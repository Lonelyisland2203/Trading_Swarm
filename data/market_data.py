"""
Market data fetching using CCXT with async caching and retry logic.

Implements point-in-time safety and data validation.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any

import aiohttp
import ccxt.async_support as ccxt
import pandas as pd
from asyncio_throttle import Throttler
from loguru import logger

from config.settings import settings
from .cache_wrapper import AsyncDiskCache, make_cache_key
from .indicators import validate_ohlcv

# Cache TTL in seconds (1 hour)
CACHE_TTL_SECONDS = 3600


class DataUnavailableError(Exception):
    """Raised when market data cannot be fetched."""
    pass


async def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple = (
        ccxt.NetworkError,
        ccxt.ExchangeNotAvailable,
        ccxt.RequestTimeout,
        ccxt.RateLimitExceeded,
    ),
) -> Any:
    """
    Retry async function with exponential backoff.

    Args:
        func: Async function to retry
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        retryable_exceptions: Exceptions that trigger retry

    Returns:
        Function result

    Raises:
        Last exception if all retries exhausted
        ExchangeError immediately (permanent failure)
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return await func()
        except retryable_exceptions as e:
            last_exception = e
            delay = min(base_delay * (2 ** attempt), max_delay)

            # Extra delay for rate limits
            if isinstance(e, ccxt.RateLimitExceeded):
                delay = max(delay, 10.0)

            logger.warning(
                "Retry attempt",
                attempt=attempt + 1,
                max_retries=max_retries,
                delay=delay,
                error_type=type(e).__name__,
                error_msg=str(e)[:100],
            )
            await asyncio.sleep(delay)
        except ccxt.ExchangeError as e:
            # Do not retry - permanent error (invalid symbol, etc.)
            logger.error("Exchange error (not retrying)", error=str(e))
            raise

    raise last_exception


class ExchangeClient:
    """
    CCXT wrapper with rate limiting and retry logic.

    Manages connection to crypto exchange with defensive error handling.
    """

    def __init__(self, exchange_id: str = "binance"):
        """
        Initialize exchange client.

        Args:
            exchange_id: CCXT exchange identifier
        """
        self.exchange_id = exchange_id
        self.exchange = getattr(ccxt, exchange_id)({
            "enableRateLimit": True,  # CCXT built-in rate limiting
            "rateLimit": 100,         # ms between requests
        })
        # Inject a session using ThreadedResolver so aiodns is bypassed.
        # aiodns (c-ares) fails on Windows; Python's built-in resolver works fine.
        connector = aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())
        self.exchange.session = aiohttp.ClientSession(connector=connector)

        # Additional throttle for burst protection
        # Binance: 1200 requests/minute = 20/second, use 15/sec to be conservative
        self.throttler = Throttler(rate_limit=15, period=1.0)

    def load_markets(self) -> dict:
        """
        Load market metadata from exchange (synchronous).

        Returns:
            Dict mapping symbols to market metadata
        """
        return self.exchange.load_markets()

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int | None = None,
        limit: int = 500,
    ) -> list[list]:
        """
        Fetch OHLCV data from exchange with rate limiting.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1h')
            since: Start timestamp in Unix ms
            limit: Maximum number of candles

        Returns:
            List of OHLCV candles [[timestamp, open, high, low, close, volume], ...]

        Raises:
            DataUnavailableError: If data cannot be fetched
        """
        async with self.throttler:
            async def _fetch():
                return await self.exchange.fetch_ohlcv(
                    symbol, timeframe, since, limit
                )

            try:
                return await retry_with_backoff(_fetch)
            except Exception as e:
                raise DataUnavailableError(
                    f"Failed to fetch {symbol} {timeframe} from {self.exchange_id}: {e}"
                )

    async def close(self):
        """Close exchange connection."""
        await self.exchange.close()


class MarketDataService:
    """
    Main interface for market data fetching with caching and point-in-time safety.

    Responsibilities:
    - Fetch OHLCV data from exchange
    - Cache data to disk
    - Enforce point-in-time correctness
    - Validate data integrity

    Usage:
        async with MarketDataService() as service:
            df = await service.fetch_ohlcv("BTC/USDT", "1h", 100)
    """

    def __init__(self):
        """Initialize market data service."""
        self.exchange_client = ExchangeClient(settings.market_data.exchange)
        self.cache = AsyncDiskCache(
            directory=settings.market_data.cache_dir,
            size_limit=settings.market_data.cache_size_limit,
        )

    async def __aenter__(self) -> "MarketDataService":
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager and cleanup resources."""
        await self.close()

    def _timeframe_to_ms(self, timeframe: str) -> int:
        """
        Convert timeframe string to milliseconds.

        Args:
            timeframe: Timeframe string (e.g., '1m', '1h', '1d')

        Returns:
            Duration in milliseconds
        """
        units = {
            "m": 60 * 1000,
            "h": 60 * 60 * 1000,
            "d": 24 * 60 * 60 * 1000,
        }

        if len(timeframe) < 2:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        num = int(timeframe[:-1])
        unit = timeframe[-1]

        if unit not in units:
            raise ValueError(f"Invalid timeframe unit: {unit}")

        return num * units[unit]

    def _compute_adaptive_ttl(self, data_timestamp: datetime, as_of: datetime) -> int:
        """
        Compute adaptive cache TTL based on data age.

        Args:
            data_timestamp: Timestamp of the data point
            as_of: Current timestamp for age calculation

        Returns:
            Cache TTL in seconds
        """
        age = as_of - data_timestamp

        if age > timedelta(days=7):
            return 86400  # Historical: 24 hours
        elif age > timedelta(hours=1):
            return 7200   # Recent: 2 hours
        else:
            return 1800   # Live: 30 minutes

    async def _load_perpetual_markets(self) -> dict[str, str]:
        """
        Load spot->perpetual symbol mapping from exchange.

        Returns:
            Dict mapping spot symbols to perpetual symbols
        """
        cache_key = "perpetual_markets_mapping"

        # Try cache first
        cached = await asyncio.to_thread(self.cache.get, cache_key)
        if cached is not None:
            return cached

        # Load markets from exchange
        markets = await asyncio.to_thread(
            self.exchange_client.load_markets
        )

        # Build spot -> perp mapping
        mapping = {}
        for symbol, market in markets.items():
            if market.get('type') == 'swap' and market.get('settle') == 'USDT':
                # Extract base symbol (BTC/USDT:USDT -> BTC/USDT)
                base_symbol = symbol.split(':')[0]
                if base_symbol != symbol:  # Ensure it's actually a perp
                    mapping[base_symbol] = symbol

        # Cache for 24 hours
        await asyncio.to_thread(
            self.cache.set,
            cache_key,
            mapping,
            expire=86400
        )

        return mapping

    async def _get_perpetual_symbol(self, spot_symbol: str) -> str | None:
        """
        Get perpetual symbol for a spot symbol.

        Args:
            spot_symbol: Spot symbol (e.g., 'BTC/USDT')

        Returns:
            Perpetual symbol (e.g., 'BTC/USDT:USDT') or None if not found
        """
        mapping = await self._load_perpetual_markets()
        return mapping.get(spot_symbol)

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int,
        end_ts: int | None = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data with caching.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            lookback_bars: Number of historical bars
            end_ts: End timestamp in Unix ms. Defaults to now.
                    Required for historical windows so the fetch targets
                    the correct time period instead of always fetching recent data.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Calculate time range - use provided end_ts or default to now
        bar_duration_ms = self._timeframe_to_ms(timeframe)
        if end_ts is None:
            end_ts = int(time.time() * 1000)
        start_ts = end_ts - (lookback_bars * bar_duration_ms)

        # Check cache first
        cache_key = make_cache_key(
            exchange=settings.market_data.exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_ts=start_ts,
            end_ts=end_ts,
        )

        cached = await self.cache.get(cache_key)
        if cached is not None:
            logger.info("Cache hit", symbol=symbol, timeframe=timeframe, bars=len(cached))
            return pd.DataFrame(cached)

        # Fetch from exchange
        logger.info("Fetching from exchange", symbol=symbol, timeframe=timeframe, bars=lookback_bars)
        ohlcv_list = await self.exchange_client.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=start_ts,
            limit=lookback_bars,
        )

        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv_list,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        # Validate data
        df = validate_ohlcv(df)

        # Cache result
        await self.cache.set(cache_key, df.to_dict("records"), expire=CACHE_TTL_SECONDS)

        logger.info("OHLCV fetched", symbol=symbol, bars=len(df))
        return df

    async def get_ohlcv_as_of(
        self,
        symbol: str,
        timeframe: str,
        as_of: int,
        lookback_bars: int,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data with point-in-time safety.

        Only returns bars that would have been "known" at the as_of timestamp.
        A candle is "known" only after it closes.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            as_of: Point-in-time timestamp (Unix ms) - "what would I have known at this moment?"
            lookback_bars: Number of historical bars

        Returns:
            DataFrame filtered for point-in-time safety

        Example:
            >>> # Get data as of 2024-01-01 10:00:00
            >>> df = await service.get_ohlcv_as_of(
            ...     'BTC/USDT', '1h', 1704106800000, 100
            ... )
            >>> # Only includes bars that closed before 10:00:00
        """
        # Fetch data anchored to as_of so the exchange query targets the
        # correct historical period, not the current time.
        df = await self.fetch_ohlcv(symbol, timeframe, lookback_bars, end_ts=as_of)

        # Calculate bar close time (when the bar becomes "known")
        bar_duration_ms = self._timeframe_to_ms(timeframe)
        df["close_time"] = df["timestamp"] + bar_duration_ms

        # Filter: only bars that would have been complete at as_of
        pit_safe = df[df["close_time"] <= as_of].copy()

        if pit_safe.empty:
            raise DataUnavailableError(
                f"No point-in-time safe data for {symbol} as of {as_of}"
            )

        return pit_safe.drop(columns=["close_time"])

    async def close(self):
        """Close connections and cache."""
        await self.exchange_client.close()
        await self.cache.close()
