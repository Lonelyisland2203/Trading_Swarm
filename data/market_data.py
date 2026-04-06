"""
Market data fetching using CCXT with async caching and retry logic.

Implements point-in-time safety and data validation.
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
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

    async def fetch_funding_rate_history(
        self,
        symbol: str,
        since: int | None = None,
        limit: int = 100
    ) -> list[dict]:
        """
        Fetch funding rate history for a perpetual contract.

        Args:
            symbol: Perpetual symbol (e.g., 'BTC/USDT:USDT')
            since: Timestamp in milliseconds (optional)
            limit: Maximum number of records to fetch

        Returns:
            List of funding rate records from CCXT

        Raises:
            DataUnavailableError: If funding rate data cannot be fetched
        """
        async with self.throttler:
            async def _fetch():
                return await asyncio.to_thread(
                    self.exchange.fetch_funding_rate_history,
                    symbol,
                    since=since,
                    limit=limit
                )

            try:
                return await retry_with_backoff(_fetch)
            except Exception as e:
                raise DataUnavailableError(
                    f"Failed to fetch funding rate history for {symbol}: {e}"
                )

    async def fetch_open_interest_history(
        self,
        symbol: str,
        timeframe: str = '1h',
        since: int | None = None,
        limit: int = 100
    ) -> list[dict]:
        """
        Fetch open interest history for a perpetual contract.

        Args:
            symbol: Perpetual symbol (e.g., 'BTC/USDT:USDT')
            timeframe: Candle timeframe (e.g., '1h', '4h')
            since: Timestamp in milliseconds (optional)
            limit: Maximum number of records to fetch

        Returns:
            List of open interest records from CCXT

        Raises:
            DataUnavailableError: If open interest data cannot be fetched
        """
        async with self.throttler:
            async def _fetch():
                return await asyncio.to_thread(
                    self.exchange.fetch_open_interest_history,
                    symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit
                )

            try:
                return await retry_with_backoff(_fetch)
            except Exception as e:
                raise DataUnavailableError(
                    f"Failed to fetch open interest history for {symbol}: {e}"
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

    def _bars_for_24_hours(self, timeframe: str) -> int:
        """
        Calculate number of bars needed for 24-hour lookback.

        Args:
            timeframe: Timeframe string (e.g., '1m', '1h', '1d')

        Returns:
            Number of bars representing 24 hours

        Examples:
            - '1m' -> 1440 bars (24 * 60)
            - '1h' -> 24 bars
            - '4h' -> 6 bars
            - '1d' -> 1 bar (but would need more data for accurate calculation)
        """
        # Map common timeframes to minutes
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }

        minutes = timeframe_minutes.get(timeframe, 60)  # default to 1h
        return int(1440 / minutes)  # 1440 minutes in 24 hours

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
        cached = await self.cache.get(cache_key)
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
        await self.cache.set(
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

    async def fetch_funding_rates(
        self,
        symbol: str,
        as_of: datetime | None = None,
        limit: int = 100
    ) -> pd.DataFrame | None:
        """
        Fetch funding rate history for a symbol with adaptive caching.

        Args:
            symbol: Spot symbol (e.g., 'BTC/USDT')
            as_of: Point-in-time timestamp (filters results)
            limit: Maximum records to fetch

        Returns:
            DataFrame with columns [timestamp, funding_rate] or None if unsupported
        """
        # Check exchange capability
        if not self.exchange_client.exchange.has.get('fetchFundingRateHistory', False):
            logger.warning(
                "Exchange does not support funding rate history",
                exchange=self.exchange_client.exchange_id
            )
            return None

        # Get perpetual symbol
        perp_symbol = await self._get_perpetual_symbol(symbol)
        if perp_symbol is None:
            logger.warning("No perpetual found for spot symbol", symbol=symbol)
            return None

        # Build cache key
        cache_key = f"funding_rates_{perp_symbol}_{limit}"

        # Try cache first
        cached = await self.cache.get(cache_key)
        if cached is not None:
            df = cached
        else:
            # Fetch from exchange
            try:
                raw_data = await self.exchange_client.fetch_funding_rate_history(
                    perp_symbol,
                    limit=limit
                )
            except DataUnavailableError as e:
                logger.error("Failed to fetch funding rates", symbol=symbol, error=str(e))
                return None

            # Convert to DataFrame
            df = pd.DataFrame(raw_data)
            if df.empty:
                return None

            # Standardize columns
            df = df.rename(columns={'fundingRate': 'funding_rate'})
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

            # Determine TTL based on most recent data timestamp
            most_recent = df['timestamp'].max()
            now = datetime.now(timezone.utc)
            ttl = self._compute_adaptive_ttl(most_recent, now)

            # Cache result
            await self.cache.set(cache_key, df, expire=ttl)

        # Apply point-in-time filter if requested
        if as_of is not None:
            df = df[df['timestamp'] <= as_of]

        return df[['timestamp', 'funding_rate']]

    async def fetch_open_interest(
        self,
        symbol: str,
        timeframe: str = '1h',
        as_of: datetime | None = None,
        limit: int = 100
    ) -> pd.DataFrame | None:
        """
        Fetch open interest history for a symbol with adaptive caching.

        Args:
            symbol: Spot symbol (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1h', '4h')
            as_of: Point-in-time timestamp (filters results)
            limit: Maximum records to fetch

        Returns:
            DataFrame with columns [timestamp, open_interest, open_interest_value] or None if unsupported
        """
        # Check exchange capability
        if not self.exchange_client.exchange.has.get('fetchOpenInterestHistory', False):
            logger.warning(
                "Exchange does not support open interest history",
                exchange=self.exchange_client.exchange_id
            )
            return None

        # Get perpetual symbol
        perp_symbol = await self._get_perpetual_symbol(symbol)
        if perp_symbol is None:
            logger.warning("No perpetual found for spot symbol", symbol=symbol)
            return None

        # Build cache key
        cache_key = f"open_interest_{perp_symbol}_{timeframe}_{limit}"

        # Try cache first
        cached = await self.cache.get(cache_key)
        if cached is not None:
            df = cached
        else:
            # Fetch from exchange
            try:
                raw_data = await self.exchange_client.fetch_open_interest_history(
                    perp_symbol,
                    timeframe=timeframe,
                    limit=limit
                )
            except DataUnavailableError as e:
                logger.error("Failed to fetch open interest", symbol=symbol, error=str(e))
                return None

            # Convert to DataFrame
            df = pd.DataFrame(raw_data)
            if df.empty:
                return None

            # Standardize columns
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

            # Extract relevant columns (may have openInterestValue or openInterest)
            if 'openInterestValue' in df.columns:
                df = df.rename(columns={'openInterestValue': 'open_interest_value'})
            if 'openInterest' in df.columns:
                df = df.rename(columns={'openInterest': 'open_interest'})

            # Determine TTL based on most recent data timestamp
            most_recent = df['timestamp'].max()
            now = datetime.now(timezone.utc)
            ttl = self._compute_adaptive_ttl(most_recent, now)

            # Cache result
            await self.cache.set(cache_key, df, expire=ttl)

        # Apply point-in-time filter if requested
        if as_of is not None:
            df = df[df['timestamp'] <= as_of]

        # Return standardized columns
        available_cols = [c for c in ['timestamp', 'open_interest', 'open_interest_value'] if c in df.columns]
        return df[available_cols]

    async def get_market_context(
        self,
        symbol: str,
        timeframe: str = '1h',
        as_of: datetime | None = None,
        limit: int = 100
    ) -> dict:
        """
        Fetch unified market context including OHLCV, funding rates, and open interest.

        Args:
            symbol: Spot symbol (e.g., 'BTC/USDT')
            timeframe: Candle timeframe for OHLCV and open interest
            as_of: Point-in-time timestamp (filters all data)
            limit: Maximum records to fetch

        Returns:
            Dict with keys:
                - ohlcv_df: OHLCV DataFrame
                - funding_rate: Most recent funding rate (float)
                - funding_rate_history: Funding rate history DataFrame
                - open_interest: Most recent open interest value (float or None)
                - open_interest_change_pct: 24-hour change in open interest (float or None)
        """
        # Fetch OHLCV (always available)
        if as_of is not None:
            # Convert datetime to Unix ms if needed
            if isinstance(as_of, datetime):
                as_of_ms = int(as_of.timestamp() * 1000)
            else:
                as_of_ms = as_of
            ohlcv_df = await self.get_ohlcv_as_of(symbol, timeframe, as_of_ms, limit)
        else:
            ohlcv_df = await self.fetch_ohlcv(symbol, timeframe, limit)

        # Fetch funding rates (optional)
        funding_df = await self.fetch_funding_rates(symbol, as_of=as_of, limit=limit)
        if funding_df is not None and len(funding_df) > 0:
            funding_rate = float(funding_df['funding_rate'].iloc[-1])
            funding_rate_history = funding_df
        else:
            funding_rate = None
            funding_rate_history = None

        # Fetch open interest (optional)
        oi_df = await self.fetch_open_interest(symbol, timeframe=timeframe, as_of=as_of, limit=limit)
        if oi_df is not None and len(oi_df) > 0:
            # Get most recent value
            if 'open_interest_value' in oi_df.columns:
                latest_oi = float(oi_df['open_interest_value'].iloc[-1])
                # Calculate 24-hour change if enough data
                # Number of bars depends on timeframe (e.g., 1m=1440 bars, 1h=24 bars, 4h=6 bars)
                bars_needed = self._bars_for_24_hours(timeframe)
                if len(oi_df) >= bars_needed:
                    oi_24h_ago = float(oi_df['open_interest_value'].iloc[-bars_needed])
                    oi_change_pct = ((latest_oi - oi_24h_ago) / oi_24h_ago) * 100
                else:
                    oi_change_pct = None
            elif 'open_interest' in oi_df.columns:
                latest_oi = float(oi_df['open_interest'].iloc[-1])
                oi_change_pct = None  # Can't calculate value change from contract count
            else:
                latest_oi = None
                oi_change_pct = None
        else:
            latest_oi = None
            oi_change_pct = None

        return {
            'ohlcv_df': ohlcv_df,
            'funding_rate': funding_rate,
            'funding_rate_history': funding_rate_history,
            'open_interest': latest_oi,
            'open_interest_change_pct': oi_change_pct
        }

    async def close(self):
        """Close connections and cache."""
        await self.exchange_client.close()
        await self.cache.close()
