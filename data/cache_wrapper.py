"""
Async wrapper for diskcache to prevent event loop blocking.

Uses asyncio.to_thread() to run synchronous diskcache operations in thread pool.
"""

import asyncio
from pathlib import Path
from typing import Any

from diskcache import Cache


class AsyncDiskCache:
    """
    Thread-safe async wrapper for diskcache.

    Prevents event loop blocking by running disk I/O in thread pool.
    Supports async context manager for automatic resource cleanup.
    """

    def __init__(self, directory: Path, size_limit: int):
        """
        Initialize async disk cache.

        Args:
            directory: Cache directory path
            size_limit: Maximum cache size in bytes
        """
        self.cache = Cache(str(directory), size_limit=size_limit)

    async def __aenter__(self) -> "AsyncDiskCache":
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager and close cache."""
        await self.close()

    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache asynchronously.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        return await asyncio.to_thread(self.cache.get, key, default)

    async def set(self, key: str, value: Any, expire: int | None = None) -> bool:
        """
        Set value in cache asynchronously.

        Args:
            key: Cache key
            value: Value to cache
            expire: Optional expiration time in seconds

        Returns:
            True if successful
        """
        return await asyncio.to_thread(self.cache.set, key, value, expire=expire)

    async def delete(self, key: str) -> bool:
        """
        Delete key from cache asynchronously.

        Args:
            key: Cache key to delete

        Returns:
            True if key existed and was deleted
        """
        return await asyncio.to_thread(self.cache.delete, key)

    async def clear(self) -> int:
        """
        Clear all cache entries asynchronously.

        Returns:
            Number of entries cleared
        """
        return await asyncio.to_thread(self.cache.clear)

    async def close(self) -> None:
        """Close cache connection."""
        await asyncio.to_thread(self.cache.close)


def make_cache_key(
    exchange: str,
    symbol: str,
    timeframe: str,
    start_ts: int,
    end_ts: int,
) -> str:
    """
    Create deterministic cache key for OHLCV data.

    Args:
        exchange: Exchange name (e.g., 'binance')
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Candle timeframe (e.g., '1h')
        start_ts: Start timestamp (Unix ms)
        end_ts: End timestamp (Unix ms)

    Returns:
        Structured cache key

    Example:
        >>> make_cache_key('binance', 'BTC/USDT', '1h', 1640000000000, 1640003600000)
        'ohlcv:binance:BTC_USDT:1h:1640000000000:1640003600000'
    """
    # Normalize exchange and symbol for consistent cache keys
    normalized_exchange = exchange.lower()
    normalized_symbol = symbol.replace("/", "_").replace("-", "_").upper()

    return f"ohlcv:{normalized_exchange}:{normalized_symbol}:{timeframe}:{start_ts}:{end_ts}"
