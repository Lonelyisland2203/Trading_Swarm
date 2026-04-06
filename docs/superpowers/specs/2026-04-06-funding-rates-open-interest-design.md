# Funding Rates and Open Interest Integration Design

**Date:** 2026-04-06
**Status:** Approved
**Goal:** Extend market data service to fetch funding rates and open interest for perpetual swap contracts, integrating with existing indicator system.

---

## Overview

Currently, `MarketDataService` fetches only OHLCV data via CCXT. This enhancement adds support for:

1. **Funding rate history** - Periodic funding payments in perpetual contracts (updates every 8 hours)
2. **Open interest** - Total value of outstanding contracts (updates continuously)

These derivatives metrics provide critical market context:
- Funding rate trends indicate leverage sentiment (positive = longs paying shorts)
- Open interest changes signal new money entering/exiting positions
- Combined with OHLCV, enables richer trading signals

**Architecture Decision:** Extend existing `MarketDataService` rather than creating separate service. Funding and OI are fundamentally market data, reuse existing infrastructure (caching, retry logic, point-in-time safety).

---

## Architecture

### Component Overview

```
MarketDataService (extended)
  ├─ ExchangeClient (extended)
  │   ├─ fetch_ohlcv() [existing]
  │   ├─ fetch_funding_rate_history() [NEW]
  │   └─ fetch_open_interest_history() [NEW]
  │
  ├─ AsyncDiskCache [existing]
  │
  ├─ _load_perpetual_markets() [NEW]
  ├─ _get_perpetual_symbol() [NEW]
  ├─ fetch_funding_rates() [NEW]
  ├─ fetch_open_interest() [NEW]
  └─ get_market_context() [NEW]
```

### Data Flow

```
get_market_context(symbol, timeframe, as_of, lookback_bars)
  │
  ├─> fetch_ohlcv(symbol, ...)           [existing - always succeeds or raises]
  │
  ├─> _get_perpetual_symbol(symbol)
  │     └─> _load_perpetual_markets()   [one-time market discovery, cached 24h]
  │
  ├─> fetch_funding_rates(perp_symbol, ...) [if perp exists, else None]
  │     ├─ Check capability: exchange.has['fetchFundingRateHistory']
  │     ├─ Fetch with retry and throttling
  │     ├─ Cache with adaptive TTL
  │     └─ Filter by as_of timestamp
  │
  └─> fetch_open_interest(perp_symbol, ...) [if perp exists, else None]
        ├─ Check capability: exchange.has['fetchOpenInterestHistory']
        ├─ Fetch with retry and throttling
        ├─ Cache with adaptive TTL
        └─ Filter by as_of timestamp
```

---

## Data Structures

### Perpetual Mapping Cache

**Key:** `"perp_mapping:{exchange}"`
**Value:** `{"BTC/USDT": "BTC/USDT:USDT", "ETH/USDT": "ETH/USDT:USDT", ...}`
**TTL:** 24 hours (markets don't change frequently)

### Funding Rate Cache

**Key:** `"funding:{exchange}:{perp_symbol}:{since}:{limit}"`
**Value:** `[{"timestamp": 1704067200000, "fundingRate": 0.0001}, ...]`
**TTL:** Adaptive (see Caching Strategy section)

### Open Interest Cache

**Key:** `"oi:{exchange}:{perp_symbol}:{timeframe}:{since}:{limit}"`
**Value:** `[{"timestamp": 1704067200000, "openInterest": 1500000000}, ...]`
**TTL:** Adaptive (see Caching Strategy section)

### get_market_context() Return Value

```python
{
    # OHLCV data (always present)
    'ohlcv_df': pd.DataFrame,  # columns: timestamp, open, high, low, close, volume

    # Funding rate data (None if perp unavailable or exchange unsupported)
    'funding_rate': float | None,              # Latest rate as of timestamp
    'funding_rate_history': list[float] | None,  # Last 7 days of rates for trend detection

    # Open interest data (None if perp unavailable or exchange unsupported)
    'open_interest': float | None,             # Latest OI value
    'open_interest_change_pct': float | None,  # Percentage change vs 24h ago
}
```

---

## Caching Strategy (Adaptive TTL)

### Cache Key Patterns

- **Perpetual mapping:** `"perp_mapping:{exchange}"` - 24-hour TTL
- **Funding rates:** `"funding:{exchange}:{perp_symbol}:{since}:{limit}"` - Adaptive TTL
- **Open interest:** `"oi:{exchange}:{perp_symbol}:{timeframe}:{since}:{limit}"` - Adaptive TTL

### Adaptive TTL Algorithm

```python
def _compute_adaptive_ttl(as_of_timestamp: int) -> int:
    """
    Compute cache TTL based on data age.

    Historical data is immutable, cache aggressively.
    Recent data may get backfilled, cache conservatively.
    """
    now = int(time.time() * 1000)
    age_hours = (now - as_of_timestamp) / (1000 * 3600)

    if age_hours > 168:  # > 7 days old
        return 86400  # 24-hour cache (historical, won't change)
    elif age_hours > 24:  # 1-7 days old
        return 7200   # 2-hour cache (recent but stable)
    else:  # < 24 hours old
        return 1800   # 30-minute cache (may still update)
```

**Rationale:**
- Historical data (> 7 days): Exchanges rarely backfill corrections this far back
- Recent data (1-7 days): Occasionally corrected, moderate caching safe
- Live data (< 24 hours): Active trading, funding settlements, short cache

### Point-in-Time Filtering

All fetched data respects the `as_of` timestamp:

**Funding rates:**
```python
filtered = [r for r in rates if r['timestamp'] <= as_of]
```

**Open interest:**
```python
# OI bars have duration like OHLCV - filter by close time
bar_duration_ms = self._timeframe_to_ms(timeframe)
filtered = [oi for oi in oi_data if oi['timestamp'] + bar_duration_ms <= as_of]
```

This ensures historical backtesting doesn't leak future information.

---

## Perpetual Symbol Mapping

### Market Discovery

On first call (or after 24h cache expiry), `_load_perpetual_markets()`:

1. **Load markets:** Call `await exchange.load_markets()`
2. **Filter perpetuals:** Markets where `market['type'] == 'swap'` and `market['swap'] == True`
3. **Build mapping:** For each perpetual, extract base/quote/settle currencies
4. **Create dict:** Map spot symbol → perp symbol
5. **Cache:** Store with 24-hour TTL

### Mapping Logic

For spot symbol `BTC/USDT`, search for perpetual where:
- `market['base'] == 'BTC'`
- `market['quote'] == 'USDT'`
- `market['settle'] == 'USDT'` (for USDT-margined perps)

CCXT unified symbol for such contracts: `BTC/USDT:USDT`

### Implementation

```python
async def _load_perpetual_markets(self) -> dict[str, str]:
    """
    Load perpetual swap markets and create spot→perp mapping.

    Returns:
        Dict mapping spot symbols to perpetual symbols
        Example: {"BTC/USDT": "BTC/USDT:USDT", "ETH/USDT": "ETH/USDT:USDT"}
    """
    cache_key = f"perp_mapping:{self.exchange_id}"
    cached = await self.cache.get(cache_key)
    if cached is not None:
        return cached

    # Fetch markets from exchange
    markets = await asyncio.to_thread(
        self.exchange_client.exchange.load_markets, True  # reload=True
    )

    # Build spot→perp mapping
    mapping = {}
    for symbol, market in markets.items():
        if market.get('type') == 'swap' and market.get('swap'):
            # Extract base/quote currencies
            base = market['base']
            quote = market['quote']
            settle = market.get('settle', quote)

            # Map spot symbol to perp symbol
            spot_symbol = f"{base}/{quote}"
            if settle == quote:  # USDT-margined perps
                mapping[spot_symbol] = symbol

    # Cache for 24 hours
    await self.cache.set(cache_key, mapping, expire=86400)

    logger.info("Loaded perpetual markets", count=len(mapping), exchange=self.exchange_id)
    return mapping

async def _get_perpetual_symbol(self, spot_symbol: str) -> str | None:
    """
    Get perpetual symbol for a spot symbol.

    Args:
        spot_symbol: Spot trading pair (e.g., 'BTC/USDT')

    Returns:
        Perpetual symbol (e.g., 'BTC/USDT:USDT') or None if not found
    """
    mapping = await self._load_perpetual_markets()

    perp_symbol = mapping.get(spot_symbol)
    if perp_symbol is None:
        logger.info("No perpetual market found", spot=spot_symbol, exchange=self.exchange_id)

    return perp_symbol
```

---

## Data Fetching Methods

### ExchangeClient Extensions

Add two wrapper methods to `ExchangeClient` class:

```python
async def fetch_funding_rate_history(
    self,
    symbol: str,
    since: int | None = None,
    limit: int = 100,
) -> list[dict]:
    """
    Fetch funding rate history with rate limiting and retry.

    Args:
        symbol: Perpetual symbol (e.g., 'BTC/USDT:USDT')
        since: Start timestamp in Unix ms
        limit: Maximum number of records

    Returns:
        List of funding rate records: [{"timestamp": ..., "fundingRate": ...}, ...]

    Raises:
        DataUnavailableError: If fetch fails after retries
    """
    async with self.throttler:
        async def _fetch():
            return await self.exchange.fetch_funding_rate_history(
                symbol, since=since, limit=limit
            )

        try:
            return await retry_with_backoff(_fetch)
        except Exception as e:
            raise DataUnavailableError(
                f"Failed to fetch funding rates for {symbol}: {e}"
            )

async def fetch_open_interest_history(
    self,
    symbol: str,
    timeframe: str,
    since: int | None = None,
    limit: int = 100,
) -> list[dict]:
    """
    Fetch open interest history with rate limiting and retry.

    Args:
        symbol: Perpetual symbol (e.g., 'BTC/USDT:USDT')
        timeframe: Candle timeframe (e.g., '1h')
        since: Start timestamp in Unix ms
        limit: Maximum number of records

    Returns:
        List of OI records: [{"timestamp": ..., "openInterest": ...}, ...]

    Raises:
        DataUnavailableError: If fetch fails after retries
    """
    async with self.throttler:
        async def _fetch():
            return await self.exchange.fetch_open_interest_history(
                symbol, timeframe, since=since, limit=limit
            )

        try:
            return await retry_with_backoff(_fetch)
        except Exception as e:
            raise DataUnavailableError(
                f"Failed to fetch open interest for {symbol}: {e}"
            )
```

### MarketDataService Wrapper Methods

#### fetch_funding_rates()

```python
async def fetch_funding_rates(
    self,
    symbol: str,
    as_of: int,
    lookback_days: int = 7,
) -> list[dict] | None:
    """
    Fetch funding rate history with capability checking and caching.

    Args:
        symbol: Perpetual symbol
        as_of: Point-in-time timestamp (Unix ms)
        lookback_days: Number of days of history to fetch

    Returns:
        List of funding rate records, or None if unsupported/unavailable
    """
    # Check exchange capability
    if not self.exchange_client.exchange.has.get('fetchFundingRateHistory', False):
        logger.info(
            "Exchange does not support funding rate history",
            exchange=self.exchange_id
        )
        return None

    # Calculate time range
    lookback_ms = lookback_days * 24 * 60 * 60 * 1000
    since = as_of - lookback_ms

    # Check cache
    cache_key = f"funding:{self.exchange_id}:{symbol.replace('/', '_')}:{since}:{as_of}"
    cached = await self.cache.get(cache_key)
    if cached is not None:
        logger.info("Cache hit", symbol=symbol, type="funding")
        return cached

    # Fetch from exchange
    try:
        logger.info("Fetching funding rates", symbol=symbol, days=lookback_days)
        rates = await self.exchange_client.fetch_funding_rate_history(
            symbol=symbol,
            since=since,
            limit=1000,  # 7 days * 3 rates/day = 21, use high limit for safety
        )

        # Filter by as_of (point-in-time safety)
        rates = [r for r in rates if r['timestamp'] <= as_of]

        # Cache with adaptive TTL
        ttl = self._compute_adaptive_ttl(as_of)
        await self.cache.set(cache_key, rates, expire=ttl)

        logger.info("Funding rates fetched", symbol=symbol, count=len(rates))
        return rates

    except Exception as e:
        logger.warning("Failed to fetch funding rates", symbol=symbol, error=str(e))
        return None
```

#### fetch_open_interest()

```python
async def fetch_open_interest(
    self,
    symbol: str,
    timeframe: str,
    as_of: int,
    lookback_bars: int,
) -> list[dict] | None:
    """
    Fetch open interest history with capability checking and caching.

    Args:
        symbol: Perpetual symbol
        timeframe: Candle timeframe
        as_of: Point-in-time timestamp (Unix ms)
        lookback_bars: Number of historical bars

    Returns:
        List of OI records, or None if unsupported/unavailable
    """
    # Check exchange capability
    if not self.exchange_client.exchange.has.get('fetchOpenInterestHistory', False):
        logger.info(
            "Exchange does not support open interest history",
            exchange=self.exchange_id
        )
        return None

    # Calculate time range
    bar_duration_ms = self._timeframe_to_ms(timeframe)
    since = as_of - (lookback_bars * bar_duration_ms)

    # Check cache
    cache_key = f"oi:{self.exchange_id}:{symbol.replace('/', '_')}:{timeframe}:{since}:{as_of}"
    cached = await self.cache.get(cache_key)
    if cached is not None:
        logger.info("Cache hit", symbol=symbol, type="open_interest")
        return cached

    # Fetch from exchange
    try:
        logger.info("Fetching open interest", symbol=symbol, timeframe=timeframe, bars=lookback_bars)
        oi_data = await self.exchange_client.fetch_open_interest_history(
            symbol=symbol,
            timeframe=timeframe,
            since=since,
            limit=lookback_bars,
        )

        # Filter by as_of (point-in-time safety - consider bar close time)
        oi_data = [
            oi for oi in oi_data
            if oi['timestamp'] + bar_duration_ms <= as_of
        ]

        # Cache with adaptive TTL
        ttl = self._compute_adaptive_ttl(as_of)
        await self.cache.set(cache_key, oi_data, expire=ttl)

        logger.info("Open interest fetched", symbol=symbol, count=len(oi_data))
        return oi_data

    except Exception as e:
        logger.warning("Failed to fetch open interest", symbol=symbol, error=str(e))
        return None
```

---

## get_market_context() API

### Method Signature

```python
async def get_market_context(
    self,
    symbol: str,
    timeframe: str,
    as_of: int,
    lookback_bars: int,
) -> dict:
    """
    Fetch comprehensive market context including OHLCV, funding, and open interest.

    This is the primary interface for retrieving all available market data for a
    symbol at a specific point in time. Respects point-in-time safety for all data.

    Args:
        symbol: Trading pair (spot symbol, e.g., 'BTC/USDT')
        timeframe: Candle timeframe (e.g., '1h')
        as_of: Point-in-time timestamp (Unix ms)
        lookback_bars: Number of historical bars

    Returns:
        Dictionary with:
        - ohlcv_df: pd.DataFrame (always present)
        - funding_rate: float | None (latest rate as of timestamp)
        - funding_rate_history: list[float] | None (last 7 days)
        - open_interest: float | None (latest OI value)
        - open_interest_change_pct: float | None (vs 24h ago)

    Raises:
        DataUnavailableError: If OHLCV cannot be fetched (critical data)

    Note:
        Funding and OI failures are graceful - returned as None with warning logs.
    """
```

### Implementation

```python
async def get_market_context(
    self,
    symbol: str,
    timeframe: str,
    as_of: int,
    lookback_bars: int,
) -> dict:
    # Step 1: Fetch OHLCV (critical - propagate errors)
    ohlcv_df = await self.get_ohlcv_as_of(symbol, timeframe, as_of, lookback_bars)

    # Initialize result with OHLCV
    result = {
        'ohlcv_df': ohlcv_df,
        'funding_rate': None,
        'funding_rate_history': None,
        'open_interest': None,
        'open_interest_change_pct': None,
    }

    # Step 2: Get perpetual symbol (if exists)
    perp_symbol = await self._get_perpetual_symbol(symbol)
    if perp_symbol is None:
        # No perpetual available - return with None derivatives data
        return result

    # Step 3: Fetch funding rates (7 days)
    try:
        funding_data = await self.fetch_funding_rates(
            symbol=perp_symbol,
            as_of=as_of,
            lookback_days=7,
        )

        if funding_data and len(funding_data) > 0:
            # Latest funding rate
            result['funding_rate'] = funding_data[-1]['fundingRate']

            # History (list of rates)
            result['funding_rate_history'] = [r['fundingRate'] for r in funding_data]

    except Exception as e:
        logger.warning("Funding rate fetch failed", symbol=perp_symbol, error=str(e))

    # Step 4: Fetch open interest (need 24h + current for change calculation)
    try:
        # Calculate lookback needed for 24h comparison
        bar_duration_ms = self._timeframe_to_ms(timeframe)
        bars_per_day = int((24 * 60 * 60 * 1000) / bar_duration_ms)
        oi_lookback = bars_per_day + 10  # Extra bars for safety

        oi_data = await self.fetch_open_interest(
            symbol=perp_symbol,
            timeframe=timeframe,
            as_of=as_of,
            lookback_bars=oi_lookback,
        )

        if oi_data and len(oi_data) > 0:
            # Latest open interest
            result['open_interest'] = oi_data[-1]['openInterest']

            # Calculate 24h change
            if len(oi_data) >= bars_per_day:
                oi_current = oi_data[-1]['openInterest']
                oi_24h_ago = oi_data[-(bars_per_day + 1)]['openInterest']

                if oi_24h_ago > 0:
                    change_pct = ((oi_current - oi_24h_ago) / oi_24h_ago) * 100
                    result['open_interest_change_pct'] = change_pct

    except Exception as e:
        logger.warning("Open interest fetch failed", symbol=perp_symbol, error=str(e))

    return result
```

---

## Error Handling

### Error Classification

**Critical errors (propagate):**
- OHLCV fetch failures → Raise `DataUnavailableError`
- Invalid timeframe format → Raise `ValueError`
- Cache corruption → Raise exception

**Non-critical errors (graceful degradation):**
- Exchange doesn't support funding rates → Return `None`, log info
- Perpetual market not found → Return `None`, log info
- Funding/OI fetch failures → Return `None`, log warning
- Network timeouts during derivatives fetch → Return `None`, log warning

### Logging Strategy

```python
# Informational (expected cases)
logger.info("No perpetual market found", spot=symbol)
logger.info("Exchange does not support funding rate history", exchange=self.exchange_id)

# Warnings (unexpected but handled)
logger.warning("Failed to fetch funding rates", symbol=symbol, error=str(e))
logger.warning("Open interest fetch failed", symbol=symbol, error=str(e))

# Errors (critical failures)
logger.error("OHLCV fetch failed", symbol=symbol, error=str(e))
```

### Capability Checking

Before attempting to fetch funding/OI:

```python
# Check once per exchange session
if not self.exchange_client.exchange.has.get('fetchFundingRateHistory', False):
    logger.info("Exchange does not support funding rate history", exchange=self.exchange_id)
    return None
```

This avoids exceptions for control flow and provides clear user feedback.

---

## Testing Strategy

### Unit Tests (New)

**Test file:** `tests/test_market_data_derivatives.py`

**Test coverage:**

1. **Symbol mapping:**
   - Test spot→perp mapping with mocked market data
   - Test missing perpetual (return None)
   - Test cache hit/miss for perpetual mapping
   - Test 24-hour TTL expiration

2. **Funding rate fetching:**
   - Mock CCXT `fetch_funding_rate_history()` responses
   - Test point-in-time filtering (rates before as_of)
   - Test adaptive caching (historical vs recent data)
   - Test exchange capability checking (unsupported exchange)
   - Test graceful failure (network error → return None)

3. **Open interest fetching:**
   - Mock CCXT `fetch_open_interest_history()` responses
   - Test point-in-time filtering (consider bar close time)
   - Test adaptive caching
   - Test 24h change calculation
   - Test insufficient data for change (< 24h history)

4. **get_market_context() integration:**
   - Mock all underlying fetches (OHLCV + funding + OI)
   - Test successful full context fetch
   - Test perp not found (derivatives fields None)
   - Test exchange doesn't support funding (graceful None)
   - Test OHLCV failure (propagates exception)
   - Test funding failure + OI success (partial data)

### Mock Response Examples

```python
# Mock funding rate history response
mock_funding_rates = [
    {"timestamp": 1704067200000, "fundingRate": 0.0001},
    {"timestamp": 1704096000000, "fundingRate": 0.00012},
    {"timestamp": 1704124800000, "fundingRate": 0.00015},
]

# Mock open interest history response
mock_oi_data = [
    {"timestamp": 1704067200000, "openInterest": 1500000000},
    {"timestamp": 1704070800000, "openInterest": 1520000000},
    {"timestamp": 1704074400000, "openInterest": 1510000000},
]

# Mock perpetual markets
mock_markets = {
    "BTC/USDT:USDT": {
        "symbol": "BTC/USDT:USDT",
        "base": "BTC",
        "quote": "USDT",
        "settle": "USDT",
        "type": "swap",
        "swap": True,
    },
}
```

### Integration Tests (Extend Existing)

Update `tests/test_data_layer.py`:

```python
@pytest.mark.asyncio
async def test_market_context_with_derivatives():
    """Test get_market_context() returns all data types."""
    # Use test exchange with mocked responses
    # Verify ohlcv_df, funding_rate, funding_rate_history, open_interest, open_interest_change_pct
    pass

@pytest.mark.asyncio
async def test_market_context_spot_only():
    """Test get_market_context() with spot symbol (no perp)."""
    # Verify derivatives fields are None
    pass
```

---

## Integration Points

### 1. PromptBuilder Integration

Current usage:
```python
# Before
df = await service.get_ohlcv_as_of(...)
indicators = compute_all_indicators(df)
```

After enhancement:
```python
# After
context = await service.get_market_context(symbol, timeframe, as_of, lookback_bars)

# Compute indicators on OHLCV
indicators = compute_all_indicators(context['ohlcv_df'])

# Merge derivatives data into indicators dict
indicators['funding_rate'] = context['funding_rate']
indicators['funding_rate_history'] = context['funding_rate_history']
indicators['open_interest'] = context['open_interest']
indicators['open_interest_change_pct'] = context['open_interest_change_pct']
```

The merged `indicators` dict can then be used in prompt building:

```python
prompt = builder.build_prompt(
    task=task,
    df=context['ohlcv_df'],
    symbol=symbol,
    timeframe=timeframe,
    market_regime=regime,
)
```

### 2. Indicator System Enhancement

The `indicators` dict already includes placeholders for funding and OI:

```python
# From data/indicators.py:1030-1034
# Crypto-specific placeholders (stubs - will be populated by actual market data)
result['funding_rate'] = None
result['open_interest'] = None
```

These placeholders can now be populated:

```python
# In calling code (e.g., dataset generation, swarm workflow)
context = await market_service.get_market_context(...)
indicators = compute_all_indicators(context['ohlcv_df'])

# Override stubs with real data
indicators['funding_rate'] = context['funding_rate']
indicators['open_interest'] = context['open_interest']
indicators['open_interest_change_pct'] = context['open_interest_change_pct']
indicators['funding_rate_history'] = context['funding_rate_history']
```

### 3. Backward Compatibility

**No breaking changes:**
- Existing code using `fetch_ohlcv()` or `get_ohlcv_as_of()` continues to work unchanged
- `get_market_context()` is a new method, doesn't affect existing callers
- `compute_all_indicators()` already has funding/OI placeholders, no signature change needed

**Migration path:**
1. Deploy enhanced `MarketDataService`
2. Update dataset generation to use `get_market_context()`
3. Update swarm workflow to use `get_market_context()`
4. Prompts automatically include funding/OI when available (no template changes needed)

---

## Configuration

### Settings Extension (Optional)

Add to `config/settings.py`:

```python
class MarketDataSettings(BaseSettings):
    # Existing fields...

    # New fields (optional)
    enable_derivatives_data: bool = True  # Feature flag
    funding_lookback_days: int = 7        # Default funding history
    oi_lookback_bars_multiplier: float = 1.5  # Extra bars for OI change calculation
```

This allows disabling derivatives data fetching if needed (e.g., for spot-only backtests).

---

## Performance Considerations

### API Call Overhead

**Before (per context fetch):**
- 1 OHLCV request

**After (per context fetch):**
- 1 OHLCV request
- 1 perpetual mapping lookup (cached 24h, amortized to ~0 after first call)
- 1 funding rate request (if perp exists)
- 1 open interest request (if perp exists)

**Total:** 2-3 additional API calls per context fetch

**Mitigation:**
- Adaptive caching reduces repeated calls for historical data
- Capability checking prevents attempts on unsupported exchanges
- Throttling prevents rate limit violations

### Cache Storage

**Disk usage estimate:**
- Perpetual mapping: ~10 KB (100 symbols)
- Funding rates (7 days): ~2 KB per symbol per cache entry
- Open interest (100 bars): ~5 KB per symbol per cache entry

**Total:** ~7 KB per symbol per cache entry (negligible compared to OHLCV)

### Memory Usage

All data processed as lists/dicts, converted to pandas only for OHLCV. Memory footprint unchanged from current implementation.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Exchange doesn't support funding/OI | Missing derivatives data | Capability checking, graceful None returns |
| Perp symbol not found for spot pair | Missing derivatives data | Info logging, return None |
| API rate limits exceeded | Temporary fetch failures | Existing throttling + retry with backoff |
| Incorrect spot→perp mapping | Wrong data fetched | Test with multiple exchanges, validate mapping logic |
| Point-in-time violation | Lookahead bias in backtest | Filter all data by as_of, include tests |
| Cache key collisions | Wrong data returned | Use unique keys with all parameters |

---

## Success Criteria

1. **Functional:**
   - `get_market_context()` returns OHLCV + funding + OI for perpetual pairs
   - Returns OHLCV only for spot-only symbols (graceful degradation)
   - Point-in-time safety maintained for all data types

2. **Performance:**
   - Adaptive caching reduces API calls for historical data
   - No significant latency increase vs current OHLCV-only fetching

3. **Quality:**
   - 100% test coverage for new methods
   - All tests pass (existing + new)
   - No breaking changes to existing code

4. **Integration:**
   - `compute_all_indicators()` can be extended with funding/OI data
   - Prompts automatically include derivatives context when available

---

## Future Enhancements

### Phase 2 (Not in Scope)

1. **Multi-exchange support:** Fetch spot from one exchange, derivatives from another
2. **Liquidation data:** Add liquidation events to market context
3. **Aggregated funding:** Compute weighted average funding across exchanges
4. **Real-time streaming:** WebSocket support for live funding rate updates
5. **Exchange-specific optimizations:** Binance has batch endpoints for funding rates

---

## Appendix: CCXT API Reference

### fetch_funding_rate_history()

```python
await exchange.fetch_funding_rate_history(
    symbol='BTC/USDT:USDT',
    since=1704067200000,  # Unix ms
    limit=100,
)
```

**Returns:**
```python
[
    {
        'info': {...},  # Raw exchange response
        'symbol': 'BTC/USDT:USDT',
        'timestamp': 1704067200000,
        'datetime': '2024-01-01T00:00:00.000Z',
        'fundingRate': 0.0001,  # 0.01%
    },
    ...
]
```

### fetch_open_interest_history()

```python
await exchange.fetch_open_interest_history(
    symbol='BTC/USDT:USDT',
    timeframe='1h',
    since=1704067200000,
    limit=100,
)
```

**Returns:**
```python
[
    {
        'symbol': 'BTC/USDT:USDT',
        'timestamp': 1704067200000,
        'datetime': '2024-01-01T00:00:00.000Z',
        'openInterest': 1500000000,  # Notional value in quote currency
    },
    ...
]
```

### Exchange Capability Flags

```python
exchange.has = {
    'fetchFundingRateHistory': True,     # Binance, Bybit, OKX
    'fetchOpenInterestHistory': True,    # Binance, Bybit, OKX
    'loadMarkets': True,                 # All exchanges
}
```

---

**End of Design Document**
