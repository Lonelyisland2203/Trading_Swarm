# Data Layer

## Indicator Groups (17 total)
1. **Trend** (4): EMA crossover, ADX, Supertrend, MACD
2. **Momentum** (4): RSI, Stochastic, CCI, Williams %R
3. **Volatility** (3): ATR, Bollinger %B, Keltner position
4. **Volume** (3): OBV slope, VWAP deviation, Volume MA ratio
5. **Market** (3): Funding rate, OI change, Liquidation imbalance

## Multi-Timeframe Voting
- Base: 4 indicators vote on primary timeframe
- Confirmation: 2 higher timeframes (e.g., 1h→4h,1d) can veto/cap
- Veto threshold: ≥75% opposing signals from higher TF
- Cap: Reduce position size by 50% if mixed higher-TF signals

## Derivatives Data (via Binance Futures API)
- Funding rate: 8h intervals, fetch at T-1 settlement
- Open interest: Absolute and 24h delta
- Liquidation data: Long/short imbalance (6h rolling)

## Adaptive TTL Caching
| Data Age    | TTL     |
|-------------|---------|
| > 7 days    | 24h     |
| 1-7 days    | 2h      |
| < 24 hours  | 30 min  |
| < 1 hour    | No cache|

## Temporal Safety
- get_ohlcv_as_of(end_ts=as_of) filters by bar close time
- All features computed point-in-time; no future data leakage
- Timestamps in UTC; bar labeled by close time
