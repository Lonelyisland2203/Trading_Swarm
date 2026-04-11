# Execution Layer

## BinanceExecutionClient Safety Controls
- Testnet by default; live requires ALLOW_LIVE_TRADING=true + typed confirmation
- Kill switch: STOP file in project root halts all trading
- Process lock via fcntl prevents concurrent execution instances

## Position Limits
- Max 3 concurrent positions
- Max 10 trades per day
- 60-second cooldown between trades on same symbol
- 1x leverage only (no margin multiplication)

## Circuit Breaker
- 2% daily drawdown limit triggers full liquidation + 24h halt
- Per-position stop-loss: 1% of portfolio value

## Fee-Aware Sizing
- Use FeeModelSettings for all cost calculations
- Position size accounts for: maker/taker fees, funding rate projection
- Round down to symbol's lot size precision
- Minimum notional check before order submission
