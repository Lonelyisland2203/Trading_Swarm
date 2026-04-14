# Execution Layer

## Multi-Exchange Support
- ExchangeRouter dispatches to HyperliquidAdapter or BinanceExecutionClient
- EXCHANGE env var: "hyperliquid" | "binance" (default: binance)
- Identical interface regardless of exchange
- Runtime switching via router.switch_exchange()

## HyperliquidAdapter
- EIP-712 signing via hyperliquid-python-sdk
- Auto exchange-side stop-loss on every position
- Connection retry: 3 attempts, exponential backoff
- Order logging to execution/order_log.jsonl

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

## Independent Watchdog
- COMPLETELY SEPARATE PROCESS (not imported by signal_loop, not part of LangGraph)
- Run via systemd/supervisor: `python execution/watchdog.py --exchange hyperliquid`
- Polls positions every 30s
- Enforces: max 2% daily loss → flatten all, position age > 48h → alert
- STOP file → immediate flatten and exit
- Writes heartbeat to dashboard/health_status.json
