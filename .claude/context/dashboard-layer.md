# Dashboard Layer

## Architecture
- Backend: FastAPI + WebSocket (real-time updates)
- Frontend: React + Vite + Tailwind + Recharts + shadcn/ui
- Design: Bloomberg terminal aesthetic. Figma design tokens → CSS variables.
- Port: 8420 (API), 5173 (UI dev)

## Design Tokens
- Background: #0D1117, Surface: #161B22, Border: #30363D
- Text: #E6EDF3 / #8B949E, Accent: #1A6B5C
- Green: #00D084, Amber: #F5A623, Red: #E24B4A
- Font data: JetBrains Mono, Font headings: Space Grotesk

## Pages
1. Live Overview — positions, P&L, drawdown gauge, kill switch, watchdog heartbeat
2. Signal Pipeline — recent signals table with XGBoost prob, LLM context, outcome
3. Performance — equity curve, rolling Sharpe, drawdown, fee drag
4. XGBoost Model — SHAP importance, IC/Brier trends, walk-forward folds
5. Autoresearch — experiment log, Sharpe_net progression, best config
6. Risk Monitor — funding heatmap, OI changes, liquidation alerts

## API Endpoints
GET  /api/positions, /api/pnl/daily, /api/signals/recent
GET  /api/performance, /api/xgboost/features, /api/xgboost/metrics
GET  /api/autoresearch/log, /api/risk/funding, /api/risk/oi
WS   /ws/live — real-time updates every 5s
GET  /api/health — watchdog heartbeat

## Constraint
Dashboard is READ-ONLY. No trade execution from UI. Architecture Constraint #11.