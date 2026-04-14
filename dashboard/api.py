"""
FastAPI dashboard backend.

Architecture Constraint #11: Dashboard is READ-ONLY.
No POST, PUT, DELETE endpoints.

Port: 8420
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from dashboard.data_readers import (
    compute_daily_pnl,
    compute_drawdown,
    compute_equity_curve,
    compute_rolling_sharpe,
    compute_win_rate,
    read_autoresearch_results,
    read_health_status,
    read_order_log,
    read_signal_log,
)


# ---------------------------------------------------------------------------
# App Configuration
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Trade Swarm Dashboard API",
    description="Read-only dashboard for Trade Swarm trading system",
    version="1.0.0",
)

# CORS for frontend at localhost:5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET"],  # READ-ONLY: Only GET allowed
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Caching for expensive operations
# ---------------------------------------------------------------------------

_positions_cache: dict[str, Any] = {"data": [], "timestamp": None}
_funding_cache: dict[str, Any] = {"data": {}, "timestamp": None}
_oi_cache: dict[str, Any] = {"data": {}, "timestamp": None}

POSITIONS_CACHE_TTL = 5  # seconds
FUNDING_CACHE_TTL = 60  # seconds
OI_CACHE_TTL = 60  # seconds


# ---------------------------------------------------------------------------
# Exchange Data Functions (to be mocked in tests)
# ---------------------------------------------------------------------------


async def get_exchange_positions() -> list[dict[str, Any]]:
    """
    Fetch live positions from exchange.

    Uses ExchangeRouter if available, otherwise returns empty.
    Results cached for 5 seconds.
    """
    now = datetime.now()

    # Check cache
    if (
        _positions_cache["timestamp"]
        and (now - _positions_cache["timestamp"]).total_seconds() < POSITIONS_CACHE_TTL
    ):
        return _positions_cache["data"]

    try:
        # Try to get positions from exchange
        from execution.exchange_router import ExchangeRouter

        private_key = os.environ.get("HYPERLIQUID_PRIVATE_KEY", "")
        if not private_key:
            return []

        router = ExchangeRouter(private_key=private_key, testnet=True)
        positions = await router.get_positions()

        result = [
            {
                "symbol": p.symbol,
                "side": p.side,
                "amount": p.amount,
                "entry_price": p.entry_price,
                "mark_price": p.mark_price,
                "unrealized_pnl": p.unrealized_pnl,
                "leverage": p.leverage,
            }
            for p in positions
        ]

        _positions_cache["data"] = result
        _positions_cache["timestamp"] = now
        return result

    except Exception as e:
        logger.warning(f"Failed to fetch positions: {e}")
        return _positions_cache.get("data", [])


async def get_funding_rates() -> dict[str, float]:
    """
    Fetch current funding rates.

    Results cached for 60 seconds.
    """
    now = datetime.now()

    if (
        _funding_cache["timestamp"]
        and (now - _funding_cache["timestamp"]).total_seconds() < FUNDING_CACHE_TTL
    ):
        return _funding_cache["data"]

    try:
        import ccxt.async_support as ccxt

        exchange = ccxt.binance({"enableRateLimit": True})
        exchange.options["defaultType"] = "future"

        # Get funding rates for common symbols
        symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
        funding = {}

        for symbol in symbols:
            try:
                ticker = await exchange.fetch_funding_rate(symbol)
                funding[symbol.split("/")[0]] = ticker.get("fundingRate", 0.0)
            except Exception:
                continue

        await exchange.close()

        _funding_cache["data"] = funding
        _funding_cache["timestamp"] = now
        return funding

    except Exception as e:
        logger.warning(f"Failed to fetch funding rates: {e}")
        return _funding_cache.get("data", {})


async def get_open_interest() -> dict[str, Any]:
    """
    Fetch open interest data.

    Results cached for 60 seconds.
    """
    now = datetime.now()

    if _oi_cache["timestamp"] and (now - _oi_cache["timestamp"]).total_seconds() < OI_CACHE_TTL:
        return _oi_cache["data"]

    try:
        import ccxt.async_support as ccxt

        exchange = ccxt.binance({"enableRateLimit": True})
        exchange.options["defaultType"] = "future"

        symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        oi_data = {}

        for symbol in symbols:
            try:
                oi = await exchange.fetch_open_interest(symbol)
                oi_data[symbol.split("/")[0]] = {
                    "oi": oi.get("openInterestValue", 0.0),
                    "timestamp": oi.get("timestamp"),
                }
            except Exception:
                continue

        await exchange.close()

        _oi_cache["data"] = oi_data
        _oi_cache["timestamp"] = now
        return oi_data

    except Exception as e:
        logger.warning(f"Failed to fetch open interest: {e}")
        return _oi_cache.get("data", {})


# ---------------------------------------------------------------------------
# XGBoost Data Functions
# ---------------------------------------------------------------------------


def get_xgboost_features() -> list[dict[str, Any]]:
    """
    Get SHAP feature importance from latest XGBoost model.

    Reads from saved evaluation if available.
    """
    # Try to load from saved evaluation
    eval_path = Path("evaluation/baseline_metrics.json")

    if eval_path.exists():
        try:
            import json

            with open(eval_path) as f:
                data = json.load(f)
                if "shap_importance" in data:
                    return data["shap_importance"]
        except Exception:
            pass

    # Return placeholder from xgboost_config feature list
    try:
        from signals.xgboost_config import FEATURE_LIST

        return [{"feature": f, "importance": 0.0} for f in FEATURE_LIST]
    except ImportError:
        return []


def get_xgboost_metrics() -> list[dict[str, Any]]:
    """
    Get IC/Brier metrics over time.

    Reads from autoresearch results if available.
    """
    results = read_autoresearch_results()

    return [
        {
            "timestamp": r.get("timestamp"),
            "ic": r.get("ic"),
            "brier": r.get("brier"),
            "sharpe_net": r.get("sharpe_net"),
        }
        for r in results
        if r.get("timestamp")
    ]


# ---------------------------------------------------------------------------
# GET Endpoints (READ-ONLY)
# ---------------------------------------------------------------------------


@app.get("/api/positions")
async def api_positions() -> list[dict[str, Any]]:
    """Get live positions from exchange (cached 5s)."""
    return await get_exchange_positions()


@app.get("/api/pnl/daily")
async def api_pnl_daily() -> dict[str, Any]:
    """Get daily P&L aggregation from order log."""
    orders = read_order_log()
    daily_pnl = compute_daily_pnl(orders)
    return {"daily_pnl": daily_pnl}


@app.get("/api/signals/recent")
async def api_signals_recent() -> list[dict[str, Any]]:
    """Get last 50 signals from signal log."""
    return read_signal_log(limit=50)


@app.get("/api/performance")
async def api_performance() -> dict[str, Any]:
    """Get performance metrics: equity curve, Sharpe, drawdown, win rate."""
    orders = read_order_log()
    equity_curve = compute_equity_curve(orders)
    rolling_sharpe = compute_rolling_sharpe(equity_curve, window=30)
    drawdown = compute_drawdown(equity_curve)
    win_rate = compute_win_rate(orders)

    return {
        "equity_curve": equity_curve,
        "rolling_sharpe": rolling_sharpe,
        "drawdown": drawdown,
        "win_rate": win_rate,
    }


@app.get("/api/xgboost/features")
async def api_xgboost_features() -> dict[str, Any]:
    """Get SHAP feature importance from latest model."""
    return {"features": get_xgboost_features()}


@app.get("/api/xgboost/metrics")
async def api_xgboost_metrics() -> dict[str, Any]:
    """Get IC/Brier metrics over time."""
    return {"metrics": get_xgboost_metrics()}


@app.get("/api/autoresearch/log")
async def api_autoresearch_log() -> list[dict[str, Any]]:
    """Get autoresearch experiment log as JSON."""
    return read_autoresearch_results()


@app.get("/api/risk/funding")
async def api_risk_funding() -> dict[str, float]:
    """Get current funding rates (cached 60s)."""
    return await get_funding_rates()


@app.get("/api/risk/oi")
async def api_risk_oi() -> dict[str, Any]:
    """Get open interest changes (cached 60s)."""
    return await get_open_interest()


@app.get("/api/health")
async def api_health() -> dict[str, Any]:
    """Get watchdog heartbeat and system health."""
    health = read_health_status()

    # Add signal loop status
    signals = read_signal_log(limit=1)
    if signals:
        last_signal_time = signals[0].get("timestamp")
        health["last_signal"] = last_signal_time

    return health


# ---------------------------------------------------------------------------
# WebSocket for Real-Time Updates
# ---------------------------------------------------------------------------


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """
    WebSocket endpoint for real-time dashboard updates.

    Pushes {positions, latest_signal, daily_pnl, health} every 5 seconds.
    """
    await websocket.accept()

    try:
        while True:
            # Gather data
            positions = await get_exchange_positions()
            signals = read_signal_log(limit=1)
            latest_signal = signals[0] if signals else None

            orders = read_order_log()
            daily_pnl = compute_daily_pnl(orders)

            health = read_health_status()

            # Send update
            await websocket.send_json(
                {
                    "positions": positions,
                    "latest_signal": latest_signal,
                    "daily_pnl": daily_pnl,
                    "health": health,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Wait 5 seconds
            await asyncio.sleep(5)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main():
    """Run dashboard API server."""
    import uvicorn

    uvicorn.run(
        "dashboard.api:app",
        host="0.0.0.0",
        port=8420,
        reload=True,
    )


if __name__ == "__main__":
    main()
