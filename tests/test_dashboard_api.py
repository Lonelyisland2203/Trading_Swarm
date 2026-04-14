"""
Tests for dashboard FastAPI backend.

TDD RED phase — all tests written before implementation.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_data_dir(tmp_path: Path):
    """Create temp directory with test data files."""
    # Signal log
    signal_log = tmp_path / "signals" / "signal_log.jsonl"
    signal_log.parent.mkdir(parents=True, exist_ok=True)
    signals = [
        {
            "timestamp": "2024-01-01T10:00:00",
            "symbol": "BTC",
            "timeframe": "1h",
            "direction": "LONG",
            "confidence": 0.75,
            "persona": "momentum",
            "market_regime": "trending",
            "reasoning": "Strong momentum",
            "current_price": 45000.0,
            "critic_score": 0.8,
            "critic_recommendation": "APPROVE",
            "critic_override": False,
            "final_direction": "LONG",
            "executed": True,
            "trade_decision_reason": "High confidence",
        },
        {
            "timestamp": "2024-01-01T11:00:00",
            "symbol": "ETH",
            "timeframe": "1h",
            "direction": "SHORT",
            "confidence": 0.65,
            "persona": "mean_reversion",
            "market_regime": "ranging",
            "reasoning": "Overbought",
            "current_price": 2500.0,
            "critic_score": 0.6,
            "critic_recommendation": "APPROVE",
            "critic_override": False,
            "final_direction": "SHORT",
            "executed": True,
            "trade_decision_reason": "Medium confidence",
        },
    ]
    for sig in signals:
        with open(signal_log, "a") as f:
            f.write(json.dumps(sig) + "\n")

    # Order log
    order_log = tmp_path / "execution" / "order_log.jsonl"
    order_log.parent.mkdir(parents=True, exist_ok=True)
    orders = [
        {
            "timestamp": "2024-01-01T10:00:05",
            "order_id": "123",
            "symbol": "BTC",
            "side": "buy",
            "order_type": "market",
            "amount": 0.1,
            "price": 45000.0,
            "status": "filled",
            "pnl": 150.0,
        },
        {
            "timestamp": "2024-01-01T11:00:05",
            "order_id": "124",
            "symbol": "ETH",
            "side": "sell",
            "order_type": "market",
            "amount": 1.0,
            "price": 2500.0,
            "status": "filled",
            "pnl": -50.0,
        },
    ]
    for order in orders:
        with open(order_log, "a") as f:
            f.write(json.dumps(order) + "\n")

    # Autoresearch results.tsv
    autoresearch_dir = tmp_path / "autoresearch"
    autoresearch_dir.mkdir(parents=True, exist_ok=True)
    results_tsv = autoresearch_dir / "results.tsv"
    results_tsv.write_text(
        "experiment_id\ttimestamp\tchange_description\tsharpe_net\tic\tbrier\taccuracy\tfalse_bullish_rate\tkept_or_reverted\n"
        "exp_001\t2024-01-01T12:00:00\tIncrease n_estimators\t0.85\t0.12\t0.22\t0.65\t0.08\tkept\n"
        "exp_002\t2024-01-02T12:00:00\tReduce learning_rate\t0.72\t0.10\t0.24\t0.60\t0.10\treverted\n"
    )

    # Health status
    health_file = tmp_path / "dashboard" / "health_status.json"
    health_file.parent.mkdir(parents=True, exist_ok=True)
    health_file.write_text(
        json.dumps(
            {
                "watchdog_heartbeat": "2024-01-01T12:00:00",
                "positions_count": 2,
                "daily_pnl_usd": 100.0,
                "max_daily_loss_triggered": False,
            }
        )
    )

    return tmp_path


@pytest.fixture
def client(temp_data_dir: Path):
    """Create test client with mocked data paths."""
    # Patch the paths used by data_readers before importing api
    with (
        patch("dashboard.data_readers.SIGNAL_LOG_PATH", temp_data_dir / "signals" / "signal_log.jsonl"),
        patch("dashboard.data_readers.ORDER_LOG_PATH", temp_data_dir / "execution" / "order_log.jsonl"),
        patch("dashboard.data_readers.AUTORESEARCH_RESULTS_PATH", temp_data_dir / "autoresearch" / "results.tsv"),
        patch("dashboard.data_readers.HEALTH_STATUS_PATH", temp_data_dir / "dashboard" / "health_status.json"),
    ):
        from dashboard.api import app

        yield TestClient(app)


@pytest.fixture
def app_instance(temp_data_dir: Path) -> FastAPI:
    """Get the FastAPI app for route inspection."""
    with (
        patch("dashboard.data_readers.SIGNAL_LOG_PATH", temp_data_dir / "signals" / "signal_log.jsonl"),
        patch("dashboard.data_readers.ORDER_LOG_PATH", temp_data_dir / "execution" / "order_log.jsonl"),
        patch("dashboard.data_readers.AUTORESEARCH_RESULTS_PATH", temp_data_dir / "autoresearch" / "results.tsv"),
        patch("dashboard.data_readers.HEALTH_STATUS_PATH", temp_data_dir / "dashboard" / "health_status.json"),
    ):
        from dashboard.api import app

        return app


# ---------------------------------------------------------------------------
# Test: All GET endpoints return 200
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "endpoint",
    [
        "/api/positions",
        "/api/pnl/daily",
        "/api/signals/recent",
        "/api/performance",
        "/api/xgboost/features",
        "/api/xgboost/metrics",
        "/api/autoresearch/log",
        "/api/risk/funding",
        "/api/risk/oi",
        "/api/health",
    ],
)
def test_all_get_endpoints_return_200(client: TestClient, endpoint: str):
    """All GET endpoints must return HTTP 200."""
    # Mock exchange calls for positions/funding/oi endpoints
    with (
        patch("dashboard.api.get_exchange_positions", new_callable=AsyncMock, return_value=[]),
        patch("dashboard.api.get_funding_rates", new_callable=AsyncMock, return_value={}),
        patch("dashboard.api.get_open_interest", new_callable=AsyncMock, return_value={}),
    ):
        response = client.get(endpoint)
        assert response.status_code == 200, f"{endpoint} returned {response.status_code}: {response.text}"


# ---------------------------------------------------------------------------
# Test: No write endpoints (Architecture Constraint #11)
# ---------------------------------------------------------------------------


def test_no_write_endpoints(app_instance: FastAPI):
    """
    Dashboard is READ-ONLY. No POST, PUT, DELETE endpoints allowed.

    Architecture Constraint #11: Dashboard is READ-ONLY.
    """
    forbidden_methods = {"POST", "PUT", "DELETE", "PATCH"}
    violations = []

    for route in app_instance.routes:
        if hasattr(route, "methods"):
            for method in route.methods:
                if method in forbidden_methods:
                    violations.append(f"{method} {route.path}")

    assert violations == [], f"Found write endpoints (violates Constraint #11): {violations}"


# ---------------------------------------------------------------------------
# Test: WebSocket connection
# ---------------------------------------------------------------------------


def test_websocket_connection(client: TestClient):
    """WebSocket /ws/live must accept connection and send at least one message."""
    with (
        patch("dashboard.api.get_exchange_positions", new_callable=AsyncMock, return_value=[]),
        patch("dashboard.api.get_funding_rates", new_callable=AsyncMock, return_value={}),
    ):
        with client.websocket_connect("/ws/live") as websocket:
            # Should receive at least one message within timeout
            data = websocket.receive_json()
            assert "positions" in data
            assert "latest_signal" in data
            assert "daily_pnl" in data
            assert "health" in data


# ---------------------------------------------------------------------------
# Test: Positions response schema
# ---------------------------------------------------------------------------


def test_positions_response_schema(client: TestClient):
    """Positions endpoint must return list with expected fields."""
    mock_positions = [
        {
            "symbol": "BTC",
            "side": "long",
            "amount": 0.1,
            "entry_price": 45000.0,
            "mark_price": 46000.0,
            "unrealized_pnl": 100.0,
            "leverage": 1,
        }
    ]

    with patch("dashboard.api.get_exchange_positions", new_callable=AsyncMock, return_value=mock_positions):
        response = client.get("/api/positions")
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        if data:
            pos = data[0]
            assert "symbol" in pos
            assert "side" in pos
            assert "amount" in pos
            assert "entry_price" in pos
            assert "unrealized_pnl" in pos


# ---------------------------------------------------------------------------
# Test: Signals recent limit
# ---------------------------------------------------------------------------


def test_signals_recent_limit(client: TestClient):
    """GET /api/signals/recent must return at most 50 signals."""
    response = client.get("/api/signals/recent")
    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) <= 50

    # Check signal has expected fields
    if data:
        sig = data[0]
        assert "timestamp" in sig
        assert "symbol" in sig
        assert "direction" in sig
        assert "confidence" in sig


# ---------------------------------------------------------------------------
# Test: Autoresearch log parsing
# ---------------------------------------------------------------------------


def test_autoresearch_log_parsing(client: TestClient):
    """GET /api/autoresearch/log must parse results.tsv into JSON array."""
    response = client.get("/api/autoresearch/log")
    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) == 2  # Two experiments in fixture

    exp = data[0]
    assert "experiment_id" in exp
    assert "timestamp" in exp
    assert "change_description" in exp
    assert "sharpe_net" in exp
    assert "ic" in exp
    assert "brier" in exp
    assert "accuracy" in exp
    assert "false_bullish_rate" in exp
    assert "kept_or_reverted" in exp


# ---------------------------------------------------------------------------
# Test: Health endpoint
# ---------------------------------------------------------------------------


def test_health_endpoint(client: TestClient):
    """GET /api/health must return watchdog heartbeat data."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()

    assert "watchdog_heartbeat" in data
    assert "positions_count" in data
    assert "daily_pnl_usd" in data


# ---------------------------------------------------------------------------
# Test: Performance metrics
# ---------------------------------------------------------------------------


def test_performance_metrics(client: TestClient):
    """GET /api/performance must include equity curve, Sharpe, drawdown."""
    response = client.get("/api/performance")
    assert response.status_code == 200
    data = response.json()

    assert "equity_curve" in data
    assert "rolling_sharpe" in data
    assert "drawdown" in data
    assert "win_rate" in data

    # Equity curve is list of {timestamp, cumulative_pnl}
    assert isinstance(data["equity_curve"], list)
    # Rolling Sharpe is list of {timestamp, sharpe}
    assert isinstance(data["rolling_sharpe"], list)
    # Drawdown is list of {timestamp, drawdown_pct}
    assert isinstance(data["drawdown"], list)


# ---------------------------------------------------------------------------
# Test: Data readers - missing files return empty
# ---------------------------------------------------------------------------


def test_data_readers_missing_files():
    """Data readers must handle missing files gracefully."""
    from dashboard.data_readers import (
        read_signal_log,
        read_order_log,
        read_autoresearch_results,
        read_health_status,
    )

    with (
        patch("dashboard.data_readers.SIGNAL_LOG_PATH", Path("/nonexistent/signal_log.jsonl")),
        patch("dashboard.data_readers.ORDER_LOG_PATH", Path("/nonexistent/order_log.jsonl")),
        patch("dashboard.data_readers.AUTORESEARCH_RESULTS_PATH", Path("/nonexistent/results.tsv")),
        patch("dashboard.data_readers.HEALTH_STATUS_PATH", Path("/nonexistent/health.json")),
    ):
        assert read_signal_log() == []
        assert read_order_log() == []
        assert read_autoresearch_results() == []

        health = read_health_status()
        assert "watchdog_heartbeat" in health  # Should have default


# ---------------------------------------------------------------------------
# Test: Equity curve computation
# ---------------------------------------------------------------------------


def test_compute_equity_curve():
    """compute_equity_curve must produce cumulative P&L from orders."""
    from dashboard.data_readers import compute_equity_curve

    orders = [
        {"timestamp": "2024-01-01T10:00:00", "pnl": 100.0},
        {"timestamp": "2024-01-01T11:00:00", "pnl": -30.0},
        {"timestamp": "2024-01-01T12:00:00", "pnl": 50.0},
    ]

    curve = compute_equity_curve(orders)

    assert len(curve) == 3
    assert curve[0]["cumulative_pnl"] == 100.0
    assert curve[1]["cumulative_pnl"] == 70.0  # 100 - 30
    assert curve[2]["cumulative_pnl"] == 120.0  # 70 + 50


# ---------------------------------------------------------------------------
# Test: Rolling Sharpe computation
# ---------------------------------------------------------------------------


def test_compute_rolling_sharpe():
    """compute_rolling_sharpe must return Sharpe values over rolling window."""
    from dashboard.data_readers import compute_rolling_sharpe

    # Need enough data points for window
    equity_curve = [
        {"timestamp": f"2024-01-{i+1:02d}T12:00:00", "cumulative_pnl": i * 10.0}
        for i in range(35)
    ]

    result = compute_rolling_sharpe(equity_curve, window=30)

    # 35 equity points → 34 returns → need 30 returns per window → 5 valid windows
    assert len(result) == 5  # 34 - 30 + 1 = 5 points with full window
    for item in result:
        assert "timestamp" in item
        assert "sharpe" in item


# ---------------------------------------------------------------------------
# Test: Drawdown computation
# ---------------------------------------------------------------------------


def test_compute_drawdown():
    """compute_drawdown must track drawdown from peak."""
    from dashboard.data_readers import compute_drawdown

    equity_curve = [
        {"timestamp": "2024-01-01T10:00:00", "cumulative_pnl": 100.0},
        {"timestamp": "2024-01-01T11:00:00", "cumulative_pnl": 150.0},  # New peak
        {"timestamp": "2024-01-01T12:00:00", "cumulative_pnl": 120.0},  # Drawdown
        {"timestamp": "2024-01-01T13:00:00", "cumulative_pnl": 100.0},  # Deeper drawdown
        {"timestamp": "2024-01-01T14:00:00", "cumulative_pnl": 200.0},  # Recovery + new peak
    ]

    result = compute_drawdown(equity_curve)

    assert len(result) == 5
    assert result[0]["drawdown_pct"] == 0.0  # First point, no drawdown
    assert result[1]["drawdown_pct"] == 0.0  # At peak
    assert result[2]["drawdown_pct"] == pytest.approx(0.2, rel=0.01)  # 30/150 = 20%
    assert result[3]["drawdown_pct"] == pytest.approx(0.333, rel=0.01)  # 50/150 = 33.3%
    assert result[4]["drawdown_pct"] == 0.0  # New peak


# ---------------------------------------------------------------------------
# Test: XGBoost features endpoint
# ---------------------------------------------------------------------------


def test_xgboost_features_endpoint(client: TestClient):
    """GET /api/xgboost/features must return SHAP feature importance."""
    response = client.get("/api/xgboost/features")
    assert response.status_code == 200
    data = response.json()

    assert "features" in data
    assert isinstance(data["features"], list)


# ---------------------------------------------------------------------------
# Test: XGBoost metrics endpoint
# ---------------------------------------------------------------------------


def test_xgboost_metrics_endpoint(client: TestClient):
    """GET /api/xgboost/metrics must return IC/Brier over time."""
    response = client.get("/api/xgboost/metrics")
    assert response.status_code == 200
    data = response.json()

    assert "metrics" in data
    assert isinstance(data["metrics"], list)


# ---------------------------------------------------------------------------
# Test: Funding rates endpoint
# ---------------------------------------------------------------------------


def test_funding_rates_endpoint(client: TestClient):
    """GET /api/risk/funding must return funding rate data."""
    mock_funding = {"BTC": 0.0001, "ETH": -0.0002}

    with patch("dashboard.api.get_funding_rates", new_callable=AsyncMock, return_value=mock_funding):
        response = client.get("/api/risk/funding")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# Test: Open interest endpoint
# ---------------------------------------------------------------------------


def test_open_interest_endpoint(client: TestClient):
    """GET /api/risk/oi must return open interest data."""
    mock_oi = {"BTC": {"oi": 50000.0, "delta_24h": 1000.0}}

    with patch("dashboard.api.get_open_interest", new_callable=AsyncMock, return_value=mock_oi):
        response = client.get("/api/risk/oi")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# Test: Daily P&L endpoint
# ---------------------------------------------------------------------------


def test_daily_pnl_endpoint(client: TestClient):
    """GET /api/pnl/daily must return daily P&L aggregation."""
    response = client.get("/api/pnl/daily")
    assert response.status_code == 200
    data = response.json()

    assert "daily_pnl" in data
    assert isinstance(data["daily_pnl"], list)
