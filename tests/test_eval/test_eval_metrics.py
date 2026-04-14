"""Tests for individual evaluation metrics."""

import numpy as np
import pytest

from eval.metrics import (
    MetricValue,
    bootstrap_confidence_interval,
    compute_calmar_ratio,
    compute_information_coefficient,
    compute_max_drawdown,
    compute_profit_factor,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_win_rate,
)


class TestMetricValue:
    """Test MetricValue dataclass."""

    def test_create_metric_value(self):
        """Test creating a metric value."""
        metric = MetricValue(
            value=0.5,
            sample_size=100,
            p_value=0.01,
            ci_lower=0.3,
            ci_upper=0.7,
            confidence_level="high",
        )
        assert metric.value == 0.5
        assert metric.sample_size == 100
        assert metric.p_value == 0.01
        assert metric.confidence_level == "high"

    def test_metric_value_is_frozen(self):
        """Test that MetricValue is immutable."""
        metric = MetricValue(value=0.5, sample_size=100)
        with pytest.raises(AttributeError):
            metric.value = 0.7  # type: ignore


class TestInformationCoefficient:
    """Test IC computation."""

    def test_perfect_positive_correlation_spearman(self):
        """Test perfect positive rank correlation."""
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        ic, p_val = compute_information_coefficient(predicted, actual, method="spearman")

        assert ic == pytest.approx(1.0, abs=0.01)
        assert p_val < 0.05

    def test_perfect_negative_correlation_spearman(self):
        """Test perfect negative rank correlation."""
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.array([50.0, 40.0, 30.0, 20.0, 10.0])

        ic, p_val = compute_information_coefficient(predicted, actual, method="spearman")

        assert ic == pytest.approx(-1.0, abs=0.01)
        assert p_val < 0.05

    def test_no_correlation(self):
        """Test zero correlation."""
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.array([30.0, 10.0, 50.0, 20.0, 40.0])

        ic, p_val = compute_information_coefficient(predicted, actual, method="spearman")

        # Should be close to zero (not exact due to small sample)
        assert abs(ic) <= 0.5  # Relaxed threshold for small samples
        assert p_val > 0.05  # Not significant

    def test_pearson_vs_spearman(self):
        """Test that Pearson and Spearman differ for non-linear relationships."""
        # Non-linear relationship: exponential
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.exp(predicted)  # Exponential growth

        spearman_ic, _ = compute_information_coefficient(predicted, actual, method="spearman")
        pearson_ic, _ = compute_information_coefficient(predicted, actual, method="pearson")

        # Spearman should be 1.0 (perfect rank correlation)
        # Pearson should be < 1.0 (non-linear)
        assert spearman_ic == pytest.approx(1.0, abs=0.01)
        assert pearson_ic < 1.0

    def test_handles_constant_array(self):
        """Test that constant arrays return 0.0 IC."""
        predicted = np.array([1.0, 1.0, 1.0, 1.0])
        actual = np.array([10.0, 20.0, 30.0, 40.0])

        ic, p_val = compute_information_coefficient(predicted, actual, method="spearman")

        assert ic == 0.0
        assert p_val == 1.0

    def test_raises_on_length_mismatch(self):
        """Test that mismatched array lengths raise error."""
        predicted = np.array([1.0, 2.0, 3.0])
        actual = np.array([10.0, 20.0])

        with pytest.raises(ValueError, match="length mismatch"):
            compute_information_coefficient(predicted, actual)

    def test_raises_on_invalid_method(self):
        """Test that invalid method raises error."""
        predicted = np.array([1.0, 2.0, 3.0])
        actual = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid method"):
            compute_information_coefficient(predicted, actual, method="kendall")


class TestSharpeRatio:
    """Test Sharpe ratio computation."""

    def test_positive_sharpe(self):
        """Test Sharpe for positive returns."""
        returns = np.array([0.01, 0.02, -0.005, 0.015, 0.01])

        sharpe = compute_sharpe_ratio(returns, annualization_factor=365)

        # Mean ≈ 0.01, std ≈ 0.0095
        # Sharpe ≈ (0.01 * 365) / (0.0095 * sqrt(365)) ≈ 20.4
        assert sharpe > 0
        assert 15 < sharpe < 25  # High Sharpe due to high mean return

    def test_negative_sharpe(self):
        """Test Sharpe for negative returns."""
        returns = np.array([-0.01, -0.02, 0.005, -0.015, -0.01])

        sharpe = compute_sharpe_ratio(returns, annualization_factor=365)

        assert sharpe < 0

    def test_zero_volatility_returns_inf(self):
        """Test that constant returns give infinite Sharpe."""
        returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01])

        sharpe = compute_sharpe_ratio(returns, annualization_factor=365)

        assert np.isinf(sharpe)
        assert sharpe > 0

    def test_zero_mean_and_volatility_returns_zero(self):
        """Test that zero returns give zero Sharpe."""
        returns = np.array([0.0, 0.0, 0.0, 0.0])

        sharpe = compute_sharpe_ratio(returns, annualization_factor=365)

        assert sharpe == 0.0

    def test_equity_annualization_factor(self):
        """Test with equity markets annualization (252)."""
        returns = np.array([0.01, 0.02, -0.005, 0.015, 0.01])

        sharpe_crypto = compute_sharpe_ratio(returns, annualization_factor=365)
        sharpe_equity = compute_sharpe_ratio(returns, annualization_factor=252)

        # Crypto should have higher Sharpe due to larger annualization
        assert sharpe_crypto > sharpe_equity

    def test_raises_on_empty_returns(self):
        """Test that empty returns array raises error."""
        returns = np.array([])

        with pytest.raises(ValueError, match="empty"):
            compute_sharpe_ratio(returns)


class TestSortinoRatio:
    """Test Sortino ratio computation."""

    def test_positive_sortino(self):
        """Test Sortino for mostly positive returns."""
        # Need at least 2 downside returns for valid std with ddof=1
        returns = np.array([0.02, 0.01, -0.005, 0.015, 0.01, -0.003, 0.02])

        sortino = compute_sortino_ratio(returns, annualization_factor=365)

        assert sortino > 0

    def test_no_downside_returns_inf(self):
        """Test that all positive returns give infinite Sortino."""
        returns = np.array([0.01, 0.02, 0.015, 0.01])

        sortino = compute_sortino_ratio(returns, annualization_factor=365)

        assert np.isinf(sortino)
        assert sortino > 0

    def test_sortino_higher_than_sharpe(self):
        """Test that Sortino is typically higher than Sharpe (same upside, less penalty)."""
        # Positive skew: few large losses, many small gains
        # Need multiple downside returns for valid std calculation
        returns = np.array([0.01, 0.01, 0.01, 0.01, -0.02, -0.01, 0.01])

        sharpe = compute_sharpe_ratio(returns, annualization_factor=365)
        sortino = compute_sortino_ratio(returns, annualization_factor=365)

        # Sortino should be higher (only penalizes downside)
        assert sortino > sharpe

    def test_raises_on_empty_returns(self):
        """Test that empty returns array raises error."""
        with pytest.raises(ValueError, match="empty"):
            compute_sortino_ratio(np.array([]))


class TestMaxDrawdown:
    """Test max drawdown computation."""

    def test_no_drawdown_uptrend(self):
        """Test zero drawdown for monotonic uptrend."""
        cumulative = np.array([0.01, 0.03, 0.06, 0.10])

        max_dd = compute_max_drawdown(cumulative)

        assert max_dd == 0.0

    def test_single_drawdown(self):
        """Test single drawdown period."""
        # Starts at 0, goes to 0.10, drops to 0.05, recovers to 0.12
        cumulative = np.array([0.0, 0.05, 0.10, 0.05, 0.08, 0.12])

        max_dd = compute_max_drawdown(cumulative)

        # Peak at 0.10, trough at 0.05 → (1.10 - 1.05) / 1.10 ≈ 0.0455
        assert 0.04 < max_dd < 0.05

    def test_multiple_drawdowns(self):
        """Test that maximum of multiple drawdowns is returned."""
        # First DD: 10% → 5% (5% DD)
        # Second DD: 12% → 3% (9% DD) ← should be max
        cumulative = np.array([0.0, 0.10, 0.05, 0.12, 0.03, 0.10])

        max_dd = compute_max_drawdown(cumulative)

        # Peak at 0.12, trough at 0.03 → (1.12 - 1.03) / 1.12 ≈ 0.0804
        assert 0.07 < max_dd < 0.09

    def test_negative_cumulative_returns(self):
        """Test drawdown with negative cumulative returns."""
        # Losing strategy overall
        cumulative = np.array([0.0, -0.05, -0.02, -0.10])

        max_dd = compute_max_drawdown(cumulative)

        # Peak at 0.0, trough at -0.10 → (1.0 - 0.9) / 1.0 = 0.1
        assert max_dd == pytest.approx(0.1, abs=0.01)

    def test_raises_on_empty_array(self):
        """Test that empty array raises error."""
        with pytest.raises(ValueError, match="empty"):
            compute_max_drawdown(np.array([]))


class TestCalmarRatio:
    """Test Calmar ratio computation."""

    def test_positive_calmar(self):
        """Test Calmar for profitable strategy with drawdown."""
        # Positive mean return with some volatility
        returns = np.array([0.01, -0.005, 0.015, -0.003, 0.01, 0.02, -0.01])

        calmar = compute_calmar_ratio(returns, annualization_factor=365)

        assert calmar > 0

    def test_no_drawdown_returns_inf(self):
        """Test that no drawdown gives infinite Calmar."""
        returns = np.array([0.01, 0.01, 0.01, 0.01])

        calmar = compute_calmar_ratio(returns, annualization_factor=365)

        assert np.isinf(calmar)
        assert calmar > 0

    def test_negative_returns_negative_calmar(self):
        """Test Calmar for losing strategy."""
        returns = np.array([-0.01, -0.02, 0.005, -0.015])

        calmar = compute_calmar_ratio(returns, annualization_factor=365)

        assert calmar < 0

    def test_raises_on_empty_returns(self):
        """Test that empty returns raises error."""
        with pytest.raises(ValueError, match="empty"):
            compute_calmar_ratio(np.array([]))


class TestWinRate:
    """Test win rate computation."""

    def test_all_wins(self):
        """Test 100% win rate."""
        profitable = np.array([True, True, True, True])

        win_rate = compute_win_rate(profitable)

        assert win_rate == 1.0

    def test_all_losses(self):
        """Test 0% win rate."""
        profitable = np.array([False, False, False, False])

        win_rate = compute_win_rate(profitable)

        assert win_rate == 0.0

    def test_fifty_percent_win_rate(self):
        """Test 50% win rate."""
        profitable = np.array([True, False, True, False])

        win_rate = compute_win_rate(profitable)

        assert win_rate == 0.5

    def test_raises_on_empty_array(self):
        """Test that empty array raises error."""
        with pytest.raises(ValueError, match="empty"):
            compute_win_rate(np.array([]))


class TestProfitFactor:
    """Test profit factor computation."""

    def test_profitable_strategy(self):
        """Test profit factor > 1 for profitable strategy."""
        # Gross profit: 0.02 + 0.03 = 0.05
        # Gross loss: |-0.01| = 0.01
        # PF = 0.05 / 0.01 = 5.0
        returns = np.array([0.02, -0.01, 0.03])

        pf = compute_profit_factor(returns)

        assert pf == pytest.approx(5.0, abs=0.1)

    def test_unprofitable_strategy(self):
        """Test profit factor < 1 for unprofitable strategy."""
        # Gross profit: 0.01
        # Gross loss: 0.05
        # PF = 0.01 / 0.05 = 0.2
        returns = np.array([0.01, -0.02, -0.03])

        pf = compute_profit_factor(returns)

        assert pf == pytest.approx(0.2, abs=0.1)

    def test_no_losses_returns_inf(self):
        """Test that all wins give infinite profit factor."""
        returns = np.array([0.01, 0.02, 0.03])

        pf = compute_profit_factor(returns)

        assert np.isinf(pf)
        assert pf > 0

    def test_no_wins_returns_zero(self):
        """Test that all losses give zero profit factor."""
        returns = np.array([-0.01, -0.02, -0.03])

        pf = compute_profit_factor(returns)

        assert pf == 0.0

    def test_raises_on_empty_returns(self):
        """Test that empty returns raises error."""
        with pytest.raises(ValueError, match="empty"):
            compute_profit_factor(np.array([]))


class TestBootstrapConfidenceInterval:
    """Test bootstrap CI computation."""

    def test_bootstrap_mean_ci(self):
        """Test bootstrap CI for mean."""
        np.random.seed(42)
        data = np.random.normal(loc=0.01, scale=0.02, size=100)

        ci_lower, ci_upper = bootstrap_confidence_interval(
            data,
            metric_fn=lambda x: np.mean(x),
            n_bootstrap=500,
        )

        # CI should contain true mean
        assert ci_lower < 0.01 < ci_upper
        assert ci_lower < ci_upper

    def test_bootstrap_sharpe_ci(self):
        """Test bootstrap CI for Sharpe ratio."""
        np.random.seed(42)
        returns = np.random.normal(loc=0.01, scale=0.02, size=100)

        ci_lower, ci_upper = bootstrap_confidence_interval(
            returns,
            metric_fn=lambda x: compute_sharpe_ratio(x, annualization_factor=365),
            n_bootstrap=500,
        )

        # CI should be reasonable
        assert ci_lower < ci_upper
        assert not np.isnan(ci_lower)
        assert not np.isnan(ci_upper)

    def test_raises_on_empty_data(self):
        """Test that empty data raises error."""
        with pytest.raises(ValueError, match="empty"):
            bootstrap_confidence_interval(
                np.array([]),
                metric_fn=lambda x: np.mean(x),
            )
