"""
Configuration for evaluation layer.

Defines evaluation settings and sample size requirements for statistical validity.
"""

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class SampleSizeRequirements:
    """
    Sample size thresholds for metric computation confidence levels.

    - minimum: Below this, return None (insufficient data)
    - marginal: Between minimum and marginal, log warning (low confidence)
    - adequate: Between marginal and adequate (moderate confidence)
    - robust: Above adequate (high confidence)
    """

    minimum: int = 30  # CLT threshold, below this return None
    marginal: int = 60  # Low confidence region
    adequate: int = 100  # Moderate confidence threshold
    robust: int = 250  # High confidence threshold

    def get_confidence_level(self, sample_size: int) -> str | None:
        """
        Determine confidence level for given sample size.

        Args:
            sample_size: Number of samples

        Returns:
            'low', 'moderate', 'high', or None if below minimum
        """
        if sample_size < self.minimum:
            return None
        if sample_size < self.marginal:
            return "low"
        if sample_size < self.adequate:
            return "moderate"
        return "high"


@dataclass(slots=True, frozen=True)
class EvaluationConfig:
    """
    Evaluation layer configuration.

    Controls metric computation parameters and statistical testing.
    """

    annualization_factor: int = 365  # 365 for crypto (24/7), 252 for equities
    min_sample_size: int = 30  # Minimum observations for any metric
    fdr_alpha: float = 0.05  # FDR threshold for multiple testing correction
    bootstrap_samples: int = 1000  # Bootstrap iterations for confidence intervals
    confidence_buckets: int = 5  # Quantile buckets for calibration analysis

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.annualization_factor <= 0:
            raise ValueError(
                f"annualization_factor must be positive, got {self.annualization_factor}"
            )
        if self.min_sample_size < 2:
            raise ValueError(f"min_sample_size must be >= 2, got {self.min_sample_size}")
        if not 0 < self.fdr_alpha < 1:
            raise ValueError(f"fdr_alpha must be in (0, 1), got {self.fdr_alpha}")
        if self.bootstrap_samples < 100:
            raise ValueError(f"bootstrap_samples must be >= 100, got {self.bootstrap_samples}")
        if self.confidence_buckets < 2:
            raise ValueError(f"confidence_buckets must be >= 2, got {self.confidence_buckets}")
