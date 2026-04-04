"""
Reward computation configuration.

Defines scaling parameters to normalize reward component magnitudes.
"""

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class RewardScaling:
    """
    Scale factors to bring reward components to similar magnitudes.
    
    These scale factors should be calibrated based on your asset's typical:
    - Return distribution (e.g., if typical returns are ~10%, use return_scale=10.0)
    - MAE distribution (e.g., if typical MAE is ~10%, use mae_scale=10.0)
    
    The directional component is already bounded [-1, +1] from confidence weighting.
    
    Attributes:
        return_scale: Multiplier for net_return to bring to [-1, 1] range
        mae_scale: Multiplier for MAE to bring to [-1, 0] range
        
    Example:
        >>> scaling = RewardScaling(return_scale=10.0, mae_scale=10.0)
        >>> # A 5% return becomes 0.5 reward contribution
        >>> # A -5% MAE becomes -0.5 penalty contribution
    """
    
    return_scale: float = 10.0
    mae_scale: float = 10.0
    
    def __post_init__(self):
        """Validate scaling parameters."""
        if self.return_scale <= 0:
            raise ValueError(f"return_scale must be positive, got {self.return_scale}")
        if self.mae_scale <= 0:
            raise ValueError(f"mae_scale must be positive, got {self.mae_scale}")
