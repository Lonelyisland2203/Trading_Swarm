"""
Configuration settings for Trading Swarm.

Uses Pydantic BaseSettings for environment variable loading and validation.
All settings are loaded from .env file and validated on startup.
"""

import os
from typing import List
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.fee_model import FeeModelSettings


class OllamaSettings(BaseModel):
    """Ollama LLM service configuration."""

    base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    generator_model: str = Field(
        default="qwen3:8b", description="Generator model tag (Qwen3-8B, non-thinking mode)"
    )
    critic_model: str = Field(
        default="deepseek-r1:14b",
        description="Critic model tag (DeepSeek-R1-14B, native reasoning)",
    )
    keep_alive: int = Field(
        default=0,
        ge=0,
        description="Model keep-alive seconds (0=immediate unload, REQUIRED for VRAM management)",
    )
    timeout: int = Field(default=300, ge=30, le=600, description="Request timeout in seconds")

    @field_validator("keep_alive")
    @classmethod
    def validate_keep_alive(cls, v: int) -> int:
        """Enforce keep_alive=0 for VRAM management."""
        if v != 0:
            raise ValueError(
                "OLLAMA_KEEP_ALIVE must be 0 to prevent VRAM exhaustion. "
                "Models must unload between generation/critique phases."
            )
        return v


class SwarmSettings(BaseModel):
    """Swarm orchestration configuration."""

    generator_personas: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of generator personas (diversity vs compute trade-off)",
    )
    critique_enabled: bool = Field(
        default=True, description="Enable critic agent (disable for testing only)"
    )
    dpo_batch_size: int = Field(
        default=4, ge=1, le=16, description="Batch size for DPO training dataset collection"
    )
    training_enabled: bool = Field(
        default=False, description="Enable DPO training (WARNING: runs in separate process)"
    )
    concurrency: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Max concurrent LLM requests (2 recommended for single GPU)",
    )


class RewardWeights(BaseModel):
    """
    Reward function weights for DPO training.

    Per-example reward computation using three components:
    - return_weight: Net return after transaction costs
    - directional_weight: Confidence-weighted directional accuracy
    - mae_weight: Max adverse excursion penalty

    All weights must sum to 1.0 for proper normalization.
    """

    return_weight: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Net return component weight (after transaction costs)",
    )
    directional_weight: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Directional accuracy component weight (confidence-weighted)",
    )
    mae_weight: float = Field(
        default=0.20, ge=0.0, le=1.0, description="Max adverse excursion penalty weight"
    )

    @model_validator(mode="after")
    def validate_sum(self) -> "RewardWeights":
        """Ensure weights sum to 1.0."""
        total = self.return_weight + self.directional_weight + self.mae_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Reward weights must sum to 1.0, got {total:.6f}. "
                f"Current: return={self.return_weight}, directional={self.directional_weight}, "
                f"mae={self.mae_weight}"
            )
        return self


class MarketDataSettings(BaseModel):
    """Market data fetching configuration."""

    exchange: str = Field(
        default="binance", description="CCXT exchange name (binance, coinbase, kraken, etc.)"
    )
    symbols: List[str] = Field(
        default=["BTC/USDT", "ETH/USDT", "BNB/USDT"], description="Trading pairs to fetch"
    )
    timeframe: str = Field(default="1h", description="Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d)")
    lookback_bars: int = Field(
        default=100,
        ge=20,
        le=1000,
        description="Number of historical bars for indicator calculation",
    )
    cache_dir: Path = Field(default=Path("data/cache"), description="Market data cache directory")
    cache_size_limit: int = Field(
        default=1_073_741_824,  # 1 GB
        ge=104_857_600,  # 100 MB minimum
        description="Disk cache size limit in bytes",
    )

    @field_validator("symbols", mode="before")
    @classmethod
    def parse_symbols(cls, v):
        """Parse comma-separated symbols from environment variable."""
        if isinstance(v, str):
            return [s.strip() for s in v.split(",")]
        return v

    @field_validator("cache_dir", mode="before")
    @classmethod
    def parse_cache_dir(cls, v):
        """Convert string path to Path object."""
        if isinstance(v, str):
            return Path(v)
        return v


class DPOTrainingSettings(BaseModel):
    """DPO fine-tuning configuration (Process B only)."""

    # LoRA hyperparameters
    lora_rank: int = Field(
        default=32,
        ge=8,
        le=128,
        description="LoRA rank (r). Higher = more expressive but more VRAM. Recommended: 32",
    )
    lora_alpha: int = Field(
        default=64,
        ge=16,
        le=256,
        description="LoRA alpha scaling. Typically 2x rank. Recommended: 64",
    )
    lora_dropout: float = Field(
        default=0.05,
        ge=0.0,
        le=0.2,
        description="LoRA dropout for regularization. Recommended: 0.05",
    )
    lora_target_modules: List[str] = Field(
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",  # Attention
            "gate_proj",
            "up_proj",
            "down_proj",  # MLP
        ],
        description="Target modules for LoRA. Attention + MLP for maximum capacity.",
    )

    # DPO hyperparameters
    dpo_beta: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        description="DPO beta parameter. Controls preference strength. Start at 0.1.",
    )
    dpo_loss_type: str = Field(
        default="sigmoid", description="DPO loss type. 'sigmoid' is standard."
    )

    # Training hyperparameters
    learning_rate: float = Field(
        default=5e-6,
        ge=1e-7,
        le=1e-4,
        description="Learning rate. Conservative for DPO. Recommended: 5e-6",
    )
    batch_size: int = Field(
        default=1, ge=1, le=4, description="Per-device batch size. Must be 1 for 16 GB VRAM."
    )
    gradient_accumulation_steps: int = Field(
        default=16,
        ge=1,
        le=64,
        description="Gradient accumulation steps. Effective batch = batch_size * this.",
    )
    max_length: int = Field(
        default=1200,
        ge=512,
        le=4096,
        description=(
            "Maximum sequence length (prompt + chosen/rejected response). "
            "Dataset prompts are ~968 tokens + ~150 token responses = ~1120 total. "
            "1200 gives headroom without padding to 2048 which quadruples attention cost."
        ),
    )
    num_epochs: int = Field(
        default=1, ge=1, le=5, description="Number of training epochs. 1 recommended for DPO."
    )

    # Walk-forward validation
    min_training_pairs: int = Field(
        default=500,
        ge=100,
        description="Minimum preference pairs before training. Below this, wait.",
    )
    train_window: int = Field(
        default=500, ge=100, description="Number of pairs in training window."
    )
    test_window: int = Field(
        default=100, ge=50, description="Number of pairs in test window (~10-20% of train)."
    )
    retrain_threshold: int = Field(
        default=250,
        ge=50,
        description="Trigger retraining after N new pairs (quarter of train window).",
    )

    # Replay buffer (catastrophic forgetting mitigation)
    replay_ratio: float = Field(
        default=0.15, ge=0.0, le=0.3, description="Fraction of training batch from replay buffer."
    )
    replay_buffer_size: int = Field(
        default=1000, ge=100, description="Maximum pairs in replay buffer."
    )

    # Evaluation & promotion criteria
    min_oos_ic: float = Field(
        default=0.05, ge=0.0, le=1.0, description="Minimum out-of-sample IC for promotion."
    )
    min_ic_delta: float = Field(
        default=0.02, ge=0.0, le=0.5, description="Minimum IC improvement for promotion."
    )
    max_brier_score: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Maximum Brier score (calibration) for promotion."
    )
    min_directional_accuracy: float = Field(
        default=0.52,
        ge=0.5,
        le=1.0,
        description="Minimum directional accuracy for promotion (> random).",
    )

    # Paths
    adapter_dir: Path = Field(
        default=Path("models/adapters/qwen3-8b-dpo"),
        description="Directory for saving LoRA adapters.",
    )
    base_model_id: str = Field(
        default="Qwen/Qwen3-8B", description="HuggingFace model ID for base model."
    )

    @field_validator("adapter_dir", mode="before")
    @classmethod
    def parse_adapter_dir(cls, v):
        """Convert string path to Path object."""
        if isinstance(v, str):
            return Path(v)
        return v


class DatasetGenerationSettings(BaseModel):
    """Configuration for comprehensive dataset generation."""

    default_window_count: int = Field(
        default=15,
        ge=1,
        le=100,
        description="Default number of historical windows per symbol/timeframe",
    )
    default_window_stride: int = Field(
        default=100, ge=10, le=1000, description="Default stride between windows (in bars)"
    )
    min_data_completeness: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Minimum data completeness ratio (skip windows below this threshold)",
    )
    save_frequency: int = Field(
        default=1, ge=1, description="Save after every N contexts (1=after each context)"
    )
    max_retries_per_job: int = Field(
        default=2, ge=0, le=5, description="Max retries for failed inference jobs"
    )
    retry_delay_seconds: int = Field(
        default=10, ge=1, le=300, description="Delay between retries (seconds)"
    )


class ExecutionSettings(BaseModel):
    """Execution layer configuration with safety limits."""

    # Connection settings
    testnet: bool = Field(default=True, description="Use testnet")
    api_key: str = Field(default="", description="Binance API key")
    api_secret: str = Field(default="", description="Binance API secret")

    # Trading mode
    mode: str = Field(default="futures", pattern="^(spot|futures)$", description="Trading mode")

    # Safety limits
    max_daily_trades: int = Field(default=10, ge=1, le=100, description="Maximum trades per day")
    max_daily_loss_pct: float = Field(
        default=2.0, ge=0.1, le=10.0, description="Maximum daily loss %"
    )
    max_open_positions: int = Field(
        default=3, ge=1, le=20, description="Maximum concurrent positions"
    )
    max_position_pct: float = Field(
        default=0.02, ge=0.001, le=0.10, description="Maximum position size %"
    )
    order_cooldown_seconds: int = Field(
        default=60, ge=0, le=3600, description="Minimum seconds between orders"
    )

    # Signal thresholds
    min_confidence: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Minimum signal confidence"
    )
    min_expected_return_pct: float = Field(
        default=0.1, ge=0.0, description="Minimum expected return %"
    )

    # State persistence
    state_dir: Path = Field(
        default=Path("execution/state"), description="Directory for state files"
    )

    @field_validator("state_dir", mode="before")
    @classmethod
    def parse_state_dir(cls, v):
        """Convert string path to Path object."""
        if isinstance(v, str):
            return Path(v)
        return v


class AppSettings(BaseSettings):
    """Main application settings.

    Loads from .env file and environment variables with validation.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Nested settings groups
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    swarm: SwarmSettings = Field(default_factory=SwarmSettings)
    reward: RewardWeights = Field(default_factory=RewardWeights)
    market_data: MarketDataSettings = Field(default_factory=MarketDataSettings)
    dpo: DPOTrainingSettings = Field(default_factory=DPOTrainingSettings)
    dataset: DatasetGenerationSettings = Field(default_factory=DatasetGenerationSettings)
    fee_model: FeeModelSettings = Field(default_factory=FeeModelSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)

    # Paths
    model_save_dir: Path = Field(
        default=Path("models"), description="Directory for saving model weights"
    )
    output_dir: Path = Field(default=Path("outputs"), description="Directory for results and logs")
    cache_dir: Path = Field(default=Path(".cache"), description="LLM response cache directory")

    # Logging
    log_level: str = Field(
        default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format",
    )

    @field_validator("model_save_dir", "output_dir", "cache_dir", mode="before")
    @classmethod
    def parse_path(cls, v):
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v

    @model_validator(mode="after")
    def create_directories(self) -> "AppSettings":
        """Create required directories if they don't exist."""
        for path in [
            self.model_save_dir,
            self.output_dir,
            self.cache_dir,
            self.market_data.cache_dir,
            self.dpo.adapter_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        return self

    def model_post_init(self, __context) -> None:
        """Post-initialization hook for nested settings from environment."""
        # Map flat environment variables to nested structure
        env_mappings = {
            # Ollama settings
            "OLLAMA_BASE_URL": ("ollama", "base_url"),
            "OLLAMA_GENERATOR_MODEL": ("ollama", "generator_model"),
            "OLLAMA_CRITIC_MODEL": ("ollama", "critic_model"),
            "OLLAMA_KEEP_ALIVE": ("ollama", "keep_alive"),
            "OLLAMA_TIMEOUT": ("ollama", "timeout"),
            # Swarm settings
            "GENERATOR_PERSONAS": ("swarm", "generator_personas"),
            "CRITIQUE_ENABLED": ("swarm", "critique_enabled"),
            "DPO_BATCH_SIZE": ("swarm", "dpo_batch_size"),
            "TRAINING_ENABLED": ("swarm", "training_enabled"),
            "CONCURRENCY": ("swarm", "concurrency"),
            # Reward weights
            "REWARD_RETURN_WEIGHT": ("reward", "return_weight"),
            "REWARD_DIRECTIONAL_WEIGHT": ("reward", "directional_weight"),
            "REWARD_MAE_WEIGHT": ("reward", "mae_weight"),
            # Market data settings
            "EXCHANGE": ("market_data", "exchange"),
            "SYMBOLS": ("market_data", "symbols"),
            "TIMEFRAME": ("market_data", "timeframe"),
            "LOOKBACK_BARS": ("market_data", "lookback_bars"),
            "DATA_CACHE_DIR": ("market_data", "cache_dir"),
            "DATA_CACHE_SIZE_LIMIT": ("market_data", "cache_size_limit"),
            # DPO training settings (Process B only)
            "DPO_LORA_RANK": ("dpo", "lora_rank"),
            "DPO_LORA_ALPHA": ("dpo", "lora_alpha"),
            "DPO_LORA_DROPOUT": ("dpo", "lora_dropout"),
            "DPO_BETA": ("dpo", "dpo_beta"),
            "DPO_LEARNING_RATE": ("dpo", "learning_rate"),
            "DPO_BATCH_SIZE": ("dpo", "batch_size"),
            "DPO_GRADIENT_ACCUMULATION_STEPS": ("dpo", "gradient_accumulation_steps"),
            "DPO_MIN_TRAINING_PAIRS": ("dpo", "min_training_pairs"),
            "DPO_ADAPTER_DIR": ("dpo", "adapter_dir"),
            # Fee model settings
            "FEE_MAKER_PCT": ("fee_model", "maker_fee_pct"),
            "FEE_TAKER_PCT": ("fee_model", "taker_fee_pct"),
            "FEE_BNB_DISCOUNT": ("fee_model", "bnb_discount_enabled"),
            "FEE_ENTRY_ORDER_TYPE": ("fee_model", "entry_order_type"),
            "FEE_EXIT_ORDER_TYPE": ("fee_model", "exit_order_type"),
            "FEE_FUNDING_RATE_PER_8H": ("fee_model", "funding_rate_pct"),
            "FEE_INCLUDE_FUNDING": ("fee_model", "include_funding"),
            "FEE_SLIPPAGE_PCT": ("fee_model", "slippage_pct"),
            # Execution settings
            "EXECUTION_TESTNET": ("execution", "testnet"),
            "BINANCE_API_KEY": ("execution", "api_key"),
            "BINANCE_API_SECRET": ("execution", "api_secret"),
            "EXECUTION_MODE": ("execution", "mode"),
            "EXECUTION_MAX_DAILY_TRADES": ("execution", "max_daily_trades"),
            "EXECUTION_MAX_DAILY_LOSS_PCT": ("execution", "max_daily_loss_pct"),
            "EXECUTION_MAX_OPEN_POSITIONS": ("execution", "max_open_positions"),
            "EXECUTION_MAX_POSITION_PCT": ("execution", "max_position_pct"),
            "EXECUTION_ORDER_COOLDOWN": ("execution", "order_cooldown_seconds"),
            "EXECUTION_MIN_CONFIDENCE": ("execution", "min_confidence"),
            "EXECUTION_MIN_EXPECTED_RETURN_PCT": ("execution", "min_expected_return_pct"),
            "EXECUTION_STATE_DIR": ("execution", "state_dir"),
        }

        for env_var, (group, field) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Get the nested settings object
                settings_obj = getattr(self, group)

                # Get field info for type conversion
                field_info = settings_obj.model_fields[field]
                field_type = field_info.annotation

                # Convert string value to appropriate type
                if field_type == bool or (
                    hasattr(field_type, "__origin__") and field_type.__origin__ == bool
                ):
                    converted = value.lower() in ("true", "1", "yes")
                elif field_type == int or (
                    hasattr(field_type, "__origin__") and field_type.__origin__ == int
                ):
                    converted = int(value)
                elif field_type == float or (
                    hasattr(field_type, "__origin__") and field_type.__origin__ == float
                ):
                    converted = float(value)
                elif field_type == Path or (
                    hasattr(field_type, "__origin__") and field_type.__origin__ == Path
                ):
                    converted = Path(value)
                elif field_type == List[str] or (
                    hasattr(field_type, "__origin__") and field_type.__origin__ == list
                ):
                    converted = [s.strip() for s in value.split(",")]
                else:
                    converted = value

                # Set the value on the nested object
                setattr(settings_obj, field, converted)

        # Re-validate nested settings after environment variable injection
        self.ollama = OllamaSettings.model_validate(self.ollama.model_dump())
        self.swarm = SwarmSettings.model_validate(self.swarm.model_dump())
        self.reward = RewardWeights.model_validate(self.reward.model_dump())
        self.market_data = MarketDataSettings.model_validate(self.market_data.model_dump())
        self.dpo = DPOTrainingSettings.model_validate(self.dpo.model_dump())
        self.fee_model = FeeModelSettings.model_validate(self.fee_model.model_dump())
        self.execution = ExecutionSettings.model_validate(self.execution.model_dump())


async def validate_ollama_models() -> dict:
    """
    Validate that required Ollama models are available.

    Returns:
        dict: Validation results with 'available', 'missing', and 'error' keys

    Example:
        >>> result = await validate_ollama_models()
        >>> if result['error']:
        ...     print(f"Ollama error: {result['error']}")
        >>> elif result['missing']:
        ...     print(f"Missing models: {result['missing']}")
    """
    import aiohttp

    result = {"available": [], "missing": [], "error": None}

    try:
        async with aiohttp.ClientSession() as session:
            url = f"{settings.ollama.base_url}/api/tags"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    result["error"] = f"Ollama API returned status {response.status}"
                    return result

                data = await response.json()
                available_models = [model["name"] for model in data.get("models", [])]

                # Check generator model
                if settings.ollama.generator_model in available_models:
                    result["available"].append(settings.ollama.generator_model)
                else:
                    result["missing"].append(settings.ollama.generator_model)

                # Check critic model
                if settings.ollama.critic_model in available_models:
                    result["available"].append(settings.ollama.critic_model)
                else:
                    result["missing"].append(settings.ollama.critic_model)

    except aiohttp.ClientError as e:
        result["error"] = f"Failed to connect to Ollama: {str(e)}"
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"

    return result


# Global settings instance
settings = AppSettings()
