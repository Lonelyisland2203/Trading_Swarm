"""
Prompt builder for LLM generator with task sampling and templating.

Samples tasks with weighted probabilities and builds prompts from market data.
"""

import random
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import pandas as pd
from loguru import logger

from .indicators import compute_rsi, compute_macd, compute_bb_position
from .regime_filter import MarketRegime

# Thread-local random generator to avoid global state pollution
_task_rng = random.Random()


class TaskType(Enum):
    """Training task types for signal generation."""

    PREDICT_DIRECTION = "predict_direction"
    IDENTIFY_SUPPORT_RESISTANCE = "identify_support_resistance"
    DETECT_TREND_REVERSAL = "detect_trend_reversal"
    ASSESS_MOMENTUM = "assess_momentum"
    IDENTIFY_PATTERN = "identify_pattern"


@dataclass(slots=True, frozen=True)
class TaskConfig:
    """Task metadata for weighted sampling."""

    task_type: TaskType
    weight: float           # Sampling probability weight
    difficulty: int         # 1-5 scale
    min_bars_required: int  # Minimum data needed


# Task configurations with weights
# NOTE: Only PREDICT_DIRECTION has a template implemented. Other tasks are
# defined for future expansion but will raise NotImplementedError if sampled.
TASK_CONFIGS = [
    TaskConfig(TaskType.PREDICT_DIRECTION, weight=1.0, difficulty=2, min_bars_required=50),
    # Temporarily disabled until templates are implemented:
    # TaskConfig(TaskType.IDENTIFY_SUPPORT_RESISTANCE, weight=0.25, difficulty=3, min_bars_required=100),
    # TaskConfig(TaskType.DETECT_TREND_REVERSAL, weight=0.20, difficulty=4, min_bars_required=100),
    # TaskConfig(TaskType.ASSESS_MOMENTUM, weight=0.15, difficulty=2, min_bars_required=30),
    # TaskConfig(TaskType.IDENTIFY_PATTERN, weight=0.10, difficulty=5, min_bars_required=200),
]


def sample_task(
    available_bars: int,
    difficulty_range: tuple[int, int] = (1, 5),
    seed: int | None = None,
) -> TaskConfig:
    """
    Sample a task weighted by probability and filtered by constraints.

    Args:
        available_bars: Number of bars available in dataset
        difficulty_range: Allowed difficulty range (min, max)
        seed: Random seed for reproducibility (uses isolated RNG, not global state)

    Returns:
        Sampled task configuration

    Raises:
        ValueError: If no eligible tasks found
    """
    # Use isolated random generator to avoid polluting global state
    rng = random.Random(seed) if seed is not None else _task_rng

    eligible = [
        t for t in TASK_CONFIGS
        if t.min_bars_required <= available_bars
        and difficulty_range[0] <= t.difficulty <= difficulty_range[1]
    ]

    if not eligible:
        raise ValueError(
            f"No eligible tasks for {available_bars} bars and difficulty {difficulty_range}"
        )

    weights = [t.weight for t in eligible]
    total = sum(weights)
    normalized_weights = [w / total for w in weights]

    task = rng.choices(eligible, weights=normalized_weights, k=1)[0]

    logger.debug(
        "Task sampled",
        task_type=task.task_type.value,
        difficulty=task.difficulty,
        weight=task.weight,
    )

    return task


class PromptTemplate(Protocol):
    """Protocol for prompt templates."""

    def render(self, **kwargs) -> str:
        """Render template with provided kwargs."""
        ...


@dataclass(slots=True)
class DirectionPredictionPrompt:
    """Prompt template for direction prediction task."""

    TEMPLATE = """/no_think

You are a quantitative trading analyst. Analyze the following market data and predict the price direction for the next {horizon} bars.

## Market Data
Symbol: {symbol}
Timeframe: {timeframe}
Current Price: ${current_price:.4f}
Market Regime: {market_regime}

## Technical Indicators
RSI(14): {rsi:.2f}
MACD: {macd:.4f} (Signal: {macd_signal:.4f})
BB Position: {bb_position:.2f} (0.0=lower band, 0.5=middle, 1.0=upper band)

## Recent Price Action (last {num_bars} bars)
{price_summary}

## Task
Predict whether the price will be HIGHER or LOWER {horizon} bars from now.

Respond in JSON format:
{{"direction": "HIGHER" | "LOWER", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
"""

    def render(
        self,
        symbol: str,
        timeframe: str,
        current_price: float,
        market_regime: str,
        rsi: float,
        macd: float,
        macd_signal: float,
        bb_position: float,
        price_summary: str,
        horizon: int = 5,
        num_bars: int = 10,
    ) -> str:
        """Render direction prediction prompt."""
        return self.TEMPLATE.format(
            symbol=symbol,
            timeframe=timeframe,
            current_price=current_price,
            market_regime=market_regime,
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            bb_position=bb_position,
            price_summary=price_summary,
            horizon=horizon,
            num_bars=num_bars,
        )


class PromptBuilder:
    """
    Main interface for building prompts from market data.

    Combines task sampling with data-driven template rendering.
    """

    def __init__(self):
        """Initialize prompt builder."""
        self.templates = {
            TaskType.PREDICT_DIRECTION: DirectionPredictionPrompt(),
            # Additional templates can be added for other task types
        }

    def _format_price_summary(self, df: pd.DataFrame, num_bars: int = 10) -> str:
        """
        Format recent price action as text summary.

        Args:
            df: OHLCV DataFrame
            num_bars: Number of recent bars to include

        Returns:
            Formatted price summary string
        """
        recent = df.tail(num_bars)

        lines = []
        for idx, row in recent.iterrows():
            timestamp = pd.to_datetime(row["timestamp"], unit="ms")
            change = ((row["close"] - row["open"]) / row["open"]) * 100
            direction = "[UP]" if change > 0 else "[DOWN]"

            lines.append(
                f"{timestamp.strftime('%Y-%m-%d %H:%M')} | "
                f"O: ${row['open']:.2f} H: ${row['high']:.2f} "
                f"L: ${row['low']:.2f} C: ${row['close']:.2f} | "
                f"{direction} {change:+.2f}%"
            )

        return "\n".join(lines)

    def build_prompt(
        self,
        task: TaskConfig,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        market_regime: MarketRegime,
    ) -> str:
        """
        Build prompt for given task and market data.

        Args:
            task: Task configuration
            df: OHLCV DataFrame with sufficient history
            symbol: Trading pair
            timeframe: Candle timeframe
            market_regime: Current market regime

        Returns:
            Rendered prompt string

        Raises:
            ValueError: If insufficient data or unsupported task type
        """
        if len(df) < task.min_bars_required:
            raise ValueError(
                f"Insufficient data: need {task.min_bars_required}, got {len(df)}"
            )

        template = self.templates.get(task.task_type)
        if template is None:
            raise ValueError(f"No template for task type: {task.task_type}")

        # Calculate indicators
        rsi = compute_rsi(df["close"]).iloc[-1]
        macd_line, macd_signal, _ = compute_macd(df["close"])
        bb_pos = compute_bb_position(df["close"]).iloc[-1]

        # Format price summary
        price_summary = self._format_price_summary(df)

        # Render template
        prompt = template.render(
            symbol=symbol,
            timeframe=timeframe,
            current_price=df["close"].iloc[-1],
            market_regime=market_regime.value,
            rsi=rsi if not pd.isna(rsi) else 50.0,
            macd=macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0.0,
            macd_signal=macd_signal.iloc[-1] if not pd.isna(macd_signal.iloc[-1]) else 0.0,
            bb_position=bb_pos if not pd.isna(bb_pos) else 0.5,
            price_summary=price_summary,
        )

        logger.info(
            "Prompt built",
            task_type=task.task_type.value,
            symbol=symbol,
            timeframe=timeframe,
            prompt_length=len(prompt),
        )

        return prompt
