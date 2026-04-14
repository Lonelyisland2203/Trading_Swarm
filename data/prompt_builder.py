"""
Prompt builder for LLM generator with task sampling and templating.

Samples tasks with weighted probabilities and builds prompts from market data.
"""

import random
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

import pandas as pd
from loguru import logger

from .indicators import compute_all_indicators, compute_bb_position
from .regime_filter import MarketRegime

if TYPE_CHECKING:
    from config.fee_model import FeeModelSettings

# Thread-local random generator to avoid global state pollution
_task_rng = random.Random()

# Timeframe hierarchy for multi-timeframe analysis
TIMEFRAME_HIERARCHY = ["1m", "5m", "15m", "1h", "4h", "1d"]

# Small number threshold for numerical comparisons
EPSILON = 1e-8


def get_higher_timeframes(current_tf: str, available_tfs: list[str]) -> list[str]:
    """
    Get up to 2 nearest higher timeframes from available set.

    Args:
        current_tf: Current timeframe (e.g., "1h")
        available_tfs: List of available higher timeframes

    Returns:
        List of up to 2 nearest higher timeframes in hierarchy order

    Examples:
        >>> get_higher_timeframes("1m", ["5m", "15m", "1h"])
        ["5m", "15m"]
        >>> get_higher_timeframes("1h", ["4h"])
        ["4h"]
        >>> get_higher_timeframes("1d", ["1h", "4h"])
        []
    """
    if current_tf not in TIMEFRAME_HIERARCHY:
        logger.warning("Unknown current timeframe", timeframe=current_tf)
        return []

    current_idx = TIMEFRAME_HIERARCHY.index(current_tf)

    # Filter to only recognized timeframes higher in hierarchy
    higher_tfs = []
    for tf in TIMEFRAME_HIERARCHY[current_idx + 1:]:
        if tf in available_tfs:
            higher_tfs.append(tf)

    # Return up to 2 nearest
    return higher_tfs[:2]


def summarize_timeframe(df: pd.DataFrame, timeframe: str) -> dict:
    """
    Generate trend summary for a timeframe from OHLCV data.

    Extracts indicators and classifies trend direction based on:
    - Ichimoku cloud position (above/below/inside)
    - KAMA slope (rising/falling/flat)
    - Donchian channel position (upper/lower/middle)
    - RSI zone (overbought/oversold/neutral)

    Args:
        df: OHLCV DataFrame with sufficient history (50+ bars recommended)
        timeframe: Timeframe label (e.g., "4h", "1d")

    Returns:
        Dict with structure:
        {
            "timeframe": str,
            "trend": "bullish" | "bearish" | "neutral",
            "cloud_position": "above" | "below" | "inside",
            "kama_slope": "rising" | "falling" | "flat",
            "donchian_position": "upper" | "lower" | "middle",
            "rsi_zone": "overbought" | "oversold" | "neutral",
            "rsi_value": float,
            "text": str (human-readable summary)
        }

    Examples:
        >>> df = create_test_df_bullish()
        >>> summarize_timeframe(df, "4h")
        {"timeframe": "4h", "trend": "bullish", ...}
    """
    # Compute all indicators
    indicators = compute_all_indicators(df, include_volume=False, include_structure=False)

    current_price = float(df["close"].iloc[-1])

    # 1. Ichimoku Cloud Position (using Ichimoku components)
    # Compute Ichimoku components if not already in indicators
    from .indicators import compute_ichimoku_cloud
    ichimoku = compute_ichimoku_cloud(df["high"], df["low"], df["close"])

    senkou_span_a = ichimoku.get("senkou_span_a")
    senkou_span_b = ichimoku.get("senkou_span_b")

    if senkou_span_a is None or senkou_span_b is None:
        cloud_position = "inside"
    else:
        cloud_top = float(max(senkou_span_a.iloc[-1], senkou_span_b.iloc[-1]))
        cloud_bottom = float(min(senkou_span_a.iloc[-1], senkou_span_b.iloc[-1]))

        if pd.isna(cloud_top) or pd.isna(cloud_bottom):
            cloud_position = "inside"
        elif current_price > cloud_top:
            cloud_position = "above"
        elif current_price < cloud_bottom:
            cloud_position = "below"
        else:
            cloud_position = "inside"

    # 2. KAMA Slope
    kama_series = indicators["series"].get("kama")
    if kama_series is None or len(kama_series) < 6:
        kama_slope = "flat"
    else:
        kama_current = kama_series.iloc[-1]
        kama_prev = kama_series.iloc[-6]  # 5-bar lookback

        if pd.isna(kama_current) or pd.isna(kama_prev):
            kama_slope = "flat"
        else:
            slope_pct = ((kama_current - kama_prev) / kama_prev) * 100
            threshold = 0.1  # 0.1% threshold

            if slope_pct > threshold:
                kama_slope = "rising"
            elif slope_pct < -threshold:
                kama_slope = "falling"
            else:
                kama_slope = "flat"

    # 3. Donchian Channel Position
    donchian_upper = indicators.get("donchian_upper")
    donchian_lower = indicators.get("donchian_lower")

    if (donchian_upper is None or donchian_lower is None or
        pd.isna(donchian_upper) or pd.isna(donchian_lower)):
        donchian_position = "middle"
    else:
        channel_range = donchian_upper - donchian_lower
        if channel_range < EPSILON:
            donchian_position = "middle"
        else:
            position_pct = (current_price - donchian_lower) / channel_range

            if position_pct > 0.65:
                donchian_position = "upper"
            elif position_pct < 0.35:
                donchian_position = "lower"
            else:
                donchian_position = "middle"

    # 4. RSI Zone
    rsi = indicators.get("rsi")
    if rsi is None or pd.isna(rsi):
        rsi = 50.0
        rsi_zone = "neutral"
    else:
        rsi = float(rsi)
        if rsi > 70:
            rsi_zone = "overbought"
        elif rsi < 30:
            rsi_zone = "oversold"
        else:
            rsi_zone = "neutral"

    # 5. Overall Trend Classification (4-indicator majority vote)
    votes = {"bullish": 0, "bearish": 0, "neutral": 0}

    # Ichimoku cloud vote
    if cloud_position == "above":
        votes["bullish"] += 1
    elif cloud_position == "below":
        votes["bearish"] += 1
    else:
        votes["neutral"] += 1

    # KAMA slope vote
    if kama_slope == "rising":
        votes["bullish"] += 1
    elif kama_slope == "falling":
        votes["bearish"] += 1
    else:
        votes["neutral"] += 1

    # Donchian position vote
    if donchian_position == "upper":
        votes["bullish"] += 1
    elif donchian_position == "lower":
        votes["bearish"] += 1
    else:
        votes["neutral"] += 1

    # RSI zone vote
    if rsi_zone == "overbought":
        votes["bullish"] += 1
    elif rsi_zone == "oversold":
        votes["bearish"] += 1
    else:
        votes["neutral"] += 1

    # Determine trend by majority vote
    trend = max(votes, key=votes.get)

    # Calculate confidence: percentage agreement of the 4 indicators
    # (highest_vote_count / 4) * 100 as a fraction (0.0-1.0)
    highest_vote_count = max(votes.values())
    confidence = highest_vote_count / 4.0

    # 6. Generate Text Summary
    trend_label = trend.capitalize()
    cloud_text = f"{cloud_position} cloud"
    kama_text = f"KAMA {kama_slope}"
    donchian_text = f"near Donchian {donchian_position}" if donchian_position != "middle" else "middle of Donchian"
    rsi_text = f"RSI {rsi:.0f} ({rsi_zone})"

    text = f"{timeframe}: {trend_label} ({cloud_text}, {kama_text}), {donchian_text}, {rsi_text}"

    logger.debug(
        "Timeframe summary",
        timeframe=timeframe,
        trend=trend,
        cloud_position=cloud_position,
        kama_slope=kama_slope,
    )

    return {
        "timeframe": timeframe,
        "trend": trend,
        "cloud_position": cloud_position,
        "kama_slope": kama_slope,
        "donchian_position": donchian_position,
        "rsi_zone": rsi_zone,
        "rsi_value": rsi,
        "confidence": confidence,
        "text": text,
    }


def compute_confluence(summaries: list[dict]) -> dict:
    """
    Analyze trend alignment across multiple timeframes.

    Args:
        summaries: List of timeframe summaries from summarize_timeframe()

    Returns:
        Dict with structure:
        {
            "status": "aligned" | "mixed" | "conflicting",
            "description": str (human-readable confluence description),
            "aligned_count": int (number of timeframes contributing to alignment)
        }

    Examples:
        >>> summaries = [{"trend": "bullish"}, {"trend": "bullish"}]
        >>> compute_confluence(summaries)
        {"status": "aligned", "description": "Confluence: Aligned with higher timeframes (bullish)", "aligned_count": 2}
    """
    if not summaries:
        return {
            "status": "aligned",
            "description": "Confluence: No higher timeframe data",
            "aligned_count": 0,
        }

    trends = [s["trend"] for s in summaries]

    # Count trend types
    bullish_count = trends.count("bullish")
    bearish_count = trends.count("bearish")
    neutral_count = trends.count("neutral")

    total = len(trends)

    # All same trend -> aligned
    if bullish_count == total:
        return {
            "status": "aligned",
            "description": "Confluence: Aligned with higher timeframes (bullish)",
            "aligned_count": total,
        }
    elif bearish_count == total:
        return {
            "status": "aligned",
            "description": "Confluence: Aligned with higher timeframes (bearish)",
            "aligned_count": total,
        }
    elif neutral_count == total:
        return {
            "status": "aligned",
            "description": "Confluence: Higher timeframes neutral",
            "aligned_count": total,
        }

    # Opposite trends (bullish vs bearish) -> conflicting
    if bullish_count > 0 and bearish_count > 0:
        return {
            "status": "conflicting",
            "description": "Confluence: Conflicting signals across timeframes (bullish vs bearish)",
            "aligned_count": 0,
        }

    # Mixed signals (includes neutral with other trends)
    return {
        "status": "mixed",
        "description": "Confluence: Mixed signals across timeframes",
        "aligned_count": max(bullish_count, bearish_count, neutral_count),
    }


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
TASK_CONFIGS = [
    TaskConfig(TaskType.PREDICT_DIRECTION, weight=1.0, difficulty=2, min_bars_required=50),
    TaskConfig(TaskType.ASSESS_MOMENTUM, weight=0.8, difficulty=2, min_bars_required=30),
    TaskConfig(TaskType.IDENTIFY_SUPPORT_RESISTANCE, weight=0.6, difficulty=3, min_bars_required=100),
    # Temporarily disabled until templates are implemented:
    # TaskConfig(TaskType.DETECT_TREND_REVERSAL, weight=0.20, difficulty=4, min_bars_required=100),
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
{higher_tf_section}{execution_context}
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
        higher_tf_context: str | None = None,
        execution_context: str = "",
    ) -> str:
        """Render direction prediction prompt."""
        # Build higher timeframe section
        higher_tf_section = ""
        if higher_tf_context:
            higher_tf_section = f"\n## Higher Timeframe Context\n{higher_tf_context}\n"

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
            higher_tf_section=higher_tf_section,
            execution_context=execution_context,
        )


@dataclass(slots=True)
class MomentumAssessmentPrompt:
    """Prompt template for momentum assessment task."""

    TEMPLATE = """/no_think

You are a momentum analyst. Analyze the following market data and assess whether momentum is increasing or decreasing.

## Market Data
Symbol: {symbol}
Timeframe: {timeframe}
Current Price: ${current_price:.4f}
Market Regime: {market_regime}
{higher_tf_section}{execution_context}
## Technical Indicators
RSI(14): {rsi:.2f} (previous: {rsi_prev:.2f}, change: {rsi_delta:+.2f})
MACD: {macd:.4f} (Signal: {macd_signal:.4f})
MACD Previous: {macd_prev:.4f} (change: {macd_delta:+.4f})
BB Width: {bb_width:.4f} ({bb_trend})

## Recent Price Action (last {num_bars} bars)
{price_summary}

## Task
Is momentum INCREASING or DECREASING? Consider:
- RSI trend (rising RSI = increasing momentum)
- MACD vs signal line (diverging upward = increasing momentum)
- Price action acceleration (larger price changes = increasing momentum)
- BB width (expanding = increasing volatility/momentum)

Respond in JSON format:
{{"direction": "INCREASING" | "DECREASING", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
"""

    def render(
        self,
        symbol: str,
        timeframe: str,
        current_price: float,
        market_regime: str,
        rsi: float,
        rsi_prev: float,
        rsi_delta: float,
        macd: float,
        macd_signal: float,
        macd_prev: float,
        macd_delta: float,
        bb_width: float,
        bb_trend: str,
        price_summary: str,
        num_bars: int = 10,
        higher_tf_context: str | None = None,
        execution_context: str = "",
    ) -> str:
        """Render momentum assessment prompt."""
        # Build higher timeframe section
        higher_tf_section = ""
        if higher_tf_context:
            higher_tf_section = f"\n## Higher Timeframe Context\n{higher_tf_context}\n"

        return self.TEMPLATE.format(
            symbol=symbol,
            timeframe=timeframe,
            current_price=current_price,
            market_regime=market_regime,
            rsi=rsi,
            rsi_prev=rsi_prev,
            rsi_delta=rsi_delta,
            macd=macd,
            macd_signal=macd_signal,
            macd_prev=macd_prev,
            macd_delta=macd_delta,
            bb_width=bb_width,
            bb_trend=bb_trend,
            price_summary=price_summary,
            num_bars=num_bars,
            higher_tf_section=higher_tf_section,
            execution_context=execution_context,
        )


@dataclass(slots=True)
class SupportResistancePrompt:
    """Prompt template for support/resistance identification task."""

    TEMPLATE = """/no_think

You are a technical analyst. Identify the nearest support and resistance levels based on recent price action.

## Market Data
Symbol: {symbol}
Timeframe: {timeframe}
Current Price: ${current_price:.4f}
Market Regime: {market_regime}
{higher_tf_section}{execution_context}
## Price History (last 50 bars)
High: ${price_high:.4f}
Low: ${price_low:.4f}
Range: ${price_range:.4f}

## Recent Swing Points
Swing Highs: {swing_highs}
Swing Lows: {swing_lows}

## Recent Price Action (last {num_bars} bars)
{price_summary}

## Task
Identify:
1. Nearest SUPPORT level below current price
2. Nearest RESISTANCE level above current price

Consider: Previous swing highs/lows, psychological levels (round numbers), high-volume areas

Respond in JSON format:
{{
  "support_price": float,
  "support_confidence": 0.0-1.0,
  "resistance_price": float,
  "resistance_confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}
"""

    def render(
        self,
        symbol: str,
        timeframe: str,
        current_price: float,
        market_regime: str,
        price_high: float,
        price_low: float,
        price_range: float,
        swing_highs: str,
        swing_lows: str,
        price_summary: str,
        num_bars: int = 10,
        higher_tf_context: str | None = None,
        execution_context: str = "",
    ) -> str:
        """Render support/resistance prompt."""
        # Build higher timeframe section
        higher_tf_section = ""
        if higher_tf_context:
            higher_tf_section = f"\n## Higher Timeframe Context\n{higher_tf_context}\n"

        return self.TEMPLATE.format(
            symbol=symbol,
            timeframe=timeframe,
            current_price=current_price,
            market_regime=market_regime,
            price_high=price_high,
            price_low=price_low,
            price_range=price_range,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            price_summary=price_summary,
            num_bars=num_bars,
            higher_tf_section=higher_tf_section,
            execution_context=execution_context,
        )


def detect_swing_highs(df: pd.DataFrame, window: int = 5, num_swings: int = 3) -> list[float]:
    """
    Detect recent swing high prices.

    A swing high is a local maximum where the high is greater than
    surrounding highs in the window.

    Args:
        df: OHLCV DataFrame
        window: Window size for local maximum detection
        num_swings: Number of most recent swings to return

    Returns:
        List of swing high prices (most recent first)
    """
    if len(df) < window * 2 + 1:
        return []

    highs = df["high"].values
    swing_highs = []

    # Start from most recent and work backward
    for i in range(len(highs) - window - 1, window, -1):
        # Check if this is a local maximum
        is_swing = True
        center_high = highs[i]

        # Check left window
        for j in range(i - window, i):
            if highs[j] >= center_high:
                is_swing = False
                break

        # Check right window
        if is_swing:
            for j in range(i + 1, min(i + window + 1, len(highs))):
                if highs[j] >= center_high:
                    is_swing = False
                    break

        if is_swing:
            swing_highs.append(float(center_high))
            if len(swing_highs) >= num_swings:
                break

    return swing_highs


def detect_swing_lows(df: pd.DataFrame, window: int = 5, num_swings: int = 3) -> list[float]:
    """
    Detect recent swing low prices.

    A swing low is a local minimum where the low is less than
    surrounding lows in the window.

    Args:
        df: OHLCV DataFrame
        window: Window size for local minimum detection
        num_swings: Number of most recent swings to return

    Returns:
        List of swing low prices (most recent first)
    """
    if len(df) < window * 2 + 1:
        return []

    lows = df["low"].values
    swing_lows = []

    # Start from most recent and work backward
    for i in range(len(lows) - window - 1, window, -1):
        # Check if this is a local minimum
        is_swing = True
        center_low = lows[i]

        # Check left window
        for j in range(i - window, i):
            if lows[j] <= center_low:
                is_swing = False
                break

        # Check right window
        if is_swing:
            for j in range(i + 1, min(i + window + 1, len(lows))):
                if lows[j] <= center_low:
                    is_swing = False
                    break

        if is_swing:
            swing_lows.append(float(center_low))
            if len(swing_lows) >= num_swings:
                break

    return swing_lows


def calculate_bb_width(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> float:
    """
    Calculate Bollinger Band width.

    Width = (Upper Band - Lower Band) / Middle Band

    Args:
        df: OHLCV DataFrame
        period: Period for moving average
        std_dev: Standard deviation multiplier

    Returns:
        BB width as ratio
    """
    if len(df) < period:
        return 0.0

    close = df["close"].tail(period)
    middle_band = close.mean()
    std = close.std()

    upper_band = middle_band + (std_dev * std)
    lower_band = middle_band - (std_dev * std)

    if middle_band == 0:
        return 0.0

    width = (upper_band - lower_band) / middle_band
    return float(width)


def get_bb_trend(current_width: float, prev_width: float) -> str:
    """
    Determine Bollinger Band trend.

    Args:
        current_width: Current BB width
        prev_width: Previous BB width

    Returns:
        "expanding", "contracting", or "stable"
    """
    if abs(current_width - prev_width) < 0.001:
        return "stable"
    elif current_width > prev_width:
        return "expanding"
    else:
        return "contracting"


class PromptBuilder:
    """
    Main interface for building prompts from market data.

    Combines task sampling with data-driven template rendering.
    """

    def __init__(self):
        """Initialize prompt builder."""
        self.templates = {
            TaskType.PREDICT_DIRECTION: DirectionPredictionPrompt(),
            TaskType.ASSESS_MOMENTUM: MomentumAssessmentPrompt(),
            TaskType.IDENTIFY_SUPPORT_RESISTANCE: SupportResistancePrompt(),
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

    def _build_execution_context(
        self,
        timeframe: str,
        horizon_bars: int,
        fee_model: "FeeModelSettings",
    ) -> str:
        """
        Build execution context section for prompts.

        Args:
            timeframe: Trading timeframe (e.g., "1h", "1d")
            horizon_bars: Prediction horizon in bars
            fee_model: Fee model settings

        Returns:
            Formatted execution context section
        """
        from verifier.constants import compute_holding_periods_8h

        holding_periods = compute_holding_periods_8h(timeframe, horizon_bars)
        round_trip_cost = fee_model.round_trip_cost_pct(holding_periods)
        min_profitable = fee_model.minimum_profitable_return_pct(holding_periods)

        # Determine mode from fee model settings
        if fee_model.include_funding:
            mode = "Futures USDT-M"
        else:
            mode = "Spot"

        return f"""## Execution Context
Exchange: Binance
Mode: {mode}
Estimated round-trip cost: {round_trip_cost:.3f}%
Minimum profitable move: {min_profitable:.3f}%

Your prediction must account for these costs. Signals with expected moves smaller than the minimum profitable threshold should be rated LOW CONFIDENCE regardless of directional conviction."""

    def build_prompt(
        self,
        task: TaskConfig,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        market_regime: MarketRegime,
        higher_tf_data: dict[str, pd.DataFrame] | None = None,
        fee_model: "FeeModelSettings | None" = None,
    ) -> str:
        """
        Build prompt for given task and market data.

        Args:
            task: Task configuration
            df: OHLCV DataFrame with sufficient history
            symbol: Trading pair
            timeframe: Candle timeframe
            market_regime: Current market regime
            higher_tf_data: Optional dict of higher timeframe OHLCV DataFrames
                           Keys are timeframe strings (e.g., "4h", "1d")
                           Values are OHLCV DataFrames
            fee_model: Optional FeeModelSettings for execution cost context.
                      When None, no execution context is added to prompts.

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

        # Validate and filter higher_tf_data
        validated_higher_tf_data = None
        if higher_tf_data is not None:
            if not isinstance(higher_tf_data, dict):
                logger.warning("Invalid higher_tf_data type", type=type(higher_tf_data))
            else:
                valid_data = {}
                for tf, tf_df in higher_tf_data.items():
                    if not isinstance(tf_df, pd.DataFrame):
                        logger.warning("Invalid DataFrame", timeframe=tf, type=type(tf_df))
                        continue
                    if len(tf_df) == 0:
                        logger.warning("Empty DataFrame", timeframe=tf)
                        continue
                    if len(tf_df) < 52:  # Minimum for Ichimoku (26*2)
                        logger.warning("Insufficient data", timeframe=tf, bars=len(tf_df))
                        continue
                    valid_data[tf] = tf_df

                validated_higher_tf_data = valid_data if valid_data else None

        # Build higher timeframe context section
        higher_tf_context = None
        if validated_higher_tf_data:
            # Select up to 2 nearest higher timeframes
            selected_tfs = get_higher_timeframes(timeframe, list(validated_higher_tf_data.keys()))

            if selected_tfs:
                summaries = []
                for tf in selected_tfs:
                    try:
                        summary = summarize_timeframe(validated_higher_tf_data[tf], tf)
                        summaries.append(summary)
                    except Exception as e:
                        logger.warning("Failed to summarize timeframe", timeframe=tf, error=str(e))
                        continue

                if summaries:
                    # Build context text
                    summary_lines = [s["text"] for s in summaries]
                    confluence_result = compute_confluence(summaries)
                    summary_lines.append(confluence_result["description"])
                    higher_tf_context = "\n".join(summary_lines)

                    logger.debug(
                        "Higher TF context built",
                        timeframes=selected_tfs,
                        num_summaries=len(summaries),
                    )

        # Calculate all indicators once
        indicators = compute_all_indicators(df, include_volume=True, include_structure=True)

        # Access scalars directly
        rsi = indicators['rsi'] if indicators['rsi'] is not None else 50.0
        macd = indicators['macd_line'] if indicators['macd_line'] is not None else 0.0
        macd_signal = indicators['macd_signal'] if indicators['macd_signal'] is not None else 0.0

        # BB position not yet in compute_all_indicators, compute separately
        bb_pos = compute_bb_position(df["close"]).iloc[-1]
        bb_pos = bb_pos if not pd.isna(bb_pos) else 0.5

        # Format price summary
        price_summary = self._format_price_summary(df)
        current_price = float(df["close"].iloc[-1])

        # Build execution context section
        execution_context = ""
        if fee_model is not None:
            from verifier.constants import get_horizon_bars
            try:
                horizon_bars = get_horizon_bars(timeframe)
            except (ValueError, KeyError):
                logger.warning("Unknown timeframe for horizon", timeframe=timeframe)
                horizon_bars = 5  # Default fallback

            execution_context = "\n" + self._build_execution_context(
                timeframe, horizon_bars, fee_model
            ) + "\n"

        logger.debug(
            "Execution context built",
            timeframe=timeframe,
            has_execution_context=bool(execution_context),
        )

        # Task-specific rendering
        if task.task_type == TaskType.PREDICT_DIRECTION:
            prompt = template.render(
                symbol=symbol,
                timeframe=timeframe,
                current_price=current_price,
                market_regime=market_regime.value,
                rsi=rsi,
                macd=macd,
                macd_signal=macd_signal,
                bb_position=bb_pos,
                price_summary=price_summary,
                higher_tf_context=higher_tf_context,
                execution_context=execution_context,
            )

        elif task.task_type == TaskType.ASSESS_MOMENTUM:
            # Calculate momentum-specific indicators using series from indicators
            rsi_series = indicators['series']['rsi']
            macd_line = indicators['series']['macd_line']

            rsi_prev = rsi_series.iloc[-2] if len(rsi_series) > 1 and not pd.isna(rsi_series.iloc[-2]) else rsi
            rsi_delta = rsi - rsi_prev

            macd_prev = macd_line.iloc[-2] if len(macd_line) > 1 and not pd.isna(macd_line.iloc[-2]) else macd
            macd_delta = macd - macd_prev

            # Calculate BB width
            current_bb_width = calculate_bb_width(df, period=20)
            prev_bb_width = calculate_bb_width(df.iloc[:-1], period=20) if len(df) > 20 else current_bb_width
            bb_trend = get_bb_trend(current_bb_width, prev_bb_width)

            prompt = template.render(
                symbol=symbol,
                timeframe=timeframe,
                current_price=current_price,
                market_regime=market_regime.value,
                rsi=rsi,
                rsi_prev=rsi_prev,
                rsi_delta=rsi_delta,
                macd=macd,
                macd_signal=macd_signal,
                macd_prev=macd_prev,
                macd_delta=macd_delta,
                bb_width=current_bb_width,
                bb_trend=bb_trend,
                price_summary=price_summary,
                higher_tf_context=higher_tf_context,
                execution_context=execution_context,
            )

        elif task.task_type == TaskType.IDENTIFY_SUPPORT_RESISTANCE:
            # Calculate support/resistance indicators
            price_high = float(df["high"].tail(50).max())
            price_low = float(df["low"].tail(50).min())
            price_range = price_high - price_low

            # Detect swing points
            swing_highs = detect_swing_highs(df, window=5, num_swings=3)
            swing_lows = detect_swing_lows(df, window=5, num_swings=3)

            # Format swing points
            swing_highs_str = ", ".join([f"${h:.2f}" for h in swing_highs]) if swing_highs else "None detected"
            swing_lows_str = ", ".join([f"${l:.2f}" for l in swing_lows]) if swing_lows else "None detected"

            prompt = template.render(
                symbol=symbol,
                timeframe=timeframe,
                current_price=current_price,
                market_regime=market_regime.value,
                price_high=price_high,
                price_low=price_low,
                price_range=price_range,
                swing_highs=swing_highs_str,
                swing_lows=swing_lows_str,
                price_summary=price_summary,
                higher_tf_context=higher_tf_context,
                execution_context=execution_context,
            )

        else:
            raise ValueError(f"Unsupported task type: {task.task_type}")

        logger.info(
            "Prompt built",
            task_type=task.task_type.value,
            symbol=symbol,
            timeframe=timeframe,
            prompt_length=len(prompt),
            has_higher_tf=higher_tf_context is not None,
        )

        return prompt
