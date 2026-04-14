"""Market snapshot builder for LLM prompts."""

from typing import Any

import pandas as pd

from config.fee_model import FeeModelSettings
from data.indicators import compute_all_indicators
from data.regime_filter import RegimeClassifier


def build_market_snapshot(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fee_model: FeeModelSettings,
    higher_tf_data: dict[str, pd.DataFrame] | None = None,
) -> str:
    """
    Build market snapshot prompt from OHLCV data.

    This is the same format the model sees in production. Contains only
    data available at bar close time - no future data.

    Args:
        df: OHLCV DataFrame with sufficient history
        symbol: Trading pair
        timeframe: Candle timeframe
        fee_model: Fee model for cost context
        higher_tf_data: Optional dict of higher timeframe DataFrames

    Returns:
        Formatted market snapshot string
    """
    # Compute all indicators
    indicators = compute_all_indicators(df, include_volume=True, include_structure=True)

    # Extract key values
    current_price = float(df["close"].iloc[-1])
    timestamp_ms = int(df["timestamp"].iloc[-1])

    # Price/Trend indicators
    rsi = indicators.get("rsi")
    macd_line = indicators.get("macd_line")
    macd_signal = indicators.get("macd_signal")
    donchian_upper = indicators.get("donchian_upper")
    donchian_middle = indicators.get("donchian_middle")
    donchian_lower = indicators.get("donchian_lower")
    kama = indicators.get("kama")

    # Volume indicators
    obv = indicators.get("obv")
    cmf = indicators.get("cmf")
    mfi = indicators.get("mfi")
    vwap = indicators.get("vwap")

    # Volatility indicators
    atr_normalized = indicators.get("atr_normalized")
    bb_width = indicators.get("bb_width")
    keltner_width = indicators.get("keltner_width")
    donchian_width = indicators.get("donchian_width")

    # Market structure
    open_fvg_count = indicators.get("open_fvg_count", 0)
    nearest_bullish_fvg_pct = indicators.get("nearest_bullish_fvg_pct")
    nearest_bearish_fvg_pct = indicators.get("nearest_bearish_fvg_pct")
    nearest_swing_high_pct = indicators.get("nearest_swing_high_pct")
    nearest_swing_low_pct = indicators.get("nearest_swing_low_pct")

    # Format recent price action (last 10 bars)
    recent = df.tail(10)
    price_lines = []
    for _, row in recent.iterrows():
        ts = pd.to_datetime(row["timestamp"], unit="ms")
        change = ((row["close"] - row["open"]) / row["open"]) * 100
        direction = "↑" if change > 0 else "↓" if change < 0 else "→"
        price_lines.append(
            f"{ts.strftime('%Y-%m-%d %H:%M')} | "
            f"O: ${row['open']:.2f} H: ${row['high']:.2f} "
            f"L: ${row['low']:.2f} C: ${row['close']:.2f} | "
            f"{direction} {change:+.2f}%"
        )
    price_summary = "\n".join(price_lines)

    # Classify market regime
    classifier = RegimeClassifier()
    regime, _ = classifier.get_current_regime(df["close"])

    # Calculate fee context
    holding_periods = 1.0  # Estimate for prompt
    round_trip_cost = fee_model.round_trip_cost_pct(holding_periods)
    min_profitable = fee_model.minimum_profitable_return_pct(holding_periods)

    # Format helper for optional values
    def fmt(val: Any, decimals: int = 2) -> str:
        if val is None:
            return "N/A"
        return f"{val:.{decimals}f}"

    # Build the market snapshot
    snapshot = f"""## Market Data
Symbol: {symbol}
Timeframe: {timeframe}
Timestamp: {pd.to_datetime(timestamp_ms, unit="ms").strftime("%Y-%m-%d %H:%M:%S UTC")}
Current Price: ${current_price:.4f}
Market Regime: {regime.name}

## Technical Indicators

### Price/Trend
RSI(14): {fmt(rsi)}
MACD Line: {fmt(macd_line, 4)} | Signal: {fmt(macd_signal, 4)}
Donchian(20): Upper ${fmt(donchian_upper, 2)} | Mid ${fmt(donchian_middle, 2)} | Lower ${fmt(donchian_lower, 2)}
KAMA(10): ${fmt(kama, 2)}

### Volume
OBV: {fmt(obv, 0)}
CMF(20): {fmt(cmf, 4)}
MFI(14): {fmt(mfi)}
VWAP: ${fmt(vwap, 2)}

### Volatility
ATR Normalized: {fmt(atr_normalized)}%
BB Width(20): {fmt(bb_width)}%
Keltner Width: {fmt(keltner_width)}%
Donchian Width: {fmt(donchian_width)}%

### Market Structure
Open FVG Count: {open_fvg_count}
Nearest Bullish FVG: {fmt(nearest_bullish_fvg_pct)}% above
Nearest Bearish FVG: {fmt(nearest_bearish_fvg_pct)}% below
Nearest Swing High: {fmt(nearest_swing_high_pct)}% above
Nearest Swing Low: {fmt(nearest_swing_low_pct)}% below

## Recent Price Action (last 10 bars)
{price_summary}

## Execution Context
Exchange: Binance Futures USDT-M
Estimated round-trip cost: {round_trip_cost:.3f}%
Minimum profitable move: {min_profitable:.3f}%"""

    # Add multi-timeframe context if available
    if higher_tf_data:
        htf_sections = []
        for htf, htf_df in higher_tf_data.items():
            if htf_df is not None and len(htf_df) >= 20:
                htf_indicators = compute_all_indicators(
                    htf_df, include_volume=True, include_structure=False
                )
                htf_rsi = htf_indicators.get("rsi")
                htf_macd = htf_indicators.get("macd_line")
                htf_signal = htf_indicators.get("macd_signal")
                htf_sections.append(
                    f"### {htf} Timeframe\n"
                    f"RSI: {fmt(htf_rsi)} | MACD: {fmt(htf_macd, 4)} (Signal: {fmt(htf_signal, 4)})"
                )
        if htf_sections:
            snapshot += "\n\n## Higher Timeframe Context\n" + "\n".join(htf_sections)

    return snapshot
