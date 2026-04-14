# Signal Layer

## Node Architecture (LangGraph)
Data Node → Indicator Compute → XGBoost Signal (calibrated probability)
                                       ↓
                             LLM Context Node (Qwen: funding/sentiment/OI narrative)
                                       ↓
                             Synthesis Node (XGBoost prob + LLM context → decision)
                                       ↓
                             Risk Filter (DeepSeek: veto/downsize if context conflicts)
                                       ↓
                             Execution Node (Hyperliquid trigger orders + watchdog)

## XGBoost Signal Contract
- Input: 17 indicators as feature vector (same compute_ functions)
- Output: {direction: LONG|SHORT|FLAT, probability: float, features: dict}
- Retrain: walk-forward, triggered every 200 verified signals or by autoresearch

## LLM Context Contract
- Input: funding rate, OI delta, liquidation data, news headlines
- Output: {bullish_factors: list, bearish_factors: list, regime_flag: str, confidence: float}
- NEVER outputs LONG/SHORT. Context only.

## Synthesis Rules
- XGBoost prob < 0.55 → FLAT (no trade regardless of context)
- XGBoost prob ≥ 0.55 + LLM regime_flag == "conflicting" → half position
- XGBoost prob ≥ 0.65 + LLM confirms → full position
- DeepSeek veto overrides everything → FLAT