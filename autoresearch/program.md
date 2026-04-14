# Autoresearch: XGBoost Signal Optimization

Karpathy-pattern autoresearch for XGBoost hyperparameter tuning.
One file to edit, one eval script (read-only), one metric to optimize.

## Setup

1. Create branch: `git checkout -b autoresearch/$(date +%Y%m%d-%H%M)`
2. Read `signals/xgboost_config.py` for current config
3. Read `autoresearch/results.tsv` for experiment history
4. Read `evaluation/baseline_metrics.json` for baseline (if exists)

## Files

| File | Role | Modifiable? |
|------|------|-------------|
| `signals/xgboost_config.py` | Config to tune | YES |
| `evaluation/xgboost_eval.py` | Eval script | NEVER |
| `autoresearch/results.tsv` | Results log | Auto-append only |
| `config/fee_model.py` | Fee model | NEVER |
| `execution/*` | Execution layer | NEVER |

## Experiment Loop

### 1. Propose ONE change to `signals/xgboost_config.py`

Changes to try (in priority order):

**a) Feature engineering** (highest signal)
- Interaction terms between correlated indicators
- Rolling ratios (e.g., RSI / MFI)
- Lag features for momentum

**b) Feature selection**
- Remove features with low SHAP importance
- Add features that SHAP shows as important but missing
- Check `shap_top_5` in eval output

**c) Hyperparameters**
- `max_depth`: [3, 4, 5, 6, 7, 8]
- `n_estimators`: [50, 100, 150, 200]
- `learning_rate`: [0.05, 0.1, 0.15, 0.2]
- `subsample`: [0.6, 0.7, 0.8, 0.9]
- `colsample_bytree`: [0.6, 0.7, 0.8, 0.9]

**d) Class weights**
- Adjust `false_bullish_penalty` (higher = more conservative)
- Adjust `false_bearish_penalty`
- Ratio should favor `false_bullish_penalty > false_bearish_penalty`

**e) Label threshold**
- Adjust fee-adjusted LONG/SHORT/FLAT boundaries
- Modify `default_holding_periods_8h` in `LabelThresholdConfig`

### 2. Run evaluation

```bash
python evaluation/xgboost_eval.py --output-json
```

### 3. Decide: keep or revert

**If `sharpe_net` improved by > 0.02:**
```bash
git add signals/xgboost_config.py
git commit -m "autoresearch: <change_description> (sharpe_net +X.XX)"
```

**If not improved:**
```bash
git checkout signals/xgboost_config.py
# Log to results.tsv (automatic via run_autoresearch.py)
```

### 4. Repeat

Continue until:
- STOP file exists (`touch STOP` to halt)
- Time budget exhausted (`--time-budget-hours`)
- Max experiments reached (`--max-experiments`)

## Rules

1. **ONLY modify `signals/xgboost_config.py`**. Nothing else.
2. **NEVER modify `evaluation/xgboost_eval.py`** — it's the ground truth.
3. **NEVER modify `config/fee_model.py`** — fees are immutable.
4. **NEVER modify `execution/*`** — execution layer is sacred.
5. **Verify temporal safety** — no lookahead in new features.
6. **Check STOP file** before each experiment — halt if exists.
7. **One change at a time** — isolate effects.
8. **Log everything** — results.tsv is audit trail.

## Primary Metric: `sharpe_net`

This is the fee-adjusted Sharpe ratio of the strategy.
Higher is better. Improvement threshold: **0.02**.

Secondary metrics (for diagnostics):
- `ic`: Information coefficient (correlation with returns)
- `brier`: Calibration score (lower is better)
- `false_bullish_rate`: Critical error rate (should decrease)

## Anti-Patterns

- Don't tune multiple params at once
- Don't ignore SHAP importance
- Don't use features with temporal leakage
- Don't reduce `false_bullish_penalty` below 1.0
- Don't skip baseline comparison

## Example Session

```
# Setup
git checkout -b autoresearch/20260414-1500
python evaluation/xgboost_eval.py  # Baseline: sharpe_net=0.85

# Experiment 1: Increase max_depth
vim signals/xgboost_config.py  # max_depth: 6 → 7
python evaluation/xgboost_eval.py  # sharpe_net=0.89 (+0.04)
git commit -am "autoresearch: max_depth 6→7 (sharpe_net +0.04)"

# Experiment 2: Increase n_estimators
vim signals/xgboost_config.py  # n_estimators: 100 → 150
python evaluation/xgboost_eval.py  # sharpe_net=0.88 (-0.01)
git checkout signals/xgboost_config.py  # Revert

# Experiment 3: Lower learning_rate
vim signals/xgboost_config.py  # learning_rate: 0.1 → 0.05
python evaluation/xgboost_eval.py  # sharpe_net=0.91 (+0.02)
git commit -am "autoresearch: learning_rate 0.1→0.05 (sharpe_net +0.02)"
```

## Automated Mode

Use `run_autoresearch.py` for automated experiments:

```bash
# Run 10 experiments
python run_autoresearch.py --max-experiments 10

# Run overnight (8 hours)
python run_autoresearch.py --time-budget-hours 8

# Dry run (no actual changes)
python run_autoresearch.py --dry-run

# Optimize different metric
python run_autoresearch.py --metric ic
```
