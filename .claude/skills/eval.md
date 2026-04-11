---
name: eval
description: Evaluate trained model on held-out test window with promotion criteria
---

# Post-Training Evaluation

Run after any training run completes. Evaluates against promotion criteria.

## Execution

```bash
python -m evaluation.evaluate_candidate \
  --checkpoint latest \
  --window held_out \
  --output_format json
```

## Output Format

### Primary Metrics

| Metric | Value | Threshold | Pass? |
|--------|-------|-----------|-------|
| IC | X.XXX | ≥ 0.05 | ✓/✗ |
| Brier | X.XXX | ≤ 0.25 | ✓/✗ |
| p-value | X.XXXX | < 0.05 | ✓/✗ |
| MACE | X.XXX | — | — |
| N | XXX | ≥ 200 | ✓/✗ |

### Promotion Criteria

All must pass:
- **IC ≥ 0.05**: Rank correlation with forward returns
- **Brier ≤ 0.25**: Probability calibration
- **p-value < 0.05**: Statistical significance (permutation test, n=1000)
- **N ≥ 200**: Minimum sample size for reliable metrics

MACE (Mean Absolute Calibration Error) is informational only.

### Regime-Stratified IC

If VIX regime data available:

| Regime | IC | N | Notes |
|--------|-----|---|-------|
| Low Vol (VIX < 15) | X.XXX | XX | |
| Normal (15-25) | X.XXX | XX | |
| High Vol (VIX > 25) | X.XXX | XX | |
| Crisis (VIX > 35) | X.XXX | XX | May have low N |

## Decision

### All Pass
```
✓ PROMOTION CANDIDATE — ready for human review.

Checkpoint: training/checkpoints/grpo_step_1000_20240115_120000
IC: 0.062 (threshold: 0.05)
Brier: 0.231 (threshold: 0.25)
p-value: 0.003 (threshold: 0.05)

Next steps:
1. Human review of sample predictions
2. Compare to previous promoted adapter
3. If approved: promote to production/adapters/
```

### Any Fail
```
✗ NOT READY

Failed criteria:
- IC: 0.042 < 0.05 threshold
- p-value: 0.082 > 0.05 threshold

Recommendations:
- [ ] Review training data quality
- [ ] Check for temporal leakage
- [ ] Consider longer training
- [ ] Examine regime breakdown for weak spots
```

## Additional Diagnostics

If evaluation fails, also report:
```bash
# Prediction distribution
python -m evaluation.diagnostics --checkpoint latest --report distribution

# Calibration curve
python -m evaluation.diagnostics --checkpoint latest --report calibration

# Confusion matrix by confidence bucket
python -m evaluation.diagnostics --checkpoint latest --report confusion
```
