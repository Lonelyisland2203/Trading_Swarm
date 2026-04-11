# Training Layer

## GRPO Configuration
- Group size G=4, β=0.04 (KL penalty), ε=0.2 (clipping)
- Baseline: mean reward within group (no value network)
- Advantage: per-sample relative to group mean

## 3-Stage Curriculum
1. **SFT**: Structured JSON output, 500 examples, 3 epochs
2. **GRPO-Structure**: Reward = format compliance + reasoning coherence
3. **GRPO-Decision**: Reward = realized PnL (fee-adjusted) + calibration

## LoRA Configuration
- r=32, alpha=64, dropout=0.05
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- 4-bit base quantization (NF4), BF16 compute

## Asymmetric Reward Matrix
| Prediction | Outcome | Reward |
|------------|---------|--------|
| Bullish    | Up      | +1.0   |
| Bullish    | Down    | -1.5   |
| Bearish    | Down    | +1.0   |
| Bearish    | Up      | -0.8   |
| Neutral    | Any     | 0.0    |

## Promotion Criteria (stage advancement)
- IC (rank correlation) ≥ 0.05
- Brier score ≤ 0.25
- Statistical significance p < 0.05 (permutation test, n=1000)

## SFT Stage (complete)
- SFT data: data/sft_training_data.jsonl — structured reasoning traces (THESIS→EVIDENCE→RISK→DECISION)
- SFT adapter: adapters/sft_base/ — this is the starting point for GRPO training, not raw qwen3:8b
- SFT trainer: lr=2e-5, batch_size=1, grad_accum=16, epochs=3, early stopping on val loss (patience=2)**