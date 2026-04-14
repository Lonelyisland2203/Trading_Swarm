# Autoresearch Layer (Karpathy-pattern adapted for trading)

## Core Concept
Autonomous XGBoost improvement loop. Agent modifies ONE file (xgboost_config.py),
evaluates via walk-forward CV, keeps or discards based on metric improvement.

## Architecture (3 files that matter)
- evaluation/xgboost_eval.py — FIXED. Walk-forward CV, metrics. DO NOT MODIFY.
- signals/xgboost_config.py — THE ONLY FILE AGENT EDITS. Features, hyperparams, class weights.
- autoresearch/program.md — Agent instructions. Human edits this.

## Evaluation Metric
Primary: Sharpe_net (net-of-fee Sharpe ratio on walk-forward OOS)
Improvement threshold: Sharpe_net increase > 0.02 to keep experiment.

## Experiment Loop
1. git checkout -b autoresearch/<tag> from main
2. Agent reads xgboost_config.py + recent results.tsv
3. Agent proposes ONE change
4. Run: python evaluation/xgboost_eval.py --config signals/xgboost_config.py
5. If Sharpe_net improved > 0.02: git commit, update results.tsv, continue
6. If not: git revert, log result, try different direction
7. Repeat until --max-experiments or --time-budget-hours

## Safety
- Only xgboost_config.py modifiable. Never fee_model, execution, data layer.
- Every experiment verified for temporal safety
- STOP file halts loop immediately