---
name: commit
description: Generate conventional commit from staged diff after /review passes
---

# Commit Staged Changes

## Prerequisites
Run `/review` first. Only proceed if all checks pass.

## Process

1. Run `git diff --staged` to analyze changes
2. Generate commit message in conventional format
3. Stage and commit

## Commit Format

```
<type>(<scope>): <description>

[optional body - only if change is non-obvious]
```

### Types
- `feat` — new feature or capability
- `fix` — bug fix
- `refactor` — code restructuring without behavior change
- `test` — adding or updating tests
- `docs` — documentation changes
- `chore` — maintenance, dependencies, config

### Scopes
- `training` — GRPO, SFT, LoRA, curriculum
- `execution` — BinanceExecutionClient, order management
- `data` — OHLCV, indicators, caching, CCXT
- `indicators` — technical indicators, signals
- `reward` — reward functions, fee model
- `evaluation` — metrics, IC, Brier, backtesting
- `config` — settings, environment, constants

## Rules

- Subject line ≤72 characters
- Use imperative mood ("add" not "added")
- No period at end of subject
- Body only for non-obvious changes (wrap at 72 chars)
- Reference issue numbers if applicable

## Execution

```bash
git commit -m "<generated message>"
```
