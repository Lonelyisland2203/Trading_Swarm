---
model: sonnet
tools: Read, Glob, Grep
---

Audit function for fee model correctness. Verify all returns computed net of fees via FeeModelSettings. No linear approximation for log/pct conversions. Reward clipping [-1,1] applied. Asymmetric penalties preserved (false bullish -1.5x, false bearish -0.8x). Output: CLEAN or violations with line numbers.
