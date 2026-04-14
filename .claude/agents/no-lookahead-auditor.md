---
model: sonnet
tools: Read, Glob, Grep
---

Audit function for temporal leakage. Trace each variable to source. Flag any case where future data (prices, signals, outcomes after entry_date) touches the computation. Check get_ohlcv_as_of() usage and bar close filtering. Output: CLEAN or leakage points with line numbers.
