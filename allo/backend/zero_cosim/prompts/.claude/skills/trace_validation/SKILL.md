# Skill: Trace Validation

## Purpose
This skill teaches the agent how to verify the accuracy of its generated performance model against the ground truth.

## Validation Steps

### 1. Structural Checks
Before comparing with ground truth, ensure the `trace.json` is healthy:
- **Causality:** Does `consumer_start` happen after (or at) `producer_start`?
- **Hierarchy:** Do all child kernel events fit within the `top()` function's start/end window?
- **Overlap:** If dataflow is used, do kernels actually overlap in time? (If they are sequential, the model is likely wrong).

### 2. Physical Validation
Use the existing scripts to check against cosim:
- **Run comparison:** `python3 compare_traces.py`
- **Goal:** Total latency error < 20%.
- **Goal:** Per-kernel error < 20% for major computation units.

### 3. Iterative Refinement
If the error is too high:
1.  **Check II:** Did you account for `Initiation Interval > 1`?
2.  **Check Stalls:** Look for FIFO bottle-necks in `kernel.cpp`.
3.  **Check Offsets:** Adjust the start-time offsets between kernels. Producers usually provide tokens quickly, but consumers might wait for a few cycles of header processing.

## Output Requirements
Final trace must be saved as `trace.json` in the project root.
Format: [Chrome Tracing JSON](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/edit).
- Required field: `displayTimeUnit: "ns"` (or `"us"` if relevant, but be consistent).
