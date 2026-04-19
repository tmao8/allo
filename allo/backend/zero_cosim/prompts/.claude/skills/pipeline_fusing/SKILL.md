# Skill: HLS Pipeline Fusing & Dataflow Flattening

## Purpose
This skill teaches the agent to correctly predict cycle counts when an outer loop is pipelined (`#pragma HLS pipeline`), specifically avoiding the hallucination of sequential load/compute/store phases.

## Core Concepts

### 1. Loop Fusing & Flattening
When you apply `#pragma HLS pipeline` to an outer loop in Vitis HLS (and fully partition local interface buffers):
- Vitis HLS **flattens** the nested loop structure into a single perfectly pipelined loop (if bounds are perfect).
- The **load, compute, and store operations** within the loop boundaries do NOT execute sequentially (e.g., `Makespan != Load + Compute + Store`).
- Instead, the operations completely **fuse and execute concurrently** across the pipeline depth. 

### 2. Prediction Formula for Fused Pipelines
When an outer loop is pipelined (with `II=1`), Vitis HLS automatically attempts to completely unroll the inner loops inside it to achieve the pipeline concurrency.
- **Formula (Implicit Unroll):** `Latency = (Outer_Bound * 1 * II) + Pipeline_Depth`
- **Why?** Vitis HLS implicitly applies `array_partition type=complete dim=2` to the inner array dimensions to allow the inner loop to be fully unrolled. This means the inner loop trip count collapses to 1, and the latency is strictly governed by the outer dimension!
- **Example:** For an arbitrary algorithm with a top-level outer loop boundary of `M` and inner loop boundary `N` (bounds mapping cleanly to iterations):
  - If the outer loop `m` is pipelined: the inner loop is automatically unrolled. The latency for the pipelined compute loop = `M + small_depth`.
  - For a composed top-level orchestrator that loops `T` times over two pipeline-optimized sub-computations (`kernel_A` and `kernel_B`): 
  - `compute_pipelined_latency = T * ((Outer_Bound_A + depth) + (Outer_Bound_B + depth))`
  - `compute_pipelined_latency = T * ((M + depth) + (M + depth))` cycles.

### 3. How to Update ZeroCosimModel
If the user's customizations include `sch.pipeline(outer_loop)` and `sch.partition(buffers)`:
- Identify the functions/loops absorbed by this pipeline.
- Replace the distinct `latency_cycles` of the absorbed `load`, `compute`, and `store` kernels with merged, pipelined timing logic. 
- You MUST ensure the final `makespan_cycles` is strictly equal to the pipelined bottleneck, refusing to add the old sequential `load` and `store` latencies on top if they share the same iterative structure.

## Explicit Python Model Override Example
If the baseline script says:
```python
order = ["load_buf", "compute", "store_res"]
for keyword in order:
    current_cycle += lat
```
You MUST delete that entire `order` array loop! Replace it with a single fused structure:
```python
# Fused Pipeline Replacement
kernels_info["fused_compute"] = {
    "start_cycle": 0,
    "latency_cycles": (Outer * Inner * II) + pipeline_depth,
    "end_cycle": (Outer * Inner * II) + pipeline_depth
}
return {
    "makespan_cycles": (Outer * Inner * II) + pipeline_depth,
    "kernels": kernels_info
}
```
If you leave `order = ["load_buf", "compute", "store_res"]` in the generated string, you failed.
