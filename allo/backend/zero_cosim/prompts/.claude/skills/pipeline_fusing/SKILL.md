# Skill: HLS Pipeline Fusing & Dataflow Flattening

## Purpose
Teach the agent to predict cycle counts when an outer loop is pipelined (`#pragma HLS pipeline`),
without hallucinating sequential load/compute/store phases AND without over-deleting sibling memory stages.

## Core Concepts

### 1. What `pipeline(outer)` actually does in Vitis HLS
When `pipeline` is applied to an OUTER loop of a perfectly-nested loop nest, Vitis HLS:
- Pipelines that outer loop with the requested `II` (usually 1).
- **Auto-unrolls (spatially) the inner loops to fit the II.** Inner loop iterations become parallel hardware, NOT sequential cycles. Required array partitioning is implicitly added so the unrolled inner body has enough memory ports.
- Result: the compute loop's cycle count is governed by the **outer trip count only**, plus a small pipeline depth that absorbs the inner work.

### 2. Prediction Formula for a Pipelined Outer Loop
For a 2D nest `for i in [0, M): for j in [0, N): body(i,j)` with `pipeline(i)` and `II=1`:

```
compute_latency = (M - 1) * II + pipeline_depth     # NOT M*N*II
                 ~= M + pipeline_depth
```

The inner dimension N does NOT multiply the latency — it spatially unrolls.
`pipeline_depth` is typically a few dozen cycles (e.g. 30–60) and accounts for all the
chained arithmetic of the unrolled inner body.

If the customization is `reuse_at(...)` + `pipeline(outer)` + `partition(reuse_buf, dim=0)`
(a classic stencil line-buffer schedule), the same formula applies — the line/window
buffer absorbs the inner-loop work spatially.

### 3. Where the pipeline applies in the MLIR
**Read the MLIR carefully** to find which function and which loop carries the pipeline:
- A loop with `pipeline_ii = K : ui32` attribute is the pipelined loop. The `K` is the
  REQUESTED II — not necessarily the achieved II (see §3a below).
- Trip count = the outer affine.for bound.
- Inner affine.for loops nested inside the pipelined loop are absorbed (spatially unrolled).
- Outer affine.for loops WRAPPING the pipelined loop (siblings in the MLIR) are NOT absorbed —
  they multiply the pipelined-loop latency by their trip count.

### 3a. Achieved II ≠ Requested II for Loops with Loop-Carried Dependencies
When the pipelined loop is a REDUCTION (`out += a*b`, `acc = acc * x`, etc.) or otherwise has a
loop-carried dependency through a register/memory location, Vitis HLS cannot pipeline at II=1.
The achieved II is bounded by the latency of the dependent operator chain on the critical path:

| Dependency op (float32)              | Typical achieved II |
| ------------------------------------ | ------------------- |
| Integer add / sub                    | 1                   |
| Float add (fadd_32)                  | ~4                  |
| Float mul (fmul_32)                  | ~4                  |
| Float mul+add fused (mac_muladd)     | ~4–5                |
| Float divide                         | ~10–15              |
| Float sqrt                           | ~15–30              |
| Chained dependent ops in one iter    | sum of stages on critical path |

**Detection in the MLIR**: look inside the pipelined loop's body for an `affine.load → arith.op → affine.store`
pattern targeting the same memref/index across iterations. If iteration K reads a value that iteration
K-1 wrote, you have a loop-carried dependency. Common patterns:
- `out[i,j] += A[i,k] * B[k,j]` inside `for k in reduction(K)` → fadd dep → II ≈ 4
- `acc = acc * x[k]` inside `for k in reduction(K)` → fmul dep → II ≈ 4
- `s = s + sqrt(...)` → fadd + fsqrt → II ≈ 4 (fadd is the recurrence path, sqrt is forwarded)

**Formula with reduction II:** `latency = (trip - 1) * achieved_II + depth ≈ trip * achieved_II + depth`.

For matmul-style reductions specifically, prefer `II = 4` for float32 over `II = 1` unless the
customizations also include a tree-reduction unroll or operator chaining annotation.

### 3b. `buffer_at` Adds Sibling Init / Writeback Loops
`sch.buffer_at(target_buf, axis=outer_iv)` hoists a scratch buffer outside the inner compute and
fills it at the start of each outer iteration, then writes back at the end. Vitis HLS inserts:
- an **init loop** before the inner compute: trip = size of inner dim, II = 1, depth small (~2–4)
- the **inner compute loop** (with the buffer accessed in registers/RAM)
- a **back / writeback loop** after the inner compute: trip = inner dim, II = 1, depth small (~2–4)

Per outer iteration these three are SIBLINGS that run sequentially. So:
```
per_outer_iter ≈ init_loop_latency + inner_compute_latency + back_loop_latency
             ≈ (inner_dim + 2) + (compute latency from §2/§3) + (inner_dim + 2)
total = outer_trip * per_outer_iter + small_FSM_overhead
```

Detection in the MLIR: a `memref.alloc {name = "<buf>"}` immediately inside the outer
`affine.for`, plus extra pipelined loops surrounding the main compute body, indicate `buffer_at`.

### 4. When the Pipelined Loop Is Inside a Sub-Function Called by a Parent Loop
This is the most common composed-kernel case. Example MLIR shape:
```
func.func @compute_A(...) {
  affine.for %i = 0 to N { ... } { pipeline_ii = 1 }   // pipelined compute body
}
func.func @kernel_top(...) {
  affine.for %t = 0 to TSTEPS {                          // outer orchestration loop
    func.call @compute_A(...)
    func.call @compute_B(...)
  }
}
```
The compute kernels each have latency `~N + depth`. The top kernel's `m` loop (TSTEPS iterations)
runs them sequentially per iteration. **You MUST multiply by the outer trip count:**
```
m_loop_latency = TSTEPS * (compute_A_latency + compute_B_latency + small_per_iter_overhead)
```

### 5. Sibling Memory Bursts at the Top Level
Vitis HLS infers AXI burst loops (`l_S_load_bufN_...`, `l_S_store_resN_...`) for each
`memref<...>` argument that the TOP function reads or writes. These appear in the baseline
report but NOT as explicit affine.for loops in the MLIR.

- The presence and count of these burst loops depends on the TOP function's signature, not on
  any pipeline pragma. Pipelining an inner compute loop does NOT remove them.
- They are **sibling** stages to the compute loop, executed sequentially around it.
- **Rule:** retain ONE `load_buf` per memref input and ONE `store_res` per memref output
  of the TOP function in the MLIR (typically what the baseline already has, unchanged).
- If the top function in the MLIR is a SUB-FUNCTION (e.g. you customized `compute_A` directly
  and the MLIR module's last `func.func` is `compute_A`, not `kernel_jacobi_2d`), then the
  burst stages are governed by `compute_A`'s memref args (typically 1 input + 1 output =
  1 `load_buf` + 1 `store_res`), NOT the parent kernel's args.

### 6. How to Update `ZeroCosimModel.report_cycle`
1. **Identify the top function in the MLIR** (the last `func.func` in the module).
2. **Find the pipelined loop** (`pipeline_ii` attribute) in the MLIR and compute
   `compute_latency = outer_trip * II + depth`.
3. **Identify outer orchestration loops** in the top function (affine.for loops that wrap
   `func.call` ops). Multiply downstream sub-function latencies by these trip counts.
4. **Keep sibling AXI burst stages** (`load_buf*`, `store_res*`) — one per memref arg of the
   top function. Their latencies (~`array_size + small_overhead`) are typically unchanged
   by compute-loop pipelining.
5. **DO NOT** multiply compute latency by inner loop trip counts when the inner loops are
   absorbed by the pipeline. DO NOT delete the sibling memory stages.

## Worked Example: 2D Stencil + TSTEPS

Customizations on `compute_A`: `reuse_at(A, i)`, `reuse_at(reuse_0, j)`, `pipeline(i)`,
`partition(reuse_0, dim=0)`, `partition(reuse_1, dim=0)`. Same on `compute_B`. Composed top
kernel has `for t in range(TSTEPS): compute_A(); compute_B()`.

With N=90, TSTEPS=40, pipeline_depth ~= 35:
- `compute_A_latency = 90 * 1 + 35 = 125`
- `compute_B_latency = 90 * 1 + 35 = 125`
- `m_loop_latency = TSTEPS * (125 + 125 + small_per_iter_overhead) ~= 40 * 254 = 10160`
- `load_buf_A ~= 8100`, `load_buf_B ~= 8100`, `store_res_A ~= 8102`, `store_res_B ~= 8102`
- `makespan ~= 8100 + 8100 + 10160 + 8102 + 8102 + small ~= 42597`

If instead you customize ONLY `compute_A` standalone (MLIR top func = `compute_A`, no TSTEPS,
no sibling compute_B), the model has just one memref input + one memref output:
- `compute_A_pipeline_latency ~= 125`
- `load_buf_A ~= 8100`, `store_res_B ~= 8101`
- `makespan ~= 8100 + 125 + 8101 + small ~= 16344`
