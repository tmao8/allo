# Skill: HLS Hardware Discovery

## Purpose
This skill enables the agent to correctly identify the **Execution Model** (Sequential vs. Dataflow) and **Physical Constraints** of an HLS design before building an analytical model. This prevents the common mistake of assuming perfect parallelism where hardware contention exists.

## Discovery Steps

### 1. FSM Launch Sequence (Top-Level Dispatch)
- **Action:** Open `top.adb.xml` and scan the `<operation>` list for `call` nodes.
- **Check `st_id`:** Compare the `st_id` of successive kernel calls. 
  - If `st_id` increases (e.g., Kernel1 at 1, Kernel2 at 3, Kernel3 at 5), the hardware launches them **sequentially**.
  - **Rule:** A kernel's base `start_cycle` is at least its `st_id - 1`.

### 2. Physical Buffer Characterization
- **Action:** Locate every buffer (FIFO/RAM) in the top-level ADB.
- **Determine Sync Model:**
  - **CORE="FIFO" or "FIFO_SRL":** Stream-based. Kernels can overlap. Use the `dataflow_modeling` skill.
  - **CORE="RAM_*" (Ping-Pong Buffer):** Block-based. The consumer **MUST** wait for the producer to finish completely. 
  - **Logic:** `consumer_start = producer_finish`.

### 3. Resource Contention (AXI Port Bottlenecks)
- **Action:** List all bus ports (e.g., `m_axi_gmem0`). Identify which kernels map to which ports in the `top_csynth.xml` interface summary.
- **Detector:** If multiple kernels share the same `m_axi` port, they compete for bandwidth.
- **Heuristic:** Serialize kernels that share a port if their runtime is dominated by memory access.

### 4. Sequential Region Detection
- **Action:** Check for `#pragma HLS dataflow` in the source. Confirm in `top.adb.xml` by looking for `<is_dataflow>1</is_dataflow>`.
- **Policy:** If a region is **not** dataflow, it is strictly sequential. Sum the latencies of all operations.

## When to use this Skill
- Use this **BEFORE** writing any simulation code (`zero_cosim_model.py`).
- Use this when the sum of synthesis latencies (`626 cycles`) is significantly higher than the max individual kernel latency (`521 cycles`).

## Related Skills
- `adb_parsing`: For extracting the `st_id` and `CORE` values.
- `dataflow_modeling`: For simulating the overlaps once discovery is complete.
