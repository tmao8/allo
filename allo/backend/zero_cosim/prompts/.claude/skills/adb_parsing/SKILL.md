# Skill: ADB Architecture Parsing

## Purpose
This skill allows the agent to extract cycle-accurate scheduling data from the HLS internal Architecture Database (`.adb.xml` or `.sched.adb.xml`). This data replaces the need for simulation (Cosim) by providing the exact RTL state machine behavior statically.

## Core Extraction logic

### 1. Identify ADB Files
Look in the `.autopilot/db/` directory for files named `[module_name].sched.adb.xml`. These contain the FSM (Finite State Machine) and operation schedule.

### 2. Parse State Machine (FSM)
- Find the `<state_list>` tag.
- Each `<state id="N">` represents a set of operations that happen at a specific cycle (or set of cycles) in the module's execution.
- Map the total number of states to the module latency.

### 3. Extract Stream/Port Offsets (Static Alignment)
To accurately overlap Dataflow kernels, you must find the **First Access Cycle**.
- **Flexible Matching**: Do not search for a fixed string like `ap_fifo`. Instead, search for the **Port Name** (e.g., `v1201`) and look for `read` or `write` operations that target it.
- **Handling Sub-Pipelines (Recursive Analysis)**: If the top-level ADB calls a function (e.g., `NEST_0_Pipeline_l_nest_i`), and the FIFO operation isn't in the top level:
  1. Open the sub-pipeline's `.adb.xml`.
  2. Find the offset within that sub-pipeline.
  3. Add the sub-pipeline's **Call Cycle** (the state ID where it's called in the parent) to the internal offset.
- **Critical Constant:** The hardware "FIFO Offset" for this connection is exactly `X - Y`.
- **Logic:** Kernel B starts at Cycle `KernelA_Start + (X - Y)`.

### 4. Memory and Pipeline CHARACTERISTICS
- **Pipeline Depth**: Check if the module is pipelined. If so, the `latency` is the total duration, but the `Interval (II)` is the cycle count between successive iterations.
- **Protocol Overhead**: ADB files show C-states (Control states). The first few states are often "Setup" and the last few "Teardown". Ensure your model accounts for these.

## Python Parsing Pattern
When implementing a parser for this, use `xml.etree.ElementTree` to navigate:
1.  Find all `read` and `write` nodes.
2.  Correlate them back to their parent `<state>`.
3.  Calculate relative cycle offsets for the internal pipeline.

## Use Case: Zero-Cosim Trace
- Stop using "Best Case Latency" from high-level reports.
- Instead, use the **ADB Total Cycle Count** (last state ID in the final transition). 
- This accounts for the true hardware protocol overhead.
