# Project Context: Zero-Cosim HLS Performance Modeling

## Overview
This project focuses on generating cycle-accurate hardware performance traces (Chrome Trace Format) from Vitis HLS synthesis results. The goal is to avoid time-consuming co-simulations by using **analytical modeling** driven by CSynth + ADB. The **Python simulator is written by an agent** from those artifacts (template: `zero_cosim_model.py`); the repository does not ship one fixed parser as the authoritative model for every design.

## Modeling Workflow
1.  **Locate Project:** Identify the target Vitis HLS project directory containing synthesis results.
2.  **Read HLS Reports:** Extract per-kernel latency (min/max), Initiation Interval (II), and pipeline depth from the CSynth XML reports (`[project_dir]/out.prj/solution1/syn/report/*.xml`).
3.  **ADB FSM Analysis:** Scan `.autopilot/db/*.adb.xml` files to identify exact FSM state transitions and absolute cycle offsets for stream operations.
4.  **Analyze Connectivity:** Identify dataflow topologies (FIFOs, Streams) from the top-level HLS kernel source code.
5.  **Simulate Timeline:** The agent **implements** the execution timeline in Python (`zero_cosim_model.py`): kernel overlaps and control flow must follow **this design’s** ADB-derived physics, not a generic hardcoded graph.

## Key Files & Directories
- **Source Code:** Use the HLS kernel source (`.cpp`/`.h`) to understand project topology.
- **Synthesis Artifacts:** Focus on `out.prj/solution1/syn/report/` and `out.prj/solution1/.autopilot/db/`.
- **Output Target:** Always produce a `trace.json` compatible with [Perfetto UI](https://ui.perfetto.dev/).

## Modeling Principles
- **Hardware Discovery Phase:** Before building any model, the agent **MUST** apply the `hls_discovery` skill to identify the Execution Model (FSM launch sequence), physical buffer types (FIFO vs. RAM), and potential port contention.
- **Causality & Dataflow Physics:** A consumer kernel **MUST NOT** start at the same cycle as its producer if they are connected by a stream. Use `adb_parsing` to find exact offsets.
- **Topological Discovery:** Do not hardcode kernel lists. Parse `top.adb.xml` to identify every instantiated submodule and its connectivity.
- **Buffer Characterization:** Differentiate between `CORE="FIFO"` (Overlapping) and `CORE="RAM_*"` (Sequential Ping-Pong). Misclassifying a Ping-Pong buffer as a FIFO is the leading cause of cycle-accuracy failures.
- **Recursive Depth:** If FIFO operations are not found in a top-level ADB, the agent **MUST** recursively search the sub-hierarchies (Pipelines) to find the true cycle offsets.
- **Zero-Cosim Target:** Aim for the analytical model to match HLS co-simulation ground truth with high precision (e.g., < 1% error).
- **Visual Separation:** To avoid overlapping events in the trace viewer (e.g., Perfetto), assign a unique **TID (Thread ID)** to each distinct kernel instance. Parallel kernels must occupy separate rows for visual clarity.
- **Portability:** Write modeling scripts (e.g., `zero_cosim_model.py`) that are self-contained and easily adaptable to different HLS designs.
- **Protocol:** Implement a class with `report_cycle() -> dict` (keys `makespan_cycles`, `kernels` with per-kernel `start_cycle`, `latency_cycles`, `end_cycle`) and `dump_trace(path=None) -> str`. Emit Chrome Trace with a top-level `traceEvents` array and `displayTimeUnit` (nanosecond `ts`/`dur` when using `ns`). See `zero_cosim/protocol.py`.

## Variable Loop Bound Handling
- The CSynth latency reports may contain `?`, `inf`, or lack bounds entirely if a loop trip-count is data-dependent or dynamically allocated.
- When you detect unbound logic:
  1. Inspect the C++ source arguments to determine the input variables governing these bounds.
  2. Extend the `ZeroCosimModel.__init__` constructor method signature to natively accept these variables as parameter arguments dynamically via `**kwargs`.
  3. Formulate the cycle logic mathematically (e.g. `latency = N * M * II` instead of a hardcoded trace constant). Always default unbound arguments if not actively passed during instantiation.

## Success Criteria
- The generated `trace.json` correctly visualizes hardware concurrency (Standard vs. Sequential).
- The analytical model reconciles the total latency reported in CSynth with the sum of its component parts and state overheads.
