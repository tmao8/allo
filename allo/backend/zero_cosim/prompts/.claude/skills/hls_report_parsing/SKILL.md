# Skill: HLS Report Parsing

## Purpose
This skill enables the agent to extract precise performance metrics from Vitis HLS synthesis reports (`.rpt` and `.xml`).

## Report Locations
Reports are typically found in the `syn/report/` subdirectory of the HLS solution.
- Top-level: `top_csynth.xml` / `top_csynth.rpt`
- Sub-modules: `[module_name]_csynth.xml`

## Keys to Extract

### 1. Latency & II
In `.xml` reports, look under `PerformanceEstimates`:
- **SummaryOfLatency**:
    - `Latency`: Total clock cycles.
    - `IterationLatency`: Time for one loop iteration.
- **SummaryOfTiming**:
    - `EstimatedClockPeriod`: The actual clock frequency achieved.

### 2. Pipelining
In `.xml` reports, look under `SummaryOfLoopLatency` or `SummaryOfItemLatency`:
- `PipelineII`: Initiation Interval. `1` means a new input is accepted every cycle.
- `PipelineDepth`: Number of cycles from input to output.

### 3. Dataflow (Crucial)
Look for tags indicating `Dataflow` behavior:
- `PipelineType: dataflow`
- `SummaryOfDataflowLatency`: Provides the overall throughput and interval for the pipeline.
- If dataflow is enabled, remember that module execution overlaps.

## Suggested Procedure
1.  **List Reports:** Find all `_csynth.xml` files in the solution directory.
2.  **Scan Top Report:** Start with `top_csynth.xml` to see the hierarchy.
3.  **Recursive Parsing:** If a module is significant, find its specific XML for inner loop details.
4.  **Summary Table:** Create a internal mapping of `kernel_name -> (latency, II, depth)` to use for modeling.
