# Skill: HLS Dataflow Modeling

## Purpose
This skill teaches the agent how to simulate a concurrent timeline for HLS designs using the `#pragma HLS dataflow` model.

## Core Concepts

### 1. Concurrent Execution (ADB-Controlled)
In HLS Dataflow, functions start as soon as their input tokens are available. To model this without Cosim, you must use **ADB Cycle Offsets**.
- **Rule:** A consumer start time is defined by the *relative cycle* it reads its first token vs when the producer writes it.
- **Formula:** `consumer_start = producer_start + (producer_first_write_cycle - consumer_first_read_cycle)`
- **Retrieval:** Use the `adb_parsing` skill to find these cycle numbers (States) in the `.adb.xml` files.

### 2. FIFO Synchronization & Backpressure
Kernels are connected by streams.
- **Token Tracking:** The simulator should track how many tokens are in each stream at any cycle `T`.
- **Stall Logic:** If `tokens == depth`, the producer stalls (stops incrementing its cycle). If `tokens == 0`, the consumer stalls.
- **Why this works:** This analytical simulation captures the exact "ripple" stalling that Cosim usually shows, but uses the synthesis schedule as the input.

### 3. Structural Analysis
To build a model, you must:
1.  **Extract Connectivity:** Read `kernel.cpp` to map which function writes to which stream, and which function reads from it.
2.  **Assign Properties:** Map the `Latency` and `II` from the reports (use the `hls_report_parsing` skill) to these functions.
3.  **Simulate:** Build a small Python script that tracks the "Ready Time" of each token across the graph.

## Python Modeling Example
```python
# Simple model for Producer -> Consumer (FIFO depth 16)
p_start = 0
p_latency = 100
p_ii = 1

# Consumer starts after 1st token is ready
# In HLS, with a stream, this is typically 1 cycle after producer starts if producer is pipelined
c_start = p_start + 1 
c_latency = 50
c_ii = 1
```

## Common Pitfall
- **Sequential assumption:** Do NOT just sum up the latencies. The total time `T` for two dataflow kernels is often `max(lat1, lat2) + small_offset`, not `lat1 + lat2`.
- **Top wrapper:** The top-level `ap_done` is only asserted when ALL dataflow processes are done.
