# Zero-Cosim

Analytical cycle modeling for Vitis HLS designs. Replaces slow C/RTL co-simulation
with a Python model written by an LLM agent from CSynth + ADB artifacts, then
asks the same agent to *predict* how user-applied Allo schedule primitives will
shift the cycle count — without re-running synthesis.

The goal is to answer "what does my schedule cost?" in seconds instead of the
~10–60 minutes Vitis cosim takes.

## How it works

There are two phases, each driven by an LLM agent over the same prompt scaffolding.

### Phase 1 — Build the baseline model
Input: a Vitis HLS project that has finished `csyn` for the **unscheduled** kernel.

The agent has filesystem tools (`list_directory`, `read_file`) and uses them to:
1. Walk `out.prj/solution1/syn/report/*.xml` to harvest per-loop `PipelineII`,
   `PipelineDepth`, `IterationLatency`, and overall latency.
2. Walk `out.prj/solution1/.autopilot/db/*.adb.xml` to extract FSM state IDs and
   dataflow/FIFO connectivity.
3. Emit a Python class `ZeroCosimModel` (saved as
   `out.prj/solution1/zero_cosim_model_impl.py`) that, when instantiated against
   the same `solution_dir`, reproduces the baseline cycle count from synthesis.

### Phase 2 — Predict the optimized cycle count
Input: the baseline `ZeroCosimModel` source from Phase 1, the new MLIR (after
the user applies `pipeline`, `partition`, `reuse_at`, `buffer_at`, `dataflow`,
etc.), and the list of customizations as `(primitive, args, kwargs)` tuples.

The agent rewrites the baseline class — same shape, new arithmetic — so its
`report_cycle()` reflects the optimized hardware. **No filesystem tools** in this
phase: everything the agent needs (MLIR, customization list, baseline code) is in
the prompt.

The returned object satisfies `protocol.ZeroCosimModel` (see `protocol.py`) and
exposes `report_cycle()` and `dump_trace()`.

## Backends

Two interchangeable agents drive both phases:

| Backend | Model | Cost | When to use |
|---|---|---|---|
| `openrouter` | `google/gemini-3-flash-preview` | API tokens | CI, no Claude CLI available, batch runs |
| `claude_cli` | local `claude --print` subprocess | uses your CLI quota | local dev, Pro/Max plan in use |

Selection precedence (highest first):
1. Explicit `backend=` arg on `build_cosim_model` / `predict_performance`.
2. `ZERO_COSIM_BACKEND` env var (`"openrouter"` or `"claude_cli"`).
3. Auto: prefers `claude_cli` if `_find_claude_cli()` resolves a binary,
   else falls back to OpenRouter.

OpenRouter mode needs `OPENROUTER_API_KEY` (env var or `api_key=` arg).
Claude CLI mode needs the CLI on `$PATH` and runs with `--dangerously-skip-permissions`
plus `--max-budget-usd` enforced by the subprocess wrapper.

## Public API

Both methods hang off `allo.customize(...)` — i.e. a `Schedule` object.

```python
import allo

# build the baseline model from synthesis artifacts
sch = allo.customize(kernel_fn, instantiate=[float32, N])
sch.build(target="vitis_hls", mode="csyn", project="my_kernel.prj")  # required first
baseline_code = sch.build_cosim_model(project="my_kernel.prj")        # phase 1

# apply Allo primitives, then predict
sch.pipeline("k0")
sch.buffer_at(sch.out, axis="i0")
predicted = sch.predict_performance(baseline_code)                    # phase 2
print(predicted.report_cycle())
# {'makespan_cycles': 7521, 'kernels': {'mm1': {...}, 'ele_add': {...}}}
```

`baseline_code` is just the Python source string of the generated class;
it's cached under `my_kernel.prj/out.prj/solution1/zero_cosim_model_impl.py`
so subsequent runs reuse it.

Both methods accept `backend=` and `api_key=`. See
`examples/polybench/jacobi_2d_interactive.py`,
`examples/polybench/gemm_interactive.py`, and
`examples/polybench/two_mm_interactive.py` for end-to-end flows that compare
predictions against actual CSynth ground truth.

## Running the examples

```bash
source /opt/xilinx/Vitis_HLS/2023.2/settings64.sh
source /work/shared/common/allo/setup-llvm-main.sh

# OpenRouter
export OPENROUTER_API_KEY=sk-or-...
export ZERO_COSIM_BACKEND=openrouter
python examples/polybench/jacobi_2d_interactive.py

# Claude CLI
export ZERO_COSIM_BACKEND=claude_cli
python examples/polybench/jacobi_2d_interactive.py
```

Each example prints the agent's predicted cycle count, then runs Vitis HLS for
real and prints the CSynth ground truth so you can eyeball the error.

## Directory layout

```
zero_cosim/
├── auto_agent.py      # agent orchestration: tool loop, prompt assembly, both backends
├── customize.py hooks # Schedule.build_cosim_model + Schedule.predict_performance
├── loader.py          # dynamic import of generated ZeroCosimModel class files
├── protocol.py        # the runtime-checkable Protocol the generated class targets
└── prompts/
    ├── .clauderules/
    │   └── project.md         # high-level rules (always loaded)
    └── .claude/skills/
        ├── pipeline_fusing/   # MLIR topology → cycles math (the main prediction rule)
        ├── dataflow_modeling/ # T = sum(stages) + (n-1)*max(stages)
        ├── hls_discovery/     # baseline-phase: FSM/FIFO/RAM identification
        ├── hls_report_parsing/# baseline-phase: XML extraction
        ├── adb_parsing/       # baseline-phase: state IDs and connectivity
        └── trace_validation/  # causality and overlap checks
```

## Prompt and skill system

The prompt sent to the agent is assembled by `build_system_prompt()`:
1. `prompts/.clauderules/project.md` — overall mission and modeling principles.
2. Every `SKILL.md` under `prompts/.claude/skills/` — concatenated in directory order.
3. A short tail with submission/protocol constraints.

To add a new rule, drop a `SKILL.md` into a new `skills/<name>/` directory.
The loader picks it up on the next run; nothing to register.

For the prediction phase, `estimate_customization*` injects an additional
inline block (above the skills) that carries the most cycle-critical rules —
topology resolution, reduction-loop II, AXI burst attribution. These live in
`auto_agent.py` rather than a skill file because the agent ignored them when
buried (see commit history).

## Knobs

- `_summarize_mlir_for_prompt(mlir_ir, budget=12000)` in `auto_agent.py` —
  trims long MLIR while *always* preserving the last `func.func` (the top
  function carries the orchestration loops).
- Claude CLI subprocess timeout: 900s for both build and prediction.
- OpenRouter model: hardcoded to `google/gemini-3-flash-preview` in
  `build_model_via_agent` / `estimate_customization`.
- Phase 1 generated model lives at
  `<project>/out.prj/solution1/zero_cosim_model_impl.py`. Delete it to force
  a rebuild.

## Known limitations

- Pipeline depth is guessed (≈5–15) rather than read from the baseline XML.
  Closing this gap is the largest remaining accuracy lever.
- Primitives without explicit prompt coverage: `unroll`, `parallel`, `split`,
  `to`, `systolic`, multi-instance `compose`. They may still predict OK but
  haven't been validated.
- Data-dependent loop bounds (cholesky, lu, ludcmp, trisolv) — the prompt
  doesn't yet teach triangular trip counts; expect ~2× over-prediction.
- Residual ~5% error on sequential matmul comes from per-loop FSM state
  transition overhead that the prompt doesn't model.

## Files at a glance for new contributors

If you're debugging an inaccurate prediction, the order to read things in is:
1. The example's printed prediction vs. CSynth ground truth.
2. The agent's generated `ZeroCosimModel` (printed near the prediction or saved
   to a temp file by the subprocess wrapper) — the math is right there.
3. `auto_agent.py: estimate_customization*` for the inline topology rule.
4. `prompts/.claude/skills/pipeline_fusing/SKILL.md` for the detailed worked
   examples.
