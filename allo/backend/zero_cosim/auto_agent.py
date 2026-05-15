import os
import glob
import json
import re
import subprocess
import shutil
from pathlib import Path
from typing import Any, Optional
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .loader import load_zero_cosim_class


def _summarize_mlir_for_prompt(mlir_ir: str, budget: int = 12000) -> str:
    """
    Keep the MLIR small enough to fit the prompt but ALWAYS preserve the
    top-level `func.func` (which contains the orchestrating outer loops
    like TSTEPS that drive how often sub-kernels are called).

    Strategy:
      - If the whole IR fits, return it.
      - Otherwise: keep the LAST `func.func` (top function in MLIR module
        ordering) intact, and prefix earlier functions only if budget allows.
    """
    if len(mlir_ir) <= budget:
        return mlir_ir

    # Find all func.func boundaries
    func_starts = []
    pos = 0
    while True:
        idx = mlir_ir.find("func.func", pos)
        if idx == -1:
            break
        func_starts.append(idx)
        pos = idx + 1
    if not func_starts:
        return mlir_ir[:budget] + "\n... [MLIR Truncated] ..."

    top_start = func_starts[-1]
    top_block = mlir_ir[top_start:]
    if len(top_block) >= budget:
        # Top function alone exceeds budget; truncate its body only.
        return mlir_ir[:200] + "\n... [earlier funcs elided] ...\n" + top_block[: budget - 250] + "\n... [MLIR Truncated] ..."

    remaining = budget - len(top_block) - 80
    prefix = mlir_ir[:remaining]
    return prefix + "\n... [some sub-function bodies elided for brevity] ...\n" + top_block


def _extract_python_from_text(text: str) -> Optional[str]:
    """Extract a Python code block containing ZeroCosimModel from raw text."""
    # Try fenced code blocks first
    patterns = [
        r'```python\s*\n(.*?)\n```',
        r'```\s*\n(.*?)\n```',
    ]
    for pat in patterns:
        matches = re.findall(pat, text, re.DOTALL)
        for m in matches:
            if 'class ZeroCosimModel' in m:
                return m.strip()
    # Fallback: find the class directly
    idx = text.find('class ZeroCosimModel')
    if idx != -1:
        # Walk backward to find any imports before the class
        line_start = text.rfind('\nimport ', 0, idx)
        if line_start == -1:
            line_start = text.rfind('\nfrom ', 0, idx)
        if line_start == -1:
            line_start = idx
        else:
            line_start += 1  # skip the leading newline
        return text[line_start:].strip()
    return None

def _load_skills(workspace_dir: Path) -> str:
    skills_text = ""
    skills_dir = workspace_dir / "prompts" / ".claude" / "skills"
    if skills_dir.exists():
        for skill_md in glob.glob(str(skills_dir / "**" / "SKILL.md"), recursive=True):
            skill_name = Path(skill_md).parent.name
            skills_text += f"=== SKILL: {skill_name} ===\n"
            skills_text += Path(skill_md).read_text() + "\n\n"
    return skills_text

def build_system_prompt(workspace_dir: Path) -> str:
    """Assembles the system prompt from the rules and skills in the repository."""
    prompt = "You are an expert Hardware Performance LLM Agent.\n\n"
    
    project_md = workspace_dir / "prompts" / ".clauderules" / "project.md"
    if project_md.exists():
        prompt += f"=== PROJECT RULES ({project_md.name}) ===\n"
        prompt += project_md.read_text() + "\n\n"
        
    prompt += _load_skills(workspace_dir)
            
    prompt += "\nYour mission is to analyze the provided `solution_dir`."
    prompt += "\nUse the provided tools to inspect the directory, read HLS XML/ADB reports, and write out a fully functioning Python ZeroCosimModel class."
    prompt += "\nYou must NOT use hardcoded logic or just write a stub. Read the ADB files to accurately extract states!"
    prompt += "\nCRITICAL: the generated class MUST be named `ZeroCosimModel` and it MUST accept `solution_dir` in its constructor: `def __init__(self, solution_dir, *, clock_period_ns=3.33):`"
    prompt += "\nCRITICAL: you MUST implement a `def report_cycle(self, **kwargs):` method that returns a dictionary like `{'makespan_cycles': int, 'kernels': {kernel_name: {'start_cycle': int, 'latency_cycles': int, 'end_cycle': int}}}`."
    prompt += "\n\nCall `submit_model` with your python code when finished."
    
    return prompt

def build_model_via_agent(
    solution_dir: str | Path,
    workspace_dir: str | Path,
    api_key: Optional[str] = None
) -> Any:
    """
    Orchestrates an agentic loop with OpenAI API (via OpenRouter) to build the zero-cosim model.
    """
    if not api_key:
        raise ValueError("An API key (e.g., OPENROUTER_API_KEY) is required to run the Auto-Agent.")
        
    solution_dir_path = Path(solution_dir).resolve()
    workspace_dir_path = Path(workspace_dir).resolve()
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List files and subdirectories in a directory path. Always append the returned path to the solution_dir to know the full path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute or relative path to list"}
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file. By default reads the first 40000 characters. Use offset and length for chunked reading.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute or relative path to the file"},
                        "offset": {"type": "integer", "description": "Starting character index (default 0)."},
                        "length": {"type": "integer", "description": "Number of characters to read (default 40000)."}
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "submit_model",
                "description": "Submit the completed python code for the ZeroCosimModel. This stops the agentic loop.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "python_code": {
                            "type": "string", 
                            "description": "The fully implemented ZeroCosimModel python code."
                        }
                    },
                    "required": ["python_code"]
                }
            }
        }
    ]
    
    sys_prompt = build_system_prompt(workspace_dir_path)
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": f"Please design the ZeroCosimModel for the Vitis project located at: {solution_dir_path}\n"
                       f"Explore the directory, analyze the .verbose.sched.rpt files for human-readable FSM tracking (and .adb files if needed), and submit your model."
        }
    ]
    
    final_python_code = None
    
    print("Agent started using OpenRouter APIs. Exploring directory...")
    while True:
        import time
        max_retries = 5
        response = None
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="google/gemini-3-flash-preview",
                    messages=messages,
                    tools=tools,
                    max_tokens=8192,
                    extra_body={"include_reasoning": True}
                )
            except Exception as e:
                print(f"API Error: {e}")
                
            if response and getattr(response, "choices", None):
                break
                
            print(f"API returned an empty/malformed response on attempt {attempt+1}/{max_retries}")
            if hasattr(response, 'error'):
                print(f"Error payload: {response.error}")
            if attempt < max_retries - 1:
                time.sleep(3)
        else:
            print("Max retries reached. Aborting.")
            break
            
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        
        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except Exception:
                    tool_args = {}
                tool_id = tool_call.id
                
                result = ""
                if tool_name == "list_directory":
                    p = Path(tool_args.get("path", "."))
                    if not p.is_absolute():
                        p = solution_dir_path / p
                    try:
                        items = os.listdir(p)
                        result = f"Items in {p}:\n" + "\n".join(items)
                    except Exception as e:
                        result = f"Error: {e}"
                        
                elif tool_name == "read_file":
                    p = Path(tool_args.get("path", ""))
                    offset = tool_args.get("offset", 0)
                    length = tool_args.get("length", 40000)
                    if not p.is_absolute():
                        p = solution_dir_path / p
                    try:
                        with open(p, 'r', encoding='utf-8') as f:
                            f.seek(offset)
                            content = f.read(length)
                            total_size = os.path.getsize(p)
                            result = f"--- File Chunk (offset={offset}, length={len(content)}, total_file_size={total_size}) ---\n{content}\n--- End Chunk ---"
                    except Exception as e:
                        result = f"Error: {e}"
                        
                elif tool_name == "submit_model":
                    final_python_code = tool_args.get("python_code", "")
                    result = "Successfully submitted model."
                    
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": result
                })
                
                if tool_name == "submit_model":
                    break
        
        if final_python_code:
            break
            
        if not message.tool_calls:
            text_response = message.content or ""
            print(f"Model responded without tools: {text_response}...")
            
            # Optionally we can try to prompt it again, but for now we break.
            if "class ZeroCosimModel" in text_response:
                print("Model wrote class in raw text instead of using submit_model. You might want to parse it.")
                
            print("Finished without model submission.")
            break
            
    if not final_python_code:
        raise RuntimeError("Agent finished without submitting the model code.")
        
    out_file = solution_dir_path / "zero_cosim_model_impl.py"
    out_file.write_text(final_python_code)
    print(f"Generated python model saved to {out_file}")
    
    cls = load_zero_cosim_class(out_file, class_name="ZeroCosimModel")
    return cls

def estimate_customization(
    baseline_model_code: str,
    customizations: list[dict],
    mlir_ir: Optional[str] = None,
    api_key: Optional[str] = None
) -> Any:
    """
    Given an existing Python model and a list of requested customizations,
    returns an updated Model class representing the predicted performance point.
    """
    if not api_key:
        raise ValueError("An API key is required to run the Auto-Agent.")
        
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "submit_model",
                "description": "Submit the completed python code for the updated ZeroCosimModel.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "python_code": {
                            "type": "string", 
                            "description": "The fully implemented ZeroCosimModel python code."
                        }
                    },
                    "required": ["python_code"]
                }
            }
        }
    ]
    
    sys_prompt = (
        "You are an expert Hardware Performance LLM Agent using Zero-Cosim analytical models. "
        "Your task is to modify a provided ZeroCosimModel Python script mathematically to predict "
        "cycle count changes after a user modifies an HLS parameter (e.g. #pragma HLS pipeline). "
        "Rewrite the class logic using your understanding of HLS scheduling and call `submit_model` "
        "with the new python file.\n\n"
        "CRITICAL — PRESERVE ALGEBRAIC PARAMETERS: If the baseline model contains parameterizable "
        "variables representing bounds (e.g. `N`, `M`, `TSTEPS`), preserve these as algebraic "
        "parameters in your cycle math.\n\n"
        "CRITICAL — TOPOLOGY FROM MLIR:\n"
        "1. The MLIR is authoritative for compute topology. The LAST `func.func` in the MLIR module "
        "is the TOP function of THIS schedule — that is what you must model. Earlier functions are "
        "callees.\n"
        "2. Find every `affine.for` in the TOP function. If a loop wraps `func.call` ops, it is an "
        "ORCHESTRATION loop (e.g. a TSTEPS time-step loop). Its trip count MULTIPLIES the summed "
        "latency of the calls inside it. Never drop such an outer loop.\n"
        "3. A loop attribute `pipeline_ii = K` marks a pipelined loop. The K is the REQUESTED II. "
        "Its latency is `outer_trip * achieved_II + pipeline_depth`. The INNER loops inside a "
        "pipelined loop are spatially unrolled and DO NOT multiply the latency — do NOT use "
        "`outer*inner*II`.\n"
        "3a. ⚠️ REDUCTION LOOPS — achieved II ≠ requested II. If the pipelined loop body has a "
        "loop-carried dependency (e.g. `acc[i,j] += a[i,k] * b[k,j]`, `s = s + x[k]`, any `+=` or "
        "`*=` on the same memory cell across iterations), the achieved II is the latency of the "
        "dependency operator, NOT 1. For float32 the typical achieved II is: fadd≈4, fmul≈4, "
        "mac/muladd≈4, fdiv≈10-15, fsqrt≈15-30. Integer add: II=1. CONCRETE EXAMPLE: pipelining "
        "a `for k in reduction(K): out += A[i,k]*B[k,j]` gives achieved_II = 4 (fadd-bound) on "
        "float32. Use `K * 4 + depth`, NOT `K * 1 + depth`. The MLIR's `pipeline_ii = 1` attribute "
        "is what was REQUESTED but Vitis HLS will not honor it under a loop-carried dep — model "
        "the achieved II instead.\n"
        "3b. `buffer_at(target, axis=outer_iv)` introduces SIBLING init and writeback loops "
        "surrounding the inner compute, each pipelined II=1 with trip = size of inner dim. "
        "Per outer iteration: `init_loop + inner_compute + back_loop`, multiplied by outer_trip. "
        "Detect via a `memref.alloc {name = \"...\"}` placed inside the outer affine.for.\n"
        "4. AXI memory bursts (`load_buf*`, `store_res*`) are inferred by Vitis HLS for the TOP "
        "function's memref arguments — and ONLY at the top level. They do NOT appear in the MLIR "
        "but DO appear in the baseline. ⚠️ Sub-function calls (`func.call @mm1(...)`, etc.) receive "
        "memref arguments that are ALREADY-LOADED on-chip buffers — sub-functions do NOT trigger "
        "their own AXI loads/stores. Do NOT inline burst stages inside each sub-kernel's latency. "
        "Burst stages are SIBLINGS of the sub-kernel calls at the TOP function level. Concretely, "
        "for a top function `kernel_gemm(A, B, C, output)` with 3 inputs + 1 output, the schedule is:\n"
        "    load_A | load_B | load_C | <fill on-chip buffers> | mm1(A_buf, B_buf, out_AB) | ele_add(out_AB, C, output) | store_output\n"
        "NOT:\n"
        "    mm1 (containing its own load_A, load_B, store_out_AB) | ele_add (containing its own load_C, store_output)\n"
        "Use one `load_buf` per memref read input and one `store_res` per memref written output "
        "OF THE TOP FUNCTION IN THE MLIR. If the MLIR top function is a SUB-FUNCTION (e.g. the user "
        "customized `compute_A` directly, not the parent kernel), do NOT carry over the parent "
        "kernel's burst stages — use only the burst stages corresponding to this sub-function's own "
        "memref args.\n"
        "5. Memory burst stages are SIBLINGS of the compute loop, executed sequentially around it. "
        "Pipelining the compute loop does NOT fuse them into the compute stage.\n\n"
        "When in doubt, consult the `pipeline_fusing` skill below.\n\n"
    )
    workspace_dir = Path(__file__).resolve().parent.parent.parent.parent
    sys_prompt += _load_skills(workspace_dir)
    user_content = (
        f"Here is the baseline ZeroCosimModel source code:\n```python\n{baseline_model_code}\n```\n\n"
        f"The user intends to apply the following customizations:\n{json.dumps(customizations, indent=2)}\n\n"
    )
    if mlir_ir:
        mlir_ir_str = _summarize_mlir_for_prompt(mlir_ir)
        user_content += (
            f"To assist you, here is the exact MLIR Abstract Syntax Tree of the hardware schedule AFTER the optimizations are applied:\n"
            f"```mlir\n{mlir_ir_str}\n```\n\n"
        )
    user_content += "Please mathematically update the python code to emulate the physical effect of these optimizations. Submit the updated python class."
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content}
    ]
    
    final_python_code = None
    print("Estimating customization effects...")
    
    while True:
        import time
        max_retries = 5
        response = None
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="google/gemini-3-flash-preview",
                    messages=messages,
                    tools=tools,
                    max_tokens=8192,
                    extra_body={"include_reasoning": True}
                )
            except Exception as e:
                print(f"API Error: {e}")
                
            if response and getattr(response, "choices", None):
                break
                
            print(f"API returned an empty/malformed response on attempt {attempt+1}/{max_retries}")
            if hasattr(response, 'error'):
                print(f"Error payload: {response.error}")
            if attempt < max_retries - 1:
                time.sleep(3)
        else:
            print("Max retries reached. Aborting.")
            break
            
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        
        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except Exception:
                    tool_args = {}
                
                if tool_name == "submit_model":
                    final_python_code = tool_args.get("python_code", "")
                    
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": "Successfully submitted model."
                })
                
                if tool_name == "submit_model":
                    break
                    
        if final_python_code:
            break
            
        if not message.tool_calls:
            text_response = message.content or ""
            print(f"Model responded without tools: {text_response[:500]}...")
            if "class ZeroCosimModel" in text_response:
                print("Model wrote class in raw text instead of using submit_model.")
            break
            
    if not final_python_code:
        raise RuntimeError("Agent finished without submitting the updated model code.")
        
    out_file = Path("/tmp/zero_cosim_model_updated.py")
    out_file.write_text(final_python_code)
    print(f"Generated predicted model saved to {out_file}")
    
    cls = load_zero_cosim_class(out_file, class_name="ZeroCosimModel")
    return cls


def _find_claude_cli() -> Optional[str]:
    """Find the claude CLI executable."""
    # Check common locations
    for candidate in [
        shutil.which("claude"),
        os.path.expanduser("~/.local/bin/claude"),
        "/usr/local/bin/claude",
    ]:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def build_model_via_claude_cli(
    solution_dir: str | Path,
    workspace_dir: str | Path,
) -> Any:
    """
    Uses the Claude CLI (claude --print) to build the zero-cosim model.
    Claude CLI has native file system access so it can read HLS artifacts directly.
    """
    claude_bin = _find_claude_cli()
    if not claude_bin:
        raise RuntimeError("Claude CLI not found. Install via: curl -fsSL https://claude.ai/install.sh | bash")

    solution_dir_path = Path(solution_dir).resolve()
    workspace_dir_path = Path(workspace_dir).resolve()

    sys_prompt = build_system_prompt(workspace_dir_path)
    # Remove the submit_model tool instruction since CLI doesn't use tools the same way
    sys_prompt = sys_prompt.replace(
        "\nCall `submit_model` with your python code when finished.",
        "\nOutput ONLY the complete Python file inside a single ```python ... ``` fenced code block. No other text outside the code block."
    )

    user_prompt = (
        f"Please design the ZeroCosimModel for the Vitis HLS project located at: {solution_dir_path}\n"
        f"Explore the directory, analyze the .verbose.sched.rpt files and csynth XML reports, then output the complete Python class.\n"
        f"The class MUST be named ZeroCosimModel with constructor `def __init__(self, solution_dir, *, clock_period_ns=3.33):` "
        f"and a `def report_cycle(self, **kwargs):` method returning {{'makespan_cycles': int, 'kernels': dict}}.\n"
        f"Output ONLY the complete Python file inside a single ```python ... ``` fenced code block."
    )

    full_prompt = f"{sys_prompt}\n\n{user_prompt}"

    print("Agent started using Claude CLI. Exploring directory...")
    try:
        result = subprocess.run(
            [
                claude_bin, "--print", full_prompt,
                "--output-format", "text",
                "--dangerously-skip-permissions",
                "--max-budget-usd", "5",
            ],
            capture_output=True,
            text=True,
            timeout=900,
            cwd=str(solution_dir_path),
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Claude CLI timed out after 900 seconds.")

    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI failed (exit {result.returncode}): {result.stderr[:500]}")

    output = result.stdout
    python_code = _extract_python_from_text(output)

    if not python_code:
        # Save the raw output for debugging
        debug_file = Path("/tmp/claude_cli_raw_output.txt")
        debug_file.write_text(output)
        raise RuntimeError(
            f"Could not extract ZeroCosimModel from Claude CLI output. "
            f"Raw output saved to {debug_file}"
        )

    out_file = solution_dir_path / "zero_cosim_model_impl.py"
    out_file.write_text(python_code)
    print(f"Generated python model saved to {out_file}")

    cls = load_zero_cosim_class(out_file, class_name="ZeroCosimModel")
    return cls


def estimate_customization_via_claude_cli(
    baseline_model_code: str,
    customizations: list[dict],
    mlir_ir: Optional[str] = None,
) -> Any:
    """
    Uses Claude CLI to predict the performance impact of customizations.
    """
    claude_bin = _find_claude_cli()
    if not claude_bin:
        raise RuntimeError("Claude CLI not found. Install via: curl -fsSL https://claude.ai/install.sh | bash")

    workspace_dir = Path(__file__).resolve().parent.parent.parent.parent
    skills_text = _load_skills(workspace_dir)

    sys_prompt = (
        "You are an expert Hardware Performance LLM Agent using Zero-Cosim analytical models. "
        "Your task is to modify a provided ZeroCosimModel Python script mathematically to predict "
        "cycle count changes after a user modifies an HLS parameter (e.g. #pragma HLS pipeline).\n\n"
        "CRITICAL — PRESERVE ALGEBRAIC PARAMETERS: If the baseline model contains parameterizable "
        "variables representing bounds (e.g. `N`, `M`, `TSTEPS`), preserve these as algebraic "
        "parameters in your cycle math.\n\n"
        "CRITICAL — TOPOLOGY FROM MLIR:\n"
        "1. The MLIR is authoritative for compute topology. The LAST `func.func` in the MLIR module "
        "is the TOP function of THIS schedule — that is what you must model. Earlier functions are "
        "callees.\n"
        "2. Find every `affine.for` in the TOP function. If a loop wraps `func.call` ops, it is an "
        "ORCHESTRATION loop (e.g. a TSTEPS time-step loop). Its trip count MULTIPLIES the summed "
        "latency of the calls inside it. Never drop such an outer loop.\n"
        "3. A loop attribute `pipeline_ii = K` marks a pipelined loop. The K is the REQUESTED II. "
        "Its latency is `outer_trip * achieved_II + pipeline_depth`. The INNER loops inside a "
        "pipelined loop are spatially unrolled and DO NOT multiply the latency — do NOT use "
        "`outer*inner*II`.\n"
        "3a. ⚠️ REDUCTION LOOPS — achieved II ≠ requested II. If the pipelined loop body has a "
        "loop-carried dependency (e.g. `acc[i,j] += a[i,k] * b[k,j]`, `s = s + x[k]`, any `+=` or "
        "`*=` on the same memory cell across iterations), the achieved II is the latency of the "
        "dependency operator, NOT 1. For float32 the typical achieved II is: fadd≈4, fmul≈4, "
        "mac/muladd≈4, fdiv≈10-15, fsqrt≈15-30. Integer add: II=1. CONCRETE EXAMPLE: pipelining "
        "a `for k in reduction(K): out += A[i,k]*B[k,j]` gives achieved_II = 4 (fadd-bound) on "
        "float32. Use `K * 4 + depth`, NOT `K * 1 + depth`. The MLIR's `pipeline_ii = 1` attribute "
        "is what was REQUESTED but Vitis HLS will not honor it under a loop-carried dep — model "
        "the achieved II instead.\n"
        "3b. `buffer_at(target, axis=outer_iv)` introduces SIBLING init and writeback loops "
        "surrounding the inner compute, each pipelined II=1 with trip = size of inner dim. "
        "Per outer iteration: `init_loop + inner_compute + back_loop`, multiplied by outer_trip. "
        "Detect via a `memref.alloc {name = \"...\"}` placed inside the outer affine.for.\n"
        "4. AXI memory bursts (`load_buf*`, `store_res*`) are inferred by Vitis HLS for the TOP "
        "function's memref arguments — and ONLY at the top level. They do NOT appear in the MLIR "
        "but DO appear in the baseline. ⚠️ Sub-function calls (`func.call @mm1(...)`, etc.) receive "
        "memref arguments that are ALREADY-LOADED on-chip buffers — sub-functions do NOT trigger "
        "their own AXI loads/stores. Do NOT inline burst stages inside each sub-kernel's latency. "
        "Burst stages are SIBLINGS of the sub-kernel calls at the TOP function level. Concretely, "
        "for a top function `kernel_gemm(A, B, C, output)` with 3 inputs + 1 output, the schedule is:\n"
        "    load_A | load_B | load_C | <fill on-chip buffers> | mm1(A_buf, B_buf, out_AB) | ele_add(out_AB, C, output) | store_output\n"
        "NOT:\n"
        "    mm1 (containing its own load_A, load_B, store_out_AB) | ele_add (containing its own load_C, store_output)\n"
        "Use one `load_buf` per memref read input and one `store_res` per memref written output "
        "OF THE TOP FUNCTION IN THE MLIR. If the MLIR top function is a SUB-FUNCTION (e.g. the user "
        "customized `compute_A` directly, not the parent kernel), do NOT carry over the parent "
        "kernel's burst stages — use only the burst stages corresponding to this sub-function's own "
        "memref args.\n"
        "5. Memory burst stages are SIBLINGS of the compute loop, executed sequentially around it. "
        "Pipelining the compute loop does NOT fuse them into the compute stage.\n\n"
        "When in doubt, consult the `pipeline_fusing` skill below.\n\n"
    )
    sys_prompt += skills_text

    user_content = (
        f"Here is the baseline ZeroCosimModel source code:\n```python\n{baseline_model_code}\n```\n\n"
        f"The user intends to apply the following customizations:\n{json.dumps(customizations, indent=2)}\n\n"
    )
    if mlir_ir:
        mlir_ir_str = _summarize_mlir_for_prompt(mlir_ir)
        user_content += (
            f"Here is the exact MLIR AST of the hardware schedule AFTER the optimizations are applied:\n"
            f"```mlir\n{mlir_ir_str}\n```\n\n"
        )
    user_content += (
        "Please mathematically update the python code to emulate the physical effect of these optimizations. "
        "Output ONLY the complete updated Python file inside a single ```python ... ``` fenced code block."
    )

    full_prompt = f"{sys_prompt}\n\n{user_content}"

    print("Estimating customization effects via Claude CLI...")
    try:
        result = subprocess.run(
            [
                claude_bin, "--print", full_prompt,
                "--output-format", "text",
                "--dangerously-skip-permissions",
                "--max-budget-usd", "5",
            ],
            capture_output=True,
            text=True,
            timeout=900,
            cwd=str(Path.cwd()),
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Claude CLI timed out after 900 seconds.")

    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI failed (exit {result.returncode}): {result.stderr[:500]}")

    output = result.stdout
    python_code = _extract_python_from_text(output)

    if not python_code:
        debug_file = Path("/tmp/claude_cli_raw_output.txt")
        debug_file.write_text(output)
        raise RuntimeError(
            f"Could not extract ZeroCosimModel from Claude CLI output. "
            f"Raw output saved to {debug_file}"
        )

    out_file = Path("/tmp/zero_cosim_model_updated.py")
    out_file.write_text(python_code)
    print(f"Generated predicted model saved to {out_file}")

    cls = load_zero_cosim_class(out_file, class_name="ZeroCosimModel")
    return cls
