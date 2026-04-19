import os
import glob
import json
from pathlib import Path
from typing import Any, Optional
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .loader import load_zero_cosim_class

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
                       f"Explore the directory, analyze the ADB files, and submit your model."
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
    
    sys_prompt = "You are an expert Hardware Performance LLM Agent using Zero-Cosim analytical models. Your task is to modify a provided ZeroCosimModel Python script mathematically to predict cycle count changes after a user modifies an HLS parameter (e.g. #pragma HLS pipeline). Do not ask to read files. Rewrite the class logic using your understanding of HLS scheduling and call `submit_model` with the new python file. CRITICAL: If the baseline model contains parameterizable variables representing bounds (e.g. `N` or `M` handled via kwargs), you must inherently preserve these algebraic parameters within your cycle math modifications!\n\nWARNING: Do NOT blindly preserve the baseline timeline sequence IF a loop is pipelined! If the user applies an outer-loop pipeline directly in the current schedule, Vitis HLS flattens the graph. For pipelined loops, you MUST delete the sequentially ordered sequence (e.g. `load_buf`, `compute`, `store`) from the `report_cycle` method, replacing it with a single fused block representing the pipelined bottleneck.\nHOWEVER: If the current schedule's outer loops are UNPIPELINED (e.g., a top-level composed kernel where only the inner-called sub-functions are pipelined), you MUST RETAIN the sequential array and mathematically SUM the sequential pieces (like hoisted memory loads and stores). Do not delete the sequential logic if the current loop layer lacks a pipeline pragma!\n\n"
    workspace_dir = Path(__file__).resolve().parent.parent.parent.parent
    sys_prompt += _load_skills(workspace_dir)
    messages = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": f"Here is the baseline ZeroCosimModel source code:\n```python\n{baseline_model_code}\n```\n\nThe user intends to apply the following customizations:\n{json.dumps(customizations, indent=2)}\n\nPlease mathematically update the python code to emulate the physical effect of these optimizations. Submit the updated python class."
        }
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
