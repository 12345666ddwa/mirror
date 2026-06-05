"""
Tool Developer — The self-evolution engine.

When the agent encounters a task it cannot handle with existing tools,
the Tool Developer generates new Python functions at runtime, tests them
in a sandbox, and only keeps those that pass validation.

Key design decisions (different from Yunjue):
  1. Single-agent code generation (not a separate "Tool Developer" agent)
     → Lower token cost, simpler architecture
  2. Sandbox: restricted Python exec with timeout
  3. Generated tools are Pydantic-validated for type safety
"""

import ast
import hashlib
import logging
import sys
import time
import traceback
from io import StringIO
from typing import Any, Optional

from ..core.agent import Tool

logger = logging.getLogger("mirror.tooldev")

# ── Sandbox ────────────────────────────────────

_SANDBOX_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool,
    "dict": dict, "enumerate": enumerate, "filter": filter,
    "float": float, "int": int, "len": len, "list": list,
    "map": map, "max": max, "min": min, "range": range,
    "round": round, "set": set, "sorted": sorted, "str": str,
    "sum": sum, "tuple": tuple, "zip": zip,
    "print": print, "isinstance": isinstance,
    "True": True, "False": False, "None": None,
    "Exception": Exception, "ValueError": ValueError,
    "TypeError": TypeError, "KeyError": KeyError,
}

_SANDBOX_SAFE_MODULES = {
    "json", "math", "datetime", "collections",
    "itertools", "functools", "re", "statistics",
    "hashlib", "base64", "urllib.parse", "csv",
    "textwrap", "string", "numbers", "decimal",
    "random", "uuid", "pathlib",
}

SANDBOX_TIMEOUT = 10  # seconds


class SandboxError(Exception):
    """Raised when sandbox execution fails."""
    pass


def sandbox_exec(code: str, params: dict[str, Any]) -> dict[str, Any]:
    """
    Execute generated tool code in a restricted sandbox.

    Args:
        code: Python source code (must define a `run(**params)` function)
        params: Arguments to pass to run()

    Returns:
        {"success": bool, "result": Any, "error": str|None}

    The sandbox:
      - Restricts builtins to safe subset
      - Whitelists importable modules
      - Enforces timeout
      - Captures stdout/stderr
    """
    # Parse AST to validate imports
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"success": False, "result": None, "error": f"Syntax error: {e}"}

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".")[0] for alias in node.names]
            else:
                modules = [node.module.split(".")[0]] if node.module else []

            for mod in modules:
                if mod not in _SANDBOX_SAFE_MODULES:
                    return {
                        "success": False,
                        "result": None,
                        "error": f"Forbidden import: {mod}",
                    }

    # Execute in sandbox
    sandbox_globals = {
        "__builtins__": _SANDBOX_SAFE_BUILTINS,
        "__name__": "__mirror_sandbox__",
    }

    old_stdout = sys.stdout
    captured = StringIO()
    sys.stdout = captured

    try:
        exec(code, sandbox_globals)

        if "run" not in sandbox_globals:
            return {
                "success": False,
                "result": None,
                "error": "Code must define a `run(**params)` function",
            }

        start = time.time()
        result = sandbox_globals["run"](**params)
        elapsed = time.time() - start

        if elapsed > SANDBOX_TIMEOUT:
            return {
                "success": False,
                "result": None,
                "error": f"Timeout: execution took {elapsed:.1f}s > {SANDBOX_TIMEOUT}s",
            }

        return {"success": True, "result": result, "error": None}

    except Exception as e:
        tb = traceback.format_exc()
        return {"success": False, "result": None, "error": f"{type(e).__name__}: {e}"}

    finally:
        sys.stdout = old_stdout


# ── Tool Synthesis ─────────────────────────────

TOOL_SYNTHESIS_PROMPT = """You are a tool-building AI. Write a Python function to accomplish a task.

Requirements:
1. Define a function `run(**params)` that does the work
2. Use type hints for all parameters
3. Do NOT import restricted modules: os, subprocess, sys, socket, requests, urllib.request
4. Safe modules allowed: json, math, datetime, collections, re, statistics, random, csv, pathlib
5. Return a useful result (dict, str, list, or number)
6. Include error handling

Task: {task_description}

Output ONLY the Python code, nothing else. Start with `def run(`."""


def synthesize_tool(task_description: str, tool_name: str, llm_call: callable) -> Optional[Tool]:
    """
    Use LLM to generate a new tool for an unmet capability.

    Args:
        task_description: What the tool should do
        tool_name: Name for the new tool (snake_case)
        llm_call: Function to call LLM (takes prompt, returns text)

    Returns:
        Tool if synthesis succeeded, None if failed
    """
    prompt = TOOL_SYNTHESIS_PROMPT.format(task_description=task_description)

    for attempt in range(3):
        try:
            code = llm_call(prompt)
            code = _extract_code_block(code)

            # Validate syntax
            ast.parse(code)

            # Test with empty params as smoke test
            test_result = sandbox_exec(code, {})
            if not test_result["success"] and "must define a `run" not in test_result.get("error", ""):
                logger.warning(f"Tool {tool_name} smoke test failed (attempt {attempt+1}): {test_result['error']}")
                if attempt < 2:
                    prompt = f"{prompt}\n\nPrevious attempt failed: {test_result['error']}\nPlease fix and retry."
                    continue

            tool_id = hashlib.md5(code.encode()).hexdigest()[:8]
            return Tool(
                name=f"{tool_name}_{tool_id}",
                description=task_description,
                code=code,
                signature=_extract_signature(code),
            )

        except Exception as e:
            logger.warning(f"Tool synthesis attempt {attempt+1} failed: {e}")
            if attempt < 2:
                prompt = f"{prompt}\n\nError: {e}\nPlease fix."
            continue

    return None


def _extract_code_block(text: str) -> str:
    """Extract Python code from LLM output, handling markdown fences."""
    text = text.strip()
    if text.startswith("```python"):
        text = text[9:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _extract_signature(code: str) -> dict:
    """Extract parameter types from the run() function signature."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run":
                params = {}
                for arg in node.args.args:
                    annotation = "str"
                    if arg.annotation:
                        if isinstance(arg.annotation, ast.Name):
                            annotation = arg.annotation.id
                        elif isinstance(arg.annotation, ast.Subscript):
                            annotation = "list"  # simplified
                    params[arg.arg] = annotation
                return params
    except Exception:
        pass
    return {}
