"""
Executor — Task execution with ReAct pattern.

The executor follows Think → Act → Observe loop, with the ability
to pause and request new tools when existing ones are insufficient.

Lightweight implementation: single LLM call per step (vs Yunjue's multi-agent).
"""

import json
import logging
from typing import Any, Optional

from ..core.agent import MirrorAgent, Tool

logger = logging.getLogger("mirror.executor")

REACT_SYSTEM_PROMPT = """You are Mirror, a personal AI agent. You have access to tools and can think step by step.

Response format (JSON):
{
    "thought": "What you're thinking about this step",
    "action": "tool_name" | "respond" | "request_tool",
    "action_input": {...} | "your response text",
    "tool_request": "description of the tool you need"  // only when action=request_tool
}

Available tools:
{tools_description}

User preferences you know about:
{preferences}

Current task: {task}
"""


async def execute_task(
    agent: MirrorAgent,
    task: str,
    llm_call: callable,
    max_steps: int = 10,
) -> dict[str, Any]:
    """
    Execute a user task using ReAct loop with tool augmentation.

    Returns:
        {"success": bool, "result": str, "steps": int, "tools_used": [...], "tools_created": int}
    """
    steps = []
    tools_created = 0
    context = ""  # Accumulated context from previous steps

    for step in range(max_steps):
        # Find relevant tools
        relevant_tools = agent.find_tools(task + " " + context)
        tools_desc = _format_tools(relevant_tools)

        # Build prompt
        prompt = REACT_SYSTEM_PROMPT.format(
            tools_description=tools_desc or "(no tools yet — use request_tool if needed)",
            preferences=json.dumps(agent.state.preferences, ensure_ascii=False),
            task=task,
        )

        if context:
            prompt += f"\n\nPrevious steps:\n{context}"

        # LLM step
        try:
            response = llm_call(prompt)
            action = json.loads(response)
        except json.JSONDecodeError:
            action = {"thought": "Parsing error", "action": "respond",
                      "action_input": response}

        steps.append(action)

        if action.get("action") == "respond":
            return {
                "success": True,
                "result": action.get("action_input", ""),
                "steps": step + 1,
                "tools_used": _extract_tool_names(steps),
                "tools_created": tools_created,
            }

        elif action.get("action") == "request_tool":
            # Need a new tool — Tool Developer will handle this
            return {
                "success": False,
                "result": None,
                "need_tool": True,
                "tool_description": action.get("tool_request", ""),
                "steps": step + 1,
                "tools_used": _extract_tool_names(steps),
                "tools_created": tools_created,
            }

        else:
            # Execute tool
            tool_name = action.get("action", "")
            tool_input = action.get("action_input", {})
            tool = _find_tool_by_name(relevant_tools, tool_name)

            if tool:
                from ..core.tool_developer import sandbox_exec
                result = sandbox_exec(tool.code, tool_input)
                agent.record_tool_result(tool_name, result["success"])
                context += f"\nStep {step+1}: {tool_name}({tool_input}) → {result}"
            else:
                context += f"\nStep {step+1}: Tool '{tool_name}' not found."

    # Max steps reached
    return {
        "success": False,
        "result": "Task exceeded maximum steps without completion.",
        "steps": max_steps,
        "tools_used": _extract_tool_names(steps),
        "tools_created": tools_created,
    }


def _format_tools(tools: list[Tool]) -> str:
    if not tools:
        return ""
    lines = []
    for t in tools:
        sig = ", ".join(f"{k}: {v}" for k, v in t.signature.items())
        lines.append(f"- {t.name}({sig}): {t.description} (success rate: {t.success_rate:.0%})")
    return "\n".join(lines)


def _find_tool_by_name(tools: list[Tool], name: str) -> Optional[Tool]:
    for t in tools:
        if t.name == name or t.name.startswith(name.split("_")[0]):
            return t
    return None


def _extract_tool_names(steps: list[dict]) -> list[str]:
    names = []
    for s in steps:
        a = s.get("action", "")
        if a not in ("respond", "request_tool", ""):
            names.append(a)
    return names
