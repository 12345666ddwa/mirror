"""
Agent Loop — The full Mirror conversation engine.

Connects LLM + Tool Developer + Executor + Memory into one run loop.
"""

import json
import logging
from typing import Optional

from .agent import MirrorAgent
from .llm import LLMClient, create_client
from .tool_developer import synthesize_tool
from ..memory.memory import extract_memory_updates

logger = logging.getLogger("mirror.loop")

SYSTEM_PROMPT = """你是赛镜 Mirror，一个自进化的个人AI Agent。你的目标是越来越了解用户、越来越有用。

核心能力：
1. 你有工具库，完成任务时优先使用现有工具
2. 如果现有工具不够用，你可以请求创造新工具
3. 你会从每次对话中学习用户的偏好和习惯

回复格式：
- 正常对话：直接回复用户
- 需要工具：以 [USE_TOOL:工具描述] 开头，描述你需要什么工具
- 对话结束可以顺便记住用户的新偏好

用户偏好（已学到的）：
{preferences}

可用工具：
{tools}"""

TOOL_SYNTHESIS_OK = """你刚才请求的工具已经创建好了。现在请直接使用它完成任务。

工具名: {tool_name}
工具描述: {tool_desc}

请直接回复用户，在回复中使用这个工具的结果。"""


class MirrorLoop:
    """Full conversation loop with LLM, tool synthesis, and memory."""

    def __init__(
        self,
        agent: MirrorAgent,
        llm: LLMClient,
    ):
        self.agent = agent
        self.llm = llm

    def chat(self, user_message: str) -> str:
        """Process one user message and return response."""

        # Build context
        tools_desc = self._describe_tools()
        prefs = json.dumps(self.agent.state.preferences, ensure_ascii=False, indent=2)

        system = SYSTEM_PROMPT.format(
            preferences=prefs if self.agent.state.preferences else "（刚开始了解你，还没有偏好记录）",
            tools=tools_desc if self.agent.state.tools else "（暂无工具，遇到需要工具的请求请用 [USE_TOOL:...] 标记）",
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]

        response = self.llm.chat(messages)

        # Check if agent requests a new tool
        if "[USE_TOOL:" in response:
            return self._handle_tool_request(response, user_message, messages, system)

        # Learn from this interaction
        self._learn(user_message, response)

        self.agent.state.interaction_count += 1
        return response

    def _handle_tool_request(self, response: str, user_message: str, history: list, system: str) -> str:
        """Agent needs a new tool — synthesize, test, retry."""

        # Extract tool description
        import re
        match = re.search(r'\[USE_TOOL:(.*?)\]', response, re.DOTALL)
        if not match:
            return "（工具请求解析失败，请重试）"

        tool_desc = match.group(1).strip()
        tool_name = tool_desc[:30].lower().replace(" ", "_").replace("/", "_")

        logger.info(f"Agent requested tool: {tool_name} — {tool_desc[:80]}")

        # Synthesize the tool
        tool = synthesize_tool(
            task_description=tool_desc,
            tool_name=f"user_{tool_name}",
            llm_call=self.llm.generate,
        )

        if tool:
            self.agent.register_tool(tool)
            logger.info(f"Tool synthesized and registered: {tool.name}")

            # Now retry with the new tool available
            tools_desc = self._describe_tools()
            system_with_tool = SYSTEM_PROMPT.format(
                preferences=json.dumps(self.agent.state.preferences, ensure_ascii=False, indent=2),
                tools=tools_desc,
            )
            followup = TOOL_SYNTHESIS_OK.format(
                tool_name=tool.name,
                tool_desc=tool_desc,
            )

            retry_messages = [
                {"role": "system", "content": system_with_tool},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": response},
                {"role": "user", "content": followup},
            ]

            final = self.llm.chat(retry_messages)
            self._learn(user_message, final)
            self.agent.state.interaction_count += 1
            return final
        else:
            return "我尝试创建一个新工具来完成你的请求，但失败了。这可能是一个超出我能力范围的任务。"

    def _learn(self, user_message: str, response: str):
        """Extract new knowledge from this interaction."""
        updates = extract_memory_updates(user_message, response, self.llm.generate)
        for k, v in updates.get("preferences", {}).items():
            self.agent.update_preference(k, v)

    def _describe_tools(self) -> str:
        if not self.agent.state.tools:
            return "（尚无工具）"
        lines = []
        for t in self.agent.state.tools[:20]:
            sig = ", ".join(f"{k}: {v}" for k, v in t.signature.items()) or "无参数"
            lines.append(f"• {t.name}({sig}) — {t.description[:60]} (成功率{t.success_rate:.0%})")
        return "\n".join(lines)


# ── Convenience ────────────────────────────────

def create_mirror(
    backend: str = "openai",
    model: str = "",
    api_key: str = "",
) -> MirrorLoop:
    """Create a ready-to-use Mirror agent."""
    agent = MirrorAgent()
    agent.load_state()
    llm = create_client(backend=backend, model=model, api_key=api_key)
    return MirrorLoop(agent=agent, llm=llm)
