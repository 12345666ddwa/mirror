"""
Mirror Agent Core — The heart of the self-evolving personal AI.

Architecture:
  Manager → ToolDeveloper → Executor → Integrator
     ↑                                      |
     └── Aggregator ← Merger ←──────────────┘

Inspired by Yunjue Agent's multi-agent design, but:
  - Simplified: single-process instead of multi-agent orchestration
  - Personalized: memory + preference model (not just tools)
  - Local-first: supports Ollama/llama.cpp
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import json
import logging

logger = logging.getLogger("mirror")


@dataclass
class Tool:
    """A self-evolved tool that the agent created at runtime."""
    name: str
    description: str
    code: str  # Python source code
    signature: dict  # Pydantic-style parameter schema
    usage_count: int = 0
    success_rate: float = 1.0
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "code": self.code,
            "signature": self.signature,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Tool":
        return cls(**d)


@dataclass
class AgentState:
    """The evolving state of the Mirror agent."""
    tools: list[Tool] = field(default_factory=list)
    preferences: dict[str, Any] = field(default_factory=dict)
    persona: dict[str, Any] = field(default_factory=dict)
    interaction_count: int = 0
    total_tool_synthesis: int = 0

    @property
    def egl(self) -> float:
        """Evolution Generality Loss — converges to 0 as agent matures.
        EGL = cumulative_synthesized_tools / cumulative_tool_calls
        Lower is better: the agent reuses tools more than it creates new ones.
        """
        total_calls = sum(t.usage_count for t in self.tools)
        if total_calls == 0:
            return float("inf")
        return self.total_tool_synthesis / total_calls


class MirrorAgent:
    """
    The main agent that orchestrates self-evolution.

    Usage:
        agent = MirrorAgent(model="gpt-4o")
        agent.load_state()  # Load persisted tools & memory
        response = await agent.run("分析我上周的睡眠数据")
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        local_model: Optional[str] = None,
        state_dir: str = "~/.mirror",
    ):
        self.model = model
        self.local_model = local_model
        self.state_dir = state_dir
        self.state = AgentState()

    # ── Tool Management ──────────────────────────

    def find_tools(self, query: str, top_k: int = 5) -> list[Tool]:
        """Retrieve relevant tools for a task using semantic similarity."""
        if not self.state.tools:
            return []

        # Simple keyword-based retrieval (upgrade to embeddings later)
        query_words = set(query.lower().split())
        scored = []
        for tool in self.state.tools:
            desc_words = set(tool.description.lower().split())
            overlap = len(query_words & desc_words)
            scored.append((overlap, tool))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:top_k] if _ > 0]

    def register_tool(self, tool: Tool) -> None:
        """Add a newly synthesized tool to the library."""
        # Check if similar tool already exists
        for existing in self.state.tools:
            if existing.name == tool.name:
                existing.code = tool.code
                existing.usage_count += 1
                return

        self.state.tools.append(tool)
        self.state.total_tool_synthesis += 1
        logger.info(f"New tool registered: {tool.name}")

    def record_tool_result(self, tool_name: str, success: bool) -> None:
        """Update tool success rate after execution."""
        for tool in self.state.tools:
            if tool.name == tool_name:
                tool.usage_count += 1
                n = tool.usage_count
                tool.success_rate = (tool.success_rate * (n - 1) + int(success)) / n
                return

    # ── Memory & Persona ─────────────────────────

    def update_preference(self, key: str, value: Any) -> None:
        """Learn a new preference about the user."""
        self.state.preferences[key] = value

    def get_preference(self, key: str, default: Any = None) -> Any:
        return self.state.preferences.get(key, default)

    # ── Persistence ──────────────────────────────

    def save_state(self) -> None:
        """Persist agent state (tools, preferences, persona) to disk."""
        import os
        path = os.path.expanduser(f"{self.state_dir}/state.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            "tools": [t.to_dict() for t in self.state.tools],
            "preferences": self.state.preferences,
            "persona": self.state.persona,
            "interaction_count": self.state.interaction_count,
            "total_tool_synthesis": self.state.total_tool_synthesis,
        }
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_state(self) -> bool:
        """Load persisted agent state. Returns False if no state exists."""
        import os
        path = os.path.expanduser(f"{self.state_dir}/state.json")
        if not os.path.exists(path):
            return False

        with open(path) as f:
            data = json.load(f)

        self.state.tools = [Tool.from_dict(t) for t in data.get("tools", [])]
        self.state.preferences = data.get("preferences", {})
        self.state.persona = data.get("persona", {})
        self.state.interaction_count = data.get("interaction_count", 0)
        self.state.total_tool_synthesis = data.get("total_tool_synthesis", 0)
        return True

    # ── Evolution Stats ──────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "tools_count": len(self.state.tools),
            "total_synthesized": self.state.total_tool_synthesis,
            "interactions": self.state.interaction_count,
            "egl": f"{self.state.egl:.4f}" if self.state.egl != float("inf") else "∞",
            "top_tools": sorted(
                self.state.tools, key=lambda t: t.usage_count, reverse=True
            )[:5],
        }
