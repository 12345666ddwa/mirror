"""
LLM Client — Unified interface for multiple model backends.

Supports: OpenAI, Anthropic, DeepSeek, Ollama (local)
"""

from abc import ABC, abstractmethod
from typing import Optional
import json
import logging
import os

logger = logging.getLogger("mirror.llm")

# ── Abstract Interface ─────────────────────────

class LLMClient(ABC):
    """Abstract LLM client. All backends implement this."""

    def __init__(self, model: str = ""):
        self.model = model

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs) -> str:
        """Send messages, return response text."""
        ...

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Simple prompt → completion."""
        ...


# ── OpenAI ─────────────────────────────────────

class OpenAIClient(LLMClient):
    def __init__(self, model: str = "gpt-4o", api_key: str = ""):
        super().__init__(model)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def chat(self, messages: list[dict], **kwargs) -> str:
        import urllib.request

        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2048),
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        resp = urllib.request.urlopen(req, timeout=60)
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    def generate(self, prompt: str, **kwargs) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)


# ── Anthropic ──────────────────────────────────

class AnthropicClient(LLMClient):
    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str = ""):
        super().__init__(model)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")

    def chat(self, messages: list[dict], **kwargs) -> str:
        import urllib.request

        # Extract system message if present
        system = ""
        msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                msgs.append(m)

        body = json.dumps({
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "system": system,
            "messages": msgs,
        }).encode()

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )

        resp = urllib.request.urlopen(req, timeout=60)
        data = json.loads(resp.read())
        return data["content"][0]["text"]

    def generate(self, prompt: str, **kwargs) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)


# ── DeepSeek ───────────────────────────────────

class DeepSeekClient(LLMClient):
    def __init__(self, model: str = "deepseek-chat", api_key: str = ""):
        super().__init__(model)
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = "https://api.deepseek.com/v1"

    def chat(self, messages: list[dict], **kwargs) -> str:
        import urllib.request

        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2048),
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        resp = urllib.request.urlopen(req, timeout=60)
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    def generate(self, prompt: str, **kwargs) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)


# ── Ollama (Local) ─────────────────────────────

class OllamaClient(LLMClient):
    def __init__(self, model: str = "llama3", host: str = ""):
        super().__init__(model)
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def chat(self, messages: list[dict], **kwargs) -> str:
        import urllib.request

        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
            },
        }).encode()

        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )

        resp = urllib.request.urlopen(req, timeout=120)
        data = json.loads(resp.read())
        return data["message"]["content"]

    def generate(self, prompt: str, **kwargs) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    @staticmethod
    def list_models() -> list[str]:
        """List available local models."""
        import urllib.request
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        req = urllib.request.Request(f"{host}/api/tags")
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        return [m["name"] for m in data.get("models", [])]


# ── Factory ────────────────────────────────────

def create_client(
    backend: str = "openai",
    model: str = "",
    api_key: str = "",
) -> LLMClient:
    """
    Create an LLM client from config.

    backend: "openai" | "anthropic" | "deepseek" | "ollama"
    """
    backends = {
        "openai": lambda: OpenAIClient(model=model or "gpt-4o", api_key=api_key),
        "anthropic": lambda: AnthropicClient(model=model or "claude-sonnet-4-20250514", api_key=api_key),
        "deepseek": lambda: DeepSeekClient(model=model or "deepseek-chat", api_key=api_key),
        "ollama": lambda: OllamaClient(model=model or "llama3"),
    }

    if backend not in backends:
        raise ValueError(f"Unknown backend: {backend}. Choose from: {list(backends.keys())}")

    return backends[backend]()
