# 🪞 Mirror (赛镜)

> A cyber mirror that reflects the real you. · 一面赛博镜子，照见真实的自己。

**Mirror** is an open-source, local-first, self-evolving personal AI agent. It learns your behavior, remembers your preferences and grows its own capabilities — the more you use it, the better it knows you.

[中文说明](README.zh.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

---

## 🎯 Why Mirror?

| | Typical AI assistant | **Mirror** |
|:--|:-----------|:--------------|
| Capability | Fixed at release | **Self-evolving** — grows with use |
| Memory | Chat history | **Personality model**: preferences, habits, thinking patterns |
| Privacy | Data uploaded to the cloud | **Local-first**, Ollama supported |
| Customization | Prompt tuning | **Tool self-synthesis** — learns missing skills on the spot |
| Openness | — | **MIT**, fully transparent |

## 🧠 Core mechanism: self-evolution

```
Day 1                    Day 7                       Day 30
0 tools ──────────────→ learns weather/alarms ────→ sleep analysis / weekly reports / proactive suggestions
chat only                starts to be useful         irreplaceable
```

The *Yunjue Agent* paper proved the feasibility of "self-evolution". Mirror builds on it with three key differentiators:

1. **Not just tools** — tools + memory + preferences + workflows, evolving across every dimension
2. **Personal** — constructs a structured model of *you*, not a generic agent
3. **Minimal deployment** — one `pip install`, running in 5 minutes

## 🚀 Quick start

```bash
# Install
pip install mirror-agent

# Initialize
mirror start

# Start from zero — Mirror knows nothing, but will gradually learn everything
```

```
🚀 Mirror v0.1.0
model: gpt-4o
tools: 0
interactions: 0
EGL: ∞ (not yet evolved)

you:  check tomorrow's weather in Hangzhou
Mirror: I don't have a weather tool — let me create one...
        ✓ new tool: get_weather synthesized
        Hangzhou tomorrow 18–26°C, cloudy turning sunny ☁️→☀️

you:  good day for a run?
Mirror: based on your preference (you like evening runs) and tomorrow's weather...
        suggested window 17:00–18:00 — comfortable temperature, low wind.
```

## 📐 Architecture

```
┌─────────────────────────────────────────┐
│              Mirror Agent               │
│                                         │
│  ┌─────────┐   ┌────────────────┐      │
│  │ Manager │──→│ Tool Developer │      │
│  │(dispatch)│  │  (synthesize)  │      │
│  └────┬────┘   └────────────────┘      │
│       │                                  │
│  ┌────▼────┐   ┌────────────────┐      │
│  │Executor │──→│   Integrator   │      │
│  │ (ReAct) │   │ (compose reply)│      │
│  └─────────┘   └────────────────┘      │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │           Memory Layer           │  │
│  │  • preference model  • personality│ │
│  │  • memory                         │ │
│  └──────────────────────────────────┘  │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │        Sensor Integration        │  │
│  │  • Apple Health  • Google Fit    │  │
│  └──────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

## 🗺️ Roadmap

- [x] **v0.1.0** — core engine: agent + tool self-synthesis + sandbox
- [ ] **v0.2.0** — LLM integrations (OpenAI / Anthropic / Ollama)
- [ ] **v0.3.0** — memory layer + preference learning
- [ ] **v0.4.0** — health data (Apple Health / Google Fit)
- [ ] **v0.5.0** — web UI + chat interface
- [ ] **v1.0.0** — stable API + full docs + skills marketplace

## 🤝 Contributing

Mirror is at an early stage — bug reports, feature ideas, documentation and PRs are all welcome.

See [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 License

MIT License — use, modify and distribute freely.

---

<p align="center">
  <i>AI should grow with you, not watch you.</i>
</p>
