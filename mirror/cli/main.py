"""
Mirror CLI — Command-line interface for the personal AI agent.

Usage:
    mirror start          Launch interactive session
    mirror status         Show agent stats & evolution
    mirror health         Connect health data sources
"""

import asyncio
import json
import os
import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

app = typer.Typer(
    name="mirror",
    help="赛镜 Mirror — 开源自进化个人AI Agent",
)
console = Console()


@app.command()
def start(
    backend: str = typer.Option("openai", help="LLM backend: openai, anthropic, deepseek, ollama"),
    model: str = typer.Option("", help="Model name (uses backend default if empty)"),
    api_key: str = typer.Option("", help="API key (or set env var)"),
):
    """Launch interactive Mirror session."""
    from mirror.core.loop import create_mirror

    try:
        mirror = create_mirror(backend=backend, model=model, api_key=api_key)
    except Exception as e:
        console.print(f"[red]启动失败: {e}[/red]")
        console.print("[dim]请检查 API key 或网络连接[/dim]")
        return

    console.print(
        Panel.fit(
            f"[bold]赛镜 Mirror v0.1.0[/bold]\n"
            f"后端: {backend} / {mirror.llm.model}\n"
            f"工具库: {len(mirror.agent.state.tools)} 个工具\n"
            f"交互次数: {mirror.agent.state.interaction_count}\n"
            f"EGL: {mirror.agent.state.egl if mirror.agent.state.egl != float('inf') else '∞ (未进化)'}",
            title="🚀 Mirror",
            border_style="yellow",
        )
    )

    if not mirror.agent.state.tools:
        console.print("[dim]首次启动 — 我还不了解你，让我们开始吧。[/dim]\n")
    else:
        console.print(f"[dim]已加载 {len(mirror.agent.state.tools)} 个工具，欢迎回来。[/dim]\n")

    console.print("[dim]输入 'exit' 退出, 'status' 查看状态, 'help' 帮助[/dim]\n")

    while True:
        try:
            user_input = console.input("[bold green]你:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]再见！[/yellow]")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            mirror.agent.save_state()
            console.print(f"[yellow]状态已保存 ({len(mirror.agent.state.tools)} 工具, {mirror.agent.state.interaction_count} 次交互)[/yellow]")
            break
        if user_input.lower() == "status":
            _show_status(mirror.agent)
            continue

        console.print("[bold]赛镜:[/bold] [dim]思考中……[/dim]")
        try:
            response = mirror.chat(user_input)
            console.print(f"[bold]赛镜:[/bold] {response}\n")
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]\n")


@app.command()
def status():
    """Show agent evolution stats."""
    from mirror.core.agent import MirrorAgent

    agent = MirrorAgent(state_dir="~/.mirror")
    loaded = agent.load_state()

    if not loaded:
        console.print("[yellow]尚未启动。运行 [bold]mirror start[/bold] 开始。[/yellow]")
        return

    _show_status(agent)


@app.command()
def health():
    """View health data integration status."""
    health_dir = os.path.expanduser("~/.mirror/health")
    if not os.path.exists(health_dir):
        console.print("[yellow]暂无健康数据。[/yellow]")
        console.print("将 Apple Health 导出文件放到 ~/.mirror/health/export.xml")
        return

    files = os.listdir(health_dir)
    console.print(f"[green]健康数据文件:[/green] {len(files)} 个")
    for f in files:
        size = os.path.getsize(os.path.join(health_dir, f))
        console.print(f"  • {f} ({size:,} bytes)")


def _show_status(agent):
    """Display agent status in a rich table."""
    table = Table(title="赛镜 Mirror — 当前状态")
    table.add_column("指标", style="cyan")
    table.add_column("值", style="green")

    table.add_row("工具数量", str(len(agent.state.tools)))
    table.add_row("总交互次数", str(agent.state.interaction_count))
    table.add_row("累计合成工具", str(agent.state.total_tool_synthesis))
    table.add_row("EGL (进化通用性)", f"{agent.state.egl:.4f}" if agent.state.egl != float("inf") else "∞")
    table.add_row("偏好数量", str(len(agent.state.preferences)))
    table.add_row("状态目录", os.path.expanduser("~/.mirror"))

    console.print(table)

    if agent.state.tools:
        tool_table = Table(title="工具库")
        tool_table.add_column("工具名")
        tool_table.add_column("使用次数")
        tool_table.add_column("成功率")
        for t in sorted(agent.state.tools, key=lambda x: x.usage_count, reverse=True)[:10]:
            tool_table.add_row(
                t.name,
                str(t.usage_count),
                f"{t.success_rate:.0%}",
            )
        console.print(tool_table)


if __name__ == "__main__":
    app()
