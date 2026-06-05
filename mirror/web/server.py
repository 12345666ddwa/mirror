"""
Mirror Web Dashboard — The user-facing interface.

Serves:
  - /           → Dashboard with insights, health data, memory browser
  - /api/health → Health data JSON
  - /api/insights → Discovered insights
  - /api/memory → Memory store (browse, edit, delete)
  - /api/chat   → Conversation endpoint
"""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from pathlib import Path

from ..core.agent import MirrorAgent
from ..core.loop import MirrorLoop, create_mirror
from ..core.insight_engine import InsightEngine, generate_demo_data


class MirrorHandler(BaseHTTPRequestHandler):

    agent = None
    loop = None
    insight_engine = None
    demo_data = None
    demo_insights = None

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._serve_dashboard()
        elif path == "/api/health":
            self._json(self.demo_data)
        elif path == "/api/insights":
            self._json([i.to_dict() for i in self.demo_insights])
        elif path == "/api/memory":
            self._json({
                "preferences": self.agent.state.preferences,
                "persona": self.agent.state.persona,
                "tool_count": len(self.agent.state.tools),
                "interactions": self.agent.state.interaction_count,
            })
        elif path == "/api/status":
            self._json(self.agent.stats)
        else:
            self._json({"error": "Not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if path == "/api/chat":
            msg = body.get("message", "")
            if msg and self.loop:
                try:
                    resp = self.loop.chat(msg)
                    self._json({"response": resp})
                except Exception as e:
                    self._json({"response": f"错误: {e}"})
            else:
                self._json({"response": "对话功能需要配置LLM。启动时加 --api-key"})

        elif path == "/api/memory/delete":
            key = body.get("key", "")
            if key in self.agent.state.preferences:
                del self.agent.state.preferences[key]
                self.agent.save_state()
                self._json({"deleted": key})
            else:
                self._json({"error": "Key not found"}, 404)

        elif path == "/api/memory/add":
            key, val = body.get("key", ""), body.get("value", "")
            if key:
                self.agent.update_preference(key, val)
                self.agent.save_state()
                self._json({"added": {key: val}})
            else:
                self._json({"error": "No key"}, 400)

    def _serve_dashboard(self):
        dp = Path(__file__).parent.parent.parent / "dashboard.html"
        if dp.exists():
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(dp.read_bytes())
        else:
            self._json({"error": "Dashboard not found"}, 404)

    def _json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, default=str).encode())


def start_server(host="127.0.0.1", port=8756, backend="openai", model="", api_key=""):
    agent = MirrorAgent()
    agent.load_state()
    loop = None
    if api_key:
        try:
            loop = create_mirror(backend=backend, model=model, api_key=api_key)
        except Exception as e:
            print(f"⚠️  LLM连接失败: {e}")

    engine = InsightEngine()
    demo_data = generate_demo_data(30)
    demo_insights = engine.analyze(demo_data)

    MirrorHandler.agent = agent
    MirrorHandler.loop = loop
    MirrorHandler.insight_engine = engine
    MirrorHandler.demo_data = demo_data
    MirrorHandler.demo_insights = demo_insights

    server = HTTPServer((host, port), MirrorHandler)
    print(f"""\n╔══════════════════════════════════════════╗
║       🪞 赛镜 Mirror Dashboard v0.2     ║
╠══════════════════════════════════════════╣
║  地址: http://{host}:{port}              ║
║  洞察: {len(demo_insights)} 条自动发现   ║
║  工具: {len(agent.state.tools)} 个       ║
║  记忆: {len(agent.state.preferences)} 条 ║
╚══════════════════════════════════════════╝
按 Ctrl+C 停止
""")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 再见")
        agent.save_state()
        server.shutdown()


def main():
    import argparse
    p = argparse.ArgumentParser(description="Mirror Dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8756)
    p.add_argument("--backend", default="openai")
    p.add_argument("--model", default="")
    p.add_argument("--api-key", default="")
    args = p.parse_args()
    start_server(args.host, args.port, args.backend, args.model, args.api_key)
