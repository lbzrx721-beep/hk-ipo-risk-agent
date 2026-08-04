"""启动新版React工作台和FastAPI服务。"""

from __future__ import annotations

import argparse
from pathlib import Path
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser

import uvicorn


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动IPO风险分析新版网页。")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="只启动服务，不自动打开浏览器。",
    )
    return parser.parse_args()


def _browser_url(host: str, port: int) -> str:
    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return f"http://{browser_host}:{port}"


def _open_browser_when_ready(host: str, port: int) -> None:
    """等待健康检查通过后打开外部默认浏览器。"""
    url = _browser_url(host, port)
    health_url = f"{url}/api/health"
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        try:
            with urlopen(health_url, timeout=1) as response:  # noqa: S310
                if response.status < 500:
                    webbrowser.open_new_tab(url)
                    return
        except (OSError, URLError):
            time.sleep(0.25)


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if not args.no_browser:
        threading.Thread(
            target=_open_browser_when_ready,
            args=(args.host, args.port),
            name="open-web-workbench",
            daemon=True,
        ).start()
    uvicorn.run(
        "ipo_risk.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        app_dir=str(project_root / "src"),
    )


if __name__ == "__main__":
    main()
