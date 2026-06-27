#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text-to-SQL Agent -- nanobot 版

对标 DeepAgents 官方 text-to-sql-agent 示例。
使用 Chinook 数据库 + Skills 渐进式加载 + 自定义 query_db 工具。

运行: python agent.py "How many customers are from Canada?"
"""

import asyncio
import os
import sqlite3
import sys
import urllib.request
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE = Path(__file__).resolve().parent
NANOBOT_ROOT = WORKSPACE.parent.parent / "nanobot-main"
if str(NANOBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(NANOBOT_ROOT))

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.base import Tool
from nanobot.bus.queue import MessageBus
from nanobot.config.loader import load_config
from nanobot.nanobot import Nanobot, _make_provider

DB_PATH = WORKSPACE / "chinook.db"
DB_URL = "https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"


class QueryDBTool(Tool):
    """SQL 查询工具 -- 在 Chinook 数据库上执行只读 SQL"""

    def __init__(self, db_path: Path):
        self._db_path = db_path

    @property
    def name(self) -> str:
        return "query_db"

    @property
    def description(self) -> str:
        return (
            "Execute a read-only SQL query against the Chinook database. "
            "Returns query results as formatted text. Only SELECT and PRAGMA statements are allowed."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The SQL query to execute (SELECT only)"
                }
            },
            "required": ["sql"]
        }

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> str:
        sql = kwargs.get("sql", "").strip()
        if not sql:
            return "Error: empty SQL query"

        upper = sql.upper().lstrip()
        if not (upper.startswith("SELECT") or upper.startswith("PRAGMA")):
            return "Error: only SELECT and PRAGMA statements are allowed"

        try:
            conn = sqlite3.connect(str(self._db_path))
            cursor = conn.cursor()
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return "Query returned 0 rows."

            lines = [" | ".join(columns)]
            lines.append("-" * len(lines[0]))
            for row in rows[:50]:
                lines.append(" | ".join(str(v) for v in row))
            if len(rows) > 50:
                lines.append(f"... ({len(rows)} total rows, showing first 50)")

            return "\n".join(lines)
        except Exception as e:
            return f"SQL Error: {e}"


class PrintHook(AgentHook):
    async def before_execute_tools(self, ctx: AgentHookContext) -> None:
        for tc in ctx.tool_calls:
            print(f"  >> {tc.name}: {str(tc.arguments)[:120]}")


def ensure_db():
    """确保 chinook.db 存在"""
    if DB_PATH.exists():
        return
    print(f"Downloading Chinook database...")
    try:
        urllib.request.urlretrieve(DB_URL, str(DB_PATH))
        print(f"Downloaded to {DB_PATH}")
    except Exception as e:
        print(f"Download failed: {e}")
        print(f"Please manually download from: {DB_URL}")
        print(f"Save as: {DB_PATH}")
        sys.exit(1)


def build_bot() -> Nanobot:
    dashscope_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not dashscope_key:
        print("[Error] DASHSCOPE_API_KEY not set")
        sys.exit(1)

    config = load_config(WORKSPACE / "config.json")
    config.providers.dashscope.api_key = dashscope_key
    config.agents.defaults.workspace = str(WORKSPACE)

    provider = _make_provider(config)
    defaults = config.agents.defaults

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=WORKSPACE,
        model=defaults.model,
        max_iterations=defaults.max_tool_iterations,
        context_window_tokens=defaults.context_window_tokens,
        max_tool_result_chars=defaults.max_tool_result_chars,
        web_config=config.tools.web,
        exec_config=config.tools.exec,
        restrict_to_workspace=False,
        timezone=defaults.timezone,
    )

    # 注册自定义 SQL 查询工具
    loop.tools.register(QueryDBTool(DB_PATH))

    return Nanobot(loop)


async def main():
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "How many customers are from Canada?"

    ensure_db()

    print(f"\nText-to-SQL Agent (nanobot)")
    print(f"Question: {question}\n")

    bot = build_bot()
    result = await bot.run(question, session_key="sql:run", hooks=[PrintHook()])

    print(f"\n{'='*60}")
    print(f"Answer: {result.content}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
