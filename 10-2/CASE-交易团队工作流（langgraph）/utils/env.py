# -*- coding: utf-8 -*-
"""
环境工具：
  1. Windows UTF-8 兼容（统一在每个入口先调一次）
  2. 路径常量均在案例目录内（lib / scripts / vendor），不依赖仓库其他章节路径
"""

import locale
import os
import sys
from pathlib import Path


def setup_utf8():
    """Windows 控制台 UTF-8 兼容，避免中文乱码"""
    if sys.platform != "win32":
        return
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(locale, "getencoding"):
        locale.getencoding = lambda: "utf-8"  # type: ignore[attr-defined]
    locale.getpreferredencoding = lambda do_setlocale=True: "utf-8"
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# 项目根目录（本 CASE 文件夹）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 本项目内部路径（自包含）
LIB_DIR = PROJECT_ROOT / "lib"          # 复制进来的纯 Python 模块
SCRIPTS_DIR = PROJECT_ROOT / "scripts"  # 复制进来的可执行脚本
# Charles：vendor/charles_agent 为与 14-16 章同源内容（agent + skills + data），随案例分发
# 维护：若本目录尚无 vendor，可在仓库内执行 python scripts/sync_charles_vendor.py 从 14-16 章同步
CHARLES_AGENT_DIR = PROJECT_ROOT / "vendor" / "charles_agent"
