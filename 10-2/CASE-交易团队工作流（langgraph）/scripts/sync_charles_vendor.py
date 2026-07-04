# -*- coding: utf-8 -*-
"""
维护用：把 14-16 章 Charles 目录同步到本案例 vendor/charles_agent。
发布给学员前在本案例根目录执行: python scripts/sync_charles_vendor.py
（要求：与本案例同属一个仓库，且根目录下存在 14-16-CASE-Charles投研Agent）
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

CASE_ROOT = Path(__file__).resolve().parent.parent
# 仓库根目录（量化交易-2）：CASE -> 20-团队架构设计 -> 根
REPO_ROOT = CASE_ROOT.parent.parent
SRC = REPO_ROOT / "14-16-CASE-Charles投研Agent"
DST = CASE_ROOT / "vendor" / "charles_agent"


def main() -> None:
    if not SRC.is_dir():
        print(f"[错误] 未找到源目录: {SRC}", file=sys.stderr)
        sys.exit(1)
    DST.parent.mkdir(parents=True, exist_ok=True)
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC / "agent.py", DST / "agent.py")
    shutil.copytree(SRC / "skills", DST / "skills")
    (DST / "data").mkdir(parents=True, exist_ok=True)
    shutil.copytree(SRC / "data" / "vector_store", DST / "data" / "vector_store")
    shutil.copytree(SRC / "data" / "financial_data", DST / "data" / "financial_data")
    print(f"[完成] 已同步到 {DST}")


if __name__ == "__main__":
    main()
