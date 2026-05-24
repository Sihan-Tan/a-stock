# -*- coding: utf-8 -*-
"""
week2 数据采集一站式运行脚本

功能: 按文件编号顺序(1~7)自动运行 week2_get_data/ 目录下的所有数据采集脚本
用法: python run_week2_get_data.py

运行顺序:
  1-行情数据采集.py  → 采集股票日K线数据
  2-财务数据采集.py  → 采集财务报表数据
  3-宏观数据采集.py  → 采集宏观经济指标
  4-新闻事件采集.py  → 采集新闻公告数据
  5-研报数据采集.py  → 采集券商研报数据
  6-财经日历采集.py  → 采集财经日历事件
  7-关键催化剂采集.py → 采集关键催化剂事件

任一步骤失败则终止后续执行并返回对应错误码
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def _leading_int(filename: str) -> int | None:
    """从文件名中提取开头的数字，如 '1-行情数据采集.py' → 1"""
    m = re.match(r"^(\d+)", filename)
    return int(m.group(1)) if m else None


def main() -> int:
    """主函数: 发现脚本 → 排序 → 逐个执行 → 返回状态码"""
    # 定位项目根目录和 week2_get_data 目录
    repo_root = Path(__file__).resolve().parent
    target_dir = repo_root / "week2_get_data"

    # 检查目录是否存在
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"[ERROR] 目录不存在: {target_dir}")
        return 2

    # 扫描目录，筛选 1~7 开头的 .py 文件
    candidates: list[tuple[int, Path]] = []
    for p in target_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() != ".py":         # 只处理 .py 文件
            continue
        n = _leading_int(p.name)              # 提取文件名中的数字编号
        if n is None:
            continue
        if 1 <= n <= 7:                       # 限定范围 1~7
            candidates.append((n, p))

    # 未找到任何脚本
    if not candidates:
        print(f"[ERROR] 未找到 1~7 开头的 .py 文件: {target_dir}")
        return 3

    # 按编号排序，确定执行顺序
    candidates.sort(key=lambda x: (x[0], x[1].name))
    scripts = [p for _, p in candidates]

    # 预览将要运行的脚本列表
    print("[INFO] 将按顺序运行以下脚本：")
    for p in scripts:
        print(f"  - {p.name}")

    # 逐个运行脚本
    for p in scripts:
        print(f"\n[RUN ] {p.name}")
        completed = subprocess.run(
            [sys.executable, str(p)],          # 用当前 Python 解释器执行
            cwd=str(target_dir),               # 工作目录设为脚本所在目录
        )
        # 任一步骤失败则终止后续执行
        if completed.returncode != 0:
            print(f"[FAIL] {p.name} (exit={completed.returncode})")
            return completed.returncode
        print(f"[OK  ] {p.name}")

    print("\n[DONE] week2_get_data 1~7 全部运行完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
