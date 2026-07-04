# -*- coding: utf-8 -*-
"""
trader_node -- 交易执行节点

复用 17-XtQuant 实盘接口对接/CASE-XtQuant实盘交易/4-miniqmt_trader.py 的 MiniQMTTrader。

为了课堂演示安全，默认 dry-run（不真下单），把订单信息打印出来即可；
要真下单时设置环境变量 TRADER_DRY_RUN=0 并准备好 QMT_PATH/ACCOUNT_ID。
"""

import os
import sys
from datetime import datetime

from utils.env import LIB_DIR

# 把 lib 加到 sys.path，便于 import vendored 模块
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))


def _load_trader_class():
    """vendored from 17章 -- 直接 import 即可"""
    from miniqmt_trader import MiniQMTTrader  # noqa: WPS433
    return MiniQMTTrader


def trader_node(state: dict) -> dict:
    """下单节点：根据 trade_signal + approved 真实下单或 dry-run"""
    signal = state.get("trade_signal", {})
    approved = state.get("approved", False)

    print()
    print("=" * 70)
    print("[Trader] 交易执行台")
    print("=" * 70)

    if not approved:
        result = {
            "dry_run": True,
            "order_id": None,
            "submitted_at": datetime.now().isoformat(timespec="seconds"),
            "note": "用户未授权，跳过下单",
        }
        print("[Trader] 用户未授权，跳过下单")
        return {"trade_result": result, "messages": [{
            "role": "trader",
            "time": datetime.now().strftime("%H:%M:%S"),
            "content": "skipped (not approved)",
        }]}

    if signal.get("direction") == "hold" or signal.get("quantity", 0) <= 0:
        result = {
            "dry_run": True,
            "order_id": None,
            "submitted_at": datetime.now().isoformat(timespec="seconds"),
            "note": "信号为 hold，无需下单",
        }
        print("[Trader] 信号为 hold，无需下单")
        return {"trade_result": result, "messages": [{
            "role": "trader",
            "time": datetime.now().strftime("%H:%M:%S"),
            "content": "skipped (hold)",
        }]}

    stock = signal["stock_code"]
    qty = int(signal["quantity"])
    price = float(signal["price"])
    direction = signal["direction"]

    dry_run = os.environ.get("TRADER_DRY_RUN", "1") == "1"

    if dry_run:
        print(f"[Trader] [DRY-RUN] {direction.upper()} {stock} {qty}股 @ {price}")
        result = {
            "dry_run": True,
            "order_id": None,
            "submitted_at": datetime.now().isoformat(timespec="seconds"),
            "note": f"dry-run: {direction} {qty} @ {price}",
        }
        return {"trade_result": result, "messages": [{
            "role": "trader",
            "time": datetime.now().strftime("%H:%M:%S"),
            "content": result["note"],
        }]}

    qmt_path = os.environ["QMT_PATH"]
    account_id = os.environ["ACCOUNT_ID"]

    MiniQMTTrader = _load_trader_class()
    trader = MiniQMTTrader(
        qmt_path=qmt_path,
        account_id=account_id,
        max_positions=10,
        max_order_amount=max(int(qty * price * 1.2), 200_000),
    )
    trader.connect()

    if direction == "buy":
        order_id = trader.buy(stock, qty, price=price,
                              strategy_name="team-workflow", remark="LangGraph 团队工作流")
    else:
        order_id = trader.sell(stock, qty, price=price,
                               strategy_name="team-workflow", remark="LangGraph 团队工作流")

    trader.disconnect()

    result = {
        "dry_run": False,
        "order_id": int(order_id) if order_id else None,
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "note": f"miniQMT 已提交 {direction} {qty} @ {price}",
    }
    print(f"[Trader] miniQMT 已提交，委托编号 {order_id}")
    return {"trade_result": result, "messages": [{
        "role": "trader",
        "time": datetime.now().strftime("%H:%M:%S"),
        "content": result["note"],
    }]}
