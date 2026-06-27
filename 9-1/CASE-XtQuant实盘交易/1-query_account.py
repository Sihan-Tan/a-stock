# -*- coding: utf-8 -*-
"""
示例1: 连接账户与信息查询

场景: "看看我的账户情况 -- 有多少钱、持了什么股、今天委托了啥"

本示例演示:
  1. 创建 XtQuantTrader 并连接 miniQMT
  2. 订阅账户
  3. 查询: 资产、持仓、当日委托、当日成交
  4. 安全断开

涉及的关键概念:
  - session_id: 会话编号，区分不同策略实例
    同一个 miniQMT 可被多个 Python 策略同时连接，
    每个策略必须使用不同的 session_id，否则会互相收到对方的回调
  - StockAccount: 账户对象，所有查询和下单都需要它
  - T+1: 当日买入的股票，can_use_volume 为 0，次日才能卖

环境要求:
  - 已安装 xtquant, python-dotenv, pandas
  - miniQMT 客户端已启动并登录（极简模式）
  - .env 中配置 QMT_PATH 和 ACCOUNT_ID
"""
import os
import time
import pandas as pd
from dotenv import load_dotenv
from xtquant.xttrader import XtQuantTrader
from xtquant.xttype import StockAccount


# ============================================================
# 从 .env 加载配置
# ============================================================
load_dotenv()
QMT_PATH = os.getenv("QMT_PATH")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
SESSION_ID = 100001


# ============================================================
# 连接 miniQMT
# ============================================================

def connect_trader(path, session_id, account_id):
    """
    创建交易实例、连接 miniQMT 并订阅账户

    返回:
        (trader, account) 元组
    """
    trader = XtQuantTrader(path, session_id)
    # 启动交易线程（必须在 connect 之前调用）
    trader.start()

    # 连接，支持重试
    for i in range(3):
        if trader.connect() == 0:
            break
        print(f"连接重试 {i + 1}/3...")
        time.sleep(2)
    else:
        raise Exception(
            f"连接 miniQMT 失败，请检查:\n"
            f"  1. miniQMT 客户端是否已启动并登录\n"
            f"  2. QMT_PATH 是否正确: {path}")

    # 订阅账户
    account = StockAccount(account_id)
    if trader.subscribe(account) != 0:
        raise Exception("订阅账户失败")

    print(f"已连接 miniQMT，账户: {account_id}，会话ID: {session_id}")
    return trader, account


# ============================================================
# 查询功能
# ============================================================

def query_asset(trader, account):
    """
    查询账户资产

    返回字段:
      - total_asset: 总资产
      - cash: 可用资金
      - market_value: 持仓市值
      - frozen_cash: 冻结资金
    """
    asset = trader.query_stock_asset(account)
    if asset is None:
        print("查询资产失败")
        return None

    print("\n--- 账户资产 ---")
    print(f"  总资产:     {asset.total_asset:>14,.2f} 元")
    print(f"  可用资金:   {asset.cash:>14,.2f} 元")
    print(f"  持仓市值:   {asset.market_value:>14,.2f} 元")
    print(f"  冻结资金:   {asset.frozen_cash:>14,.2f} 元")
    return asset


def query_positions(trader, account):
    """
    查询股票持仓

    返回字段:
      - stock_code: 股票代码
      - volume: 持仓数量
      - can_use_volume: 可用数量 (T+1制度下，当日买入不可卖)
      - open_price: 开仓均价 (成本价)
      - market_value: 持仓市值
    """
    positions = trader.query_stock_positions(account)
    if not positions:
        print("\n--- 股票持仓 ---")
        print("  当前无持仓")
        return []

    rows = []
    for pos in positions:
        if pos.volume > 0:
            rows.append({
                '股票代码': pos.stock_code,
                '持仓数量': pos.volume,
                '可用数量': pos.can_use_volume,
                '成本价': round(pos.open_price, 3),
                '持仓市值': round(pos.market_value, 2),
            })

    print(f"\n--- 股票持仓 ({len(rows)} 只) ---")
    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
    else:
        print("  当前无持仓")

    return positions


def query_orders(trader, account):
    """
    查询当日委托

    委托状态码:
      48=未报, 49=待报, 50=已报, 51=已报待撤,
      52=部成待撤, 53=部撤, 54=已撤, 55=部成, 56=已成, 57=废单
    """
    orders = trader.query_stock_orders(account)
    if not orders:
        print("\n--- 当日委托 ---")
        print("  今日暂无委托")
        return []

    status_map = {
        48: '未报', 49: '待报', 50: '已报', 51: '已报待撤',
        52: '部成待撤', 53: '部撤', 54: '已撤', 55: '部成',
        56: '已成', 57: '废单',
    }

    rows = []
    for order in orders:
        rows.append({
            '委托编号': order.order_id,
            '股票代码': order.stock_code,
            '买卖': '买入' if order.order_type == 23 else '卖出',
            '委托数量': order.order_volume,
            '委托价格': round(order.price, 3),
            '成交数量': order.traded_volume,
            '状态': status_map.get(order.order_status,
                                   f'未知({order.order_status})'),
        })

    print(f"\n--- 当日委托 ({len(orders)} 笔) ---")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return orders


def query_trades(trader, account):
    """
    查询当日成交

    一笔委托可能分多次成交，每次都会产生一条成交记录。
    """
    trades = trader.query_stock_trades(account)
    if not trades:
        print("\n--- 当日成交 ---")
        print("  今日暂无成交")
        return []

    rows = []
    for trade in trades:
        rows.append({
            '成交编号': trade.traded_id,
            '股票代码': trade.stock_code,
            '成交价格': round(trade.traded_price, 3),
            '成交数量': trade.traded_volume,
            '成交金额': round(trade.traded_amount, 2),
            '委托编号': trade.order_id,
        })

    print(f"\n--- 当日成交 ({len(trades)} 笔) ---")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return trades


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("示例1: 连接账户与信息查询")
    print("场景: 看看我的账户情况")
    print("=" * 60)

    trader, account = connect_trader(QMT_PATH, SESSION_ID, ACCOUNT_ID)

    query_asset(trader, account)
    query_positions(trader, account)
    query_orders(trader, account)
    query_trades(trader, account)

    input("\n按回车键退出...")
    trader.stop()
