# -*- coding: utf-8 -*-
"""
示例2: 下单与撤单

场景: "买一手纳指ETF试试，然后把没成交的撤掉"

本示例演示 order_stock 下单和 cancel_order_stock 撤单的完整流程:
  1. 限价买入 (FIX_PRICE) -- 挂指定价格
  2. 市价买入 -- 五档即时成交
  3. 查询委托状态
  4. 撤销指定委托
  5. 批量撤销所有可撤委托

order_stock 参数说明:
  account       : StockAccount 账户对象
  stock_code    : 股票代码, 如 "513100.SH"
  order_type    : xtconstant.STOCK_BUY (买入) / STOCK_SELL (卖出)
  order_volume  : 委托数量 (必须是100的整数倍)
  price_type    : 报价类型 (见下方说明)
  price         : 委托价格 (限价时填价格, 市价时填0)
  strategy_name : 策略名称 (可选)
  order_remark  : 委托备注 (可选)

报价类型:
  FIX_PRICE                   : 限价 -- 指定价格挂单
  LATEST_PRICE                : 最新价 -- 以当前最新价作为限价
  MARKET_SH_CONVERT_5_LIMIT   : 沪市市价 -- 五档即时成交，剩余转限价
  MARKET_SZ_CONVERT_5_CANCEL  : 深市市价 -- 五档即时成交，剩余撤销

环境要求:
  - 已安装 xtquant, python-dotenv
  - miniQMT 客户端已启动并登录

注意: 本示例仅操作 513100.SH (纳指ETF)，买卖一手(100股)
"""
import os
import time
from dotenv import load_dotenv
from xtquant.xttrader import XtQuantTrader
from xtquant.xttype import StockAccount
from xtquant import xtconstant


# ============================================================
# 从 .env 加载配置
# ============================================================
load_dotenv()
QMT_PATH = os.getenv("QMT_PATH")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
SESSION_ID = 100002

STOCK_CODE = "513100.SH"   # 纳指ETF
VOLUME = 100               # 一手 = 100股


# ============================================================
# 连接
# ============================================================

def connect_trader(path, session_id, account_id):
    """创建交易实例、连接并订阅账户"""
    trader = XtQuantTrader(path, session_id)
    trader.start()
    for i in range(3):
        if trader.connect() == 0:
            break
        time.sleep(1)
    else:
        raise Exception("连接 miniQMT 失败")

    account = StockAccount(account_id)
    if trader.subscribe(account) != 0:
        raise Exception("订阅账户失败")
    print(f"已连接，账户: {account_id}\n")
    return trader, account


# ============================================================
# 下单函数
# ============================================================

def buy_limit(trader, account, stock_code, volume, price):
    """
    限价买入

    以指定价格挂单，只有市场价达到或低于委托价时才成交。
    适合: 想在某个目标价格买入时使用。
    """
    order_id = trader.order_stock(
        account, stock_code, xtconstant.STOCK_BUY,
        volume, xtconstant.FIX_PRICE, price,
        'demo', '限价买入'
    )
    print(f"[限价买入] {stock_code} {volume}股 价格:{price} 委托编号:{order_id}")
    return order_id


def buy_market(trader, account, stock_code, volume):
    """
    市价买入

    沪市和深市市价委托类型不同:
      沪市(.SH): MARKET_SH_CONVERT_5_LIMIT -- 五档即时成交，剩余转限价
      深市(.SZ): MARKET_SZ_CONVERT_5_CANCEL -- 五档即时成交，剩余撤销
    """
    if stock_code.endswith('.SH'):
        price_type = xtconstant.MARKET_SH_CONVERT_5_LIMIT
        type_name = "沪市-五档转限价"
    else:
        price_type = xtconstant.MARKET_SZ_CONVERT_5_CANCEL
        type_name = "深市-五档撤余"

    order_id = trader.order_stock(
        account, stock_code, xtconstant.STOCK_BUY,
        volume, price_type, 0,
        'demo', '市价买入'
    )
    print(f"[市价买入] {stock_code} {volume}股 ({type_name}) 委托编号:{order_id}")
    return order_id


def sell_market(trader, account, stock_code, volume):
    """市价卖出"""
    if stock_code.endswith('.SH'):
        price_type = xtconstant.MARKET_SH_CONVERT_5_LIMIT
    else:
        price_type = xtconstant.MARKET_SZ_CONVERT_5_CANCEL

    order_id = trader.order_stock(
        account, stock_code, xtconstant.STOCK_SELL,
        volume, price_type, 0,
        'demo', '市价卖出'
    )
    print(f"[市价卖出] {stock_code} {volume}股 委托编号:{order_id}")
    return order_id


# ============================================================
# 撤单函数
# ============================================================

def cancel_order(trader, account, order_id):
    """
    撤销指定委托

    只有 "已报(50)" 和 "部成(55)" 状态的委托才能撤。
    已成交(56) 或 已撤(54) 的委托无法撤销。
    """
    result = trader.cancel_order_stock(account, order_id)
    status = "撤单请求已提交" if result == 0 else f"撤单失败(错误码:{result})"
    print(f"[撤单] 编号:{order_id} {status}")
    return result


def cancel_all_orders(trader, account):
    """批量撤销所有可撤委托 (状态为 已报/部成)"""
    orders = trader.query_stock_orders(account)
    if not orders:
        print("[批量撤单] 今日暂无委托")
        return

    cancelable = [o for o in orders if o.order_status in (50, 55)]
    if not cancelable:
        print("[批量撤单] 没有可撤的委托")
        return

    print(f"[批量撤单] 找到 {len(cancelable)} 笔可撤委托:")
    for order in cancelable:
        print(f"  -> {order.stock_code} 编号:{order.order_id} "
              f"委托{order.order_volume}股 已成交{order.traded_volume}股")
        cancel_order(trader, account, order.order_id)
        time.sleep(0.5)


# ============================================================
# 主流程: 下单 -> 查看 -> 撤单，完整闭环
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("示例2: 下单与撤单")
    print(f"场景: 买一手 {STOCK_CODE}，然后撤掉")
    print("=" * 60)

    trader, account = connect_trader(QMT_PATH, SESSION_ID, ACCOUNT_ID)

    # ---- 第一步: 限价买入 (挂低价，不会成交，用于演示撤单) ----
    print("[步骤1] 限价买入 -- 挂 0.50 元低价，不会成交")
    order_id = buy_limit(trader, account, STOCK_CODE, VOLUME, 0.50)

    # ---- 第二步: 等待委托上报 ----
    print("\n等待 2 秒让委托上报...")
    time.sleep(2)

    # ---- 第三步: 查看委托状态 ----
    print("\n[步骤2] 查看委托状态")
    status_map = {
        48: '未报', 49: '待报', 50: '已报', 51: '已报待撤',
        52: '部成待撤', 53: '部撤', 54: '已撤', 55: '部成',
        56: '已成', 57: '废单',
    }
    orders = trader.query_stock_orders(account)
    if orders:
        for o in orders:
            print(f"  {o.stock_code} 编号:{o.order_id} "
                  f"委托{o.order_volume}股 "
                  f"状态:{status_map.get(o.order_status, o.order_status)}")

    # ---- 第四步: 撤单 ----
    print(f"\n[步骤3] 撤销委托 {order_id}")
    cancel_order(trader, account, order_id)

    # ---- 第五步: 确认撤单结果 ----
    time.sleep(2)
    print("\n[步骤4] 确认撤单结果")
    orders = trader.query_stock_orders(account)
    if orders:
        for o in orders:
            if o.order_id == order_id:
                s = status_map.get(o.order_status, str(o.order_status))
                print(f"  委托 {o.order_id}: {s}")

    # ---- 附加: 批量撤单 ----
    print("\n" + "-" * 40)
    print("[附加] 批量撤销所有可撤委托")
    cancel_all_orders(trader, account)

    input("\n按回车键退出...")
    trader.stop()
