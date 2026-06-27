# -*- coding: utf-8 -*-
"""
示例3: 回调机制 -- 异步推送处理

场景: "下单后怎么知道成交了没? -- 不用轮询，交易所主动告诉你"

回调是事件驱动编程的核心:
  你下单后不需要反复查询状态，交易所有任何变化都会通过回调主动推送给你。

本示例演示:
  1. 继承 XtQuantTraderCallback 实现自定义回调类
  2. 注册回调，接收交易推送
  3. 下单/撤单并观察每个回调的触发
  4. 在回调中记录事件日志

回调方法一览:
  on_disconnected()              连接断开
  on_account_status(status)      账户状态变更
  on_stock_order(order)          委托状态推送 (每次状态变化都会推送)
  on_stock_trade(trade)          成交回报推送 (有成交时推送)
  on_order_error(order_error)    委托失败推送
  on_cancel_error(cancel_error)  撤单失败推送
  on_order_stock_async_response  异步下单响应

环境要求:
  - 已安装 xtquant, python-dotenv
  - miniQMT 客户端已启动并登录

注意: 本示例仅操作 513100.SH (纳指ETF)，一手(100股)
"""
import os
import time
import datetime
from dotenv import load_dotenv
from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
from xtquant.xttype import StockAccount
from xtquant import xtconstant


# ============================================================
# 从 .env 加载配置
# ============================================================
load_dotenv()
QMT_PATH = os.getenv("QMT_PATH")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
SESSION_ID = 100003

STOCK_CODE = "513100.SH"
VOLUME = 100


# ============================================================
# 自定义回调类
# ============================================================

class MyTraderCallback(XtQuantTraderCallback):
    """
    自定义交易回调类

    继承 XtQuantTraderCallback 并重写需要处理的回调方法。
    所有回调方法都在独立线程中执行，注意线程安全。
    """

    def __init__(self):
        super().__init__()
        self.event_log = []

    def _log(self, event_type, message):
        """记录回调事件"""
        now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        entry = f"[{now}] [{event_type}] {message}"
        self.event_log.append(entry)
        print(entry)

    def on_disconnected(self):
        """连接断开 -- 实际项目中应在此处实现重连逻辑"""
        self._log("断开连接", "与 miniQMT 的连接已断开")

    def on_account_status(self, status):
        """账户状态变更"""
        self._log("账户状态",
                  f"账户:{status.account_id} 类型:{status.account_type} "
                  f"状态:{status.status}")

    def on_stock_order(self, order):
        """
        委托状态推送 (最常用的回调)

        每当委托状态变化时触发: 未报->已报->已成/已撤
        """
        direction = "买入" if order.order_type == 23 else "卖出"
        status_map = {
            48: '未报', 49: '待报', 50: '已报', 51: '已报待撤',
            52: '部成待撤', 53: '部撤', 54: '已撤', 55: '部成',
            56: '已成', 57: '废单',
        }
        status_text = status_map.get(order.order_status,
                                     f'未知({order.order_status})')
        self._log("委托推送",
                  f"{order.stock_code} {direction} "
                  f"委托{order.order_volume}股 价格:{order.price:.2f} "
                  f"已成交{order.traded_volume}股 状态:{status_text} "
                  f"编号:{order.order_id}")

    def on_stock_trade(self, trade):
        """成交回报 -- 有成交时推送，一笔委托可能分多次成交"""
        self._log("成交回报",
                  f"{trade.stock_code} "
                  f"成交{trade.traded_volume}股 价格:{trade.traded_price:.2f} "
                  f"金额:{trade.traded_amount:.2f} "
                  f"成交编号:{trade.traded_id}")

    def on_order_error(self, order_error):
        """委托失败 -- 资金不足、涨跌停限制、股票停牌等"""
        self._log("委托失败",
                  f"编号:{order_error.order_id} "
                  f"错误码:{order_error.error_id} "
                  f"原因:{order_error.error_msg}")

    def on_cancel_error(self, cancel_error):
        """撤单失败 -- 委托已成交无法撤单等"""
        self._log("撤单失败",
                  f"编号:{cancel_error.order_id} "
                  f"错误码:{cancel_error.error_id} "
                  f"原因:{cancel_error.error_msg}")

    def on_order_stock_async_response(self, response):
        """异步下单响应 -- 返回委托编号"""
        self._log("异步响应",
                  f"账户:{response.account_id} "
                  f"委托编号:{response.order_id} 序列号:{response.seq}")


# ============================================================
# 主流程: 下单/撤单/异步下单，观察回调触发
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("示例3: 回调机制 -- 异步推送处理")
    print(f"场景: 下单后怎么知道成交了没?")
    print("=" * 60)

    # 创建回调实例
    callback = MyTraderCallback()

    # 创建交易实例并注册回调
    trader = XtQuantTrader(QMT_PATH, SESSION_ID)
    trader.register_callback(callback)
    trader.start()

    # 连接
    for i in range(3):
        if trader.connect() == 0:
            break
        time.sleep(1)
    else:
        print("连接 miniQMT 失败")
        trader.stop()
        exit(1)

    # 订阅账户
    account = StockAccount(ACCOUNT_ID)
    if trader.subscribe(account) != 0:
        print("订阅账户失败")
        trader.stop()
        exit(1)

    print(f"已连接，账户: {ACCOUNT_ID}\n")

    # ---- 场景1: 下单，观察 on_stock_order 回调 ----
    print("-" * 40)
    print("下限价单 (低价不会成交)，观察 on_stock_order 回调...")
    order_id = trader.order_stock(
        account, STOCK_CODE, xtconstant.STOCK_BUY,
        VOLUME, xtconstant.FIX_PRICE, 0.50,
        'callback_demo', '回调演示'
    )
    print(f"委托已提交，编号: {order_id}")
    print("等待回调推送...\n")
    time.sleep(3)

    # ---- 场景2: 撤单，观察状态变为已撤 ----
    print("-" * 40)
    print("撤单，观察状态从 '已报' 变为 '已撤'...")
    trader.cancel_order_stock(account, order_id)
    print("等待回调推送...\n")
    time.sleep(3)

    # ---- 场景3: 异步下单，观察 on_order_stock_async_response ----
    print("-" * 40)
    print("异步下单，观察 on_order_stock_async_response...")
    seq = trader.order_stock_async(
        account, STOCK_CODE, xtconstant.STOCK_BUY,
        VOLUME, xtconstant.FIX_PRICE, 0.50,
        'callback_demo', '异步回调演示'
    )
    print(f"异步下单序列号: {seq}")
    print("等待回调推送...\n")
    time.sleep(3)

    # 清理: 撤掉异步下单的委托
    orders = trader.query_stock_orders(account)
    if orders:
        for order in orders:
            if order.order_status in (50, 55):
                trader.cancel_order_stock(account, order.order_id)
    time.sleep(2)

    # ---- 展示事件日志 ----
    print("\n" + "=" * 60)
    print(f"回调事件日志 (共 {len(callback.event_log)} 条):")
    print("=" * 60)
    for entry in callback.event_log:
        print(f"  {entry}")

    input("\n按回车键退出...")
    trader.stop()
