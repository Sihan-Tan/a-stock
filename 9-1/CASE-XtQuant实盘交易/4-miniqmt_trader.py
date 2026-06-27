# -*- coding: utf-8 -*-
"""
示例4: MiniQMTTrader 封装类

场景: "能不能一行代码搞定? -- 把连接、下单、撤单、查询、回调全封装"

将前面学到的所有能力封装为一个可复用的交易工具类:
  1. 连接管理: connect() / disconnect()
  2. 买卖操作: buy() / sell() -- 自动区分沪深市价类型
  3. 撤单: cancel() / cancel_all()
  4. 查询: query_asset() / query_positions() / query_orders() / query_trades()
  5. 内置回调: 自动记录委托、成交、错误等事件
  6. 风控检查: 持仓上限、单笔金额上限

环境要求:
  - 已安装 xtquant, python-dotenv
  - miniQMT 客户端已启动并登录
"""
import os
import time
import datetime
from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
from xtquant.xttype import StockAccount
from xtquant import xtconstant


class _TraderCallback(XtQuantTraderCallback):
    """内置回调类，将交易事件记录到 MiniQMTTrader 的事件列表"""

    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def _now(self):
        return datetime.datetime.now().strftime("%H:%M:%S")

    def on_disconnected(self):
        msg = f"[{self._now()}] 连接断开"
        self.owner._events.append(msg)
        print(msg)

    def on_account_status(self, status):
        msg = (f"[{self._now()}] 账户状态: "
               f"{status.account_id} 状态:{status.status}")
        self.owner._events.append(msg)

    def on_stock_order(self, order):
        direction = "买入" if order.order_type == 23 else "卖出"
        status_map = {
            48: '未报', 49: '待报', 50: '已报', 51: '已报待撤',
            52: '部成待撤', 53: '部撤', 54: '已撤', 55: '部成',
            56: '已成', 57: '废单',
        }
        status_text = status_map.get(order.order_status,
                                     str(order.order_status))
        msg = (f"[{self._now()}] 委托推送: {order.stock_code} {direction} "
               f"委托{order.order_volume}股 已成交{order.traded_volume}股 "
               f"状态:{status_text}")
        self.owner._events.append(msg)
        print(msg)

    def on_stock_trade(self, trade):
        msg = (f"[{self._now()}] 成交回报: {trade.stock_code} "
               f"成交{trade.traded_volume}股 价格:{trade.traded_price:.2f} "
               f"金额:{trade.traded_amount:.2f}")
        self.owner._events.append(msg)
        print(msg)

    def on_order_error(self, order_error):
        msg = (f"[{self._now()}] 委托失败: "
               f"编号:{order_error.order_id} {order_error.error_msg}")
        self.owner._events.append(msg)
        print(msg)

    def on_cancel_error(self, cancel_error):
        msg = (f"[{self._now()}] 撤单失败: "
               f"编号:{cancel_error.order_id} {cancel_error.error_msg}")
        self.owner._events.append(msg)
        print(msg)

    def on_order_stock_async_response(self, response):
        msg = (f"[{self._now()}] 异步响应: "
               f"委托编号:{response.order_id} 序列号:{response.seq}")
        self.owner._events.append(msg)


class MiniQMTTrader:
    """
    miniQMT 交易封装类

    用法:
        trader = MiniQMTTrader(qmt_path, account_id)
        trader.connect()

        trader.buy("513100.SH", 100, price=1.00)   # 限价买入
        trader.buy("513100.SH", 100)                # 市价买入
        trader.sell("513100.SH", 100)               # 市价卖出

        positions = trader.query_positions()
        asset = trader.query_asset()

        trader.cancel_all()
        trader.disconnect()

    风控参数:
        max_positions: 最大持仓只数 (默认10)
        max_order_amount: 单笔最大委托金额 (默认500000元)
    """

    def __init__(self, qmt_path, account_id, session_id=None,
                 max_positions=10, max_order_amount=500000):
        self.qmt_path = qmt_path
        self.account_id = account_id
        self.session_id = session_id or int(time.time())
        self.max_positions = max_positions
        self.max_order_amount = max_order_amount

        self._trader = None
        self._account = None
        self._connected = False
        self._events = []

    # ----------------------------------------------------------
    # 连接管理
    # ----------------------------------------------------------

    def connect(self, max_retry=3):
        """连接 miniQMT 并订阅账户"""
        self._trader = XtQuantTrader(self.qmt_path, self.session_id)
        self._trader.register_callback(_TraderCallback(self))
        self._trader.start()

        for i in range(max_retry):
            result = self._trader.connect()
            if result == 0:
                break
            print(f"连接重试 {i + 1}/{max_retry}...")
            time.sleep(2)
        else:
            raise Exception(
                f"连接 miniQMT 失败。请确保客户端已启动，路径: {self.qmt_path}")

        self._account = StockAccount(self.account_id)
        sub_result = self._trader.subscribe(self._account)
        if sub_result != 0:
            raise Exception(f"订阅账户失败，错误码: {sub_result}")

        self._connected = True
        print(f"[MiniQMTTrader] 连接成功 账户:{self.account_id}")
        return True

    def disconnect(self):
        """断开连接"""
        if self._trader:
            self._trader.stop()
            self._connected = False
            print("[MiniQMTTrader] 已断开连接")

    @property
    def connected(self):
        return self._connected

    @property
    def events(self):
        """获取回调事件记录"""
        return list(self._events)

    # ----------------------------------------------------------
    # 风控检查
    # ----------------------------------------------------------

    def _check_risk(self, stock_code, volume, price):
        """
        下单前的风控检查

        检查项:
          1. 持仓只数是否超限
          2. 单笔委托金额是否超限
        """
        positions = self._trader.query_stock_positions(self._account)
        if positions:
            hold_count = sum(1 for p in positions if p.volume > 0)
            if hold_count >= self.max_positions:
                held_codes = {p.stock_code for p in positions if p.volume > 0}
                if stock_code not in held_codes:
                    print(f"[风控] 持仓已达上限 {self.max_positions} 只，拒绝新开仓")
                    return False

        if price > 0:
            order_amount = volume * price
            if order_amount > self.max_order_amount:
                print(f"[风控] 单笔金额 {order_amount:,.0f} 超过上限 "
                      f"{self.max_order_amount:,.0f}，拒绝下单")
                return False

        return True

    # ----------------------------------------------------------
    # 买卖操作
    # ----------------------------------------------------------

    def buy(self, stock_code, volume, price=0, strategy_name='', remark=''):
        """
        买入股票

        参数:
            stock_code: 股票代码, 如 "513100.SH"
            volume: 买入数量 (必须是100的整数倍)
            price: 委托价格, 0表示市价
            strategy_name: 策略名称
            remark: 委托备注

        返回:
            委托编号 (int), 失败返回 None
        """
        if not self._connected:
            print("[MiniQMTTrader] 未连接，请先调用 connect()")
            return None

        if price > 0:
            price_type = xtconstant.FIX_PRICE
        else:
            if stock_code.endswith('.SH'):
                price_type = xtconstant.MARKET_SH_CONVERT_5_LIMIT
            else:
                price_type = xtconstant.MARKET_SZ_CONVERT_5_CANCEL

        if not self._check_risk(stock_code, volume, price):
            return None

        order_id = self._trader.order_stock(
            self._account,
            stock_code,
            xtconstant.STOCK_BUY,
            volume,
            price_type,
            price if price > 0 else 0,
            strategy_name,
            remark
        )

        price_str = f"{price:.2f}" if price > 0 else "市价"
        print(f"[买入] {stock_code} {volume}股 {price_str} 编号:{order_id}")
        return order_id

    def sell(self, stock_code, volume, price=0, strategy_name='', remark=''):
        """
        卖出股票

        参数同 buy()
        """
        if not self._connected:
            print("[MiniQMTTrader] 未连接，请先调用 connect()")
            return None

        if price > 0:
            price_type = xtconstant.FIX_PRICE
        else:
            if stock_code.endswith('.SH'):
                price_type = xtconstant.MARKET_SH_CONVERT_5_LIMIT
            else:
                price_type = xtconstant.MARKET_SZ_CONVERT_5_CANCEL

        order_id = self._trader.order_stock(
            self._account,
            stock_code,
            xtconstant.STOCK_SELL,
            volume,
            price_type,
            price if price > 0 else 0,
            strategy_name,
            remark
        )

        price_str = f"{price:.2f}" if price > 0 else "市价"
        print(f"[卖出] {stock_code} {volume}股 {price_str} 编号:{order_id}")
        return order_id

    # ----------------------------------------------------------
    # 撤单
    # ----------------------------------------------------------

    def cancel(self, order_id):
        """撤销指定委托"""
        if not self._connected:
            return None
        result = self._trader.cancel_order_stock(self._account, order_id)
        print(f"[撤单] 编号:{order_id} 结果:{'成功' if result == 0 else f'失败({result})'}")
        return result

    def cancel_all(self):
        """撤销所有可撤委托"""
        if not self._connected:
            return
        orders = self._trader.query_stock_orders(self._account)
        if not orders:
            print("[撤单] 无委托可撤")
            return

        cancelable = [o for o in orders if o.order_status in (50, 55)]
        if not cancelable:
            print("[撤单] 无可撤委托")
            return

        print(f"[撤单] 批量撤销 {len(cancelable)} 笔委托")
        for order in cancelable:
            self.cancel(order.order_id)
            time.sleep(0.3)

    # ----------------------------------------------------------
    # 查询
    # ----------------------------------------------------------

    def query_asset(self):
        """查询账户资产，返回 dict"""
        if not self._connected:
            return {}
        asset = self._trader.query_stock_asset(self._account)
        if asset is None:
            return {}
        return {
            'total_asset': asset.total_asset,
            'cash': asset.cash,
            'market_value': asset.market_value,
            'frozen_cash': asset.frozen_cash,
        }

    def query_positions(self):
        """查询股票持仓，返回 list[dict]"""
        if not self._connected:
            return []
        positions = self._trader.query_stock_positions(self._account)
        if not positions:
            return []
        return [
            {
                'stock_code': p.stock_code,
                'volume': p.volume,
                'can_use_volume': p.can_use_volume,
                'open_price': p.open_price,
                'market_value': p.market_value,
            }
            for p in positions if p.volume > 0
        ]

    def query_orders(self):
        """查询当日委托，返回 list[dict]"""
        if not self._connected:
            return []
        orders = self._trader.query_stock_orders(self._account)
        if not orders:
            return []
        return [
            {
                'order_id': o.order_id,
                'stock_code': o.stock_code,
                'order_type': o.order_type,
                'order_volume': o.order_volume,
                'price': o.price,
                'traded_volume': o.traded_volume,
                'order_status': o.order_status,
                'order_time': o.order_time,
            }
            for o in orders
        ]

    def query_trades(self):
        """查询当日成交，返回 list[dict]"""
        if not self._connected:
            return []
        trades = self._trader.query_stock_trades(self._account)
        if not trades:
            return []
        return [
            {
                'traded_id': t.traded_id,
                'stock_code': t.stock_code,
                'traded_price': t.traded_price,
                'traded_volume': t.traded_volume,
                'traded_amount': t.traded_amount,
                'order_id': t.order_id,
                'traded_time': t.traded_time,
            }
            for t in trades
        ]


# ============================================================
# 主流程: 演示 MiniQMTTrader 的使用
# ============================================================
if __name__ == "__main__":
    from dotenv import load_dotenv

    print("=" * 60)
    print("示例4: MiniQMTTrader 封装类")
    print("场景: 能不能一行代码搞定?")
    print("=" * 60)

    load_dotenv()
    QMT_PATH = os.getenv("QMT_PATH")
    ACCOUNT_ID = os.getenv("ACCOUNT_ID")

    STOCK_CODE = "513100.SH"
    VOLUME = 100

    # ---- 一行创建 + 一行连接 ----
    trader = MiniQMTTrader(
        qmt_path=QMT_PATH,
        account_id=ACCOUNT_ID,
        max_positions=10,
        max_order_amount=500000,
    )
    trader.connect()

    # ---- 查询资产 ----
    print("\n--- 账户资产 ---")
    asset = trader.query_asset()
    for key, value in asset.items():
        print(f"  {key}: {value:,.2f}")

    # ---- 查询持仓 ----
    print("\n--- 股票持仓 ---")
    positions = trader.query_positions()
    if positions:
        for pos in positions:
            print(f"  {pos['stock_code']}: "
                  f"持仓{pos['volume']} 可用{pos['can_use_volume']} "
                  f"成本{pos['open_price']:.2f}")
    else:
        print("  无持仓")

    # ---- 下单示例 ----
    print(f"\n--- 下单: {STOCK_CODE} ---")
    order_id = trader.buy(STOCK_CODE, VOLUME, price=0.50,
                          strategy_name='demo', remark='封装类演示')
    time.sleep(2)

    # ---- 查看委托 ----
    print("\n--- 当日委托 ---")
    orders = trader.query_orders()
    for o in orders:
        print(f"  {o['stock_code']} 委托{o['order_volume']}股 "
              f"成交{o['traded_volume']}股 状态:{o['order_status']}")

    # ---- 撤单 ----
    if order_id:
        print("\n--- 撤单 ---")
        trader.cancel(order_id)
        time.sleep(2)

    trader.cancel_all()

    # ---- 事件日志 ----
    print(f"\n--- 事件日志 ({len(trader.events)} 条) ---")
    for event in trader.events:
        print(f"  {event}")

    input("\n按回车键退出...")
    trader.disconnect()
