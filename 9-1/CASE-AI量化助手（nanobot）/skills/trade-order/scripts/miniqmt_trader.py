# -*- coding: utf-8 -*-
"""
MiniQMTTrader 封装类

将 miniQMT 的连接、下单、撤单、查询、回调能力封装为可复用的交易工具类。
从 CASE-XtQuant实盘交易/4-miniqmt_trader.py 复制而来，供 trade-order 技能内部使用。
"""
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

    def on_stock_trade(self, trade):
        msg = (f"[{self._now()}] 成交回报: {trade.stock_code} "
               f"成交{trade.traded_volume}股 价格:{trade.traded_price:.2f} "
               f"金额:{trade.traded_amount:.2f}")
        self.owner._events.append(msg)

    def on_order_error(self, order_error):
        msg = (f"[{self._now()}] 委托失败: "
               f"编号:{order_error.order_id} {order_error.error_msg}")
        self.owner._events.append(msg)

    def on_cancel_error(self, cancel_error):
        msg = (f"[{self._now()}] 撤单失败: "
               f"编号:{cancel_error.order_id} {cancel_error.error_msg}")
        self.owner._events.append(msg)

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
        trader.buy("513100.SH", 100, price=1.00)
        trader.sell("513100.SH", 100)
        trader.disconnect()
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

    def connect(self, max_retry=3):
        """连接 miniQMT 并订阅账户"""
        self._trader = XtQuantTrader(self.qmt_path, self.session_id)
        self._trader.register_callback(_TraderCallback(self))
        self._trader.start()

        for i in range(max_retry):
            result = self._trader.connect()
            if result == 0:
                break
            time.sleep(2)
        else:
            raise Exception(
                f"连接 miniQMT 失败。请确保客户端已启动，路径: {self.qmt_path}")

        self._account = StockAccount(self.account_id)
        sub_result = self._trader.subscribe(self._account)
        if sub_result != 0:
            raise Exception(f"订阅账户失败，错误码: {sub_result}")

        self._connected = True
        return True

    def disconnect(self):
        """断开连接"""
        if self._trader:
            self._trader.stop()
            self._connected = False

    @property
    def connected(self):
        return self._connected

    @property
    def events(self):
        """获取回调事件记录"""
        return list(self._events)

    def _check_risk(self, stock_code, volume, price):
        """下单前的风控检查: 持仓只数、单笔金额"""
        positions = self._trader.query_stock_positions(self._account)
        if positions:
            hold_count = sum(1 for p in positions if p.volume > 0)
            if hold_count >= self.max_positions:
                held_codes = {p.stock_code for p in positions if p.volume > 0}
                if stock_code not in held_codes:
                    return False, "持仓已达上限，拒绝新开仓"

        if price > 0:
            order_amount = volume * price
            if order_amount > self.max_order_amount:
                return False, f"单笔金额 {order_amount:,.0f} 超过上限 {self.max_order_amount:,.0f}"

        return True, ""

    def buy(self, stock_code, volume, price=0, strategy_name='', remark=''):
        """买入股票，返回 (order_id, message)"""
        if not self._connected:
            return None, "未连接，请先调用 connect()"

        if price > 0:
            price_type = xtconstant.FIX_PRICE
        else:
            if stock_code.endswith('.SH'):
                price_type = xtconstant.MARKET_SH_CONVERT_5_LIMIT
            else:
                price_type = xtconstant.MARKET_SZ_CONVERT_5_CANCEL

        ok, err_msg = self._check_risk(stock_code, volume, price)
        if not ok:
            return None, err_msg

        order_id = self._trader.order_stock(
            self._account, stock_code, xtconstant.STOCK_BUY,
            volume, price_type, price if price > 0 else 0,
            strategy_name, remark
        )

        price_str = f"{price:.2f}" if price > 0 else "市价"
        return order_id, f"买入 {stock_code} {volume}股 {price_str} 编号:{order_id}"

    def sell(self, stock_code, volume, price=0, strategy_name='', remark=''):
        """卖出股票，返回 (order_id, message)"""
        if not self._connected:
            return None, "未连接，请先调用 connect()"

        if price > 0:
            price_type = xtconstant.FIX_PRICE
        else:
            if stock_code.endswith('.SH'):
                price_type = xtconstant.MARKET_SH_CONVERT_5_LIMIT
            else:
                price_type = xtconstant.MARKET_SZ_CONVERT_5_CANCEL

        order_id = self._trader.order_stock(
            self._account, stock_code, xtconstant.STOCK_SELL,
            volume, price_type, price if price > 0 else 0,
            strategy_name, remark
        )

        price_str = f"{price:.2f}" if price > 0 else "市价"
        return order_id, f"卖出 {stock_code} {volume}股 {price_str} 编号:{order_id}"

    def cancel(self, order_id):
        """撤销指定委托，返回 (result_code, message)"""
        if not self._connected:
            return -1, "未连接"
        result = self._trader.cancel_order_stock(self._account, order_id)
        status = "成功" if result == 0 else f"失败({result})"
        return result, f"撤单 编号:{order_id} {status}"

    def cancel_all(self):
        """撤销所有可撤委托"""
        if not self._connected:
            return
        orders = self._trader.query_stock_orders(self._account)
        if not orders:
            return
        cancelable = [o for o in orders if o.order_status in (50, 55)]
        for order in cancelable:
            self.cancel(order.order_id)
            time.sleep(0.3)

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
        """查询持仓，返回 list[dict]"""
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
