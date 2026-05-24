# -*- coding: utf-8 -*-
"""
中枢网格 + 趋势联动策略 - 中枢内做网格, 出中枢后跟趋势
====================================================

核心问题:
---------
脚本2的中枢网格在出中枢后就停止交易等待 -- 浪费了趋势行情。
本脚本实现: 中枢内做网格收租, 出中枢后切换为趋势跟踪,
两种模式自动切换, 让策略既赚震荡的钱, 也赚趋势的钱。

模式切换逻辑:
-----------
1. chan.py 识别出中枢 → 进入 GRID 模式 (中枢内做网格)
2. 价格向上突破 ZG → 切换为 TREND_UP 模式 (趋势跟踪, ATR止损)
3. 价格向下跌破 ZD → 切换为 WAIT 模式 (空仓防守)
4. 新中枢形成 → 回到 GRID 模式

这是"自适应策略"的进化版:
  第6讲: ADX判断趋势/震荡 → 切换MACD/RSI
  本讲:  缠论中枢判断趋势/震荡 → 切换网格/趋势跟踪

三种模式详解:
  GRID:     中枢内做网格, 低买高卖, 赚震荡的钱
  TREND_UP: 向上突破中枢, ATR跟踪止损, 赚趋势的钱
  WAIT:     向下跌破中枢或等待新中枢, 空仓防守

运行方式:
  python 3-中枢网格+趋势联动.py
"""
import numpy as np
import talib
import backtrader as bt
from data_loader import (
    load_stock_data, ChanPandasData,
    run_and_report, calc_buy_and_hold,
)
from chanpy_wrapper import run_chan, chan_to_signal_df
from grid_engine import ChanGridEngine
from db_config import INITIAL_CASH


# ============================================================
# 中枢网格 + 趋势联动策略
# ============================================================

class ChanHybridStrategy(bt.Strategy):
    """
    中枢网格 + 趋势联动策略 (核心策略类)。

    三种模式:
      GRID:     中枢内做网格, 低买高卖
      TREND_UP: 向上突破中枢, ATR跟踪止损持仓
      WAIT:     向下跌破中枢或等待新中枢, 空仓防守

    参数:
      num_grids: 中枢内格子数 (默认6)
      capital_ratio: 网格资金比例 (默认80%)
      atr_period: ATR计算周期 (默认14)
      atr_trail_mult: ATR跟踪止损倍数 (默认2.5)
      breakout_confirm: 突破确认幅度 (默认0.5%)
    """
    params = (
        ('num_grids', 6),
        ('capital_ratio', 0.80),
        ('atr_period', 14),           # ATR 计算周期
        ('atr_trail_mult', 2.5),      # ATR 跟踪止损倍数: 2.5倍ATR
        ('breakout_confirm', 0.005),  # 突破确认: 超出ZG 0.5%才算真突破
    )

    def __init__(self):
        self.grid = None                    # ChanGridEngine 实例
        self.order = None                   # 当前挂单
        self.mode = 'WAIT'                  # 当前模式: WAIT/GRID/TREND_UP
        self.current_zg = 0.0               # 当前中枢ZG
        self.current_zd = 0.0               # 当前中枢ZD
        self.trail_stop = 0.0               # ATR跟踪止损价
        self.highest_since_breakout = 0.0   # 突破以来最高价
        self.trend_entry_price = 0.0        # 趋势入场价格
        self.mode_log = []                  # 模式切换日志

    def notify_order(self, order):
        """订单状态回调: 完成后重置挂单标志"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        self.order = None

    def _calc_atr(self):
        """
        计算当前 ATR (平均真实波幅) 值。

        ATR 用于:
          - TREND_UP 模式中设置跟踪止损
          - 确保止损幅度能适应市场波动

        返回值:
            float ATR 值, 或 None (数据不足时)
        """
        size = min(len(self.data), self.p.atr_period + 5)
        if size < self.p.atr_period:
            return None
        high = np.array([self.data.high[-i] for i in range(size, 0, -1)], dtype=float)
        low = np.array([self.data.low[-i] for i in range(size, 0, -1)], dtype=float)
        close = np.array([self.data.close[-i] for i in range(size, 0, -1)], dtype=float)
        atr = talib.ATR(high, low, close, timeperiod=self.p.atr_period)
        if atr is None or np.isnan(atr[-1]):
            return None
        return float(atr[-1])

    def _build_grid(self, zg, zd):
        """
        构建或重建中枢网格。
        """
        grid_capital = self.broker.getvalue() * self.p.capital_ratio
        if self.grid is None:
            self.grid = ChanGridEngine(zg=zg, zd=zd, num_grids=self.p.num_grids,
                                       total_capital=grid_capital)
        else:
            self.grid.switch_zhongshu(zg, zd)
            self.grid.total_capital = grid_capital
            self.grid.capital_per_grid = grid_capital / self.p.num_grids
        self.current_zg = zg
        self.current_zd = zd

    def _switch_mode(self, new_mode):
        """
        记录模式切换日志, 并切换到新模式。

        记录内容包括: 切换日期、旧模式、新模式、当前价格。
        """
        if new_mode != self.mode:
            self.mode_log.append({
                'date': self.data.datetime.date(0),
                'from': self.mode,
                'to': new_mode,
                'price': self.data.close[0],
            })
            self.mode = new_mode

    def next(self):
        """
        每个 Bar 的回调: 核心模式切换和交易逻辑。

        模式切换流程:
          WAIT → (中枢形成) → GRID
          GRID → (向上突破) → TREND_UP
          GRID → (向下跌破) → WAIT
          TREND_UP → (止损/三卖/新中枢) → WAIT 或 GRID
        """
        if self.order:
            return

        zg = self.data.chan_zg[0]
        zd = self.data.chan_zd[0]
        price = self.data.close[0]
        has_zs = (not np.isnan(zg)) and (not np.isnan(zd)) and zg > 0 and zd > 0

        # ==================== WAIT模式 ====================
        # 等待中枢形成, 形成后进入 GRID 模式
        if self.mode == 'WAIT':
            if has_zs:
                zg_changed = abs(zg - self.current_zg) > self.current_zg * 0.001 if self.current_zg > 0 else True
                zd_changed = abs(zd - self.current_zd) > self.current_zd * 0.001 if self.current_zd > 0 else True
                if zg_changed or zd_changed:
                    self._build_grid(zg, zd)
                    self._switch_mode('GRID')
            return

        # ==================== GRID模式 ====================
        # 在中枢内做网格, 突破中枢则切换模式
        if self.mode == 'GRID':
            if not has_zs:
                # 中枢消失 → 平仓, 回到 WAIT
                if self.position.size > 0:
                    self.order = self.close()
                if self.grid:
                    self.grid.deactivate()
                self._switch_mode('WAIT')
                return

            # 检查中枢是否切换
            zg_changed = abs(zg - self.current_zg) > self.current_zg * 0.001 if self.current_zg > 0 else False
            zd_changed = abs(zd - self.current_zd) > self.current_zd * 0.001 if self.current_zd > 0 else False
            if zg_changed or zd_changed:
                if self.position.size > 0:
                    self.order = self.close()
                    return
                self._build_grid(zg, zd)

            # ---- 向上突破 ZG → TREND_UP 模式 ----
            if price > self.current_zg * (1 + self.p.breakout_confirm):
                if self.position.size > 0:
                    self.order = self.close()
                    return
                if self.grid:
                    self.grid.deactivate()
                # 趋势买入: 用50%资金
                atr = self._calc_atr()
                if atr and atr > 0:
                    size = int((self.broker.getvalue() * 0.5) / price // 100) * 100
                    if size >= 100:
                        self.order = self.buy(size=size)
                        self.trend_entry_price = price
                        self.highest_since_breakout = price
                        # 初始止损 = 入场价 - 2.5倍ATR
                        self.trail_stop = price - self.p.atr_trail_mult * atr
                        self._switch_mode('TREND_UP')
                return

            # ---- 向下跌破 ZD → WAIT 模式 ----
            if price < self.current_zd * (1 - self.p.breakout_confirm):
                if self.position.size > 0:
                    self.order = self.close()
                if self.grid:
                    self.grid.deactivate()
                self._switch_mode('WAIT')
                return

            # ---- 正常网格交易 ----
            if self.grid and self.grid.active:
                signals = self.grid.update(price)
                for sig in signals:
                    if sig['action'] == 'BUY':
                        cash = self.broker.getcash()
                        if cash >= price * sig['size'] * 1.01 and sig['size'] > 0:
                            self.order = self.buy(size=sig['size'])
                            break
                    elif sig['action'] == 'SELL':
                        if self.position.size >= sig['size']:
                            self.order = self.sell(size=sig['size'])
                            break
            return

        # ==================== TREND_UP模式 ====================
        # 趋势跟踪: ATR动态止损, 三卖离场, 新中枢回到网格
        if self.mode == 'TREND_UP':
            # 更新突破以来的最高价
            if price > self.highest_since_breakout:
                self.highest_since_breakout = price

            # 动态调整止损: ATR跟踪止损
            atr = self._calc_atr()
            if atr and atr > 0:
                new_stop = self.highest_since_breakout - self.p.atr_trail_mult * atr
                self.trail_stop = max(self.trail_stop, new_stop)  # 只上移不下移

            # ATR跟踪止损触发
            if price < self.trail_stop:
                self.order = self.close()
                self._switch_mode('WAIT')
                return

            # 缠论三卖信号离场
            if self.data.chan_signal[0] == -3:
                self.order = self.close()
                self._switch_mode('WAIT')
                return

            # 检查是否进入新中枢 → 回到 GRID 模式
            zg_changed = abs(zg - self.current_zg) > self.current_zg * 0.001 if self.current_zg > 0 else False
            zd_changed = abs(zd - self.current_zd) > self.current_zd * 0.001 if self.current_zd > 0 else False
            if has_zs and (zg_changed or zd_changed):
                if self.position.size > 0:
                    self.order = self.close()
                    return
                self._build_grid(zg, zd)
                self._switch_mode('GRID')
            return

    def stop(self):
        """
        回测结束时打印网格运行报告和模式切换日志。
        """
        if self.grid:
            self.grid.summary()
        if self.mode_log:
            print(f"  模式切换: {len(self.mode_log)}次")
            for m in self.mode_log[:10]:
                print(f"    {m['date']} | {m['from']:>8} -> {m['to']:<8} | 价格={m['price']:.2f}")
            if len(self.mode_log) > 10:
                print(f"    ... 还有 {len(self.mode_log) - 10} 次切换")


# ============================================================
# 纯中枢网格 (对照组, 脚本2的策略)
# ============================================================

class PureChanGridStrategy(bt.Strategy):
    """
    纯中枢网格策略(对照组), 出中枢就停止。
    逻辑与脚本2的 ChanGridStrategy 完全相同。
    """
    params = (('num_grids', 6), ('capital_ratio', 0.80),)

    def __init__(self):
        self.grid = None
        self.order = None
        self.current_zg = 0.0
        self.current_zd = 0.0
        self.mode = 'WAIT'

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        self.order = None

    def next(self):
        if self.order:
            return
        zg, zd = self.data.chan_zg[0], self.data.chan_zd[0]
        price = self.data.close[0]
        has_zs = (not np.isnan(zg)) and (not np.isnan(zd)) and zg > 0 and zd > 0

        if not has_zs:
            if self.mode == 'GRID' and self.position.size > 0:
                self.order = self.close()
            if self.grid:
                self.grid.deactivate()
            self.mode = 'WAIT'
            return

        zg_c = abs(zg - self.current_zg) > self.current_zg * 0.001 if self.current_zg > 0 else True
        zd_c = abs(zd - self.current_zd) > self.current_zd * 0.001 if self.current_zd > 0 else True

        if zg_c or zd_c:
            if self.position.size > 0:
                self.order = self.close()
                return
            cap = self.broker.getvalue() * self.p.capital_ratio
            if self.grid is None:
                self.grid = ChanGridEngine(zg, zd, self.p.num_grids, cap)
            else:
                self.grid.switch_zhongshu(zg, zd)
                self.grid.total_capital = cap
                self.grid.capital_per_grid = cap / self.p.num_grids
            self.current_zg, self.current_zd = zg, zd
            self.mode = 'GRID'

        if self.mode != 'GRID' or not self.grid or not self.grid.active:
            return

        if price > self.current_zg * 1.005 or price < self.current_zd * 0.995:
            if self.position.size > 0:
                self.order = self.close()
            self.grid.deactivate()
            self.mode = 'WAIT'
            return

        signals = self.grid.update(price)
        for sig in signals:
            if sig['action'] == 'BUY':
                if self.broker.getcash() >= price * sig['size'] * 1.01 and sig['size'] > 0:
                    self.order = self.buy(size=sig['size'])
                    break
            elif sig['action'] == 'SELL':
                if self.position.size >= sig['size']:
                    self.order = self.sell(size=sig['size'])
                    break


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    start_date = '2023-01-01'
    end_date = '2025-12-31'

    print("=" * 70)
    print("中枢网格 + 趋势联动策略")
    print("=" * 70)
    print("\n模式切换:")
    print("  GRID:     中枢内做网格 (ZD <= 价格 <= ZG)")
    print("  TREND_UP: 向上突破ZG, ATR跟踪止损, 赚趋势钱")
    print("  WAIT:     向下跌破ZD或等待新中枢, 空仓防守")

    stocks = [
        ('600519.SH', '贵州茅台'),
        ('688981.SH', '中芯国际'),
        ('000001.SZ', '平安银行'),
        ('159941.SZ', '纳指ETF'),
    ]

    all_results = {}

    for code, name in stocks:
        try:
            print(f"\n{'=' * 70}")
            print(f"{name}({code})")
            print(f"{'=' * 70}")

            df = load_stock_data(code, start_date, end_date)
            chan_data = run_chan(df, symbol=code)
            signal_df = chan_to_signal_df(df, chan_data)

            zs_count = len(chan_data['zs_list'])
            bsp_count = len(chan_data['bsp_list'])
            print(f"  chan.py: {len(chan_data['bi_list'])}笔, {zs_count}个中枢, {bsp_count}个买卖点")

            # 纯中枢网格 (对照组)
            print(f"\n  [纯中枢网格]")
            r_pure = run_and_report(
                PureChanGridStrategy, stock_code=code,
                label=f'{name}-纯网格', plot=True, use_sizer=False,
                df=signal_df, data_class=ChanPandasData,
            )

            # 混合策略 (中枢网格 + 趋势联动)
            print(f"\n  [中枢网格+趋势联动]")
            r_hybrid = run_and_report(
                ChanHybridStrategy, stock_code=code,
                label=f'{name}-混合策略', plot=True, use_sizer=False,
                df=signal_df, data_class=ChanPandasData,
            )

            bh = calc_buy_and_hold(code, start_date, end_date)

            all_results[code] = {
                'name': name, 'pure': r_pure, 'hybrid': r_hybrid,
                'bh': bh, 'zs_count': zs_count,
            }

        except Exception as e:
            print(f"  跳过 - {e}")

    # ---- 汇总对比 ----
    if all_results:
        print(f"\n{'=' * 70}")
        print("三策略汇总对比")
        print(f"{'=' * 70}")

        for code, d in all_results.items():
            bh_val = (d['bh'] or 0) * 100
            print(f"\n  {d['name']}({code}) - {d['zs_count']}个中枢:")
            print(f"  {'指标':<12} {'买入持有':>12} {'纯中枢网格':>12} {'网格+趋势':>12}")
            print(f"  {'-' * 52}")
            print(f"  {'总收益':<12} {bh_val:>+11.2f}% {d['pure']['total_return']*100:>+11.2f}% {d['hybrid']['total_return']*100:>+11.2f}%")
            print(f"  {'最大回撤':<12} {'--':>12} {d['pure']['max_drawdown']*100:>11.2f}% {d['hybrid']['max_drawdown']*100:>11.2f}%")
            print(f"  {'夏普比率':<12} {'--':>12} {d['pure']['sharpe_ratio']:>12.2f} {d['hybrid']['sharpe_ratio']:>12.2f}")
            print(f"  {'交易次数':<12} {'--':>12} {d['pure']['total_trades']:>12d} {d['hybrid']['total_trades']:>12d}")
            print(f"  {'胜率':<12} {'--':>12} {d['pure']['win_rate']*100:>11.1f}% {d['hybrid']['win_rate']*100:>11.1f}%")

    print("\n关键发现:")
    print("  - 混合策略 = 纯网格的低风险 + 趋势跟踪的高收益")
    print("  - 中枢内: 网格收租, 稳定赚差价")
    print("  - 出中枢: ATR跟踪止损, 不错过趋势行情")
    print("  - 跌破中枢: 空仓防守, 回避下跌风险")
    print("  - 模式切换由缠论结构驱动, 不是人为判断")
