# -*- coding: utf-8 -*-
"""
第09讲：缠论精华量化
脚本6：缠论 + 量价增强策略

教学目标:
  - 在缠论三买的基础上，改进出场逻辑提升策略表现
  - 对比"固定止盈止损" vs "动态跟踪止损"的效果
  - 理解出场管理对策略收益的关键影响

核心认知:
  入场决定"能不能赚钱", 出场决定"赚多少"。
  很多策略胜率不低但最终亏损, 原因就是出场管理太差——赚小亏大。

策略对比:

  基础策略 (ChanBasicStrategy):
    入场: 三买信号 → 买入
    止损: 固定 ZG 止损 (或 7% 固定止损)
    止盈: 固定 15%
    问题: 遇到大行情时过早止盈, 错过主升浪

  增强策略 (ChanTrailingStopStrategy):
    入场: 三买信号 → 买入
    止损: 阶梯式动态调整
      - 初始: ZG 止损 (或 7% 固定)
      - 盈利 > 5%: 止损上移至成本价 (保本止损)
      - 盈利 > 10%: 止损上移至锁定 5% 利润
      - ATR 跟踪止损: 最高价 - 2.5倍ATR (动态跟随)
    止盈: 无固定止盈, 由跟踪止损决定
    分析: 让利润奔跑, 截断亏损

  出场管理的核心原则:
    不要因为"已经赚了XX%"就卖出,
    而要因为"趋势结束了"才卖出。
    跟踪止损是实现这一理念的利器。

ATR (Average True Range, 平均真实波幅):
  衡量市场波动率的指标。
  ATR 大 → 市场活跃, 允许更大的价格回撤
  ATR 小 → 市场平静, 应该收紧止损
  使用 ATR 跟踪止损的好处是"自适应"——波动大时止损宽, 波动小时止损窄。
"""

import backtrader as bt
import numpy as np
import talib
from data_loader import (
    load_stock_data, ChanPandasData,
    run_and_report, calc_buy_and_hold,
)
from chan_analyzer import ChanAnalyzer

# ============================================================
# 参数配置
# ============================================================

STOCK_CODE = '300782.SZ'
START_DATE = '2024-01-01'
END_DATE = '2025-06-01'


# ============================================================
# 策略定义
# ============================================================

class ChanBasicStrategy(bt.Strategy):
    """
    基础缠论三买策略: 固定止盈止损

    作为对照组的"基线"策略。
    出场逻辑简单但粗暴:
      - 跌到 ZG 以下 → 止损
      - 涨 15% → 止盈
      - 出现三卖 → 离场
    """

    params = (
        ('take_profit_pct', 0.15),
    )

    def __init__(self):
        self.entry_price = None
        self.stop_price = None
        self.order = None

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.entry_price = order.executed.price
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.data.chan_signal[0] == 3:
                self.order = self.buy()
                # 止损 = ZG, 如果ZG不可用则用93%兜底
                self.stop_price = self.data.chan_zg[0] if self.data.chan_zg[0] > 0 else self.data.close[0] * 0.93
        else:
            current_price = self.data.close[0]
            if self.stop_price and current_price < self.stop_price:
                self.order = self.close()
                return
            if self.entry_price and (current_price / self.entry_price - 1) >= self.p.take_profit_pct:
                self.order = self.close()
                return
            if self.data.chan_signal[0] == -3:
                self.order = self.close()


class ChanTrailingStopStrategy(bt.Strategy):
    """
    增强策略: 阶梯式跟踪止损 + ATR 动态管理

    核心理念:
      1. 保本止损: 涨起来后, 先把止损移到成本价 → "至少不亏"
      2. 阶梯锁定: 涨更多后, 逐步锁定利润 → "赚到的要留住"
      3. ATR跟踪: 用波动率动态调整止损距离 → "自适应市场"

    阶梯规则:
      盈利 < 5%:    止损 = ZG (初始)
      盈利 >= 5%:   止损 = 成本价 (保本)
      盈利 >= 10%:  止损 = 成本 + 5% (锁定部分利润)
      以上 + ATR: if 最高价 - 2.5*ATR > 当前止损, 用ATR止损

    ATR参数说明:
      atr_period=14:   计算14日ATR (常用参数)
      atr_exit_mult=2.5: 止损 = 最高价 - 2.5 * ATR
        2.5倍的意义: 正常情况下日波动在1倍ATR内,
        2.5倍允许较大的异常波动, 避免被噪音震出。
    """

    params = (
        ('atr_period', 14),           # ATR 计算周期
        ('atr_exit_mult', 2.5),       # ATR 止损倍数
        ('breakeven_pct', 0.05),      # 保本止损触发阈值 (盈利5%)
        ('lock_profit_pct', 0.10),    # 锁定利润触发阈值 (盈利10%)
        ('lock_amount_pct', 0.05),    # 锁定利润量 (5%)
    )

    def __init__(self):
        self.entry_price = None
        self.stop_price = None
        self.highest_since_entry = None  # 入场后的最高价 (用于跟踪止损)
        self.order = None

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.entry_price = order.executed.price
                self.highest_since_entry = order.executed.price
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def _calc_atr(self):
        """
        计算 ATR (Average True Range)

        使用 TA-Lib 实现:
          TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
          ATR = SMA(TR, period)

        为什么需要足够的缓存数据 (period + 5):
          ATR 计算需要前期数据预热, 否则结果是 NaN。
        """
        size = min(len(self.data), self.p.atr_period + 5)
        if size < self.p.atr_period:
            return None
        high_arr = np.array([self.data.high[-i] for i in range(size, 0, -1)], dtype=float)
        low_arr = np.array([self.data.low[-i] for i in range(size, 0, -1)], dtype=float)
        close_arr = np.array([self.data.close[-i] for i in range(size, 0, -1)], dtype=float)
        atr = talib.ATR(high_arr, low_arr, close_arr, timeperiod=self.p.atr_period)
        if atr is None or np.isnan(atr[-1]):
            return None
        return float(atr[-1])

    def next(self):
        if self.order:
            return

        if not self.position:
            # 入场逻辑 (与基础策略相同)
            if self.data.chan_signal[0] == 3:
                self.order = self.buy()
                zg_val = self.data.chan_zg[0]
                self.stop_price = zg_val if zg_val > 0 else self.data.close[0] * 0.93
        else:
            current_price = self.data.close[0]

            # 更新最高价记录
            if current_price > self.highest_since_entry:
                self.highest_since_entry = current_price

            profit_pct = (current_price / self.entry_price) - 1 if self.entry_price else 0

            # ===== 阶梯式止损调整 =====
            # 盈利 >= 10%: 止损锁定 5% 利润
            if profit_pct >= self.p.lock_profit_pct:
                new_stop = self.entry_price * (1 + self.p.lock_amount_pct)
                self.stop_price = max(self.stop_price or 0, new_stop)
            # 盈利 >= 5%: 止损移到成本价 (至少不亏)
            elif profit_pct >= self.p.breakeven_pct:
                new_stop = self.entry_price
                self.stop_price = max(self.stop_price or 0, new_stop)

            # ===== ATR 跟踪止损 =====
            # 用"最高价 - ATR倍数"作为动态止损, 跟随股价上行
            atr = self._calc_atr()
            if atr and self.highest_since_entry:
                atr_stop = self.highest_since_entry - atr * self.p.atr_exit_mult
                if atr_stop > (self.stop_price or 0):
                    self.stop_price = atr_stop

            # ===== 止损触发检查 =====
            if self.stop_price and current_price < self.stop_price:
                self.order = self.close()
                return

            # ===== 三卖信号离场 =====
            if self.data.chan_signal[0] == -3:
                self.order = self.close()


# ============================================================
# 主逻辑
# ============================================================

def main():
    print("=" * 60)
    print("第09讲 | 脚本6: 缠论 + 量价增强策略")
    print("=" * 60)

    # 1. 加载数据
    print(f"\n[1] 加载 {STOCK_CODE} 日线数据...")
    df = load_stock_data(STOCK_CODE, START_DATE, END_DATE)
    print(f"    共 {len(df)} 根K线")

    # 2. 缠论分析
    print("\n[2] 执行缠论分析...")
    analyzer = ChanAnalyzer(df)
    analyzer.analyze()

    signal_df = analyzer.get_signal_df()
    third_buy_count = (signal_df['chan_signal'] == 3).sum()
    print(f"    三买信号: {third_buy_count} 个")

    if third_buy_count == 0:
        print("    没有三买信号，无法回测。")
        return

    # 3. 回测: 基础策略（固定止盈止损）
    print(f"\n[3] 回测: 基础策略 (ZG止损 + 15%止盈)")
    result_basic = run_and_report(
        ChanBasicStrategy,
        stock_code=STOCK_CODE,
        label='基础-固定止盈',
        plot=True,
        df=signal_df,
        data_class=ChanPandasData,
    )

    # 4. 回测: 增强策略（跟踪止损）
    print(f"\n[4] 回测: 增强策略 (阶梯止损 + ATR跟踪)")
    result_enhanced = run_and_report(
        ChanTrailingStopStrategy,
        stock_code=STOCK_CODE,
        label='缠论+量价增强-跟踪止损',
        plot=True,
        df=signal_df,
        data_class=ChanPandasData,
    )

    # 5. 对比汇总
    bh_return = calc_buy_and_hold(STOCK_CODE, START_DATE, END_DATE)

    print(f"\n{'=' * 60}")
    print(f"[5] 策略对比汇总")
    print(f"{'=' * 60}")
    print(f"\n    {'指标':>12} | {'基础策略':>12} | {'增强策略':>12} | {'买入持有':>12}")
    print(f"    {'-'*60}")

    metrics = [
        ('总收益', 'total_return', '%', 100),
        ('年化收益', 'annual_return', '%', 100),
        ('最大回撤', 'max_drawdown', '%', 100),
        ('夏普比率', 'sharpe_ratio', '', 1),
        ('胜率', 'win_rate', '%', 100),
        ('盈亏比', 'profit_loss_ratio', '', 1),
        ('交易次数', 'total_trades', '', 1),
    ]

    for label, key, unit, mult in metrics:
        v1 = result_basic.get(key, 0) * mult
        v2 = result_enhanced.get(key, 0) * mult
        fmt = f"+.2f" if key in ['total_return', 'annual_return'] else '.2f'
        print(f"    {label:>12} | {v1:{fmt}}{unit:>2} | {v2:{fmt}}{unit:>2} |", end='')
        if key == 'total_return' and bh_return is not None:
            print(f" {bh_return*100:+.2f}%")
        else:
            print(f" {'':>12}")

    print(f"\n    增强策略改进点:")
    print(f"      - 盈利>5%: 止损移至成本价（保本止损）")
    print(f"      - 盈利>10%: 止损锁定5%利润")
    print(f"      - ATR跟踪: 最高价 - 2.5倍ATR 作为动态止损")
    print(f"      - 三卖信号也会触发离场")

    print("\n完成!")


if __name__ == '__main__':
    main()
