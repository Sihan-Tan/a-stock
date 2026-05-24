# -*- coding: utf-8 -*-
"""
第09讲：缠论精华量化
脚本7：多周期缠论策略

教学目标:
  - 理解"区间套"思想：大周期定方向，小周期找入场
  - 周线确定趋势方向（中枢上移 = 上升趋势）
  - 日线在上升趋势中寻找三买入场点
  - 对比单周期 vs 多周期的策略表现

缠论区间套原理:
  区间套是缠论的核心思想之一, 源自数学中的"闭区间套定理":
    大级别的走势由小级别走势构成, 小级别走势又由更小级别构成。
    通过多级别联立分析, 可以精确定位买卖点。

  本策略的区间套实现:
    周线 (大级别): 判断趋势方向 → 解决"该不该做多"
    日线 (小级别): 寻找具体入场点 → 解决"什么时候买"

策略逻辑:

  周线分析 - 中枢方向判断:
    最近两个周线中枢:
      - 中枢上移 (ZG/ZD 都抬高) → 上升趋势 (trend = 1)
      - 中枢下移 (ZG/ZD 都降低) → 下跌趋势 (trend = -1)
      - 其他 → 震荡 (trend = 0)

  日线交易:
    入场: 周线趋势向上 (trend=1) 且 日线出现三买信号
    为什么: 周线上升趋势中的三买成功率远高于下跌趋势中的三买
    止损: 日线 ZG (跌回中枢)
    止盈: 15% 或 三卖信号 或 周线转空

  为什么多周期比单周期好:
    单周期策略的问题:
      在下跌趋势中也会出现三买信号 (小级别反弹),
      但这些"逆势三买"往往是陷阱, 进去就跌。

    多周期过滤:
      周线趋势向上 → 只做多, 过滤掉逆势信号
      周线趋势向下 → 不交易, 防止抄底被套
      这个简单的过滤就能大幅提升胜率。

  趋势判断的三种方法 (优先级从高到低):
    1. 中枢方向 (>=2个中枢时最可靠)
    2. 价格与中枢位置关系 + 笔方向 (1个中枢时)
    3. E/(E)MA20趋势 (兜底, 没有中枢时使用)
"""

import backtrader as bt
import pandas as pd
import numpy as np
from data_loader import (
    load_stock_data, ChanPandasData,
    run_and_report, calc_buy_and_hold,
)
from chan_analyzer import ChanAnalyzer

# ============================================================
# 参数配置
# ============================================================

STOCK_CODE = '600519.SH'
START_DATE = '2023-01-01'
END_DATE = '2025-12-31'


# ============================================================
# 周线趋势计算
# ============================================================

def calc_weekly_trend(df):
    """
    基于周线缠论分析 + 均线辅助 计算趋势方向

    实现思路:
      将日线数据重采样为周线 (W), 然后在周线上运行 ChanAnalyzer。

    重采样规则:
      open  = 周一开盘价
      high  = 本周最高价
      low   = 本周最低价
      close = 周五收盘价
      volume = 本周总成交量

    趋势判断方法 (三层次, 逐层兜底):
      方法1 (最优): 周线中枢方向
        >=2个中枢: 比较最后两个中枢的 ZG/ZD 变化
        中枢上移 = 上升趋势, 中枢下移 = 下跌趋势

      方法2 (次优): 价格与中枢位置 + 笔方向
        价格在 ZG 之上且笔向上 → 上升
        价格在 ZD 之下且笔向下 → 下跌

      方法3 (兜底): MA20 均线
        价格在 MA20 之上 → 上升
        价格在 MA20 之下 → 下跌

    参数:
        df: 日线 DataFrame

    返回:
        Series: 索引为日期, 值为趋势方向 (1=上升, -1=下跌, 0=震荡)
    """
    # 将日线重采样为周线
    weekly_df = df.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    if len(weekly_df) < 20:
        return pd.Series(0, index=df.index)

    # 在周线上运行缠论分析
    w_analyzer = ChanAnalyzer(weekly_df)
    w_analyzer.analyze()

    print(f"    周线分析: {len(weekly_df)}根周K线, "
          f"{len(w_analyzer.bi_list)}笔, {len(w_analyzer.zhongshu_list)}个中枢")

    weekly_trend = pd.Series(0, index=weekly_df.index)

    # 方法1: 多个中枢时, 按中枢方向判断
    if len(w_analyzer.zhongshu_list) >= 2:
        for i in range(1, len(w_analyzer.zhongshu_list)):
            curr = w_analyzer.zhongshu_list[i]
            prev = w_analyzer.zhongshu_list[i - 1]

            if curr['ZG'] > prev['ZG'] and curr['ZD'] > prev['ZD']:
                trend = 1
            elif curr['ZG'] < prev['ZG'] and curr['ZD'] < prev['ZD']:
                trend = -1
            else:
                trend = 0

            # 从该中枢开始之后都标记为该趋势
            mask = weekly_trend.index >= curr['start_date']
            weekly_trend.loc[mask] = trend

    # 方法2: 有1个中枢时, 用价格与中枢位置 + 笔方向
    if len(w_analyzer.zhongshu_list) >= 1:
        for zs in w_analyzer.zhongshu_list:
            for idx in range(len(weekly_df)):
                date = weekly_df.index[idx]
                if date < zs['start_date']:
                    continue
                close = weekly_df['close'].iloc[idx]
                if weekly_trend.loc[date] != 0:
                    continue
                if close > zs['ZG']:
                    weekly_trend.loc[date] = 1
                elif close < zs['ZD']:
                    weekly_trend.loc[date] = -1

    # 方法3: 兜底 — 用周线 MA20 判断
    ma20 = weekly_df['close'].rolling(20).mean()
    for idx in range(20, len(weekly_df)):
        date = weekly_df.index[idx]
        if weekly_trend.loc[date] != 0:
            continue
        if weekly_df['close'].iloc[idx] > ma20.iloc[idx]:
            weekly_trend.loc[date] = 1
        elif weekly_df['close'].iloc[idx] < ma20.iloc[idx]:
            weekly_trend.loc[date] = -1

    # 将周线趋势映射回日线 (用前向填充)
    daily_trend = weekly_trend.reindex(df.index, method='ffill').fillna(0).astype(int)
    return daily_trend


# ============================================================
# 策略定义
# ============================================================

class ChanSinglePeriodStrategy(bt.Strategy):
    """
    单周期缠论三买策略（对照组）

    与之前脚本5的策略相同, 但不加任何过滤。
    用于与多周期策略进行对比, 证明"周线趋势过滤"的价值。
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


class ChanMultiPeriodStrategy(bt.Strategy):
    """
    多周期缠论策略: 周线趋势 + 日线三买

    入场条件 (两个条件同时满足):
      1. 周线趋势向上: self.data.weekly_trend[0] == 1
      2. 日线三买信号: self.data.chan_signal[0] == 3

    出场逻辑:
      止损: 收盘价 < ZG (与单周期相同)
      止盈: 15% 固定止盈 (与单周期相同)
      三卖信号离场
      新增: 周线趋势转空 → 离场
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
            # 多周期入场条件: 周线趋势向上 + 日线三买
            weekly_up = self.data.weekly_trend[0] == 1
            daily_third_buy = self.data.chan_signal[0] == 3

            if weekly_up and daily_third_buy:
                self.order = self.buy()
                self.stop_price = self.data.chan_zg[0] if self.data.chan_zg[0] > 0 else self.data.close[0] * 0.93
        else:
            current_price = self.data.close[0]

            # 止损
            if self.stop_price and current_price < self.stop_price:
                self.order = self.close()
                return

            # 止盈
            if self.entry_price and (current_price / self.entry_price - 1) >= self.p.take_profit_pct:
                self.order = self.close()
                return

            # 三卖离场
            if self.data.chan_signal[0] == -3:
                self.order = self.close()
                return

            # 多周期新增: 周线转空 → 离场
            if self.data.weekly_trend[0] == -1:
                self.order = self.close()


# ============================================================
# 主逻辑
# ============================================================

def main():
    print("=" * 60)
    print("第09讲 | 脚本7: 多周期缠论策略")
    print("=" * 60)

    # 1. 加载数据
    print(f"\n[1] 加载 {STOCK_CODE} 日线数据 ({START_DATE} ~ {END_DATE})...")
    df = load_stock_data(STOCK_CODE, START_DATE, END_DATE)
    print(f"    共 {len(df)} 根K线")

    # 2. 日线缠论分析
    print("\n[2] 日线缠论分析...")
    daily_analyzer = ChanAnalyzer(df)
    daily_analyzer.analyze()

    signal_df = daily_analyzer.get_signal_df()
    third_buy_count = (signal_df['chan_signal'] == 3).sum()
    print(f"    日线三买信号: {third_buy_count} 个")

    # 3. 周线趋势分析
    print("\n[3] 周线趋势分析...")
    weekly_trend = calc_weekly_trend(df)
    signal_df['weekly_trend'] = weekly_trend  # 将周线趋势写入信号DataFrame

    up_days = (weekly_trend == 1).sum()
    down_days = (weekly_trend == -1).sum()
    flat_days = (weekly_trend == 0).sum()
    print(f"    趋势分布: 上升={up_days}天, 下跌={down_days}天, 震荡={flat_days}天")

    # 统计经过周线过滤后的三买信号数量
    filtered_count = ((signal_df['chan_signal'] == 3) & (signal_df['weekly_trend'] == 1)).sum()
    print(f"    周线上升+日线三买: {filtered_count} 个 (过滤前: {third_buy_count} 个)")

    if third_buy_count == 0:
        print("    没有三买信号，无法回测。")
        return

    # 4. 回测: 单周期策略
    print(f"\n[4] 回测: 单周期缠论三买策略")
    result_single = run_and_report(
        ChanSinglePeriodStrategy,
        stock_code=STOCK_CODE,
        label='单周期三买',
        plot=True,
        df=signal_df,
        data_class=ChanPandasData,
    )

    # 5. 回测: 多周期策略
    print(f"\n[5] 回测: 多周期缠论策略 (周线+日线)")
    result_multi = run_and_report(
        ChanMultiPeriodStrategy,
        stock_code=STOCK_CODE,
        label='多周期三买',
        plot=True,
        df=signal_df,
        data_class=ChanPandasData,
    )

    # 6. 对比汇总
    bh_return = calc_buy_and_hold(STOCK_CODE, START_DATE, END_DATE)

    print(f"\n{'=' * 60}")
    print(f"[6] 策略对比汇总")
    print(f"{'=' * 60}")
    print(f"\n    {'指标':>12} | {'单周期':>12} | {'多周期':>12} | {'买入持有':>12}")
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
        v1 = result_single.get(key, 0) * mult
        v2 = result_multi.get(key, 0) * mult
        fmt = f"+.2f" if key in ['total_return', 'annual_return'] else '.2f'
        print(f"    {label:>12} | {v1:{fmt}}{unit:>2} | {v2:{fmt}}{unit:>2} |", end='')
        if key == 'total_return' and bh_return is not None:
            print(f" {bh_return*100:+.2f}%")
        else:
            print(f" {'':>12}")

    print(f"\n    多周期策略优势:")
    print(f"      - 周线趋势过滤: 只在上升趋势中做多，避免逆势交易")
    print(f"      - 信号质量提升: 从 {third_buy_count} 个三买过滤到 {filtered_count} 个")
    print(f"      - 周线转空自动离场: 趋势反转时及时止损")

    print("\n完成!")


if __name__ == '__main__':
    main()
