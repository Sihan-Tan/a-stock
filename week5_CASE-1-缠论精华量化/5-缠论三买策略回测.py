# -*- coding: utf-8 -*-
"""
第09讲：缠论精华量化
脚本5：缠论三买策略回测

教学目标:
  - 将缠论第三类买点信号接入 Backtrader 进行策略回测
  - 设计合理的止损止盈规则
  - 与买入持有进行对比

策略逻辑:

  入场: 缠论三买信号触发
    三买 = 股价突破中枢 ZG 后回踩, 但不跌回中枢内部
    这是缠论中最具实战价值的信号

  止损: 价格跌回中枢 (收盘价 < ZG)
    如果三买判断失误, 股价重新跌回中枢 → 说明突破是假突破 → 必须离场
    这是缠论逻辑自洽的止损位: ZG 就是三买的"生命线"

  止盈: 固定比例止盈 (默认15%)
    或者出现三卖信号离场

数据流:
  1. 加载原始数据 → ChanAnalyzer 分析 → 生成信号列 (get_signal_df)
  2. 信号列合并到 DataFrame → 通过 ChanPandasData 送入 Backtrader
  3. 策略在 next() 中逐日检查 chan_signal 线执行交易

与之前脚本的区别:
  之前的脚本只是检测信号, 本脚本真正执行交易并评估绩效。
  这是一个完整的"从分析到交易"的流程闭环。

策略类说明:
  ChanThirdBuyStrategy 包含三个关键变量:
    - entry_price: 入场价格 (用于计算止盈)
    - stop_price: 止损价格 (ZG 或 93% 入场价兜底)
    - order: 当前挂单 (防止重复下单)

风险控制:
  - 每次只挂一个订单, 有未完成订单时不发新单
  - 兜底止损: 如果 ZG 不可用, 用入场价 93% 作为止损
"""

import backtrader as bt
from data_loader import (
    load_stock_data, ChanPandasData,
    run_and_report, calc_buy_and_hold,
)
from chan_analyzer import ChanAnalyzer

# ============================================================
# 参数配置
# ============================================================

# 使用3年数据以确保足够的交易样本
STOCK_CODE = '600519.SH'
START_DATE = '2023-01-01'
END_DATE = '2025-12-31'


# ============================================================
# 策略定义
# ============================================================

class ChanThirdBuyStrategy(bt.Strategy):
    """
    缠论第三类买点策略

    核心逻辑:
      1. 空仓时: 检查 chan_signal == 3 (三买信号) → 买入
      2. 持仓时:
         - 止损: 收盘价 < 止损价 (ZG 或兜底)
         - 止盈: 涨幅 >= 15%
         - 三卖信号离场: chan_signal == -3
    """

    params = (
        ('take_profit_pct', 0.15),    # 止盈比例: 15%
        ('use_chan_stop', True),      # 使用缠论止损 (ZG)
    )

    def __init__(self):
        self.entry_price = None  # 入场价格 (成交后记录)
        self.stop_price = None   # 止损价格
        self.order = None        # 当前订单 (有单时不允许重复下单)

    def notify_order(self, order):
        """
        订单状态回调

        Completed: 成交, 记录入场价并清空订单
        Canceled/Margin/Rejected: 失败, 清空订单以待下次重试
        """
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def next(self):
        """每个交易日执行一次"""
        if self.order:
            return  # 有未完成订单, 跳过

        if not self.position:
            # ===== 入场条件: 三买信号 (chan_signal == 3) =====
            if self.data.chan_signal[0] == 3:
                self.order = self.buy()
                zg_val = self.data.chan_zg[0]  # 获取当前中枢的ZG
                if self.p.use_chan_stop and zg_val > 0:
                    self.stop_price = zg_val      # 止损 = ZG (缠论逻辑: 跌回中枢即失败)
                else:
                    self.stop_price = self.data.close[0] * 0.93  # 兜底: 7%止损
        else:
            # ===== 持仓管理 =====
            current_price = self.data.close[0]

            # 止损: 价格跌回中枢 (ZG)
            if self.stop_price and current_price < self.stop_price:
                self.order = self.close()
                return

            # 止盈: 达到目标涨幅
            if self.entry_price:
                profit_pct = (current_price / self.entry_price) - 1
                if profit_pct >= self.p.take_profit_pct:
                    self.order = self.close()
                    return

            # 三卖信号离场: 即使没触发止损止盈, 出现三卖也要走
            if self.data.chan_signal[0] == -3:
                self.order = self.close()
                return


# ============================================================
# 主逻辑
# ============================================================

def main():
    print("=" * 60)
    print("第09讲 | 脚本5: 缠论三买策略回测")
    print("=" * 60)

    # 1. 加载数据
    print(f"\n[1] 加载 {STOCK_CODE} 日线数据...")
    df = load_stock_data(STOCK_CODE, START_DATE, END_DATE)
    print(f"    共 {len(df)} 根K线")

    # 2. 缠论分析 + 生成信号列
    # get_signal_df() 将信号列表映射为逐日标记的DataFrame
    # 包含 chan_signal/chan_zg/chan_zd 列
    print("\n[2] 执行缠论分析...")
    analyzer = ChanAnalyzer(df)
    analyzer.analyze()
    analyzer.summary()

    signal_df = analyzer.get_signal_df()
    third_buy_count = (signal_df['chan_signal'] == 3).sum()
    third_sell_count = (signal_df['chan_signal'] == -3).sum()
    print(f"\n    信号统计: 三买={third_buy_count}, 三卖={third_sell_count}")

    if third_buy_count == 0:
        print("    没有三买信号，无法回测。请尝试更长的时间区间或其他股票。")
        return

    # 3. 运行回测
    # 使用 run_and_report 一站式完成: 配置→运行→评估→绘图
    # 注意传入 data_class=ChanPandasData, 框架才会识别 chan_signal 等额外数据线
    print(f"\n[3] 运行回测...")
    result = run_and_report(
        ChanThirdBuyStrategy,
        stock_code=STOCK_CODE,
        label='缠论三买策略',
        plot=True,
        df=signal_df,
        data_class=ChanPandasData,
    )

    # 4. 对比买入持有
    # 策略必须跑赢简单的买入持有才有存在价值
    bh_return = calc_buy_and_hold(STOCK_CODE, START_DATE, END_DATE)
    print(f"\n[4] 策略 vs 买入持有:")
    print(f"    策略收益:   {result['total_return']*100:+.2f}%")
    if bh_return is not None:
        print(f"    买入持有:   {bh_return*100:+.2f}%")
        excess = result['total_return'] - bh_return
        print(f"    超额收益:   {excess*100:+.2f}%")

    # 5. 交易明细
    # 展示每笔交易的具体信息
    trades = result.get('trades', [])
    if trades:
        print(f"\n[5] 交易明细 ({len(trades)} 笔):")
        print(f"    {'日期':>12} | {'操作':>4} | {'价格':>8} | {'数量':>6}")
        print("    " + "-" * 45)
        for t in trades:
            print(f"    {t['date']} | {t['type']:>4} | {t['price']:>8.2f} | {t['size']:>6}")

    print("\n完成!")


if __name__ == '__main__':
    main()
