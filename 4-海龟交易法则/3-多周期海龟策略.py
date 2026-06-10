# -*- coding: utf-8 -*-
"""
多周期融合海龟策略 - 周线定方向，日线找入场

====================================================================
核心思想: "大周期过滤，小周期执行"
====================================================================

这是交易中最经典的多周期分析框架之一:

  大周期（周线）的作用: 判断主要趋势方向
    - 好比"看森林"，避免陷入短期波动的细节
    - 周线级别的趋势一旦形成，持续时间长，不容易被突破

  小周期（日线）的作用: 寻找精确入场时机
    - 好比"看树木"，在确定的方向上找到最佳时机
    - 日线级别的信号更灵敏，比周线信号出现得更早

  为什么要这样做？
    如果只用日线: 可能逆着大趋势做多，在周线下跌中接飞刀
    如果只用周线: 信号太少，可能一年只交易几次，错过很多机会
    两者结合: 既有大趋势的保护，又有足够的交易机会

====================================================================
周线过滤规则详解
====================================================================

周线的唐奇安通道（8 周）将市场分为三种状态:

  1. 周线趋势向上 (weekly_trend = 'up'):
     周收盘价 > 8 周最高价
     操作: 允许做多（正常交易）

  2. 周线趋势中性 (weekly_trend = 'neutral'):
     周收盘价在 8 周通道内部
     操作: 允许做多（不能太严格，否则错过太多）

  3. 周线趋势向下 (weekly_trend = 'down'):
     周收盘价 < 8 周最低价
     操作: 禁止开新仓；已持仓则平仓（趋势反转保护）

为什么中性状态也允许做多？
  如果只在周线向上时才做多，标准太严格了。
  很多趋势的启动阶段，周线可能刚从低位回升但还没突破通道上轨。
  过于严格的过滤会让我们错过趋势启动的最佳入场点。

====================================================================
Backtrader 多周期实现
====================================================================

Backtrader 支持多数据源策略，实现方式:

  data0: 日线数据，通过 cerebro.adddata() 添加
  data1: 周线数据，通过 cerebro.resampledata() 从日线重采样得到

  resampledata 的工作原理:
    - 它将日线数据按周聚合（5 个交易日一周）
    - 每周的 OHLC: 使用第一条的开盘价、最高/最低/收盘价取周内值
    - 这样我们不需要额外的周线数据源，只需要有足够的日线数据即可

  在策略中通过 self.data0 (日线) 和 self.data1 (周线) 访问对应数据。

运行: python 3-多周期海龟策略.py
"""
import numpy as np
import backtrader as bt
from data_loader import load_stock_data, _wrap_strategy, _calc_metrics, plot_backtest, calc_buy_and_hold
from db_config import INITIAL_CASH, COMMISSION


# ============================================================
# 单周期海龟 (仅日线, 对照组)
# ============================================================
# 使用纯日线信号的海龟策略，用于和多周期版本对比

class SingleTFTurtle(bt.Strategy):
    """单周期海龟策略 - 只用日线信号，没有大周期过滤"""
    params = (
        ('entry_period', 20), ('exit_period', 10), ('atr_period', 20),
        ('risk_pct', 0.01), ('max_units', 4), ('add_n', 0.5), ('stop_n', 2.0),
    )

    def __init__(self):
        self.entry_high = bt.ind.Highest(self.data.high, period=self.p.entry_period)
        self.exit_low = bt.ind.Lowest(self.data.low, period=self.p.exit_period)
        self.atr = bt.ind.ATR(period=self.p.atr_period)
        self.units = 0; self.entry_prices = []; self.stop_price = 0.0
        self.last_add_price = 0.0; self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: return
        if order.status == order.Completed:
            if order.isbuy():
                fp = order.executed.price; self.entry_prices.append(fp)
                self.units = len(self.entry_prices)
                self.stop_price = fp - self.p.stop_n * self.atr[0]; self.last_add_price = fp
            elif order.issell():
                self.units = 0; self.entry_prices = []; self.stop_price = 0.0; self.last_add_price = 0.0
        self.order = None

    def _calc_unit_size(self):
        pv = self.broker.getvalue(); a = self.atr[0]
        if a <= 0: return 0
        return max(int((pv * self.p.risk_pct) / a // 100) * 100, 100)

    def next(self):
        if self.order: return
        a = self.atr[0]
        if np.isnan(a) or a <= 0: return
        c = self.data.close[0]
        if not self.position:
            if c > self.entry_high[-1]:
                s = self._calc_unit_size()
                if s > 0: self.order = self.buy(size=s)
        else:
            if c < self.stop_price: self.order = self.close(); return
            if c < self.exit_low[-1]: self.order = self.close(); return
            if self.units < self.p.max_units:
                if c >= self.last_add_price + self.p.add_n * a:
                    s = self._calc_unit_size(); cash = self.broker.getcash()
                    if s > 0 and cash > c * s * 1.01: self.order = self.buy(size=s)


# ============================================================
# 多周期海龟 (周线过滤 + 日线入场) - 核心改进
# ============================================================

class MultiTFTurtle(bt.Strategy):
    """
    多周期海龟策略 - 周线定方向，日线找入场

    =============================================================
    数据源说明
    =============================================================

    本策略使用 Backtrader 的多数据源功能:

      self.data0 (日线数据)
        - 通过 cerebro.adddata() 添加
        - 提供日级别的 OHLC 数据
        - 用于: 计算日线唐奇安通道、日线 ATR、出入场信号

      self.data1 (周线数据)
        - 通过 cerebro.resampledata(timeframe=bt.TimeFrame.Weeks) 添加
        - Backtrader 自动将日线数据重采样为周线
        - 用于: 计算周线唐奇安通道、判断大趋势方向

    重采样规则:
      周线的开盘 = 周一的日线开盘
      周线的最高 = 周一~周五的日线最高价的最大值
      周线的最低 = 周一~周五的日线最低价的最小值
      周线的收盘 = 周五的日线收盘价

    =============================================================
    策略逻辑
    =============================================================

    只挡住"最差的情况": 大趋势明确向下时禁止做多。
    其他情况（无趋势、上升趋势）都允许正常交易。

    这样做的原因是我们不希望过滤掉太多信号:
      - 趋势启动时，周线通常还在低位，严格过滤会错过
      - 海龟本身的止损机制已经能控制亏损
      - 我们只需要挡住"逆势做多"这种最危险的行为

    参数:
        daily_entry:  日线入场通道周期，默认 20
        daily_exit:   日线出场通道周期，默认 10
        weekly_period: 周线通道周期，默认 8 周
                       (约等于 40 个交易日，相当于日线的 40 日通道)
    """
    params = (
        ('daily_entry', 20),       # 日线唐奇安入场通道周期
        ('daily_exit', 10),        # 日线唐奇安出场通道周期
        ('weekly_period', 8),      # 周线唐奇安通道周期 (8 周 ≈ 40 个交易日)
        ('atr_period', 20),        # ATR 计算周期
        ('risk_pct', 0.01), ('max_units', 4), ('add_n', 0.5), ('stop_n', 2.0),
    )

    def __init__(self):
        # ---- 日线级别指标 (用于出入场和仓位计算) ----
        self.daily_entry_high = bt.ind.Highest(self.data0.high, period=self.p.daily_entry)
        self.daily_exit_low = bt.ind.Lowest(self.data0.low, period=self.p.daily_exit)
        self.daily_atr = bt.ind.ATR(self.data0, period=self.p.atr_period)

        # ---- 周线级别指标 (用于大趋势判断) ----
        # 这里使用 data1（周线数据）计算周线唐奇安通道
        self.weekly_high = bt.ind.Highest(self.data1.high, period=self.p.weekly_period)
        self.weekly_low = bt.ind.Lowest(self.data1.low, period=self.p.weekly_period)

        # ---- 状态变量 ----
        self.units = 0; self.entry_prices = []; self.stop_price = 0.0
        self.last_add_price = 0.0; self.order = None

    def notify_order(self, order):
        """订单成交回调"""
        if order.status in [order.Submitted, order.Accepted]: return
        if order.status == order.Completed:
            if order.isbuy():
                fp = order.executed.price; self.entry_prices.append(fp)
                self.units = len(self.entry_prices)
                self.stop_price = fp - self.p.stop_n * self.daily_atr[0]; self.last_add_price = fp
            elif order.issell():
                self.units = 0; self.entry_prices = []; self.stop_price = 0.0; self.last_add_price = 0.0
        self.order = None

    def _calc_unit_size(self):
        """ATR 仓位公式（基于日线 ATR）"""
        pv = self.broker.getvalue(); a = self.daily_atr[0]
        if a <= 0: return 0
        return max(int((pv * self.p.risk_pct) / a // 100) * 100, 100)

    def _get_weekly_trend(self):
        """
        判断周线级别的大趋势方向

        使用周线唐奇安通道判断趋势:
          - 周收盘价 > 8 周最高价: 上升趋势 ('up')
          - 周收盘价 < 8 周最低价: 下降趋势 ('down')
          - 其他情况: 无明确趋势 ('neutral')

        返回:
            str: 'up' / 'down' / 'neutral'
        """
        try:
            wc = self.data1.close[0]      # 本周收盘价（周线）
            wh = self.weekly_high[-1]     # 8 周最高价
            wl = self.weekly_low[-1]      # 8 周最低价
            if np.isnan(wh) or np.isnan(wl):
                return 'neutral'           # 数据不足，中性处理
            if wc > wh:
                return 'up'
            if wc < wl:
                return 'down'
            return 'neutral'
        except Exception:
            # 异常保护: 如果周线数据还没准备好，返回中性
            return 'neutral'

    def next(self):
        """
        每个交易日的核心逻辑

        相比单周期海龟，多周期版本在入场前多了一步"周线趋势判断":

        空仓时:
          1. 检查周线趋势
          2. 如果周线趋势向下 -> 跳过（不开仓），其他情况允许
          3. 如果允许 -> 用日线信号判断是否入场

        持仓时:
          1. 日线止损（2N 止损线）
          2. 日线出场（跌破 10 日低点）
          3. 新增: 周线趋势转为向下 -> 平仓（趋势逆转保护）
          4. 加仓逻辑不变（日线 0.5N 加仓）
        """
        if self.order: return
        a = self.daily_atr[0]
        if np.isnan(a) or a <= 0: return
        c = self.data0.close[0]

        # 获取当前周线趋势状态
        weekly_trend = self._get_weekly_trend()

        if not self.position:
            # ============================================================
            # 周线过滤规则: 只在周线趋势非"向下"时允许开仓
            # ============================================================
            # 这个过滤器的设计哲学是"宽松但有底线":
            #   - 'up' 和 'neutral' 都允许开仓（保留足够多的交易机会）
            #   - 只有 'down' 禁止开仓（挡住最危险的情况）
            #
            # 为什么是 8 周周期?
            #   8 周 ≈ 40 个交易日，比日线的 20 日通道长一倍
            #   这保证了周线通道确实代表了"大趋势"
            # ============================================================
            if weekly_trend == 'down':
                return  # 大趋势向下，不开仓

            # 日线入场条件: 突破 20 日最高价
            if c > self.daily_entry_high[-1]:
                s = self._calc_unit_size()
                if s > 0: self.order = self.buy(size=s)
        else:
            # ---- 持仓逻辑 ----
            # 1. 日线 2N 止损（第一道风控）
            if c < self.stop_price: self.order = self.close(); return
            # 2. 日线通道出场（第二道风控）
            if c < self.daily_exit_low[-1]: self.order = self.close(); return
            # 3. 周线趋势逆转保护（第三道风控，多周期特有）
            #    如果大趋势从向上转为向下，即使日线还没触发止损也平仓
            #    这是多周期策略的核心优势: 比单周期更早识别趋势变化
            if weekly_trend == 'down':
                self.order = self.close(); return
            # 4. 金字塔加仓（日线级别）
            if self.units < self.p.max_units:
                if c >= self.last_add_price + self.p.add_n * a:
                    s = self._calc_unit_size(); cash = self.broker.getcash()
                    if s > 0 and cash > c * s * 1.01: self.order = self.buy(size=s)


# ============================================================
# 多周期回测引擎
# ============================================================

def run_multi_tf_backtest(strategy_class, stock_code, start_date, end_date,
                          label='', plot=False, **kwargs):
    """
    多周期回测引擎: 配置日线 + 周线双数据源

    与单周期回测的区别:
      1. 创建两个 PandasData 对象（同一个 DataFrame 的副本）
      2. data_daily 通过 adddata() 添加-> self.data0 (日线)
      3. data_weekly 通过 resampledata() 添加 -> self.data1 (周线)
         resampledata 会自动将日线重采样为周线

    参数:
        strategy_class: 策略类（需要支持双数据源）
        stock_code: 标的代码
        start_date/end_date: 日期范围
        label: 显示名称
        plot: 是否保存图表
        **kwargs: 传给策略类的参数

    返回:
        dict 绩效指标 + 交易日志 + 净值数据
    """
    # 1. 加载日线数据
    df = load_stock_data(stock_code, start_date, end_date)

    # 2. 包装策略类（自动记录交易和净值）
    wrapped = _wrap_strategy(strategy_class)

    # 3. 配置 Cerebro 引擎
    cerebro = bt.Cerebro()
    cerebro.addstrategy(wrapped, **kwargs)

    # 4. 添加日线数据源
    data_daily = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data_daily)

    # 5. 添加周线数据源（通过重采样）
    #    resampledata 将日线数据压缩为周线
    #    timeframe=bt.TimeFrame.Weeks 指定压缩为周线
    data_weekly = bt.feeds.PandasData(dataname=df)
    cerebro.resampledata(data_weekly, timeframe=bt.TimeFrame.Weeks)

    # 6. 配置资金、手续费、分析器
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=COMMISSION)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    # 7. 打印回测信息
    if label:
        print(f"{label} | {stock_code} | {df.index[0].strftime('%Y-%m-%d')} ~ "
              f"{df.index[-1].strftime('%Y-%m-%d')} | {len(df)}个交易日")

    # 8. 运行回测
    results = cerebro.run()
    strat = results[0]

    # 9. 计算绩效
    m = _calc_metrics(cerebro, strat, df)

    # 10. 打印报告
    print(f"  总收益: {m['total_return']*100:+.2f}% | 年化: {m['annual_return']*100:+.2f}% | "
          f"最大回撤: {m['max_drawdown']*100:.2f}% | 夏普: {m['sharpe_ratio']:.2f} | "
          f"卡玛: {m['calmar_ratio']:.2f}")
    print(f"  交易: {m['total_trades']}次 | 胜率: {m['win_rate']*100:.1f}% | "
          f"盈亏比: {m['profit_loss_ratio']:.2f} | 利润因子: {m['profit_factor']:.2f} | "
          f"最大连亏: {m['max_consecutive_losses']}次")

    # 11. 返回结果
    result = {**m, 'df': df, 'trades': strat._trade_log, 'nav': strat._nav_log}
    if plot:
        plot_backtest(result, stock_code, label or strategy_class.__name__)
    return result


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    # 使用沪深 300 ETF 作为测试标的
    # 选择理由: ETF 趋势较明显，适合验证多周期过滤的效果
    stock_code = '510300.SH'
    start_date = '2024-01-01'
    end_date = '2025-12-31'

    print("=" * 70)
    print("多周期融合海龟策略: 周线定方向 + 日线找入场")
    print("=" * 70)
    print("\n多周期思想:")
    print("  周线通道(8周): 判断大趋势方向")
    print("  日线通道(20日): 寻找入场时机")
    print("  规则: 只有周线趋势向下时才禁止做多, 其他情况正常交易")
    print("  效果: 避免在大趋势下跌时逆势做多")
    print("\nBacktrader 多周期实现:")
    print("  data0 = 日线数据 (adddata)")
    print("  data1 = 周线数据 (resampledata 重采样)")

    # 买入持有基准
    bh = calc_buy_and_hold(stock_code, start_date, end_date)

    try:
        from data_loader import run_and_report

        # ---- 单周期海龟 (对照组) ----
        print(f"\n{'-' * 70}")
        print(f"[单周期] 仅日线海龟 | 买入持有: {bh*100:+.1f}%")
        print(f"{'-' * 70}")
        r_single = run_and_report(
            SingleTFTurtle, stock_code, start_date, end_date,
            label='单周期海龟', plot=True, use_sizer=False,
        )

        # ---- 多周期海龟 (实验组) ----
        print(f"\n{'-' * 70}")
        print("[多周期] 周线过滤(只挡下跌) + 日线入场")
        print(f"{'-' * 70}")
        r_multi = run_multi_tf_backtest(
            MultiTFTurtle, stock_code, start_date, end_date,
            label='多周期海龟', plot=True,
        )

        # ---- 对比 ----
        print(f"\n{'=' * 70}")
        print("对比总结")
        print(f"{'=' * 70}")
        print(f"  {'指标':<12} {'单周期':>14} {'多周期':>14}")
        print(f"  {'-' * 42}")
        print(f"  {'买入持有':<12} {bh*100:>+13.1f}% {bh*100:>+13.1f}%")
        print(f"  {'海龟收益':<12} {r_single['total_return']*100:>+13.2f}% {r_multi['total_return']*100:>+13.2f}%")
        print(f"  {'最大回撤':<12} {r_single['max_drawdown']*100:>13.2f}% {r_multi['max_drawdown']*100:>13.2f}%")
        print(f"  {'夏普比率':<12} {r_single['sharpe_ratio']:>14.2f} {r_multi['sharpe_ratio']:>14.2f}")
        print(f"  {'交易次数':<12} {r_single['total_trades']:>14d} {r_multi['total_trades']:>14d}")
        print(f"  {'胜率':<12} {r_single['win_rate']*100:>13.1f}% {r_multi['win_rate']*100:>13.1f}%")
        print(f"  {'盈亏比':<12} {r_single['profit_loss_ratio']:>14.2f} {r_multi['profit_loss_ratio']:>14.2f}")

        print("\n关键发现:")
        print("  - 周线过滤只挡住'大趋势明确向下'的情况, 不过度限制")
        print("  - 减少了逆势交易, 但保留了趋势启动时的入场机会")
        print("  - 多周期适合中长线交易, ETF/指数类标的效果较好")
        print("  - Backtrader的resampledata是实现多周期的关键API")

    except ValueError as e:
        print(f"\n错误: {e}")
        print("提示: 如果没有 510300.SH 数据, 可以改为 600519.SH 或其他有数据的标的")
