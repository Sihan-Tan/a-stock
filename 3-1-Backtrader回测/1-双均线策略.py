# -*- coding: utf-8 -*-
"""
双均线策略 - 趋势跟踪入门

类别: 趋势跟踪
逻辑: 快线(10日均线)上穿慢线(30日均线) -> 买入; 快线下穿慢线 -> 卖出
核心: 理解 Cerebro 大脑 + 跑通第一个策略

运行: python 1-双均线策略.py
"""
import backtrader as bt
from data_loader import load_stock_data, run_and_report, INITIAL_CASH, COMMISSION, POSITION_PCT


class DoubleMAStrategy(bt.Strategy):
    """双均线金叉/死叉策略

    使用两条移动平均线判断趋势方向:
    - 快线(10日均线)上穿慢线(30日均线) -> 金叉，买入信号
    - 快线下穿慢线 -> 死叉，卖出信号
    """
    # 策略参数: 快线和慢线的周期，可在外部覆盖
    params = (('fast', 10), ('slow', 30))

    def __init__(self):
        # 计算快线: 10日简单移动平均线 (SMA)
        self.ma_fast = bt.indicators.SMA(self.data.close, period=self.p.fast)
        # 计算慢线: 30日简单移动平均线 (SMA)
        self.ma_slow = bt.indicators.SMA(self.data.close, period=self.p.slow)
        # 交叉信号: >0 表示快线上穿慢线(金叉), <0 表示快线下穿慢线(死叉)
        self.crossover = bt.indicators.CrossOver(self.ma_fast, self.ma_slow)

    def next(self):
        """每个交易日触发一次，定义买卖逻辑"""
        if not self.position:
            # 当前无持仓 + 出现金叉信号 -> 全仓买入
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            # 当前有持仓 + 出现死叉信号 -> 清仓卖出
            self.close()


if __name__ == '__main__':
    # 回测参数配置
    stock_code = '600519.SH'    # 股票代码 (贵州茅台)
    start_date = '2025-01-01'   # 回测起始日期
    end_date = '2025-12-31'     # 回测结束日期

    # 步骤1: 从MySQL数据库加载股票日线数据，返回Pandas DataFrame
    df = load_stock_data(stock_code, start_date, end_date)
    print(f"股票: {stock_code}")
    print(f"数据: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}  共{len(df)}个交易日")

    # 步骤2: 创建 Cerebro 大脑 - Backtrader的核心引擎，负责调度策略、数据、资金、分析器
    cerebro = bt.Cerebro()

    # 步骤3: 将策略类注入Cerebro（可添加多个策略同时运行）
    cerebro.addstrategy(DoubleMAStrategy)

    # 步骤4: 将DataFrame转换为Backtrader的数据格式并注入（可添加多只股票）
    cerebro.adddata(bt.feeds.PandasData(dataname=df))

    # 步骤5: 设置回测参数（从.env配置文件读取）
    cerebro.broker.setcash(INITIAL_CASH)                              # 设置初始资金
    cerebro.broker.setcommission(commission=COMMISSION)               # 设置手续费率
    cerebro.addsizer(bt.sizers.PercentSizer, percents=POSITION_PCT)   # 设置仓位比例(占总资金的百分比)

    # 步骤6: 添加分析器 - 回测结束后自动计算各项绩效指标
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)  # 夏普比率(无风险利率2%)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')                       # 最大回撤
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')                     # 交易统计(次数/胜率等)

    # 步骤7: 运行回测，返回策略实例列表
    results = cerebro.run()
    strat = results[0]  # 取第一个(也是唯一的)策略实例

    # 步骤8: 提取并输出回测绩效结果

    # 资金相关: 获取回测结束后的最终账户总价值
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - INITIAL_CASH) / INITIAL_CASH        # 计算总收益率

    # 从夏普比率分析器中提取年化夏普比率
    sharpe = strat.analyzers.sharpe.get_analysis()
    sharpe_ratio = sharpe.get('sharperatio', 0) or 0

    # 从回撤分析器中提取最大回撤比例
    dd = strat.analyzers.drawdown.get_analysis()
    max_drawdown = dd.get('max', {}).get('drawdown', 0) / 100          # 转为小数(原始值为百分比)

    # 从交易分析器中提取交易次数和胜率
    ta = strat.analyzers.trades.get_analysis()
    trade_count = ta.get('total', {}).get('total', 0)                  # 总交易次数(买入+卖出算一次)
    won = ta.get('won', {}).get('total', 0)                            # 盈利交易次数
    win_rate = won / trade_count if trade_count > 0 else 0             # 胜率 = 盈利次数 / 总次数

    # 打印回测结果面板
    print(f"\n{'='*50}")
    print(f"初始资金:  {INITIAL_CASH:>14,.2f}")
    print(f"手续费:    {COMMISSION*10000:>11.1f} (万分之)")             # 手续费以万分之一为单位显示
    print(f"仓位比例:  {POSITION_PCT:>13d}%")                           # 仓位比例为整数百分比
    print(f"最终资金:  {final_value:>14,.2f}")
    print(f"总收益率:  {total_return*100:>13.2f}%")
    print(f"最大回撤:  {max_drawdown*100:>13.2f}%")
    print(f"夏普比率:  {sharpe_ratio:>14.4f}")
    print(f"交易次数:  {trade_count:>14d}")
    print(f"胜率:      {win_rate*100:>13.2f}%")
    print(f"{'='*50}")

    # 上面是展开写的教学版本，每一行都能看清细节
    # 实际使用时可以一行搞定（封装函数 + 自动生成K线图）:
    run_and_report(DoubleMAStrategy, '600519.SH', '2024-01-01', '2025-12-31', label='双均线策略', plot=True)
