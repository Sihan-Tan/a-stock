# -*- coding: utf-8 -*-
"""
数据加载与回测工具模块

本模块是整个海龟策略回测系统的"基础设施层"，提供以下功能:
  1. 数据加载: 从 MySQL 数据库 (trade_stock_daily 表) 读取日 K 线数据
  2. Cerebro 引擎配置: 统一设置初始资金、手续费、分析器 (夏普/回撤/交易统计)
  3. 策略包装器: 自动记录每笔交易日志和每日净值，无需在每个策略中重复编写
  4. 绩效计算: 从 Backtrader 分析器中提取并计算完整的绩效指标 (年化/回撤/夏普/卡玛/盈亏比/利润因子等)
  5. 可视化图表: 绘制三合一图表 (K 线+买卖点 / 净值曲线 / 回撤曲线)

设计理念:
  - 将重复的回测样板代码集中在这里，策略文件只需关注策略逻辑本身
  - 基于 CASE-Talib 技术指标库的 data_loader.py，针对海龟策略做了以下扩展:
    - 增加 use_sizer 参数: 海龟策略使用自定义 ATR 仓位管理，不需要默认的 Sizer
    - 增加 calc_buy_and_hold: 计算买入持有收益率作为基准对比

与 CASE-Talib 模块的差异:
  - CASE-Talib: 通用回测工具，默认使用 PercentSizer
  - 本模块: 针对海龟策略优化，支持"无 Sizer"模式（海龟自行管理仓位）
"""
import pandas as pd          # 数据处理：DataFrame 操作、日期处理
import numpy as np            # 数值计算：NaN 判断、数组操作
import backtrader as bt       # 回测框架：策略引擎、分析器、数据馈送
import os                     # 文件路径操作：创建输出目录
from db_config import execute_query, INITIAL_CASH, COMMISSION, POSITION_PCT


# ============================================================
# 数据加载
# ============================================================

def load_stock_data(stock_code, start_date=None, end_date=None):
    """
    从 MySQL 数据库加载日 K 线数据，并转换为 Backtrader 可用的格式

    查询字段映射:
      trade_date  -> 索引 (pandas DatetimeIndex)
      open_price  -> open   (开盘价)
      high_price  -> high   (最高价)
      low_price   -> low    (最低价)
      close_price -> close  (收盘价)
      volume      -> volume (成交量)

    参数:
        stock_code: 股票/ETF 代码，如 '600519.SH' 贵州茅台、'510300.SH' 沪深300ETF
                    注意：后缀 .SH 表示上交所，.SZ 表示深交所
        start_date: 开始日期，格式 'YYYY-MM-DD'，如 '2024-01-01'
                    传 None 表示从最早有数据的日期开始
        end_date:   结束日期，格式 'YYYY-MM-DD'，如 '2025-12-31'
                    传 None 表示到最晚有数据的日期结束

    返回:
        pandas DataFrame，结构如下:
          - index: DatetimeIndex，按日期升序排列
          - columns: ['open', 'high', 'low', 'close', 'volume']
          - 所有列均为 float64 类型

    数据源说明:
      数据来自 wucai_trade.trade_stock_daily 表，该表由数据采集模块
      (CASE-股票数据 项目) 每日定时写入，包含 A 股/ETF 的日 K 线数据

    异常:
        ValueError: 数据库中没有该标的的数据时抛出，提示先运行数据采集
    """
    # 构建动态 SQL 查询：根据是否传入起止日期，动态拼接 WHERE 条件
    # 使用参数化查询 (%s 占位符) 防止 SQL 注入
    conditions = ["stock_code = %s"]
    params = [stock_code]

    if start_date:
        conditions.append("trade_date >= %s")
        params.append(start_date)
    if end_date:
        conditions.append("trade_date <= %s")
        params.append(end_date)

    # SQL 语句说明:
    #   - 按 trade_date ASC 排序，保证时间序列的连续性
    #   - 字段名来自 trade_stock_daily 表，需要映射为回测可用的标准名称
    sql = f"""
        SELECT trade_date, open_price, high_price, low_price, close_price, volume
        FROM trade_stock_daily
        WHERE {' AND '.join(conditions)}
        ORDER BY trade_date ASC
    """
    rows = execute_query(sql, params)

    # 数据为空检查：如果查询结果为空，说明数据库中还没有该标的的数据
    if not rows:
        raise ValueError(f"没有找到 {stock_code} 的数据，请检查数据库或先运行数据采集")

    # 数据格式转换：
    #   1. 将 SQL 查询结果 (list[dict]) 转为 pandas DataFrame
    #   2. trade_date 转为 datetime 类型并设为索引（Backtrader 要求 DatetimeIndex）
    #   3. 列名重命名为标准英文名（Backtrader PandasData 默认读取的列名）
    #   4. 所有列转为数值类型（数据库可能返回 Decimal 类型，导致后续计算出错）
    df = pd.DataFrame(rows)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def list_available_symbols(start_date=None, end_date=None):
    """
    查询数据库中有哪些股票/ETF 有完整数据

    用于在运行回测前了解数据覆盖范围，方便选择测试标的。
    如果不传日期参数，返回数据库中的所有标的。

    参数:
        start_date: 开始日期（可选），只返回该日期之后有数据的标的
        end_date:   结束日期（可选），只返回该日期之前有数据的标的

    返回:
        list[str] 股票代码列表，按代码排序，如 ['159941.SZ', '300750.SZ', '510300.SH', ...]
    """
    conditions, params = [], []
    if start_date:
        conditions.append("trade_date >= %s")
        params.append(start_date)
    if end_date:
        conditions.append("trade_date <= %s")
        params.append(end_date)
    # 使用 DISTINCT 去重，每个标的只返回一次
    sql = "SELECT DISTINCT stock_code FROM trade_stock_daily"
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY stock_code"
    rows = execute_query(sql, params)
    return [r['stock_code'] for r in rows]


def get_symbol_data_summary(codes, start_date=None, end_date=None):
    """
    查询指定标的数据天数统计（按年份分组）

    用于评估每个标的数据覆盖的完整程度。如果某个年份数据天数明显偏少
    （如 2024 年只有 100 天，正常应有 ~240 个交易日），说明数据可能不完整。

    参数:
        codes: 标的代码列表，如 ['600519.SH', '510300.SH']
        start_date/end_date: 日期范围（可选）

    返回:
        dict 格式:
        {
            '600519.SH': {
                'total': 483,           # 总数据天数
                'yearly': {2024: 243, 2025: 240}  # 按年份统计的天数
            },
            ...
        }
    """
    if not codes:
        return {}
    # 动态生成占位符：根据 codes 数量生成 %s, %s, %s ...
    placeholders = ','.join(['%s'] * len(codes))
    conditions = [f"stock_code IN ({placeholders})"]
    params = list(codes)
    if start_date:
        conditions.append("trade_date >= %s")
        params.append(start_date)
    if end_date:
        conditions.append("trade_date <= %s")
        params.append(end_date)
    # 使用 GROUP BY 按标的+年份分组统计
    sql = (f"SELECT stock_code, YEAR(trade_date) AS yr, COUNT(*) AS cnt "
           f"FROM trade_stock_daily WHERE {' AND '.join(conditions)} "
           f"GROUP BY stock_code, YEAR(trade_date)")
    rows = execute_query(sql, params)
    result = {}
    for r in rows or []:
        code = r['stock_code']
        if code not in result:
            result[code] = {'total': 0, 'yearly': {}}
        result[code]['yearly'][r['yr']] = r['cnt']
        result[code]['total'] += r['cnt']
    return result


def calc_buy_and_hold(stock_code, start_date, end_date):
    """
    计算区间买入持有 (Buy & Hold) 收益率

    买入持有是最简单的被动投资策略，也是检验任何主动策略的基准。
    如果一个策略连买入持有都跑不赢，说明策略本身可能是负价值的。

    计算方法:
      收益率 = (期末收盘价 / 期初收盘价) - 1
      例如: 买入时 100 元，卖出时 115 元，收益率为 +15%

    参数:
        stock_code: 标的代码
        start_date/end_date: 日期范围

    返回:
        float 收益率（小数形式）:
          - 0.15 表示 +15%
          - -0.10 表示 -10%
        None: 数据不足（少于 2 根 K 线）或查询失败时返回 None
    """
    try:
        df = load_stock_data(stock_code, start_date, end_date)
        if len(df) < 2:
            return None
        # 收益率 = (最后一天收盘价 / 第一天收盘价) - 1
        return float(df['close'].iloc[-1] / df['close'].iloc[0] - 1)
    except Exception:
        # 任何异常（数据库连接失败、数据不存在等）都返回 None，不中断主流程
        return None


def get_instrument_names(codes):
    """
    从 trade_stock_status 表查询标的名称

    股票代码对人类不友好，"600519.SH" 远不如 "贵州茅台" 直观。
    该函数将代码映射为中文名称，用于图表标题和输出报告。

    参数:
        codes: 标的代码列表，如 ['600519.SH', '510300.SH']

    返回:
        dict {code: name}
        如 {'600519.SH': '贵州茅台', '510300.SH': '沪深300ETF'}
        未查到的标的用其代码本身作为名称（兜底策略）
    """
    if not codes:
        return {}
    placeholders = ','.join(['%s'] * len(codes))
    sql = f"SELECT stock_code, stock_name FROM trade_stock_status WHERE stock_code IN ({placeholders})"
    rows = execute_query(sql, codes)
    result = {}
    for r in rows or []:
        # 如果 stock_name 为空 (NULL)，用 stock_code 作为兜底名称
        result[r['stock_code']] = (r['stock_name'] or r['stock_code'])
    return result


# ============================================================
# 策略包装器 - 自动记录交易和净值
# ============================================================
# 这里的实现技巧是"装饰器模式"：
#   我们不修改策略类本身，而是创建一个继承自它的新类，在关键方法中插入记录逻辑
#   这样做的好处是：策略类保持纯净，专注于交易逻辑；记录逻辑集中在这里，一处修改处处生效

def _wrap_strategy(strategy_class):
    """
    包装任意策略类，自动记录交易日志和每日净值

    包装效果:
      所有通过 run_and_report 运行的策略都会自动获得以下属性:
        - strat._trade_log: list[dict]，每笔交易的日期/方向/价格/数量
        - strat._nav_log:   list[dict]，每日的净值快照

    实现原理:
      通过 Python 的动态类创建，生成 WrappedStrategy 继承自原始策略类:
        - notify_order: 订单成交时记录交易日志（买入/卖出/价格/数量）
        - next: 每个 Bar 记录当日净值

    参数:
        strategy_class: 任何继承自 bt.Strategy 的策略类

    返回:
        type 一个新的策略类（继承自 strategy_class），可直接传递给 Cerebro
    """
    class WrappedStrategy(strategy_class):
        def __init__(self):
            super().__init__()
            # _trade_log: 记录每笔成交的交易明细
            # 每笔记录包含: date(日期), type(BUY/SELL), price(成交价), size(数量)
            self._trade_log = []
            # _nav_log: 记录每个交易日的账户净值
            # 每条记录包含: date(日期), nav(账户总价值)
            self._nav_log = []

        def notify_order(self, order):
            """
            订单状态变化时自动调用，用于记录成交的交易

            注意: 订单从提交(submitted)到接受(accepted)再到成交(completed)，
            会多次触发 notify_order。我们只关心 completed 状态的订单。
            """
            if order.status == order.Completed:
                # 记录交易日志:
                #   - order.isbuy() 判断买入还是卖出
                #   - order.executed.price 是实际成交价（可能和下单价格不同）
                #   - order.executed.size 是成交数量（正数表示买入，负数表示卖出）
                self._trade_log.append({
                    'date': self.data.datetime.date(0),  # 当前 Bar 的日期
                    'type': 'BUY' if order.isbuy() else 'SELL',
                    'price': round(order.executed.price, 2),  # 成交价，保留2位小数
                    'size': abs(int(order.executed.size)),    # 成交数量，取绝对值
                })
            # 如果原始策略类有自己的 notify_order 实现，仍然调用它
            # 这样包装器不会破坏原始策略的逻辑
            if hasattr(super(), 'notify_order'):
                super().notify_order(order)

        def next(self):
            """
            每个新的 K 线到达时自动调用，用于记录当日净值

            净值记录是在原始策略的 next() 之前还是之后？
            这里是在之前记录，确保每个交易日都有净值快照。
            """
            self._nav_log.append({
                'date': self.data.datetime.date(0),
                'nav': self.broker.getvalue(),  # 账户总价值 = 现金 + 持仓市值
            })
            super().next()

    # 保持原始策略类的元数据（名称、模块路径等），方便调试和日志输出
    WrappedStrategy.__name__ = strategy_class.__name__
    WrappedStrategy.__qualname__ = strategy_class.__qualname__
    WrappedStrategy.__module__ = strategy_class.__module__
    return WrappedStrategy


# ============================================================
# Cerebro 配置
# ============================================================

def setup_cerebro(strategy_class, stock_code, start_date=None, end_date=None,
                  use_sizer=True, **strategy_kwargs):
    """
    创建并配置 Backtrader Cerebro 引擎

    这个函数封装了 Cerebro 的标准化配置流程，避免每个策略文件都重复写:
      - 创建 Cerebro 实例
      - 添加策略
      - 添加数据
      - 设置初始资金和手续费
      - 添加分析器 (夏普比率、回撤、交易分析)

    参数:
        use_sizer: 是否使用默认仓位管理器 (PercentSizer)
                   - True: 使用固定仓位管理（适合 SimpleTurtle 等不需要动态仓位的策略）
                   - False: 策略自行管理仓位（适合海龟策略，使用 ATR 动态仓位）
        **strategy_kwargs: 传递给策略类的额外参数
                   例如 run_and_report(TurtleStrategy, code, risk_pct=0.02)
                   会将 risk_pct=0.02 传给 TurtleStrategy 的 params

    返回:
        (cerebro, df)
        - cerebro: 已配置好的 Backtrader Cerebro 引擎，调用 cerebro.run() 即可回测
        - df: 加载的 DataFrame，后续用于计算绩效指标和绘图
    """
    # 1. 加载数据
    df = load_stock_data(stock_code, start_date, end_date)

    # 2. 创建 Cerebro 引擎
    cerebro = bt.Cerebro()

    # 3. 添加策略，传入用户指定的参数
    cerebro.addstrategy(strategy_class, **strategy_kwargs)

    # 4. 添加数据源：PandasData 将 pandas DataFrame 包装为 Backtrader 可用的数据馈送
    #    Backtrader 会从 DataFrame 的 index 中读取日期，列名必须为 open/high/low/close/volume
    cerebro.adddata(bt.feeds.PandasData(dataname=df))

    # 5. 设置初始资金（来自 .env 的 INITIAL_CASH）
    cerebro.broker.setcash(INITIAL_CASH)

    # 6. 设置手续费（来自 .env 的 COMMISSION）
    #    Backtrader 的手续费计算方式: commission * price * size
    #    所以万二的意思是: 0.0002 * 成交金额
    cerebro.broker.setcommission(commission=COMMISSION)

    # 7. 添加仓位管理器（仅在 use_sizer=True 时）
    #    PercentSizer 根据账户净值的一定比例计算仓位
    #    POSITION_PCT 来自 .env 配置，默认 95%
    if use_sizer:
        cerebro.addsizer(bt.sizers.PercentSizer, percents=POSITION_PCT)

    # 8. 添加三个内置分析器，用于评估策略表现
    #    分析器在 cerebro.run() 时自动收集数据，可在运行后通过 strat.analyzers 访问
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    return cerebro, df


# ============================================================
# 绩效计算
# ============================================================

def _calc_metrics(cerebro, strat, df):
    """
    从 Backtrader 运行结果中提取并计算完整的绩效指标体系

    计算的指标分为五类:
      1. 收益类: 总收益、年化收益
      2. 风险类: 最大回撤、最大回撤时长
      3. 风险调整收益: 夏普比率、卡玛比率
      4. 交易统计: 总交易次数、胜率、盈亏比
      5. 综合指标: 利润因子、最大连续亏损、期望值

    参数:
        cerebro: 运行后的 Cerebro 引擎（用于获取最终净值）
        strat:   运行后的策略实例（用于获取分析器结果）
        df:      原始 DataFrame（用于计算交易日数和年化）

    返回:
        dict 包含所有绩效指标
    """
    # ---- 基础收益 ----
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - INITIAL_CASH) / INITIAL_CASH

    # ---- 年化收益 ----
    # 使用 252 个交易日作为一年的标准（A股年均约 242-250 个交易日）
    # 年化公式: (1 + 总收益) ^ (1 / 年数) - 1
    # 这是几何年化收益率，比算术平均更准确
    trading_days = len(df)
    years = trading_days / 252
    if years > 0 and total_return > -1:
        annual_return = (1 + total_return) ** (1 / years) - 1
    else:
        # 总收益 <= -100%（亏光）或年数为 0 时，年化收益等于总收益
        annual_return = total_return

    # ---- 夏普比率 ----
    # 衡量每承担一单位风险（波动）能获得多少超额收益
    # 夏普 > 1: 好；夏普 > 2: 非常好；夏普 > 3: 极好
    # riskfreerate=0.02 表示无风险利率为 2%（参考国债收益率）
    sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0

    # ---- 最大回撤 ----
    # 衡量策略从最高点回落到最低点的最大幅度
    # 这是评估策略风险的最重要指标之一
    dd = strat.analyzers.drawdown.get_analysis()
    max_drawdown = dd.get('max', {}).get('drawdown', 0) / 100  # 转小数
    max_dd_len = dd.get('max', {}).get('len', 0)  # 最大回撤持续天数

    # ---- 卡玛比率 = 年化收益 / 最大回撤 ----
    # 和夏普类似，但用最大回撤代替标准差作为风险度量
    # 对趋势跟踪策略来说，卡玛比率比夏普更有参考价值
    # 因为趋势策略的收益分布通常非正态（尖峰肥尾），标准差不能很好描述风险
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

    # ---- 交易统计 ----
    # TradeAnalyzer 提供详细的交易统计，包括胜/负交易数、盈亏金额等
    ta = strat.analyzers.trades.get_analysis()
    total_trades = ta.get('total', {}).get('total', 0)
    won_trades = ta.get('won', {}).get('total', 0)
    lost_trades = ta.get('lost', {}).get('total', 0)
    # 胜率 = 盈利交易次数 / 总交易次数
    # 海龟策略的胜率通常不高（30-40%），但盈亏比高
    win_rate = won_trades / total_trades if total_trades > 0 else 0

    # ---- 盈亏比 (Profit/Loss Ratio) ----
    # 平均盈利 / 平均亏损的绝对值
    # 海龟策略的核心: 低胜率 + 高盈亏比 (> 2:1)
    avg_win = ta.get('won', {}).get('pnl', {}).get('average', 0) or 0
    avg_loss = ta.get('lost', {}).get('pnl', {}).get('average', 0) or 0
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # ---- 利润因子 (Profit Factor) ----
    # 总盈利 / 总亏损的绝对值
    # 利润因子 > 2 说明策略质量很好
    # 利润因子 < 1 说明策略总体是亏损的
    gross_profit = ta.get('won', {}).get('pnl', {}).get('total', 0) or 0
    gross_loss = ta.get('lost', {}).get('pnl', {}).get('total', 0) or 0
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else 0

    # ---- 最大连续亏损次数 ----
    # 衡量策略在不利行情下的回撤忍耐度
    max_consecutive_losses = _calc_max_consecutive_losses(ta)

    # ---- 期望值 (Expectancy) ----
    # 每笔交易的平均期望盈亏
    # 公式: 胜率 * 平均盈利 + 负率 * 平均亏损
    # 期望值 > 0 说明策略长期有效
    expected_value = win_rate * avg_win + (1 - win_rate) * avg_loss if total_trades > 0 else 0

    return {
        'final_value': round(final_value, 2),           # 最终账户价值
        'total_return': total_return,                    # 总收益率（小数）
        'annual_return': annual_return,                  # 年化收益率（小数）
        'max_drawdown': max_drawdown,                    # 最大回撤（小数）
        'max_dd_len': max_dd_len,                        # 最大回撤持续天数
        'sharpe_ratio': round(sharpe_ratio, 4),          # 夏普比率
        'calmar_ratio': round(calmar_ratio, 4),          # 卡玛比率
        'total_trades': total_trades,                    # 总交易次数
        'won_trades': won_trades,                        # 盈利交易次数
        'lost_trades': lost_trades,                      # 亏损交易次数
        'win_rate': win_rate,                            # 胜率
        'avg_win': avg_win,                              # 平均盈利金额
        'avg_loss': avg_loss,                            # 平均亏损金额
        'profit_loss_ratio': round(profit_loss_ratio, 2), # 盈亏比
        'profit_factor': round(profit_factor, 2),        # 利润因子
        'max_consecutive_losses': max_consecutive_losses, # 最大连续亏损
        'expected_value': round(expected_value, 2),      # 每笔期望值
        'years': round(years, 2),                        # 回测年数
        'trading_days': trading_days,                    # 交易日数
    }


def _calc_max_consecutive_losses(ta):
    """
    从 TradeAnalyzer 提取最大连续亏损次数

    连续亏损次数是一个重要的心理指标：
      如果策略最大连续亏损 10 次，投资者能否坚持执行？
      海龟法则中说"最重要的规则是坚持规则"，但连续亏损是最考验纪律性的时刻。

    TradeAnalyzer 的 streak 字段直接提供了最长连续亏损记录。
    """
    streak = ta.get('streak', {})
    lost_streak = streak.get('lost', {})
    return lost_streak.get('longest', 0) if lost_streak else 0


# ============================================================
# 运行回测 + 输出报告
# ============================================================

def run_and_report(strategy_class, stock_code, start_date=None, end_date=None,
                   label='', plot=False, quiet=False, use_sizer=True, **strategy_kwargs):
    """
    一键运行回测并打印绩效报告

    这是外部策略文件调用最多的函数，整合了:
      1. 策略包装 (自动记录交易和净值)
      2. Cerebro 配置 (数据/资金/手续费/分析器)
      3. 回测运行
      4. 绩效计算与报告打印
      5. 可选: 保存可视化图表

    典型用法:
      r = run_and_report(
          TurtleStrategy, '510300.SH',
          start_date='2024-01-01', end_date='2025-12-31',
          label='海龟策略', plot=True, use_sizer=False,
      )

    参数:
        strategy_class: 策略类 (继承 bt.Strategy)
        stock_code: 股票/ETF 代码
        start_date/end_date: 日期范围
        label: 显示名称，用于输出和图表标题
        plot: 是否输出可视化图表到 outputs/ 目录
        quiet: True 时不打印任何内容，仅返回绩效结果（用于批量回测）
        use_sizer: 是否使用默认仓位管理器（海龟策略需要设为 False）
        **strategy_kwargs: 传递给策略类的额外参数

    返回:
        dict 格式:
        {
            'final_value': 1200000.00,    # 最终资产
            'total_return': 0.20,          # 总收益率
            'annual_return': 0.095,        # 年化收益率
            'max_drawdown': 0.15,          # 最大回撤
            'sharpe_ratio': 1.2,           # 夏普比率
            ... 其他绩效指标 ...
            'df': DataFrame,               # 原始行情数据
            'trades': list[dict],          # 交易日志
            'nav': list[dict],             # 净值曲线
        }
    """
    # 第一步：包装策略类，使其自动记录交易和净值
    wrapped = _wrap_strategy(strategy_class)

    # 第二步：配置 Cerebro 引擎（加载数据、设置资金/手续费/分析器）
    cerebro, df = setup_cerebro(wrapped, stock_code, start_date, end_date,
                                use_sizer=use_sizer, **strategy_kwargs)

    # 第三步：打印回测信息（非静默模式下）
    if not quiet and label:
        print(f"{label} | {stock_code} | {df.index[0].strftime('%Y-%m-%d')} ~ "
              f"{df.index[-1].strftime('%Y-%m-%d')} | {len(df)}个交易日")

    # 第四步：运行回测
    results = cerebro.run()
    strat = results[0]  # cerebro.run() 返回列表，取第一个（因为我们只运行了一个策略）

    # 第五步：计算绩效指标
    m = _calc_metrics(cerebro, strat, df)

    # 第六步：打印绩效报告（非静默模式下）
    if not quiet:
        print(f"  总收益: {m['total_return']*100:+.2f}% | 年化: {m['annual_return']*100:+.2f}% | "
              f"最大回撤: {m['max_drawdown']*100:.2f}% | 夏普: {m['sharpe_ratio']:.2f} | "
              f"卡玛: {m['calmar_ratio']:.2f}")
        print(f"  交易: {m['total_trades']}次 | 胜率: {m['win_rate']*100:.1f}% | "
              f"盈亏比: {m['profit_loss_ratio']:.2f} | 利润因子: {m['profit_factor']:.2f} | "
              f"最大连亏: {m['max_consecutive_losses']}次")

    # 第七步：组装返回结果（绩效 + 数据 + 日志）
    result = {**m, 'df': df, 'trades': strat._trade_log, 'nav': strat._nav_log}

    # 第八步：可选 - 生成可视化图表
    if plot:
        chart_name = label or strategy_class.__name__
        plot_backtest(result, stock_code, chart_name)

    return result


# ============================================================
# 可视化图表
# ============================================================

def plot_backtest(result, stock_code='', title=''):
    """
    绘制三合一回测图表:
      上图: K 线(收盘价) + 买卖点标记 (红色三角买入, 绿色三角卖出)
      中图: 策略净值曲线 vs 买入持有基准
      下图: 回撤曲线 (红色填充)

    图表保存在 outputs/ 目录下，以 title 命名。

    参数:
        result: run_and_report 的返回值字典
        stock_code: 股票代码，用于图表标题
        title: 图表标题，也是保存的文件名
    """
    import matplotlib.pyplot as plt
    import matplotlib
    # 设置中文字体：SimHei（黑体）在 Windows 和 Linux 上通常都有
    # 如果系统没有 SimHei，可以改为 'Arial Unicode MS' (macOS) 或 'WenQuanYi Micro Hei' (Linux)
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    # 正确显示负号（默认的 Unicode 负号在某些字体中显示为方框）
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 创建输出目录
    os.makedirs('outputs', exist_ok=True)

    df = result['df']
    trades = result.get('trades', [])
    nav_data = result.get('nav', [])

    if not nav_data:
        print("没有净值数据，跳过绘图")
        return

    # ---- 构建净值 DataFrame ----
    nav_df = pd.DataFrame(nav_data)
    nav_df['date'] = pd.to_datetime(nav_df['date'])
    nav_df.set_index('date', inplace=True)
    # 净值百分比: 当前净值 / 初始资金，初始值为 1.0
    nav_df['nav_pct'] = nav_df['nav'] / INITIAL_CASH
    # 净值历史最高点（用于计算回撤）
    nav_df['peak'] = nav_df['nav'].cummax()
    # 回撤百分比: (当前值 - 最高点) / 最高点 * 100
    nav_df['drawdown'] = (nav_df['nav'] - nav_df['peak']) / nav_df['peak'] * 100

    # ---- 买入持有基准 ----
    # 将股价归一化到和净值相同的起点（1.0），方便对比
    close_start = float(df['close'].iloc[0])
    benchmark = df['close'] / close_start

    # ---- 分离买卖点 ----
    # 买入标记: 红色上三角 (^)
    buy_dates = [t['date'] for t in trades if t['type'] == 'BUY']
    buy_prices = [t['price'] for t in trades if t['type'] == 'BUY']
    # 卖出标记: 绿色下三角 (v)
    sell_dates = [t['date'] for t in trades if t['type'] == 'SELL']
    sell_prices = [t['price'] for t in trades if t['type'] == 'SELL']

    m = result
    # 创建三行子图，高度比例为 3:2:1
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12),
                                         gridspec_kw={'height_ratios': [3, 2, 1]})

    # ---- 上图: K 线(收盘价) + 买卖点 ----
    ax1.plot(df.index, df['close'], 'gray', linewidth=1, alpha=0.8, label='收盘价')
    if buy_dates:
        # 红色上三角标记买入点
        ax1.scatter(buy_dates, buy_prices, color='#e74c3c', marker='^', s=80,
                    zorder=5, label=f'买入({len(buy_dates)}次)')
    if sell_dates:
        # 绿色下三角标记卖出点
        ax1.scatter(sell_dates, sell_prices, color='#2ecc71', marker='v', s=80,
                    zorder=5, label=f'卖出({len(sell_dates)}次)')
    ax1.set_ylabel('价格')
    ax1.set_title(f'{title}  {stock_code}', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 右上角绩效摘要信息框
    info_text = (
        f"Return:    {m['total_return']*100:+.2f}%\n"
        f"Annual:    {m['annual_return']*100:+.2f}%\n"
        f"MaxDD:     {m['max_drawdown']*100:.2f}%\n"
        f"Sharpe:    {m['sharpe_ratio']:.2f}\n"
        f"Calmar:    {m['calmar_ratio']:.2f}\n"
        f"WinRate:   {m['win_rate']*100:.1f}%\n"
        f"P/L Ratio: {m['profit_loss_ratio']:.2f}\n"
        f"ProfitF:   {m['profit_factor']:.2f}"
    )
    ax1.text(0.98, 0.97, info_text, transform=ax1.transAxes,
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8),
             family='monospace')

    # ---- 中图: 净值曲线 vs 基准 ----
    ax2.plot(nav_df.index, nav_df['nav_pct'], '#2980b9', linewidth=1.5, label='策略净值')
    ax2.plot(benchmark.index, benchmark, 'gray', linewidth=1, alpha=0.6, label='买入持有')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)  # 基准线 y=1.0
    ax2.set_ylabel('净值 (初始=1.0)')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ---- 下图: 回撤曲线 ----
    ax3.fill_between(nav_df.index, nav_df['drawdown'], 0, color='#e74c3c', alpha=0.4)
    ax3.plot(nav_df.index, nav_df['drawdown'], '#c0392b', linewidth=0.8)
    ax3.set_ylabel('回撤(%)')
    ax3.set_xlabel('日期')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    safe_name = title.replace(' ', '_').replace('/', '_')
    plot_file = os.path.join('outputs', f'{safe_name}.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  图表已保存: {plot_file}")
    plt.close()
