# -*- coding: utf-8 -*-
"""
多因子引擎
==========

提供因子计算、因子打分、股票池筛选的核心模块。

什么是多因子选股?
-----------------
多因子选股是量化投资中最经典的策略框架之一。核心思路是:

  1. 找到多个能预测股票未来收益的指标 (因子)
  2. 对每个股票计算这些因子值
  3. 综合打分, 选出得分最高的股票构建组合

本模块的因子体系:
-----------------
目前定义了 8 个技术因子:

  1. momentum_20d  (权重 20%): 20日动量 (ROC), 反映短期价格趋势
  2. momentum_60d  (权重 15%): 60日动量, 反映中期价格趋势
  3. volatility    (权重 15%): 归一化波动率 (ATR/Close), 低波动优先
  4. rsi_14        (权重 10%): RSI(14), 超卖区间优先
  5. adx_14        (权重 10%): ADX(14), 趋势强度
  6. turnover_ratio (权重 10%): 换手率指标, 量能放大信号
  7. price_position (权重 10%): 价格在60日区间内的位置, 低位优先
  8. macd_signal   (权重 10%): MACD柱状图信号

每个因子配置了:
  - direction: 方向 (1=正向, 值越大越好; -1=反向, 值越小越好)
  - weight:    权重 (所有因子权重之和 = 1.0)
  - desc:      因子描述

使用方式:
    from factor_engine import (
        calc_all_factors, batch_calc_factors,
        score_stocks, select_top_stocks, print_factor_report
    )

    factors = calc_all_factors(df)           # 单只股票计算
    factor_df = batch_calc_factors(all_data)  # 批量计算
    scored = score_stocks(factor_df)          # 综合打分
    top = select_top_stocks(factor_df, 10)    # 选Top-N
"""
import numpy as np
import pandas as pd
import talib  # 技术分析库, 提供 ROC/ATR/RSI/ADX/MACD 等函数


# ============================================================
# 因子定义与权重
# ============================================================

FACTOR_CONFIG = {
    'momentum_20d': {
        'name': '20日动量',
        'direction': 1,    # 正向: 值越大越好 (上涨趋势强)
        'weight': 0.20,    # 权重 20%, 最重要的因子
        'desc': 'ROC(20), 反映短期价格趋势',
    },
    'momentum_60d': {
        'name': '60日动量',
        'direction': 1,
        'weight': 0.15,
        'desc': 'ROC(60), 反映中期价格趋势',
    },
    'volatility': {
        'name': '波动率',
        'direction': -1,   # 反向: 值越小越好 (低波动优先)
        'weight': 0.15,
        'desc': 'ATR(14)/Close, 归一化波动率',
    },
    'rsi_14': {
        'name': 'RSI(14)',
        'direction': -1,   # 反向: RSI越低越可能反弹 (超卖区)
        'weight': 0.10,
        'desc': 'RSI(14), 超卖区间更优',
    },
    'adx_14': {
        'name': 'ADX(14)',
        'direction': 1,    # 正向: 趋势越强越好
        'weight': 0.10,
        'desc': 'ADX(14), 趋势强度指标',
    },
    'turnover_ratio': {
        'name': '换手率指标',
        'direction': 1,
        'weight': 0.10,
        'desc': '当日量/20日均量, 量能放大信号',
    },
    'price_position': {
        'name': '价格位置',
        'direction': -1,   # 反向: 价格在区间越低越好 (捡便宜)
        'weight': 0.10,
        'desc': '当前价在60日区间内的位置 (0~1)',
    },
    'macd_signal': {
        'name': 'MACD信号',
        'direction': 1,
        'weight': 0.10,
        'desc': 'MACD柱状图 > 0 为正',
    },
}


# ============================================================
# 因子计算
# ============================================================

def calc_all_factors(df):
    """
    对单只股票的 DataFrame 计算全部技术因子。

    使用 TA-Lib 库计算, 需要至少 60 根 K 线数据。

    参数:
        df: DataFrame, 含 open/high/low/close/volume 五列,
            DatetimeIndex, 按时间升序排列

    返回值:
        dict {factor_name: float}, 包含所有因子的最新值。
        如果数据不足 (len(df) < 60) 或计算失败, 返回 None。

    计算因子列表:
      - momentum_20d: 20日动量 (ROC)
      - momentum_60d: 60日动量 (ROC)
      - volatility:   ATR(14) / 收盘价, 归一化波动率
      - rsi_14:       RSI(14) 相对强弱指标
      - adx_14:       ADX(14) 趋势强度
      - turnover_ratio: 当日成交量 / 20日均量
      - price_position: (收盘价 - 60日最低) / (60日最高 - 60日最低)
      - macd_signal:  MACD 柱状图 (Histogram) 最新值
      - close:        收盘价 (用于后续计算, 不是因子)
    """
    if len(df) < 60:
        return None

    # 转换为 numpy float64 数组, TA-Lib 要求这种格式
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    o = df['open'].values.astype(np.float64)
    v = df['volume'].values.astype(np.float64)

    if c[-1] <= 0 or np.isnan(c[-1]):
        return None

    try:
        # ---- TA-Lib 因子计算 ----
        roc_20 = talib.ROC(c, timeperiod=20)      # 20日变化率
        roc_60 = talib.ROC(c, timeperiod=60)      # 60日变化率
        atr = talib.ATR(h, l, c, timeperiod=14)    # 平均真实波幅
        rsi = talib.RSI(c, timeperiod=14)          # 相对强弱指标
        adx = talib.ADX(h, l, c, timeperiod=14)    # 平均趋向指数
        vol_ma = talib.SMA(v, timeperiod=20)       # 20日均量
        macd_line, macd_signal, macd_hist = talib.MACD(c)  # MACD

        # ---- 自定义因子 ----
        # 60日最高/最低价 (用于价格位置因子)
        high_60 = np.nanmax(h[-60:])
        low_60 = np.nanmin(l[-60:])
        price_range = high_60 - low_60

        # 防止除以零
        vol_ma_val = vol_ma[-1] if not np.isnan(vol_ma[-1]) and vol_ma[-1] > 0 else 1

        factors = {
            'momentum_20d': float(roc_20[-1]) if not np.isnan(roc_20[-1]) else 0,
            'momentum_60d': float(roc_60[-1]) if not np.isnan(roc_60[-1]) else 0,
            'volatility': float(atr[-1] / c[-1]) if not np.isnan(atr[-1]) and c[-1] > 0 else 0,
            'rsi_14': float(rsi[-1]) if not np.isnan(rsi[-1]) else 50,
            'adx_14': float(adx[-1]) if not np.isnan(adx[-1]) else 0,
            'turnover_ratio': float(v[-1] / vol_ma_val) if vol_ma_val > 0 else 1,
            'price_position': float((c[-1] - low_60) / price_range) if price_range > 0 else 0.5,
            'macd_signal': float(macd_hist[-1]) if not np.isnan(macd_hist[-1]) else 0,
            'close': float(c[-1]),
        }
        return factors
    except Exception:
        return None


def batch_calc_factors(all_data, calc_date=None):
    """
    批量计算所有股票的因子。

    遍历字典中的每只股票, 调用 calc_all_factors() 计算因子,
    将所有结果合并成一个 DataFrame。

    参数:
        all_data: dict {stock_code: DataFrame}, 包含多只股票的 K 线数据
        calc_date: 截止日期 (pd.Timestamp), 只使用此日期之前的数据计算因子。
                   None 表示使用全部数据。

    返回值:
        DataFrame, 索引为 stock_code, 列为各因子名称。
        每行对应一只股票的因子值。
    """
    factor_dict = {}
    for code, df in all_data.items():
        # 如果指定了 calc_date, 截取该日期之前的数据
        if calc_date is not None:
            df = df[df.index <= calc_date]
        f = calc_all_factors(df)
        if f is not None:
            factor_dict[code] = f

    return pd.DataFrame(factor_dict).T  # 转置: 行=股票, 列=因子


# ============================================================
# 因子打分
# ============================================================

def score_stocks(factor_df, factor_config=None):
    """
    对股票池进行多因子综合打分。

    打分逻辑 (三步法):
      1. 横截面排名: 对每个因子, 在全体股票中排名, 归一化到 0~1
      2. 方向调整: 反向因子的排名取反 (1 - rank)
      3. 加权求和: 各因子排名 × 权重, 得到综合得分

    为什么用排名而不是原始值?
      不同因子的量纲不同 (动量是 %, RSI 是 0~100, 波动率是小数),
      直接加权没有意义。排名将所有因子统一到 0~1 的量纲。

    参数:
        factor_df: DataFrame, batch_calc_factors() 的输出,
                   每行是一只股票, 每列是一个因子
        factor_config: 因子配置字典 (默认用 FACTOR_CONFIG)

    返回值:
        DataFrame, 包含原始因子列 + 排名列 ('{factor_name}_rank') + 'score' 列,
        按得分降序排列。
    """
    config = factor_config or FACTOR_CONFIG
    result = factor_df.copy()
    result['score'] = 0.0

    for fname, cfg in config.items():
        if fname not in result.columns:
            continue

        # Step 1: 横截面排名归一化 (0~1)
        # rank(pct=True) 返回百分等级, 值越大排名越高
        rank = result[fname].rank(pct=True)

        # Step 2: 反向因子取反
        if cfg['direction'] < 0:
            rank = 1 - rank

        result[f'{fname}_rank'] = rank
        # Step 3: 加权求和
        result['score'] += rank * cfg['weight']

    # 按得分降序排列, 得分越高越好
    result = result.sort_values('score', ascending=False)
    return result


def select_top_stocks(factor_df, top_n=10, factor_config=None):
    """
    选出得分最高的 Top-N 股票。

    这是多因子选股的最后一步: 根据综合得分, 选择排名靠前的股票。

    参数:
        factor_df: batch_calc_factors() 的输出
        top_n:     选股数量
        factor_config: 因子配置 (默认用 FACTOR_CONFIG)

    返回值:
        (scored_df, top_codes) 二元组
        - scored_df: score_stocks() 的结果
        - top_codes: list[str], 得分最高的 top_n 只股票代码
    """
    scored = score_stocks(factor_df, factor_config)
    top_codes = scored.head(top_n).index.tolist()
    return scored, top_codes


def print_factor_report(scored_df, top_n=10, title=''):
    """
    打印因子打分报告, 用于查看选股结果。

    显示 Top-N 股票的排名、代码、得分和各因子值。
    同时显示得分最低的5只股票作为对比。

    参数:
        scored_df: score_stocks() 的结果
        top_n:     显示前多少名
        title:     报告标题
    """
    if title:
        print(f"\n  {title}")

    top = scored_df.head(top_n)
    print(f"\n  {'排名':>4} {'代码':<12} {'得分':>8} {'20D动量':>10} {'波动率':>10} "
          f"{'RSI':>8} {'价格位置':>10}")
    print(f"  {'-' * 66}")

    for i, (code, row) in enumerate(top.iterrows()):
        m20 = row.get('momentum_20d', 0)
        vol = row.get('volatility', 0)
        rsi = row.get('rsi_14', 0)
        pp = row.get('price_position', 0)
        print(f"  {i+1:>4} {code:<12} {row['score']:>8.4f} {m20:>+9.2f}% "
              f"{vol:>10.4f} {rsi:>8.1f} {pp:>10.3f}")

    bottom = scored_df.tail(5)
    print(f"\n  ... 得分最低 5 只:")
    for i, (code, row) in enumerate(bottom.iterrows()):
        rank_pos = len(scored_df) - len(bottom) + i + 1
        print(f"  {rank_pos:>4} {code:<12} {row['score']:>8.4f}")
