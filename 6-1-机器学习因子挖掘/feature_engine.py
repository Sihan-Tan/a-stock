# -*- coding: utf-8 -*-
"""
工业级特征工程引擎

本模块是"机器学习因子挖掘"的核心，负责将原始 OHLCV 数据转化为
模型可用的特征（因子）。包含完整的因子计算、预处理、中性化流水线。

模块架构（6 层流水线）:
  原始 OHLCV
    -> 1. calc_features():            计算 50+ 技术因子（6 大类别）
    -> 2. calc_fundamental_features(): 补充基本面因子（PE/ROE 等）
    -> 3. preprocess_features():       MAD 去极值 + Z-score 标准化
    -> 4. preprocess_cross_section():  截面预处理（多股票同期处理）
    -> 5. neutralize():               行业市值中性化（回归取残差）

参考:
  华泰证券金工团队 XGBoost 选股模型:
  - 231 个因子体系（本模块实现核心子集）
  - MAD 去极值（中位数 +/- 5*MAD）
  - 行业市值中性化
  - Z-score 标准化
  - 核心发现：价量因子重要性 > 基本面因子（基于 SHAP 分析）

为什么因子工程如此重要？
  在量化投资中，因子就是模型的"原材料"。
  好的因子 = 好的预测信号；垃圾因子 = 模型学到的随机噪声。
  "数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限。"
  -- Andrew Ng

依赖关系：
  - 需要 TA-Lib 库（技术分析指标库），用 pip install TA-Lib 安装
  - data_loader.py 提供原始 OHLCV 数据
  - ml_engine.py 使用本模块计算的特征进行训练和预测
"""
import numpy as np
import pandas as pd
import talib  # TA-Lib: 技术分析指标库, 提供 150+ 种技术指标


# ============================================================
# 因子分类体系
# ============================================================

# FACTOR_TAXONOMY 是本模块的"因子字典"。
# 它不仅定义了有哪些因子，还对这些因子进行了分类，方便后续分析
# 不同类别因子的预测能力和稳定性差异。
#
# 6 大因子类别:
#   1. 价量因子 (price_volume) - 最基础的量价衍生
#   2. 动量因子 (momentum)     - 趋势强度
#   3. 波动率因子 (volatility) - 风险度量
#   4. 技术指标因子 (technical) - TA-Lib 经典指标
#   5. 均线与形态因子 (ma_pattern) - 均线偏离和K线形态
#   6. 交互因子 (interaction)   - 因子间非线性组合
#
# 华泰证券研究发现：各因子类别在不同市场环境下表现不同，
# 组合使用比任何单一类别都更稳健。

FACTOR_TAXONOMY = {
    'price_volume': {
        'name': '价量因子',
        'desc': '直接从价格和成交量衍生的基础因子，反映市场交易行为。'
                '价量关系是技术分析的基石："量是价的先行指标"',
        'features': [
            'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d',
            'amplitude_5d', 'amplitude_10d',
            'vol_ratio_5d', 'vol_ratio_10d',
            'price_volume_corr_10d', 'turnover_change_5d',
        ],
    },
    'momentum': {
        'name': '动量因子',
        'desc': '衡量价格趋势的持续性和强度。ROC是最常用的动量指标。'
                'A股市场存在"动量效应"：过去一段时间表现好的股票，短期内继续走好',
        'features': [
            'momentum_5d', 'momentum_10d', 'momentum_20d', 'momentum_60d',
            'momentum_slope_10d', 'momentum_slope_20d',
            'momentum_accel_10d', 'momentum_accel_20d',
        ],
    },
    'volatility': {
        'name': '波动率因子',
        'desc': '衡量价格波动的剧烈程度。研究表明低波动股票往往有更高的风险调整收益'
                '（低波动异象 Low Volatility Anomaly）',
        'features': [
            'atr_norm_14', 'hist_vol_10d', 'hist_vol_20d', 'hist_vol_60d',
            'vol_change_10d', 'vol_change_20d',
        ],
    },
    'technical': {
        'name': '技术指标因子',
        'desc': 'TA-Lib计算的经典技术分析指标，捕捉超买超卖和趋势信号。'
                '这些指标在传统技术分析中广泛应用，但在ML框架下'
                '它们不再是交易信号，而是模型的输入特征',
        'features': [
            'rsi_14', 'rsi_6',
            'adx_14',
            'macd_hist', 'macd_signal', 'macd_dif',
            'bbands_position',
            'kdj_k', 'kdj_d',
            'cci_14',
            'willr_14',
            'obv_slope_10d',
        ],
    },
    'ma_pattern': {
        'name': '均线与形态因子',
        'desc': '均线偏离度和K线形态特征，反映技术面的多空力量对比。'
                '均线多头排列 = 趋势向上，空头排列 = 趋势向下',
        'features': [
            'ma5_bias', 'ma10_bias', 'ma20_bias', 'ma60_bias',
            'ma_bull_score',
            'upper_shadow_ratio', 'lower_shadow_ratio',
            'body_ratio',
            'new_high_20d', 'new_low_20d',
        ],
    },
    'interaction': {
        'name': '交互因子',
        'desc': '多个因子的交叉组合，捕捉因子间的非线性关系。'
                '华泰研究发现部分因子（如动量+波动率）存在强交互',
        'features': [
            'mom_vol_cross', 'adx_rsi_cross',
            'vol_ratio_mom_cross', 'rsi_bbands_cross',
            'macd_adx_cross', 'vol_mom_accel_cross',
        ],
    },
}


def get_feature_names():
    """
    返回所有特征名称及其分类信息。

    与 get_all_feature_cols() 的区别：
      - 这个函数返回详细的分类信息（每个特征属于哪个大类）
      - get_all_feature_cols() 只返回特征名列表

    返回:
        list[dict]: 每个字典包含 feature, category, category_name
    """
    all_features = []
    for cat_key, cat_info in FACTOR_TAXONOMY.items():
        for feat in cat_info['features']:
            all_features.append({
                'feature': feat,
                'category': cat_key,
                'category_name': cat_info['name'],
            })
    return all_features


def get_all_feature_cols():
    """
    返回所有特征列名的扁平列表。

    这个函数主要用于：
      1. 在预处理时确定要对哪些列做去极值和标准化
      2. 在训练时指定特征列名
      3. 在因子分析时遍历所有因子

    返回:
        list[str]: 所有特征列名
    """
    cols = []
    for cat_info in FACTOR_TAXONOMY.values():
        cols.extend(cat_info['features'])
    return cols


# ============================================================
# 技术特征计算（单只股票）
# ============================================================

def calc_features(df):
    """
    从 OHLCV 数据计算 50+ 技术特征（核心函数）。

    输入：原始 OHLCV（开盘价、最高价、最低价、收盘价、成交量）
    输出：原始列 + 6 大类 50+ 技术因子

    参数:
        df: DataFrame, 需要包含 open/high/low/close/volume 列, 索引为日期

    返回:
        DataFrame, 在原数据基础上增加 50+ 特征列

    计算要点：
      - 使用 TA-Lib 加速技术指标计算（C 语言实现，性能极快）
      - 对数值稳定性做处理（如除以 0 时用 np.nan 代替）
      - 大部分因子需要 10~60 天的历史数据，前期的值是 NaN
    """
    df = df.copy()
    o = df['open'].values.astype(float)
    h = df['high'].values.astype(float)
    lo = df['low'].values.astype(float)
    c = df['close'].values.astype(float)
    v = df['volume'].values.astype(float)

    # ================================================================
    # 第1类：价量因子 (Price-Volume Factors)
    #   直接从价格和成交量计算，不经过复杂变换。
    #   这些是最原始的因子，也是其他因子的"原料"。
    # ================================================================

    # 收益率：衡量不同周期的涨跌幅
    # pct_change(N) = (当前价 / N天前价 - 1)
    df['ret_1d'] = df['close'].pct_change(1)     # 日收益率
    df['ret_3d'] = df['close'].pct_change(3)     # 3日收益率
    df['ret_5d'] = df['close'].pct_change(5)     # 5日收益率（一周）
    df['ret_10d'] = df['close'].pct_change(10)   # 10日收益率（两周）

    # 振幅：N天内的价格波动范围（相对价格水平归一化）
    # 高振幅 = 多空分歧大，低振幅 = 趋势一致
    df['amplitude_5d'] = (df['high'].rolling(5).max() - df['low'].rolling(5).min()) / df['close'].rolling(5).mean()
    df['amplitude_10d'] = (df['high'].rolling(10).max() - df['low'].rolling(10).min()) / df['close'].rolling(10).mean()

    # 量比：当前成交量相对 N 天均量的比值
    # > 1 表示放量，< 1 表示缩量
    avg_vol_5 = df['volume'].rolling(5).mean()
    avg_vol_10 = df['volume'].rolling(10).mean()
    avg_vol_20 = df['volume'].rolling(20).mean()
    df['vol_ratio_5d'] = df['volume'] / avg_vol_5.replace(0, np.nan)
    df['vol_ratio_10d'] = df['volume'] / avg_vol_10.replace(0, np.nan)

    # 量价相关性：价格和成交量的滚动相关系数
    # 正相关 = 价涨量增（健康的上涨），负相关 = 价涨量缩（可疑的上涨）
    df['price_volume_corr_10d'] = df['close'].rolling(10).corr(df['volume'])

    # 换手率变化：5日均量 / 20日均量
    # 反映短期资金流入/流出速度
    df['turnover_change_5d'] = avg_vol_5 / avg_vol_20.replace(0, np.nan)

    # ================================================================
    # 第2类：动量因子 (Momentum Factors)
    #   ROC（Price Rate of Change）= 当前价格 / N天前价格 - 1
    #   动量因子是表现最好的因子之一（Fama-French 五因子模型包含动量）
    # ================================================================

    df['momentum_5d'] = talib.ROC(c, timeperiod=5)
    df['momentum_10d'] = talib.ROC(c, timeperiod=10)
    df['momentum_20d'] = talib.ROC(c, timeperiod=20)
    df['momentum_60d'] = talib.ROC(c, timeperiod=60)

    # 动量斜率：动量的变化速度（一阶导）
    # 动量上升 = 趋势增强，动量下降 = 趋势减弱
    mom_10 = pd.Series(talib.ROC(c, timeperiod=10), index=df.index)
    mom_20 = pd.Series(talib.ROC(c, timeperiod=20), index=df.index)
    df['momentum_slope_10d'] = mom_10 - mom_10.shift(5)
    df['momentum_slope_20d'] = mom_20 - mom_20.shift(10)

    # 动量加速度：动量斜率的变化速度（二阶导）
    # 加速度为正 = 趋势加速，加速度为负 = 趋势减速
    df['momentum_accel_10d'] = df['momentum_slope_10d'] - pd.Series(df['momentum_slope_10d']).shift(5).values
    df['momentum_accel_20d'] = df['momentum_slope_20d'] - pd.Series(df['momentum_slope_20d']).shift(10).values

    # ================================================================
    # 第3类：波动率因子 (Volatility Factors)
    #   低波动异象：低波动股票长期表现优于高波动股票
    #   波动率因子在 A 股和美股都有效
    # ================================================================

    # ATR（Average True Range）：真实波幅均值，衡量价格波动幅度
    # norm 表示用价格归一化，消除高价股和低价股的量纲差异
    atr_14 = talib.ATR(h, lo, c, timeperiod=14)
    df['atr_norm_14'] = atr_14 / np.where(c > 0, c, np.nan)

    # 历史波动率：日收益率的标准差，年化到 252 个交易日
    # 年化公式：日波动率 * sqrt(252)
    df['hist_vol_10d'] = df['ret_1d'].rolling(10).std() * np.sqrt(252)
    df['hist_vol_20d'] = df['ret_1d'].rolling(20).std() * np.sqrt(252)
    df['hist_vol_60d'] = df['ret_1d'].rolling(60).std() * np.sqrt(252)

    # 波动率变化：当前波动率相对 N 天前的变化率
    # 波动率突然升高 = 市场情绪变化（可能是机会，也可能是风险）
    hv_10 = df['hist_vol_10d']
    hv_20 = df['hist_vol_20d']
    df['vol_change_10d'] = hv_10 / hv_10.shift(10).replace(0, np.nan) - 1
    df['vol_change_20d'] = hv_20 / hv_20.shift(20).replace(0, np.nan) - 1

    # ================================================================
    # 第4类：技术指标因子 (Technical Indicator Factors)
    #   经典的技术分析指标，作为模型的输入特征而非交易信号
    #   TA-Lib 提供这些指标的快速计算
    # ================================================================

    # RSI（Relative Strength Index）：相对强弱指数
    # 取值范围 0~100，>70 超买，<30 超卖
    # 行情软件常用周期是 14 天，短线交易者也会看 6 天
    df['rsi_14'] = talib.RSI(c, timeperiod=14)
    df['rsi_6'] = talib.RSI(c, timeperiod=6)

    # ADX（Average Directional Index）：平均趋向指数
    # 衡量趋势强度，>25 强趋势，<20 弱趋势或无趋势
    df['adx_14'] = talib.ADX(h, lo, c, timeperiod=14)

    # MACD（Moving Average Convergence Divergence）：指数平滑异同移动平均线
    # DIF = 快线(12日EMA) - 慢线(26日EMA)
    # Signal = DIF的9日EMA
    # Histogram = DIF - Signal（柱状线，最敏感的指标）
    macd_dif, macd_signal, macd_hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd_dif'] = macd_dif
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist

    # BOLL（Bollinger Bands）：布林带
    # 中轨 = 20日SMA，上下轨 = 中轨 +/- 2倍标准差
    # bbands_position = 价格在布林带中的相对位置（0~1）
    upper, middle, lower = talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)
    band_width = np.where((upper - lower) > 0, upper - lower, np.nan)
    df['bbands_position'] = (c - lower) / band_width

    # KDJ（随机指标）：融合了动量、强弱指标和移动平均线的优点
    # K值 = 快速随机线，D值 = 慢速随机线（K的移动平均）
    slowk, slowd = talib.STOCH(h, lo, c, fastk_period=9, slowk_period=3, slowk_matype=0,
                                slowd_period=3, slowd_matype=0)
    df['kdj_k'] = slowk
    df['kdj_d'] = slowd

    # CCI（Commodity Channel Index）：商品通道指数
    # 衡量价格偏离统计均值的程度，>100 超买，<-100 超卖
    df['cci_14'] = talib.CCI(h, lo, c, timeperiod=14)

    # Williams %R：威廉指标，与 CCI 互补的超买超卖指标
    df['willr_14'] = talib.WILLR(h, lo, c, timeperiod=14)

    # OBV（On-Balance Volume）：能量潮指标
    # 将成交量与价格变化结合，量化资金流向
    # obv_slope_10d = OBV 的短期斜率，衡量资金流入/流出的速度
    obv = talib.OBV(c, v)
    obv_series = pd.Series(obv, index=df.index)
    obv_ma = obv_series.rolling(10).mean()
    df['obv_slope_10d'] = (obv_series - obv_ma) / obv_ma.abs().replace(0, np.nan)

    # ================================================================
    # 第5类：均线与形态因子 (MA Pattern Factors)
    #   均线是技术分析中最基础也最有效的工具
    #   K 线形态（上影线/下影线/实体）反映多空博弈的细节
    # ================================================================

    # 均线偏离度：(价格 - 均线) / 均线
    # 正偏离 = 价格在均线上方，负偏离 = 价格在均线下方
    # 偏离度越大，回调压力越大
    ma5 = talib.SMA(c, timeperiod=5)
    ma10 = talib.SMA(c, timeperiod=10)
    ma20 = talib.SMA(c, timeperiod=20)
    ma60 = talib.SMA(c, timeperiod=60)

    df['ma5_bias'] = (c - ma5) / np.where(ma5 > 0, ma5, np.nan)
    df['ma10_bias'] = (c - ma10) / np.where(ma10 > 0, ma10, np.nan)
    df['ma20_bias'] = (c - ma20) / np.where(ma20 > 0, ma20, np.nan)
    df['ma60_bias'] = (c - ma60) / np.where(ma60 > 0, ma60, np.nan)

    # 均线多头排列得分：6 个条件的满足程度（0~1）
    # 条件：价格 > MA5, > MA10, > MA20, > MA60, MA5 > MA10, MA10 > MA20
    # 得分 = 1.0：完美多头排列，趋势强劲
    # 得分 = 0.0：完全空头排列，趋势疲弱
    bull_score = np.zeros(len(c))
    bull_score += np.where(c > ma5, 1, 0)
    bull_score += np.where(c > ma10, 1, 0)
    bull_score += np.where(c > ma20, 1, 0)
    bull_score += np.where(c > ma60, 1, 0)
    bull_score += np.where(ma5 > ma10, 1, 0)
    bull_score += np.where(ma10 > ma20, 1, 0)
    df['ma_bull_score'] = bull_score / 6.0

    # K 线形态因子：反映多空力量的单日博弈
    body = np.abs(c - o)                      # 实体长度
    full_range = h - lo                        # 全天振幅
    full_range_safe = np.where(full_range > 0, full_range, np.nan)
    df['upper_shadow_ratio'] = (h - np.maximum(c, o)) / full_range_safe  # 上影线比例：卖盘压力
    df['lower_shadow_ratio'] = (np.minimum(c, o) - lo) / full_range_safe  # 下影线比例：买盘支撑
    df['body_ratio'] = body / full_range_safe                               # 实体比例：多空确定性

    # 20日新高/新低：价格是否突破近期高点或低点
    # 新高 = 强势延续，新低 = 弱势延续
    high_20 = pd.Series(h, index=df.index).rolling(20).max()
    low_20 = pd.Series(lo, index=df.index).rolling(20).min()
    df['new_high_20d'] = (pd.Series(h, index=df.index) >= high_20).astype(float)
    df['new_low_20d'] = (pd.Series(lo, index=df.index) <= low_20).astype(float)

    # ================================================================
    # 第6类：交互因子 (Interaction Factors)
    #   单个因子的预测能力有限，但因子之间的交互往往能揭示更深层次的市场规律
    #   例如：强动量 + 低波动 = 更可靠的上涨信号
    # ================================================================

    df['mom_vol_cross'] = df['momentum_20d'] * df['atr_norm_14']             # 动量 * 波动率
    df['adx_rsi_cross'] = df['adx_14'] * (df['rsi_14'] - 50) / 50            # 趋势强度 * RSI 位置
    df['vol_ratio_mom_cross'] = df['vol_ratio_5d'] * df['momentum_10d']       # 量比 * 动量
    df['rsi_bbands_cross'] = (df['rsi_14'] - 50) / 50 * df['bbands_position'] # RSI * 布林位置
    df['macd_adx_cross'] = df['macd_hist'] * df['adx_14']                     # MACD柱 * 趋势强度
    df['vol_mom_accel_cross'] = df['hist_vol_10d'] * df['momentum_accel_10d']  # 波动率 * 动量加速度

    return df


# ============================================================
# 基本面特征
# ============================================================

def calc_fundamental_features(price_df, fin_df, stock_code):
    """
    从财务数据计算基本面因子。

    基本面因子与价格因子的重要区别：
      - 价格因子：每天变化，反映市场情绪和短期交易行为
      - 基本面因子：每季度更新，反映公司的内在价值
      - 两者的结合可以同时捕捉短期动量和长期价值

    参数:
        price_df: 日K线 DataFrame（需含 close 列）
        fin_df: 财务数据 DataFrame（含 stock_code/report_date/eps/roe 等）
        stock_code: 股票代码

    返回:
        DataFrame, 与 price_df 索引对齐，含基本面因子列

    关键实现细节：
      - 财务数据是按季度发布的（年报/季报），需要向前填充（ffill）
        到每个交易日：report_date 发布后，每天都知道这个财务数据
      - 同一报告期可能有多个版本（如修正），用 keep='last' 取最新版
    """
    stock_fin = fin_df[fin_df['stock_code'] == stock_code].copy()
    if stock_fin.empty:
        # 无财务数据时，返回 NaN 填充的 DataFrame
        result = pd.DataFrame(index=price_df.index)
        for col in ['pe_ratio', 'roe_factor', 'gross_margin_factor', 'debt_ratio_factor']:
            result[col] = np.nan
        return result

    stock_fin = stock_fin.sort_values('report_date')
    stock_fin = stock_fin.drop_duplicates(subset=['report_date'], keep='last')
    stock_fin.set_index('report_date', inplace=True)

    result = pd.DataFrame(index=price_df.index)

    # 市盈率 PE = 收盘价 / 每股收益(EPS)
    # PE 是最常用的估值指标，高 PE = 高估值，低 PE = 低估值
    eps_daily = stock_fin['eps'].reindex(price_df.index, method='ffill')
    eps_safe = eps_daily.replace(0, np.nan)
    result['pe_ratio'] = price_df['close'] / eps_safe

    # ROE、毛利率、负债率直接向前填充到每个交易日
    for col_src, col_dst in [('roe', 'roe_factor'),
                              ('gross_margin', 'gross_margin_factor'),
                              ('debt_ratio', 'debt_ratio_factor')]:
        if col_src in stock_fin.columns:
            result[col_dst] = stock_fin[col_src].reindex(price_df.index, method='ffill')
        else:
            result[col_dst] = np.nan

    return result


# ============================================================
# 预处理
# ============================================================

def preprocess_features(df, feature_cols=None, method='mad'):
    """
    华泰标准预处理流水线（对每只股票独立处理）。

    预处理步骤：
      1. MAD 去极值：消除极端价格的噪音
         - 原理：中位数 +/- 5 * 1.4826 * MAD
         - 为什么用 MAD 而不是标准差？MAD 对极端值更鲁棒
         - 系数 1.4826：让 MAD 在正态分布下等价于标准差
      2. 缺失值填充：用列中位数填充
         - 为什么中位数而不是均值？中位数对极端值更鲁棒
      3. Z-score 标准化：(x - mean) / std
         - 将所有因子统一到均值0/标准差1的量纲
         - 消除量纲差异对模型的影响（如 PE=100 和 RSI=50 可以比较）

    参数:
        df: DataFrame
        feature_cols: 要处理的列名列表（None 则自动检测所有特征列）
        method: 去极值方法，'mad'（推荐）或 'sigma'（3-sigma）

    返回:
        DataFrame，预处理后的数据

    注意：
      这是一个"时间序列内"（单只股票的时间维度）的预处理，
      与 preprocess_cross_section 不同。
    """
    df = df.copy()

    if feature_cols is None:
        feature_cols = get_all_feature_cols()
        feature_cols = [c for c in feature_cols if c in df.columns]

    for col in feature_cols:
        series = df[col].copy()

        # ---- 第一步：去极值 ----
        if method == 'mad':
            # MAD 方法：中位数 +/- 5*MAD
            # 为什么是 5？华泰研究的经验值，约 99% 的数据在范围内
            median = series.median()
            mad = (series - median).abs().median()
            if mad > 0:
                upper = median + 5 * 1.4826 * mad
                lower = median - 5 * 1.4826 * mad
                series = series.clip(lower=lower, upper=upper)
        elif method == 'sigma':
            # 3-sigma 方法（传统方法，但不如 MAD 鲁棒）
            mean = series.mean()
            std = series.std()
            if std > 0:
                series = series.clip(lower=mean - 3 * std, upper=mean + 3 * std)

        # ---- 第二步：缺失值填充 ----
        fill_val = series.median()
        series = series.fillna(fill_val)

        # ---- 第三步：Z-score 标准化 ----
        mean = series.mean()
        std = series.std()
        if std > 0:
            series = (series - mean) / std
        else:
            # 标准差为 0（所有值相同），直接减均值
            series = series - mean

        df[col] = series

    return df


def preprocess_cross_section(all_data, feature_cols):
    """
    截面预处理：对同一时间截面的所有股票做去极值和标准化。

    与 preprocess_features 的关键区别：
      - preprocess_features: 对单只股票的时间序列处理
        目标：消除单只股票的时序极端值
      - preprocess_cross_section: 对同一时间点的多只股票处理
        目标：消除截面上的极端值，使不同股票在同一时间可比

    在实战中的应用：
      当同时分析数百只股票时，我们关心的是某只股票在同期的
      相对表现。截面预处理让因子值在横截面上可比。

    参数:
        all_data: DataFrame，必须含 'trade_date' 列和 feature_cols
        feature_cols: 特征列名列表

    返回:
        DataFrame，预处理后的数据
    """
    result = all_data.copy()

    for date, group in result.groupby('trade_date'):
        for col in feature_cols:
            if col not in group.columns:
                continue
            series = group[col].copy()

            # MAD 去极值（截面）
            median = series.median()
            mad = (series - median).abs().median()
            if mad > 0:
                upper = median + 5 * 1.4826 * mad
                lower = median - 5 * 1.4826 * mad
                series = series.clip(lower=lower, upper=upper)

            # 缺失值填充 + Z-score 标准化
            series = series.fillna(series.median())
            mean = series.mean()
            std = series.std()
            if std > 0:
                series = (series - mean) / std

            result.loc[group.index, col] = series

    return result


def neutralize(factor_series, industry_dummies, mktcap_log=None):
    """
    行业市值中性化（回归取残差法）。

    为什么需要中性化？
      很多因子实际上只是行业或市值的"代理变量"。例如：
      - PE 因子：银行股天生 PE 低，消费股天生 PE 高
      - 动量因子：大盘股和小盘股的动量特征不同
      如果我们直接使用这些因子，模型学到的是"行业偏见"而非真正的预测信号。

    原理：
      factor = beta_industry * industry + beta_mktcap * ln(mktcap) + residual
      我们取残差 residual 作为中性化后的因子值。

      残差 = 因子值中无法被行业和市值解释的部分
      这部分才是股票"自身的"alpha 信号。

    参数:
        factor_series: Series，单个因子值
        industry_dummies: DataFrame，行业哑变量（one-hot 编码）
        mktcap_log: Series，市值的自然对数（可选）

    返回:
        Series，中性化后的因子值

    注意事项：
      - 中性化会降低因子与行业/市值的相关性，但也可能减弱因子本身的效果
      - 实践中需要对比中性化前后的 IC 表现
    """
    from sklearn.linear_model import LinearRegression

    valid_mask = factor_series.notna()
    if valid_mask.sum() < 10:
        return factor_series

    # 构建自变量：行业哑变量 + 可选市值对数
    X_parts = [industry_dummies.loc[valid_mask]]
    if mktcap_log is not None:
        X_parts.append(mktcap_log.loc[valid_mask].to_frame('mktcap'))

    X = pd.concat(X_parts, axis=1).fillna(0)
    y = factor_series.loc[valid_mask].values

    # 线性回归：因子值 ~ 行业 + 市值
    model = LinearRegression()
    model.fit(X.values, y)
    # 残差 = 实际值 - 预测值 = 中性化后的因子值
    residual = y - model.predict(X.values)

    result = factor_series.copy()
    result.loc[valid_mask] = residual
    return result
