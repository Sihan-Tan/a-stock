# -*- coding: utf-8 -*-
"""
工业级特征工程引擎 —— 从原始 OHLCV 数据到标准化 ML 特征的全流程

本模块是实现量化策略的"数据基础"。深度学习模型的效果 80% 取决于特征质量，
本引擎提供了从原始行情数据到标准化 ML 特征的一站式解决方案。

三大核心功能:
  1. 因子计算（calc_features）
     - 基于 TA-Lib 计算 50+ 个技术指标，覆盖 6 大类因子
     - 价量/动量/波动率/技术指标/均线形态/交互因子
  2. 基本面因子（calc_fundamental_features）
     - 从财务数据计算 PE/ROE/毛利率等基本面指标
     - 自动与日期对齐，支持前向填充
  3. 预处理管线（preprocess_features / preprocess_cross_section）
     - MAD 去极值 + 缺失值填充 + Z-score 标准化
     - 支持截面预处理（每个交易日独立处理所有股票）

参考:
  华泰证券金工团队 XGBoost 选股模型（L11 研报）:
    - 231 个因子，MAD 去极值，行业市值中性化，Z-score 标准化
    - 核心结论："价量因子重要性 > 基本面因子"（基于 SHAP 分析）

  因子分类体系灵感来自 MASTER 论文（AAAI 2024）:
    - MASTER 使用 158 维 Alpha158 因子 + 63 维市场信息
    - 本引擎的 52 维因子是其轻量级子集
"""
import numpy as np
import pandas as pd
import talib  # Technical Analysis Library: 业界标准的技术指标计算库


# ============================================================
# 因子分类体系
# ============================================================

FACTOR_TAXONOMY = {
    'price_volume': {
        'name': '价量因子',
        'desc': '直接从价格和成交量衍生的基础因子，反映市场交易行为。'
                '收益率是最基础的预测变量，成交量的变化反映市场参与者的情绪变化。',
        'features': [
            'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d',           # 不同周期的收益率
            'amplitude_5d', 'amplitude_10d',                    # 价格振幅（波动区间宽度）
            'vol_ratio_5d', 'vol_ratio_10d',                    # 成交量相对比值（当前 vs 过去）
            'price_volume_corr_10d', 'turnover_change_5d',      # 价量相关性、换手率变化
        ],
    },
    'momentum': {
        'name': '动量因子',
        'desc': '衡量价格趋势的持续性和强度。动量效应是A股市场最显著的异象之一，'
                'Jegadeesh & Titman(1993)首次系统性地证明了动量策略的有效性。'
                'ROC（Price Rate of Change）是最常用的动量指标。',
        'features': [
            'momentum_5d', 'momentum_10d', 'momentum_20d', 'momentum_60d',  # 不同周期的ROC动量
            'momentum_slope_10d', 'momentum_slope_20d',          # 动量斜率（一阶导数，趋势加速度）
            'momentum_accel_10d', 'momentum_accel_20d',          # 动量加速度（二阶导数，趋势变化率）
        ],
    },
    'volatility': {
        'name': '波动率因子',
        'desc': '衡量价格波动的剧烈程度。低波动异象（Low Volatility Anomaly）表明：'
                '低波动股票往往有更高的风险调整后收益。'
                'ATR（Average True Range）是衡量波动幅度的经典指标。',
        'features': [
            'atr_norm_14',                                      # 归一化平均真实波幅
            'hist_vol_10d', 'hist_vol_20d', 'hist_vol_60d',     # 不同周期的历史波动率
            'vol_change_10d', 'vol_change_20d',                  # 波动率变化率
        ],
    },
    'technical': {
        'name': '技术指标因子',
        'desc': 'TA-Lib 计算的经典技术分析指标。这些指标捕捉了市场中的'
                '超买超卖状态和趋势强度信号，是技术分析流派的核心工具集。',
        'features': [
            'rsi_14', 'rsi_6',             # 相对强弱指标（14日和6日）
            'adx_14',                      # 平均趋向指数（趋势强度）
            'macd_hist', 'macd_signal', 'macd_dif',  # MACD 三线（异同移动平均）
            'bbands_position',             # 布林带位置（价格在带中的相对位置）
            'kdj_k', 'kdj_d',              # KDJ 随机指标
            'cci_14',                      # 商品通道指数
            'willr_14',                    # 威廉指标
            'obv_slope_10d',               # 能量潮斜率（价量配合度）
        ],
    },
    'ma_pattern': {
        'name': '均线与形态因子',
        'desc': '均线偏离度和 K 线形态特征。反映技术面的多空力量对比，'
                '均线多头排列（短期均线 > 长期均线）是典型的多头信号。',
        'features': [
            'ma5_bias', 'ma10_bias', 'ma20_bias', 'ma60_bias',   # 不同周期均线偏离度
            'ma_bull_score',                 # 均线多头排列得分（0~1）
            'upper_shadow_ratio', 'lower_shadow_ratio',  # 上下影线比例
            'body_ratio',                    # 实体比例（反映K线力度）
            'new_high_20d', 'new_low_20d',    # 20日新高/新低标志
        ],
    },
    'interaction': {
        'name': '交互因子',
        'desc': '多个因子的交叉组合，捕捉因子间的非线性关系。'
                '华泰研究发现部分因子存在强交互作用，'
                '例如：高动量 + 低波动 = 更强的选股信号。'
                '交互因子是提升模型预测能力的高性价比方式。',
        'features': [
            'mom_vol_cross',                 # 动量 x 波动率交互
            'adx_rsi_cross',                 # 趋势强度 x 超买超卖交互
            'vol_ratio_mom_cross',           # 成交量比 x 动量交互
            'rsi_bbands_cross',              # RSI x 布林带位置交互
            'macd_adx_cross',                # MACD x ADX 交互（趋势+动量双确认）
            'vol_mom_accel_cross',           # 波动率 x 动量加速度交互
        ],
    },
}


def get_feature_names():
    """返回所有特征名称和分类信息（元数据）"""
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
    """返回所有特征列名的平铺列表"""
    cols = []
    for cat_info in FACTOR_TAXONOMY.values():
        cols.extend(cat_info['features'])
    return cols


# ============================================================
# 技术特征计算 (单只股票)
# ============================================================

def calc_features(df):
    """
    从 OHLCV 数据计算 50+ 技术特征（六大类因子）

    本函数是特征工程的核心。输入一只股票的原始 OHLCV 数据，
    输出包含 50+ 列技术因子的 DataFrame。

    参数:
        df: DataFrame，必须包含 open/high/low/close/volume 列，索引为日期

    返回:
        DataFrame，原始列 + 所有新增特征列

    设计原则:
      1. 所有特征基于滚动窗口计算，不引入未来信息（look-ahead bias）
      2. 缺失值保留为 NaN，由后续 preprocess_features 统一处理
      3. 计算性能优化：使用 numpy 数组操作和 TA-Lib 编译函数
    """
    df = df.copy()
    # 将 OHLCV 转换为 numpy 数组，加速计算
    o = df['open'].values.astype(float)
    h = df['high'].values.astype(float)
    lo = df['low'].values.astype(float)
    c = df['close'].values.astype(float)
    v = df['volume'].values.astype(float)

    # --- 价量因子 ---
    # 收益率是最基础的预测变量，通过 pct_change 计算百分比变化
    df['ret_1d'] = df['close'].pct_change(1)      # 日收益率
    df['ret_3d'] = df['close'].pct_change(3)      # 3日收益率
    df['ret_5d'] = df['close'].pct_change(5)      # 5日（一周）收益率
    df['ret_10d'] = df['close'].pct_change(10)    # 10日（两周）收益率

    # 振幅：衡量一段时间的价格波动区间
    df['amplitude_5d'] = (df['high'].rolling(5).max() - df['low'].rolling(5).min()) / df['close'].rolling(5).mean()
    df['amplitude_10d'] = (df['high'].rolling(10).max() - df['low'].rolling(10).min()) / df['close'].rolling(10).mean()

    # 成交量比率：当前成交量与历史平均成交量的比值
    # > 1 表示放量，< 1 表示缩量。放量上涨是看多信号，放量下跌是看空信号
    avg_vol_5 = df['volume'].rolling(5).mean()
    avg_vol_10 = df['volume'].rolling(10).mean()
    avg_vol_20 = df['volume'].rolling(20).mean()
    df['vol_ratio_5d'] = df['volume'] / avg_vol_5.replace(0, np.nan)
    df['vol_ratio_10d'] = df['volume'] / avg_vol_10.replace(0, np.nan)

    # 价量相关性：价格和成交量的 10 日滚动相关系数
    # 正相关 = 上涨放量（健康），负相关 = 上涨缩量（需要警惕）
    df['price_volume_corr_10d'] = df['close'].rolling(10).corr(df['volume'])
    df['turnover_change_5d'] = avg_vol_5 / avg_vol_20.replace(0, np.nan)

    # --- 动量因子 ---
    # ROC (Rate of Change) 是衡量动量的经典方法
    # 正动量 = 价格上涨，负动量 = 价格下跌
    df['momentum_5d'] = talib.ROC(c, timeperiod=5)
    df['momentum_10d'] = talib.ROC(c, timeperiod=10)
    df['momentum_20d'] = talib.ROC(c, timeperiod=20)
    df['momentum_60d'] = talib.ROC(c, timeperiod=60)

    # 动量斜率：动量本身的变化速度（相当于价格的"加速度"）
    # 斜率为正 = 动量在增强，斜率为负 = 动量在衰减
    mom_10 = pd.Series(talib.ROC(c, timeperiod=10), index=df.index)
    mom_20 = pd.Series(talib.ROC(c, timeperiod=20), index=df.index)
    df['momentum_slope_10d'] = mom_10 - mom_10.shift(5)
    df['momentum_slope_20d'] = mom_20 - mom_20.shift(10)

    # 动量加速度：动量斜率的变化率（"加加速度"）
    # 正值表示趋势加速，负值表示趋势减速
    df['momentum_accel_10d'] = df['momentum_slope_10d'] - pd.Series(df['momentum_slope_10d']).shift(5).values
    df['momentum_accel_20d'] = df['momentum_slope_20d'] - pd.Series(df['momentum_slope_20d']).shift(10).values

    # --- 波动率因子 ---
    # ATR (Average True Range)：真实波幅均值，衡量价格的日内波动幅度
    atr_14 = talib.ATR(h, lo, c, timeperiod=14)
    df['atr_norm_14'] = atr_14 / np.where(c > 0, c, np.nan)  # 归一化：除以价格消除量纲

    # 历史波动率：日收益率的标准差，年化后用于不同周期比较
    df['hist_vol_10d'] = df['ret_1d'].rolling(10).std() * np.sqrt(252)   # sqrt(252) 将日波动年化
    df['hist_vol_20d'] = df['ret_1d'].rolling(20).std() * np.sqrt(252)
    df['hist_vol_60d'] = df['ret_1d'].rolling(60).std() * np.sqrt(252)

    # 波动率变化：衡量市场是否进入"放量"或"缩量"状态
    hv_10 = df['hist_vol_10d']
    hv_20 = df['hist_vol_20d']
    df['vol_change_10d'] = hv_10 / hv_10.shift(10).replace(0, np.nan) - 1
    df['vol_change_20d'] = hv_20 / hv_20.shift(20).replace(0, np.nan) - 1

    # --- 技术指标因子 ---
    # RSI (Relative Strength Index)：相对强弱指数
    # 值 > 70 为超买（可能回调），< 30 为超卖（可能反弹）
    df['rsi_14'] = talib.RSI(c, timeperiod=14)
    df['rsi_6'] = talib.RSI(c, timeperiod=6)  # 短期 RSI，更灵敏

    # ADX (Average Directional Index)：平均趋向指数
    # 值 > 25 表示强趋势，< 20 表示盘整
    df['adx_14'] = talib.ADX(h, lo, c, timeperiod=14)

    # MACD (Moving Average Convergence Divergence)：指数平滑移动均线
    # DIF = 快线(12日EMA) - 慢线(26日EMA)
    # Signal = DIF的9日EMA
    # Hist = DIF - Signal（柱状线），上穿零轴为买入信号
    macd_dif, macd_signal, macd_hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd_dif'] = macd_dif
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist

    # 布林带位置：价格在布林带中的相对位置
    # 值 0 = 下轨，值 1 = 上轨，值 0.5 = 中轨
    # 接近上轨可能超买，接近下轨可能超卖
    upper, middle, lower = talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)
    band_width = np.where((upper - lower) > 0, upper - lower, np.nan)
    df['bbands_position'] = (c - lower) / band_width

    # KDJ 随机指标：经典短线技术指标
    # K 值上穿 D 值为买入信号，K 值下穿 D 值为卖出信号
    slowk, slowd = talib.STOCH(h, lo, c, fastk_period=9, slowk_period=3, slowk_matype=0,
                                slowd_period=3, slowd_matype=0)
    df['kdj_k'] = slowk
    df['kdj_d'] = slowd

    # CCI (Commodity Channel Index)：商品通道指数
    # 值 > 100 表示超买，< -100 表示超卖
    df['cci_14'] = talib.CCI(h, lo, c, timeperiod=14)

    # Williams %R：威廉指标，与 RSI 类似，但计算方法不同
    # 值 < -80 超卖，> -20 超买
    df['willr_14'] = talib.WILLR(h, lo, c, timeperiod=14)

    # OBV (On-Balance Volume)：能量潮
    # 价格上涨日加成交量，价格下跌日减成交量
    # OBV 斜率反映了价量配合关系
    obv = talib.OBV(c, v)
    obv_series = pd.Series(obv, index=df.index)
    obv_ma = obv_series.rolling(10).mean()
    df['obv_slope_10d'] = (obv_series - obv_ma) / obv_ma.abs().replace(0, np.nan)

    # --- 均线与形态因子 ---
    # 简单移动平均线（SMA）
    ma5 = talib.SMA(c, timeperiod=5)
    ma10 = talib.SMA(c, timeperiod=10)
    ma20 = talib.SMA(c, timeperiod=20)
    ma60 = talib.SMA(c, timeperiod=60)

    # 均线偏离度：(价格 - 均线) / 均线
    # 正值表示价格在均线上方（多头），负值表示下方（空头）
    df['ma5_bias'] = (c - ma5) / np.where(ma5 > 0, ma5, np.nan)
    df['ma10_bias'] = (c - ma10) / np.where(ma10 > 0, ma10, np.nan)
    df['ma20_bias'] = (c - ma20) / np.where(ma20 > 0, ma20, np.nan)
    df['ma60_bias'] = (c - ma60) / np.where(ma60 > 0, ma60, np.nan)

    # 均线多头排列得分（0~1）
    # 每一项条件成立加 1 分，满分 6 分，归一化到 [0, 1]
    # 得分越高表示多头趋势越强
    bull_score = np.zeros(len(c))
    bull_score += np.where(c > ma5, 1, 0)      # 价格在 MA5 之上
    bull_score += np.where(c > ma10, 1, 0)     # 价格在 MA10 之上
    bull_score += np.where(c > ma20, 1, 0)     # 价格在 MA20 之上
    bull_score += np.where(c > ma60, 1, 0)     # 价格在 MA60 之上
    bull_score += np.where(ma5 > ma10, 1, 0)   # MA5 在 MA10 之上
    bull_score += np.where(ma10 > ma20, 1, 0)  # MA10 在 MA20 之上
    df['ma_bull_score'] = bull_score / 6.0

    # K 线形态特征：实体比例、上下影线
    # 实体 = |收盘 - 开盘|，影线是超出实体的部分
    # 长上影线 = 卖方压力大（见顶信号），长下影线 = 买方支撑强（见底信号）
    body = np.abs(c - o)
    full_range = h - lo
    full_range_safe = np.where(full_range > 0, full_range, np.nan)
    df['upper_shadow_ratio'] = (h - np.maximum(c, o)) / full_range_safe
    df['lower_shadow_ratio'] = (np.minimum(c, o) - lo) / full_range_safe
    df['body_ratio'] = body / full_range_safe

    # 20 日新高/新低信号
    high_20 = pd.Series(h, index=df.index).rolling(20).max()
    low_20 = pd.Series(lo, index=df.index).rolling(20).min()
    df['new_high_20d'] = (pd.Series(h, index=df.index) >= high_20).astype(float)
    df['new_low_20d'] = (pd.Series(lo, index=df.index) <= low_20).astype(float)

    # --- 交互因子 ---
    # 交互因子通过乘法组合捕捉指标间的协同效应
    # 单个因子可能预测力有限，但组合后可能产生强信号
    df['mom_vol_cross'] = df['momentum_20d'] * df['atr_norm_14']  # 强趋势 + 高波动
    df['adx_rsi_cross'] = df['adx_14'] * (df['rsi_14'] - 50) / 50  # 趋势确认
    df['vol_ratio_mom_cross'] = df['vol_ratio_5d'] * df['momentum_10d']  # 放量 + 动量
    df['rsi_bbands_cross'] = (df['rsi_14'] - 50) / 50 * df['bbands_position']  # 双重超买/超卖
    df['macd_adx_cross'] = df['macd_hist'] * df['adx_14']  # MACD 信号 + 趋势强度
    df['vol_mom_accel_cross'] = df['hist_vol_10d'] * df['momentum_accel_10d']

    return df


# ============================================================
# 基本面特征
# ============================================================

def calc_fundamental_features(price_df, fin_df, stock_code):
    """
    从财务数据计算基本面因子

    基本面因子与技术因子的区别：
    - 技术因子：反映市场交易行为（短期，分钟~月）
    - 基本面因子：反映公司经营状况（长期，季度~年）
    两者的结合可以构建更稳健的选股模型。

    参数:
        price_df: 日 K 线 DataFrame（含 close 列）
        fin_df: 财务数据 DataFrame
        stock_code: 股票代码

    返回:
        DataFrame，含基本面因子列（按日期对齐到 price_df）
    """
    stock_fin = fin_df[fin_df['stock_code'] == stock_code].copy()
    if stock_fin.empty:
        result = pd.DataFrame(index=price_df.index)
        for col in ['pe_ratio', 'roe_factor', 'gross_margin_factor', 'debt_ratio_factor']:
            result[col] = np.nan
        return result

    stock_fin = stock_fin.sort_values('report_date')
    stock_fin = stock_fin.drop_duplicates(subset=['report_date'], keep='last')
    stock_fin.set_index('report_date', inplace=True)

    result = pd.DataFrame(index=price_df.index)

    # PE 比率（市盈率）= 股价 / 每股收益
    # 注意：使用前向填充（ffill）确保财报发布后使用最新数据
    # 这是避免未来信息泄露（look-ahead bias）的关键步骤
    eps_daily = stock_fin['eps'].reindex(price_df.index, method='ffill')
    eps_safe = eps_daily.replace(0, np.nan)
    result['pe_ratio'] = price_df['close'] / eps_safe

    # 其他基本面指标：ROE（净资产收益率）、毛利率、负债率
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
    华泰标准预处理流水线 —— 将原始因子处理为机器学习可用格式

    预处理三步骤（每步都是必要的）:
      1. MAD 去极值：消除极端值的影响
         - 为什么？原始因子中存在大量极端值（如涨停日的收益率为 +10%），
           这些极端值会严重影响后续标准化和模型训练
         - 使用中位数和 MAD（而非均值和标准差），因为中位数和 MAD 本身对异常值免疫
      2. 缺失值填充：用列中位数填充 NaN
         - 为什么？大部分 ML 模型不能处理 NaN
         - 选择中位数而非均值，同样是出于对异常值的鲁棒性考虑
      3. Z-score 标准化：让所有因子处于同一量纲
         - 为什么？不同因子的数值范围差异巨大（RSI 是 0~100，收益率是 -0.1~+0.1），
           标准化后模型可以公平地比较不同因子的重要性
         - (x - mean) / std 将所有因子变换到均值为 0、标准差为 1 的分布

    参数:
        df: DataFrame，包含原始因子列
        feature_cols: 要处理的列名列表（None 则自动检测所有因子列）
        method: 去极值方法
            - 'mad': 中位数 +/- 5*MAD 截断（华泰标准，推荐）
            - 'sigma': 均值 +/- 3*sigma 截断（传统方法，对异常值敏感）

    返回:
        DataFrame，预处理后的数据
    """
    df = df.copy()

    if feature_cols is None:
        feature_cols = get_all_feature_cols()
        feature_cols = [c for c in feature_cols if c in df.columns]

    for col in feature_cols:
        series = df[col].copy()

        # 第一步：去极值
        if method == 'mad':
            # MAD (Median Absolute Deviation)：中位数绝对偏差
            # 相比于标准差，MAD 对异常值不敏感（breakdown point = 50%）
            # 即数据中最多 50% 的异常值才会显著影响 MAD 的估计
            median = series.median()
            mad = (series - median).abs().median()
            if mad > 0:
                # 1.4826 是 MAD 到标准差的换算系数（正态分布下）
                # 5 * 1.4826 * MAD 约等于 7.4 个标准差
                # 华泰研报使用 5 倍 MAD 作为截断阈值
                upper = median + 5 * 1.4826 * mad
                lower = median - 5 * 1.4826 * mad
                series = series.clip(lower=lower, upper=upper)
        elif method == 'sigma':
            # 传统 3-sigma 方法（对异常值不够 robust）
            mean = series.mean()
            std = series.std()
            if std > 0:
                series = series.clip(lower=mean - 3 * std, upper=mean + 3 * std)

        # 第二步：缺失值填充
        fill_val = series.median()
        series = series.fillna(fill_val)

        # 第三步：Z-score 标准化
        mean = series.mean()
        std = series.std()
        if std > 0:
            series = (series - mean) / std
        else:
            series = series - mean

        df[col] = series

    return df


def preprocess_cross_section(all_data, feature_cols):
    """
    截面预处理：对同一时间截面的所有股票独立做去极值和标准化

    为什么需要截面预处理？
      在截面预测中，我们关心的是"同一时间点不同股票的相对价值"。
      因此标准化应该基于同一截面（同一交易日）的所有股票进行，
      而不是基于时间序列。

    例如：
      时间序列标准化：某只股票的 RSI 在时间维度上标准化
        -> 告诉你这只股票的 RSI 相对于其自身历史是高还是低
      截面标准化：所有股票的 RSI 在同一交易日标准化
        -> 告诉你这只股票的 RSI 相对于其他股票是高还是低
      在截面预测中，我们更需要后者——因为我们想找出"明天哪只股票会涨"。

    参数:
        all_data: DataFrame，必须含 'trade_date' 列和 feature_cols 中的列
        feature_cols: 特征列名列表

    返回:
        DataFrame，每个截面独立标准化后的数据
    """
    result = all_data.copy()

    for date, group in result.groupby('trade_date'):
        for col in feature_cols:
            if col not in group.columns:
                continue
            series = group[col].copy()

            # MAD 去极值
            median = series.median()
            mad = (series - median).abs().median()
            if mad > 0:
                upper = median + 5 * 1.4826 * mad
                lower = median - 5 * 1.4826 * mad
                series = series.clip(lower=lower, upper=upper)

            # 缺失值填充
            series = series.fillna(series.median())

            # Z-score 标准化
            mean = series.mean()
            std = series.std()
            if std > 0:
                series = (series - mean) / std

            result.loc[group.index, col] = series

    return result


def neutralize(factor_series, industry_dummies, mktcap_log=None):
    """
    行业市值中性化 —— 消除因子中的行业和市值偏见

    为什么需要中性化？
      某个因子可能天生对某些行业或大市值股票"偏好"。
      例如，高 ROE 因子天然偏好银行股（银行业 ROE 普遍较高）。
      中性化通过回归去除这些偏见，让因子"公平"地评估所有股票。

    原理（回归取残差法）：
      factor = beta0 + beta1*industry_1 + ... + beta_k*industry_k + beta_mktcap*ln(mktcap) + residual
      其中 residual 就是中性化后的因子值

    通俗理解：
      中性化后的因子衡量的是"在排除了行业和市值影响后，这只股票在这个因子上有多突出"。

    参数:
        factor_series: Series，单个因子值（连续值）
        industry_dummies: DataFrame，行业哑变量（0/1 one-hot 编码）
        mktcap_log: Series，市值对数（可选），用于市值中性化

    返回:
        Series，中性化后的因子值
    """
    from sklearn.linear_model import LinearRegression

    valid_mask = factor_series.notna()
    if valid_mask.sum() < 10:
        return factor_series  # 样本太少，不做中性化

    # 构建回归自变量：行业哑变量 + 可选的对数市值
    X_parts = [industry_dummies.loc[valid_mask]]
    if mktcap_log is not None:
        X_parts.append(mktcap_log.loc[valid_mask].to_frame('mktcap'))

    X = pd.concat(X_parts, axis=1).fillna(0)
    y = factor_series.loc[valid_mask].values

    # 线性回归：因子 = beta * 行业/市值 + 残差
    model = LinearRegression()
    model.fit(X.values, y)
    residual = y - model.predict(X.values)

    # 残差就是中性化后的因子值
    result = factor_series.copy()
    result.loc[valid_mask] = residual
    return result
