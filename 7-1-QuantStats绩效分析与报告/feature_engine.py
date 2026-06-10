# -*- coding: utf-8 -*-
"""
工业级特征工程引擎

功能:
  1. FACTOR_TAXONOMY       - 因子分类体系(6大类)
  2. calc_features()       - 从单只股票OHLCV计算50+技术特征
  3. calc_fundamental_features() - 基本面因子(PE/ROE等)
  4. preprocess_features()  - 华泰标准预处理(MAD去极值/Z-score/中性化)
  5. neutralize()          - 行业市值中性化
  6. get_feature_names()   - 返回特征名与分类信息

参考:
  华泰证券金工团队 XGBoost 选股模型:
  - 231个因子, MAD去极值, 行业市值中性化, Z-score标准化
  - "价量因子重要性 > 基本面因子" (SHAP分析结论)
"""
import numpy as np
import pandas as pd
import talib


# ============================================================
# 因子分类体系
# ============================================================

FACTOR_TAXONOMY = {
    'price_volume': {
        'name': '价量因子',
        'desc': '直接从价格和成交量衍生的基础因子，反映市场交易行为',
        'features': [
            'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d',
            'amplitude_5d', 'amplitude_10d',
            'vol_ratio_5d', 'vol_ratio_10d',
            'price_volume_corr_10d', 'turnover_change_5d',
        ],
    },
    'momentum': {
        'name': '动量因子',
        'desc': '衡量价格趋势的持续性和强度，ROC是最常用的动量指标',
        'features': [
            'momentum_5d', 'momentum_10d', 'momentum_20d', 'momentum_60d',
            'momentum_slope_10d', 'momentum_slope_20d',
            'momentum_accel_10d', 'momentum_accel_20d',
        ],
    },
    'volatility': {
        'name': '波动率因子',
        'desc': '衡量价格波动的剧烈程度，低波动股票往往有更高的风险调整收益',
        'features': [
            'atr_norm_14', 'hist_vol_10d', 'hist_vol_20d', 'hist_vol_60d',
            'vol_change_10d', 'vol_change_20d',
        ],
    },
    'technical': {
        'name': '技术指标因子',
        'desc': 'TA-Lib计算的经典技术分析指标，捕捉超买超卖和趋势信号',
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
        'desc': '均线偏离度和K线形态特征，反映技术面的多空力量对比',
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
        'desc': '多个因子的交叉组合，捕捉因子间的非线性关系(华泰发现部分因子存在强交互)',
        'features': [
            'mom_vol_cross', 'adx_rsi_cross',
            'vol_ratio_mom_cross', 'rsi_bbands_cross',
            'macd_adx_cross', 'vol_mom_accel_cross',
        ],
    },
}


def get_feature_names():
    """返回所有特征名称和分类信息"""
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
    """返回所有特征列名的列表"""
    cols = []
    for cat_info in FACTOR_TAXONOMY.values():
        cols.extend(cat_info['features'])
    return cols


# ============================================================
# 技术特征计算 (单只股票)
# ============================================================

def calc_features(df):
    """
    从OHLCV数据计算50+技术特征

    参数:
        df: DataFrame, 需要包含 open/high/low/close/volume 列, 索引为日期

    返回:
        DataFrame, 原始列 + 新增特征列
    """
    df = df.copy()
    o = df['open'].values.astype(float)
    h = df['high'].values.astype(float)
    lo = df['low'].values.astype(float)
    c = df['close'].values.astype(float)
    v = df['volume'].values.astype(float)

    # --- 价量因子 ---
    df['ret_1d'] = df['close'].pct_change(1)
    df['ret_3d'] = df['close'].pct_change(3)
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_10d'] = df['close'].pct_change(10)

    df['amplitude_5d'] = (df['high'].rolling(5).max() - df['low'].rolling(5).min()) / df['close'].rolling(5).mean()
    df['amplitude_10d'] = (df['high'].rolling(10).max() - df['low'].rolling(10).min()) / df['close'].rolling(10).mean()

    avg_vol_5 = df['volume'].rolling(5).mean()
    avg_vol_10 = df['volume'].rolling(10).mean()
    avg_vol_20 = df['volume'].rolling(20).mean()
    df['vol_ratio_5d'] = df['volume'] / avg_vol_5.replace(0, np.nan)
    df['vol_ratio_10d'] = df['volume'] / avg_vol_10.replace(0, np.nan)

    df['price_volume_corr_10d'] = df['close'].rolling(10).corr(df['volume'])
    df['turnover_change_5d'] = avg_vol_5 / avg_vol_20.replace(0, np.nan)

    # --- 动量因子 ---
    df['momentum_5d'] = talib.ROC(c, timeperiod=5)
    df['momentum_10d'] = talib.ROC(c, timeperiod=10)
    df['momentum_20d'] = talib.ROC(c, timeperiod=20)
    df['momentum_60d'] = talib.ROC(c, timeperiod=60)

    mom_10 = pd.Series(talib.ROC(c, timeperiod=10), index=df.index)
    mom_20 = pd.Series(talib.ROC(c, timeperiod=20), index=df.index)
    df['momentum_slope_10d'] = mom_10 - mom_10.shift(5)
    df['momentum_slope_20d'] = mom_20 - mom_20.shift(10)
    df['momentum_accel_10d'] = df['momentum_slope_10d'] - pd.Series(df['momentum_slope_10d']).shift(5).values
    df['momentum_accel_20d'] = df['momentum_slope_20d'] - pd.Series(df['momentum_slope_20d']).shift(10).values

    # --- 波动率因子 ---
    atr_14 = talib.ATR(h, lo, c, timeperiod=14)
    df['atr_norm_14'] = atr_14 / np.where(c > 0, c, np.nan)

    df['hist_vol_10d'] = df['ret_1d'].rolling(10).std() * np.sqrt(252)
    df['hist_vol_20d'] = df['ret_1d'].rolling(20).std() * np.sqrt(252)
    df['hist_vol_60d'] = df['ret_1d'].rolling(60).std() * np.sqrt(252)

    hv_10 = df['hist_vol_10d']
    hv_20 = df['hist_vol_20d']
    df['vol_change_10d'] = hv_10 / hv_10.shift(10).replace(0, np.nan) - 1
    df['vol_change_20d'] = hv_20 / hv_20.shift(20).replace(0, np.nan) - 1

    # --- 技术指标因子 ---
    df['rsi_14'] = talib.RSI(c, timeperiod=14)
    df['rsi_6'] = talib.RSI(c, timeperiod=6)
    df['adx_14'] = talib.ADX(h, lo, c, timeperiod=14)

    macd_dif, macd_signal, macd_hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd_dif'] = macd_dif
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist

    upper, middle, lower = talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)
    band_width = np.where((upper - lower) > 0, upper - lower, np.nan)
    df['bbands_position'] = (c - lower) / band_width

    slowk, slowd = talib.STOCH(h, lo, c, fastk_period=9, slowk_period=3, slowk_matype=0,
                                slowd_period=3, slowd_matype=0)
    df['kdj_k'] = slowk
    df['kdj_d'] = slowd

    df['cci_14'] = talib.CCI(h, lo, c, timeperiod=14)
    df['willr_14'] = talib.WILLR(h, lo, c, timeperiod=14)

    obv = talib.OBV(c, v)
    obv_series = pd.Series(obv, index=df.index)
    obv_ma = obv_series.rolling(10).mean()
    df['obv_slope_10d'] = (obv_series - obv_ma) / obv_ma.abs().replace(0, np.nan)

    # --- 均线与形态因子 ---
    ma5 = talib.SMA(c, timeperiod=5)
    ma10 = talib.SMA(c, timeperiod=10)
    ma20 = talib.SMA(c, timeperiod=20)
    ma60 = talib.SMA(c, timeperiod=60)

    df['ma5_bias'] = (c - ma5) / np.where(ma5 > 0, ma5, np.nan)
    df['ma10_bias'] = (c - ma10) / np.where(ma10 > 0, ma10, np.nan)
    df['ma20_bias'] = (c - ma20) / np.where(ma20 > 0, ma20, np.nan)
    df['ma60_bias'] = (c - ma60) / np.where(ma60 > 0, ma60, np.nan)

    bull_score = np.zeros(len(c))
    bull_score += np.where(c > ma5, 1, 0)
    bull_score += np.where(c > ma10, 1, 0)
    bull_score += np.where(c > ma20, 1, 0)
    bull_score += np.where(c > ma60, 1, 0)
    bull_score += np.where(ma5 > ma10, 1, 0)
    bull_score += np.where(ma10 > ma20, 1, 0)
    df['ma_bull_score'] = bull_score / 6.0

    body = np.abs(c - o)
    full_range = h - lo
    full_range_safe = np.where(full_range > 0, full_range, np.nan)
    df['upper_shadow_ratio'] = (h - np.maximum(c, o)) / full_range_safe
    df['lower_shadow_ratio'] = (np.minimum(c, o) - lo) / full_range_safe
    df['body_ratio'] = body / full_range_safe

    high_20 = pd.Series(h, index=df.index).rolling(20).max()
    low_20 = pd.Series(lo, index=df.index).rolling(20).min()
    df['new_high_20d'] = (pd.Series(h, index=df.index) >= high_20).astype(float)
    df['new_low_20d'] = (pd.Series(lo, index=df.index) <= low_20).astype(float)

    # --- 交互因子 ---
    df['mom_vol_cross'] = df['momentum_20d'] * df['atr_norm_14']
    df['adx_rsi_cross'] = df['adx_14'] * (df['rsi_14'] - 50) / 50
    df['vol_ratio_mom_cross'] = df['vol_ratio_5d'] * df['momentum_10d']
    df['rsi_bbands_cross'] = (df['rsi_14'] - 50) / 50 * df['bbands_position']
    df['macd_adx_cross'] = df['macd_hist'] * df['adx_14']
    df['vol_mom_accel_cross'] = df['hist_vol_10d'] * df['momentum_accel_10d']

    return df


# ============================================================
# 基本面特征
# ============================================================

def calc_fundamental_features(price_df, fin_df, stock_code):
    """
    从财务数据计算基本面因子

    参数:
        price_df: 日K线DataFrame (含close列)
        fin_df: 财务数据DataFrame (含stock_code, report_date, eps, roe等)
        stock_code: 股票代码

    返回:
        DataFrame, 含基本面因子列(按日期对齐到price_df)
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

    eps_daily = stock_fin['eps'].reindex(price_df.index, method='ffill')
    eps_safe = eps_daily.replace(0, np.nan)
    result['pe_ratio'] = price_df['close'] / eps_safe

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
    华泰标准预处理流水线

    步骤:
      1. MAD去极值: 中位数 +/- 5*MAD 截断
      2. 缺失值填充: 列中位数
      3. Z-score标准化: (x - mean) / std

    参数:
        df: DataFrame
        feature_cols: 要处理的列名列表（None则自动检测）
        method: 去极值方法, 'mad' 或 'sigma'

    返回:
        DataFrame, 预处理后
    """
    df = df.copy()

    if feature_cols is None:
        feature_cols = get_all_feature_cols()
        feature_cols = [c for c in feature_cols if c in df.columns]

    for col in feature_cols:
        series = df[col].copy()

        if method == 'mad':
            median = series.median()
            mad = (series - median).abs().median()
            if mad > 0:
                upper = median + 5 * 1.4826 * mad
                lower = median - 5 * 1.4826 * mad
                series = series.clip(lower=lower, upper=upper)
        elif method == 'sigma':
            mean = series.mean()
            std = series.std()
            if std > 0:
                series = series.clip(lower=mean - 3 * std, upper=mean + 3 * std)

        fill_val = series.median()
        series = series.fillna(fill_val)

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
    截面预处理: 对同一时间截面的所有股票做去极值和标准化

    参数:
        all_data: DataFrame, 必须含 'trade_date' 列和 feature_cols 中的列
        feature_cols: 特征列名列表

    返回:
        DataFrame, 预处理后
    """
    result = all_data.copy()

    for date, group in result.groupby('trade_date'):
        for col in feature_cols:
            if col not in group.columns:
                continue
            series = group[col].copy()

            median = series.median()
            mad = (series - median).abs().median()
            if mad > 0:
                upper = median + 5 * 1.4826 * mad
                lower = median - 5 * 1.4826 * mad
                series = series.clip(lower=lower, upper=upper)

            series = series.fillna(series.median())

            mean = series.mean()
            std = series.std()
            if std > 0:
                series = (series - mean) / std

            result.loc[group.index, col] = series

    return result


def neutralize(factor_series, industry_dummies, mktcap_log=None):
    """
    行业市值中性化 (回归取残差)

    原理: factor = beta0 + beta_industry * industry + beta_mktcap * ln(mktcap) + residual
    残差residual就是中性化后的因子值

    参数:
        factor_series: Series, 单个因子值
        industry_dummies: DataFrame, 行业哑变量 (one-hot)
        mktcap_log: Series, 市值对数 (可选)

    返回:
        Series, 中性化后的因子值
    """
    from sklearn.linear_model import LinearRegression

    valid_mask = factor_series.notna()
    if valid_mask.sum() < 10:
        return factor_series

    X_parts = [industry_dummies.loc[valid_mask]]
    if mktcap_log is not None:
        X_parts.append(mktcap_log.loc[valid_mask].to_frame('mktcap'))

    X = pd.concat(X_parts, axis=1).fillna(0)
    y = factor_series.loc[valid_mask].values

    model = LinearRegression()
    model.fit(X.values, y)
    residual = y - model.predict(X.values)

    result = factor_series.copy()
    result.loc[valid_mask] = residual
    return result
