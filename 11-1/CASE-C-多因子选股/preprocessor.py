# -*- coding: utf-8 -*-
# 21-CASE-C 多因子: 因子预处理
"""
FactorPreprocessor -- 因子预处理三件套

三件套 (按顺序应用):

    1. 去极值 (Winsorize)
        把超出 N 倍 MAD (中位绝对偏差) 的值截断到 N×MAD 边界
        作用: 防止个别异常值 (极端涨幅 / 财报暴雷) 主导整个因子分布

    2. 标准化 (Z-score)
        (x - mean) / std
        作用: 把所有因子拉到 mean=0 / std=1 的同一量纲, 可比可加

    3. 行业中性化 (Industry Neutralization)
        对每个因子在每个行业内做 Z-score (而不是全市场)
        作用: 排除"某个行业天生 PE 高"造成的伪 alpha

"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ============================================================
# 1. 去极值 (Winsorize)
# ============================================================

def winsorize_mad(series: pd.Series, n: float = 3.0) -> pd.Series:
    """
    用 MAD (中位绝对偏差) 做去极值

    公式:
        median = series.median()
        mad    = (series - median).abs().median()
        upper  = median + n × 1.4826 × mad
        lower  = median - n × 1.4826 × mad
        把超出 [lower, upper] 的值截断到边界

    1.4826 是高斯分布 MAD -> std 的换算系数
    """
    s = series.copy()
    median = s.median()
    mad = (s - median).abs().median()
    if mad == 0 or np.isnan(mad):
        return s
    upper = median + n * 1.4826 * mad
    lower = median - n * 1.4826 * mad
    return s.clip(lower=lower, upper=upper)


# ============================================================
# 2. 标准化 (Z-score)
# ============================================================

def zscore(series: pd.Series) -> pd.Series:
    """Z-score 标准化"""
    s = series.copy()
    mean = s.mean()
    std = s.std(ddof=1)
    if std == 0 or np.isnan(std):
        return s * 0.0
    return (s - mean) / std


# ============================================================
# 3. 行业中性化
# ============================================================

def industry_neutralize(factor_series: pd.Series,
                        industry_map: dict) -> pd.Series:
    """
    在每个行业内单独做 Z-score, 排除行业差异

    例如: 银行股 PE 普遍低 (5-10), 科技股 PE 普遍高 (30-100)
    如果不中性化, "选低 PE" 会变成"全选银行" -- 这是行业 beta, 不是 alpha
    """
    df = pd.DataFrame({
        "factor": factor_series,
        "industry": pd.Series(industry_map),
    })
    df = df.dropna(subset=["industry"])
    # 对每个行业组单独 Z-score
    return df.groupby("industry")["factor"].transform(zscore)


# ============================================================
# 完整 pipeline: 一行调用
# ============================================================

def preprocess_factors(factor_df: pd.DataFrame,
                       industry_map: dict = None,
                       winsorize_n: float = 3.0,
                       neutralize: bool = True) -> pd.DataFrame:
    """
    把整张因子矩阵走完三件套

    参数:
        factor_df:    DataFrame, index=股票代码, columns=因子名
        industry_map: {stock_code: industry_name}, neutralize=True 时必须
        winsorize_n:  去极值倍数 (默认 3 倍 MAD)
        neutralize:   是否做行业中性化

    返回: 预处理后的 DataFrame, 同样 shape
    """
    result = pd.DataFrame(index=factor_df.index)
    for col in factor_df.columns:
        s = factor_df[col].dropna()
        if len(s) == 0:
            result[col] = factor_df[col]
            continue
        # 1) 去极值
        s_w = winsorize_mad(s, n=winsorize_n)
        # 2) 标准化
        s_z = zscore(s_w)
        # 3) 行业中性化 (可选)
        if neutralize and industry_map:
            s_z = industry_neutralize(s_z, industry_map)
            # 中性化后再做一次全市场 Z-score, 让所有行业可比
            s_z = zscore(s_z)
        result[col] = s_z
    return result


# ============================================================
# Demo
# ============================================================

def demo():
    """构造一份模拟数据演示三件套效果"""
    np.random.seed(42)

    # 100 只股票, 假数据
    codes = [f"00000{i:04d}.SZ" for i in range(100)]
    df = pd.DataFrame({
        "MOM_1M": np.random.normal(0.05, 0.10, 100),
        "VOL_20": np.random.normal(-0.30, 0.15, 100),
    }, index=codes)
    # 加几个极端值
    df.loc[codes[0], "MOM_1M"] = 5.0    # 极端涨幅
    df.loc[codes[1], "MOM_1M"] = -3.0   # 极端跌幅

    print("=" * 60)
    print("  因子预处理三件套 demo")
    print("=" * 60)

    print("\n[原始数据] MOM_1M 的统计:")
    print(df["MOM_1M"].describe().round(3).to_string())

    print("\n[1. 去极值后] MOM_1M:")
    s_w = winsorize_mad(df["MOM_1M"])
    print(s_w.describe().round(3).to_string())

    print("\n[2. 标准化后] MOM_1M:")
    s_z = zscore(s_w)
    print(s_z.describe().round(3).to_string())

    # 加上假行业
    industries = ["银行", "科技", "消费", "医药"]
    industry_map = {c: industries[i % 4] for i, c in enumerate(codes)}

    print("\n[3. 行业中性化后] (每个行业内 Z-score) MOM_1M:")
    s_n = industry_neutralize(s_z, industry_map)
    s_n = zscore(s_n)
    print(s_n.describe().round(3).to_string())

    print("\n[Pipeline 一行调用]")
    out = preprocess_factors(df, industry_map=industry_map)
    print(out.describe().round(3).to_string())


if __name__ == "__main__":
    demo()
