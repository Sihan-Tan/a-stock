# -*- coding: utf-8 -*-
# 21-CASE-C 多因子: 因子合成器
"""
FactorSynthesizer -- 把多个因子合成为单个 alpha 信号

三种主流合成方式:

    1. 等权 (Equal Weight)
        alpha = mean(factor1, factor2, ...)
        最简单, 假设所有因子同等重要
        基线方案, 适合入门

    2. IC 加权 (IC-Weighted)
        权重 = 该因子过去 N 期的 IC 均值 (信息系数)
        IC 越高的因子权重越大, 自动给"有效因子"加分
        最常用的工业方案

    3. Lasso 回归
        把因子矩阵作为 X, 未来收益作为 y, Lasso 找最优系数
        可以自动剔除无效因子 (系数缩到 0)
        过拟合风险高, 需要 walk-forward 验证

本模块都实现, 默认推荐 IC 加权.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def equal_weight_synthesis(factor_df: pd.DataFrame) -> pd.Series:
    """
    等权合成
    
    factor_df: index=股票, columns=因子, 都已经 Z-score 化
    返回: index=股票, value=alpha 分数
    """
    return factor_df.mean(axis=1)


def ic_weighted_synthesis(factor_df: pd.DataFrame,
                          ic_dict: dict) -> pd.Series:
    """
    IC 加权合成

    参数:
        factor_df: index=股票, columns=因子
        ic_dict:   {factor_name: ic_value} 每个因子的历史 IC 均值

    权重 = IC / sum(|IC|)  (有的因子可能反向, 但 |IC| 越大越好)
    """
    weights = pd.Series(ic_dict)
    weights = weights.reindex(factor_df.columns).fillna(0)
    if weights.abs().sum() == 0:
        return factor_df.mean(axis=1)
    weights_norm = weights / weights.abs().sum()
    return (factor_df * weights_norm).sum(axis=1)


def lasso_synthesis(X_train: pd.DataFrame, y_train: pd.Series,
                    X_predict: pd.DataFrame, alpha: float = 0.01) -> pd.Series:
    """
    Lasso 回归合成

    用过去的因子值 + 未来收益训练, 然后预测当前
    
    参数:
        X_train:    历史因子矩阵 (T 行 × K 列)
        y_train:    历史未来收益 (T 行)
        X_predict:  当前因子矩阵 (N 行 × K 列), 行是股票
        alpha:      Lasso 正则化强度

    返回: 预测的 alpha (N 行)
    """
    from sklearn.linear_model import Lasso
    model = Lasso(alpha=alpha, max_iter=5000)
    mask = ~(X_train.isna().any(axis=1) | y_train.isna())
    model.fit(X_train[mask], y_train[mask])
    return pd.Series(model.predict(X_predict), index=X_predict.index)


# ============================================================
# IC 计算 (信息系数)
# ============================================================

def calc_ic(factor_series: pd.Series, future_return: pd.Series,
            method: str = "spearman") -> float:
    """
    计算单期 IC

    spearman 相关系数 = rank-based, 不受极值影响, 最常用
    pearson 相关系数 = 线性, 容易被极端值带偏
    """
    df = pd.DataFrame({"f": factor_series, "r": future_return}).dropna()
    if len(df) < 10:
        return np.nan
    return df["f"].corr(df["r"], method=method)


def calc_ir(ic_series: pd.Series) -> float:
    """
    Information Ratio = IC 的均值 / IC 的标准差
    
    含义: IC 不仅要高, 还要稳定. IR > 0.5 算优秀
    """
    if len(ic_series) < 2:
        return np.nan
    mean = ic_series.mean()
    std = ic_series.std(ddof=1)
    return mean / std if std > 0 else np.nan


# ============================================================
# Demo
# ============================================================

def demo():
    np.random.seed(42)

    # 假设 100 只股票, 5 个因子
    codes = [f"S{i:03d}" for i in range(100)]
    factor_names = ["MOM_1M", "MOM_3M", "VOL_20", "RSI_14", "BIAS_20"]

    factor_df = pd.DataFrame(
        np.random.randn(100, 5),
        index=codes, columns=factor_names,
    )

    # 假 IC: 让 MOM_1M 和 MOM_3M 比较有效, VOL_20 中等, 其他低
    ic_dict = {
        "MOM_1M":  0.06,
        "MOM_3M":  0.05,
        "VOL_20":  0.03,
        "RSI_14":  0.01,
        "BIAS_20": 0.02,
    }

    print("=" * 60)
    print("  因子合成器 demo")
    print("=" * 60)

    print("\n[1] 等权合成")
    eq = equal_weight_synthesis(factor_df)
    print(f"  分数分布: mean={eq.mean():.3f}, std={eq.std():.3f}")
    print(f"  Top 5 股票: {list(eq.nlargest(5).index)}")

    print("\n[2] IC 加权合成")
    print(f"  IC 字典: {ic_dict}")
    ic_w = ic_weighted_synthesis(factor_df, ic_dict)
    weights = pd.Series(ic_dict) / pd.Series(ic_dict).abs().sum()
    print(f"  归一化权重: {weights.round(3).to_dict()}")
    print(f"  分数分布: mean={ic_w.mean():.3f}, std={ic_w.std():.3f}")
    print(f"  Top 5 股票: {list(ic_w.nlargest(5).index)}")
    print(f"  注意: 高 IC 因子的权重大, 所以 Top 5 大概率跟 MOM 因子高的股票吻合")


if __name__ == "__main__":
    demo()
