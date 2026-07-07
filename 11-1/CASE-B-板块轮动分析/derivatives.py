# -*- coding: utf-8 -*-
# 21-CASE-B: 一阶导 + 二阶导 (速度 + 加速度)
"""
Derivatives -- 板块趋势的一阶导和二阶导
==================================================
本模块计算的指标
==================================================

一阶导 (速度类):
    1. ROC_20            20 日变化率 = (close_t - close_t-20) / close_t-20
    2. MA20_SLOPE        MA20 的最小二乘线性回归斜率, 年化为 % / 年
    3. MA20_SLOPE_NORM   MA20_SLOPE / |MA20|, 量纲无关

二阶导 (加速度类):
    4. MACD_HIST         经典 MACD 柱状 (DIF - DEA), 是"速度的变化"的代理指标
                          DIF = EMA(close, 12) - EMA(close, 26)   <- 短中期速度差
                          DEA = EMA(DIF, 9)                       <- 速度的平滑
                          HIST = DIF - DEA                        <- 速度的二阶导近似

    5. MA20_ACCEL        MA20 斜率的 5 日变化 (用相邻两段斜率差)
                          这是真正意义上的二阶导

    6. ROC_ACCEL         ROC_20 的 5 日变化, 简单粗暴的"加速度"

==================================================
使用建议
==================================================

- 选股 / 板块筛选用 ROC_20 + MA20_SLOPE 一组
- 拐点预警用 MACD_HIST + MA20_ACCEL 一组
- 完整四象限分析见 inflection_detector.py
"""
from __future__ import annotations
from typing import Dict

import numpy as np
import pandas as pd


# ============================================================
# 一阶导 (速度类)
# ============================================================

def roc_n(close: pd.Series, n: int = 20) -> pd.Series:
    """N 日变化率 (Rate of Change), 一阶导的离散近似"""
    return close.pct_change(n)


def ma_slope(close: pd.Series, ma_window: int = 20, slope_window: int = 10) -> pd.Series:
    """
    对 MA(close, ma_window) 做最小二乘线性回归, 取斜率
    斜率年化为 % / 年 (假设一年 252 个交易日)

    返回: pd.Series, 同长度, 头部 NaN
    """
    ma = close.rolling(ma_window).mean()

    def _fit(window: np.ndarray) -> float:
        if np.isnan(window).any():
            return np.nan
        y = window
        x = np.arange(len(y), dtype=float)
        # 最小二乘斜率 = cov(x,y) / var(x)
        x_mean = x.mean()
        y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0:
            return np.nan
        slope = ((x - x_mean) * (y - y_mean)).sum() / denom
        # 年化: 每日斜率 * 252 / |y_mean| -> 年化百分比
        if y_mean == 0:
            return np.nan
        return slope * 252.0 / abs(y_mean) * 100  # 单位 %

    return ma.rolling(slope_window).apply(_fit, raw=True)


def ma_slope_norm(close: pd.Series, ma_window: int = 20, slope_window: int = 10) -> pd.Series:
    """
    返回归一化的 MA20 斜率 (年化百分比), 与 ma_slope 一致
    保留这个函数是为了语义清晰: "我要的是量纲无关的斜率"
    """
    return ma_slope(close, ma_window, slope_window)


# ============================================================
# 二阶导 (加速度类)
# ============================================================

def macd_hist(close: pd.Series,
              fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    经典 MACD: 返回 DIF / DEA / HIST 三列
    HIST = DIF - DEA, 是 "速度差的变化", 业界公认的二阶导代理指标

    含义对照:
        HIST > 0 且增大: 上涨加速, 强势确认
        HIST > 0 但减小: 上涨减速, 顶部预警 (顶背离前兆)
        HIST < 0 且减小: 下跌加速, 弱势确认
        HIST < 0 但增大: 下跌减速, 见底信号 (底背离前兆)
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2.0   # 通达信约定乘 2, 业内习惯
    return pd.DataFrame({"DIF": dif, "DEA": dea, "HIST": hist})


def ma_accel(close: pd.Series,
             ma_window: int = 20, slope_window: int = 10,
             accel_lag: int = 5) -> pd.Series:
    """
    MA 斜率的二阶导: 当前斜率 - N 日前斜率
    单位与 ma_slope 一致 (年化百分比)
    """
    slope = ma_slope(close, ma_window, slope_window)
    return slope - slope.shift(accel_lag)


def roc_accel(close: pd.Series, roc_n_days: int = 20, accel_lag: int = 5) -> pd.Series:
    """
    ROC 的二阶导: 当前 ROC - N 日前 ROC
    简单粗暴, 适合做横截面比较
    """
    roc = roc_n(close, roc_n_days)
    return roc - roc.shift(accel_lag)


# ============================================================
# 一站式: 算所有导数指标
# ============================================================

def calc_all_derivatives(close: pd.Series) -> pd.DataFrame:
    """
    给定一条 close 序列, 返回所有导数指标 (DataFrame, 同长度)
    """
    macd = macd_hist(close)
    return pd.DataFrame({
        # 一阶导 (速度)
        "ROC_20":           roc_n(close, 20),
        "MA20_SLOPE":       ma_slope(close, 20, 10),

        # 二阶导 (加速度)
        "MACD_DIF":         macd["DIF"],
        "MACD_DEA":         macd["DEA"],
        "MACD_HIST":        macd["HIST"],
        "MA20_ACCEL":       ma_accel(close, 20, 10, 5),
        "ROC_ACCEL":        roc_accel(close, 20, 5),
    })


def latest_derivatives(close: pd.Series) -> Dict[str, float]:
    """取最近一日的全部导数指标 (单字典, 方便横截面拼)"""
    df = calc_all_derivatives(close)
    last = df.iloc[-1]
    return {k: float(v) if pd.notna(v) else np.nan for k, v in last.items()}
