# -*- coding: utf-8 -*-
"""
1-MASTER数据与因子 —— 数据探索与预处理对比实战

本脚本是"MASTER论文复现与策略进化"系列的第一步，也是后续所有建模工作的基础。
在把数据喂给机器学习模型之前，必须先理解数据的特征——这就是探索性数据分析（EDA）。

核心目标:
  1. 从 MySQL 加载真实 A 股数据，使用 feature_engine 计算 50+ 技术因子
  2. 诊断原始因子的分布特征（偏度、峰度、异常值率）
  3. 对比两种主流预处理方法，理解它们的差异和适用场景
     - 方法A: RobustZScoreNorm（MASTER 论文使用）
     - 方法B: MAD + Z-Score（华泰 L11 标准）
  4. 分析因子间相关性，识别冗余因子
  5. 对比 MASTER 论文配套的 63 维市场信息数据

为什么要做 EDA？
  如果没有 EDA，你可能直接把含大量极端值的因子喂给模型，
  导致模型训练不稳定、预测能力差。EDA 帮你回答：
  - 哪些因子分布正常？哪些需要去极值？
  - 因子间是否存在高度冗余？
  - 应该选择哪种预处理方法？

与 MASTER 论文的关系：
  MASTER 论文使用 Qlib 框架的 Alpha158 因子（158 维）+ 63 维市场信息 = 221 维输入。
  本脚本使用 feature_engine 的 52 维 TA-Lib 技术因子，是其轻量级子集。
  两种预处理方法的对比有助于理解 MASTER 论文中 RobustZScoreNorm 的设计动机。

MASTER 论文: Li et al., "MASTER: Market-Guided Stock Transformer (AAAI 2024)"
"""

import os
import time
import numpy as np
import pandas as pd

from data_loader import load_stock_data
from feature_engine import calc_features, get_all_feature_cols, preprocess_features

# ============================================================
# 配置
# ============================================================

START_DATE = '2023-01-01'
END_DATE = '2025-12-31'

# 选取 10 只代表性股票做 EDA（覆盖 A 股主要行业板块）
# 为什么选 10 只而不是全部？
#   目的是快速做 EDA，降低数据库查询量和计算时间。
#   10 只覆盖了消费、金融、制造、新能源、医药、科技等主要行业，
#   因子分布的统计特征已经具有代表性。
EDA_STOCKS = [
    '600519.SH',  # 贵州茅台 - 消费（白酒龙头）
    '601318.SH',  # 中国平安 - 金融（保险龙头）
    '000333.SZ',  # 美的集团 - 制造（家电龙头）
    '300750.SZ',  # 宁德时代 - 新能源（电池龙头）
    '002594.SZ',  # 比亚迪   - 汽车（新能源车龙头）
    '000858.SZ',  # 五粮液   - 消费（白酒）
    '600036.SH',  # 招商银行 - 银行（零售银行龙头）
    '600276.SH',  # 恒瑞医药 - 医药（创新药龙头）
    '002415.SZ',  # 海康威视 - 科技（安防龙头）
    '601012.SH',  # 隆基绿能 - 新能源（光伏龙头）
]

# MASTER 论文开源数据目录（用于第五部分的对比分析）
# 如果尚未下载 MASTER 代码库，此部分会自动跳过
MASTER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "MASTER-master")
MARKET_INFO_PATH = os.path.join(MASTER_DIR, "data", "csi_market_information.csv")


# ============================================================
# 第一部分: 加载真实 A 股数据并计算因子
# ============================================================

def load_and_compute():
    """
    从 MySQL 加载 10 只股票，用 feature_engine 计算 50+ 技术因子

    处理流程:
      1. 遍历 EDA_STOCKS 列表，逐只调用 load_stock_data 从数据库读取 K 线
      2. 对每只股票调用 calc_features 计算 50+ 技术因子
      3. 添加股票代码和交易日信息
      4. 将所有股票的数据拼接成面板（Panel）数据

    返回:
        panel: DataFrame, 所有股票的因子面板数据
        feature_cols: list, 可用的特征列名列表

    为什么用 Panel 结构？
      截面分析需要在同一时间点比较不同股票的因子值，
      因此所有股票的数据必须组织在一起，以交易日为索引。
    """
    print("=" * 80)
    print("第一部分: 加载A股数据并计算因子")
    print("=" * 80)
    print(f"  股票池: {len(EDA_STOCKS)} 只(覆盖消费/金融/制造/新能源/医药/科技)")
    print(f"  日期范围: {START_DATE} ~ {END_DATE}")

    t0 = time.time()
    all_frames = []
    loaded = 0

    for code in EDA_STOCKS:
        try:
            df = load_stock_data(code, START_DATE, END_DATE)
            if len(df) < 200:
                # 要求至少 200 个交易日数据（约一年），确保因子计算有足够的历史窗口
                continue
            feat_df = calc_features(df)
            feat_df['stock_code'] = code
            feat_df['trade_date'] = feat_df.index      # 将索引转为列，便于 concat
            all_frames.append(feat_df)
            loaded += 1
        except Exception as e:
            print(f"  [跳过] {code}: {e}")

    elapsed = time.time() - t0
    print(f"\n  加载成功: {loaded}/{len(EDA_STOCKS)} 只, 耗时: {elapsed:.1f}s")

    if loaded < 3:
        print("  [错误] 有效股票不足, 无法继续分析")
        return None, None

    # 将所有个股数据拼接为面板数据
    panel = pd.concat(all_frames, ignore_index=True)

    # 确定实际可用的特征列（可能在计算过程中某些列全部为 NaN）
    feature_cols = get_all_feature_cols()
    feature_cols = [c for c in feature_cols if c in panel.columns]

    print(f"  面板大小: {len(panel):,} 行 x {len(feature_cols)} 因子")

    return panel, feature_cols


# ============================================================
# 第二部分: 因子分布诊断
# ============================================================

def diagnose_factor_distribution(panel, feature_cols):
    """
    分析原始因子的分布特征：偏度、峰度、异常值率

    为什么需要分析因子分布？
      1. 偏度（Skewness）：衡量分布的对称性
         - 正偏（skew > 0）：右尾长，有大量极端大值
         - 负偏（skew < 0）：左尾长，有大量极端小值
         - |skew| > 1 表示偏离正态分布较远，需要去极值
      2. 峰度（Kurtosis）：衡量分布的"厚尾"程度
         - 正态分布的峰度约为 3
         - kurtosis > 5 表示存在明显的厚尾（极端值较多）
      3. 异常值率：超出 3 倍 IQR（四分位距）的数据比例
         - 正常情况下约 0.3%（正态分布下 3-sigma 之外的比例）

    参数:
        panel: DataFrame, 因子面板数据
        feature_cols: list, 特征列名

    返回:
        stats_df: DataFrame, 每个因子的统计分析结果
    """
    print("\n" + "=" * 80)
    print("第二部分: 因子分布诊断 (建模前必须了解的)")
    print("=" * 80)

    stats = []
    for col in feature_cols:
        series = panel[col].dropna()
        if len(series) < 100:
            continue

        # IQR 方法检测异常值
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        outlier_rate = ((series < q1 - 3 * iqr) | (series > q3 + 3 * iqr)).mean()

        stats.append({
            'factor': col,
            'mean': series.mean(),
            'std': series.std(),
            'skew': series.skew(),          # 偏度：>0 右偏，<0 左偏
            'kurtosis': series.kurtosis(),   # 峰度：>0 比正态分布更集中（厚尾）
            'outlier_pct': outlier_rate * 100,  # 异常值百分比
            'nan_pct': panel[col].isna().mean() * 100,  # 缺失值百分比
        })

    stats_df = pd.DataFrame(stats)

    # 展示异常值率最高的 5 个因子
    top_outlier = stats_df.nlargest(5, 'outlier_pct')
    print("\n  异常值率最高的5个因子 (IQR 3倍标准):")
    print(f"  {'因子':<25s} {'偏度':>8s} {'峰度':>8s} {'异常值%':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for _, row in top_outlier.iterrows():
        print(f"  {row['factor']:<25s} {row['skew']:>8.2f} {row['kurtosis']:>8.1f} {row['outlier_pct']:>7.2f}%")

    # 展示偏度最大的 5 个因子（右尾最厚）
    top_skew = stats_df.nlargest(5, 'skew')
    print(f"\n  正偏最严重的5个因子 (右尾厚):")
    print(f"  {'因子':<25s} {'偏度':>8s} {'峰度':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8}")
    for _, row in top_skew.iterrows():
        print(f"  {row['factor']:<25s} {row['skew']:>8.2f} {row['kurtosis']:>8.1f}")

    # 汇总统计
    avg_outlier = stats_df['outlier_pct'].mean()
    high_skew_count = (stats_df['skew'].abs() > 1).sum()
    high_kurtosis_count = (stats_df['kurtosis'] > 5).sum()

    print(f"\n  汇总:")
    print(f"    平均异常值率: {avg_outlier:.2f}%")
    print(f"    高偏度因子(|skew|>1): {high_skew_count}/{len(stats_df)}")
    print(f"    高峰度因子(kurtosis>5): {high_kurtosis_count}/{len(stats_df)}")
    print(f"    --> 说明原始因子普遍存在厚尾分布, 需要去极值处理")

    return stats_df


# ============================================================
# 第三部分: 两种预处理方法对比
# ============================================================

def robust_zscore_norm(series, clip_range=3.0):
    """
    MASTER 论文的 RobustZScoreNorm —— 鲁棒 Z 分数标准化

    这是 MASTER 论文中使用的预处理方法，与华泰 L11 的 MAD+Z-Score 方法有相似之处，
    但也有关键差异。

    步骤:
      1. 计算中位数（median）和 MAD（Median Absolute Deviation）
      2. 鲁棒标准差 = MAD * 1.4826
         - 为什么乘以 1.4826？在正态分布下，MAD 约等于 0.6745 倍标准差
         - 所以 标准差 = MAD / 0.6745 = MAD * 1.4826
      3. 标准化：(x - median) / robust_std
         - 使用中位数而非均值：中位数对异常值免疫（breakdown point = 50%）
         - 使用鲁棒标准差而非普通标准差：同样是对异常值免疫
      4. 裁剪到 [-clip_range, clip_range]
         - 这是 MASTER 方法独有的步骤！直接将极端值截断

    与 MAD+Z-Score 的关键差异:
      1. MASTER 用中位数 + 鲁棒标准差，华泰用均值 + 普通标准差
      2. MASTER 在标准化后硬截断，华泰在标准化前去极值
      3. MASTER 的输出严格限制在 [-3, 3] 区间，适合深度学习
      4. 华泰方法输出范围不固定，取决于数据，但对树模型无影响

    参数:
        series: array-like, 原始因子值
        clip_range: 截断范围，默认 3.0

    返回:
        array, 标准化后的因子值，范围在 [-clip_range, clip_range]
    """
    median = np.nanmedian(series)     # 中位数（忽略 NaN）
    mad = np.nanmedian(np.abs(series - median))  # MAD（中位数绝对偏差）
    robust_std = mad * 1.4826         # 换算为鲁棒标准差

    if robust_std < 1e-10:
        return np.zeros_like(series)  # 如果所有值相同，返回全 0

    normalized = (series - median) / robust_std
    return np.clip(normalized, -clip_range, clip_range)  # 硬截断


def compare_preprocessing(panel, feature_cols):
    """
    对比 RobustZScoreNorm (MASTER) vs MAD+Z-Score (华泰 L11)

    通过实际数据对比两种方法的效果差异，帮助理解它们的适用场景。

    对比维度：
      1. 处理后的数据范围：MASTER 严格限制在 [-3, 3]，华泰方法范围不固定
      2. 处理后数据的均值/标准差/偏度
      3. 被截断/去极值影响的数据比例
    """
    print("\n" + "=" * 80)
    print("第三部分: 预处理方法对比")
    print("  方法A: RobustZScoreNorm (MASTER论文) - 中位数 + MAD + clip[-3,3]")
    print("  方法B: MAD + Z-Score (L11华泰标准) - MAD去极值 + mean/std标准化")
    print("=" * 80)

    # 选择 5 个有代表性的因子做详细对比
    # 包括 RSI（范围受限的指标）、动量（可能有极端值）、波动率（偏度较大）
    demo_factors = [f for f in ['rsi_14', 'momentum_20d', 'hist_vol_20d',
                                'vol_ratio_5d', 'adx_14'] if f in feature_cols]

    if not demo_factors:
        demo_factors = feature_cols[:5]

    # 方法 A: RobustZScoreNorm
    robust_panel = panel.copy()
    for col in feature_cols:
        vals = robust_panel[col].values.astype(float)
        robust_panel[col] = robust_zscore_norm(vals)

    # 方法 B: MAD + Z-Score（华泰标准预处理）
    mad_panel = preprocess_features(panel, feature_cols, method='mad')

    # 展示对比结果
    print(f"\n  {'因子':<25s} | {'原始范围':>20s} | {'RobustZScore范围':>20s} | {'MAD+ZScore范围':>20s}")
    print(f"  {'-'*25}-+-{'-'*20}-+-{'-'*20}-+-{'-'*20}")

    for col in demo_factors:
        raw_vals = panel[col].dropna().values
        r_vals = robust_panel[col].dropna().values
        m_vals = mad_panel[col].dropna().values

        raw_range = f"[{np.min(raw_vals):>7.2f}, {np.max(raw_vals):>7.2f}]"
        r_range = f"[{np.min(r_vals):>7.2f}, {np.max(r_vals):>7.2f}]"
        m_range = f"[{np.min(m_vals):>7.2f}, {np.max(m_vals):>7.2f}]"

        print(f"  {col:<25s} | {raw_range:>20s} | {r_range:>20s} | {m_range:>20s}")

    # 对第一个因子做详细的统计对比
    test_col = demo_factors[0]
    raw = panel[test_col].dropna().values
    r_out = robust_panel[test_col].dropna().values
    m_out = mad_panel[test_col].dropna().values

    print(f"\n  详细对比 ({test_col}):")
    print(f"    原始数据:  均值={np.mean(raw):.4f}  标准差={np.std(raw):.4f}  偏度={pd.Series(raw).skew():.3f}")
    print(f"    RobustZ:   均值={np.mean(r_out):.4f}  标准差={np.std(r_out):.4f}  偏度={pd.Series(r_out).skew():.3f}")
    print(f"    MAD+ZScore: 均值={np.mean(m_out):.4f}  标准差={np.std(m_out):.4f}  偏度={pd.Series(m_out).skew():.3f}")

    # 计算被裁剪的比例
    robust_clipped = (np.abs(r_out) >= 2.99).mean() * 100
    print(f"\n    RobustZ clip到[-3,3]被裁比例: {robust_clipped:.2f}%")

    print(f"\n  核心差异:")
    print(f"    RobustZScoreNorm: 输出严格限制在[-3,3], 对极端值硬截断")
    print(f"                     适合深度学习(梯度稳定, 激活函数不饱和)")
    print(f"    MAD+Z-Score:     先去极值再标准化, 输出范围取决于数据")
    print(f"                     适合树模型(不依赖数值范围, 只看排序)")


# ============================================================
# 第四部分: 因子相关性分析
# ============================================================

def analyze_factor_correlation(panel, feature_cols):
    """
    分析因子间相关性，找出高度冗余的因子对

    为什么需要分析因子相关性？
      1. 高相关性意味着信息冗余：两个高度相关的因子携带几乎相同的信息
      2. 对线性模型和神经网络：高共线性（multicollinearity）会导致模型不稳定
      3. 对树模型（XGBoost/LightGBM）：共线性影响不大，但冗余特征会增加计算量
      4. 高度相关的因子可以合并或剔除，降低模型复杂度

    经验法则：
      - |r| > 0.9：高度冗余，建议只保留一个
      - 0.7 < |r| < 0.9：中等相关，可以不处理（对树模型无害）
      - |r| < 0.7：基本独立
    """
    print("\n" + "=" * 80)
    print("第四部分: 因子相关性分析 (发现冗余因子)")
    print("=" * 80)

    valid_panel = panel[feature_cols].dropna()
    if len(valid_panel) < 100:
        print("  有效数据不足, 跳过相关性分析")
        return

    corr_matrix = valid_panel.corr()

    # 提取上三角的所有高相关对（只取上三角避免重复统计）
    high_corr_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.8:  # 0.8 为高相关阈值
                high_corr_pairs.append((feature_cols[i], feature_cols[j], r))

    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"\n  总因子数: {len(feature_cols)}")
    print(f"  高相关对(|r|>0.8): {len(high_corr_pairs)} 对")

    if high_corr_pairs:
        print(f"\n  Top 10 高相关因子对:")
        print(f"  {'因子A':<25s}  {'因子B':<25s}  {'相关系数':>8s}")
        print(f"  {'-'*25}  {'-'*25}  {'-'*8}")
        for fa, fb, r in high_corr_pairs[:10]:
            print(f"  {fa:<25s}  {fb:<25s}  {r:>8.3f}")

        print(f"\n  实践建议:")
        print(f"    - 相关性>0.9的因子对可以考虑只保留其中一个")
        print(f"    - 树模型(XGBoost)对共线性不敏感, 影响不大")
        print(f"    - 线性模型/神经网络对共线性敏感, 需要去冗余")
    else:
        print(f"\n  未发现高相关因子对, 因子体系正交性较好")


# ============================================================
# 第五部分: MASTER 市场信息数据对比
# ============================================================

def analyze_master_csv():
    """
    加载 MASTER 论文的 63 维市场信息 CSV，与我们的因子做维度对比

    MASTER 论文的一个重要创新是引入了"市场信息"：
      - 3 个指数（沪深300/中证500/中证800）x 21 个指标 = 63 维
      - 这些指标包括各指数的动量、波动率、换手率等
      - 通过 Gate 机制，这些市场信息动态调整各个因子的权重
      - 相当于让模型知道"当前市场环境如何，应该更看重哪些因子"

    这解释了为什么 MASTER 的因子维度（221 维）远多于我们的 52 维：
      52（技术因子）+ 0（市场信息）= 我们自己
      158（Alpha158）+ 63（市场信息）= MASTER
    """
    print("\n" + "=" * 80)
    print("第五部分: MASTER市场信息数据 (63维)")
    print("=" * 80)

    if not os.path.exists(MARKET_INFO_PATH):
        print(f"  [跳过] 未找到MASTER CSV: {MARKET_INFO_PATH}")
        print(f"  提示: 这是论文附带的中国A股指数级别数据, 非必需")
        return

    df = pd.read_csv(MARKET_INFO_PATH, header=[0, 1], index_col=0)
    print(f"\n  数据形状: {df.shape[0]} 天 x {df.shape[1]} 维")
    print(f"  时间范围: {df.index[0]} ~ {df.index[-1]}")

    feature_names = [col[1] for col in df.columns]

    index_map = {"SH000300": "沪深300", "SH000905": "中证500", "SH000906": "中证800"}
    print(f"\n  覆盖指数:")
    for code, name in index_map.items():
        count = sum(1 for fn in feature_names if code in fn)
        print(f"    {name}({code}): {count} 维")

    val_min, val_max = df.values.min(), df.values.max()
    print(f"\n  数据范围: [{val_min:.4f}, {val_max:.4f}]")
    # 检查数据是否已经被 RobustZScoreNorm+clip 处理过
    if abs(val_min + 3.0) < 0.01 and abs(val_max - 3.0) < 0.01:
        print(f"  --> 范围恰好为[-3, 3], 已经过RobustZScoreNorm+clip处理")

    print(f"\n  对比总结:")
    print(f"    {'维度':<15s} {'MASTER':>15s}  {'我们(L11)':>15s}")
    print(f"    {'-'*15} {'-'*15}  {'-'*15}")
    print(f"    {'因子数量':<15s} {'158(Alpha158)':>15s}  {'52(TA-Lib)':>15s}")
    print(f"    {'市场信息':<15s} {'63维(3指数x21)':>15s}  {'无':>15s}")
    print(f"    {'总特征维度':<15s} {'221':>15s}  {'52':>15s}")
    print(f"    {'预处理方法':<15s} {'RobustZScoreNorm':>15s}  {'MAD+Z-Score':>15s}")
    print(f"    {'数据来源':<15s} {'Qlib框架':>15s}  {'MySQL+TA-Lib':>15s}")


# ============================================================
# 主流程
# ============================================================

if __name__ == "__main__":
    print("MASTER数据与因子 - 数据探索与预处理对比实战")
    print("=" * 80)

    # 第一步：从 MySQL 加载数据并计算因子
    panel, feature_cols = load_and_compute()

    # 第二步到第四步：因子分析和预处理对比
    if panel is not None:
        diagnose_factor_distribution(panel, feature_cols)
        compare_preprocessing(panel, feature_cols)
        analyze_factor_correlation(panel, feature_cols)

    # 第五步：对比 MASTER 市场信息数据
    analyze_master_csv()

    print(f"\n{'=' * 80}")
    print("[完成] 数据探索结束, 接下来运行 3-XGBoost截面预测.py 进行截面预测")
