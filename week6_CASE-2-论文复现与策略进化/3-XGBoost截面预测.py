# -*- coding: utf-8 -*-
"""
3-XGBoost截面预测 —— 截面预测与 IC 评估实战

本脚本是"MASTER论文复现"系列的实践环节，目的是用 XGBoost（一种更传统、
更易上手的模型）来实践 MASTER 论文的评估方法论（IC/ICIR/RankIC/RankICIR）。

核心流程:
  1. 从 MySQL 加载 50 只 A 股大盘股（模拟 CSI300 的子集）
  2. 使用 feature_engine 计算 50+ 技术因子
  3. 单因子 IC 分析：逐个因子评估预测能力，找出最有预测力的因子
  4. 滚动 XGBoost 截面预测：用过去 N 天训练，预测未来第 5 日的收益率排名
  5. 用 IC/ICIR/RankIC/RankICIR 评估预测质量
  6. 与 MASTER 论文的 CSI300 结果做对比分析（明确差距和改进方向）

为什么用 XGBoost 而不是直接跑 MASTER？
  1. MASTER 的 221 维因子需要 Qlib 数据框架，部署成本高
  2. XGBoost 是量化界的"工业标准"，在中小型私募中仍是主力模型
  3. 本脚本的 52 维因子可以从 MySQL 自行计算，不依赖外部数据
  4. 通过对比 XGBoost 和 MASTER 的 IC 差异，可以量化"深度学习 vs 传统方法"的差距

什么是截面预测？
  - 时间序列预测: "用茅台过去 N 天的数据预测茅台明天的涨跌"
  - 截面预测: "用所有股票今天的数据预测它们明天的相对排名"
  截面预测更接近量化选股的真实场景——我们总是在所有股票中选择最好的那些。

IC（Information Coefficient）方法论:
  IC 本质上衡量的是"预测排名 vs 实际排名"的一致性。
  对于量化选股来说，排序的准确性比点预测的准确性更重要。
  一个策略只要排名准，就能选出超越平均的组合。

MASTER 论文: Li et al., "MASTER: Market-Guided Stock Transformer (AAAI 2024)"
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data_loader import load_stock_data
from db_config import execute_query
from feature_engine import calc_features, get_all_feature_cols

# ============================================================
# 配置
# ============================================================

START_DATE = '2023-01-01'
END_DATE = '2025-12-31'
TRAIN_WINDOW = 120        # 训练窗口：用过去 120 个交易日训练
PREDICT_HORIZON = 5       # 预测未来第几日的收益（5 日 = 一周）
ROLL_STEP = 5             # 滚动步长：每 5 天预测一次

# 关于参数选择的说明:
#   TRAIN_WINDOW=120: 约半年的交易日，足够的样本量但不超过"市场风格切换"的周期
#   PREDICT_HORIZON=5: 5 日收益是短线预测的典型目标，过长则信号衰减，过短则噪声太大
#   ROLL_STEP=5: 每 5 天滚动一次，与预测周期匹配

# 50 只 A 股代表性大盘股
# 为什么选 50 只而不是 CSI300 的全部 300 只？
#   1. 数据库查询和因子计算的时间可控
#   2. 50 只已经足够展示截面预测的方法论
#   3. 覆盖了金融/消费/科技/制造/医药等主要行业
STOCK_POOL = [
    '600519.SH', '000858.SZ', '601318.SH', '600036.SH', '000333.SZ',  # 消费+金融+制造
    '600900.SH', '601166.SH', '000001.SZ', '600276.SH', '601888.SH',  # 电力+银行+医药
    '002594.SZ', '300750.SZ', '601398.SH', '601939.SH', '600030.SH',  # 新能源+大行+券商
    '000651.SZ', '002415.SZ', '600309.SH', '600887.SH', '601012.SH',  # 家电+安防+化工
    '000568.SZ', '002304.SZ', '600050.SH', '601668.SH', '600000.SH',  # 白酒+通信+建筑
    '000002.SZ', '601857.SH', '600585.SH', '002352.SZ', '600104.SH',  # 地产+石油+水泥
    '601601.SH', '600690.SH', '601288.SH', '600028.SH', '601138.SH',  # 保险+家电+石化
    '002714.SZ', '300059.SZ', '002475.SZ', '600031.SH', '300760.SZ',  # 农业+金融IT+医疗
    '601899.SH', '600809.SH', '000725.SZ', '002230.SZ', '601919.SH',  # 矿业+白酒+面板
    '300015.SZ', '002142.SZ', '600438.SH', '601225.SH', '002027.SZ',  # 眼科+银行+光伏
]


# ============================================================
# 第一部分: 数据加载与因子计算
# ============================================================

def load_and_compute_factors():
    """
    从 MySQL 加载股票数据，用 feature_engine 计算 50+ 因子

    处理流程:
      1. 遍历 STOCK_POOL，逐只加载 K 线数据
      2. 使用 calc_features 计算 50+ 技术因子
      3. 计算未来 PREDICT_HORIZON 日的收益率作为预测目标（标签）
         - 使用 shift(-PREDICT_HORIZON) 避免未来信息泄露
         - pct_change(PREDICT_HORIZON) 计算 N 日总收益率
      4. 拼接成面板数据（Panel DataFrame）
      5. 过滤有缺失标签的样本

    关键设计:
      future_ret 的计算必须小心——shift(-N) 把未来的收益率"拉"到当前时间点
      这样在训练时才能用今天的因子去预测未来的收益。
      但回测时需要注意：当前日期之后 N 天的收益率实际上是未知的！
      这就是为什么我们做"滚动预测"——用历史数据训练，预测未来。

    返回:
        panel: DataFrame, 包含因子和未来收益率的面板数据
        feature_cols: list, 可用的特征列名
    """
    print("=" * 80)
    print("第一部分: 加载A股数据并计算因子")
    print("=" * 80)
    print(f"  股票池: {len(STOCK_POOL)} 只代表性A股大盘股")
    print(f"  日期范围: {START_DATE} ~ {END_DATE}")
    print(f"  因子引擎: L11 feature_engine (TA-Lib, 50+维)")

    t0 = time.time()
    all_frames = []
    loaded = 0

    for code in STOCK_POOL:
        try:
            df = load_stock_data(code, START_DATE, END_DATE)
            if len(df) < 200:
                continue

            feat_df = calc_features(df)
            # 计算未来 PREDICT_HORIZON 日的收益率
            # 为什么用 pct_change + shift？
            #   pct_change(N): 计算从 N 天前到今天的收益率
            #   shift(-N): 把 N 天后的收益率拉到今天的位置
            # 组合效果: 今天的位置上存储的是"从今天到 N 天后的收益率"
            feat_df['future_ret'] = feat_df['close'].pct_change(PREDICT_HORIZON).shift(-PREDICT_HORIZON)
            feat_df['stock_code'] = code
            feat_df['trade_date'] = feat_df.index
            all_frames.append(feat_df)
            loaded += 1
        except Exception as e:
            print(f"  [跳过] {code}: {e}")

    elapsed = time.time() - t0
    print(f"\n  成功加载: {loaded}/{len(STOCK_POOL)} 只, 耗时: {elapsed:.1f}s")

    if loaded < 10:
        print("  [错误] 有效股票不足10只, 无法进行截面分析")
        return None, None

    panel = pd.concat(all_frames, ignore_index=True)
    panel = panel.dropna(subset=['future_ret'])  # 删除没有标签的行

    feature_cols = get_all_feature_cols()
    feature_cols = [c for c in feature_cols if c in panel.columns]

    dates = sorted(panel['trade_date'].unique())
    daily_counts = panel.groupby('trade_date')['stock_code'].nunique()

    print(f"  面板大小: {len(panel):,} 行 x {len(feature_cols)} 个因子")
    print(f"  交易日数: {len(dates)} ({dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')})")
    print(f"  平均每日股票: {daily_counts.mean():.0f} 只")
    print(f"  未来{PREDICT_HORIZON}日收益率: 均值={panel['future_ret'].mean()*100:.3f}%, "
          f"标准差={panel['future_ret'].std()*100:.2f}%")

    return panel, feature_cols


# ============================================================
# 第二部分: 单因子 IC 分析
# ============================================================

def analyze_factor_ic(panel, feature_cols):
    """
    单因子 IC 分析 —— 评估每个因子独立预测未来收益的能力

    为什么做单因子 IC 分析？
      1. 筛选有效因子：剔除那些与未来收益完全无关的因子
      2. 理解因子逻辑：IC 高的因子背后有经济学/行为金融学解释
      3. 诊断因子质量：ICIR 比 IC 更重要——稳定的中等预测优于不稳定的强预测
      4. 与文献对比：可以检查我们的因子是否与学术研究一致（如动量因子的 IC 应为正）

    计算逻辑:
      对每个交易日，计算所有股票的因子值和未来收益率之间的截面相关性，
      然后对时间序列取均值（IC）和均值/标准差（ICIR）。

    参数:
        panel: DataFrame, 面板数据
        feature_cols: list, 特征列名

    返回:
        results_df: DataFrame, 每个因子的 IC/ICIR/RankIC/RankICIR
    """
    print("\n" + "=" * 80)
    print("第二部分: 单因子IC分析 (哪些因子最有预测力?)")
    print("=" * 80)

    dates = sorted(panel['trade_date'].unique())
    factor_stats = {col: {'ics': [], 'rics': []} for col in feature_cols}

    # 逐日计算截面 IC
    # 为什么逐日而非整体？整体计算会混入时间序列效应（比如趋势性上涨导致所有因子都"有效"）
    # 逐日计算后取平均，得到的是"纯截面预测能力"
    for dt in dates:
        daily = panel[panel['trade_date'] == dt]
        if len(daily) < 10:
            continue

        for col in feature_cols:
            valid = daily[[col, 'future_ret']].dropna()
            if len(valid) < 10:
                continue

            # Pearson 相关系数（线性关系）
            ic = valid[col].corr(valid['future_ret'])
            # Spearman 秩相关系数（排序关系，对异常值不敏感）
            ric, _ = spearmanr(valid[col], valid['future_ret'])

            if not np.isnan(ic):
                factor_stats[col]['ics'].append(ic)
            if not np.isnan(ric):
                factor_stats[col]['rics'].append(ric)

    results = []
    for col in feature_cols:
        ics = factor_stats[col]['ics']
        rics = factor_stats[col]['rics']
        if len(ics) < 30:  # 至少 30 个交易日的数据才统计
            continue

        ic_mean = np.mean(ics)
        ic_std = np.std(ics)
        ric_mean = np.mean(rics)
        ric_std = np.std(rics)

        results.append({
            'factor': col,
            'IC': ic_mean,
            'ICIR': ic_mean / ic_std if ic_std > 0 else 0,        # 信息比率
            'RankIC': ric_mean,
            'RankICIR': ric_mean / ric_std if ric_std > 0 else 0, # 排序信息比率
            'IC_positive': sum(1 for x in ics if x > 0) / len(ics),  # IC 为正的天数比例
            'n_days': len(ics),
        })

    results_df = pd.DataFrame(results)
    results_df['abs_ICIR'] = results_df['ICIR'].abs()
    results_df = results_df.sort_values('abs_ICIR', ascending=False)

    # 打印 Top 15 因子的 IC 详情
    print(f"\n因子IC排名 (Top 15, 按|ICIR|排序):")
    print(f"  {'因子':<28} {'IC':>8} {'ICIR':>8} {'RankIC':>8} {'RICIR':>8} {'IC>0':>6}")
    print("  " + "-" * 75)
    for _, row in results_df.head(15).iterrows():
        print(f"  {row['factor']:<28} {row['IC']:>8.4f} {row['ICIR']:>8.4f} "
              f"{row['RankIC']:>8.4f} {row['RankICIR']:>8.4f} {row['IC_positive']:>5.1%}")

    # 按因子类别汇总平均 ICIR
    from feature_engine import FACTOR_TAXONOMY
    print(f"\n各因子类别平均|ICIR|:")
    for cat_key, cat_info in FACTOR_TAXONOMY.items():
        cat_factors = [r for r in results if r['factor'] in cat_info['features']]
        if cat_factors:
            avg_icir = np.mean([abs(f['ICIR']) for f in cat_factors])
            best = max(cat_factors, key=lambda x: abs(x['ICIR']))
            print(f"  {cat_info['name']:<14} 平均|ICIR|={avg_icir:.3f}  "
                  f"最佳: {best['factor']} (ICIR={best['ICIR']:.3f})")

    return results_df


# ============================================================
# 第三部分: 滚动 XGBoost 截面预测
# ============================================================

def rolling_prediction(panel, feature_cols):
    """
    滚动 XGBoost 截面预测 —— 模拟真实的量化选股流程

    这是本脚本的核心功能，模拟了量化交易中的"滚动训练-预测"流程：
      1. 用过去 N 天（TRAIN_WINDOW）的数据训练 XGBoost
      2. 预测下一个交易日所有股票的未来收益率
      3. 计算预测值和实际值之间的 IC
      4. 向前滚动 ROLL_STEP 天，重复 1-3

    为什么用滚动而非固定训练集？
      A 股市场存在明显的风格切换（如 2021 年成长风格 vs 2022 年价值风格），
      固定训练集可能包含过时的市场模式。滚动训练让模型始终学习最新的市场特征。

    参数:
        panel: DataFrame, 面板数据
        feature_cols: list, 特征列名

    返回:
        metrics: dict, 包含 IC/ICIR/RankIC/RankICIR 等评估指标
        如果失败返回 None
    """
    print("\n" + "=" * 80)
    print("第三部分: 滚动XGBoost截面预测")
    print("=" * 80)

    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("[错误] 需要安装 xgboost: pip install xgboost")
        return None

    dates = sorted(panel['trade_date'].unique())

    if len(dates) < TRAIN_WINDOW + 20:
        print(f"  [错误] 交易日数({len(dates)})不足, 需要至少 {TRAIN_WINDOW + 20}")
        return None

    print(f"  训练窗口: {TRAIN_WINDOW} 交易日")
    print(f"  预测目标: 未来{PREDICT_HORIZON}日收益率")
    print(f"  滚动步长: 每{ROLL_STEP}天预测一次")

    predict_indices = list(range(TRAIN_WINDOW, len(dates), ROLL_STEP))
    print(f"  预计预测: {len(predict_indices)} 次")

    daily_ics = []   # 每次预测的 IC
    daily_rics = []  # 每次预测的 RankIC
    t0 = time.time()

    for step, pred_idx in enumerate(predict_indices):
        # 训练集：预测日前 TRAIN_WINDOW 天
        train_dates = dates[pred_idx - TRAIN_WINDOW: pred_idx]
        # 预测日（用训练好的模型预测这天所有股票的未来收益）
        pred_date = dates[pred_idx]

        train_data = panel[panel['trade_date'].isin(train_dates)]
        test_data = panel[panel['trade_date'] == pred_date]

        if len(test_data) < 5 or len(train_data) < 100:
            continue

        X_train = train_data[feature_cols].fillna(0).values
        y_train = train_data['future_ret'].fillna(0).values
        X_test = test_data[feature_cols].fillna(0).values
        y_test = test_data['future_ret'].values

        # XGBoost 参数说明:
        #   n_estimators=50: 树的数量，50 棵足够（更多树容易过拟合）
        #   max_depth=4: 树深 4，防止单棵树的过拟合
        #   learning_rate=0.05: 学习率，较小的学习率配合较少的树
        #   subsample=0.8: 行采样，每棵树只用 80% 的样本（增加随机性，防止过拟合）
        #   colsample_bytree=0.8: 列采样，每棵树只用 80% 的特征
        model = XGBRegressor(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        valid_mask = ~np.isnan(y_test)
        if valid_mask.sum() < 5:
            continue

        # 计算截面 IC
        ic = np.corrcoef(y_pred[valid_mask], y_test[valid_mask])[0, 1]
        ric, _ = spearmanr(y_pred[valid_mask], y_test[valid_mask])

        if not np.isnan(ic):
            daily_ics.append(ic)
        if not np.isnan(ric):
            daily_rics.append(ric)

        if (step + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  进度: {step+1}/{len(predict_indices)} "
                  f"({elapsed:.0f}s, 累计IC均值={np.mean(daily_ics):.4f})")

    elapsed = time.time() - t0
    print(f"  完成: {len(daily_ics)} 次有效预测, 耗时 {elapsed:.1f}s")

    if not daily_ics:
        print("  [错误] 没有有效的预测结果")
        return None

    # 计算最终评估指标
    ic_mean = np.mean(daily_ics)
    ic_std = np.std(daily_ics)
    icir = ic_mean / ic_std if ic_std > 0 else 0
    ric_mean = np.mean(daily_rics)
    ric_std = np.std(daily_rics)
    ricir = ric_mean / ric_std if ric_std > 0 else 0
    ic_positive = sum(1 for x in daily_ics if x > 0) / len(daily_ics)

    metrics = {
        'IC': ic_mean,
        'ICIR': icir,
        'RankIC': ric_mean,
        'RankICIR': ricir,
    }

    print(f"\n--- XGBoost截面预测结果 ---")
    print(f"  IC:        {ic_mean:.4f}")
    print(f"  ICIR:      {icir:.4f}")
    print(f"  RankIC:    {ric_mean:.4f}")
    print(f"  RankICIR:  {ricir:.4f}")
    print(f"  IC>0占比:  {ic_positive:.1%} ({sum(1 for x in daily_ics if x > 0)}/{len(daily_ics)})")
    print(f"  IC最大值:  {max(daily_ics):.4f}")
    print(f"  IC最小值:  {min(daily_ics):.4f}")

    return metrics


# ============================================================
# 第四部分: 与 MASTER 论文对比
# ============================================================

def compare_with_master(our_metrics):
    """
    与 MASTER 论文 CSI300 结果对比

    本函数诚实地展示 XGBoost + 52 因子与 MASTER + 221 因子的差距，
    并分析差距的来源——帮我们理解"从 XGBoost 到 MASTER"的改进路径。

    预期结果：
      XGBoost + 50 只股票的 IC 通常在 0.02~0.04 左右，
      MASTER + 300 只 + 221 维 + Transformer 的 IC 在 0.06~0.08。
      差距不仅来自模型，还来自因子维度和股票数量。

    参数:
        our_metrics: dict, 我们 XGBoost 模型的评估指标
    """
    print("\n" + "=" * 80)
    print("第四部分: 与MASTER论文(CSI300)对比")
    print("=" * 80)

    # MASTER 在 CSI300 上的典型结果（论文 Table 2 的范围）
    # 为什么给范围而非固定值？因为论文报告了 5 次不同 seed 的均值和标准差
    master_range = {
        'IC':       (0.050, 0.080),
        'ICIR':     (0.400, 0.700),
        'RankIC':   (0.080, 0.120),
        'RankICIR': (0.700, 1.100),
    }

    print(f"\n  {'指标':<12} {'我们(50只)':>12} {'MASTER(300只)':>15} {'差距':>8} {'评估'}")
    print("  " + "-" * 60)

    for key in ['IC', 'ICIR', 'RankIC', 'RankICIR']:
        ours = our_metrics.get(key, 0)
        m_lo, m_hi = master_range[key]
        m_mid = (m_lo + m_hi) / 2

        if abs(ours) >= m_lo:
            assessment = '达标'          # 达到了 MASTER 的下限
        elif abs(ours) >= m_lo * 0.6:
            assessment = '接近'          # 达到了 MASTER 的 60%
        else:
            assessment = '差距大'        # 差距明显

        gap = abs(ours) - m_mid
        print(f"  {key:<12} {ours:>12.4f} {m_lo:.3f}~{m_hi:.3f}      {gap:>+8.4f} {assessment}")

    print(f"""
  条件差异分析:
    1. 股票数量: 50只 vs MASTER 300只
       -> 50只截面较小, IC的统计噪声更大
    2. 因子维度: 52维 vs MASTER 221维(158+63)
       -> MASTER覆盖了更多K线形态和回归因子
    3. 模型架构: XGBoost(截面独立) vs Transformer(双注意力)
       -> MASTER的S-Attention能捕捉板块联动, T-Attention捕捉多日模式
    4. 特征选择: 固定因子 vs Gate动态调整
       -> MASTER根据市场环境自动调整因子权重

  实践启示:
    - XGBoost + 50因子在A股上{'' if abs(our_metrics.get('IC',0)) > 0.03 else '仍可'}产生有效IC信号
    - 要进一步提升, 可从三个方向:
      a) 扩大因子库 (加入Alpha158中我们缺少的因子)
      b) 引入市场状态特征 (类似Gate的63维市场信息)
      c) 升级模型架构 (Transformer + 时序/截面注意力)""")


# ============================================================
# 主流程
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print("  截面预测与IC评估 -- XGBoost实践MASTER论文评估方法论")
    print("  论文: MASTER (AAAI 2024), 目标市场: 中国A股(CSI300/CSI800)")
    print("=" * 80)

    # 第一步: 加载数据并计算因子
    result = load_and_compute_factors()
    if result is None:
        sys.exit(1)

    panel, feature_cols = result

    # 第二步: 单因子 IC 分析
    factor_ic_df = analyze_factor_ic(panel, feature_cols)

    # 第三步: 滚动 XGBoost 截面预测
    xgb_metrics = rolling_prediction(panel, feature_cols)

    # 第四步: 与 MASTER 论文对比
    if xgb_metrics:
        compare_with_master(xgb_metrics)

    print(f"\n{'=' * 80}")
    print("[完成] 3-XGBoost截面预测.py 运行结束")
