# -*- coding: utf-8 -*-
"""
XGBoost 涨跌二分类预测模型（教学演示脚本）

本脚本是"机器学习因子挖掘"系列的第三步，使用 XGBoost 模型
对贵州茅台和中芯国际进行"次日涨跌预测"。

完整流程：
  1. 加载日K线数据
  2. 计算 50+ 技术因子作为特征
  3. 构建标签（次日涨=1，跌=0）
  4. MAD 去极值 + Z-score 标准化
  5. 滚动训练与预测（避免未来信息泄露）
  6. 评估模型性能（AUC/准确率/精确率/召回率/F1）
  7. 混淆矩阵分析
  8. 按月统计准确率变化

教学重点：
  - XGBoost 的工作原理和参数含义
  - 滚动窗口预测如何避免未来信息泄露
  - AUC 为什么比准确率更能反映模型质量
  - 不同股票的可预测性差异

关于 AUC（Area Under ROC Curve）：
  这是分类模型最重要的指标，衡量模型区分涨跌的能力：
    - AUC = 0.5：随机猜测，模型无效
    - AUC = 0.55：弱预测能力，但可作为因子使用
    - AUC = 0.60：有实际意义的预测能力
    - AUC = 0.70+：强预测能力（在股票上几乎不可能达到）
  相比之下，准确率容易受涨跌不平衡的"误导"。

运行方式: python 3-XGBoost涨跌预测.py

依赖:
  - data_loader.py: 加载日K线
  - feature_engine.py: 计算因子和预处理
  - ml_engine.py: 标签构建、滚动预测、评估
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from data_loader import load_stock_data
from feature_engine import calc_features, preprocess_features, get_all_feature_cols
from ml_engine import make_labels, rolling_train_predict, evaluate_classification


# ============================================================
# 配置区
# ============================================================

# 教学用两只标的：一只消费龙头（茅台），一只科技龙头（中芯国际）
# 它们的市场特性不同，可预测性也不同
XGB_STOCKS = {
    '600519.SH': '贵州茅台',
    '688981.SH': '中芯国际',
}

START_DATE = '2023-01-01'
END_DATE = '2025-12-31'

# 预测时间窗口：1 = 预测次日涨跌
# 如果改成 5，就是预测"5个交易日后的涨跌"
# 窗口越大，预测难度越大（不确定性增加），但信号周期更长
LABEL_HORIZON = 1


def prepare_stock_features(stock_code, start_date, end_date, min_bars=120):
    """
    一站式特征准备：加载 -> 计算因子 -> 预处理。

    参数:
        stock_code: 股票代码
        start_date: 起始日期
        end_date: 结束日期
        min_bars: 最小交易天数（少于这个值不处理）

    返回:
        df: 含特征列的 DataFrame
        feature_cols: 可用特征列名列表
    """
    df = load_stock_data(stock_code, start_date, end_date)
    if len(df) < min_bars:
        raise ValueError(f'{stock_code} 有效交易日 {len(df)} < {min_bars}')
    df = calc_features(df)
    feature_cols = [c for c in get_all_feature_cols() if c in df.columns]
    df = preprocess_features(df, feature_cols)
    return df, feature_cols


# ============================================================
# 单只股票：特征工程 + 滚动预测 + 评估
# ============================================================

def run_single_stock(stock_code, stock_name):
    """
    对单只股票执行完整的 XGBoost 滚动预测流程。

    这是本脚本的核心函数，展示了 ML 因子挖掘的完整流程：
      数据 -> 特征 -> 标签 -> 训练 -> 预测 -> 评估 -> 分析

    参数:
        stock_code: 股票代码
        stock_name: 股票名称（用于显示）

    返回:
        dict: 包含评估指标、月度数据和预测结果
    """

    print(f"\n{'='*60}")
    print(f"  {stock_name} ({stock_code})")
    print(f"{'='*60}")

    # --- 第1步：加载数据 ---
    print(f"\n[1] 加载数据: {START_DATE} ~ {END_DATE}")
    df, feature_cols = prepare_stock_features(stock_code, START_DATE, END_DATE)
    print(f"    共 {len(df)} 个交易日, 价格区间: {df['close'].min():.2f} ~ {df['close'].max():.2f}")

    # --- 第2步：特征计算完成 ---
    print("[2] 计算技术特征 + 预处理（已完成）")
    print(f"    特征数量: {len(feature_cols)}")

    # --- 第3步：构建标签 ---
    # 监督学习的三要素：特征、标签、模型
    # 这里构建的是"标签"——告诉模型什么是正确的预测
    # make_labels 用 shift(-horizon) 把未来的涨跌拉到今天
    # 因此，今天的特征对应的是未来 horizon 天的涨跌
    print(f"[3] 构建标签: 未来第{LABEL_HORIZON}日涨=1, 跌=0 (相对今日收盘)")
    df['label'] = make_labels(df, horizon=LABEL_HORIZON, method='binary')
    df.dropna(subset=['label'], inplace=True)
    label_dist = df['label'].value_counts().sort_index()
    print(f"    标签分布: 跌(0)={label_dist.get(0,0)}, 涨(1)={label_dist.get(1,0)}, "
          f"涨占比={label_dist.get(1,0)/len(df)*100:.1f}%")

    # --- 第4步：滚动预测 ---
    # 关键差异：这里用的是"滚动窗口"而非随机划分
    # 普通机器学习：随机取 80% 训练，20% 测试
    # 时序预测：用 1~120 天训练，预测 121 天，然后滑动窗口
    # 详见 ml_engine.py 中 rolling_train_predict 的文档
    print("[4] XGBoost 滚动预测 (train_days=120, retrain_interval=20)")
    df_reset = df.reset_index()
    pred_df = rolling_train_predict(
        df_reset, feature_cols, label_col='label',
        model_type='xgboost', train_days=120, retrain_interval=20,
        verbose=True,
    )
    print(f"    预测样本数: {len(pred_df)}")

    # --- 第5步：整体评估 ---
    # AUC 是最重要的指标
    # 在股票预测中，AUC > 0.55 已经说明模型学到了有意义的信号
    print("\n[5] 整体评估指标")
    metrics = evaluate_classification(
        pred_df['y_true'].values,
        pred_df['y_pred'].values,
        pred_df['y_prob'].values,
    )
    print(f"    AUC:       {metrics['auc']:.4f}")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1:        {metrics['f1']:.4f}")

    # --- 第6步：混淆矩阵 ---
    # 混淆矩阵有四个格子：
    #   TN（真跌，预测跌）：模型正确预测了下跌
    #   FP（假涨，预测涨但实际跌）：模型误报上涨（最危险！会亏钱）
    #   FN（假跌，预测跌但实际涨）：模型错过了上涨机会
    #   TP（真涨，预测涨且实际涨）：模型正确预测了上涨
    print("\n[6] 混淆矩阵")
    cm = confusion_matrix(pred_df['y_true'], pred_df['y_pred'])
    print(f"              预测跌  预测涨")
    print(f"    实际跌    {cm[0,0]:>5d}   {cm[0,1]:>5d}")
    print(f"    实际涨    {cm[1,0]:>5d}   {cm[1,1]:>5d}")

    # --- 第7步：按月统计准确率 ---
    # 模型在不同月份的预测能力是否稳定？
    # 如果某些月份准确率突然下降，可能说明：
    #   - 市场风格发生了切换
    #   - 模型过拟合了特定市场环境
    #   - 需要重新训练或调整特征
    print("\n[7] 按月准确率变化")
    pred_df['date'] = pd.to_datetime(pred_df['date'])
    pred_df['month'] = pred_df['date'].dt.to_period('M')
    monthly = pred_df.groupby('month').apply(
        lambda g: pd.Series({
            'accuracy': (g['y_true'] == g['y_pred']).mean(),
            'samples': len(g),
            'auc': evaluate_classification(
                g['y_true'].values, g['y_pred'].values, g['y_prob'].values
            )['auc'] if len(g['y_true'].unique()) > 1 else float('nan'),
        })
    )
    print(f"    {'月份':<12s} {'样本':>6s} {'Accuracy':>10s} {'AUC':>8s}")
    print(f"    {'-'*38}")
    for idx, row in monthly.iterrows():
        auc_str = f"{row['auc']:.4f}" if not np.isnan(row['auc']) else '  N/A '
        print(f"    {str(idx):<12s} {int(row['samples']):>6d} {row['accuracy']:>10.4f} {auc_str:>8s}")

    return {
        'stock_code': stock_code,
        'stock_name': stock_name,
        'metrics': metrics,
        'monthly': monthly,
        'pred_df': pred_df,
        'n_samples': len(pred_df),
    }


# ============================================================
# 两只股票对比
# ============================================================

def compare_stocks(results_list):
    """
    对比多只股票的预测结果。

    为什么要对比？
      不同股票的"可预测性"不同：
        - 流动性好的大盘股（如茅台）：价格更有效，预测更难
        - 波动大的股票：噪声更多，但信号也可能更强
      通过对比，我们可以了解模型在不同股票上的表现差异。
    """
    print(f"\n{'='*60}")
    print("  XGBoost 预测结果对比")
    print(f"{'='*60}")

    header = f"{'股票':<12s} {'AUC':>8s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s} {'样本数':>8s}"
    print(f"\n    {header}")
    print(f"    {'-'*len(header)}")

    for r in results_list:
        m = r['metrics']
        print(f"    {r['stock_name']:<12s} "
              f"{m['auc']:>8.4f} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['f1']:>8.4f} {r['n_samples']:>8d}")

    # 月度准确率波动对比（稳定性分析）
    print("\n    月度准确率统计:")
    for r in results_list:
        acc_series = r['monthly']['accuracy']
        print(f"    {r['stock_name']}: "
              f"均值={acc_series.mean():.4f}, "
              f"标准差={acc_series.std():.4f}, "
              f"最高={acc_series.max():.4f}, "
              f"最低={acc_series.min():.4f}")


# ============================================================
# 可预测性分析
# ============================================================

def analyze_predictability(results_list):
    """
    分析不同类型股票的可预测性差异。

    影响股票可预测性的因素：
      - 市值：大盘股更有效，小盘股可预测性更强
      - 波动率：波动越大的股票，短期可预测性越强
      - 流动性：高流动性股票的定价更有效
      - 行业：某些行业（如周期股）的趋势持续性更强
    """
    print(f"\n{'='*60}")
    print("  可预测性分析")
    print(f"{'='*60}")
    print('  (可预测性的分析维度详见课件 Part1)')

    for r in results_list:
        pred_df = r['pred_df']
        acc = r['metrics']['accuracy']
        monthly_std = r['monthly']['accuracy'].std()
        print(f"\n    {r['stock_name']}:")
        print(f"      - 整体准确率: {acc:.4f}")
        print(f"      - 月度准确率波动(std): {monthly_std:.4f}")
        stability = "稳定" if monthly_std < 0.08 else "波动较大"
        print(f"      - 预测稳定性: {stability}")


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 60)
    title = "次日涨跌" if LABEL_HORIZON == 1 else f"未来第{LABEL_HORIZON}日涨跌"
    print(f"  XGBoost {title} 二分类预测模型")
    print("=" * 60)
    stocks = XGB_STOCKS
    print(f"  数据区间: {START_DATE} ~ {END_DATE}")
    print(f"  标的（{len(stocks)}只）: {', '.join(f'{v}({k})' for k, v in stocks.items())}")
    print(f"  标签定义: 未来第{LABEL_HORIZON}个交易日收盘 > 今日收盘 => 1(涨), 否则 => 0(跌)")
    print(f"  模型: XGBoost | 滚动窗口=120天 | 重训间隔=20天")

    results = []
    for code, name in stocks.items():
        try:
            r = run_single_stock(code, name)
            results.append(r)
        except Exception as e:
            print(f"\n  [错误] {name}({code}) 处理失败: {e}")

    if len(results) < 2:
        print("\n  不足两只成功完成的股票, 跳过对比分析")
    else:
        compare_stocks(results)
        analyze_predictability(results)

    # --- 最终结论 ---
    if results:
        avg_acc = np.mean([r['metrics']['accuracy'] for r in results])
        print(f"\n{'='*60}")
        print(f"  结论")
        print(f"{'='*60}")
        hz = "次日" if LABEL_HORIZON == 1 else f"未来第{LABEL_HORIZON}日"
        print(f"\n  XGBoost二分类预测模型完成，{hz}方向准确率约{avg_acc*100:.0f}%，")
        print(f"  输出的概率值(0~1)就是'上涨概率因子'。")
        print(f"\n  该因子可作为多因子选股模型的alpha信号之一，")
        print(f"  与基本面因子、动量因子等组合使用，构建综合选股策略。")
    print()


if __name__ == '__main__':
    main()
