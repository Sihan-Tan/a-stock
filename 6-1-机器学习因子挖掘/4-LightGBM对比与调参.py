# -*- coding: utf-8 -*-
"""
LightGBM 对比实验与 Optuna 超参数优化（教学演示脚本）

本脚本是"机器学习因子挖掘"系列的第四步，核心目标是：
  1. 对比 LightGBM 和 XGBoost 在股票预测上的性能差异
  2. 使用 Optuna 自动搜索 LightGBM 的最优超参数
  3. 验证调优后的 LightGBM 是否优于默认参数

完整流程（7 个部分）:
  [第一部分] 数据准备 - 加载数据、特征工程、标签构建
  [第二部分] LightGBM 默认参数 - 滚动预测
  [第三部分] XGBoost 滚动预测（用来做对比基准）
  [第四部分] Optuna 超参数搜索 - 50 次试验找最优参数
  [第五部分] 最优参数 LightGBM - 重新滚动预测
  [第六部分] Purged K-Fold 交叉验证 - 详细评估模型稳定性
  [第七部分] XGBoost vs LightGBM 对比表

LightGBM vs XGBoost 核心差异：
  - LightGBM 使用"叶子-wise"生长策略（按叶子分裂），XGBoost 用"层-wise"（按层生长）
  - LightGBM 引入 GOSS（单边梯度采样）加速训练
  - LightGBM 训练速度通常比 XGBoost 快 2-3 倍
  - 在中小数据集上，XGBoost 有时精度略高；在大型数据集上，LightGBM 表现更好

什么是超参数优化？
  超参数（如学习率、树深度）是在训练前设置的，模型无法自己学习。
  手工调参效率低且容易遗漏最优组合。Optuna 通过贝叶斯优化自动搜索：
    - 定义搜索空间（每个参数的范围）
    - 每次试验选取一组参数组合
    - 用 Purged K-Fold AUC 作为优化目标
    - 逐步收敛到最优参数组合

运行方式: python 4-LightGBM对比与调参.py

依赖:
  - data_loader.py: 加载日K线
  - feature_engine.py: 计算因子和预处理
  - ml_engine.py: 标签构建、滚动预测、评估、交叉验证
  - optuna: 超参数优化框架
"""

import time
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import confusion_matrix

from data_loader import load_stock_data
from feature_engine import calc_features, preprocess_features, get_all_feature_cols
from ml_engine import (make_labels, rolling_train_predict, evaluate_classification,
                       purged_kfold_cv)

# 设置 Optuna 日志级别为 WARNING，减少不必要的输出
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================
# 配置区
# ============================================================

# 与脚本 3 保持一致，确保对比公平
STOCKS = {
    '600519.SH': '贵州茅台',
    '688981.SH': '中芯国际',
}
START_DATE = '2023-01-01'
END_DATE = '2025-12-31'


# ============================================================
# 数据准备（与脚本3共用逻辑，保持一致的实验条件）
# ============================================================

def prepare_data(stock_code, stock_name):
    """
    加载数据 -> 特征工程 -> 标签构建。

    与脚本3的 prepare_stock_features 逻辑完全一致，
    确保 XGBoost 和 LightGBM 在完全相同的数据上对比。

    返回:
        df: 含特征和标签的 DataFrame
        feature_cols: 特征列名列表
    """
    print(f"\n  加载 {stock_name}({stock_code}) ...")
    df = load_stock_data(stock_code, START_DATE, END_DATE)
    print(f"    交易日: {len(df)}, 价格区间: {df['close'].min():.2f} ~ {df['close'].max():.2f}")

    df = calc_features(df)
    feature_cols = get_all_feature_cols()
    feature_cols = [c for c in feature_cols if c in df.columns]
    df = preprocess_features(df, feature_cols)

    df['label'] = make_labels(df, horizon=1, method='binary')
    df.dropna(subset=['label'], inplace=True)

    return df, feature_cols


# ============================================================
# 滚动预测并计时
# ============================================================

def run_rolling(df, feature_cols, model_type, params=None, verbose=True):
    """
    执行滚动预测并返回结果 + 耗时。

    这个函数封装了滚动预测的公共逻辑，同时记录训练耗时，
    方便后续对比 XGBoost 和 LightGBM 在速度上的差异。

    参数:
        df: 原始数据
        feature_cols: 特征列
        model_type: 'xgboost' 或 'lightgbm'
        params: 模型参数（None 使用默认参数）
        verbose: 是否显示进度

    返回:
        (pred_df, metrics, elapsed): 预测结果、评估指标、耗时（秒）
    """
    df_reset = df.reset_index()
    t0 = time.time()
    pred_df = rolling_train_predict(
        df_reset, feature_cols, label_col='label',
        model_type=model_type, train_days=120, retrain_interval=20,
        params=params, verbose=verbose,
    )
    elapsed = time.time() - t0

    metrics = evaluate_classification(
        pred_df['y_true'].values,
        pred_df['y_pred'].values,
        pred_df['y_prob'].values,
    )
    return pred_df, metrics, elapsed


# ============================================================
# Optuna 超参数优化
# ============================================================

def optuna_lgb_search(X, y, n_trials=50):
    """
    用 Optuna 搜索 LightGBM 最优超参数。

    搜索策略：Tree-structured Parzen Estimator（TPE）
    Optuna 的默认搜索算法，比随机搜索更高效：
      - 先用随机搜索探索参数空间
      - 根据历史结果，建立"哪些参数组合表现好"的概率模型
      - 在可能产生好结果的区域密集采样
      - 逐步收敛到最优区域

    搜索空间说明：
      num_leaves (15-63): 叶子节点数，越大模型越复杂
      learning_rate (0.01-0.15): 学习率，越小越稳定但需要更多树
      max_depth (3-8): 树深度，限制复杂度
      min_child_samples (10-50): 叶子最少样本数，越大越保守
      subsample (0.6-1.0): 样本采样比例
      colsample_bytree (0.6-1.0): 特征采样比例
      reg_alpha (0.0-1.0): L1 正则化
      reg_lambda (0.0-1.0): L2 正则化

    优化目标：Purged K-Fold 平均 AUC
    为什么用 Purged K-Fold 而不是普通交叉验证？
    见 ml_engine.py 中 purged_kfold_cv 的详细说明。

    参数:
        X: 特征矩阵
        y: 标签数组
        n_trials: 试验次数（50 次约需 5-10 分钟）

    返回:
        optuna.Study 对象，含最优参数和最优值
    """
    def objective(trial):
        # trial.suggest_* 是 Optuna 的"参数建议API"
        # 每次 trial 会从搜索空间中选取一组参数
        params = {
            'num_leaves':        trial.suggest_int('num_leaves', 15, 63),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'max_depth':         trial.suggest_int('max_depth', 3, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha':         trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda':        trial.suggest_float('reg_lambda', 0.0, 1.0),
            'n_estimators': 200,
            'random_state': 42,
            'verbose': -1,
        }

        # Purged K-Fold 会返回每折的评估结果
        fold_results = purged_kfold_cv(X, y, model_type='lightgbm',
                                        n_splits=5, gap=5, params=params)
        if not fold_results:
            return 0.5  # 没有有效折时返回 0.5（随机猜测水平）

        # 用平均 AUC 作为优化目标
        mean_auc = np.mean([f['auc'] for f in fold_results])
        return mean_auc

    # direction='maximize' 表示我们要最大化 AUC
    # TPESampler 是 Optuna 的默认采样器，基于贝叶斯优化
    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 70)
    print("  LightGBM 对比实验与 Optuna 超参数优化")
    print("=" * 70)
    print(f"  数据区间: {START_DATE} ~ {END_DATE}")
    print(f"  目标股票: {', '.join(f'{v}({k})' for k,v in STOCKS.items())}")

    # ----------------------------------------------------------
    # 第一部分：数据准备
    # ----------------------------------------------------------
    print(f"\n{'='*70}")
    print("  [第一部分] 数据准备")
    print(f"{'='*70}")

    stock_data = {}
    for code, name in STOCKS.items():
        df, feature_cols = prepare_data(code, name)
        stock_data[code] = {'df': df, 'feature_cols': feature_cols, 'name': name}

    # ----------------------------------------------------------
    # 第二部分：LightGBM 默认参数 - 滚动预测
    # ----------------------------------------------------------
    # 先用 LightGBM 的默认参数做滚动预测，作为"基准线"
    # 后续调优后的结果将与这个基准线对比
    print(f"\n{'='*70}")
    print("  [第二部分] LightGBM 默认参数 - 滚动预测")
    print(f"{'='*70}")

    lgb_default_results = {}
    for code, info in stock_data.items():
        print(f"\n  --- {info['name']} ({code}) ---")
        pred_df, metrics, elapsed = run_rolling(
            info['df'], info['feature_cols'], model_type='lightgbm')
        lgb_default_results[code] = {
            'pred_df': pred_df, 'metrics': metrics, 'elapsed': elapsed}
        print(f"    AUC={metrics['auc']:.4f}  Acc={metrics['accuracy']:.4f}  "
              f"F1={metrics['f1']:.4f}  耗时={elapsed:.1f}s")

    # ----------------------------------------------------------
    # 第三部分：XGBoost 滚动预测（对比基准）
    # ----------------------------------------------------------
    # 在完全相同的数据上运行 XGBoost，作为对比基准
    print(f"\n{'='*70}")
    print("  [第三部分] XGBoost 滚动预测 (对比基准)")
    print(f"{'='*70}")

    xgb_results = {}
    for code, info in stock_data.items():
        print(f"\n  --- {info['name']} ({code}) ---")
        pred_df, metrics, elapsed = run_rolling(
            info['df'], info['feature_cols'], model_type='xgboost')
        xgb_results[code] = {
            'pred_df': pred_df, 'metrics': metrics, 'elapsed': elapsed}
        print(f"    AUC={metrics['auc']:.4f}  Acc={metrics['accuracy']:.4f}  "
              f"F1={metrics['f1']:.4f}  耗时={elapsed:.1f}s")

    # ----------------------------------------------------------
    # 第四部分：Optuna 超参数搜索（以茅台为主）
    # ----------------------------------------------------------
    # 为什么只用茅台的数搜索参数？
    #   - 减少搜索时间（只搜索一次参数，然后在两只股票上验证）
    #   - 检验调参的泛化能力：茅台找到的参数在中芯国际上也有效吗？
    ref_code = '600519.SH'
    ref_info = stock_data[ref_code]

    print(f"\n{'='*70}")
    print(f"  [第四部分] Optuna 超参数优化 ({ref_info['name']})")
    print(f"{'='*70}")
    print(f"  搜索空间: num_leaves/learning_rate/max_depth/min_child_samples/")
    print(f"            subsample/colsample_bytree/reg_alpha/reg_lambda")
    print(f"  目标: 最大化 Purged K-Fold AUC (n_splits=5, gap=5)")
    print(f"  试验次数: 50")

    df_ref = ref_info['df']
    fc = ref_info['feature_cols']
    df_clean = df_ref.dropna(subset=fc + ['label'])
    X_all = df_clean[fc].values
    y_all = df_clean['label'].values

    t0 = time.time()
    study = optuna_lgb_search(X_all, y_all, n_trials=50)
    search_time = time.time() - t0

    best = study.best_params
    print(f"\n  搜索完成, 耗时 {search_time:.1f}s")
    print(f"  最优 AUC: {study.best_value:.4f}")
    print(f"  最优参数:")
    for k, v in best.items():
        if isinstance(v, float):
            print(f"    {k:<22s} = {v:.6f}")
        else:
            print(f"    {k:<22s} = {v}")

    # ----------------------------------------------------------
    # 第五部分：最优参数重新滚动预测
    # ----------------------------------------------------------
    # 用 Optuna 找到的最优参数重新运行滚动预测
    # 对比默认参数的结果，看调优是否有效
    print(f"\n{'='*70}")
    print("  [第五部分] 最优参数 LightGBM - 滚动预测")
    print(f"{'='*70}")

    best_params = {**best, 'n_estimators': 200, 'random_state': 42, 'verbose': -1}

    lgb_tuned_results = {}
    for code, info in stock_data.items():
        print(f"\n  --- {info['name']} ({code}) ---")
        pred_df, metrics, elapsed = run_rolling(
            info['df'], info['feature_cols'],
            model_type='lightgbm', params=best_params)
        lgb_tuned_results[code] = {
            'pred_df': pred_df, 'metrics': metrics, 'elapsed': elapsed}
        print(f"    AUC={metrics['auc']:.4f}  Acc={metrics['accuracy']:.4f}  "
              f"F1={metrics['f1']:.4f}  耗时={elapsed:.1f}s")

    # ----------------------------------------------------------
    # 第六部分：Purged K-Fold 交叉验证详情
    # ----------------------------------------------------------
    # 交叉验证可以更全面地评估模型稳定性：
    #   - 每折的性能是否一致？
    #   - 有没有某折表现特别差（说明模型不稳定）？
    #   - 平均性能是否有统计意义？
    print(f"\n{'='*70}")
    print(f"  [第六部分] Purged K-Fold 交叉验证 (n_splits=5, gap=5)")
    print(f"{'='*70}")

    for code, info in stock_data.items():
        df_c = info['df'].dropna(subset=info['feature_cols'] + ['label'])
        X = df_c[info['feature_cols']].values
        y = df_c['label'].values

        print(f"\n  --- {info['name']} ({code}) ---")
        fold_results = purged_kfold_cv(X, y, model_type='lightgbm',
                                        n_splits=5, gap=5, params=best_params)

        print(f"    {'Fold':>6s} {'AUC':>8s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s}")
        print(f"    {'-'*52}")
        for fr in fold_results:
            print(f"    {fr['fold']:>6d} {fr['auc']:>8.4f} {fr['accuracy']:>10.4f} "
                  f"{fr['precision']:>10.4f} {fr['recall']:>8.4f} {fr['f1']:>8.4f}")

        if fold_results:
            avg_auc = np.mean([f['auc'] for f in fold_results])
            avg_acc = np.mean([f['accuracy'] for f in fold_results])
            avg_f1 = np.mean([f['f1'] for f in fold_results])
            print(f"    {'均值':>6s} {avg_auc:>8.4f} {avg_acc:>10.4f} {'':>10s} {'':>8s} {avg_f1:>8.4f}")

    # ----------------------------------------------------------
    # 第七部分：XGBoost vs LightGBM 对比表
    # ----------------------------------------------------------
    # 最终对比：在一个表格中展示三种配置的预测性能
    #   - XGBoost（默认参数）
    #   - LightGBM（默认参数）
    #   - LightGBM（Optuna 调优）
    # 对比维度：AUC、Accuracy、F1、训练时间
    print(f"\n{'='*70}")
    print("  [第七部分] XGBoost vs LightGBM 性能对比")
    print(f"{'='*70}")

    print(f"\n  {'股票':<10s} {'模型':<18s} {'AUC':>8s} {'Accuracy':>10s} {'F1':>8s} {'训练时间(s)':>12s}")
    print(f"  {'-'*68}")

    for code, info in stock_data.items():
        name = info['name']
        xm = xgb_results[code]['metrics']
        xt = xgb_results[code]['elapsed']
        print(f"  {name:<10s} {'XGBoost':<18s} "
              f"{xm['auc']:>8.4f} {xm['accuracy']:>10.4f} {xm['f1']:>8.4f} {xt:>12.1f}")

        lm_d = lgb_default_results[code]['metrics']
        lt_d = lgb_default_results[code]['elapsed']
        print(f"  {'':10s} {'LightGBM(默认)':<18s} "
              f"{lm_d['auc']:>8.4f} {lm_d['accuracy']:>10.4f} {lm_d['f1']:>8.4f} {lt_d:>12.1f}")

        lm_t = lgb_tuned_results[code]['metrics']
        lt_t = lgb_tuned_results[code]['elapsed']
        print(f"  {'':10s} {'LightGBM(调优)':<18s} "
              f"{lm_t['auc']:>8.4f} {lm_t['accuracy']:>10.4f} {lm_t['f1']:>8.4f} {lt_t:>12.1f}")
        print()

    # 汇总平均
    avg_xgb_auc = np.mean([xgb_results[c]['metrics']['auc'] for c in STOCKS])
    avg_xgb_time = np.mean([xgb_results[c]['elapsed'] for c in STOCKS])
    avg_lgb_d_auc = np.mean([lgb_default_results[c]['metrics']['auc'] for c in STOCKS])
    avg_lgb_d_time = np.mean([lgb_default_results[c]['elapsed'] for c in STOCKS])
    avg_lgb_t_auc = np.mean([lgb_tuned_results[c]['metrics']['auc'] for c in STOCKS])
    avg_lgb_t_time = np.mean([lgb_tuned_results[c]['elapsed'] for c in STOCKS])

    print(f"  平均 AUC 对比:")
    print(f"    XGBoost:         {avg_xgb_auc:.4f}  (平均耗时 {avg_xgb_time:.1f}s)")
    print(f"    LightGBM(默认):  {avg_lgb_d_auc:.4f}  (平均耗时 {avg_lgb_d_time:.1f}s)")
    print(f"    LightGBM(调优):  {avg_lgb_t_auc:.4f}  (平均耗时 {avg_lgb_t_time:.1f}s)")

    speed_ratio = avg_xgb_time / avg_lgb_d_time if avg_lgb_d_time > 0 else 0
    print(f"\n  训练速度: LightGBM 比 XGBoost 快约 {speed_ratio:.1f} 倍")

    # ----------------------------------------------------------
    # 分析洞察
    # ----------------------------------------------------------
    print(f"\n{'='*70}")
    print("  分析洞察")
    print(f"{'='*70}")
    print(f"\n  Optuna调优后 LightGBM AUC: {avg_lgb_d_auc:.4f} -> {avg_lgb_t_auc:.4f}")
    print(f"  LightGBM 训练速度约为 XGBoost 的 {speed_ratio:.1f} 倍")

    # 核心结论：
    # 1. LightGBM 在测试的股票上通常能取得与 XGBoost 相近或更好的 AUC
    # 2. LightGBM 的训练速度显著快于 XGBoost（2~3 倍）
    # 3. Optuna 调优可以带来小幅但稳定的提升
    # 4. 两种模型在日内涨跌预测上的 AUC 通常在 0.50~0.60 之间
    #    这表明市场有一定的可预测性，但随机性很大


if __name__ == '__main__':
    main()
