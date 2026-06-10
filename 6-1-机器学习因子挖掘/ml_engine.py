# -*- coding: utf-8 -*-
"""
ML 训练/预测/评估引擎

本模块是机器学习选股的核心，提供完整的模型训练和评估流水线：

  核心功能:
    1. make_labels()            - 构建预测标签（次日涨跌的监督信号）
    2. rolling_train_predict()  - 滚动窗口训练与预测（避免未来信息泄露）
    3. purged_kfold_cv()        - 时序交叉验证（带清洗期，防止数据泄漏）
    4. train_xgboost/lightgbm/rf() - 三种主流树模型的训练接口
    5. evaluate_classification()    - 分类模型评估指标
    6. evaluate_factor()            - 概率因子的 IC/分层回测分析
    7. ensemble_predict()           - 多模型集成预测
    8. stacking_train()             - Stacking 集成（用各模型输出训练元模型）

  为什么要使用树模型预测股票涨跌？
    - XGBoost/LightGBM 能自动处理特征间的非线性关系
    - 对缺失值和不规范数据鲁棒性强
    - 输出概率值（0~1）本身就是天然的"alpha 因子"
    - 相比神经网络，树模型在表格数据上往往表现更好且可解释性更强

依赖关系：
  - feature_engine.py 提供特征计算
  - 1-4 号脚本调用本模块的各项功能
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 标签构建
# ============================================================

def make_labels(df, horizon=1, method='binary'):
    """
    构建预测标签（监督学习的目标变量）。

    在监督学习中，我们需要告诉模型"什么是正确的预测"。
    对于股票涨跌预测，自然的选择是未来 N 天的涨跌方向。

    参数:
        df: DataFrame, 必须包含 'close' 列（收盘价）
        horizon: 预测时间窗口，单位：交易日
                 =1 表示预测"明天"涨跌（短周期）
                 =5 表示预测"5天后"涨跌（中周期）
        method: 标签生成方法
                'binary'  - 二分类：涨=1, 跌=0
                'ternary' - 三分类：大涨>2%=2, 震荡=1, 大跌<-2%=0

    返回:
        pandas.Series, 标签值（与 df 索引对齐）

    关键设计考虑：
      - shift(-horizon) 让标签对齐到"今天"，即今天收盘时就能知道 horizon 天后的涨跌
      - 末尾 horizon 行的标签为 NaN（没有足够未来的数据），需要后续 dropna
      - 二分类是简化版，实际量化中三分类或回归更常见

    示例：
      make_labels(df, horizon=1, method='binary')
      -> 明天如果涨，今天这条记录的 label=1；明天如果跌，label=0
    """
    # 未来 horizon 天的收益率：shift(-horizon) 把未来数据拉到当前位置
    future_ret = df['close'].shift(-horizon) / df['close'] - 1

    if method == 'binary':
        # 二分类：收益率 > 0 为 1（涨），否则为 0（跌/平）
        label = (future_ret > 0).astype(int)
    elif method == 'ternary':
        # 三分类：涨>2% -> 2, 跌<-2% -> 0, 其余 -> 1（震荡）
        # 为什么设 2% 阈值？A 股日涨跌幅限制 10%，2% 是一个有意义的方向性波动
        label = pd.Series(1, index=df.index)
        label[future_ret > 0.02] = 2
        label[future_ret < -0.02] = 0
    else:
        raise ValueError(f"不支持的标签方法: {method}")

    return label


# ============================================================
# 模型训练接口
# ============================================================

def train_xgboost(X_train, y_train, params=None):
    """
    训练 XGBoost 分类器。

    XGBoost（eXtreme Gradient Boosting）是梯度提升决策树（GBDT）的优化实现。
    它的核心思想是：串行训练多棵决策树，每棵新树都去纠正前一棵树的错误。

    默认参数的设计思路：
      - n_estimators=200: 200棵树足够捕捉规律，又不会过拟合
      - max_depth=5: 限制树深度，防止学到噪声
      - learning_rate=0.05: 较小的学习率配合较多棵树，更好泛化
      - subsample=0.8: 每棵树只用 80% 样本，增加随机性防过拟合
      - colsample_bytree=0.8: 每棵树只用 80% 特征，类似随机森林
      - min_child_weight=10: 子节点最小权重和，防止分裂过于细碎
      - reg_alpha/reg_lambda: L1/L2 正则化，惩罚大权重

    参数:
        X_train: 训练特征矩阵 (n_samples, n_features)
        y_train: 训练标签数组 (n_samples,)
        params: 自定义参数字典，会覆盖默认参数

    返回:
        XGBClassifier 模型实例（已训练）
    """
    import xgboost as xgb

    default_params = {
        'n_estimators': 200,           # 树的数量
        'max_depth': 5,                # 每棵树的最大深度
        'learning_rate': 0.05,         # 学习率（步长）
        'subsample': 0.8,              # 样本采样比例
        'colsample_bytree': 0.8,       # 特征采样比例
        'min_child_weight': 10,        # 叶子节点最小权重
        'reg_alpha': 0.1,              # L1 正则化系数
        'reg_lambda': 1.0,             # L2 正则化系数
        'random_state': 42,            # 随机种子，确保结果可复现
        'use_label_encoder': False,    # 关闭旧的标签编码器（新版 XGBoost）
        'eval_metric': 'logloss',      # 评估指标：对数损失
        'verbosity': 0,                # 不打印训练过程信息
    }
    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(**default_params)
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train, params=None):
    """
    训练 LightGBM 分类器。

    LightGBM 与 XGBoost 同属 GBDT 家族，但有两个关键创新：
      1. 单边梯度采样（GOSS）：保留大梯度样本，对小梯度样本随机采样
      2. 互斥特征捆绑（EFB）：将互斥特征捆绑，降维加速

    这些优化让 LightGBM 在保持精度的同时，训练速度比 XGBoost 快 2-3 倍。

    num_leaves 是 LightGBM 特有的关键参数：
      - XGBoost 用 max_depth 控制树复杂度
      - LightGBM 用 num_leaves + max_depth 共同控制
      - 更大的 num_leaves 让树更宽，能捕捉更复杂的模式，但也更容易过拟合

    参数:
        X_train: 训练特征矩阵
        y_train: 训练标签数组
        params: 自定义参数

    返回:
        LGBMClassifier 模型实例（已训练）
    """
    import lightgbm as lgb

    default_params = {
        'n_estimators': 200,           # 树的数量
        'max_depth': 5,                # 最大深度
        'learning_rate': 0.05,         # 学习率
        'num_leaves': 31,              # 每棵树的叶子节点数（LightGBM 核心参数）
        'subsample': 0.8,              # 样本采样比例
        'colsample_bytree': 0.8,       # 特征采样比例
        'min_child_samples': 20,       # 叶子节点最少样本数
        'reg_alpha': 0.1,              # L1 正则化
        'reg_lambda': 1.0,             # L2 正则化
        'random_state': 42,            # 随机种子
        'verbose': -1,                 # 不打印训练信息
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)
    model.fit(X_train, y_train)
    return model


def train_rf(X_train, y_train, params=None):
    """
    训练随机森林（Random Forest）分类器。

    随机森林与 GBDT 的重要区别：
      - 随机森林：并行训练多棵树，每棵树独立预测，最终投票
      - GBDT（XGBoost/LightGBM）：串行训练，每棵树纠正前一棵的残差

    随机森林的优势：
      - 天然抗过拟合（多棵树投票的平均效应）
      - 训练速度快（可并行）
      - 参数少，调参简单

    但不适合做最终预测的原因：
      - 随机森林在大数据集上通常不如 GBDT 精确
      - 对"弱信号"的捕捉能力不如 GBDT（因为 GBDT 会迭代增强微弱信号）

    参数:
        X_train: 训练特征矩阵
        y_train: 训练标签数组
        params: 自定义参数

    返回:
        RandomForestClassifier 模型实例（已训练）
    """
    default_params = {
        'n_estimators': 200,        # 树的数量
        'max_depth': 8,             # 最大深度
        'min_samples_leaf': 20,     # 叶子节点最少样本数
        'max_features': 'sqrt',     # 每棵树随机选取 sqrt(n_features) 个特征
        'random_state': 42,         # 随机种子
        'n_jobs': -1,               # 使用所有 CPU 核并行训练
    }
    if params:
        default_params.update(params)

    model = RandomForestClassifier(**default_params)
    model.fit(X_train, y_train)
    return model


# 模型训练函数注册表：通过名称字符串选择模型
# 这样设计的好处是：调用者只需传 'xgboost'/'lightgbm'/'rf' 字符串，
# 而不需要 import 各模型库和选择正确的函数
TRAIN_FUNCS = {
    'xgboost': train_xgboost,
    'lightgbm': train_lightgbm,
    'rf': train_rf,
}


# ============================================================
# 滚动训练预测
# ============================================================

def rolling_train_predict(df, feature_cols, label_col='label',
                          model_type='xgboost', train_days=120,
                          retrain_interval=20, params=None,
                          verbose=True):
    """
    滚动窗口训练与预测（量化预测的核心技术）。

    为什么不能使用普通的 train/test split？
      股票数据是时间序列，存在时间依赖性。如果随机划分训练集和测试集，
      未来数据会泄露到训练集中，造成"未来信息泄露"，导致回测表现虚高。

    滚动窗口的工作方式：
      ```
      训练集         预测
      [0...119]天  -> 120天
      [0...119]天  -> 121天
      ...
      [0...119]天  -> 139天
      [20...139]天 -> 140天  （重新训练）
      [20...139]天 -> 141天
      ...
      ```

    关键参数：
      - train_days=120: 用最近 120 个交易日的数据训练（约半年）
      - retrain_interval=20: 每隔 20 个交易日重新训练一次（约一个月）

    为什么每隔 retrain_interval 天而不是每天重新训练？
      每天重新训练的代价太高（时间 + 计算资源），而市场规律变化相对缓慢。
      每隔一段时间重训，在"模型新鲜度"和"计算效率"之间取得平衡。

    参数:
        df: DataFrame，必须包含特征列和标签列（日期列可以是 index 或 trade_date 列）
        feature_cols: 特征列名列表
        label_col: 标签列名
        model_type: 'xgboost' / 'lightgbm' / 'rf'
        train_days: 训练窗口大小（交易日天数）
        retrain_interval: 重训间隔天数
        params: 模型参数字典
        verbose: 是否打印进度

    返回:
        DataFrame, 包含以下列：
          date:   预测日期
          y_true: 真实标签
          y_pred: 预测分类（0/1）
          y_prob: 预测概率（0~1），即"上涨概率因子"
    """
    train_func = TRAIN_FUNCS.get(model_type)
    if train_func is None:
        raise ValueError(f"不支持的模型类型: {model_type}")

    # 删除含缺失值的行（特征或标签缺失都无法预测）
    df_clean = df.dropna(subset=feature_cols + [label_col]).reset_index(drop=False)

    # 智能检测日期列名：
    # 先检查是否有 trade_date 列（常见命名），再看 index 名
    if 'trade_date' in df_clean.columns:
        date_col = 'trade_date'
    elif df_clean.index.name and 'date' in df_clean.index.name.lower():
        df_clean = df_clean.reset_index()
        date_col = df_clean.columns[0]
    else:
        date_col = df_clean.columns[0]

    results = []
    model = None
    last_train_idx = -retrain_interval  # 初始时确保第一次训练

    total = len(df_clean) - train_days
    for i in range(train_days, len(df_clean)):
        # 判断是否需要重新训练
        if model is None or (i - last_train_idx) >= retrain_interval:
            train_start = max(0, i - train_days)
            train_data = df_clean.iloc[train_start:i]

            X_train = train_data[feature_cols].values
            y_train = train_data[label_col].values

            # 如果训练集中只有一个类别（全涨或全跌），跳过
            # 因为树模型无法从单类别数据中学习到有意义的决策边界
            if len(np.unique(y_train)) < 2:
                continue

            model = train_func(X_train, y_train, params)
            last_train_idx = i

        # 用当前模型预测下一天
        row = df_clean.iloc[i]
        X_test = row[feature_cols].values.reshape(1, -1)

        y_pred = model.predict(X_test)[0]
        # predict_proba 返回每个类别的概率，[0,1] 分别对应跌/涨的概率
        # 取 [0,1] 中的第二个元素（索引1），即"上涨概率"
        y_prob = model.predict_proba(X_test)[0, 1]

        results.append({
            'date': row[date_col],
            'y_true': int(row[label_col]),
            'y_pred': int(y_pred),
            'y_prob': float(y_prob),
        })

        if verbose and len(results) % 100 == 0:
            print(f"  [{model_type}] 已预测 {len(results)}/{total} 天")

    return pd.DataFrame(results)


# ============================================================
# Purged K-Fold 时序交叉验证
# ============================================================

def purged_kfold_cv(X, y, model_type='xgboost', n_splits=5, gap=5, params=None):
    """
    Purged K-Fold 时序交叉验证。

    为什么需要这种特殊的交叉验证？
      标准 K-Fold 随机划分数据，但时间序列数据不能这样：
      如果用 1 月的数据训练、2 月的数据验证，这没问题。
      但用 2 月训练、1 月验证就有未来信息泄露了。

    Purged K-Fold 的划分方式：
      ```
      Fold 1: 训练 [0..20] | gap [21..25] | 验证 [26..30]
      Fold 2: 训练 [0..40] | gap [41..45] | 验证 [46..50]
      ...
      ```

      为什么需要 gap（清洗期）？
      股票收益率存在自相关性（尤其是高频数据），紧邻验证集的训练集
      可能会通过"信息泄漏"提高验证集表现。gap 天的时间间隔确保
      训练集和验证集之间没有时间重叠和自相关影响。

    参数:
        X: 特征矩阵 (n_samples, n_features)
        y: 标签数组 (n_samples,)
        model_type: 模型类型
        n_splits: 交叉验证折数
        gap: 训练集和验证集之间的间隔天数
        params: 模型参数

    返回:
        list[dict]: 每折的评估指标，包含 auc/accuracy/precision/recall/f1
    """
    train_func = TRAIN_FUNCS.get(model_type)
    n = len(X)
    fold_size = n // n_splits

    metrics_list = []

    for fold in range(n_splits):
        val_start = fold * fold_size
        val_end = min(val_start + fold_size, n)

        # 训练集在验证集之前，且留出 gap 天的间隔
        train_end = max(0, val_start - gap)
        if train_end < 30:
            # 训练数据太少时跳过（至少需要 30 个样本才有统计意义）
            continue

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[val_start:val_end], y[val_start:val_end]

        # 确保训练集和验证集都包含两个类别
        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            continue

        model = train_func(X_train, y_train, params)
        y_prob = model.predict_proba(X_val)[:, 1]  # 所有验证样本的上涨概率
        y_pred = model.predict(X_val)

        fold_metrics = evaluate_classification(y_val, y_pred, y_prob)
        fold_metrics['fold'] = fold
        metrics_list.append(fold_metrics)

    return metrics_list


# ============================================================
# 评估函数
# ============================================================

def evaluate_classification(y_true, y_pred, y_prob=None):
    """
    评估分类模型的性能。

    关键指标解读：
      - AUC（最重要！）：衡量模型区分上涨和下跌的能力。
        0.5 = 随机猜测，0.6 = 有一定预测能力，0.7+ = 较强预测能力。
        在量化中，AUC > 0.55 通常就可以作为有效因子。
      - Accuracy: 整体准确率，但股票涨跌往往不平衡（涨的天数多于跌的），
        因此 accuracy 可能虚高，需要结合其他指标看。
      - Precision: 预测"涨"中有多少是真的涨了。
        高 precision 意味着做多信号可靠，但可能错过很多机会。
      - Recall: 实际涨的股票中有多少被正确预测。
        高 recall 意味着能抓住大部分上涨机会，但可能有较多假信号。
      - F1: Precision 和 Recall 的调和平均，综合衡量。

    参数:
        y_true: 真实标签数组
        y_pred: 预测标签数组
        y_prob: 预测概率数组（可选，用于计算 AUC）

    返回:
        dict: 含 accuracy/precision/recall/f1/auc
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    # AUC 需要概率值和至少两个类别
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    else:
        metrics['auc'] = 0.0

    return metrics


def evaluate_factor(dates, probs, returns, n_groups=5):
    """
    评估概率因子的预测能力（因子分析的核心）。

    当我们用 ML 模型输出"上涨概率"后，这个概率本身就是一个 alpha 因子。
    本函数从以下几个维度评估因子的有效性：

    1. Rank IC（Spearman 相关系数）：
       - 衡量预测概率和未来收益的秩相关性
       - IC > 0 说明因子有正向预测能力
       - IC 的绝对值越大，预测能力越强

    2. ICIR（IC 的信息比率）：
       - IC 的均值 / IC 的标准差
       - 衡量因子预测能力的稳定性和一致性
       - ICIR > 0.5 通常被认为是有效的因子

    3. 分层回测（Quintile Analysis）：
       - 将股票按概率值分成 5 组
       - 计算每组未来收益的均值
       - 理想情况下：概率越高（第4组）的组收益越大，呈单调递增趋势

    参数:
        dates: 日期序列
        probs: 预测概率序列（0~1）
        returns: 实际收益率序列
        n_groups: 分层数量（默认 5 组）

    返回:
        dict: 含 ic/icir/分层收益等
    """
    df = pd.DataFrame({
        'date': dates,
        'prob': probs,
        'return': returns,
    }).dropna()

    if len(df) < 20:
        return {'ic': 0, 'icir': 0, 'quintile_returns': {}}

    # ---- IC 计算 ----
    # Spearman 秩相关：不要求线性关系，对异常值鲁棒
    ic = df['prob'].corr(df['return'], method='spearman')

    # ---- 按月计算 IC 序列 ----
    # 为什么按月？因为单日 IC 噪声太大，按月聚合可以消除噪声影响
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
    monthly_ic = df.groupby('month').apply(
        lambda g: g['prob'].corr(g['return'], method='spearman')
        if len(g) > 5 else np.nan
    ).dropna()

    # ICIR：IC 的 t 统计量，衡量 IC 的稳定性
    icir = monthly_ic.mean() / monthly_ic.std() if monthly_ic.std() > 0 else 0

    # ---- 分层回测 ----
    # pd.qcut 按分位数等分成 n_groups 组
    # labels=False 表示用 0,1,2,... 而不是区间标签
    # duplicates='drop' 处理重复分位数边界的情况
    df['group'] = pd.qcut(df['prob'], n_groups, labels=False, duplicates='drop')
    quintile_returns = df.groupby('group')['return'].mean().to_dict()

    return {
        'ic': round(ic, 4),
        'icir': round(icir, 4),
        'monthly_ic_mean': round(monthly_ic.mean(), 4) if len(monthly_ic) > 0 else 0,
        'monthly_ic_std': round(monthly_ic.std(), 4) if len(monthly_ic) > 0 else 0,
        'ic_positive_rate': round((monthly_ic > 0).mean(), 4) if len(monthly_ic) > 0 else 0,
        'quintile_returns': quintile_returns,
    }


# ============================================================
# 多模型集成
# ============================================================

def ensemble_predict(models, X, method='blending'):
    """
    多模型集成预测。

    为什么集成有效？
      不同模型（XGBoost/LightGBM/RF）有不同的"偏见"：
        - XGBoost: 对异常值敏感，但能捕捉复杂的非线性关系
        - LightGBM: 速度快，对大规模数据效果好
        - RF: 方差小，不易过拟合
      组合它们的预测可以抵消各自的偏见，得到更稳健的结果。

    两种集成方法：
      1. Blending（概率平均）：
         - 对多个模型的概率取均值
         - 相当于"每个模型投票，按置信度加权"
         - 比 voting 更平滑，因为用了概率而非 0/1 判别

      2. Voting（多数投票）：
         - 每个模型输出 0/1 预测
         - 取多数作为最终预测
         - 相当于"一人一票，少数服从多数"

    参数:
        models: 模型列表
        X: 特征矩阵
        method: 'blending'（概率平均）或 'voting'（多数投票）

    返回:
        (y_pred, y_prob): 预测标签和平均概率
    """
    probs = []
    for model in models:
        prob = model.predict_proba(X)[:, 1]
        probs.append(prob)

    probs = np.array(probs)

    if method == 'blending':
        # 概率平均：取所有模型概率的均值
        avg_prob = probs.mean(axis=0)
        y_pred = (avg_prob > 0.5).astype(int)
        return y_pred, avg_prob

    elif method == 'voting':
        # 多数投票：每个模型先判断涨跌，然后过半为涨
        preds = (probs > 0.5).astype(int)
        y_pred = (preds.sum(axis=0) > len(models) / 2).astype(int)
        avg_prob = probs.mean(axis=0)
        return y_pred, avg_prob

    else:
        raise ValueError(f"不支持的集成方法: {method}")


def stacking_train(models, X_train, y_train, X_test, meta_model=None):
    """
    Stacking 集成学习。

    Stacking 是比 Blending/Voting 更高级的集成方法：

    流程：
      1. 基模型预测：每个基模型对训练集做预测
      2. 构建元特征：将各模型的预测概率作为"元特征"
      3. 训练元模型：用元特征训练一个"元学习器"
      4. 最终预测：基模型先预测，元模型再基于基模型的预测做最终判断

    类比理解：
      - Blending 就像"三个医生投票"
      - Stacking 就像"三个医生各自诊断，然后一位主任医师综合意见做最终诊断"

    参数:
        models: 基模型列表（已训练好的）
        X_train: 训练特征（用于训练元模型）
        y_train: 训练标签（用于训练元模型）
        X_test: 测试特征
        meta_model: 元学习器，默认使用逻辑回归（LogisticRegression）

    返回:
        (y_pred, y_prob, meta_model): 预测结果和训练好的元模型
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    # 构建元特征：每列是一个模型的预测概率
    meta_features_train = np.column_stack([
        m.predict_proba(X_train)[:, 1] for m in models
    ])
    meta_features_test = np.column_stack([
        m.predict_proba(X_test)[:, 1] for m in models
    ])

    # 为什么用逻辑回归作为元模型？
    # 1. 逻辑回归简单、可解释性强
    # 2. 基模型的概率输出已经包含了丰富的非线性信息
    # 3. 逻辑回归可以学习到每个基模型的"权重"（即基模型的可靠性）
    if meta_model is None:
        meta_model = LogisticRegression(random_state=42)

    meta_model.fit(meta_features_train, y_train)
    y_pred = meta_model.predict(meta_features_test)
    y_prob = meta_model.predict_proba(meta_features_test)[:, 1]

    return y_pred, y_prob, meta_model
