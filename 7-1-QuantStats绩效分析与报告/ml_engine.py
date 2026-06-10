# -*- coding: utf-8 -*-
"""
ML训练/预测/评估引擎

功能:
  1. make_labels()               - 构建二分类标签 (次日涨=1, 跌=0)
  2. rolling_train_predict()     - 滚动窗口训练与预测
  3. purged_kfold_cv()           - Purged时序交叉验证
  4. train_xgboost/lightgbm/rf() - 三种模型训练接口
  5. evaluate_classification()   - 分类评估指标
  6. evaluate_factor()           - 概率因子的IC/分层分析
  7. ensemble_predict()          - 多模型集成预测
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
    构建预测标签

    参数:
        df: DataFrame, 需包含 close 列
        horizon: 预测时间窗口 (天数)
        method: 'binary' - 涨=1,跌=0; 'ternary' - 大涨=2,震荡=1,大跌=0

    返回:
        Series, 标签值
    """
    future_ret = df['close'].shift(-horizon) / df['close'] - 1

    if method == 'binary':
        label = (future_ret > 0).astype(int)
    elif method == 'ternary':
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
    """训练XGBoost分类器"""
    import xgboost as xgb

    default_params = {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 10,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'verbosity': 0,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(**default_params)
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train, params=None):
    """训练LightGBM分类器"""
    import lightgbm as lgb

    default_params = {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'verbose': -1,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)
    model.fit(X_train, y_train)
    return model


def train_rf(X_train, y_train, params=None):
    """训练RandomForest分类器"""
    default_params = {
        'n_estimators': 200,
        'max_depth': 8,
        'min_samples_leaf': 20,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1,
    }
    if params:
        default_params.update(params)

    model = RandomForestClassifier(**default_params)
    model.fit(X_train, y_train)
    return model


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
    滚动窗口训练与预测

    流程: 过去train_days天训练 -> 预测下一天 -> 滑动窗口
    每隔retrain_interval天重新训练模型

    参数:
        df: DataFrame, 含特征列和标签列
        feature_cols: 特征列名列表
        label_col: 标签列名
        model_type: 'xgboost' / 'lightgbm' / 'rf'
        train_days: 训练窗口大小
        retrain_interval: 重训间隔天数
        params: 模型参数
        verbose: 是否打印进度

    返回:
        DataFrame, 含 date/y_true/y_pred/y_prob 列
    """
    train_func = TRAIN_FUNCS.get(model_type)
    if train_func is None:
        raise ValueError(f"不支持的模型类型: {model_type}")

    df_clean = df.dropna(subset=feature_cols + [label_col]).reset_index(drop=False)
    if 'trade_date' in df_clean.columns:
        date_col = 'trade_date'
    elif df_clean.index.name and 'date' in df_clean.index.name.lower():
        df_clean = df_clean.reset_index()
        date_col = df_clean.columns[0]
    else:
        date_col = df_clean.columns[0]

    results = []
    model = None
    last_train_idx = -retrain_interval

    total = len(df_clean) - train_days
    for i in range(train_days, len(df_clean)):
        if model is None or (i - last_train_idx) >= retrain_interval:
            train_start = max(0, i - train_days)
            train_data = df_clean.iloc[train_start:i]

            X_train = train_data[feature_cols].values
            y_train = train_data[label_col].values

            if len(np.unique(y_train)) < 2:
                continue

            model = train_func(X_train, y_train, params)
            last_train_idx = i

        row = df_clean.iloc[i]
        X_test = row[feature_cols].values.reshape(1, -1)

        y_pred = model.predict(X_test)[0]
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
    Purged K-Fold 时序交叉验证

    在训练集和验证集之间留出gap天的间隔，防止未来信息泄露

    参数:
        X: 特征矩阵
        y: 标签数组
        model_type: 模型类型
        n_splits: 折数
        gap: 训练/验证间隔天数
        params: 模型参数

    返回:
        dict, 含每折的AUC/Accuracy等指标
    """
    train_func = TRAIN_FUNCS.get(model_type)
    n = len(X)
    fold_size = n // n_splits

    metrics_list = []

    for fold in range(n_splits):
        val_start = fold * fold_size
        val_end = min(val_start + fold_size, n)

        train_end = max(0, val_start - gap)
        if train_end < 30:
            continue

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[val_start:val_end], y[val_start:val_end]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            continue

        model = train_func(X_train, y_train, params)
        y_prob = model.predict_proba(X_val)[:, 1]
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
    分类模型评估

    返回:
        dict, 含 accuracy/precision/recall/f1/auc
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    else:
        metrics['auc'] = 0.0

    return metrics


def evaluate_factor(dates, probs, returns, n_groups=5):
    """
    评估概率因子的预测能力

    参数:
        dates: 日期序列
        probs: 预测概率序列
        returns: 实际收益率序列
        n_groups: 分层数量

    返回:
        dict, 含 ic/icir/分层收益等
    """
    df = pd.DataFrame({
        'date': dates,
        'prob': probs,
        'return': returns,
    }).dropna()

    if len(df) < 20:
        return {'ic': 0, 'icir': 0, 'quintile_returns': {}}

    # IC: 概率与实际收益的rank相关
    ic = df['prob'].corr(df['return'], method='spearman')

    # 按月计算IC序列
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
    monthly_ic = df.groupby('month').apply(
        lambda g: g['prob'].corr(g['return'], method='spearman')
        if len(g) > 5 else np.nan
    ).dropna()

    icir = monthly_ic.mean() / monthly_ic.std() if monthly_ic.std() > 0 else 0

    # 分层回测
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
    多模型集成预测

    参数:
        models: 模型列表
        X: 特征矩阵
        method: 'blending' (概率平均) 或 'voting' (多数投票)

    返回:
        y_pred, y_prob
    """
    probs = []
    for model in models:
        prob = model.predict_proba(X)[:, 1]
        probs.append(prob)

    probs = np.array(probs)

    if method == 'blending':
        avg_prob = probs.mean(axis=0)
        y_pred = (avg_prob > 0.5).astype(int)
        return y_pred, avg_prob

    elif method == 'voting':
        preds = (probs > 0.5).astype(int)
        y_pred = (preds.sum(axis=0) > len(models) / 2).astype(int)
        avg_prob = probs.mean(axis=0)
        return y_pred, avg_prob

    else:
        raise ValueError(f"不支持的集成方法: {method}")


def stacking_train(models, X_train, y_train, X_test, meta_model=None):
    """
    Stacking集成: 各模型预测作为元特征，用逻辑回归做最终预测

    参数:
        models: 基模型列表(已训练)
        X_train, y_train: 用于训练元模型的数据
        X_test: 测试特征
        meta_model: 元学习器(默认LogisticRegression)

    返回:
        y_pred, y_prob
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    # 构建元特征(各模型的概率预测)
    meta_features_train = np.column_stack([
        m.predict_proba(X_train)[:, 1] for m in models
    ])
    meta_features_test = np.column_stack([
        m.predict_proba(X_test)[:, 1] for m in models
    ])

    if meta_model is None:
        meta_model = LogisticRegression(random_state=42)

    meta_model.fit(meta_features_train, y_train)
    y_pred = meta_model.predict(meta_features_test)
    y_prob = meta_model.predict_proba(meta_features_test)[:, 1]

    return y_pred, y_prob, meta_model
