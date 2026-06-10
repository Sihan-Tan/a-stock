# -*- coding: utf-8 -*-
"""
第09讲：缠论精华量化
脚本8：ML增强缠论策略 - 用 LightGBM 过滤三买信号

核心思路:
  缠论三买信号并非100%可靠(参见平安银行案例: 三买后继续下跌)。
  用机器学习模型预测"这个三买会不会成功", 只在模型看好时入场。

与海龟ML策略(CASE 4)的对比:
  海龟: 预测"突破"是否成功, 特征基于价格/量价通用指标
  缠论: 预测"三买"是否成功, 特征包含缠论结构指标

关键设计:

  1. 多股票训练 (6只跨行业股票):
     单只股票的三买样本太少 (可能只有个位数), 不够训练模型。
     通过多只股票合并, 扩大样本量。

  2. 时间分割: 2022-2024年训练, 2025年测试
     Walk-Forward 思想: 用历史数据训练, 在未来数据上验证。

  3. 特征设计 - 分为两类:
     结构特征 (基于缠论):
       - zs_height_ratio: 中枢高度/价格 (中枢越窄, 突破越容易延续)
       - zs_width: 中枢包含的K线数 (盘整越久, 突破能量越大)
       - bi_slope: 三买前最后一笔的斜率 (上升力度)
       - bi_in_zs: 中枢内的笔数 (标准中枢3笔, 更多笔=更强共识)

     技术指标特征:
       - atr_ratio: ATR/Close (波动率环境)
       - adx: 趋势强度
       - vol_ratio: 成交量/20日均量 (放量突破更可靠)
       - rsi: RSI (避免超买追高)
       - macd_hist: MACD柱状值 (动能方向)
       - momentum_10d: 10日动量 (中期趋势)

  4. Label 定义: 三买后20日内最大涨幅 > 5% = 成功 (1), 否则失败 (0)
     注意用的是"最大涨幅"而非"期末涨幅", 因为交易者可以在期间任意时刻止盈。
     这给了模型更多的"正样本", 不容易过拟合。

  5. 模型: LightGBM (浅树 max_depth=3 + 正则化防过拟合)
     - n_estimators=60: 树的数量不多, 防止过拟合
     - max_depth=3: 每棵树深度小, 特征交互有限
     - reg_alpha/reg_lambda: L1/L2正则化
     - is_unbalance=True: 正负样本不均衡时自动调整权重

训练流程:
  多股票提取特征 → 合并 → 时间分割 → 训练LightGBM → 目标股票预测 → 回测
"""

import sys
import numpy as np
import pandas as pd
import talib
import backtrader as bt
from data_loader import (
    load_stock_data, ChanPandasData,
    run_and_report, calc_buy_and_hold,
)
from chan_analyzer import ChanAnalyzer
from db_config import execute_query

# ============================================================
# 参数配置
# ============================================================

# 目标股票: 中芯国际 (测试集, 不参与训练)
TARGET_STOCK = '688981.SH'
TARGET_NAME = '中芯国际'
START_DATE = '2024-01-01'
END_DATE = '2026-06-02'
SPLIT_DATE = pd.Timestamp('2026-01-01')  # 训练/测试分割点
ML_THRESHOLD = 0.6  # 模型预测概率 >= 0.6 时才入场

# 特征缓存: 避免每次运行都重新提取缠论特征 (提取很慢)
# 缓存文件保存在当前脚本同目录下, CSV 格式便于检查和修改
from pathlib import Path
_SCRIPT_DIR = Path(__file__).parent
FEATURES_CACHE_PATH = _SCRIPT_DIR / '__cahce__/chan_features_cache.csv'
FORCE_REFRESH = True  # True=强制重新提取特征, False=优先使用缓存

# 模型持久化: 训练完成后保存模型, 下次运行可直接加载
# 支持训练中断恢复: 如果缓存文件存在, 跳过训练直接加载
MODEL_CACHE_PATH = _SCRIPT_DIR / '__cahce__/chan_ml_model.pkl'
FORCE_RETRAIN = False  # True=强制重新训练, False=优先加载已保存的模型

# 训练股票池: 从 stock_list 表随机采样 N 只股票
# 设定随机种子保证每次采样结果一致, 方便复现
import random
TRAIN_SAMPLE_SIZE = 500
RANDOM_SEED = 23

def get_train_stocks_from_db(sample_size=TRAIN_SAMPLE_SIZE, seed=RANDOM_SEED):
    """
    从 stock_list 表中获取所有股票, 随机抽取 sample_size 只作为训练集,
    返回与 TRAIN_STOCKS 相同结构的列表: [(code, name), ...]
    """
    rows = execute_query(
        "SELECT stock_code, stock_name FROM stock_list ORDER BY stock_code"
    )
    all_stocks = [(r['stock_code'], r['stock_name']) for r in rows]

    if len(all_stocks) <= sample_size:
        return all_stocks

    rng = random.Random(seed)
    return rng.sample(all_stocks, sample_size)


TRAIN_STOCKS = get_train_stocks_from_db()


# ============================================================
# Step 1: 特征工程 - 在每个三买信号点提取特征
# ============================================================

def extract_chan_features(stock_code, start_date, end_date):
    """
    对单只股票执行缠论分析, 在每个三买信号点提取特征

    流程:
      1. 加载数据 → ChanAnalyzer 分析
      2. 遍历所有三买信号
      3. 对每个信号提取: 结构特征 + 技术指标特征
      4. 标记 label: 20日内最大涨幅 > 5%?

    参数:
        stock_code: 股票代码
        start_date: 起始日期
        end_date:   结束日期

    返回:
        features_df: 特征DataFrame (行为三买事件, 列为特征)
        labels: numpy 数组 (1=成功, 0=失败)
        signal_df: 缠论信号DataFrame (用于后续回测)
    """
    df = load_stock_data(stock_code, start_date, end_date)
    analyzer = ChanAnalyzer(df)
    analyzer.analyze()
    signal_df = analyzer.get_signal_df()

    # 预计算技术指标
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    close = df['close'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)

    # ATR: 波动率
    atr = talib.ATR(high, low, close, timeperiod=14)
    # ADX: 趋势强度 (25以上表示强趋势)
    adx = talib.ADX(high, low, close, timeperiod=14)
    # RSI: 相对强弱 (避免超买追高)
    rsi = talib.RSI(close, timeperiod=14)
    # 成交量均线
    vol_ma = talib.SMA(volume, timeperiod=20)
    # MACD: 动能方向
    _, _, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

    features_list = []
    labels_list = []
    dates_list = []

    for sig in analyzer.signals:
        if sig['type'] != 'third_buy':
            continue  # 只处理三买信号

        sig_date = sig['date']
        idx = df.index.get_loc(sig_date)
        if idx < 30:
            continue  # 数据太靠前, 指标计算可能不准确

        price = close[idx]
        if np.isnan(atr[idx]) or atr[idx] <= 0:
            continue  # 无效ATR
        if np.isnan(adx[idx]) or np.isnan(rsi[idx]):
            continue  # 无效ADX/RSI

        # ===== 结构特征 (缠论特有) =====

        # 找到触发这个三买的中枢 (取最后一个)
        zs_height_ratio = 0.0
        zs_width = 0
        bi_in_zs = 0
        if analyzer.zhongshu_list:
            for zs in reversed(analyzer.zhongshu_list):
                if zs['end_date'] <= sig_date:
                    zs_height_ratio = (zs['ZG'] - zs['ZD']) / price  # 中枢高度/价格
                    zs_start_idx = df.index.get_loc(zs['start_date'])
                    zs_end_idx = df.index.get_loc(zs['end_date'])
                    zs_width = zs_end_idx - zs_start_idx  # 中枢包含K线数
                    bi_in_zs = zs.get('bi_count', 3)  # 中枢内的笔数
                    break

        # 三买前最后一笔的斜率
        bi_slope = 0.0
        if len(analyzer.bi_list) >= 2:
            for bi in reversed(analyzer.bi_list):
                if bi['end_date'] <= sig_date and bi['direction'] == 'up':
                    bi_len = (bi['end_date'] - bi['start_date']).days
                    if bi_len > 0:
                        bi_slope = (bi['end_price'] - bi['start_price']) / bi['start_price'] / bi_len * 100
                    break

        # ===== 技术指标特征 =====
        momentum_10d = close[idx] / close[max(0, idx-10)] - 1 if idx >= 10 else 0
        vol_ratio_val = volume[idx] / vol_ma[idx] if not np.isnan(vol_ma[idx]) and vol_ma[idx] > 0 else 1.0
        macd_val = macd_hist[idx] if not np.isnan(macd_hist[idx]) else 0.0

        features_list.append({
            'zs_height_ratio': zs_height_ratio,  # 中枢相对高度 (窄→好突破)
            'zs_width': zs_width,                  # 中枢宽度 (久→蓄力足)
            'bi_in_zs': bi_in_zs,                  # 中枢内笔数 (多→共识强)
            'bi_slope': bi_slope,                  # 上升笔斜率 (陡→强势)
            'atr_ratio': atr[idx] / price,         # 波动率/价格 (高→风险大)
            'adx': adx[idx],                       # 趋势强度 (大→趋势可靠)
            'vol_ratio': vol_ratio_val,            # 量比 (大→真突破)
            'rsi': rsi[idx],                       # RSI (避免超买区追高)
            'macd_hist': macd_val / price * 100,   # MACD柱 (正→多头动能)
            'momentum_10d': momentum_10d,          # 10日动量 (加速→强势)
        })

        # Label: 三买后20日内最高涨幅 > 5% = 成功
        if idx + 20 < len(df):
            future_max = np.max(close[idx+1:idx+21])
            labels_list.append(1 if (future_max / price - 1) > 0.05 else 0)
        else:
            labels_list.append(np.nan)

        dates_list.append(sig_date)

    if not features_list:
        return pd.DataFrame(), np.array([]), signal_df

    features_df = pd.DataFrame(features_list, index=dates_list)
    labels = np.array(labels_list)

    # 过滤掉 label NaN 的样本 (数据末尾不足20日)
    valid = ~np.isnan(labels)
    features_df = features_df[valid]
    labels = labels[valid].astype(int)

    # 记录股票代码, 方便缓存时区分来源
    features_df['stock_code'] = stock_code

    return features_df, labels, signal_df


def collect_multi_stock_features(stocks, start_date, end_date):
    """
    从多只股票收集三买事件特征

    为什么需要多只股票:
      单只股票的三买事件通常只有个位数, 不足以训练模型。
      通过多只股票合并, 可以积累几十到上百个训练样本。

    合并方式: 纵向拼接 (concat), 用日期索引区分不同股票的事件。
    """
    all_features = []
    all_labels = []
    stock_info = []

    for code, name in stocks:
        try:
            feat, lab, _ = extract_chan_features(code, start_date, end_date)
            if len(feat) > 0:
                all_features.append(feat)
                all_labels.append(lab)
                rate = lab.mean() * 100 if len(lab) > 0 else 0
                stock_info.append(f"    {name}({code}): {len(feat)}个三买, 成功率 {rate:.0f}%")
            else:
                stock_info.append(f"    {name}({code}): 无三买信号")
        except Exception as e:
            stock_info.append(f"    {name}({code}): 跳过({e})")

    for info in stock_info:
        print(info)

    if not all_features:
        return pd.DataFrame(), np.array([])

    combined_features = pd.concat(all_features).sort_index()
    combined_labels = np.concatenate(all_labels)
    return combined_features, combined_labels


# ============================================================
# Step 1.5: 特征持久化 - 保存/加载/缓存
# ============================================================

def save_features_to_csv(features_df, labels, filepath):
    """
    将特征和标签保存为 CSV 文件

    保存格式:
      - stock_code: 股票代码
      - signal_date: 三买信号日期 (从 features_df 的 index 取出)
      - 10 个特征列 (zs_height_ratio ... momentum_10d)
      - label: 标签 (1=成功, 0=失败)

    参数:
        features_df: 特征 DataFrame (index=信号日期, 包含 stock_code 列)
        labels: numpy 数组
        filepath: 保存路径 (str 或 Path)
    """
    df = features_df.copy()
    df['signal_date'] = df.index  # 把日期 index 存为一列
    df['label'] = labels
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"  特征已保存: {filepath} ({len(df)} 条样本)")


def load_features_from_csv(filepath):
    """
    从 CSV 文件加载特征和标签

    返回:
        features_df: 特征 DataFrame (index=signal_date, 不含 stock_code/label)
        labels: numpy 数组
        all_stock_codes: list[str] (样本对应的股票代码, 与 features_df 同序)

    如果文件不存在, 返回 (None, None, None)
    """
    if not Path(filepath).exists():
        return None, None, None

    df = pd.read_csv(filepath, encoding='utf-8-sig')
    labels = df['label'].values.astype(int)
    stock_codes = df['stock_code'].tolist()

    # 恢复 index 为 signal_date, 去掉辅助列
    feature_cols = [c for c in df.columns if c not in ('stock_code', 'signal_date', 'label')]
    features_df = df.set_index('signal_date')[feature_cols]
    features_df.index = pd.to_datetime(features_df.index)

    return features_df, labels, stock_codes


def get_or_collect_features(stocks, start_date, end_date,
                            cache_path=FEATURES_CACHE_PATH,
                            force_refresh=FORCE_REFRESH):
    """
    获取训练特征 (优先读缓存, 缓存不存在时提取并保存)

    使用逻辑:
      1. 如果缓存文件存在 且 force_refresh=False → 从 CSV 加载
      2. 否则 → 从数据库/行情提取特征, 保存到 CSV

    参数:
        stocks: [(code, name), ...] 训练股票列表
        start_date, end_date: 数据区间
        cache_path: 缓存文件路径
        force_refresh: True=强制重新提取

    返回:
        features_df, labels (与 collect_multi_stock_features 返回值一致)
    """
    cache_path = Path(cache_path)

    # --- 尝试从缓存加载 ---
    if cache_path.exists() and not force_refresh:
        print(f"  从缓存加载特征: {cache_path}")
        print(f"  文件大小: {cache_path.stat().st_size / 1024:.1f} KB")
        features_df, labels, stock_codes = load_features_from_csv(cache_path)
        if features_df is not None and len(features_df) > 0:
            # 打印缓存中的股票覆盖情况
            cached_codes = set(stock_codes)
            requested_codes = {code for code, _ in stocks}
            missing = requested_codes - cached_codes
            print(f"  缓存覆盖: {len(cached_codes)} 只股票, {len(features_df)} 条样本")
            if missing:
                print(f"  ⚠ 缓存中缺少 {len(missing)} 只股票: {missing}")
                print(f"    如需包含这些股票, 设置 FORCE_REFRESH=True 重新提取")
            return features_df, labels
        else:
            print(f"  缓存文件为空, 将重新提取")

    # --- 重新提取特征 ---
    print(f"  提取特征 (共 {len(stocks)} 只股票)...")
    features_df, labels = collect_multi_stock_features(stocks, start_date, end_date)

    if len(features_df) > 0:
        save_features_to_csv(features_df, labels, cache_path)

    return features_df, labels

def prepare_train_data(features_df, labels, split_date):
    """
    将特征按时间分割为训练集和验证集

    Walk-Forward 分割:
      训练集: 信号日期 < split_date  (历史数据)
      验证集: 信号日期 >= split_date (未来数据)

    返回:
        X_train, y_train: 训练特征和标签
        X_test, y_test:   验证特征和标签
        feature_cols:     参与训练的特征列名 (已排除 stock_code)
    """
    train_mask = features_df.index < split_date
    test_mask = features_df.index >= split_date

    # 去掉非特征列 (stock_code 仅用于缓存标识, 不参与训练)
    feature_cols = [c for c in features_df.columns if c != 'stock_code']
    X_train = features_df.loc[train_mask, feature_cols]
    y_train = labels[np.where(train_mask)[0]]
    X_test = features_df.loc[test_mask, feature_cols]
    y_test = labels[np.where(test_mask)[0]]

    if len(X_train) < 3 or len(X_test) < 2:
        print(f"  样本不足: 训练{len(X_train)}, 验证{len(X_test)}")
        return None, None, None, None, feature_cols

    print(f"\n  训练集: {len(X_train)}个三买 | 成功率: {y_train.mean()*100:.0f}%")
    print(f"  验证集: {len(X_test)}个三买 | 成功率: {y_test.mean()*100:.0f}%")

    return X_train, y_train, X_test, y_test, feature_cols


def _detect_ml_engine():
    """检测可用的ML引擎: LightGBM > XGBoost > sklearn"""
    from importlib.util import find_spec
    if find_spec('lightgbm'):
        return 'lightgbm'
    if find_spec('xgboost'):
        return 'xgboost'
    return 'sklearn'


def _create_model(ml_engine, y_train):
    """
    创建模型实例 (不训练)

    防过拟合设计:
      - max_depth=3: 每棵树深度有限
      - reg_alpha/reg_lambda: L1/L2正则化
      - min_child_samples=2 / min_samples_leaf=2: 叶子最少样本数
      - n_estimators=60: 有限棵数
    """
    if ml_engine == 'lightgbm':
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=60, max_depth=3, learning_rate=0.1,
            min_child_samples=2, reg_alpha=0.1, reg_lambda=1.0,
            is_unbalance=True, verbose=-1, random_state=42,
            device='gpu', gpu_platform_id=0, gpu_device_id=0,
        )
    elif ml_engine == 'xgboost':
        import xgboost as xgb
        pos_w = max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1)
        return xgb.XGBClassifier(
            n_estimators=60, max_depth=3, learning_rate=0.1,
            min_child_weight=2, reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=pos_w, eval_metric='logloss',
            verbosity=0, random_state=42,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=60, max_depth=3, learning_rate=0.1,
            min_samples_leaf=2, random_state=42,
        )


def save_model(model, filepath):
    """
    保存模型到磁盘 (joblib 格式)

    joblib 专门优化了 numpy 数组和 scikit-learn 模型的序列化,
    比 pickle 更快、文件更小。
    """
    import joblib
    filepath = Path(filepath)
    joblib.dump(model, filepath)
    size_kb = filepath.stat().st_size / 1024
    print(f"  模型已保存: {filepath} ({size_kb:.1f} KB)")


def load_model(filepath):
    """
    从磁盘加载模型

    返回:
        model 对象, 文件不存在时返回 None
    """
    import joblib
    filepath = Path(filepath)
    if not filepath.exists():
        return None
    print(f"  模型已加载: {filepath} ({filepath.stat().st_size / 1024:.1f} KB)")
    return joblib.load(filepath)


def fit_model(X_train, y_train, ml_engine=None):
    """
    训练模型 (纯训练, 不做评估)

    参数:
        X_train, y_train: 训练数据
        ml_engine: 指定引擎, None时自动检测

    返回:
        model: 训练好的模型
        ml_engine: 实际使用的引擎名称
    """
    if ml_engine is None:
        ml_engine = _detect_ml_engine()

    print(f"  引擎: {ml_engine}")
    model = _create_model(ml_engine, y_train)

    # 检测 GPU 是否实际启用 (LightGBM GPU 不可用时静默回退 CPU)
    if ml_engine == 'lightgbm':
        params = model.get_params()
        gpu_enabled = params.get('device') in ('gpu', 'cuda')
        print(f"  GPU加速: {'✓ 已启用' if gpu_enabled else '✗ 未启用 (检查: pip install lightgbm GPU版)'}"
              f" (device={params.get('device', '未设置')})")

    model.fit(X_train, y_train)
    print(f"  训练完成")

    return model, ml_engine


def evaluate_model(model, X_test, y_test, feature_cols):
    """
    在验证集上评估模型

    与训练完全分离: 只做预测和指标计算, 不修改模型。

    评估指标:
      - accuracy: 总体准确率
      - precision: 预测"成功"的准确率
      - recall: 找出了多少"真成功"
      - F1: 综合指标

    返回:
        metrics: dict
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }

    print(f"\n  验证集评估:")
    print(f"    准确率:  {metrics['accuracy']*100:.1f}%")
    print(f"    精确率:  {metrics['precision']*100:.1f}%")
    print(f"    召回率:  {metrics['recall']*100:.1f}%")
    print(f"    F1分数:  {metrics['f1']*100:.1f}%")

    # 打印特征重要性 (帮助理解模型学到了什么)
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        importances = importances.sort_values(ascending=False)
        imp_max = importances.max()
        imp_norm = importances / imp_max if imp_max > 0 else importances
        print(f"\n  特征重要性:")
        for feat, imp_n in imp_norm.items():
            bar = '#' * int(imp_n * 25)
            print(f"    {feat:<22} {imp_n:.2f} {bar}")

    return metrics


def get_or_train_model(X_train, y_train,
                       cache_path=MODEL_CACHE_PATH,
                       force_retrain=FORCE_RETRAIN):
    """
    获取模型 (优先从缓存加载, 支持训练中断恢复)

    使用逻辑:
      1. 如果缓存文件存在 且 force_retrain=False → 加载已有模型
         (训练中断后重启, 直接从这里恢复)
      2. 否则 → 训练新模型并保存到磁盘

    参数:
        X_train, y_train: 训练数据
        cache_path: 模型缓存路径
        force_retrain: True=强制重新训练

    返回:
        model, ml_engine
    """
    cache_path = Path(cache_path)

    # --- 尝试从缓存加载 (中断恢复入口) ---
    if cache_path.exists() and not force_retrain:
        model = load_model(cache_path)
        if model is not None:
            ml_engine = _detect_ml_engine()
            return model, ml_engine

    # --- 训练新模型 ---
    model, ml_engine = fit_model(X_train, y_train)

    # --- 保存到缓存 (下次运行 / 中断恢复) ---
    save_model(model, cache_path)

    return model, ml_engine


# ============================================================
# Step 3: 策略定义
# ============================================================

class ChanMLStrategy(bt.Strategy):
    """
    ML增强缠论三买策略: 只在ML模型看好时入场

    与基础策略的唯一区别: 在入场前增加ML过滤
      ML预测概率 >= ML_THRESHOLD (0.5) → 入场
      ML预测概率 < ML_THRESHOLD (0.5) → 跳过

    ML过滤的效果:
      - 减少交易次数 (只做有把握的交易)
      - 提升胜率 (过滤掉模型判断为"失败"的信号)
      - 期望值提高 (坏信号被过滤, 好信号留下)
    """
    params = (
        ('take_profit_pct', 0.15),
        ('ml_threshold', 0.5),
        ('predictions', {}),  # {日期: ML预测概率}
    )

    def __init__(self):
        self.entry_price = None
        self.stop_price = None
        self.order = None
        self.ml_passed = 0      # ML通过的信号数
        self.ml_filtered = 0    # ML过滤掉的信号数

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.entry_price = order.executed.price
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.data.chan_signal[0] == 3:
                current_date = self.data.datetime.date(0)
                prob = self.p.predictions.get(current_date, 0.0)  # 获取ML预测概率
                if prob >= self.p.ml_threshold:
                    # ML看好 → 入场
                    self.order = self.buy()
                    zg_val = self.data.chan_zg[0]
                    self.stop_price = zg_val if zg_val > 0 else self.data.close[0] * 0.93
                    self.ml_passed += 1
                else:
                    # ML不看好 → 跳过
                    self.ml_filtered += 1
        else:
            # 持仓管理与基础策略相同
            current_price = self.data.close[0]
            if self.stop_price and current_price < self.stop_price:
                self.order = self.close()
                return
            if self.entry_price and (current_price / self.entry_price - 1) >= self.p.take_profit_pct:
                self.order = self.close()
                return
            if self.data.chan_signal[0] == -3:
                self.order = self.close()

    def stop(self):
        """策略结束时的回调: 打印ML过滤统计"""
        total = self.ml_passed + self.ml_filtered
        if total > 0:
            print(f"  ML过滤: 三买信号{total}个 | "
                  f"通过{self.ml_passed}({self.ml_passed/total*100:.0f}%) | "
                  f"过滤{self.ml_filtered}({self.ml_filtered/total*100:.0f}%)")


class ChanBasicStrategy(bt.Strategy):
    """基础缠论三买策略(对照组), 不做ML过滤"""
    params = (('take_profit_pct', 0.15),)

    def __init__(self):
        self.entry_price = None
        self.stop_price = None
        self.order = None

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.entry_price = order.executed.price
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def next(self):
        if self.order:
            return
        if not self.position:
            if self.data.chan_signal[0] == 3:
                self.order = self.buy()
                self.stop_price = self.data.chan_zg[0] if self.data.chan_zg[0] > 0 else self.data.close[0] * 0.93
        else:
            c = self.data.close[0]
            if self.stop_price and c < self.stop_price:
                self.order = self.close()
                return
            if self.entry_price and (c / self.entry_price - 1) >= self.p.take_profit_pct:
                self.order = self.close()
                return
            if self.data.chan_signal[0] == -3:
                self.order = self.close()


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 70)
    print("第09讲 | 脚本8: ML增强缠论策略")
    print("=" * 70)
    print("\n设计思路:")
    print("  1. 提取每个三买信号点的缠论结构特征 + 技术指标特征")
    print("  2. 时间分割: 历史数据训练, 未来数据验证")
    print("  3. 训练LightGBM模型 (可中断恢复, 模型持久化)")
    print("  4. 验证集评估模型预测能力")
    print("  5. 只在模型预测成功率 >= 50% 时入场")
    print("  6. 对比: 基础三买策略 vs ML增强三买策略")

    # ---- Step 1: 多股票特征收集 (优先读缓存) ----
    print(f"\n{'=' * 70}")
    print("Step 1: 多股票三买特征收集")
    print(f"{'=' * 70}")

    features_df, labels = get_or_collect_features(
        TRAIN_STOCKS, START_DATE, END_DATE,
        cache_path=FEATURES_CACHE_PATH,
        force_refresh=FORCE_REFRESH,
    )

    if len(features_df) < 5:
        print(f"\n三买样本不足({len(features_df)}个), 无法训练")
        return

    print(f"\n  合计: {len(features_df)}个三买事件")
    print(f"  成功(20日涨>5%): {labels.sum()} ({labels.mean()*100:.0f}%)")
    print(f"  失败: {len(labels)-labels.sum()} ({(1-labels.mean())*100:.0f}%)")

    # ---- Step 2: 准备训练/验证数据 (时间分割) ----
    print(f"\n{'=' * 70}")
    print(f"Step 2: 准备训练/验证数据 (分割: {SPLIT_DATE.strftime('%Y-%m-%d')})")
    print(f"{'=' * 70}")

    X_train, y_train, X_test, y_test, feature_cols = prepare_train_data(
        features_df, labels, SPLIT_DATE
    )
    if X_train is None:
        print("数据准备失败")
        return

    # ---- Step 3: 训练模型 (可中断恢复) ----
    print(f"\n{'=' * 70}")
    print(f"Step 3: 训练模型")
    print(f"{'=' * 70}")

    model, ml_engine = get_or_train_model(
        X_train, y_train,
        cache_path=MODEL_CACHE_PATH,
        force_retrain=FORCE_RETRAIN,
    )

    # ---- Step 4: 验证模型 ----
    print(f"\n{'=' * 70}")
    print(f"Step 4: 验证模型 (测试集)")
    print(f"{'=' * 70}")

    model_metrics = evaluate_model(model, X_test, y_test, feature_cols)

    # ---- Step 5: 为目标股票生成预测 ----
    print(f"\n{'=' * 70}")
    print(f"Step 5: 为 {TARGET_NAME}({TARGET_STOCK}) 生成预测")
    print(f"{'=' * 70}")

    target_feat, _, target_signal_df = extract_chan_features(TARGET_STOCK, START_DATE, END_DATE)

    predictions = {}
    if len(target_feat) > 0:
        probas = model.predict_proba(target_feat[feature_cols])[:, 1]
        for date, prob in zip(target_feat.index, probas):
            d = date.date() if hasattr(date, 'date') else date
            predictions[d] = float(prob)

        print(f"\n  三买事件: {len(predictions)}个")
        for d, p in sorted(predictions.items()):
            status = "OK" if p >= ML_THRESHOLD else "SKIP"
            print(f"    {d} | 概率={p:.2f} | {status}")
    else:
        print("  无三买信号")

    # ---- Step 6: 回测对比 ----
    print(f"\n{'=' * 70}")
    print(f"Step 4: 回测对比 ({TARGET_NAME})")
    print(f"{'=' * 70}")

    bh = calc_buy_and_hold(TARGET_STOCK, START_DATE, END_DATE)
    print(f"\n  买入持有: {bh*100:+.1f}%\n")

    tb = (target_signal_df['chan_signal'] == 3).sum()
    print(f"  三买信号: {tb}个")

    print(f"\n[基础三买策略]")
    r_basic = run_and_report(
        ChanBasicStrategy,
        stock_code=TARGET_STOCK,
        label='基础三买',
        plot=True,
        df=target_signal_df,
        data_class=ChanPandasData,
    )

    print(f"\n[ML增强三买策略] 阈值={ML_THRESHOLD}, 引擎={ml_engine}")
    r_ml = run_and_report(
        ChanMLStrategy,
        stock_code=TARGET_STOCK,
        label='ML三买',
        plot=True,
        df=target_signal_df,
        data_class=ChanPandasData,
        ml_threshold=ML_THRESHOLD,
        predictions=predictions,
    )

    # ---- 结果对比 ----
    print(f"\n{'=' * 70}")
    print("对比总结")
    print(f"{'=' * 70}")
    print(f"  {'指标':<12} {'基础三买':>14} {'ML三买':>14}")
    print(f"  {'-' * 42}")
    if bh is not None:
        print(f"  {'买入持有':<12} {bh*100:>+13.1f}% {bh*100:>+13.1f}%")
    print(f"  {'策略收益':<12} {r_basic['total_return']*100:>+13.2f}% {r_ml['total_return']*100:>+13.2f}%")
    print(f"  {'最大回撤':<12} {r_basic['max_drawdown']*100:>13.2f}% {r_ml['max_drawdown']*100:>13.2f}%")
    print(f"  {'夏普比率':<12} {r_basic['sharpe_ratio']:>14.2f} {r_ml['sharpe_ratio']:>14.2f}")
    print(f"  {'交易次数':<12} {r_basic['total_trades']:>14d} {r_ml['total_trades']:>14d}")
    print(f"  {'胜率':<12} {r_basic['win_rate']*100:>13.1f}% {r_ml['win_rate']*100:>13.1f}%")
    print(f"  {'盈亏比':<12} {r_basic['profit_loss_ratio']:>14.2f} {r_ml['profit_loss_ratio']:>14.2f}")

    print(f"\n  核心发现:")
    print(f"    - ML过滤利用缠论结构特征(中枢宽度/高度)判断三买质量")
    print(f"    - 结合技术指标(ADX/RSI/ATR)提升信号可靠性")
    print(f"    - 与海龟ML策略思路一致: 不改变策略逻辑, 只增强信号质量")

    print(f"\n  延伸方向:")
    print(f"    - LSTM时序模型: 用笔的序列特征预测趋势转折")
    print(f"    - 特征扩展: 大盘趋势、板块热度、资金流向")
    print(f"    - Walk-Forward: 滚动训练窗口, 适应市场风格变化")

    print("\n完成!")


if __name__ == '__main__':
    main()
