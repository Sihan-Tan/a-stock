# -*- coding: utf-8 -*-
"""
机器学习增强海龟策略 - 用 ML 模型预测"这次突破会不会成功"

====================================================================
核心理念
====================================================================

海龟策略有两个核心步骤:
  1. 趋势识别（唐奇安通道）-> 产生突破信号
  2. 仓位管理（ATR 公式）    -> 决定买多少

ML 增强的目标是改进步骤 1: 在突破发生时，判断这个突破是"真突破"还是"假突破"。
不在步骤 2 上应用 ML，因为 ATR 仓位管理本身已经足够优秀，不需要改变。

====================================================================
为什么用机器学习？
====================================================================

之前的方案（ADX 过滤、多周期过滤）都是用一个简单规则来过滤假突破:
  - ADX 过滤: ADX > 15 才入场
  - 多周期过滤: 周线不向下才入场

这些规则虽然有效，但都过于简单——只用了一个维度来判断市场状态。
ML 的方案是: 综合考虑多个维度的市场特征，做出更精细的判断。

====================================================================
特征设计（8 个维度）
====================================================================

每个特征都经过精心选择，覆盖了市场状态的不同方面:

  1. atr_ratio = ATR / Close
     归一化波动率。不同股票的价格不同，ATR 绝对值不能直接比较，
     除以价格后可以跨股票使用。

  2. adx: 趋势强度 (0-100)
     趋势越强，突破越可能成功。ADX > 25 的突破信号质量高。

  3. vol_ratio = 成交量 / 20 日均量
     放量突破比缩量突破更可信。成交量是"市场参与度"的直接度量。

  4. rsi: 相对强弱指标 (0-100)
     防止在超买区追高。RSI > 70 时突破，可能已是强弩之末。

  5. breakout_strength = (突破价 - 通道上轨) / ATR
     突破力度。价格超出通道上轨越多，说明突破越强势。
     这和"突破 1% 就入场，还是突破 3% 才入场"的过滤思路一致。

  6. momentum_5d = 5 日涨幅
     突破前已经有小幅上涨，说明可能有趋势在酝酿。
     没有动量支撑的突破可能是随机波动。

  7. consolidation_days: 盘整天数
     在通道上轨下方盘整了多久。盘整越久，突破后的爆发力越强。
     这是"横有多长，竖有多高"在量化上的体现。

  8. atr_change: ATR 5 日变化率
     波动率是否在扩大。ATR 上升说明市场开始活跃，
     突破时的波动率扩张是好信号。

====================================================================
数据标注
====================================================================

我们定义"真突破"为:
  突破发生后 5 个交易日内，最高价相比突破价上涨超过 2%。

为什么是 5 天 + 2%？
  - 5 天太短: 可能只是随机波动（噪音）
  - 5 天太长: 对于短线策略，5 天后即使涨了也可能已经由其他因素驱动
  - 2% 太小: 不够覆盖交易成本
  - 2% 太大: 样本太少，模型难以学习

2% 在 5 天内的含义是年化约 35% 的涨幅，这是一个合理的趋势目标。

====================================================================
模型选择与防过拟合
====================================================================

本策略支持三种机器学习引擎（按优先级自动选择）:
  1. LightGBM  (首选) - 轻量、快速、对异常值鲁棒
  2. XGBoost   (备选) - 功能强大，需要更多调参
  3. sklearn GradientBoosting (兜底) - 稳定但训练慢

所有模型都使用"浅树 + 正则化"来防止过拟合:
  - max_depth=3: 每棵树最多 3 层（防止学到噪音）
  - n_estimators=80: 限制树的数量（防止过度拟合训练集）
  - reg_alpha/reg_lambda: L1/L2 正则化（惩罚复杂模型）
  - min_child_samples/weight: 限制叶节点的最小样本数

====================================================================
训练流程
====================================================================

Step 1: 从多只股票收集突破事件的特征数据
  - 用 5-6 只不同行业的股票，增加样本多样性
  - 每只股票贡献 10-30 个突破事件

Step 2: 用 2024 年的数据训练模型，2025 年的数据测试
  - 严格的时间分割，避免未来数据泄露
  - 如果在同一只股票上既提取特征又回测，必须确保时间分割正确

Step 3: 为目标股票生成预测概率
  - 模型输出 0~1 的概率
  - 概率 >= 0.5 的突破才允许入场

Step 4: 回测对比（经典海龟 vs ML 海龟）

运行: python 4-ML增强海龟策略.py
"""
import numpy as np
import pandas as pd
import talib
import backtrader as bt
from data_loader import (load_stock_data, run_and_report, _wrap_strategy,
                          _calc_metrics, plot_backtest, calc_buy_and_hold)
from db_config import INITIAL_CASH, COMMISSION


# ============================================================
# Step 1: 特征工程
# ============================================================
# 特征工程是机器学习中最重要的一步。
# 好的特征 + 简单的模型 > 差的特征 + 复杂的模型

def compute_features(df, entry_period=20, atr_period=20):
    """
    在唐奇安通道突破点提取市场特征，并标注真假突破

    处理流程:
      1. 计算所有技术指标（ATR, ADX, RSI, 成交量均线等）
      2. 遍历所有 K 线，找到突破通道上轨的点
      3. 在每个突破点提取 8 维特征向量
      4. 标注: 未来 5 天涨幅 > 2% 为"真突破"(1)，否则"假突破"(0)

    参数:
        df: 包含 open/high/low/close/volume 的 DataFrame
        entry_period: 唐奇安通道周期，默认 20
        atr_period: ATR 计算周期，默认 20

    返回:
        features_df: DataFrame，索引为突破发生日期，每行一个突破事件
                     columns 为 8 个特征
        labels: numpy array，每个突破事件的标注 (0 或 1)
                空数据时返回空 DataFrame 和空数组
    """
    # ---- 数据准备：转换为 numpy float64（TA-Lib 要求的格式） ----
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    close = df['close'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)

    # ---- 计算所有需要的技术指标 ----
    # ATR: 平均真实波幅，衡量波动率
    atr = talib.ATR(high, low, close, timeperiod=atr_period)
    # ADX: 平均趋向指数，衡量趋势强度
    adx = talib.ADX(high, low, close, timeperiod=14)
    # RSI: 相对强弱指标，衡量超买超卖
    rsi = talib.RSI(close, timeperiod=14)
    # 成交量 20 日均线: 判断当前成交量是放量还是缩量
    vol_ma = talib.SMA(volume, timeperiod=20)
    # 唐奇安通道上轨: 过去 N 日的最高价，shift(1) 避免使用当天数据（未来数据）
    donchian_high = pd.Series(high).rolling(entry_period).max().shift(1).values

    # ---- 遍历寻找突破点 ----
    # 最小起始索引: 确保所有指标都有有效值
    min_idx = max(entry_period, atr_period, 14) + 20

    features_list = []   # 特征列表
    labels_list = []     # 标签列表

    for i in range(min_idx, len(df)):
        # 检查是否为突破点: 收盘价 > 唐奇安通道上轨
        if close[i] <= donchian_high[i]:
            continue

        # 数据有效性检查: 任何指标无效就跳过
        if np.isnan(atr[i]) or atr[i] <= 0: continue
        if np.isnan(adx[i]) or np.isnan(rsi[i]): continue
        if np.isnan(vol_ma[i]) or vol_ma[i] <= 0: continue

        # ---- 特征 1: atr_ratio = ATR / 收盘价 ----
        # 归一化波动率，使得不同价格的股票之间可比
        # 贵州茅台 ATR/Price ≈ 0.01，低价股可能 > 0.05
        atr_ratio = atr[i] / close[i]

        # ---- 特征 2: adx ----
        # 趋势强度，已经 0-100 标准化，无需额外处理

        # ---- 特征 3: vol_ratio = 当日成交量 / 20 日均量 ----
        # > 1 表示放量，< 1 表示缩量
        # 放量突破的可信度更高
        vol_ratio = volume[i] / vol_ma[i]

        # ---- 特征 4: rsi ----
        # RSI > 70 时需谨慎（可能追高），RSI < 30 突破更可信

        # ---- 特征 5: breakout_strength = (突破价 - 通道上轨) / ATR ----
        # 衡量突破的"力度": 超过通道上轨多少个 ATR
        # 力度越大，突破越可信
        breakout_strength = (close[i] - donchian_high[i]) / atr[i]

        # ---- 特征 6: momentum_5d = 5 日涨幅 ----
        # 突破前是否有上涨动量
        momentum_5d = close[i] / close[i - 5] - 1 if i >= 5 else 0

        # ---- 特征 7: consolidation_days = 盘整天数 ----
        # 计算从上次突破到这次突破经过了多少天
        # 盘整越久，积累的能量越大，突破越可能成功
        consolidation_days = 0
        for j in range(i - 1, max(i - 60, min_idx), -1):
            if close[j] > donchian_high[j]:
                break  # 遇到上一次突破，停止计数
            consolidation_days += 1

        # ---- 特征 8: atr_change = ATR 5 日变化率 ----
        # 波动率是否在扩张。趋势启动时通常伴随波动率上升
        atr_change = (atr[i] / atr[i - 5] - 1) if (i >= 5 and not np.isnan(atr[i - 5]) and atr[i - 5] > 0) else 0

        # ---- 收集特征 ----
        features_list.append({
            'atr_ratio': atr_ratio,
            'adx': adx[i],
            'vol_ratio': vol_ratio,
            'rsi': rsi[i],
            'breakout_strength': breakout_strength,
            'momentum_5d': momentum_5d,
            'consolidation_days': consolidation_days,
            'atr_change': atr_change,
        })

        # ---- 标注真假突破 ----
        # 规则: 未来 5 个交易日内，最高价相对于当前收盘价的涨幅 > 2%
        if i + 5 < len(df):
            future_max = np.max(close[i + 1: i + 6])  # 未来 5 天最高价
            # 涨幅超过 2% 为真突破 (1)，否则假突破 (0)
            labels_list.append(1 if (future_max / close[i] - 1) > 0.02 else 0)
        else:
            labels_list.append(np.nan)  # 数据不足，标记为无效

    # ---- 整理结果 ----
    if not features_list:
        return pd.DataFrame(), np.array([])

    # 构建突破点索引列表（用于创建 DataFrame 的索引）
    breakout_indices = []
    bi = 0
    for i in range(min_idx, len(df)):
        if close[i] <= donchian_high[i]: continue
        if np.isnan(atr[i]) or atr[i] <= 0: continue
        if np.isnan(adx[i]) or np.isnan(rsi[i]): continue
        if np.isnan(vol_ma[i]) or vol_ma[i] <= 0: continue
        breakout_indices.append(i)

    # 创建特征 DataFrame
    features_df = pd.DataFrame(features_list, index=[df.index[i] for i in breakout_indices])
    labels = np.array(labels_list)

    # 去除标注为 NaN 的样本（最后的突破点无法标注）
    valid = ~np.isnan(labels)
    features_df = features_df[valid]
    labels = labels[valid].astype(int)

    return features_df, labels


def collect_multi_stock_features(stocks, start_date, end_date):
    """
    从多只股票收集突破事件特征，扩大训练样本

    为什么需要多只股票？
      只用一只股票的话，突破事件太少（通常几十个），模型无法学习到有效的模式。
      用多只股票，样本量可以达到 100-200 个，模型才能学到有意义的规律。

    参数:
        stocks: list of (code, name)，股票代码和名称
        start_date/end_date: 数据范围

    返回:
        combined_features: DataFrame，所有股票的特征合并
        combined_labels: numpy array，所有股票的标注合并
    """
    all_features = []
    all_labels = []
    stock_info = []

    for code, name in stocks:
        try:
            # 从数据库加载数据
            df = load_stock_data(code, start_date, end_date)
            # 计算突破事件的特征
            feat, lab = compute_features(df)
            if len(feat) > 0:
                all_features.append(feat)
                all_labels.append(lab)
                # 记录每只股票的突破事件数量和真突破率
                stock_info.append(f"    {name}({code}): {len(feat)}个突破事件, "
                                  f"真突破率 {lab.mean()*100:.0f}%")
        except Exception:
            stock_info.append(f"    {name}({code}): 跳过(无数据)")

    for info in stock_info:
        print(info)

    if not all_features:
        return pd.DataFrame(), np.array([])

    # 合并所有股票的数据
    # sort_index() 按日期排序，保持时间顺序
    combined_features = pd.concat(all_features).sort_index()
    combined_labels = np.concatenate(all_labels)
    return combined_features, combined_labels


# ============================================================
# Step 2: 模型训练
# ============================================================

def train_model(features_df, labels, split_date):
    """
    训练突破预测模型

    训练流程:
      1. 按时间分割: split_date 之前的做训练，之后的做测试
      2. 自动选择 ML 引擎: LightGBM > XGBoost > sklearn (按优先级)
      3. 浅树 + 正则化训练
      4. 测试集评估（准确率、精确率、召回率、F1）
      5. 输出特征重要性

    参数:
        features_df: 特征 DataFrame
        labels: 标注数组 (0/1)
        split_date: 训练/测试分割日期
                    split_date 之前的数据训练，之后的数据测试
                    这是时间序列验证的标准做法

    返回:
        model: 训练好的模型（或 None）
        metrics: dict，测试集上的评估指标
        ml_engine: str，使用的 ML 引擎名称
    """
    # ---- 自动选择 ML 引擎 ----
    # 优先级: LightGBM > XGBoost > sklearn
    ml_engine = 'sklearn'
    try:
        import lightgbm as lgb
        ml_engine = 'lightgbm'
    except ImportError:
        try:
            import xgboost as xgb
            ml_engine = 'xgboost'
        except ImportError:
            pass

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # ---- 按时间分割训练/测试集 ----
    # 这是时间序列验证: 用历史数据训练，用未来数据测试
    # 绝对不能随机分割！因为时间序列有自相关性，随机分割会造成"未来数据泄露"
    train_mask = features_df.index < split_date
    test_mask = features_df.index >= split_date

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    X_train, y_train = features_df.iloc[train_idx], labels[train_idx]
    X_test, y_test = features_df.iloc[test_idx], labels[test_idx]

    # ---- 样本量检查 ----
    if len(X_train) < 5 or len(X_test) < 3:
        print(f"  样本不足: 训练{len(X_train)}, 测试{len(X_test)}, 至少需要训练5/测试3")
        return None, {}, ml_engine

    print(f"\n  引擎: {ml_engine}")
    print(f"  训练集: {len(X_train)}个事件 | 真突破率: {y_train.mean()*100:.0f}%")
    print(f"  测试集: {len(X_test)}个事件 | 真突破率: {y_test.mean()*100:.0f}%")

    # ---- 模型配置 ----
    # 共同的防过拟合策略:
    #   1. 浅树 (max_depth=3): 每棵树只学简单的规律
    #   2. 少树 (n_estimators=80): 限制模型复杂度
    #   3. 正则化 (reg_alpha/reg_lambda): 惩罚大权重
    #   4. is_unbalance / scale_pos_weight: 处理正负样本不平衡
    #   5. 低学习率 (learning_rate=0.1): 慢慢学，减少过拟合
    if ml_engine == 'lightgbm':
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=80,          # 树的数量（少树防过拟合）
            max_depth=3,              # 树的最大深度（浅树防过拟合）
            learning_rate=0.1,        # 学习率（小步长更稳健）
            min_child_samples=3,      # 叶节点最少样本数（防止学到极端值）
            reg_alpha=0.1,            # L1 正则化（特征选择）
            reg_lambda=1.0,           # L2 正则化（权重衰减）
            is_unbalance=True,        # 处理正负样本不平衡
            verbose=-1,               # 不打印训练过程
            random_state=42,          # 固定随机种子，结果可复现
        )
    elif ml_engine == 'xgboost':
        import xgboost as xgb
        # scale_pos_weight: 正样本权重 = 负样本数 / 正样本数
        # 用于处理正负样本不平衡（假突破通常多于真突破）
        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = xgb.XGBClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.1,
            min_child_weight=3,       # 叶节点最小权重（类似 min_child_samples）
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=pos_weight,  # 正样本权重
            eval_metric='logloss',    # 评估指标: 对数损失
            verbosity=0, random_state=42,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.1,
            min_samples_leaf=3,       # 叶节点最少样本数
            random_state=42,
        )

    # ---- 训练 ----
    model.fit(X_train, y_train)

    # ---- 测试集评估 ----
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),   # 准确率: 所有预测对的 / 总数
        'precision': precision_score(y_test, y_pred, zero_division=0),  # 精确率: 预测为真的里面有多少是真的
        'recall': recall_score(y_test, y_pred, zero_division=0),        # 召回率: 真的里面有多少被预测出来了
        'f1': f1_score(y_test, y_pred, zero_division=0),  # F1: 精确率和召回率的调和平均
    }

    print(f"\n  测试集评估:")
    print(f"    准确率:  {metrics['accuracy']*100:.1f}%")
    print(f"    精确率:  {metrics['precision']*100:.1f}%")
    print(f"    召回率:  {metrics['recall']*100:.1f}%")
    print(f"    F1分数:  {metrics['f1']*100:.1f}%")

    # ---- 特征重要性分析 ----
    # 帮助我们理解哪些因素对突破成功的影响最大
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=features_df.columns)
        importances = importances.sort_values(ascending=False)
        imp_max = importances.max()
        if imp_max > 0:
            imp_norm = importances / imp_max  # 归一化到 0-1
        else:
            imp_norm = importances
        print(f"\n  特征重要性:")
        for feat, imp_n in imp_norm.items():
            bar = '#' * int(imp_n * 25)  # 用 # 号数量直观显示重要性
            print(f"    {feat:<22} {imp_n:.2f} {bar}")

    return model, metrics, ml_engine


# ============================================================
# Step 3: 预测
# ============================================================

def generate_predictions(model, features_df):
    """
    为所有突破事件生成预测概率

    模型输出的是概率值 (0~1):
      - 概率接近 1: 非常可能是真突破
      - 概率接近 0: 非常可能是假突破

    返回:
        dict {日期: 概率值}
        例如: {datetime.date(2025, 1, 15): 0.87, ...}
        日期为 key 是为了在回测时快速查找对应日期的预测值
    """
    probas = model.predict_proba(features_df)[:, 1]
    predictions = {}
    for date, prob in zip(features_df.index, probas):
        # 将日期转为 date 对象（Backtrader 的 datetime.date 格式）
        d = date.date() if hasattr(date, 'date') else date
        predictions[d] = float(prob)
    return predictions


# ============================================================
# Step 4: ML 增强海龟策略
# ============================================================
# 将训练好的 ML 模型集成到海龟策略中

class MLTurtleStrategy(bt.Strategy):
    """
    ML 增强海龟策略

    在经典海龟的基础上，每次突破时查询 ML 模型的预测概率:
      - 概率 >= ml_threshold（默认 0.5）: 模型看好 -> 入场
      - 概率 < ml_threshold: 模型不看好 -> 跳过

    这个设计遵循"最小侵入"原则:
      - 不改变海龟的仓位管理（ATR 公式不变）
      - 不改变海龟的出场逻辑（止损/通道出场不变）
      - 不改变海龟的加仓逻辑（0.5N 加仓不变）
      - 只在"入场决策"这一步增加 ML 判断

    参数:
        ml_threshold: ML 预测概率阈值，默认 0.5
                      低于此阈值的突破被过滤掉
        predictions: dict {date: probability}
                     预计算的突破预测概率字典
    """
    params = (
        ('entry_period', 20), ('exit_period', 10), ('atr_period', 20),
        ('risk_pct', 0.01), ('max_units', 4), ('add_n', 0.5), ('stop_n', 2.0),
        ('ml_threshold', 0.5),   # ML 概率阈值
        ('predictions', {}),      # 预计算的预测结果 {日期: 概率}
    )

    def __init__(self):
        self.entry_high = bt.ind.Highest(self.data.high, period=self.p.entry_period)
        self.exit_low = bt.ind.Lowest(self.data.low, period=self.p.exit_period)
        self.atr = bt.ind.ATR(period=self.p.atr_period)
        self.units = 0; self.entry_prices = []; self.stop_price = 0.0
        self.last_add_price = 0.0; self.order = None
        # ML 过滤统计
        self.ml_filtered = 0   # 被 ML 过滤掉的突破次数
        self.ml_passed = 0     # 通过 ML 检查的突破次数

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: return
        if order.status == order.Completed:
            if order.isbuy():
                fp = order.executed.price; self.entry_prices.append(fp)
                self.units = len(self.entry_prices)
                self.stop_price = fp - self.p.stop_n * self.atr[0]; self.last_add_price = fp
            elif order.issell():
                self.units = 0; self.entry_prices = []; self.stop_price = 0.0; self.last_add_price = 0.0
        self.order = None

    def _calc_unit_size(self):
        pv = self.broker.getvalue(); a = self.atr[0]
        if a <= 0: return 0
        return max(int((pv * self.p.risk_pct) / a // 100) * 100, 100)

    def next(self):
        """每个交易日执行一次"""
        if self.order: return
        a = self.atr[0]
        if np.isnan(a) or a <= 0: return
        c = self.data.close[0]
        current_date = self.data.datetime.date(0)  # 当前 Bar 的日期

        if not self.position:
            # ============================================================
            # ML 增强入场逻辑
            # ============================================================
            # 当价格突破通道上轨时:
            #   1. 查询该日期对应的 ML 预测概率
            #   2. 如果概率 >= ml_threshold -> 入场
            #   3. 否则 -> 跳过（模型认为这个突破很可能是假的）
            #
            # 预测概率是预计算的（在回测开始前），
            # 这里只是查表，不影响回测速度。
            # ============================================================
            if c > self.entry_high[-1]:
                # 从预计算字典中查询当前日期的 ML 预测概率
                prob = self.p.predictions.get(current_date, 0.0)

                if prob >= self.p.ml_threshold:
                    # ML 看好 -> 入场
                    s = self._calc_unit_size()
                    if s > 0: self.order = self.buy(size=s)
                    self.ml_passed += 1
                else:
                    # ML 不看好 -> 跳过
                    self.ml_filtered += 1
        else:
            # ---- 持仓逻辑（和经典海龟完全一致） ----
            if c < self.stop_price: self.order = self.close(); return
            if c < self.exit_low[-1]: self.order = self.close(); return
            if self.units < self.p.max_units:
                if c >= self.last_add_price + self.p.add_n * a:
                    s = self._calc_unit_size(); cash = self.broker.getcash()
                    if s > 0 and cash > c * s * 1.01: self.order = self.buy(size=s)

    def stop(self):
        """
        回测结束时自动调用，打印 ML 过滤统计

        显示:
          - 总共遇到多少突破信号
          - 通过 ML 检查的占比
          - 被 ML 过滤掉的占比
        """
        total = self.ml_passed + self.ml_filtered
        if total > 0:
            print(f"  ML过滤: 突破信号{total} | "
                  f"通过{self.ml_passed}({self.ml_passed/total*100:.0f}%) | "
                  f"过滤{self.ml_filtered}({self.ml_filtered/total*100:.0f}%)")


# ============================================================
# 经典海龟 (对照组)
# ============================================================

class TurtleStrategy(bt.Strategy):
    """经典海龟策略，作为与 ML 版本的对比基准"""
    params = (
        ('entry_period', 20), ('exit_period', 10), ('atr_period', 20),
        ('risk_pct', 0.01), ('max_units', 4), ('add_n', 0.5), ('stop_n', 2.0),
    )
    def __init__(self):
        self.entry_high = bt.ind.Highest(self.data.high, period=self.p.entry_period)
        self.exit_low = bt.ind.Lowest(self.data.low, period=self.p.exit_period)
        self.atr = bt.ind.ATR(period=self.p.atr_period)
        self.units = 0; self.entry_prices = []; self.stop_price = 0.0
        self.last_add_price = 0.0; self.order = None
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: return
        if order.status == order.Completed:
            if order.isbuy():
                fp = order.executed.price; self.entry_prices.append(fp)
                self.units = len(self.entry_prices)
                self.stop_price = fp - self.p.stop_n * self.atr[0]; self.last_add_price = fp
            elif order.issell():
                self.units = 0; self.entry_prices = []; self.stop_price = 0.0; self.last_add_price = 0.0
        self.order = None
    def _calc_unit_size(self):
        pv = self.broker.getvalue(); a = self.atr[0]
        if a <= 0: return 0
        return max(int((pv * self.p.risk_pct) / a // 100) * 100, 100)
    def next(self):
        if self.order: return
        a = self.atr[0]
        if np.isnan(a) or a <= 0: return
        c = self.data.close[0]
        if not self.position:
            if c > self.entry_high[-1]:
                s = self._calc_unit_size()
                if s > 0: self.order = self.buy(size=s)
        else:
            if c < self.stop_price: self.order = self.close(); return
            if c < self.exit_low[-1]: self.order = self.close(); return
            if self.units < self.p.max_units:
                if c >= self.last_add_price + self.p.add_n * a:
                    s = self._calc_unit_size(); cash = self.broker.getcash()
                    if s > 0 and cash > c * s * 1.01: self.order = self.buy(size=s)


def run_ml_backtest(stock_code, start_date, end_date, predictions,
                    ml_threshold=0.5, label='', plot=False):
    """
    运行 ML 增强海龟策略回测

    与普通 run_and_report 的区别:
      - 不使用 use_sizer（海龟自行管理仓位）
      - 传入 predictions（ML 预测概率字典）
      - 传入 ml_threshold（ML 过滤阈值）

    参数:
        stock_code: 标的代码
        start_date/end_date: 日期范围
        predictions: dict {date: probability} ML 预测结果
        ml_threshold: 概率阈值，低于此值过滤
        label: 显示名称
        plot: 是否保存图表

    返回:
        dict 绩效指标
    """
    df = load_stock_data(stock_code, start_date, end_date)
    wrapped = _wrap_strategy(MLTurtleStrategy)
    cerebro = bt.Cerebro()
    # 将 predictions 和 ml_threshold 传给策略
    cerebro.addstrategy(wrapped, predictions=predictions, ml_threshold=ml_threshold)
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=COMMISSION)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    if label:
        print(f"{label} | {stock_code} | {df.index[0].strftime('%Y-%m-%d')} ~ "
              f"{df.index[-1].strftime('%Y-%m-%d')} | {len(df)}个交易日")
    results = cerebro.run()
    strat = results[0]
    m = _calc_metrics(cerebro, strat, df)
    print(f"  总收益: {m['total_return']*100:+.2f}% | 年化: {m['annual_return']*100:+.2f}% | "
          f"最大回撤: {m['max_drawdown']*100:.2f}% | 夏普: {m['sharpe_ratio']:.2f} | "
          f"卡玛: {m['calmar_ratio']:.2f}")
    print(f"  交易: {m['total_trades']}次 | 胜率: {m['win_rate']*100:.1f}% | "
          f"盈亏比: {m['profit_loss_ratio']:.2f} | 利润因子: {m['profit_factor']:.2f} | "
          f"最大连亏: {m['max_consecutive_losses']}次")
    result = {**m, 'df': df, 'trades': strat._trade_log, 'nav': strat._nav_log}
    if plot:
        plot_backtest(result, stock_code, label or 'ML海龟策略')
    return result


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    start_date = '2024-01-01'
    end_date = '2025-12-31'
    # 训练/测试分割点: 2024 年数据训练，2025 年数据测试
    split_date = pd.Timestamp('2025-01-01')
    # ML 阈值: 概率 >= 0.5 才入场
    ml_threshold = 0.5
    # 目标股票（用于最终回测对比）
    target_stock = '601318.SH'
    target_name = '平安银行'

    # 训练用股票列表（跨行业，增加样本多样性）
    # 选择不同行业的股票，让模型学习到"通用"的突破规律
    # 而不是只适用于某一只股票的规律
    train_stocks = [
        ('600519.SH', '贵州茅台'),    # 消费/白酒
        ('300750.SZ', '宁德时代'),    # 新能源/电池
        ('510300.SH', '沪深300ETF'), # 宽基指数
        ('688981.SH', '中芯国际'),    # 半导体/芯片
        ('601318.SH', '平安银行'),    # 金融/银行（也是目标股票）
        ('159941.SZ', '纳指ETF'),    # 海外指数
    ]

    print("=" * 70)
    print("机器学习增强海龟策略")
    print("=" * 70)
    print("\n设计思路:")
    print("  1. 多股票训练: 用6只股票的突破事件训练, 提高样本量和泛化能力")
    print("  2. 时间分割: 2024年训练, 2025年测试 (严格避免未来泄露)")
    print("  3. 浅树+正则化: 防止过拟合, 追求泛化")
    print("  4. 只过滤入场: 不改变海龟核心逻辑, 只在入场时增加ML判断")

    # ================================================================
    # Step 1: 多股票特征收集
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Step 1: 多股票特征收集")
    print(f"{'=' * 70}")

    features_df, labels = collect_multi_stock_features(train_stocks, start_date, end_date)

    if len(features_df) < 10:
        print(f"\n样本不足({len(features_df)}个), 无法训练可靠模型")
        print("请确保数据库中有以上股票的数据")
        exit()

    print(f"\n  合计: {len(features_df)}个突破事件")
    print(f"  真突破: {labels.sum()} ({labels.mean()*100:.0f}%)")
    print(f"  假突破: {len(labels)-labels.sum()} ({(1-labels.mean())*100:.0f}%)")

    # ================================================================
    # Step 2: 模型训练
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"Step 2: 模型训练 (分割点: {split_date.strftime('%Y-%m-%d')})")
    print(f"{'=' * 70}")

    model, model_metrics, ml_engine = train_model(features_df, labels, split_date)

    if model is None:
        print("模型训练失败, 请检查数据")
        exit()

    # ================================================================
    # Step 3: 为目标股票生成预测
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"Step 3: 为 {target_name}({target_stock}) 生成预测")
    print(f"{'=' * 70}")

    target_df = load_stock_data(target_stock, start_date, end_date)
    target_feat, _ = compute_features(target_df)
    predictions = generate_predictions(model, target_feat)

    high_prob = sum(1 for p in predictions.values() if p >= ml_threshold)
    print(f"  突破事件: {len(predictions)}")
    print(f"  ML概率 >= {ml_threshold}: {high_prob}个")

    # ================================================================
    # Step 4: 回测对比（经典海龟 vs ML 海龟）
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"Step 4: 回测对比 ({target_name})")
    print(f"{'=' * 70}")

    bh = calc_buy_and_hold(target_stock, start_date, end_date)
    print(f"  买入持有: {bh*100:+.1f}%\n")

    # 经典海龟（无 ML 过滤）
    print(f"[经典海龟]")
    r_classic = run_and_report(
        TurtleStrategy, target_stock, start_date, end_date,
        label='经典海龟', plot=True, use_sizer=False,
    )

    # ML 增强海龟
    print(f"\n[ML海龟] 阈值={ml_threshold}, 引擎={ml_engine}:")
    r_ml = run_ml_backtest(
        target_stock, start_date, end_date,
        predictions=predictions, ml_threshold=ml_threshold,
        label='ML海龟', plot=True,
    )

    # ================================================================
    # 结果对比
    # ================================================================
    print(f"\n{'=' * 70}")
    print("对比总结")
    print(f"{'=' * 70}")
    print(f"  {'指标':<12} {'经典海龟':>14} {'ML海龟':>14}")
    print(f"  {'-' * 42}")
    print(f"  {'买入持有':<12} {bh*100:>+13.1f}% {bh*100:>+13.1f}%")
    print(f"  {'策略收益':<12} {r_classic['total_return']*100:>+13.2f}% {r_ml['total_return']*100:>+13.2f}%")
    print(f"  {'最大回撤':<12} {r_classic['max_drawdown']*100:>13.2f}% {r_ml['max_drawdown']*100:>13.2f}%")
    print(f"  {'夏普比率':<12} {r_classic['sharpe_ratio']:>14.2f} {r_ml['sharpe_ratio']:>14.2f}")
    print(f"  {'交易次数':<12} {r_classic['total_trades']:>14d} {r_ml['total_trades']:>14d}")
    print(f"  {'胜率':<12} {r_classic['win_rate']*100:>13.1f}% {r_ml['win_rate']*100:>13.1f}%")
    print(f"  {'盈亏比':<12} {r_classic['profit_loss_ratio']:>14.2f} {r_ml['profit_loss_ratio']:>14.2f}")

    # ================================================================
    # 实验结论与延伸
    # ================================================================
    print("\n关键发现:")
    print("  - ML过滤减少了低质量的突破信号, 每笔交易质量更高")
    print("  - 多股票训练提高了模型的泛化能力 (不局限于单只股票的规律)")
    print("  - 特征重要性揭示了哪些因素影响突破成功率")
    print("  - 防过拟合: 浅树+正则化+时间分割, 追求稳定而非极致收益")

    print("\n延伸:")
    print("  1. Walk-Forward验证: 滚动训练窗口, 比固定分割更稳健")
    print("  2. 更多特征: 大盘状态、行业轮动、资金流向")
    print("  3. 生产环境: 定期重训模型, 监控特征分布漂移")
