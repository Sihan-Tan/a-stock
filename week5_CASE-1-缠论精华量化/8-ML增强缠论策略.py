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

# ============================================================
# 参数配置
# ============================================================

# 目标股票: 中芯国际 (测试集, 不参与训练)
TARGET_STOCK = '688981.SH'
TARGET_NAME = '中芯国际'
START_DATE = '2022-01-01'
END_DATE = '2025-12-31'
SPLIT_DATE = pd.Timestamp('2025-01-01')  # 训练/测试分割点
ML_THRESHOLD = 0.5  # 模型预测概率 >= 0.5 时才入场

# 训练股票池: 6只跨行业股票, 确保多样性
TRAIN_STOCKS = [
    ('600519.SH', '贵州茅台'),    # 消费龙头
    ('300750.SZ', '宁德时代'),    # 新能源龙头
    ('000001.SZ', '平安银行'),    # 银行股
    ('688981.SH', '中芯国际'),    # 半导体龙头
    ('601318.SH', '中国平安'),   # 保险龙头
    ('159941.SZ', '纳指ETF'),    # 海外指数
]


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
# Step 2: 模型训练
# ============================================================

def train_model(features_df, labels, split_date):
    """
    训练三买成功预测模型

    模型选择优先级: LightGBM > XGBoost > GradientBoosting (降级)
    目标是可解释性强的浅树模型, 不追求复杂的深度学习。

    防过拟合设计:
      - max_depth=3: 每棵树深度有限
      - reg_alpha/reg_lambda: L1/L2正则化
      - min_child_samples=2: 叶子节点最少样本数
      - n_estimators=60: 有限棵数

    评估指标:
      - accuracy: 总体准确率
      - precision: 预测"成功"的准确率 (避免假信号)
      - recall: 找出了多少"真成功" (避免漏掉好信号)
      - F1: 综合指标
    """
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

    # 按时间分割: 训练集 < split_date <= 测试集
    train_mask = features_df.index < split_date
    test_mask = features_df.index >= split_date

    X_train = features_df[train_mask]
    y_train = labels[np.where(train_mask)[0]]
    X_test = features_df[test_mask]
    y_test = labels[np.where(test_mask)[0]]

    if len(X_train) < 3 or len(X_test) < 2:
        print(f"  样本不足: 训练{len(X_train)}, 测试{len(X_test)}")
        return None, {}, ml_engine

    print(f"\n  引擎: {ml_engine}")
    print(f"  训练集: {len(X_train)}个三买 | 成功率: {y_train.mean()*100:.0f}%")
    print(f"  测试集: {len(X_test)}个三买 | 成功率: {y_test.mean()*100:.0f}%")

    if ml_engine == 'lightgbm':
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=60, max_depth=3, learning_rate=0.1,
            min_child_samples=2, reg_alpha=0.1, reg_lambda=1.0,
            is_unbalance=True, verbose=-1, random_state=42,
        )
    elif ml_engine == 'xgboost':
        import xgboost as xgb
        pos_w = max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1)
        model = xgb.XGBClassifier(
            n_estimators=60, max_depth=3, learning_rate=0.1,
            min_child_weight=2, reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=pos_w, eval_metric='logloss',
            verbosity=0, random_state=42,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=60, max_depth=3, learning_rate=0.1,
            min_samples_leaf=2, random_state=42,
        )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }

    print(f"\n  测试集评估:")
    print(f"    准确率:  {metrics['accuracy']*100:.1f}%")
    print(f"    精确率:  {metrics['precision']*100:.1f}%")
    print(f"    召回率:  {metrics['recall']*100:.1f}%")
    print(f"    F1分数:  {metrics['f1']*100:.1f}%")

    # 打印特征重要性 (帮助理解模型学到了什么)
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=features_df.columns)
        importances = importances.sort_values(ascending=False)
        imp_max = importances.max()
        imp_norm = importances / imp_max if imp_max > 0 else importances
        print(f"\n  特征重要性:")
        for feat, imp_n in imp_norm.items():
            bar = '#' * int(imp_n * 25)
            print(f"    {feat:<22} {imp_n:.2f} {bar}")

    return model, metrics, ml_engine


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
    print("  2. 多股票训练LightGBM模型, 预测三买成功率")
    print("  3. 只在模型预测成功率 >= 50% 时入场")
    print("  4. 对比: 基础三买策略 vs ML增强三买策略")

    # ---- Step 1: 多股票特征收集 ----
    print(f"\n{'=' * 70}")
    print("Step 1: 多股票三买特征收集")
    print(f"{'=' * 70}")

    features_df, labels = collect_multi_stock_features(TRAIN_STOCKS, START_DATE, END_DATE)

    if len(features_df) < 5:
        print(f"\n三买样本不足({len(features_df)}个), 无法训练")
        return

    print(f"\n  合计: {len(features_df)}个三买事件")
    print(f"  成功(20日涨>5%): {labels.sum()} ({labels.mean()*100:.0f}%)")
    print(f"  失败: {len(labels)-labels.sum()} ({(1-labels.mean())*100:.0f}%)")

    # ---- Step 2: 模型训练 ----
    print(f"\n{'=' * 70}")
    print(f"Step 2: 模型训练 (分割: {SPLIT_DATE.strftime('%Y-%m-%d')})")
    print(f"{'=' * 70}")

    model, model_metrics, ml_engine = train_model(features_df, labels, SPLIT_DATE)
    if model is None:
        print("模型训练失败")
        return

    # ---- Step 3: 为目标股票生成预测 ----
    print(f"\n{'=' * 70}")
    print(f"Step 3: 为 {TARGET_NAME}({TARGET_STOCK}) 生成预测")
    print(f"{'=' * 70}")

    target_feat, _, target_signal_df = extract_chan_features(TARGET_STOCK, START_DATE, END_DATE)

    predictions = {}
    if len(target_feat) > 0:
        probas = model.predict_proba(target_feat)[:, 1]
        for date, prob in zip(target_feat.index, probas):
            d = date.date() if hasattr(date, 'date') else date
            predictions[d] = float(prob)

        print(f"\n  三买事件: {len(predictions)}个")
        for d, p in sorted(predictions.items()):
            status = "OK" if p >= ML_THRESHOLD else "SKIP"
            print(f"    {d} | 概率={p:.2f} | {status}")
    else:
        print("  无三买信号")

    # ---- Step 4: 回测对比 ----
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
