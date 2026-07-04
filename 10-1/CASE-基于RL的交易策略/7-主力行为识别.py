# -*- coding: utf-8 -*-
"""
CASE: 主力行为识别

Ethan 的第七步 -- 从市场微观结构特征识别主力行为

场景:
    Ethan 已经理解了做市和拆单的原理
    现在他站在散户的角度，学习如何从公开数据中
    识别主力机构 RL 策略留下的市场痕迹

核心内容:
  1. 模拟三种典型交易行为的 tick 数据:
     - 普通散户交易 (随机、低频)
     - RL 做市商 (高频对称报价、库存回归)
     - RL 拆单 (TWAP/VWAP变体、有节奏的大单拆分)
  2. 提取高频特征:
     - 订单流不平衡 (OFI)
     - 大单识别 (成交额阈值)
     - 撤单率
     - 价格冲击恢复速度
     - 成交节奏规律性
  3. 用随机森林分类器识别行为类型
  4. 可视化: 特征雷达图、典型模式示例
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================
# 模拟 tick 数据生成器
# ============================================================

class TickDataGenerator:
    """
    模拟不同交易行为的 tick 级别数据

    生成的每条 tick 包含:
        timestamp, price, volume, direction(1=买,-1=卖),
        order_type(limit/market), cancel_flag
    """

    def __init__(self, base_price=100.0, seed=None):
        self.base_price = base_price
        self.rng = np.random.RandomState(seed)

    def generate_retail(self, n_ticks=500):
        """
        模拟散户交易行为:
        - 交易间隔不规律 (指数分布)
        - 成交量小 (100-300股)
        - 买卖方向随机
        - 偶尔有大单 (追涨杀跌)
        - 撤单率低 (~5%)
        """
        price = self.base_price
        ticks = []

        for i in range(n_ticks):
            dt = self.rng.exponential(2.0)
            direction = self.rng.choice([1, -1])

            # 散户偶尔追涨杀跌
            if abs(price - self.base_price) / self.base_price > 0.01:
                if price > self.base_price:
                    direction = self.rng.choice([1, -1], p=[0.7, 0.3])
                else:
                    direction = self.rng.choice([1, -1], p=[0.3, 0.7])

            volume = self.rng.choice([100, 200, 300])
            # 5% 概率大单
            if self.rng.random() < 0.05:
                volume = self.rng.randint(1000, 3000)

            price += direction * self.rng.uniform(0.01, 0.05)
            cancel = 1 if self.rng.random() < 0.05 else 0

            ticks.append({
                'timestamp': i * dt,
                'price': round(price, 4),
                'volume': volume,
                'direction': direction,
                'order_type': 'market' if self.rng.random() < 0.6 else 'limit',
                'cancel': cancel,
            })

        return pd.DataFrame(ticks)

    def generate_mm_rl(self, n_ticks=500):
        """
        模拟 RL 做市商行为:
        - 交易间隔极短且规律 (近似等间隔)
        - 买卖成对出现 (对称报价)
        - 成交量稳定 (100股为主)
        - 库存回归行为 (大量后自动反向)
        - 撤单率高 (~30%, 频繁调整报价)
        - 价差异常稳定
        """
        price = self.base_price
        ticks = []
        inventory = 0

        for i in range(n_ticks):
            dt = self.rng.uniform(0.1, 0.3)

            # 对称报价: 买卖几乎同时
            if self.rng.random() < 0.7:
                # 库存管理: 库存过大时倾向反向
                if inventory > 200:
                    direction = -1
                elif inventory < -200:
                    direction = 1
                else:
                    direction = self.rng.choice([1, -1])
            else:
                direction = self.rng.choice([1, -1])

            volume = 100  # 做市商固定小单
            price += direction * self.rng.uniform(0.005, 0.02)
            inventory += direction * volume

            # 高撤单率
            cancel = 1 if self.rng.random() < 0.30 else 0

            ticks.append({
                'timestamp': i * dt,
                'price': round(price, 4),
                'volume': volume,
                'direction': direction,
                'order_type': 'limit',
                'cancel': cancel,
            })

        return pd.DataFrame(ticks)

    def generate_execution_rl(self, n_ticks=500):
        """
        模拟 RL 拆单执行行为:
        - 只有一个方向 (如持续买入)
        - 成交量有节奏 (类TWAP但有自适应)
        - U型分布 (开头和结尾执行量大)
        - 遇到大幅价格不利时暂停
        - 撤单率中等 (~15%)
        """
        price = self.base_price
        ticks = []
        total_target = 50000
        executed = 0

        for i in range(n_ticks):
            dt = self.rng.uniform(0.3, 0.8)

            # 主方向: 买入
            direction = 1

            # U型执行: 开头和结尾量大
            progress = i / n_ticks
            u_weight = 1.5 * (progress - 0.5) ** 2 + 0.3
            base_vol = int(total_target / n_ticks * u_weight * 2)
            volume = max(100, min(base_vol, 1000))
            volume = (volume // 100) * 100

            # 价格冲击: 持续买入推高价格
            impact = 0.001 * (executed / total_target)
            price += direction * self.rng.uniform(0.005, 0.03) + impact

            # 价格不利时减量
            if price > self.base_price * 1.01:
                volume = max(100, volume // 2)

            executed += volume
            cancel = 1 if self.rng.random() < 0.15 else 0

            ticks.append({
                'timestamp': i * dt,
                'price': round(price, 4),
                'volume': volume,
                'direction': direction,
                'order_type': 'limit' if self.rng.random() < 0.7 else 'market',
                'cancel': cancel,
            })

        return pd.DataFrame(ticks)


# ============================================================
# 高频特征提取
# ============================================================

def extract_features(tick_df, window=50):
    """
    从 tick 数据中提取行为特征

    参数:
        tick_df: tick 级 DataFrame
        window: 滚动窗口大小

    返回:
        dict: 特征名 -> 特征值
    """
    prices = tick_df['price'].values
    volumes = tick_df['volume'].values
    directions = tick_df['direction'].values
    timestamps = tick_df['timestamp'].values
    cancels = tick_df['cancel'].values
    n = len(tick_df)

    # 1. 订单流不平衡 (OFI)
    buy_vol = np.sum(volumes[directions == 1])
    sell_vol = np.sum(volumes[directions == -1])
    total_vol = buy_vol + sell_vol
    ofi = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0
    ofi_abs = abs(ofi)

    # 2. 大单比例 (成交额超过均值3倍)
    amounts = prices * volumes
    large_threshold = amounts.mean() * 3
    large_ratio = np.sum(amounts > large_threshold) / n

    # 3. 撤单率
    cancel_rate = cancels.mean()

    # 4. 成交间隔规律性 (间隔的变异系数, 越小越规律)
    if len(timestamps) > 1:
        intervals = np.diff(timestamps)
        interval_cv = np.std(intervals) / (np.mean(intervals) + 1e-8)
    else:
        interval_cv = 1.0

    # 5. 价格冲击恢复速度
    # 找到大单后的价格恢复
    recovery_speeds = []
    for i in range(n - window):
        if amounts[i] > large_threshold:
            pre_price = prices[max(0, i - 5):i].mean() if i > 5 else prices[i]
            post_prices = prices[i + 1:i + 11]
            if len(post_prices) > 5:
                recovery = abs(post_prices[-1] - prices[i]) / (abs(prices[i] - pre_price) + 1e-8)
                recovery_speeds.append(recovery)
    avg_recovery = np.mean(recovery_speeds) if recovery_speeds else 0.5

    # 6. 方向持续性 (连续同方向交易的平均长度)
    run_lengths = []
    current_run = 1
    for i in range(1, n):
        if directions[i] == directions[i - 1]:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    run_lengths.append(current_run)
    avg_run_length = np.mean(run_lengths) if run_lengths else 1

    # 7. 成交量变异系数
    vol_cv = np.std(volumes) / (np.mean(volumes) + 1e-8)

    # 8. 对称性 (买卖笔数差异)
    buy_count = np.sum(directions == 1)
    sell_count = np.sum(directions == -1)
    direction_symmetry = min(buy_count, sell_count) / (max(buy_count, sell_count) + 1e-8)

    # 9. 限价单比例
    limit_ratio = np.sum(tick_df['order_type'] == 'limit') / n

    # 10. 价格波动率 (归一化)
    price_vol = np.std(np.diff(prices)) / np.mean(prices) if len(prices) > 1 else 0

    return {
        'ofi_abs': ofi_abs,
        'large_ratio': large_ratio,
        'cancel_rate': cancel_rate,
        'interval_cv': interval_cv,
        'recovery_speed': avg_recovery,
        'run_length': avg_run_length,
        'vol_cv': vol_cv,
        'direction_symmetry': direction_symmetry,
        'limit_ratio': limit_ratio,
        'price_volatility': price_vol,
    }


# ============================================================
# 构建训练数据
# ============================================================

def build_dataset(n_samples_per_class=200, seed=42):
    """
    生成模拟数据集

    参数:
        n_samples_per_class: 每类样本数

    返回:
        X: 特征矩阵
        y: 标签 (0=散户, 1=RL做市, 2=RL拆单)
        feature_names: 特征名列表
    """
    gen = TickDataGenerator(seed=seed)
    features_list = []
    labels = []
    label_names = {0: '散户交易', 1: 'RL做市', 2: 'RL拆单'}

    for cls, (gen_fn, label) in enumerate([
        (gen.generate_retail, 0),
        (gen.generate_mm_rl, 1),
        (gen.generate_execution_rl, 2),
    ]):
        for i in range(n_samples_per_class):
            gen.rng = np.random.RandomState(seed + cls * 1000 + i)
            ticks = gen_fn(n_ticks=300)
            feats = extract_features(ticks)
            features_list.append(feats)
            labels.append(label)

    feature_names = list(features_list[0].keys())
    X = np.array([[f[fn] for fn in feature_names] for f in features_list])
    y = np.array(labels)

    return X, y, feature_names, label_names


# ============================================================
# 可视化
# ============================================================

def plot_feature_radar(X, y, feature_names, label_names):
    """特征雷达图: 三种行为的特征分布"""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
    angles += angles[:1]

    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for cls in range(3):
        mask = y == cls
        means = X[mask].mean(axis=0)
        # 归一化到 0-1
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        normalized = (means - mins) / (maxs - mins + 1e-8)
        values = normalized.tolist()
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=label_names[cls],
                color=colors[cls])
        ax.fill(angles, values, alpha=0.1, color=colors[cls])

    ax.set_xticks(angles[:-1])
    # 中文特征名映射
    cn_names = {
        'ofi_abs': 'OFI绝对值',
        'large_ratio': '大单比例',
        'cancel_rate': '撤单率',
        'interval_cv': '间隔规律性',
        'recovery_speed': '冲击恢复',
        'run_length': '方向持续',
        'vol_cv': '量变异系数',
        'direction_symmetry': '方向对称性',
        'limit_ratio': '限价单比例',
        'price_volatility': '价格波动',
    }
    labels_display = [cn_names.get(fn, fn) for fn in feature_names]
    ax.set_xticklabels(labels_display, fontsize=9)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title('三种交易行为的特征雷达图', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('outputs/7-行为特征雷达图.png', dpi=150, bbox_inches='tight')
    print("  图表已保存: outputs/7-行为特征雷达图.png")
    plt.close()


def plot_typical_patterns(seed=42):
    """展示三种典型行为的 tick 模式"""
    gen = TickDataGenerator(seed=seed)

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    data_configs = [
        ('散户交易', gen.generate_retail, '#3498db'),
        ('RL做市商', gen.generate_mm_rl, '#e74c3c'),
        ('RL拆单执行', gen.generate_execution_rl, '#2ecc71'),
    ]

    for row, (name, gen_fn, color) in enumerate(data_configs):
        ticks = gen_fn(300)

        # 价格轨迹
        ax = axes[row, 0]
        ax.plot(ticks['timestamp'], ticks['price'], color=color, linewidth=0.8)
        ax.set_title(f'{name} - 价格轨迹')
        ax.set_ylabel('价格')
        ax.grid(True, alpha=0.3)

        # 成交量分布
        ax = axes[row, 1]
        buy_mask = ticks['direction'] == 1
        sell_mask = ticks['direction'] == -1
        ax.bar(ticks.index[buy_mask], ticks.loc[buy_mask, 'volume'],
               color='#e74c3c', alpha=0.6, label='买入', width=1.5)
        ax.bar(ticks.index[sell_mask], -ticks.loc[sell_mask, 'volume'],
               color='#2ecc71', alpha=0.6, label='卖出', width=1.5)
        ax.set_title(f'{name} - 买卖量')
        ax.set_ylabel('成交量')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 交易间隔分布
        ax = axes[row, 2]
        intervals = np.diff(ticks['timestamp'].values)
        ax.hist(intervals, bins=30, color=color, alpha=0.7, edgecolor='white')
        cv = np.std(intervals) / (np.mean(intervals) + 1e-8)
        ax.set_title(f'{name} - 间隔分布 (CV={cv:.2f})')
        ax.set_xlabel('间隔')
        ax.grid(True, alpha=0.3)

    axes[2, 0].set_xlabel('时间')
    axes[2, 1].set_xlabel('Tick序号')

    plt.suptitle('三种交易行为的典型模式', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/7-典型行为模式.png', dpi=150, bbox_inches='tight')
    print("  图表已保存: outputs/7-典型行为模式.png")
    plt.close()


def plot_confusion_matrix(cm, label_names):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')

    labels = [label_names[i] for i in range(len(label_names))]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')

    for i in range(len(labels)):
        for j in range(len(labels)):
            text = f'{cm[i, j]}'
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=14)

    plt.colorbar(im)
    ax.set_title('主力行为分类 - 混淆矩阵', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/7-混淆矩阵.png', dpi=150, bbox_inches='tight')
    print("  图表已保存: outputs/7-混淆矩阵.png")
    plt.close()


def plot_feature_importance(clf, feature_names):
    """特征重要性排名"""
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    cn_names = {
        'ofi_abs': 'OFI绝对值',
        'large_ratio': '大单比例',
        'cancel_rate': '撤单率',
        'interval_cv': '间隔规律性',
        'recovery_speed': '冲击恢复',
        'run_length': '方向持续',
        'vol_cv': '量变异系数',
        'direction_symmetry': '方向对称性',
        'limit_ratio': '限价单比例',
        'price_volatility': '价格波动',
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [cn_names.get(feature_names[i], feature_names[i]) for i in indices]
    values = [importances[i] for i in indices]
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(names)))

    ax.barh(range(len(names)), values, color=colors, edgecolor='white')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('特征重要性')
    ax.set_title('随机森林 - 特征重要性排名', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('outputs/7-特征重要性.png', dpi=150, bbox_inches='tight')
    print("  图表已保存: outputs/7-特征重要性.png")
    plt.close()


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("第18讲 脚本7: 主力行为识别")
    print("Ethan 从市场微观结构特征识别主力 RL 策略的痕迹")
    print("=" * 60)

    SEED = 42

    # 1. 展示三种典型模式
    print(f"\n[1] 展示三种典型交易行为")
    plot_typical_patterns(seed=SEED)

    # 2. 构建数据集
    print(f"\n[2] 构建训练数据集")
    X, y, feature_names, label_names = build_dataset(
        n_samples_per_class=200, seed=SEED
    )
    print(f"  样本数: {len(X)} (每类 {len(X)//3} 个)")
    print(f"  特征数: {len(feature_names)}")
    print(f"  特征: {', '.join(feature_names)}")

    # 3. 各类别特征统计
    print(f"\n[3] 各类别关键特征均值")
    print(f"  {'特征':<18} | {'散户':>10} | {'RL做市':>10} | {'RL拆单':>10}")
    print(f"  {'-'*55}")
    for i, fn in enumerate(feature_names):
        vals = [X[y == c, i].mean() for c in range(3)]
        print(f"  {fn:<18} | {vals[0]:>10.4f} | {vals[1]:>10.4f} | {vals[2]:>10.4f}")

    # 4. 训练分类器
    print(f"\n[4] 训练随机森林分类器")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"  训练集准确率: {train_acc:.4f}")
    print(f"  测试集准确率: {test_acc:.4f}")

    # 5. 分类报告
    print(f"\n[5] 分类报告")
    y_pred = clf.predict(X_test)
    target_names = [label_names[i] for i in range(3)]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 6. 可视化
    print(f"\n[6] 可视化")
    plot_feature_radar(X, y, feature_names, label_names)

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, label_names)

    plot_feature_importance(clf, feature_names)

    # 7. 总结: 关键识别指标
    print(f"\n[7] Ethan 的主力行为识别要点")
    print(f"  {'='*50}")
    print(f"  疑似 RL 做市的特征:")
    print(f"    - 撤单率 > 25% (频繁调整报价)")
    print(f"    - 交易间隔CV < 0.5 (高度规律)")
    print(f"    - 方向对称性 > 0.8 (买卖均衡)")
    print(f"    - 成交量CV < 0.3 (固定小单)")
    print(f"  疑似 RL 拆单的特征:")
    print(f"    - 方向持续性 > 3.0 (持续单边)")
    print(f"    - OFI绝对值 > 0.5 (明显买/卖压)")
    print(f"    - 限价单比例 > 60% (耐心执行)")
    print(f"    - 成交量有U型分布 (开盘收盘量大)")
    print(f"  散户的典型特征:")
    print(f"    - 撤单率 < 10%")
    print(f"    - 偶尔大单 (追涨杀跌)")
    print(f"    - 交易间隔不规律 (CV > 1.0)")

    print(f"\n{'='*60}")
    print("第18讲全部脚本完成!")
    print("Ethan 已掌握: RL择时 -> 智能拆单 -> 高频做市 -> 主力识别")
    print(f"{'='*60}")
