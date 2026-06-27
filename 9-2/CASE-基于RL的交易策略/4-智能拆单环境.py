# -*- coding: utf-8 -*-
"""
第18讲 脚本4: 智能拆单环境

Ethan 的第四步 -- 构建订单执行环境，模拟大单拆分

场景:
    Ethan 需要在1个交易日内买入 10万股 某股票
    如果一次性下单，会造成巨大的市场冲击（价格被推高）
    他需要将大单拆成若干小单，在一天的不同时段分批执行
    目标: 最小化执行成本（实际均价 vs 到达价格）

核心内容:
  1. 市场冲击模型: price_impact = k * sqrt(v / ADV)
  2. 构建 OrderExecutionEnv (Gymnasium):
     - State: 剩余量比例、剩余时间比例、价格偏移、波动率、VWAP偏离
     - Action: 当前时段执行比例 (连续空间 0~1)
     - Reward: -执行缺口 (最小化执行成本)
  3. 实现 TWAP / VWAP 基线策略
  4. 单日模拟可视化
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from data_loader import load_stock_data, generate_intraday_data


# ============================================================
# 市场冲击模型
# ============================================================

class MarketImpactModel:
    """
    市场冲击模型

    线性-平方根混合模型:
        临时冲击 = eta * sigma * (v / V_bar) ^ beta
        永久冲击 = gamma * sigma * (v / V_bar)

    其中:
        v = 本次执行量
        V_bar = 日均成交量 (ADV)
        sigma = 波动率
        eta, gamma, beta = 模型参数

    参考: Almgren-Chriss (2000) 最优执行模型
    """
    def __init__(self, adv, sigma, eta=0.1, gamma=0.05, beta=0.5):
        """
        参数:
            adv: 日均成交量
            sigma: 日波动率
            eta: 临时冲击系数
            gamma: 永久冲击系数
            beta: 冲击指数 (0.5=平方根)
        """
        self.adv = adv
        self.sigma = sigma
        self.eta = eta
        self.gamma = gamma
        self.beta = beta

    def calc_impact(self, volume, current_price):
        """
        计算市场冲击导致的价格偏移

        返回:
            (临时冲击, 永久冲击, 冲击后价格)
        """
        if volume <= 0:
            return 0.0, 0.0, current_price

        participation = volume / self.adv
        temp_impact = self.eta * self.sigma * (participation ** self.beta)
        perm_impact = self.gamma * self.sigma * participation

        impacted_price = current_price * (1 + temp_impact + perm_impact)
        return temp_impact, perm_impact, impacted_price


# ============================================================
# 订单执行环境
# ============================================================

class OrderExecutionEnv(gym.Env):
    """
    订单执行环境

    将一个交易日分成 num_steps 个时段
    Agent 在每个时段决定执行总订单的多少比例
    目标: 最小化执行缺口 (Implementation Shortfall)

    执行缺口 = (实际执行均价 - 到达价格) / 到达价格
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, total_order, daily_data_list, adv,
                 num_steps=48, impact_eta=0.1, impact_gamma=0.05):
        """
        参数:
            total_order: 总订单量（股）
            daily_data_list: 日线数据 DataFrame 的行列表 (每行含 OHLCV)
            adv: 日均成交量
            num_steps: 每天分成多少个时段 (48 对应 5分钟)
            impact_eta: 临时冲击系数
            impact_gamma: 永久冲击系数
        """
        super().__init__()

        self.total_order = total_order
        self.daily_data_list = daily_data_list
        self.adv = adv
        self.num_steps = num_steps

        # 状态空间: [剩余量比例, 剩余时间比例, 价格偏移, 波动率, VWAP偏离]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(5,), dtype=np.float32
        )

        # 动作空间: 当前时段执行比例 [0, 1]
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

        # 冲击模型
        self.sigma = 0.02  # 默认日波动率 2%
        self.impact_model = MarketImpactModel(
            adv=adv, sigma=self.sigma,
            eta=impact_eta, gamma=impact_gamma
        )

        # 环境状态变量
        self.current_step = 0
        self.remaining_qty = total_order
        self.arrival_price = 0.0
        self.executed_qty = 0
        self.executed_cost = 0.0
        self.intraday = None

        # 记录轨迹
        self.history = {
            'step': [], 'price': [], 'exec_qty': [],
            'exec_price': [], 'remaining': [], 'vwap': [],
        }

        self._day_idx = 0

    def _load_day(self, day_idx):
        """加载并模拟某一天的分钟数据"""
        row = self.daily_data_list[day_idx % len(self.daily_data_list)]
        self.sigma = (row['high'] - row['low']) / row['close']
        self.impact_model.sigma = self.sigma
        self.intraday = generate_intraday_data(row, self.num_steps, seed=day_idx)
        self.arrival_price = self.intraday['prices'][0]

    def _get_obs(self):
        """构造观测向量"""
        remaining_ratio = self.remaining_qty / self.total_order
        time_ratio = (self.num_steps - self.current_step) / self.num_steps

        current_price = self.intraday['prices'][self.current_step]
        price_drift = (current_price - self.arrival_price) / self.arrival_price

        vol = self.sigma

        if self.executed_qty > 0:
            current_vwap = self.executed_cost / self.executed_qty
            vwap_drift = (current_vwap - self.arrival_price) / self.arrival_price
        else:
            vwap_drift = 0.0

        return np.array([
            remaining_ratio,
            time_ratio,
            price_drift,
            vol,
            vwap_drift,
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.remaining_qty = self.total_order
        self.executed_qty = 0
        self.executed_cost = 0.0
        self.history = {
            'step': [], 'price': [], 'exec_qty': [],
            'exec_price': [], 'remaining': [], 'vwap': [],
        }

        self._load_day(self._day_idx)
        self._day_idx += 1

        return self._get_obs(), {}

    def step(self, action):
        exec_ratio = float(np.clip(action[0], 0, 1))

        # 最后一步必须执行完剩余量
        if self.current_step == self.num_steps - 1:
            exec_qty = self.remaining_qty
        else:
            exec_qty = int(self.remaining_qty * exec_ratio)
            exec_qty = min(exec_qty, self.remaining_qty)

        # 计算市场冲击后的成交价
        market_price = self.intraday['prices'][self.current_step + 1]
        if exec_qty > 0:
            temp_impact, perm_impact, impacted_price = \
                self.impact_model.calc_impact(exec_qty, market_price)
            exec_price = impacted_price
        else:
            exec_price = market_price

        # 更新状态
        self.executed_cost += exec_qty * exec_price
        self.executed_qty += exec_qty
        self.remaining_qty -= exec_qty

        # 记录
        current_vwap = self.executed_cost / self.executed_qty if self.executed_qty > 0 else 0
        self.history['step'].append(self.current_step)
        self.history['price'].append(market_price)
        self.history['exec_qty'].append(exec_qty)
        self.history['exec_price'].append(exec_price if exec_qty > 0 else 0)
        self.history['remaining'].append(self.remaining_qty)
        self.history['vwap'].append(current_vwap)

        self.current_step += 1
        terminated = self.current_step >= self.num_steps

        # ---- 奖励设计 ----
        # 每步即时奖励 = -市场冲击成本 - 集中度惩罚
        reward = 0.0
        if exec_qty > 0:
            # (1) 冲击成本: 成交价偏离市场价越多，惩罚越大
            impact_cost = (exec_price - market_price) / market_price
            reward -= impact_cost * 50

            # (2) 集中度惩罚: 单步执行超过均匀量的部分额外惩罚
            uniform_qty = self.total_order / self.num_steps
            if exec_qty > uniform_qty * 2:
                excess = (exec_qty - uniform_qty * 2) / self.total_order
                reward -= excess * 5

        # (3) 未按时完成惩罚: 临近结束但剩余量过多
        time_left = (self.num_steps - self.current_step) / self.num_steps
        remaining_ratio = self.remaining_qty / self.total_order
        if time_left < 0.2 and remaining_ratio > 0.5:
            reward -= remaining_ratio * 2

        # (4) 终端奖励: 执行缺口
        if terminated and self.executed_qty > 0:
            actual_vwap = self.executed_cost / self.executed_qty
            is_cost = (actual_vwap - self.arrival_price) / self.arrival_price
            reward -= abs(is_cost) * 30

        return self._get_obs(), reward, terminated, False, {
            'exec_qty': exec_qty,
            'exec_price': exec_price,
        }

    def get_execution_summary(self):
        """获取执行摘要"""
        if self.executed_qty <= 0:
            return None
        actual_vwap = self.executed_cost / self.executed_qty
        is_cost = (actual_vwap - self.arrival_price) / self.arrival_price
        market_vwap = self.intraday['vwap']
        vwap_slip = (actual_vwap - market_vwap) / market_vwap

        return {
            'arrival_price': self.arrival_price,
            'actual_vwap': actual_vwap,
            'market_vwap': market_vwap,
            'implementation_shortfall': is_cost,
            'vwap_slippage': vwap_slip,
            'total_executed': self.executed_qty,
            'num_slices': sum(1 for q in self.history['exec_qty'] if q > 0),
        }


# ============================================================
# 基线策略
# ============================================================

class TWAPStrategy:
    """
    TWAP 策略: 均匀分配到每个时段

    最简单的拆单策略，不考虑市场状态
    """
    def __init__(self, total_order, num_steps):
        self.qty_per_step = total_order // num_steps
        self.remainder = total_order - self.qty_per_step * num_steps
        self.step = 0
        self.num_steps = num_steps

    def get_action(self, obs):
        self.step += 1
        if self.step == self.num_steps:
            ratio = 1.0
        else:
            ratio = self.qty_per_step / max(obs[0] * 100000, 1)
            ratio = np.clip(ratio, 0, 1)
        return np.array([ratio], dtype=np.float32)


class VWAPStrategy:
    """
    VWAP 策略: 按历史成交量分布分配

    使用 U 型成交量分布 (开盘和收盘量大)
    """
    def __init__(self, total_order, num_steps):
        x = np.linspace(0, 1, num_steps)
        weights = 1.5 * (x - 0.5) ** 2 + 0.3
        self.weights = weights / weights.sum()
        self.total_order = total_order
        self.num_steps = num_steps
        self.step = 0

    def get_action(self, obs):
        remaining_ratio = obs[0]
        if self.step < self.num_steps:
            target_ratio = self.weights[self.step]
            if remaining_ratio > 0:
                ratio = target_ratio / remaining_ratio
            else:
                ratio = 0
            self.step += 1
        else:
            ratio = 1.0
        return np.array([np.clip(ratio, 0, 1)], dtype=np.float32)


def run_baseline(env, strategy_class, total_order, num_steps, n_episodes=50):
    """运行基线策略多次，返回统计结果"""
    results = []
    for ep in range(n_episodes):
        strategy = strategy_class(total_order, num_steps)
        obs, _ = env.reset()
        while True:
            action = strategy.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        summary = env.get_execution_summary()
        if summary:
            results.append(summary)
    return results


# ============================================================
# 可视化
# ============================================================

def plot_single_execution(env, title="单日执行轨迹"):
    """可视化单日执行过程"""
    h = env.history
    if not h['step']:
        print("没有执行记录")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10),
                                         gridspec_kw={'height_ratios': [2, 1, 1]})

    steps = h['step']
    prices = h['price']
    exec_qtys = h['exec_qty']
    remaining = h['remaining']
    vwaps = h['vwap']

    # 上图: 价格走势 + 执行价格
    ax1.plot(steps, prices, 'gray', linewidth=1.5, label='市场价格')
    exec_steps = [s for s, q in zip(steps, exec_qtys) if q > 0]
    exec_prices = [p for p, q in zip(h['exec_price'], exec_qtys) if q > 0]
    if exec_steps:
        ax1.scatter(exec_steps, exec_prices, color='#e74c3c', s=50,
                    zorder=5, label='成交价格')

    ax1.axhline(y=env.arrival_price, color='#2ecc71', linestyle='--',
                alpha=0.7, label=f'到达价格: {env.arrival_price:.3f}')
    if vwaps[-1] > 0:
        ax1.axhline(y=vwaps[-1], color='#3498db', linestyle='--',
                    alpha=0.7, label=f'执行VWAP: {vwaps[-1]:.3f}')

    ax1.set_ylabel('价格')
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 中图: 每个时段的执行量
    ax2.bar(steps, exec_qtys, color='#3498db', alpha=0.7, width=0.8)
    ax2.set_ylabel('执行量 (股)')
    ax2.set_title('各时段执行量分布')
    ax2.grid(True, alpha=0.3)

    # 下图: 剩余量变化
    ax3.plot(steps, remaining, '#e74c3c', linewidth=1.5, marker='o', markersize=3)
    ax3.fill_between(steps, remaining, alpha=0.2, color='#e74c3c')
    ax3.set_ylabel('剩余量 (股)')
    ax3.set_xlabel('时段')
    ax3.set_title('剩余订单量')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/4-拆单执行轨迹.png', dpi=150, bbox_inches='tight')
    print(f"  图表已保存: outputs/4-拆单执行轨迹.png")
    plt.close()


def plot_baseline_comparison(twap_results, vwap_results):
    """对比 TWAP 和 VWAP 基线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 执行缺口分布
    ax = axes[0]
    twap_is = [r['implementation_shortfall'] * 10000 for r in twap_results]
    vwap_is = [r['implementation_shortfall'] * 10000 for r in vwap_results]

    ax.hist(twap_is, bins=20, alpha=0.6, color='#3498db', label='TWAP', edgecolor='white')
    ax.hist(vwap_is, bins=20, alpha=0.6, color='#e74c3c', label='VWAP', edgecolor='white')
    ax.axvline(x=np.mean(twap_is), color='#3498db', linestyle='--')
    ax.axvline(x=np.mean(vwap_is), color='#e74c3c', linestyle='--')
    ax.set_xlabel('执行缺口 (bps)')
    ax.set_ylabel('频次')
    ax.set_title('执行缺口分布对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # VWAP 滑点分布
    ax = axes[1]
    twap_slip = [r['vwap_slippage'] * 10000 for r in twap_results]
    vwap_slip = [r['vwap_slippage'] * 10000 for r in vwap_results]

    ax.hist(twap_slip, bins=20, alpha=0.6, color='#3498db', label='TWAP', edgecolor='white')
    ax.hist(vwap_slip, bins=20, alpha=0.6, color='#e74c3c', label='VWAP', edgecolor='white')
    ax.set_xlabel('VWAP滑点 (bps)')
    ax.set_ylabel('频次')
    ax.set_title('VWAP滑点分布对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('TWAP vs VWAP 基线对比', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/4-基线策略对比.png', dpi=150, bbox_inches='tight')
    print(f"  图表已保存: outputs/4-基线策略对比.png")
    plt.close()


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("第18讲 脚本4: 智能拆单环境")
    print("Ethan 构建订单执行环境，模拟大单拆分")
    print("=" * 60)

    # 加载数据
    STOCK_CODE = '510050.SH'
    START_DATE = '2023-01-01'
    END_DATE = '2026-04-09'
    TOTAL_ORDER = 100_000  # 10万股

    print(f"\n[1] 加载数据: {STOCK_CODE}")
    df = load_stock_data(STOCK_CODE, START_DATE, END_DATE)
    print(f"  数据量: {len(df)} 个交易日")

    # 计算日均成交量
    adv = df['volume'].mean()
    print(f"  日均成交量 (ADV): {adv:,.0f}")
    print(f"  订单占ADV比例: {TOTAL_ORDER / adv * 100:.2f}%")

    # 构建环境
    print(f"\n[2] 构建订单执行环境")
    daily_data_list = [df.iloc[i] for i in range(len(df))]
    NUM_STEPS = 48

    env = OrderExecutionEnv(
        total_order=TOTAL_ORDER,
        daily_data_list=daily_data_list,
        adv=adv,
        num_steps=NUM_STEPS,
        impact_eta=0.1,
        impact_gamma=0.05,
    )
    print(f"  时段数: {NUM_STEPS} (每时段约5分钟)")
    print(f"  观测空间: {env.observation_space.shape}")
    print(f"  动作空间: {env.action_space.shape}")

    # 验证环境: TWAP 单次执行
    print(f"\n[3] TWAP 单次执行验证")
    twap = TWAPStrategy(TOTAL_ORDER, NUM_STEPS)
    obs, _ = env.reset()
    while True:
        action = twap.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    summary = env.get_execution_summary()
    print(f"  到达价格:    {summary['arrival_price']:.4f}")
    print(f"  执行VWAP:    {summary['actual_vwap']:.4f}")
    print(f"  市场VWAP:    {summary['market_vwap']:.4f}")
    print(f"  执行缺口:    {summary['implementation_shortfall']*10000:.2f} bps")
    print(f"  VWAP滑点:    {summary['vwap_slippage']*10000:.2f} bps")
    print(f"  分片数:      {summary['num_slices']}")

    # 可视化单次执行
    print(f"\n[4] 可视化 TWAP 执行轨迹")
    plot_single_execution(env, "TWAP拆单执行轨迹")

    # 批量对比 TWAP vs VWAP
    print(f"\n[5] 批量对比 TWAP vs VWAP (各50次)")
    twap_results = run_baseline(env, TWAPStrategy, TOTAL_ORDER, NUM_STEPS, 50)
    vwap_results = run_baseline(env, VWAPStrategy, TOTAL_ORDER, NUM_STEPS, 50)

    twap_is = [r['implementation_shortfall'] * 10000 for r in twap_results]
    vwap_is = [r['implementation_shortfall'] * 10000 for r in vwap_results]

    print(f"\n  TWAP 执行缺口: 均值={np.mean(twap_is):.2f}bps, "
          f"标准差={np.std(twap_is):.2f}bps")
    print(f"  VWAP 执行缺口: 均值={np.mean(vwap_is):.2f}bps, "
          f"标准差={np.std(vwap_is):.2f}bps")

    plot_baseline_comparison(twap_results, vwap_results)

    print(f"\n{'='*60}")
    print("拆单环境构建完成! 下一步: 5-TWAP与RL拆单对比.py")
    print(f"{'='*60}")
