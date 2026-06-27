# -*- coding: utf-8 -*-
"""
第18讲 脚本1: 搭建RL交易环境

Ethan 的第一步 -- 用 Gymnasium 构建股票交易环境

核心内容:
  1. 构建 StockTradingEnv (继承 gymnasium.Env)
     - State: 回看5日 OHLC 的 Z-score 标准化 (20维)
     - Action: {0:持仓, 1:买入, 2:卖出}
     - Reward: 基于持仓状态+动作的四种组合收益率
     - 内置交易成本 (佣金万1.5 + 印花税万5，仅卖出)
  2. 用随机Agent跑通完整 episode，验证环境正确性
  3. 可视化一个 episode 的价格与动作轨迹
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib
from data_loader import load_stock_data
from db_config import COMMISSION, STAMP_TAX, INITIAL_CASH

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================
# 股票交易环境
# ============================================================

class StockTradingEnv(gym.Env):
    """
    股票日线择时环境

    状态空间 (20维):
        回看 lookback 日的 [open, high, low, close]，
        每个值相对于过去 norm_window 日收盘价做 Z-score 标准化

    动作空间 (离散3):
        0 = 持仓不动 (hold)
        1 = 买入 (全仓做多)
        2 = 卖出 (平多)

    奖励设计 (四种情况):
        未持仓 + 买入 -> 下一日多头收益率 (扣手续费)
        未持仓 + 其他 -> 下一日空仓收益 (=0)
        已持仓 + 持有  -> 下一日持仓收益率
        已持仓 + 卖出  -> 平仓收益 (扣手续费)
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, df, lookback=5, norm_window=252,
                 commission=COMMISSION, stamp_tax=STAMP_TAX,
                 initial_cash=INITIAL_CASH):
        """
        参数:
            df: 日线DataFrame (open/high/low/close/volume)
            lookback: 回看天数
            norm_window: Z-score 标准化的窗口
            commission: 佣金费率 (买卖各收, 默认万1.5)
            stamp_tax: 印花税 (仅卖出收, 默认万5)
            initial_cash: 初始资金
        """
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.lookback = lookback
        self.norm_window = norm_window
        self.commission = commission
        self.stamp_tax = stamp_tax

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(lookback * 4,), dtype=np.float32
        )

        # 有效交易区间: 需要 norm_window 的历史来计算 Z-score
        self.start_idx = max(lookback, norm_window)
        self.end_idx = len(self.df) - 2  # 需要 t+1 来计算奖励

        self.current_step = self.start_idx
        self.position = 0        # 0=空仓, 1=持仓
        self.entry_price = 0.0   # 买入价格

        # 记录轨迹用于可视化
        self.history = {
            'step': [], 'price': [], 'action': [],
            'reward': [], 'position': [], 'nav': [],
        }
        self.initial_cash = float(initial_cash)
        self.cash = self.initial_cash
        self.shares = 0

    def _get_obs(self):
        """构造观测向量: lookback 日 OHLC 的 Z-score"""
        idx = self.current_step
        lb = self.lookback

        # 取 norm_window 的历史收盘价计算均值和标准差
        hist_start = max(0, idx - self.norm_window)
        hist_close = self.df['close'].iloc[hist_start:idx].values
        mu = hist_close.mean()
        sigma = hist_close.std()
        if sigma < 1e-8:
            sigma = 1.0

        obs = []
        for offset in range(lb, 0, -1):
            row = self.df.iloc[idx - offset]
            obs.extend([
                (row['open'] - mu) / sigma,
                (row['high'] - mu) / sigma,
                (row['low'] - mu) / sigma,
                (row['close'] - mu) / sigma,
            ])

        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.start_idx
        self.position = 0
        self.entry_price = 0.0
        self.cash = self.initial_cash
        self.shares = 0
        self.history = {
            'step': [], 'price': [], 'action': [],
            'reward': [], 'position': [], 'nav': [],
        }
        return self._get_obs(), {}

    def step(self, action):
        current_price = self.df['close'].iloc[self.current_step]
        next_price = self.df['close'].iloc[self.current_step + 1]
        daily_return = (next_price - current_price) / current_price

        reward = 0.0

        if self.position == 0:
            if action == 1:
                # 空仓 -> 买入
                cost = current_price * (1 + self.commission)
                self.shares = int(self.cash / cost / 100) * 100
                if self.shares > 0:
                    self.cash -= self.shares * cost
                    self.entry_price = current_price
                    self.position = 1
                    reward = daily_return - self.commission
                else:
                    reward = 0.0
            else:
                # 空仓 + 持有/卖出 -> 无操作
                reward = 0.0
        else:
            if action == 2:
                # 持仓 -> 卖出
                sell_price = current_price * (1 - self.commission - self.stamp_tax)
                self.cash += self.shares * sell_price
                pnl = (current_price - self.entry_price) / self.entry_price
                reward = pnl - self.commission - self.stamp_tax
                self.shares = 0
                self.position = 0
                self.entry_price = 0.0
            else:
                # 持仓 + 持有/买入 -> 继续持有
                reward = daily_return

        # 计算当前净值
        nav = self.cash + self.shares * current_price

        # 记录轨迹
        self.history['step'].append(self.current_step)
        self.history['price'].append(current_price)
        self.history['action'].append(action)
        self.history['reward'].append(reward)
        self.history['position'].append(self.position)
        self.history['nav'].append(nav)

        self.current_step += 1
        terminated = self.current_step >= self.end_idx
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    def get_nav(self):
        """获取最终净值"""
        price = self.df['close'].iloc[self.current_step]
        return self.cash + self.shares * price


# ============================================================
# 随机 Agent 验证环境
# ============================================================

def run_random_agent(env, episodes=3):
    """用随机Agent测试环境是否正常工作"""
    print("=" * 60)
    print("随机 Agent 验证环境")
    print("=" * 60)

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        nav = env.get_nav()
        ret = (nav - env.initial_cash) / env.initial_cash * 100
        print(f"  Episode {ep+1}: {steps}步 | "
              f"累计奖励: {total_reward:.4f} | "
              f"最终净值: {nav:,.0f} | "
              f"收益率: {ret:+.2f}%")

    return env.history


# ============================================================
# 可视化
# ============================================================

def plot_episode(env, title="随机Agent交易轨迹"):
    """可视化一个 episode 的价格、动作和净值"""
    h = env.history
    if not h['step']:
        print("没有交易记录可绘制")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8),
                                    gridspec_kw={'height_ratios': [2, 1]})

    steps = h['step']
    prices = h['price']
    actions = h['action']
    navs = h['nav']

    # 上图: 价格 + 买卖点
    ax1.plot(steps, prices, 'gray', linewidth=1, alpha=0.8, label='收盘价')

    buy_steps = [s for s, a in zip(steps, actions) if a == 1]
    buy_prices = [p for p, a in zip(prices, actions) if a == 1]
    sell_steps = [s for s, a in zip(steps, actions) if a == 2]
    sell_prices = [p for p, a in zip(prices, actions) if a == 2]

    if buy_steps:
        ax1.scatter(buy_steps, buy_prices, color='#e74c3c', marker='^',
                    s=60, zorder=5, label=f'买入({len(buy_steps)}次)')
    if sell_steps:
        ax1.scatter(sell_steps, sell_prices, color='#2ecc71', marker='v',
                    s=60, zorder=5, label=f'卖出({len(sell_steps)}次)')

    ax1.set_ylabel('价格')
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 下图: 净值曲线
    nav_norm = [n / navs[0] for n in navs]
    price_norm = [p / prices[0] for p in prices]

    ax2.plot(steps, nav_norm, '#2980b9', linewidth=1.5, label='策略净值')
    ax2.plot(steps, price_norm, 'gray', linewidth=1, alpha=0.6, label='买入持有')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)
    ax2.set_ylabel('净值 (初始=1.0)')
    ax2.set_xlabel('交易日')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, '1-随机Agent交易轨迹.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  图表已保存: {save_path}")
    plt.close()


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("第18讲 脚本1: 搭建RL交易环境")
    print("Ethan 构建 Gymnasium 股票交易环境")
    print("=" * 60)

    # 加载数据: 上证50ETF
    STOCK_CODE = '510050.SH'
    START_DATE = '2022-01-01'
    END_DATE = '2026-04-09'

    print(f"\n[1] 加载数据: {STOCK_CODE}")
    df = load_stock_data(STOCK_CODE, START_DATE, END_DATE)
    print(f"  数据量: {len(df)} 个交易日")
    print(f"  区间: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  价格范围: {df['close'].min():.3f} ~ {df['close'].max():.3f}")

    # 创建环境
    print(f"\n[2] 创建 StockTradingEnv")
    env = StockTradingEnv(df, lookback=5, norm_window=252)
    print(f"  观测空间: {env.observation_space.shape}")
    print(f"  动作空间: {env.action_space.n} (0=持仓, 1=买入, 2=卖出)")
    print(f"  有效交易区间: 第{env.start_idx}天 ~ 第{env.end_idx}天 "
          f"(共{env.end_idx - env.start_idx}步)")

    # 验证环境: 检查 Gymnasium API 一致性
    print(f"\n[3] 验证环境")
    obs, info = env.reset()
    print(f"  初始观测 shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"  观测样例 (前8维): {obs[:8]}")

    obs, reward, terminated, truncated, info = env.step(0)
    print(f"  执行 hold 后: reward={reward:.6f}, terminated={terminated}")

    obs, reward, terminated, truncated, info = env.step(1)
    print(f"  执行 buy  后: reward={reward:.6f}, position={env.position}")

    obs, reward, terminated, truncated, info = env.step(2)
    print(f"  执行 sell 后: reward={reward:.6f}, position={env.position}")

    # 随机Agent完整测试
    print(f"\n[4] 随机Agent完整测试")
    env.reset()
    run_random_agent(env, episodes=3)

    # 可视化最后一个 episode 的轨迹
    print(f"\n[5] 可视化")
    env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    plot_episode(env)

    print(f"\n{'='*60}")
    print("环境构建完成! 下一步: 2-DQN择时策略.py")
    print(f"{'='*60}")
