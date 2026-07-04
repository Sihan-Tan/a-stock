# -*- coding: utf-8 -*-
"""
CASE: 高频做市模拟

Ethan 的第六步 -- 模拟订单簿和做市Agent，理解高频做市的核心机制

核心内容:
  1. 构建简化的订单簿模拟器 (OrderBookSimulator):
     - 5档买卖盘
     - 订单到达遵循泊松过程
     - 支持限价单和市价单
  2. 做市 Agent 环境:
     - State: 买卖价差、订单不平衡(OFI)、库存水平、波动率
     - Action: 报价偏移量 (离散: 对称/偏买/偏卖)
     - Reward: 已实现价差收益 - 库存风险惩罚
  3. DQN 做市 Agent vs 固定价差做市基线
  4. 可视化: 库存变化、累计PnL、报价行为
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================
# 简化订单簿模拟器
# ============================================================

class OrderBookSimulator:
    """
    简化的限价订单簿模拟器

    模拟一个具有5档买卖盘的市场:
    - 基础价格围绕 mid_price 波动
    - 订单到达遵循泊松过程
    - 做市商可以在买卖两侧挂单
    - 市价单按价格优先成交
    """

    def __init__(self, mid_price=100.0, tick_size=0.01,
                 arrival_rate=10.0, volatility=0.001, seed=None):
        """
        参数:
            mid_price: 初始中间价
            tick_size: 最小价格变动单位
            arrival_rate: 每步订单到达率 (泊松参数)
            volatility: 每步价格波动率
            seed: 随机种子
        """
        self.mid_price = mid_price
        self.tick_size = tick_size
        self.arrival_rate = arrival_rate
        self.volatility = volatility
        self.rng = np.random.RandomState(seed)

        # 买卖盘 (price -> quantity)
        self.bids = {}  # 买盘
        self.asks = {}  # 卖盘

        # 做市商的挂单追踪
        self.mm_bids = {}  # 做市商的买单 {price: qty}
        self.mm_asks = {}  # 做市商的卖单 {price: qty}

        self._init_book()

    def _init_book(self):
        """初始化5档买卖盘"""
        self.bids = {}
        self.asks = {}
        for i in range(1, 6):
            bid_price = round(self.mid_price - i * self.tick_size, 4)
            ask_price = round(self.mid_price + i * self.tick_size, 4)
            self.bids[bid_price] = self.rng.randint(100, 500)
            self.asks[ask_price] = self.rng.randint(100, 500)

    def get_best_bid(self):
        """最优买价"""
        return max(self.bids.keys()) if self.bids else self.mid_price - self.tick_size

    def get_best_ask(self):
        """最优卖价"""
        return min(self.asks.keys()) if self.asks else self.mid_price + self.tick_size

    def get_spread(self):
        """买卖价差"""
        return self.get_best_ask() - self.get_best_bid()

    def get_order_imbalance(self):
        """
        订单流不平衡 (OFI)
        = (买盘总量 - 卖盘总量) / (买盘总量 + 卖盘总量)
        范围 [-1, 1], 正值表示买压大
        """
        bid_vol = sum(self.bids.values()) if self.bids else 0
        ask_vol = sum(self.asks.values()) if self.asks else 0
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0

    def step(self):
        """
        模拟一步市场演化:
        1. 随机游走更新中间价
        2. 随机到达的市价买/卖单消耗挂单
        3. 补充流动性

        返回:
            dict: 包含成交信息
        """
        # 价格随机游走
        price_change = self.rng.normal(0, self.volatility) * self.mid_price
        self.mid_price = round(self.mid_price + price_change, 4)

        fills = {'mm_bid_fills': [], 'mm_ask_fills': []}

        # 随机市价单到达
        n_orders = self.rng.poisson(self.arrival_rate)
        for _ in range(n_orders):
            is_buy = self.rng.random() < 0.5 + 0.1 * self.get_order_imbalance()
            size = self.rng.randint(50, 200)

            if is_buy and self.asks:
                # 市价买单 -> 吃掉卖盘
                best_ask = min(self.asks.keys())
                consumed = min(size, self.asks[best_ask])
                self.asks[best_ask] -= consumed
                if self.asks[best_ask] <= 0:
                    del self.asks[best_ask]

                # 检查是否成交了做市商的卖单
                if best_ask in self.mm_asks and self.mm_asks[best_ask] > 0:
                    mm_filled = min(consumed, self.mm_asks[best_ask])
                    self.mm_asks[best_ask] -= mm_filled
                    fills['mm_ask_fills'].append({
                        'price': best_ask, 'qty': mm_filled
                    })

            elif not is_buy and self.bids:
                # 市价卖单 -> 吃掉买盘
                best_bid = max(self.bids.keys())
                consumed = min(size, self.bids[best_bid])
                self.bids[best_bid] -= consumed
                if self.bids[best_bid] <= 0:
                    del self.bids[best_bid]

                # 检查是否成交了做市商的买单
                if best_bid in self.mm_bids and self.mm_bids[best_bid] > 0:
                    mm_filled = min(consumed, self.mm_bids[best_bid])
                    self.mm_bids[best_bid] -= mm_filled
                    fills['mm_bid_fills'].append({
                        'price': best_bid, 'qty': mm_filled
                    })

        # 补充流动性 (模拟其他参与者挂单)
        self._replenish_book()

        return fills

    def _replenish_book(self):
        """补充订单簿流动性"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        # 确保至少有3档
        for i in range(1, 6):
            bid_p = round(self.mid_price - i * self.tick_size, 4)
            ask_p = round(self.mid_price + i * self.tick_size, 4)

            if bid_p not in self.bids:
                self.bids[bid_p] = self.rng.randint(50, 300)
            if ask_p not in self.asks:
                self.asks[ask_p] = self.rng.randint(50, 300)

        # 清理过远的挂单
        if len(self.bids) > 10:
            sorted_bids = sorted(self.bids.keys(), reverse=True)
            for p in sorted_bids[10:]:
                del self.bids[p]
        if len(self.asks) > 10:
            sorted_asks = sorted(self.asks.keys())
            for p in sorted_asks[10:]:
                del self.asks[p]

    def place_mm_orders(self, bid_offset, ask_offset, qty=100):
        """
        做市商挂单

        参数:
            bid_offset: 买单偏移 (距mid_price的tick数)
            ask_offset: 卖单偏移 (距mid_price的tick数)
            qty: 挂单量
        """
        # 先撤掉旧挂单
        for p, q in self.mm_bids.items():
            if p in self.bids:
                self.bids[p] = max(0, self.bids[p] - q)
        for p, q in self.mm_asks.items():
            if p in self.asks:
                self.asks[p] = max(0, self.asks[p] - q)

        self.mm_bids = {}
        self.mm_asks = {}

        # 新挂单
        bid_price = round(self.mid_price - bid_offset * self.tick_size, 4)
        ask_price = round(self.mid_price + ask_offset * self.tick_size, 4)

        self.mm_bids[bid_price] = qty
        self.mm_asks[ask_price] = qty

        # 加入订单簿
        self.bids[bid_price] = self.bids.get(bid_price, 0) + qty
        self.asks[ask_price] = self.asks.get(ask_price, 0) + qty


# ============================================================
# 做市环境
# ============================================================

class MarketMakingEnv(gym.Env):
    """
    做市 Agent 环境

    做市商在每一步:
    1. 观察市场状态 (价差、OFI、库存、波动率)
    2. 决定报价策略 (对称报价 / 偏买 / 偏卖 / 宽价差 / 窄价差)
    3. 获得奖励 = 价差收益 - 库存风险惩罚

    动作空间 (离散5):
        0: 对称报价 (bid_offset=1, ask_offset=1)
        1: 偏向买入 (bid_offset=1, ask_offset=2) -- 想买更多
        2: 偏向卖出 (bid_offset=2, ask_offset=1) -- 想卖更多
        3: 宽价差   (bid_offset=2, ask_offset=2) -- 保守
        4: 窄价差   (bid_offset=1, ask_offset=1, qty更大) -- 激进
    """

    ACTION_MAP = {
        0: (1, 1, 100),   # 对称报价
        1: (1, 2, 100),   # 偏买
        2: (2, 1, 100),   # 偏卖
        3: (2, 2, 80),    # 宽价差(保守)
        4: (1, 1, 150),   # 窄价差(激进)
    }

    def __init__(self, mid_price=100.0, tick_size=0.01,
                 max_steps=1000, inventory_limit=1000,
                 inventory_penalty=0.01, seed=None):
        """
        参数:
            mid_price: 初始中间价
            tick_size: 最小变动
            max_steps: 每个 episode 步数
            inventory_limit: 库存上限(绝对值)
            inventory_penalty: 库存风险惩罚系数
        """
        super().__init__()
        self.initial_mid = mid_price
        self.tick_size = tick_size
        self.max_steps = max_steps
        self.inventory_limit = inventory_limit
        self.inventory_penalty = inventory_penalty
        self.seed_val = seed

        self.action_space = spaces.Discrete(5)
        # [价差(归一化), OFI, 库存(归一化), 波动率估计, PnL变化]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(5,), dtype=np.float32
        )

        self.book = None
        self.inventory = 0
        self.cash = 0.0
        self.current_step = 0
        self.prev_pnl = 0.0

        # 历史记录
        self.history = {
            'step': [], 'mid_price': [], 'spread': [],
            'inventory': [], 'pnl': [], 'action': [],
            'ofi': [],
        }

    def _get_obs(self):
        spread = self.book.get_spread() / self.book.mid_price * 10000  # bps
        ofi = self.book.get_order_imbalance()
        inv_norm = self.inventory / self.inventory_limit
        vol_est = self.book.volatility * 100
        pnl = self._calc_pnl()
        pnl_change = pnl - self.prev_pnl

        return np.array([spread, ofi, inv_norm, vol_est, pnl_change],
                        dtype=np.float32)

    def _calc_pnl(self):
        """计算未实现 + 已实现 PnL"""
        mark_to_market = self.inventory * self.book.mid_price
        return self.cash + mark_to_market

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        s = self.seed_val if seed is None else seed
        self.book = OrderBookSimulator(
            mid_price=self.initial_mid,
            tick_size=self.tick_size,
            seed=s,
        )
        self.inventory = 0
        self.cash = 0.0
        self.current_step = 0
        self.prev_pnl = 0.0
        self.history = {
            'step': [], 'mid_price': [], 'spread': [],
            'inventory': [], 'pnl': [], 'action': [],
            'ofi': [],
        }
        return self._get_obs(), {}

    def step(self, action):
        bid_offset, ask_offset, qty = self.ACTION_MAP[action]

        # 根据库存调整: 库存过多时倾向卖出
        if abs(self.inventory) > self.inventory_limit * 0.5:
            if self.inventory > 0:
                ask_offset = max(1, ask_offset - 1)
            else:
                bid_offset = max(1, bid_offset - 1)

        # 做市商挂单
        self.book.place_mm_orders(bid_offset, ask_offset, qty)

        # 市场演化
        fills = self.book.step()

        # 处理做市商成交
        spread_pnl = 0.0
        for fill in fills['mm_bid_fills']:
            self.inventory += fill['qty']
            self.cash -= fill['price'] * fill['qty']
            spread_pnl += (self.book.mid_price - fill['price']) * fill['qty']

        for fill in fills['mm_ask_fills']:
            self.inventory -= fill['qty']
            self.cash += fill['price'] * fill['qty']
            spread_pnl += (fill['price'] - self.book.mid_price) * fill['qty']

        # 奖励 = 价差收益 - 库存风险惩罚
        inv_risk = self.inventory_penalty * self.inventory ** 2 * self.book.volatility
        reward = spread_pnl - inv_risk

        pnl = self._calc_pnl()
        self.prev_pnl = pnl

        # 记录
        self.history['step'].append(self.current_step)
        self.history['mid_price'].append(self.book.mid_price)
        self.history['spread'].append(self.book.get_spread())
        self.history['inventory'].append(self.inventory)
        self.history['pnl'].append(pnl)
        self.history['action'].append(action)
        self.history['ofi'].append(self.book.get_order_imbalance())

        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        # 库存超限则终止
        if abs(self.inventory) > self.inventory_limit:
            reward -= 10.0  # 大额惩罚
            terminated = True

        return self._get_obs(), reward, terminated, False, {
            'pnl': pnl, 'inventory': self.inventory
        }


# ============================================================
# 做市 DQN Agent (复用脚本2的框架，简化版)
# ============================================================

class MMQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class MMDQNAgent:
    """做市专用的轻量 DQN Agent"""

    def __init__(self, state_dim, action_dim, lr=5e-4,
                 gamma=0.95, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay=0.998, buffer_size=5000, batch_size=32):

        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_net = MMQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = MMQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)
        self.train_step = 0

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_net(state_t).argmax(dim=1).item()

    def store(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def train(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s2, d = zip(*batch)

        s_t = torch.FloatTensor(np.array(s)).to(self.device)
        a_t = torch.LongTensor(a).to(self.device)
        r_t = torch.FloatTensor(r).to(self.device)
        s2_t = torch.FloatTensor(np.array(s2)).to(self.device)
        d_t = torch.FloatTensor(d).to(self.device)

        q = self.q_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target_net(s2_t).max(1)[0]
            target = r_t + self.gamma * q_next * (1 - d_t)

        loss = nn.MSELoss()(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step += 1
        if self.train_step % 50 == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# ============================================================
# 固定价差做市基线
# ============================================================

class FixedSpreadMM:
    """固定对称1-tick报价的基线做市商"""
    def select_action(self, obs, training=False):
        return 0  # 始终对称报价


# ============================================================
# 可视化
# ============================================================

def plot_mm_results(history_dqn, history_fixed, title="做市策略对比"):
    """对比 DQN 做市商 vs 固定价差做市商"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. 价格 + 库存 (DQN)
    ax = axes[0, 0]
    ax.plot(history_dqn['step'], history_dqn['mid_price'],
            'gray', linewidth=0.8, alpha=0.8)
    ax.set_ylabel('中间价', color='gray')
    ax.set_title('DQN做市商 - 价格与库存')
    ax2 = ax.twinx()
    ax2.plot(history_dqn['step'], history_dqn['inventory'],
             '#e74c3c', linewidth=0.8, alpha=0.8)
    ax2.set_ylabel('库存', color='#e74c3c')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.2)
    ax.grid(True, alpha=0.2)

    # 2. 价格 + 库存 (固定)
    ax = axes[0, 1]
    ax.plot(history_fixed['step'], history_fixed['mid_price'],
            'gray', linewidth=0.8, alpha=0.8)
    ax.set_ylabel('中间价', color='gray')
    ax.set_title('固定价差做市商 - 价格与库存')
    ax2 = ax.twinx()
    ax2.plot(history_fixed['step'], history_fixed['inventory'],
             '#3498db', linewidth=0.8, alpha=0.8)
    ax2.set_ylabel('库存', color='#3498db')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.2)
    ax.grid(True, alpha=0.2)

    # 3. 累计PnL对比
    ax = axes[1, 0]
    ax.plot(history_dqn['step'], history_dqn['pnl'],
            '#e74c3c', linewidth=1.5, label='DQN做市')
    ax.plot(history_fixed['step'], history_fixed['pnl'],
            '#3498db', linewidth=1.5, label='固定价差')
    ax.set_xlabel('步数')
    ax.set_ylabel('PnL')
    ax.set_title('累计PnL对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 动作分布 (DQN)
    ax = axes[1, 1]
    action_labels = ['对称', '偏买', '偏卖', '宽价差', '窄价差']
    actions = history_dqn['action']
    counts = [actions.count(i) for i in range(5)]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#95a5a6', '#9b59b6']
    ax.bar(action_labels, counts, color=colors, edgecolor='white')
    for i, v in enumerate(counts):
        ax.text(i, v + 2, str(v), ha='center', fontsize=9)
    ax.set_title('DQN做市商动作分布')
    ax.set_ylabel('次数')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/6-做市策略对比.png', dpi=150, bbox_inches='tight')
    print("  图表已保存: outputs/6-做市策略对比.png")
    plt.close()


def plot_mm_training(episode_pnls):
    """做市Agent训练曲线"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episode_pnls, alpha=0.3, color='#3498db')
    w = min(30, len(episode_pnls) // 5) or 1
    if len(episode_pnls) >= w:
        ma = np.convolve(episode_pnls, np.ones(w)/w, mode='valid')
        ax.plot(range(w-1, len(episode_pnls)), ma,
                color='#e74c3c', linewidth=2, label=f'{w}轮均值')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('最终PnL')
    ax.set_title('做市Agent训练PnL曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/6-做市训练曲线.png', dpi=150, bbox_inches='tight')
    print("  图表已保存: outputs/6-做市训练曲线.png")
    plt.close()


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("第18讲 脚本6: 高频做市模拟")
    print("Ethan 模拟订单簿，训练DQN做市Agent")
    print("=" * 60)

    MID_PRICE = 100.0
    TICK_SIZE = 0.01
    MAX_STEPS = 500
    NUM_EPISODES = 300

    # 创建做市环境
    print(f"\n[1] 创建做市环境")
    env = MarketMakingEnv(
        mid_price=MID_PRICE,
        tick_size=TICK_SIZE,
        max_steps=MAX_STEPS,
        inventory_limit=1000,
        inventory_penalty=0.01,
    )
    print(f"  中间价: {MID_PRICE}, tick: {TICK_SIZE}")
    print(f"  每episode: {MAX_STEPS}步")
    print(f"  观测空间: {env.observation_space.shape}")
    print(f"  动作空间: {env.action_space.n}")

    # 创建 DQN 做市Agent
    print(f"\n[2] 创建 DQN 做市 Agent")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = MMDQNAgent(state_dim, action_dim)
    print(f"  参数量: {sum(p.numel() for p in agent.q_net.parameters()):,}")

    # 训练
    print(f"\n[3] 训练做市 Agent ({NUM_EPISODES} episodes)")
    episode_pnls = []
    episode_rewards = []

    for ep in range(1, NUM_EPISODES + 1):
        obs, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.store(obs, action, reward, next_obs, float(terminated))
            agent.train()
            obs = next_obs
            total_reward += reward
            if terminated or truncated:
                break

        agent.decay_epsilon()
        pnl = env.history['pnl'][-1] if env.history['pnl'] else 0
        episode_pnls.append(pnl)
        episode_rewards.append(total_reward)

        if ep % 50 == 0 or ep == 1:
            avg_pnl = np.mean(episode_pnls[-50:])
            print(f"  Episode {ep:>4}: PnL={pnl:>+8.2f}, "
                  f"平均PnL(50)={avg_pnl:>+8.2f}, "
                  f"epsilon={agent.epsilon:.3f}")

    # 训练曲线
    print(f"\n[4] 训练可视化")
    plot_mm_training(episode_pnls)

    # 评估: DQN vs 固定价差
    print(f"\n[5] 评估: DQN做市 vs 固定价差做市")

    # DQN 做市
    env_eval = MarketMakingEnv(
        mid_price=MID_PRICE, tick_size=TICK_SIZE,
        max_steps=MAX_STEPS, seed=999
    )
    obs, _ = env_eval.reset()
    while True:
        action = agent.select_action(obs, training=False)
        obs, reward, terminated, truncated, info = env_eval.step(action)
        if terminated or truncated:
            break
    history_dqn = env_eval.history
    pnl_dqn = history_dqn['pnl'][-1]

    # 固定价差做市
    env_fixed = MarketMakingEnv(
        mid_price=MID_PRICE, tick_size=TICK_SIZE,
        max_steps=MAX_STEPS, seed=999
    )
    fixed_mm = FixedSpreadMM()
    obs, _ = env_fixed.reset()
    while True:
        action = fixed_mm.select_action(obs)
        obs, reward, terminated, truncated, info = env_fixed.step(action)
        if terminated or truncated:
            break
    history_fixed = env_fixed.history
    pnl_fixed = history_fixed['pnl'][-1]

    print(f"  DQN做市 最终PnL:    {pnl_dqn:+.2f}")
    print(f"  固定价差 最终PnL:    {pnl_fixed:+.2f}")
    print(f"  DQN最终库存:         {history_dqn['inventory'][-1]}")
    print(f"  固定价差最终库存:     {history_fixed['inventory'][-1]}")

    # 批量评估
    print(f"\n  批量评估 (各30次)...")
    dqn_pnls, fixed_pnls = [], []
    for i in range(30):
        # DQN
        env_t = MarketMakingEnv(mid_price=MID_PRICE, tick_size=TICK_SIZE,
                                max_steps=MAX_STEPS, seed=i*7)
        obs, _ = env_t.reset()
        while True:
            action = agent.select_action(obs, training=False)
            obs, r, d, t, _ = env_t.step(action)
            if d or t:
                break
        dqn_pnls.append(env_t.history['pnl'][-1])

        # 固定
        env_t2 = MarketMakingEnv(mid_price=MID_PRICE, tick_size=TICK_SIZE,
                                 max_steps=MAX_STEPS, seed=i*7)
        obs, _ = env_t2.reset()
        while True:
            action = fixed_mm.select_action(obs)
            obs, r, d, t, _ = env_t2.step(action)
            if d or t:
                break
        fixed_pnls.append(env_t2.history['pnl'][-1])

    print(f"  DQN做市  平均PnL: {np.mean(dqn_pnls):+.2f} "
          f"(std={np.std(dqn_pnls):.2f})")
    print(f"  固定价差 平均PnL: {np.mean(fixed_pnls):+.2f} "
          f"(std={np.std(fixed_pnls):.2f})")

    # 对比可视化
    print(f"\n[6] 可视化对比")
    plot_mm_results(history_dqn, history_fixed)

    print(f"\n{'='*60}")
    print("做市模拟完成! 下一步: 7-主力行为识别.py")
    print(f"{'='*60}")
