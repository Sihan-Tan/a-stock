# -*- coding: utf-8 -*-
"""
第18讲 脚本2: DQN择时策略

Ethan 的第二步 -- 训练 DQN Agent 进行买卖择时

核心内容:
  1. 从零实现 DQN 核心组件:
     - Q网络: 3层MLP (64-64-3)
     - 经验回放池 ReplayBuffer (容量10000)
     - Epsilon-Greedy 探索策略 (1.0 -> 0.01 指数衰减)
     - 目标网络 (每100步软更新)
  2. 训练流程: 500 episodes, 每50 episode 打印进度
  3. 训练过程可视化 (奖励曲线、epsilon衰减)
  4. 保存训练好的模型权重
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from data_loader import load_stock_data

# 导入脚本1的环境
import importlib.util
_script1_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '1-搭建RL交易环境.py')
_spec = importlib.util.spec_from_file_location('env_module', _script1_path)
_env_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_env_module)
StockTradingEnv = _env_module.StockTradingEnv


# ============================================================
# Q 网络
# ============================================================

class QNetwork(nn.Module):
    """
    Q值估计网络

    结构: state_dim -> 64 -> 64 -> action_dim
    激活函数: ReLU
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 经验回放池
# ============================================================

class ReplayBuffer:
    """
    经验回放池

    存储 (state, action, reward, next_state, done) 五元组
    随机采样 mini-batch 用于训练，打破时间相关性
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# DQN Agent
# ============================================================

class DQNAgent:
    """
    DQN 择时 Agent

    关键超参数:
        gamma=0.99: 折扣因子，重视长期收益
        lr=1e-3: 学习率
        epsilon: 1.0 -> 0.01 指数衰减
        tau=0.005: 目标网络软更新系数
        batch_size=64: 训练 mini-batch 大小
        target_update_freq=100: 目标网络更新频率
    """
    def __init__(self, state_dim, action_dim,
                 gamma=0.99, lr=1e-3,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 tau=0.005, batch_size=64,
                 buffer_capacity=10000, target_update_freq=100,
                 device=None):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Q网络和目标网络
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_capacity)
        self.train_step = 0

        # 训练记录
        self.losses = []
        self.epsilons = []

    def select_action(self, state, training=True):
        """Epsilon-Greedy 动作选择"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        self.buffer.push(state, action, reward, next_state, done)

    def train(self):
        """从回放池采样并更新 Q 网络"""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # 当前 Q 值: Q(s, a)
        current_q = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # 目标 Q 值: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 软更新目标网络
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self._soft_update()

        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val

    def _soft_update(self):
        """目标网络软更新: target = tau * q_net + (1 - tau) * target"""
        for target_param, q_param in zip(
            self.target_net.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * q_param.data + (1 - self.tau) * target_param.data
            )

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilons.append(self.epsilon)

    def save(self, path):
        """保存模型"""
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
        }, path)
        print(f"  模型已保存: {path}")

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']
        print(f"  模型已加载: {path}")


# ============================================================
# 训练流程
# ============================================================

def train_dqn(env, agent, num_episodes=500, print_every=50):
    """
    DQN 训练主循环

    参数:
        env: StockTradingEnv 实例
        agent: DQNAgent 实例
        num_episodes: 训练轮次
        print_every: 每隔多少轮打印进度

    返回:
        episode_rewards: 每个 episode 的累计奖励
    """
    episode_rewards = []
    episode_navs = []
    best_reward = -float('inf')

    print(f"\n  开始训练 DQN (共 {num_episodes} episodes)")
    print(f"  设备: {agent.device}")
    print(f"  {'Episode':>8} | {'Reward':>10} | {'Epsilon':>8} | "
          f"{'NAV':>12} | {'Return':>8} | {'Loss':>10}")
    print(f"  {'-'*70}")

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)

            agent.store_transition(obs, action, reward, next_obs,
                                   float(terminated or truncated))
            agent.train()

            obs = next_obs
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        agent.decay_epsilon()
        nav = env.get_nav()
        ret = (nav - env.initial_cash) / env.initial_cash * 100

        episode_rewards.append(total_reward)
        episode_navs.append(nav)

        if total_reward > best_reward:
            best_reward = total_reward
            agent.save('models/dqn_best.pth')

        if ep % print_every == 0 or ep == 1:
            avg_loss = np.mean(agent.losses[-100:]) if agent.losses else 0
            print(f"  {ep:>8} | {total_reward:>+10.4f} | {agent.epsilon:>8.4f} | "
                  f"{nav:>12,.0f} | {ret:>+7.2f}% | {avg_loss:>10.6f}")

    return episode_rewards, episode_navs


# ============================================================
# 训练可视化
# ============================================================

def plot_training(episode_rewards, agent, title="DQN训练过程"):
    """可视化训练过程"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # 1. 奖励曲线
    ax = axes[0, 0]
    ax.plot(episode_rewards, alpha=0.3, color='#3498db', label='每轮奖励')
    window = min(50, len(episode_rewards) // 5) or 1
    if len(episode_rewards) >= window:
        ma = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(episode_rewards)), ma,
                color='#e74c3c', linewidth=2, label=f'{window}轮均值')
    ax.set_xlabel('Episode')
    ax.set_ylabel('累计奖励')
    ax.set_title('训练奖励曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Epsilon 衰减
    ax = axes[0, 1]
    ax.plot(agent.epsilons, color='#2ecc71', linewidth=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('探索率衰减')
    ax.grid(True, alpha=0.3)

    # 3. 损失曲线
    ax = axes[1, 0]
    if agent.losses:
        losses = agent.losses
        step = max(1, len(losses) // 500)
        sampled = losses[::step]
        ax.plot(sampled, alpha=0.3, color='#9b59b6')
        if len(sampled) >= 50:
            loss_ma = np.convolve(sampled, np.ones(50)/50, mode='valid')
            ax.plot(range(49, len(sampled)), loss_ma,
                    color='#e74c3c', linewidth=1.5)
    ax.set_xlabel('训练步数 (采样)')
    ax.set_ylabel('Loss')
    ax.set_title('训练损失')
    ax.grid(True, alpha=0.3)

    # 4. 奖励分布
    ax = axes[1, 1]
    ax.hist(episode_rewards, bins=30, color='#3498db', alpha=0.7, edgecolor='white')
    ax.axvline(x=np.mean(episode_rewards), color='#e74c3c',
               linestyle='--', label=f'均值: {np.mean(episode_rewards):.4f}')
    ax.set_xlabel('累计奖励')
    ax.set_ylabel('频次')
    ax.set_title('奖励分布')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/2-DQN训练过程.png', dpi=150, bbox_inches='tight')
    print("  图表已保存: outputs/2-DQN训练过程.png")
    plt.close()


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("第18讲 脚本2: DQN择时策略")
    print("Ethan 训练 DQN Agent 学习买卖时机")
    print("=" * 60)

    # 加载训练数据 (使用前几年作为训练集)
    STOCK_CODE = '510050.SH'
    TRAIN_START = '2022-01-01'
    TRAIN_END = '2025-06-30'

    print(f"\n[1] 加载训练数据: {STOCK_CODE}")
    df_train = load_stock_data(STOCK_CODE, TRAIN_START, TRAIN_END)
    print(f"  训练集: {len(df_train)} 个交易日")
    print(f"  区间: {df_train.index[0].strftime('%Y-%m-%d')} ~ "
          f"{df_train.index[-1].strftime('%Y-%m-%d')}")

    # 创建环境
    print(f"\n[2] 创建训练环境")
    env = StockTradingEnv(df_train, lookback=5, norm_window=252)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"  状态维度: {state_dim}")
    print(f"  动作维度: {action_dim}")

    # 创建 Agent
    print(f"\n[3] 创建 DQN Agent")
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=0.99,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        tau=0.005,
        batch_size=64,
        buffer_capacity=10000,
        target_update_freq=100,
    )
    total_params = sum(p.numel() for p in agent.q_net.parameters())
    print(f"  Q网络参数量: {total_params:,}")
    print(f"  设备: {agent.device}")

    # 训练
    print(f"\n[4] 训练 DQN Agent")
    episode_rewards, episode_navs = train_dqn(
        env, agent, num_episodes=500, print_every=50
    )

    # 保存最终模型
    agent.save('models/dqn_final.pth')

    # 训练统计
    print(f"\n[5] 训练统计")
    print(f"  总训练步数: {agent.train_step:,}")
    print(f"  最终 epsilon: {agent.epsilon:.4f}")
    print(f"  平均奖励 (后100轮): {np.mean(episode_rewards[-100:]):.4f}")
    print(f"  最佳奖励: {max(episode_rewards):.4f}")
    print(f"  最终净值 (后10轮均值): {np.mean(episode_navs[-10:]):,.0f}")

    # 可视化
    print(f"\n[6] 可视化训练过程")
    plot_training(episode_rewards, agent)

    print(f"\n{'='*60}")
    print("DQN训练完成! 下一步: 3-策略回测与评估.py")
    print(f"{'='*60}")
