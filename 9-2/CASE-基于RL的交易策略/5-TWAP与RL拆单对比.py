# -*- coding: utf-8 -*-
"""
第18讲 脚本5: TWAP与RL拆单对比

Ethan 的第五步 -- 用 PPO 训练智能拆单 Agent，对比基线策略

核心内容:
  1. 用 Stable-Baselines3 的 PPO 训练拆单 Agent
  2. 三种策略对比:
     - TWAP: 均匀拆单
     - VWAP: 按量拆单
     - PPO Agent: 自适应拆单
  3. 100次模拟统计对比
  4. 可视化: 执行轨迹对比、成本分布箱线图、不同市场状态下的表现
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from data_loader import load_stock_data

import importlib.util
_exec_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '4-智能拆单环境.py')
_spec = importlib.util.spec_from_file_location('exec_module', _exec_path)
_exec_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_exec_module)
OrderExecutionEnv = _exec_module.OrderExecutionEnv
TWAPStrategy = _exec_module.TWAPStrategy
VWAPStrategy = _exec_module.VWAPStrategy
MarketImpactModel = _exec_module.MarketImpactModel


# ============================================================
# 训练回调 - 打印进度
# ============================================================

class TrainingCallback(BaseCallback):
    """每隔一定步数打印训练进度"""
    def __init__(self, print_freq=5000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.episode_rewards = []

    def _on_step(self):
        if self.n_calls % self.print_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                print(f"    Step {self.n_calls:>6}: 平均奖励 = {mean_reward:.4f}")
        return True


# ============================================================
# PPO 训练
# ============================================================

def train_ppo(env_fn, total_timesteps=50000, save_path='models/ppo_execution.zip'):
    """
    训练 PPO 拆单 Agent

    参数:
        env_fn: 环境创建函数
        total_timesteps: 训练总步数
        save_path: 模型保存路径

    返回:
        训练好的 PPO 模型
    """
    vec_env = DummyVecEnv([env_fn])

    model = PPO(
        'MlpPolicy',
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64])
        ),
    )

    callback = TrainingCallback(print_freq=5000)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(save_path)
    print(f"  PPO 模型已保存: {save_path}")

    vec_env.close()
    return model


# ============================================================
# 策略评估
# ============================================================

def evaluate_ppo(env, model, n_episodes=100):
    """评估 PPO 策略"""
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        summary = env.get_execution_summary()
        if summary:
            results.append(summary)
    return results


def evaluate_baseline(env, strategy_class, total_order, num_steps, n_episodes=100):
    """评估基线策略"""
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

def plot_execution_comparison(env, model, total_order, num_steps):
    """
    在同一天上对比三种策略的执行轨迹
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    strategies = {
        'TWAP': lambda: TWAPStrategy(total_order, num_steps),
        'VWAP': lambda: VWAPStrategy(total_order, num_steps),
        'PPO': None,
    }
    colors = {'TWAP': '#3498db', 'VWAP': '#e67e22', 'PPO': '#e74c3c'}

    # 固定同一天进行对比
    env._day_idx = 42

    for idx, (name, strategy_fn) in enumerate(strategies.items()):
        env._day_idx = 42
        obs, _ = env.reset()

        while True:
            if name == 'PPO':
                action, _ = model.predict(obs, deterministic=True)
            else:
                strategy = strategy_fn()
                # 重新初始化环境和策略
                env._day_idx = 42
                obs, _ = env.reset()
                while True:
                    action = strategy.get_action(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                break
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        h = env.history
        color = colors[name]

        # 上行: 价格 + 成交量
        ax = axes[0, idx]
        ax.plot(h['step'], h['price'], 'gray', linewidth=1, alpha=0.8)
        exec_steps = [s for s, q in zip(h['step'], h['exec_qty']) if q > 0]
        exec_prices = [p for p, q in zip(h['exec_price'], h['exec_qty']) if q > 0]
        if exec_steps:
            ax.scatter(exec_steps, exec_prices, color=color, s=30, zorder=5)
        ax.axhline(y=env.arrival_price, color='green', linestyle='--', alpha=0.5)
        ax.set_title(f'{name} 执行轨迹', fontweight='bold')
        ax.set_ylabel('价格')
        ax.grid(True, alpha=0.3)

        summary = env.get_execution_summary()
        if summary:
            ax.text(0.02, 0.98,
                    f"IS: {summary['implementation_shortfall']*10000:.1f}bps",
                    transform=ax.transAxes, va='top',
                    fontsize=10, fontweight='bold', color=color)

        # 下行: 执行量分布
        ax = axes[1, idx]
        ax.bar(h['step'], h['exec_qty'], color=color, alpha=0.7, width=0.8)
        ax.set_title(f'{name} 执行量分布')
        ax.set_xlabel('时段')
        ax.set_ylabel('执行量')
        ax.grid(True, alpha=0.3)

    plt.suptitle('三种拆单策略执行轨迹对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/5-拆单策略执行对比.png', dpi=150, bbox_inches='tight')
    print("  图表已保存: outputs/5-拆单策略执行对比.png")
    plt.close()


def plot_cost_comparison(twap_results, vwap_results, ppo_results):
    """
    箱线图对比三种策略的执行成本
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 执行缺口 (IS)
    ax = axes[0]
    twap_is = [r['implementation_shortfall'] * 10000 for r in twap_results]
    vwap_is = [r['implementation_shortfall'] * 10000 for r in vwap_results]
    ppo_is = [r['implementation_shortfall'] * 10000 for r in ppo_results]

    data = [twap_is, vwap_is, ppo_is]
    bp = ax.boxplot(data, labels=['TWAP', 'VWAP', 'PPO'],
                    patch_artist=True, widths=0.5)
    colors_box = ['#3498db', '#e67e22', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # 标注均值
    means = [np.mean(d) for d in data]
    for i, m in enumerate(means):
        ax.text(i + 1, m, f'{m:.1f}', ha='center', va='bottom',
                fontweight='bold', fontsize=10)

    ax.set_ylabel('执行缺口 (bps)')
    ax.set_title('执行缺口分布对比')
    ax.grid(True, alpha=0.3, axis='y')

    # VWAP滑点
    ax = axes[1]
    twap_slip = [r['vwap_slippage'] * 10000 for r in twap_results]
    vwap_slip = [r['vwap_slippage'] * 10000 for r in vwap_results]
    ppo_slip = [r['vwap_slippage'] * 10000 for r in ppo_results]

    data = [twap_slip, vwap_slip, ppo_slip]
    bp = ax.boxplot(data, labels=['TWAP', 'VWAP', 'PPO'],
                    patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    means = [np.mean(d) for d in data]
    for i, m in enumerate(means):
        ax.text(i + 1, m, f'{m:.1f}', ha='center', va='bottom',
                fontweight='bold', fontsize=10)

    ax.set_ylabel('VWAP滑点 (bps)')
    ax.set_title('VWAP滑点分布对比')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('TWAP vs VWAP vs PPO -- 执行成本对比 (100次模拟)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/5-执行成本箱线图.png', dpi=150, bbox_inches='tight')
    print("  图表已保存: outputs/5-执行成本箱线图.png")
    plt.close()


def plot_summary_table(twap_results, vwap_results, ppo_results):
    """绘制策略对比汇总表"""
    def summarize(results, name):
        is_vals = [r['implementation_shortfall'] * 10000 for r in results]
        slip_vals = [r['vwap_slippage'] * 10000 for r in results]
        return {
            'name': name,
            'is_mean': np.mean(is_vals),
            'is_std': np.std(is_vals),
            'slip_mean': np.mean(slip_vals),
            'slip_std': np.std(slip_vals),
            'avg_slices': np.mean([r['num_slices'] for r in results]),
        }

    summaries = [
        summarize(twap_results, 'TWAP'),
        summarize(vwap_results, 'VWAP'),
        summarize(ppo_results, 'PPO'),
    ]

    print(f"\n  {'策略':<8} | {'IS均值(bps)':>12} | {'IS标准差':>10} | "
          f"{'VWAP滑点均值':>14} | {'分片数':>8}")
    print(f"  {'-'*65}")
    for s in summaries:
        print(f"  {s['name']:<8} | {s['is_mean']:>12.2f} | {s['is_std']:>10.2f} | "
              f"{s['slip_mean']:>14.2f} | {s['avg_slices']:>8.1f}")

    # PPO 相对改善
    ppo = summaries[2]
    twap = summaries[0]
    if twap['is_mean'] != 0:
        improvement = (twap['is_mean'] - ppo['is_mean']) / abs(twap['is_mean']) * 100
        print(f"\n  PPO 相对 TWAP 执行缺口改善: {improvement:+.1f}%")


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("第18讲 脚本5: TWAP与RL拆单对比")
    print("Ethan 用 PPO 训练智能拆单 Agent")
    print("=" * 60)

    STOCK_CODE = '510050.SH'
    START_DATE = '2023-01-01'
    END_DATE = '2026-04-09'
    TOTAL_ORDER = 100_000
    NUM_STEPS = 48
    MODEL_PATH = 'models/ppo_execution.zip'

    # 加载数据
    print(f"\n[1] 加载数据: {STOCK_CODE}")
    df = load_stock_data(STOCK_CODE, START_DATE, END_DATE)
    daily_data_list = [df.iloc[i] for i in range(len(df))]
    adv = df['volume'].mean()
    print(f"  数据量: {len(df)} 个交易日, ADV: {adv:,.0f}")

    # 创建环境工厂
    def make_env():
        return OrderExecutionEnv(
            total_order=TOTAL_ORDER,
            daily_data_list=daily_data_list,
            adv=adv,
            num_steps=NUM_STEPS,
        )

    # 训练 PPO
    print(f"\n[2] 训练 PPO 拆单 Agent")
    if os.path.exists(MODEL_PATH):
        print(f"  加载已有模型: {MODEL_PATH}")
        model = PPO.load(MODEL_PATH)
    else:
        print(f"  开始训练 (50000步)...")
        model = train_ppo(make_env, total_timesteps=50000, save_path=MODEL_PATH)

    # 评估三种策略
    print(f"\n[3] 评估三种策略 (各100次模拟)")
    eval_env = make_env()

    print(f"  评估 TWAP...")
    twap_results = evaluate_baseline(eval_env, TWAPStrategy, TOTAL_ORDER, NUM_STEPS, 100)

    print(f"  评估 VWAP...")
    vwap_results = evaluate_baseline(eval_env, VWAPStrategy, TOTAL_ORDER, NUM_STEPS, 100)

    print(f"  评估 PPO...")
    ppo_results = evaluate_ppo(eval_env, model, 100)

    # 打印汇总
    print(f"\n[4] 策略对比汇总")
    plot_summary_table(twap_results, vwap_results, ppo_results)

    # 可视化
    print(f"\n[5] 可视化")
    plot_execution_comparison(eval_env, model, TOTAL_ORDER, NUM_STEPS)
    plot_cost_comparison(twap_results, vwap_results, ppo_results)

    print(f"\n{'='*60}")
    print("拆单对比完成! 下一步: 6-高频做市模拟.py")
    print(f"{'='*60}")
