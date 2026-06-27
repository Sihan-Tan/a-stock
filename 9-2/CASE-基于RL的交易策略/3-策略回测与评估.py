# -*- coding: utf-8 -*-
"""
第18讲 脚本3: 策略回测与评估

Ethan 的第三步 -- 在样本外数据上回测 DQN 策略

核心内容:
  1. 加载训练好的 DQN 模型
  2. 在样本外区间 (最近1年) 回测
  3. 输出关键绩效指标: 年化收益、夏普比率、最大回撤、胜率
  4. 与买入持有基准对比
  5. 绘制: 净值曲线、持仓信号叠加K线、交易统计
"""
import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from data_loader import load_stock_data

import importlib.util

def _load_module(name, filename):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_env_mod = _load_module('env_module', '1-搭建RL交易环境.py')
StockTradingEnv = _env_mod.StockTradingEnv

_dqn_mod = _load_module('dqn_module', '2-DQN择时策略.py')
DQNAgent = _dqn_mod.DQNAgent


# ============================================================
# 回测引擎
# ============================================================

def backtest(env, agent):
    """
    回测 DQN 策略

    返回:
        dict: 包含逐日净值、交易记录、绩效指标
    """
    obs, _ = env.reset()
    total_reward = 0
    trades = []
    daily_navs = []
    daily_actions = []

    while True:
        action = agent.select_action(obs, training=False)
        next_obs, reward, terminated, truncated, info = env.step(action)

        daily_navs.append(env.get_nav())
        daily_actions.append(action)
        total_reward += reward
        obs = next_obs

        if terminated or truncated:
            break

    # 从环境历史中提取交易信号
    h = env.history
    return {
        'navs': daily_navs,
        'actions': daily_actions,
        'prices': h['price'],
        'steps': h['step'],
        'total_reward': total_reward,
    }


# ============================================================
# 绩效指标计算
# ============================================================

def calc_metrics(navs, initial_cash):
    """
    计算完整绩效指标

    参数:
        navs: 逐日净值列表
        initial_cash: 初始资金

    返回:
        dict: 各项绩效指标
    """
    navs = np.array(navs)
    returns = np.diff(navs) / navs[:-1]

    total_return = (navs[-1] - initial_cash) / initial_cash
    trading_days = len(navs)
    years = trading_days / 252

    if years > 0 and total_return > -1:
        annual_return = (1 + total_return) ** (1 / years) - 1
    else:
        annual_return = total_return

    # 夏普比率 (无风险利率 2%)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = (np.mean(returns) - 0.02 / 252) / np.std(returns) * np.sqrt(252)
    else:
        sharpe = 0.0

    # 最大回撤
    peak = np.maximum.accumulate(navs)
    drawdown = (peak - navs) / peak
    max_drawdown = np.max(drawdown)
    max_dd_end = np.argmax(drawdown)
    max_dd_start = np.argmax(navs[:max_dd_end + 1]) if max_dd_end > 0 else 0

    # 卡玛比率
    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0

    # 胜率 (按日计算)
    win_days = np.sum(returns > 0)
    total_days = len(returns)
    daily_win_rate = win_days / total_days if total_days > 0 else 0

    # 盈亏比
    avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0
    avg_loss = np.mean(returns[returns < 0]) if np.any(returns < 0) else 0
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # 波动率
    annual_vol = np.std(returns) * np.sqrt(252)

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'max_dd_start': max_dd_start,
        'max_dd_end': max_dd_end,
        'calmar': calmar,
        'daily_win_rate': daily_win_rate,
        'profit_loss_ratio': profit_loss_ratio,
        'trading_days': trading_days,
        'final_nav': navs[-1],
    }


# ============================================================
# 可视化
# ============================================================

def plot_backtest_result(result, df_test, metrics_dqn, metrics_bh, stock_code):
    """
    绘制回测结果图表:
      上图: K线 + DQN持仓信号
      中图: 净值对比 (DQN vs 买入持有)
      下图: 回撤曲线
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12),
                                         gridspec_kw={'height_ratios': [3, 2, 1]})

    navs = np.array(result['navs'])
    prices = np.array(result['prices'])
    actions = np.array(result['actions'])
    n = len(prices)
    x = np.arange(n)

    # 上图: 价格 + 持仓区域 + 买卖点
    ax1.plot(x, prices, 'gray', linewidth=1, alpha=0.8, label='收盘价')

    # 标记持仓区间 (绿色背景)
    position = 0
    for i, a in enumerate(actions):
        if a == 1 and position == 0:
            start = i
            position = 1
        elif a == 2 and position == 1:
            ax1.axvspan(start, i, alpha=0.1, color='#2ecc71')
            position = 0

    buy_idx = [i for i, a in enumerate(actions) if a == 1]
    sell_idx = [i for i, a in enumerate(actions) if a == 2]

    if buy_idx:
        ax1.scatter(buy_idx, [prices[i] for i in buy_idx],
                    color='#e74c3c', marker='^', s=80, zorder=5,
                    label=f'买入({len(buy_idx)}次)')
    if sell_idx:
        ax1.scatter(sell_idx, [prices[i] for i in sell_idx],
                    color='#2ecc71', marker='v', s=80, zorder=5,
                    label=f'卖出({len(sell_idx)}次)')

    ax1.set_ylabel('价格')
    ax1.set_title(f'DQN择时策略 - {stock_code} 样本外回测', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 绩效信息框
    info_text = (
        f"--- DQN策略 ---\n"
        f"总收益:   {metrics_dqn['total_return']*100:+.2f}%\n"
        f"年化收益: {metrics_dqn['annual_return']*100:+.2f}%\n"
        f"最大回撤: {metrics_dqn['max_drawdown']*100:.2f}%\n"
        f"夏普比率: {metrics_dqn['sharpe']:.2f}\n"
        f"卡玛比率: {metrics_dqn['calmar']:.2f}\n"
        f"--- 买入持有 ---\n"
        f"总收益:   {metrics_bh['total_return']*100:+.2f}%\n"
        f"最大回撤: {metrics_bh['max_drawdown']*100:.2f}%\n"
        f"夏普比率: {metrics_bh['sharpe']:.2f}"
    )
    ax1.text(0.98, 0.97, info_text, transform=ax1.transAxes,
             fontsize=8, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8),
             family='monospace')

    # 中图: 净值对比
    nav_norm = navs / navs[0]
    bh_norm = prices / prices[0]

    ax2.plot(x, nav_norm, '#2980b9', linewidth=1.5, label='DQN策略')
    ax2.plot(x, bh_norm, 'gray', linewidth=1, alpha=0.6, label='买入持有')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)
    ax2.fill_between(x, nav_norm, bh_norm,
                     where=(nav_norm > bh_norm), alpha=0.1, color='#2ecc71')
    ax2.fill_between(x, nav_norm, bh_norm,
                     where=(nav_norm < bh_norm), alpha=0.1, color='#e74c3c')
    ax2.set_ylabel('净值 (初始=1.0)')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 下图: 回撤曲线
    peak = np.maximum.accumulate(navs)
    dd = (peak - navs) / peak * 100

    ax3.fill_between(x, dd, 0, color='#e74c3c', alpha=0.4)
    ax3.plot(x, dd, '#c0392b', linewidth=0.8)
    ax3.set_ylabel('回撤(%)')
    ax3.set_xlabel('交易日')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/3-DQN样本外回测.png', dpi=150, bbox_inches='tight')
    print("  图表已保存: outputs/3-DQN样本外回测.png")
    plt.close()


def plot_trade_analysis(result, df_test):
    """绘制交易统计分析"""
    actions = np.array(result['actions'])
    prices = np.array(result['prices'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. 动作分布
    ax = axes[0]
    action_counts = [np.sum(actions == i) for i in range(3)]
    labels = ['持仓', '买入', '卖出']
    colors = ['#95a5a6', '#e74c3c', '#2ecc71']
    ax.bar(labels, action_counts, color=colors, edgecolor='white')
    for i, v in enumerate(action_counts):
        ax.text(i, v + 1, str(v), ha='center', fontweight='bold')
    ax.set_title('动作分布')
    ax.set_ylabel('次数')

    # 2. 买卖价格分布
    ax = axes[1]
    buy_prices = [prices[i] for i, a in enumerate(actions) if a == 1]
    sell_prices = [prices[i] for i, a in enumerate(actions) if a == 2]
    if buy_prices:
        ax.hist(buy_prices, bins=15, alpha=0.6, color='#e74c3c', label='买入价', edgecolor='white')
    if sell_prices:
        ax.hist(sell_prices, bins=15, alpha=0.6, color='#2ecc71', label='卖出价', edgecolor='white')
    ax.set_title('买卖价格分布')
    ax.set_xlabel('价格')
    ax.legend()

    # 3. 持仓时长分布
    ax = axes[2]
    hold_durations = []
    in_position = False
    entry_idx = 0
    for i, a in enumerate(actions):
        if a == 1 and not in_position:
            entry_idx = i
            in_position = True
        elif a == 2 and in_position:
            hold_durations.append(i - entry_idx)
            in_position = False
    if hold_durations:
        ax.hist(hold_durations, bins=15, color='#3498db', alpha=0.7, edgecolor='white')
        ax.axvline(x=np.mean(hold_durations), color='#e74c3c', linestyle='--',
                   label=f'均值: {np.mean(hold_durations):.1f}天')
        ax.legend()
    ax.set_title('持仓时长分布')
    ax.set_xlabel('天数')

    plt.suptitle('DQN策略交易分析', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/3-DQN交易分析.png', dpi=150, bbox_inches='tight')
    print("  图表已保存: outputs/3-DQN交易分析.png")
    plt.close()


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("第18讲 脚本3: 策略回测与评估")
    print("Ethan 在样本外数据上回测 DQN 择时策略")
    print("=" * 60)

    STOCK_CODE = '510050.SH'
    TEST_START = '2025-07-01'
    TEST_END = '2026-04-09'
    MODEL_PATH = 'models/dqn_best.pth'

    # 加载测试数据
    # 注意: 环境需要 norm_window=252 的历史数据来计算 Z-score
    # 所以实际加载更早的数据，但只在测试区间内交易
    print(f"\n[1] 加载测试数据: {STOCK_CODE}")
    DATA_START = '2024-01-01'
    df_all = load_stock_data(STOCK_CODE, DATA_START, TEST_END)
    print(f"  数据量: {len(df_all)} 个交易日")
    print(f"  测试区间: {TEST_START} ~ {TEST_END}")

    # 创建测试环境
    print(f"\n[2] 创建测试环境")
    env = StockTradingEnv(df_all, lookback=5, norm_window=252)

    # 加载模型
    print(f"\n[3] 加载 DQN 模型")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)

    if os.path.exists(MODEL_PATH):
        agent.load(MODEL_PATH)
    else:
        print(f"  模型文件不存在: {MODEL_PATH}")
        print("  请先运行 2-DQN择时策略.py 进行训练")
        print("  使用未训练的模型进行演示...")

    # 回测
    print(f"\n[4] 执行回测")
    result = backtest(env, agent)
    print(f"  总交易步数: {len(result['navs'])}")
    print(f"  买入次数: {sum(1 for a in result['actions'] if a == 1)}")
    print(f"  卖出次数: {sum(1 for a in result['actions'] if a == 2)}")

    # 计算 DQN 策略绩效
    print(f"\n[5] 绩效评估")
    metrics_dqn = calc_metrics(result['navs'], env.initial_cash)
    print(f"\n  --- DQN 择时策略 ---")
    print(f"  总收益:     {metrics_dqn['total_return']*100:+.2f}%")
    print(f"  年化收益:   {metrics_dqn['annual_return']*100:+.2f}%")
    print(f"  年化波动:   {metrics_dqn['annual_vol']*100:.2f}%")
    print(f"  夏普比率:   {metrics_dqn['sharpe']:.4f}")
    print(f"  最大回撤:   {metrics_dqn['max_drawdown']*100:.2f}%")
    print(f"  卡玛比率:   {metrics_dqn['calmar']:.4f}")
    print(f"  日胜率:     {metrics_dqn['daily_win_rate']*100:.1f}%")
    print(f"  盈亏比:     {metrics_dqn['profit_loss_ratio']:.2f}")

    # 计算买入持有基准
    prices = np.array(result['prices'])
    bh_navs = env.initial_cash * prices / prices[0]
    metrics_bh = calc_metrics(bh_navs.tolist(), env.initial_cash)
    print(f"\n  --- 买入持有基准 ---")
    print(f"  总收益:     {metrics_bh['total_return']*100:+.2f}%")
    print(f"  年化收益:   {metrics_bh['annual_return']*100:+.2f}%")
    print(f"  最大回撤:   {metrics_bh['max_drawdown']*100:.2f}%")
    print(f"  夏普比率:   {metrics_bh['sharpe']:.4f}")

    # 超额收益
    excess = metrics_dqn['annual_return'] - metrics_bh['annual_return']
    print(f"\n  >>> 年化超额收益: {excess*100:+.2f}%")

    # 可视化
    print(f"\n[6] 可视化")
    df_test = df_all[df_all.index >= TEST_START]
    plot_backtest_result(result, df_test, metrics_dqn, metrics_bh, STOCK_CODE)
    plot_trade_analysis(result, df_test)

    print(f"\n{'='*60}")
    print("回测完成! 下一步: 4-智能拆单环境.py")
    print(f"{'='*60}")
