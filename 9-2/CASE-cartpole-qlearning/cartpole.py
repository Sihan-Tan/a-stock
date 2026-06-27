import os
import math
import argparse
import gymnasium as gym
from agent import Qlearning, Agent, Trainer

# 用于存放
RECORD_PATH = './upload'
print(RECORD_PATH)

def main(episodes, render, monitor):
    # 创建游戏环境
    # monitor 录制需要 rgb_array 模式，实时渲染用 human 模式
    if monitor:
        render_mode = "rgb_array"
    elif render:
        render_mode = "human"
    else:
        render_mode = None
    env = gym.make("CartPole-v1", render_mode=render_mode)
    # 创建Qlearning算法实例
    q = Qlearning(
        env.action_space.n, 
        env.observation_space, 
        # 连续值离散化，分成多少个bin
        bin_size=[5, 5, 8, 5],
        # cart位置、速度，pole角度，角速度
        # 传入None，会计算env实际的low和high
        low_bound=[None, -4, None, -math.radians(50)],  # 将角度转化为弧度
        high_bound=[None, 4, None, math.radians(50)]
        # low_bound=[None, -3, None, -1], 
        # high_bound=[None, 3, None, 1]
        )
    agent = Agent(q, epsilon=0.05)
    # 随之时间变化的函数，首先设置最大值，然后随着时间增加，learning_decay和epsilon_decay数值都在减少
    learning_decay = lambda lr, t: max(0.1, min(0.5, 1.0 - math.log10((t + 1) / 25)))
    epsilon_decay = lambda eps, t: max(0.01, min(1.0, 1.0 - math.log10((t + 1) / 25)))
    # 训练Agent
    trainer = Trainer(
        agent, 
        gamma=0.95,
        learning_rate=0.5, learning_rate_decay=learning_decay, 
        epsilon=1.0, epsilon_decay=epsilon_decay,
        max_step=250)
    print('monitor=', monitor)
    print('render=', render)
    print('episodes=', episodes)
    if monitor:
        # 将环境打包，记录 Agent 在环境中的表现
        os.makedirs(RECORD_PATH, exist_ok=True)
        env = gym.wrappers.RecordVideo(env, RECORD_PATH, disable_logger=True)
    # 训练episodes次
    trainer.train(env, episode_count=episodes, render=render)
    env.close()

if __name__ == "__main__":
    # 设置默认参数
    parser = argparse.ArgumentParser(description="train & run cartpole ")
    parser.add_argument("--episode", type=int, default=100, help="episode to train")
    parser.add_argument("--render", default=True, action="store_true", help="render the screen")
    # 是否将结果进行记录，保存为json
    parser.add_argument("--monitor", default=True, action="store_true", help="monitor")
    # 解析参数
    args = parser.parse_args()

    # 训练episode次
    main(args.episode, args.render, args.monitor)
