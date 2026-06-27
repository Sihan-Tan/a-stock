import random
import copy
from collections import defaultdict
from collections import deque
from collections import namedtuple
import numpy as np

# QLearning算法
class Qlearning():
    def __init__(self, n_actions, observation_space, bin_size, low_bound=None, high_bound=None, initial_mean=0.0, initial_std=0.0):
        self.n_actions = n_actions # Agent一共有多少action
        self._observation_dimension = 1

        for d in observation_space.shape:
            #print('d=', d) #d=4，因为state中一共记录了4个数值
            self._observation_dimension *= d
        # 传参过来的，人工定义这4个数值，想要分成多少个分箱
        self._bin_sizes = bin_size if isinstance(bin_size, list) else [bin_size] * self._observation_dimension
        #print('bin_sizes=', self._bin_sizes)
        self._dimension_bins = []
        for i, low, high in self._low_high_iter(observation_space, low_bound, high_bound):
            b_size = self._bin_sizes[i]
            # 基于low, high, b_size创建分箱 bins
            bins = self._make_bins(low, high, b_size)
            print('low={}, high={}, b_size={}'.format(low, high, b_size))
            print('bins=', bins)
            self._dimension_bins.append(bins)

        # if we encounter the new observation, we initialize action evaluations
        # 保存算法计算的 Q Table, 字典格式 key: state, value: Q Value
        self.table = defaultdict(lambda: initial_std * np.random.randn(self.n_actions) + initial_mean)
    
    @classmethod # 表示：对应的函数，不需要实例化
    def _make_bins(cls, low, high, bin_size):
        # 计算分箱
        print('low:{} high:{} span:{}'.format(low, high, (float(high) - float(low)) / (bin_size - 2)))
        bins = np.arange(low, high, (float(high) - float(low)) / (bin_size - 2))  # exclude both ends
        bins = np.sort(np.append(bins, [high]))
        # if min(bins) < 0 and 0 not in bins:
        #     bins = np.sort(np.append(bins, [0]))  # 0 centric bins
        return bins
    
    @classmethod
    def _low_high_iter(cls, observation_space, low_bound, high_bound):
        # 这里会基于env，更新lows, highs
        lows = observation_space.low
        highs = observation_space.high
        for i in range(len(lows)):
            low = lows[i]
            if low_bound is not None:
                _low_bound = low_bound if not isinstance(low_bound, list) else low_bound[i]
                low = low if _low_bound is None else max(low, _low_bound)
            
            high = highs[i]
            if high_bound is not None:
                _high_bound = high_bound if not isinstance(high_bound, list) else high_bound[i]
                high = high if _high_bound is None else min(high, _high_bound)
            
            yield i, low, high

    def observation_to_state(self, observation):
        state = 0
        # caution: bin_size over 10 will not work accurately
        unit = max(self._bin_sizes)
        for d, o in enumerate(observation.flatten()):
            # o在分箱中的位置
            temp=np.digitize(o, self._dimension_bins[d]) * pow(unit, d)
            #print('unit={}, d={}, pow={}, temp={}'.format(unit, d, pow(unit, d), temp))
            state = state + np.digitize(o, self._dimension_bins[d]) * pow(unit, d)  # bin_size numeral system
        #print('state=', state)
        return state

    # 基于env obs，查找QTable中的value
    def values(self, observation):
        # 通过env obs，进行state编码
        state = self.observation_to_state(observation)
        return self.table[state]

class Agent():
    # q即为Agent的算法
    # epsilon-greedy 贪心算法（95%的情况下 选择最大值，5%的情况是随机选择）
    def __init__(self, q, epsilon=0.05):
        self.q = q # q learning算法
        self.epsilon = epsilon # 随机探索的比例 E&E 利用&探索

    # 通过state，计算action（通过Q Table）
    def act(self, observation):
        action = -1
        if np.random.random() < self.epsilon: # 5%
            # 随机从action space中选择一个
            action = np.random.choice(self.q.n_actions)
        else: # 选择Q Value最大的Action
            # 通过Q Table进行计算，输入的是obs，输出的是 action（对应最大的Q值的action）
            action = np.argmax(self.q.values(observation))
        return action

# 用于Agent训练，使用时需要实例化
class Trainer():
    # 初始化
    def __init__(self, agent, gamma=0.95, learning_rate=0.1, learning_rate_decay=None, epsilon=0.05, epsilon_decay=None, max_step=-1):
        self.agent = agent
        self.gamma = gamma  # 贴现因子 （未来你期望能挣100元，相当于现在可以挣到 100*gamma元）
        self.learning_rate = learning_rate  
        self.learning_rate_decay = learning_rate_decay # 变化的lr，先大后小
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay # 变化的epsilon 探索比例，先大后小
        self.max_step = max_step

    def train(self, env, episode_count, render=False):
        default_epsilon = self.agent.epsilon
        self.agent.epsilon = self.epsilon
        values = []
        # 记录最近100次的结果
        steps = deque(maxlen=100)
        lr = self.learning_rate
        for i in range(episode_count):
            # 游戏初始化 (gymnasium 返回 obs, info)
            obs, _ = env.reset()
            # 记录一共坚持了多少step
            step = 0
            done = False
            while not done: # 如果没有结束，就一直step
                # gymnasium 通过 render_mode 控制渲染，不再需要手动调用 env.render()
                # Agent通过obs，进行action
                action = self.agent.act(obs)
                # Agent执行action，从环境中得到反馈 (gymnasium 返回 terminated, truncated)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                # 通过obs，计算state编码
                state = self.agent.q.observation_to_state(obs)
                # next_obs 期望获得的最大reward
                future = 0 if done else np.max(self.agent.q.values(next_obs))
                # 当前action，Q Table中的value
                value = self.agent.q.table[state][action]
                # 更新q table
                self.agent.q.table[state][action] += lr * (reward + self.gamma * future - value)

                obs = next_obs
                values.append(value)
                step += 1
                # 如果超出了max_step，也会结束
                if self.max_step > 0 and step > self.max_step:
                    done = True
            else:
                # 所以比赛结果的平均值
                mean = np.mean(values)
                # 用于保留最近100局的结果
                steps.append(step)
                # 最近100次的平均值
                mean_step = np.mean(steps)
                print("Episode {}: {}steps(最近的avg{}). epsilon={:.3f}, lr={:.3f}, 整体平均q_value={:.2f}".format(
                    i, step, mean_step, self.agent.epsilon, lr, mean)
                    )
                
                if self.epsilon_decay is not None:       
                    # 按照epsilon_decay函数，计算当前的epsilon
                    self.agent.epsilon = self.epsilon_decay(self.agent.epsilon, i)
                if self.learning_rate_decay is not None:
                    # 按照learning_rate_decay函数，计算当前的lr
                    lr = self.learning_rate_decay(lr, i)
