# -*- coding: utf-8 -*-
"""
策略进化引擎 —— 遗传算法（Genetic Algorithm）驱动的量化策略参数优化器

本模块受到 MASTER 论文（Li et al., AAAI 2024）中"策略进化"理念的启发，
但实现的是经典的遗传算法（Genetic Algorithm, GA）而非论文中的强化学习进化框架。

核心思想：
  将交易策略的参数视为"基因"，通过"物竞天择，适者生存"的进化机制，
  在参数空间中自动搜索最优解。与传统网格搜索相比，遗传算法有如下优势：
    1. 高维搜索：网格搜索在 5 个以上参数时维度灾难严重，GA 不受此限制
    2. 全局优化：GA 的变异机制帮助跳出局部最优
    3. 多目标优化：通过 Pareto 前沿可以同时优化多个冲突目标（如收益 vs 回撤）

遗传算法流程：
  初始化种群 -> 评估适应度 -> 选择(锦标赛) -> 交叉 -> 变异 -> 精英保留 -> 下一代

本模块与 MASTER 论文的关联：
  MASTER 论文提出用强化学习（RL）来进化交易策略，本模块提供了一个更轻量的替代方案
  ——遗传算法。两者都是"策略进化"思想的体现，但 GA 更简单、更易落地。

核心组件:
  1. Individual          - 策略参数个体（基因编码）
  2. Population          - 种群管理
  3. evolve()            - 遗传算法进化主循环
  4. pareto_front()      - Pareto 前沿计算（多目标优化）
  5. run_backtest_fitness() - 适应度评估（包装 Backtrader 回测）
"""
import numpy as np
import random
import copy
import backtrader as bt
from db_config import INITIAL_CASH, COMMISSION


# ============================================================
# 个体: 策略参数的基因编码
# ============================================================

class Individual:
    """
    策略参数个体 —— 遗传算法中的"染色体"

    在遗传算法中，一个个体代表参数空间中的一个候选解。
    例如，一个移动平均线交叉策略的个体可能包含如下基因：
      {fast_ma: 5, slow_ma: 20, stop_loss: 0.05}

    属性:
        genes: dict, 参数名 -> 参数值的映射（基因型）
        fitness: dict, 适应度指标（表型表现），如 {'sharpe': 1.5, 'max_dd': 0.15}
        param_space: dict, 参数搜索空间定义（哪些参数可以优化以及它们的范围）

    参数空间的格式:
        {param_name: (min_value, max_value, type)}
        其中 type 可以是 'int'（整数参数）或 'float'（浮点数参数）
        示例: {'ma_period': (5, 60, 'int'), 'stop_loss': (0.01, 0.1, 'float')}
    """

    def __init__(self, param_space, genes=None):
        """
        初始化个体

        参数:
            param_space: 参数搜索空间定义
            genes: 如果提供，使用指定的基因值创建个体；否则随机初始化
        """
        self.param_space = param_space
        if genes is not None:
            self.genes = genes.copy()      # 深拷贝避免引用共享
        else:
            self.genes = self._random_init()  # 随机生成基因型
        self.fitness = {}                  # 适应度初始为空，评估后填充

    def _random_init(self):
        """
        随机初始化基因 —— 在参数空间内随机采样

        这是遗传算法"初始化种群"步骤的核心操作。
        随机初始化保证了初始种群的多样性——种群的多样性直接影响搜索质量。
        """
        genes = {}
        for name, (lo, hi, ptype) in self.param_space.items():
            if ptype == 'int':
                # 整数参数：在 [lo, hi] 区间内均匀采样整数
                genes[name] = random.randint(int(lo), int(hi))
            elif ptype == 'float':
                # 浮点数参数：在 [lo, hi] 区间内均匀采样浮点数
                genes[name] = random.uniform(lo, hi)
            else:
                # 未指定类型默认按浮点数处理
                genes[name] = random.uniform(lo, hi)
        return genes

    def mutate(self, mutation_rate=0.2, mutation_strength=0.3):
        """
        基因变异 —— 引入随机扰动以维持种群多样性

        变异是遗传算法中"探索"（exploration）的核心机制：
        - 如果只有交叉没有变异，算法会快速收敛但容易陷入局部最优
        - 变异以一定概率扰动基因值，帮助跳出局部最优
        - 变异率太高则搜索退化为随机搜索，需要平衡探索与利用

        参数:
            mutation_rate: 每个基因发生变异的概率
                典型值 0.1~0.3。太大则优秀基因被破坏，太小则易早熟收敛
            mutation_strength: 变异幅度（相对于搜索范围的比例）
                如 range=100, strength=0.3, 则变异最大偏移 30
        """
        for name, (lo, hi, ptype) in self.param_space.items():
            if random.random() < mutation_rate:
                range_size = hi - lo
                # 使用高斯分布（正态分布）产生变异偏移
                # 均值 0 表示偏移方向随机，标准差控制偏移幅度
                # 大多数变异幅度较小（靠近原值），少数变异幅度较大
                delta = random.gauss(0, mutation_strength * range_size)
                new_val = self.genes[name] + delta
                new_val = max(lo, min(hi, new_val))  # 边界截断，保证不越界
                if ptype == 'int':
                    new_val = int(round(new_val))    # 浮点数转整数
                self.genes[name] = new_val

    def __repr__(self):
        """字符串表示，便于调试和打印"""
        fitness_str = ', '.join(f'{k}={v:.4f}' for k, v in self.fitness.items())
        params_str = ', '.join(f'{k}={v}' for k, v in self.genes.items())
        return f"Individual({params_str} | {fitness_str})"


def crossover(parent1, parent2):
    """
    均匀交叉（Uniform Crossover）：每对父母生育两个子代

    交叉是遗传算法中"利用"（exploitation）的核心机制：
    它从两个优秀的父代个体中组合出新个体，期望子代能继承双方的优点。

    均匀交叉 vs 单点交叉：
    - 单点交叉：随机选一个分割点，前后段互换（适合有结构关联的参数）
    - 均匀交叉：每位基因独立随机选择父方或母方（适合无结构关联的参数）
    - 本模块使用均匀交叉，因为策略参数之间通常没有位置关联

    参数:
        parent1, parent2: 两个父代 Individual 对象

    返回:
        (child1, child2) 两个子代 Individual 对象
    """
    child1_genes = {}
    child2_genes = {}
    for name in parent1.param_space:
        # 每个基因位以 50% 概率交换
        # child1 和 child2 在同一个基因位上取不同的父方——保证两个子代不同
        if random.random() < 0.5:
            child1_genes[name] = parent1.genes[name]
            child2_genes[name] = parent2.genes[name]
        else:
            child1_genes[name] = parent2.genes[name]
            child2_genes[name] = parent1.genes[name]

    return (Individual(parent1.param_space, child1_genes),
            Individual(parent1.param_space, child2_genes))


def tournament_select(population, k=3):
    """
    锦标赛选择（Tournament Selection）

    选择策略对比：
    - 轮盘赌：按适应度比例选择，适应度高的被选概率高（易早熟）
    - 锦标赛：随机选 k 个，取其中最好的（平衡选择压力和多样性）
    - 排名选择：按排名分配概率，不受适应度绝对数值影响

    锦标赛选择的优点：
    1. 选择压力可通过 k 调节：k 越大，选择压力越大，收敛越快
    2. 计算简单，不需要全局排序
    3. 适应度尺度不敏感（避免了某些个体适应度极高时主导种群的"超级个体"问题）

    参数:
        population: 个体列表
        k: 锦标赛规模，即每次随机抽取多少个体进行比赛

    返回:
        选中的最优个体
    """
    # 从种群中随机抽取 k 个个体（不放回抽样）
    candidates = random.sample(population, min(k, len(population)))
    # 返回其中适应度最高（夏普比率最大）的个体
    # 使用 get('sharpe', -999) 确保即使 fitness 为空也会返回合理的默认值
    return max(candidates, key=lambda ind: ind.fitness.get('sharpe', -999))


# ============================================================
# 种群与进化
# ============================================================

class Population:
    """
    种群管理 —— 遗传算法的"进化容器"

    种群是所有个体的集合，进化过程就是在种群层面迭代优化。
    每一代进化包含五个步骤：精英保留 -> 选择 -> 交叉 -> 变异 -> 评估。
    """

    def __init__(self, param_space, size=30):
        """
        初始化种群

        参数:
            param_space: 参数搜索空间
            size: 种群大小（个体数量），典型值 30~100
                - 太小：快速收敛但易陷入局部最优
                - 太大：搜索充分但每代计算量巨大
        """
        self.param_space = param_space
        # 创建 size 个随机初始化的个体
        self.individuals = [Individual(param_space) for _ in range(size)]
        self.generation = 0      # 当前进化代数
        self.history = []        # 进化历史记录（追踪每代最优/平均适应度）

    def evaluate(self, fitness_fn):
        """
        评估种群中所有个体的适应度

        适应度评估是整个遗传算法中最耗时的步骤（通常占 95% 以上时间），
        因为需要为每个个体运行一次完整的回测。

        参数:
            fitness_fn: 适应度函数，输入 genes dict，输出 fitness dict
                如: lambda genes: {'sharpe': 1.5, 'max_dd': 0.1}
        """
        for ind in self.individuals:
            if not ind.fitness:
                ind.fitness = fitness_fn(ind.genes)

    def evolve_one_generation(self, fitness_fn, elite_count=3,
                               mutation_rate=0.2, mutation_strength=0.3):
        """
        进化一代 —— 完整的"生老病死"循环

        进化步骤详解：
        1. 评估（Evaluate）：计算当前代所有个体的适应度
        2. 排序（Sort）：按夏普比率降序排列
        3. 精英保留（Elite Preservation）：将前 N 个最优个体直接复制到下一代
           - 为什么要保留精英？保证最优解不会在交叉/变异中丢失（理论保证收敛性）
        4. 选择-交叉-变异（Selection-Crossover-Mutation）：
           - 锦标赛选择两个父代 -> 均匀交叉产生两个子代 -> 子代变异
           - 重复这个过程直到子代数量填满种群
        5. 评估新个体：新生成的个体（非精英）需要计算适应度

        参数:
            fitness_fn: 适应度函数
            elite_count: 每代保留的精英数量
            mutation_rate: 变异概率
            mutation_strength: 变异幅度
        """
        # 第一步：评估适应度（如果尚未评估）
        self.evaluate(fitness_fn)

        # 第二步：按夏普比率降序排序
        sorted_pop = sorted(self.individuals,
                            key=lambda ind: ind.fitness.get('sharpe', -999),
                            reverse=True)

        # 记录当前代的最佳和平均夏普比率
        best = sorted_pop[0]
        avg_sharpe = np.mean([ind.fitness.get('sharpe', 0) for ind in self.individuals])
        self.history.append({
            'generation': self.generation,
            'best_sharpe': best.fitness.get('sharpe', 0),
            'avg_sharpe': avg_sharpe,
            'best_genes': best.genes.copy(),  # 保存最优个体的基因，便于后续分析
        })

        # 第三步：精英保留 —— 直接复制前 elite_count 个个体到下一代
        # 注意：必须 deepcopy，否则后面的交叉/变异会修改精英个体的基因
        elites = [copy.deepcopy(ind) for ind in sorted_pop[:elite_count]]

        # 第四步：用选择-交叉-变异机制生成剩余的子代
        new_pop = list(elites)
        target_size = len(self.individuals)

        while len(new_pop) < target_size:
            p1 = tournament_select(sorted_pop)
            p2 = tournament_select(sorted_pop)
            c1, c2 = crossover(p1, p2)
            c1.mutate(mutation_rate, mutation_strength)
            c2.mutate(mutation_rate, mutation_strength)
            new_pop.extend([c1, c2])

        self.individuals = new_pop[:target_size]

        # 第五步：为新个体（非精英）评估适应度
        # 精英个体的 fitness 已被保留，无需重新评估（节省计算量）
        for ind in self.individuals[elite_count:]:
            ind.fitness = fitness_fn(ind.genes)

        self.generation += 1

    def best(self):
        """返回当前种群中的最优个体（夏普比率最高）"""
        return max(self.individuals,
                   key=lambda ind: ind.fitness.get('sharpe', -999))


def evolve(param_space, fitness_fn, pop_size=30, generations=50,
           elite_count=3, mutation_rate=0.2, verbose=True):
    """
    遗传算法进化主循环 —— "一键启动"遗传算法优化

    这是一个便捷的顶层函数，封装了 Population 管理的细节。
    调用方只需要提供参数空间和适应度函数，剩下的事情都由 evolve 处理。

    参数:
        param_space: dict, 参数搜索空间
            {参数名: (最小值, 最大值, 类型)}
            类型为 'int' 或 'float'
        fitness_fn: callable, 适应度函数
            输入: genes dict, 如 {'ma_period': 20, 'stop_loss': 0.05}
            输出: fitness dict, 如 {'sharpe': 1.5, 'annual_return': 0.15}
        pop_size: 种群大小，默认 30
        generations: 进化代数，默认 50
        elite_count: 精英保留数量，默认 3
        mutation_rate: 变异概率，默认 0.2
        verbose: 是否打印进化进度

    返回:
        Population 对象（包含进化历史记录和最优个体）

    使用示例:
        >> def fitness(genes):
        >>     # 运行回测，返回绩效指标
        >>     return run_backtest_fitness(MyStrategy, genes, df)
        >> pop = evolve(param_space, fitness, pop_size=30, generations=50)
        >> best_params = pop.best().genes
    """
    pop = Population(param_space, pop_size)

    for gen in range(generations):
        pop.evolve_one_generation(fitness_fn, elite_count, mutation_rate)
        if verbose:
            best = pop.best()
            print(f"  第{gen+1}代 | 最优夏普: {best.fitness.get('sharpe', 0):.4f} | "
                  f"平均夏普: {pop.history[-1]['avg_sharpe']:.4f}")

    return pop


# ============================================================
# Pareto 前沿 (多目标优化)
# ============================================================

def dominates(ind1, ind2, objectives):
    """
    判断 ind1 是否 Pareto 支配 ind2

    Pareto 支配的定义：
    一个解 A 支配另一个解 B，当且仅当：
      1. A 在所有目标上都不比 B 差（better_or_equal）
      2. A 在至少一个目标上严格优于 B（strictly_better）

    为什么要用 Pareto 优化？
      在交易策略优化中，我们往往有多个冲突的目标：
      - 最大化收益（annual_return） vs 最小化回撤（max_dd）
      - 最大化夏普比率 vs 增加交易次数
      没有单一的最优解，而是一组 Pareto 最优解——你在某个目标上变好，
      就必然在另一个目标上变差。

    参数:
        ind1, ind2: 两个 Individual 对象
        objectives: 目标定义列表，[(目标名, 方向), ...]
            方向: 'max' 表示越大越好，'min' 表示越小越好
            示例: [('annual_return', 'max'), ('max_dd', 'min')]

    返回:
        bool，ind1 是否支配 ind2
    """
    better_or_equal = True
    strictly_better = False

    for obj_name, direction in objectives:
        v1 = ind1.fitness.get(obj_name, 0)
        v2 = ind2.fitness.get(obj_name, 0)

        if direction == 'max':
            if v1 < v2:
                better_or_equal = False
            if v1 > v2:
                strictly_better = True
        else:  # direction == 'min'
            if v1 > v2:
                better_or_equal = False
            if v1 < v2:
                strictly_better = True

    return better_or_equal and strictly_better


def pareto_front(population_list, objectives):
    """
    计算 Pareto 前沿（非支配解集）

    Pareto 前沿是一组互不支配的解的集合，位于目标空间中的"最前线"。
    直观理解：Pareto 前沿上的每个解都在至少一个目标上是"最优的"。

    在量化策略中的应用：
      比如我们有 100 组参数，看它们的年化收益和最大回撤：
      - 有些收益高但回撤大（激进策略）
      - 有些收益低但回撤小（保守策略）
      - Pareto 前沿就是那些"在收益和回撤之间做了最佳权衡"的策略

    参数:
        population_list: Individual 列表
        objectives: [(目标名, 方向), ...] 目标定义

    返回:
        Pareto 前沿上的 Individual 列表
    """
    front = []
    for ind in population_list:
        # 检查 ind 是否被种群中的任何其他个体支配
        is_dominated = False
        for other in population_list:
            if other is ind:
                continue
            if dominates(other, ind, objectives):
                is_dominated = True
                break
        # 如果不被任何个体支配，则在前沿上
        if not is_dominated:
            front.append(ind)
    return front


# ============================================================
# Backtrader 适应度评估
# ============================================================

def run_backtest_fitness(strategy_class, params, df,
                         initial_cash=None, commission=None):
    """
    运行 Backtrader 回测并返回适应度指标

    这是连接遗传算法和 Backtrader 回测的桥梁。
    在遗传算法的每一代中，每个个体都需要调用此函数来评估其"生存价值"。

    参数:
        strategy_class: Backtrader 策略类
        params: dict, 策略参数（个体的基因型）
        df: DataFrame, OHLCV 数据
        initial_cash: 初始资金，默认使用 .env 配置
        commission: 手续费率，默认使用 .env 配置

    返回:
        dict, 适应度指标，包含:
          - sharpe: 夏普比率（遗传算法默认优化目标）
          - annual_return: 年化收益率
          - max_dd: 最大回撤
          - total_return: 总收益率
          - total_trades: 交易次数
          - win_rate: 胜率

    异常处理：
      如果回测过程中发生任何异常（比如参数不合法导致 Backtrader 崩溃），
      返回一个"惩罚"适应度（sharpe=-999），让遗传算法自动淘汰这些有问题的个体。
    """
    if initial_cash is None:
        initial_cash = INITIAL_CASH
    if commission is None:
        commission = COMMISSION

    try:
        cerebro = bt.Cerebro()
        # params 用 ** 解包为关键字参数传递给策略构造函数
        cerebro.addstrategy(strategy_class, **params)
        cerebro.adddata(bt.feeds.PandasData(dataname=df))
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=commission)
        cerebro.addsizer(bt.sizers.PercentSizer, percents=95)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

        results = cerebro.run()
        strat = results[0]

        final_value = cerebro.broker.getvalue()
        total_return = (final_value - initial_cash) / initial_cash

        trading_days = len(df)
        years = trading_days / 252
        if years > 0 and total_return > -1:
            annual_return = (1 + total_return) ** (1 / years) - 1
        else:
            annual_return = total_return

        sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0

        dd = strat.analyzers.drawdown.get_analysis()
        max_dd = dd.get('max', {}).get('drawdown', 0) / 100  # 转换为小数

        ta = strat.analyzers.trades.get_analysis()
        total_trades = ta.get('total', {}).get('total', 0)
        won = ta.get('won', {}).get('total', 0)
        win_rate = won / total_trades if total_trades > 0 else 0

        return {
            'sharpe': sharpe,
            'annual_return': annual_return,
            'max_dd': max_dd,
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
        }

    except Exception as e:
        # 任何异常都返回惩罚适应度
        # 这个惩罚值（-999）足够低，保证有问题的个体在锦标赛中被自然淘汰
        return {
            'sharpe': -999,
            'annual_return': -1,
            'max_dd': 1.0,
            'total_return': -1,
            'total_trades': 0,
            'win_rate': 0,
        }
