# -*- coding: utf-8 -*-
"""
网格交易引擎
============

提供两种网格引擎:
  1. GridEngine     - 固定价格区间网格
  2. ChanGridEngine - 缠论中枢网格 (支持中枢切换)

网格交易的核心原理:
-------------------
网格交易是一种"震荡市收割机"策略, 核心思想是:

  将价格区间 [lower, upper] 等分为 N 个格子 (levels)
  价格每下穿一个格子 → 买入 1 份 (越跌越买)
  价格每上穿一个格子 → 卖出 1 份 (越涨越卖)

  每完成一次"买入→卖出"循环, 赚取 grid_size 的差价。

  示例: 区间 [100, 110], 5格, grid_size=2
    格子价格: [100, 102, 104, 106, 108, 110]
    价格从 106 跌到 102: 在 104 买入, 在 102 买入
    价格从 102 涨到 108: 在 104 卖出(赚2), 在 106 卖出(赚2)

  关键洞察:
    - 网格不预测方向, 只利用波动
    - 价格在区间内震荡越久, 网格赚得越多
    - 单边上涨会踏空 (卖光仓位), 单边下跌会满仓被套

四要素:
  1. 网格区间 [lower, upper]: 价格波动范围
  2. 格子数量 num_grids: 区间切分数, 越多交易越频繁
  3. 每格仓位 capital_per_grid: 总资金/格子数, 每格买入金额
  4. 出界处理: 价格超出区间时的应对策略
"""


class GridEngine:
    """
    固定网格引擎。

    这是最基础的网格实现: 在预设的价格区间内等距切格,
    价格每穿越一格就执行一次买入或卖出。

    属性:
        upper/lower: 网格上下边界
        num_grids: 格子数量
        grid_size: 每格间距 = (upper - lower) / num_grids
        levels: 各格价位列表 [lower, lower+gs, lower+2gs, ..., upper]
        position_at: 每格的持仓状态, position_at[i] = 在第i格买入的股数
        prev_cell: 上一个交易日的格子位置, 用于判断价格穿越

    统计属性:
        buy_count / sell_count: 买卖次数
        total_profit: 网格累计利润
        max_layers: 最大同时持仓层数
    """

    def __init__(self, upper, lower, num_grids, total_capital):
        """
        初始化固定网格。

        参数:
            upper: 网格上界 (价格超过此值则全部卖出)
            lower: 网格下界 (价格跌破此值则满仓被套)
            num_grids: 格子数量, 越多交易越频繁
            total_capital: 分配给网格的总资金
                          (用于计算每格的买入股数)
        """
        self.upper = upper
        self.lower = lower
        self.num_grids = num_grids
        self.grid_size = (upper - lower) / num_grids      # 每格间距
        self.levels = [lower + i * self.grid_size for i in range(num_grids + 1)]
        self.total_capital = total_capital
        self.capital_per_grid = total_capital / num_grids  # 每格分配资金

        # position_at[i] 表示在第 i 格 (价格 = levels[i]) 买入的股份数
        # 0 表示该格没有持仓
        self.position_at = [0] * (num_grids + 1)
        self.prev_cell = None  # 上一个 bar 的格子位置

        # 统计信息
        self.buy_count = 0
        self.sell_count = 0
        self.total_profit = 0.0
        self.max_layers = 0   # 最大同时持仓层数

    def get_cell(self, price):
        """
        获取价格所在的格子索引。

        格子索引的含义:
          -1:              价格低于网格下界 (出界)
          num_grids:       价格高于或等于网格上界 (出界)
          0 ~ num_grids-1: 价格在网格内的正常格子

        参数:
            price: 当前价格

        返回值:
            int, 格子索引
        """
        if price < self.lower:
            return -1
        if price >= self.upper:
            return self.num_grids
        # 计算所在格子: (price - lower) / grid_size 取整
        return int((price - self.lower) / self.grid_size)

    def calc_shares(self, price):
        """
        计算每格应买入的股数。

        策略: 每格用固定金额 (capital_per_grid) 买入,
        股数取整到 100 股 (A股交易单位为手, 1手=100股)。

        参数:
            price: 买入价格

        返回值:
            int, 股数 (100的倍数), 最少 100 股
        """
        if price <= 0:
            return 0
        shares = self.capital_per_grid / price
        shares = int(shares // 100) * 100  # 取整到100股
        return max(shares, 100)

    def current_layers(self):
        """
        当前持仓层数: 有多少个格子持有仓位。

        返回值:
            int, 0 表示空仓, num_grids 表示满仓
        """
        return sum(1 for s in self.position_at if s > 0)

    def update(self, price):
        """
        核心方法: 根据当前价格更新网格状态, 返回交易信号。

        价格穿越格子的判断逻辑:
          1. 如果 curr_cell < prev_cell: 价格下跌, 穿越格子
             从高到低遍历进入的每个格子 → 发出买入信号
          2. 如果 curr_cell > prev_cell: 价格上涨, 穿越格子
             从低到高遍历离开的每个格子 → 发出卖出信号

        参数:
            price: 当前价格

        返回值:
            list of dict, 每个 dict 代表一个交易信号:
              BUY:  {'action': 'BUY', 'price': float, 'size': int, 'grid_level': int}
              SELL: {'action': 'SELL', 'price': float, 'size': int,
                     'grid_level': int, 'profit': float}
            空列表表示没有交易信号。
        """
        curr_cell = self.get_cell(price)
        signals = []

        # 第一个 bar, 只记录位置, 不交易
        if self.prev_cell is None:
            self.prev_cell = curr_cell
            return signals

        prev_cell = self.prev_cell

        if curr_cell < prev_cell:
            # 价格下跌: 穿越格子 → 买入
            # 遍历的格子: prev_cell-1, prev_cell-2, ..., curr_cell
            # 从高往低, 确保每个进入的格子都被触发
            for cell in range(prev_cell - 1, curr_cell - 1, -1):
                if 0 <= cell < self.num_grids and self.position_at[cell] == 0:
                    size = self.calc_shares(self.levels[cell])
                    if size > 0:
                        signals.append({
                            'action': 'BUY',
                            'price': self.levels[cell],   # 以格子价格买入
                            'size': size,
                            'grid_level': cell,
                        })
                        self.position_at[cell] = size      # 标记该格已持仓
                        self.buy_count += 1

        elif curr_cell > prev_cell:
            # 价格上涨: 穿越格子 → 卖出持仓
            # 遍历的格子: prev_cell, prev_cell+1, ..., curr_cell-1
            # 从低往高, 卖出之前买入的仓位
            for cell in range(prev_cell, curr_cell):
                if 0 <= cell < self.num_grids and self.position_at[cell] > 0:
                    size = self.position_at[cell]
                    sell_price = self.levels[cell + 1]    # 以下一格价格卖出
                    # 利润 = (卖出价 - 买入价) * 股数
                    profit = (sell_price - self.levels[cell]) * size
                    signals.append({
                        'action': 'SELL',
                        'price': sell_price,
                        'size': size,
                        'grid_level': cell,
                        'profit': round(profit, 2),
                    })
                    self.position_at[cell] = 0  # 该格仓位清空
                    self.sell_count += 1
                    self.total_profit += profit

        layers = self.current_layers()
        self.max_layers = max(self.max_layers, layers)
        self.prev_cell = curr_cell
        return signals

    def is_out_of_range(self, price):
        """
        判断价格是否超出网格范围。

        超出范围时策略需要特殊处理:
          - 跌破下界: 已满仓, 只能持有等待
          - 涨破上界: 已空仓, 只能观望

        参数:
            price: 当前价格

        返回值:
            bool
        """
        return price < self.lower or price >= self.upper

    def get_stats(self):
        """
        获取网格的统计信息摘要。

        返回值:
            dict:
              - buy_count: 买入次数
              - sell_count: 卖出次数
              - total_profit: 累计网格利润
              - max_layers: 最大持仓层数
              - current_layers: 当前持仓层数
              - grid_utilization: 卖出/买入比率, 接近1表示网格运转良好
        """
        return {
            'buy_count': self.buy_count,
            'sell_count': self.sell_count,
            'total_profit': round(self.total_profit, 2),
            'max_layers': self.max_layers,
            'current_layers': self.current_layers(),
            'grid_utilization': f"{self.sell_count}/{self.buy_count}" if self.buy_count > 0 else "0/0",
        }

    def summary(self):
        """
        打印网格的详细运行报告。
        """
        s = self.get_stats()
        print(f"  网格参数: [{self.lower:.2f} ~ {self.upper:.2f}], "
              f"{self.num_grids}格, 间距={self.grid_size:.2f}")
        print(f"  网格交易: 买入{s['buy_count']}次, 卖出{s['sell_count']}次, "
              f"利用率={s['grid_utilization']}")
        print(f"  网格利润: {s['total_profit']:.2f}, "
              f"最大持仓层数: {s['max_layers']}")


class ChanGridEngine(GridEngine):
    """
    缠论中枢网格引擎。

    继承自 GridEngine, 但网格边界不由人为设定,
    而是由缠论中枢的 ZG (中枢上沿) 和 ZD (中枢下沿) 决定。

    核心思路:
      缠论中枢 = 市场中价格反复震荡的区间
      网格 = 在震荡区间内低买高卖
      两者天然契合: 中枢的 ZG/ZD 就是网格最合理的上下界

    相比固定网格的优势:
      1. 边界由市场结构决定, 不是拍脑袋
      2. 中枢切换时自动重建网格, 适应市场变化
      3. 价格突破中枢时停止网格, 避免单边行情中继续加仓

    属性 (新增):
      zg/zd: 当前中枢的上下沿
      active: 网格是否活跃 (突破中枢时设为 False)
      switch_count: 中枢切换次数
    """

    def __init__(self, zg, zd, num_grids=6, total_capital=0):
        """
        参数:
            zg: 中枢上沿 (ZG)
            zd: 中枢下沿 (ZD)
            num_grids: 格子数量 (默认6格, 比固定网格少, 因为中枢区间通常较小)
            total_capital: 分配给网格的总资金
        """
        super().__init__(upper=zg, lower=zd, num_grids=num_grids,
                         total_capital=total_capital)
        self.zg = zg
        self.zd = zd
        self.active = True        # 网格是否处于活跃状态
        self.switch_count = 0     # 中枢切换次数统计

    def is_in_zhongshu(self, price):
        """
        判断价格是否在中枢区间内。

        参数:
            price: 当前价格

        返回值:
            bool, True 表示价格在 [ZD, ZG] 内
        """
        return self.zd <= price <= self.zg

    def is_breakout_up(self, price):
        """
        判断价格是否向上突破中枢 (价格 > ZG)。

        向上突破 → 趋势可能开始 → 应切换到趋势跟踪模式。

        参数:
            price: 当前价格

        返回值:
            bool
        """
        return price > self.zg

    def is_breakdown(self, price):
        """
        判断价格是否向下跌破中枢 (价格 < ZD)。

        向下跌破 → 空头趋势 → 应清仓防守。

        参数:
            price: 当前价格

        返回值:
            bool
        """
        return price < self.zd

    def switch_zhongshu(self, new_zg, new_zd):
        """
        切换到新的中枢, 重建网格。

        当缠论识别出新中枢时调用此方法。旧网格的未平仓不会被自动清仓,
        需要在策略层面先平仓再切换 (否则旧格子的仓位会遗留下来)。

        参数:
            new_zg: 新的中枢上沿
            new_zd: 新的中枢下沿
        """
        self.zg = new_zg
        self.zd = new_zd
        self.upper = new_zg
        self.lower = new_zd
        self.grid_size = (new_zg - new_zd) / self.num_grids
        # 重建格子价位列表
        self.levels = [new_zd + i * self.grid_size for i in range(self.num_grids + 1)]
        # 重置所有持仓 (注意: 策略层应先平仓!)
        self.position_at = [0] * (self.num_grids + 1)
        self.prev_cell = None
        self.active = True
        self.switch_count += 1

    def deactivate(self):
        """
        停用网格。

        当价格突破中枢时调用, 停止进一步的网格交易。
        停用后 update() 会返回空列表, 不再产生交易信号。
        """
        self.active = False

    def update(self, price):
        """
        在活跃状态下执行网格逻辑。

        如果网格已停用 (deactivate), 直接返回空列表。

        参数:
            price: 当前价格

        返回值:
            list of dict, 交易信号列表 (同 GridEngine.update)
        """
        if not self.active:
            return []
        return super().update(price)

    def summary(self):
        """
        打印中枢网格的详细运行报告。
        """
        s = self.get_stats()
        print(f"  中枢网格: ZG={self.zg:.2f}, ZD={self.zd:.2f}, "
              f"{self.num_grids}格, 间距={self.grid_size:.2f}")
        print(f"  中枢切换: {self.switch_count}次, "
              f"当前状态: {'活跃' if self.active else '停用'}")
        print(f"  网格交易: 买入{s['buy_count']}次, 卖出{s['sell_count']}次")
        print(f"  网格利润: {s['total_profit']:.2f}, "
              f"最大持仓层数: {s['max_layers']}")
