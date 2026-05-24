# -*- coding: utf-8 -*-
"""
自定义策略开发与加载

1-6号脚本的策略是"写死在脚本里"的，直接运行即可。
本脚本演示的是"插件机制"：
  - 把策略文件丢到 strategies/ 目录
  - 系统自动发现并加载，不用改主程序代码
  - 这是线上 Zoe 回测系统使用的架构

自定义策略规范（3步）：
  1. 在 strategies/ 目录下创建 .py 文件
  2. 定义 STRATEGY_META 字典（策略元信息）
  3. 定义 Strategy 类，继承 bt.Strategy

示例: strategies/macd_divergence.py (MACD底背离策略)

运行: python 7-自定义策略.py
"""
import sys
import importlib.util
from pathlib import Path
import backtrader as bt
from data_loader import run_and_report

# 策略文件存放目录: 当前脚本同级目录下的 strategies/ 文件夹
STRATEGY_DIR = Path(__file__).parent / 'strategies'


def load_custom_strategies():
    """扫描 strategies/ 目录，动态加载所有自定义策略

    实现原理:
      使用 Python 的 importlib 在运行时动态导入 .py 文件，
      无需提前 import，新增策略文件后自动生效。

    返回:
        dict: {策略键名: {'meta': 元信息字典, 'class': 策略类}, ...}
    """
    strategies = {}
    if not STRATEGY_DIR.exists():
        print(f"策略目录不存在: {STRATEGY_DIR}")
        return strategies

    # 遍历 strategies/ 目录下所有 .py 文件
    for py_file in sorted(STRATEGY_DIR.glob('*.py')):
        # 跳过以下划线开头的文件 (如 __init__.py, _utils.py)
        if py_file.name.startswith('_'):
            continue

        key = py_file.stem                                                # 文件名(不含扩展名)作为策略键
        module_name = f'custom_{key}'                                     # 构造唯一模块名，避免冲突

        # 动态导入: 从文件路径创建模块规范 → 创建模块对象 → 执行模块代码
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod                                    # 注册到 sys.modules
        spec.loader.exec_module(mod)                                      # 执行模块代码

        # 提取策略的元信息和类 (必须同时存在才有效)
        meta = getattr(mod, 'STRATEGY_META', None)
        cls = getattr(mod, 'Strategy', None)
        if not meta or not cls:
            print(f"跳过 {py_file.name}: 缺少 STRATEGY_META 或 Strategy 类")
            continue

        strategies[key] = {'meta': meta, 'class': cls}
        print(f"加载: {key} ({meta.get('name', '')})")
    return strategies


def show_template():
    """打印自定义策略模板，方便用户复制粘贴快速开发新策略"""
    print("""
# ============================================================
# 自定义策略模板 - 保存到 strategies/ 目录即可自动加载
# ============================================================
import backtrader as bt

# 策略元信息 (必须定义，用于系统注册和展示)
STRATEGY_META = {
    'name': '策略中文名',            # 策略显示名称
    'category': 'custom',            # 策略分类
    'params': {'period': 20},        # 策略参数及默认值
    'logic': '买入条件 -> 买入; 卖出条件 -> 卖出',  # 交易逻辑简述
}

# 策略类 (必须命名为 Strategy，继承 bt.Strategy)
class Strategy(bt.Strategy):
    params = (('period', 20),)       # Backtrader 参数定义格式

    def __init__(self):
        # 初始化指标: 计算20日均线
        self.sma = bt.indicators.SMA(self.data.close, period=self.p.period)

    def next(self):
        # 每个交易日执行一次
        if not self.position:
            # 收盘价站上均线 → 买入
            if self.data.close[0] > self.sma[0]:
                self.buy()
        else:
            # 收盘价跌破均线 → 卖出
            if self.data.close[0] < self.sma[0]:
                self.close()
""")


if __name__ == '__main__':
    # 第一部分: 展示自定义策略模板
    print("=" * 60)
    print("自定义策略模板")
    print("=" * 60)
    show_template()

    # 第二部分: 加载并运行 strategies/ 目录下的所有自定义策略
    strategies = load_custom_strategies()
    if not strategies:
        print("没有找到自定义策略，请在 strategies/ 目录下创建策略文件")
    else:
        for key, info in strategies.items():
            meta = info['meta']
            print(f"\n策略: {meta.get('name', key)}")
            print(f"逻辑: {meta.get('logic', '')}")
            # 一键运行回测 + 生成绩效报告 + 绘制图表
            run_and_report(info['class'], '600519.SH', '2024-01-01', '2025-12-31',
                          label=meta.get('name', key), plot=True)
