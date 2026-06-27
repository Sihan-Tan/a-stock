---
name: strategy-backtest
description: "对指定股票运行量化策略回测(MACD/双均线)，输出胜率、收益率、最大回撤、最新信号等。在用户询问策略表现、是否有买卖信号、要不要买入时使用。"
keywords: 回测, 策略, MACD, 均线, 胜率, 收益率, 信号, 金叉, 死叉, 买入, 卖出
---

# strategy-backtest 技能指南

## 适用场景
- 用户问某只股票是否有买卖信号（MACD金叉/死叉、均线交叉）
- 用户想看某个策略的历史回测表现（胜率、收益率、最大回撤）
- 下单前先回测验证策略的有效性

## 前置条件
- MiniQMT 客户端已安装并保持登录运行状态
- xtquant 包已安装到 Python 环境中

## 可用脚本

| 脚本 | 功能 | 参数 |
|------|------|------|
| `scripts/run_backtest.py` | 运行策略回测 | `--code <股票代码>` `[--strategy macd\|double_ma]` `[--start YYYYMMDD]` `[--count 天数]` |

## 参数说明
- `--code`: 股票代码，带后缀，如 `513100.SH`（必填）
- `--strategy`: 策略类型，`macd`（默认）或 `double_ma`（双均线）
- `--start`: 回测开始日期，格式 `YYYYMMDD`，默认取最近数据
- `--count`: 获取K线条数，默认 `250`（约一年交易日）

## 执行流程（必须遵守）
1. 识别用户提到的股票，转换为带后缀的代码
2. 直接执行 `python skills/strategy-backtest/scripts/run_backtest.py --code <代码> [其他参数]`
3. 解读返回的 JSON 数据，重点关注:
   - `latest_signal`: 当前最新信号（golden_cross/death_cross/none）
   - `win_rate`: 历史胜率
   - `total_return`: 总收益率
   - `max_drawdown`: 最大回撤
4. 用通俗易懂的方式向用户展示回测结果

禁止：安装依赖、检查环境、验证包版本。

## 示例对话

用户: "513100 最近有MACD金叉吗？"
步骤:
1. 执行 `python skills/strategy-backtest/scripts/run_backtest.py --code 513100.SH --strategy macd`
2. 查看 `latest_signal` 字段
3. 返回信号情况和回测绩效

用户: "回测双均线策略在纳指ETF上的表现"
步骤:
1. 执行 `python skills/strategy-backtest/scripts/run_backtest.py --code 513100.SH --strategy double_ma --count 500`
2. 解读胜率、收益率、最大回撤
3. 给出分析和建议

注意：所有路径参数使用相对路径（不带前导 /）。
