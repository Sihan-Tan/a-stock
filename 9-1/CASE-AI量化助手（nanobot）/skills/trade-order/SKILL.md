---
name: trade-order
description: "通过 miniQMT 执行实盘交易操作，包括买入、卖出、撤单、查询持仓和委托。在用户要求下单、查持仓、查委托时使用。"
keywords: 买入, 卖出, 下单, 持仓, 委托, 撤单, 交易, 账户, 资产, 买, 卖
---

# trade-order 技能指南

## 适用场景
- 用户要求买入/卖出某只股票
- 用户想查看账户资产、持仓情况
- 用户想查看当日委托或成交记录
- 用户要求撤销委托

## 前置条件
- MiniQMT 客户端已安装并保持登录运行状态
- xtquant 包已安装到 Python 环境中
- `.env` 文件已配置 `QMT_PATH` 和 `ACCOUNT_ID`

## 交易原则（必须严格遵守）

1. **AI 建议，人类授权**: 先给出完整的交易建议方案（股票、方向、股数、价格、理由），用户明确授权后才执行
2. **建议要有依据**: 结合策略信号（如 MACD 金叉）和当前持仓/资金情况给出股数建议
3. 绝不自动连续下单，每次交易都需要单独授权

## 可用脚本

| 脚本 | 功能 | 参数 |
|------|------|------|
| `scripts/query_account.py` | 查询账户信息 | `--action asset\|positions\|orders\|trades` |
| `scripts/place_order.py` | 下单/撤单 | `--action buy\|sell\|cancel --code <代码> --volume <数量> [--price 价格] [--order_id 编号]` |

## query_account.py 参数说明
- `--action asset`: 查询账户总资产、可用资金、持仓市值
- `--action positions`: 查询所有持仓（代码、数量、可用、成本、市值）
- `--action orders`: 查询当日所有委托
- `--action trades`: 查询当日所有成交

## place_order.py 参数说明
- `--action buy`: 买入股票
- `--action sell`: 卖出股票
- `--action cancel`: 撤销委托（需提供 `--order_id`）
- `--code`: 股票代码，如 `513100.SH`（buy/sell 必填）
- `--volume`: 买卖数量，必须为100的整数倍（buy/sell 必填）
- `--price`: 委托价格，不填或为0时使用市价
- `--order_id`: 委托编号（cancel 时必填）

## 执行流程（必须遵守）

### 下单流程（AI建议 -> 人类授权 -> 执行）
1. **分析**: 调用 strategy-backtest 获取策略信号，调用 query_account.py 查看持仓和可用资金
2. **建议**: 向用户展示完整交易方案:
   - 策略依据（如"MACD 金叉，历史胜率55%"）
   - 当前持仓情况
   - 建议操作: 股票代码、方向、股数、价格方式
   - 示例: "513100.SH 出现 MACD 金叉，当前无持仓，可用资金 5 万。建议市价买入 200 股（约 xx 元），是否授权执行？"
3. **授权**: 等用户明确回复"确认"/"好"/"买"/"授权"等肯定语
4. **执行**: `python skills/trade-order/scripts/place_order.py --action buy --code 513100.SH --volume 200`
5. **反馈**: 解读返回的 JSON，报告委托编号和状态

### 查询流程
1. 直接执行 `python skills/trade-order/scripts/query_account.py --action <类型>`
2. 解读返回的 JSON，格式化展示给用户

禁止：安装依赖、检查环境、验证包版本。

## 示例对话

用户: "MACD 出了金叉，帮我买点 513100"
步骤:
1. 调用 strategy-backtest 确认信号: `python skills/strategy-backtest/scripts/run_backtest.py --code 513100.SH --strategy macd`
2. 查询持仓和资金: `python skills/trade-order/scripts/query_account.py --action positions` 和 `--action asset`
3. 给出建议: "513100.SH MACD 金叉确认，历史胜率55%。当前无持仓，可用资金3万。建议市价买入 200 股（约 xx 元），是否授权？"
4. 用户授权后执行 place_order.py
5. 返回委托结果

用户: "看下我的持仓"
步骤:
1. 执行 `python skills/trade-order/scripts/query_account.py --action positions`
2. 格式化展示持仓信息

注意：所有路径参数使用相对路径（不带前导 /）。
