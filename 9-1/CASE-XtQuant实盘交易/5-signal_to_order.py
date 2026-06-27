# -*- coding: utf-8 -*-
"""
示例5: 策略信号转实盘订单

场景: "MACD 金叉自动买入 -- 让策略自己盯盘下单"

这是整个 XtQuant 交易系列的核心示例 -- 将策略信号转化为真实订单。

核心架构: 信号层 + 执行层分离

  +-----------------------+         +------------------------+
  |      信号层           |         |      执行层            |
  |  (xtdata + 策略计算)  | ------> |  (MiniQMTTrader)       |
  |                       |  信号   |                        |
  |  1. 获取实时行情      |         |  1. 接收交易指令       |
  |  2. 计算技术指标      |         |  2. 风控检查           |
  |  3. 检测买卖信号      |         |  3. 下单到交易所       |
  |  4. 输出交易指令      |         |  4. 回调跟踪成交       |
  +-----------------------+         +------------------------+

为什么要分离?
  - 信号逻辑和交易逻辑解耦，便于独立修改和测试
  - 可以先用历史数据验证信号，再接入实盘
  - 执行层可以加风控，不影响信号计算
  - 信号层可以替换为任何策略 (MACD、均线、AI模型等)

本示例流程:
  1. 用 xtdata 获取 513100.SH 的日线数据
  2. 计算 MACD 指标，检测金叉/死叉
  3. 金叉 -> 调用 MiniQMTTrader.buy() 买入一手
  4. 死叉 -> 调用 MiniQMTTrader.sell() 卖出持仓

环境要求:
  - 已安装 xtquant, python-dotenv, pandas
  - miniQMT 客户端已启动并登录
  - .env 中配置 QMT_PATH 和 ACCOUNT_ID
"""
import os
import sys
import time
import pandas as pd
from dotenv import load_dotenv
from xtquant import xtdata


# ============================================================
# 从 .env 加载配置
# ============================================================
load_dotenv()
QMT_PATH = os.getenv("QMT_PATH")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")

STOCK_CODE = "513100.SH"
STOCK_NAME = "纳指ETF"
MACD_SHORT = 12
MACD_LONG = 26
MACD_SIGNAL = 9
VOLUME = 100


# ============================================================
# 信号层: MACD 信号计算
# ============================================================

def calc_macd(close_prices, short=12, long=26, signal=9):
    """
    计算 MACD 指标

    返回:
        dif: DIF线 (快线EMA - 慢线EMA)
        dea: DEA线 (DIF的EMA)
        macd_bar: MACD柱 = (DIF - DEA) * 2
    """
    close = pd.Series(close_prices)
    ema_short = close.ewm(span=short, adjust=False).mean()
    ema_long = close.ewm(span=long, adjust=False).mean()
    dif = ema_short - ema_long
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd_bar = (dif - dea) * 2
    return dif.values, dea.values, macd_bar.values


def check_signal(dif, dea):
    """
    检查最新一根K线是否产生金叉/死叉信号

    返回:
        'buy'  -- 金叉 (DIF 从下方穿越 DEA)
        'sell' -- 死叉 (DIF 从上方穿越 DEA)
        None   -- 无信号
    """
    if len(dif) < 2:
        return None

    prev_dif, curr_dif = dif[-2], dif[-1]
    prev_dea, curr_dea = dea[-2], dea[-1]

    if prev_dif <= prev_dea and curr_dif > curr_dea:
        return 'buy'
    if prev_dif >= prev_dea and curr_dif < curr_dea:
        return 'sell'
    return None


# ============================================================
# 数据层: 通过 xtdata 获取行情 (使用 get_market_data_ex)
# ============================================================

def fetch_kline(stock_code, period='1d', start_date='20240101', count=500):
    """
    通过 xtdata 获取 K 线数据

    使用 get_market_data_ex (推荐的新版 API)

    返回:
        DataFrame: 包含 date, open, high, low, close, volume 列
    """
    xtdata.download_history_data(
        stock_code, period=period,
        start_time=start_date, end_time='',
        incrementally=True
    )

    data = xtdata.get_market_data_ex(
        field_list=['open', 'high', 'low', 'close', 'volume'],
        stock_list=[stock_code],
        period=period,
        start_time=start_date,
        end_time='',
        count=count
    )

    if not data or stock_code not in data:
        raise Exception(f"获取 {stock_code} 数据失败")

    df = data[stock_code]
    if df is None or len(df) == 0:
        raise Exception(f"{stock_code} 无数据")

    df.index = pd.to_datetime(df.index)
    df['date'] = df.index
    df = df.reset_index(drop=True)
    return df


# ============================================================
# 执行层: 加载 MiniQMTTrader
# ============================================================

def load_trader_class():
    """从 4-miniqmt_trader.py 加载 MiniQMTTrader 类"""
    from importlib.util import spec_from_file_location, module_from_spec
    current_dir = os.path.dirname(os.path.abspath(__file__))
    spec = spec_from_file_location(
        "miniqmt_trader",
        os.path.join(current_dir, "4-miniqmt_trader.py"))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MiniQMTTrader


# ============================================================
# 核心流程: 信号 -> 订单
# ============================================================

def run_signal_to_order(check_interval=60):
    """
    策略信号转实盘订单的主循环

    流程:
      1. 连接 MiniQMTTrader
      2. 每隔 check_interval 秒:
         a. 获取最新行情
         b. 计算 MACD
         c. 检测金叉/死叉
         d. 金叉 -> 买入一手 / 死叉 -> 卖出持仓
      3. 通过回调跟踪订单状态
    """
    MiniQMTTrader = load_trader_class()

    print("=" * 60)
    print("策略信号转实盘订单")
    print(f"标的: {STOCK_CODE} ({STOCK_NAME})")
    print(f"策略: MACD ({MACD_SHORT}/{MACD_LONG}/{MACD_SIGNAL})")
    print(f"每次交易: {VOLUME}股 (一手)")
    print(f"检查间隔: {check_interval}秒")
    print("=" * 60)

    # --- 连接交易 ---
    print("\n[1] 连接 MiniQMTTrader...")
    trader = MiniQMTTrader(
        qmt_path=QMT_PATH,
        account_id=ACCOUNT_ID,
        max_positions=10,
        max_order_amount=50000,
    )
    trader.connect()

    asset = trader.query_asset()
    print(f"    总资产: {asset.get('total_asset', 0):,.0f} "
          f"可用: {asset.get('cash', 0):,.0f}")

    positions = trader.query_positions()
    for pos in positions:
        if pos['stock_code'] == STOCK_CODE:
            print(f"    已持有 {STOCK_CODE}: {pos['volume']}股 "
                  f"可用{pos['can_use_volume']}股")

    # --- 获取初始数据 ---
    print(f"\n[2] 获取 {STOCK_CODE} 历史数据...")
    df = fetch_kline(STOCK_CODE, '1d', '20240101')
    print(f"    获取 {len(df)} 条K线 "
          f"({df['date'].iloc[0].strftime('%Y-%m-%d')} ~ "
          f"{df['date'].iloc[-1].strftime('%Y-%m-%d')})")

    dif, dea, _ = calc_macd(df['close'].values,
                             MACD_SHORT, MACD_LONG, MACD_SIGNAL)

    last_dif_above = dif[-1] > dea[-1]
    curr_price = df['close'].iloc[-1]
    print(f"    当前价格: {curr_price:.3f}")
    print(f"    DIF: {dif[-1]:.6f}  DEA: {dea[-1]:.6f}")
    print(f"    当前状态: {'DIF > DEA (多头)' if last_dif_above else 'DIF < DEA (空头)'}")

    # --- 监控循环 ---
    print(f"\n[3] 开始监控，每 {check_interval} 秒检查一次...")
    print("    按 Ctrl+C 停止\n")

    try:
        while True:
            try:
                df = fetch_kline(STOCK_CODE, '1d', '20240101')

                if len(df) < MACD_LONG + MACD_SIGNAL + 2:
                    print("    数据不足，跳过...")
                    time.sleep(check_interval)
                    continue

                dif, dea, _ = calc_macd(df['close'].values,
                                         MACD_SHORT, MACD_LONG, MACD_SIGNAL)

                curr_price = df['close'].iloc[-1]
                curr_date = df['date'].iloc[-1].strftime('%Y-%m-%d')
                dif_above = dif[-1] > dea[-1]

                print(f"  [{curr_date}] 价格:{curr_price:.3f} "
                      f"DIF:{dif[-1]:.6f} DEA:{dea[-1]:.6f} "
                      f"{'多头' if dif_above else '空头'}")

                signal = check_signal(dif, dea)

                if signal == 'buy':
                    print(f"\n  *** MACD 金叉 -> 买入信号 ***")
                    positions = trader.query_positions()
                    already_held = any(p['stock_code'] == STOCK_CODE
                                       for p in positions)
                    if not already_held:
                        print(f"  执行: 市价买入 {STOCK_CODE} {VOLUME}股")
                        trader.buy(STOCK_CODE, VOLUME,
                                   strategy_name='MACD',
                                   remark=f'金叉 DIF={dif[-1]:.6f}')
                    else:
                        print(f"  跳过: 已持有 {STOCK_CODE}")

                elif signal == 'sell':
                    print(f"\n  *** MACD 死叉 -> 卖出信号 ***")
                    positions = trader.query_positions()
                    sold = False
                    for pos in positions:
                        if pos['stock_code'] == STOCK_CODE:
                            if pos['can_use_volume'] >= VOLUME:
                                print(f"  执行: 市价卖出 {STOCK_CODE} {VOLUME}股")
                                trader.sell(STOCK_CODE, VOLUME,
                                            strategy_name='MACD',
                                            remark=f'死叉 DIF={dif[-1]:.6f}')
                                sold = True
                            else:
                                print(f"  跳过: 可用持仓不足 "
                                      f"(可用{pos['can_use_volume']}股, "
                                      f"需要{VOLUME}股, T+1限制)")
                            break

                    if not sold and not any(p['stock_code'] == STOCK_CODE
                                            for p in positions):
                        print(f"  跳过: 未持有 {STOCK_CODE}")

                last_dif_above = dif_above

            except Exception as e:
                print(f"  本轮异常: {e}")

            time.sleep(check_interval)

    except KeyboardInterrupt:
        print("\n\n用户中断，停止监控")

    # --- 最终状态 ---
    print("\n" + "=" * 60)
    print("最终状态")
    print("=" * 60)

    asset = trader.query_asset()
    print(f"  总资产: {asset.get('total_asset', 0):,.0f}")
    print(f"  可用资金: {asset.get('cash', 0):,.0f}")

    positions = trader.query_positions()
    if positions:
        print("  持仓:")
        for pos in positions:
            print(f"    {pos['stock_code']}: {pos['volume']}股 "
                  f"市值:{pos['market_value']:,.0f}")

    print(f"\n  回调事件 ({len(trader.events)} 条):")
    for event in trader.events[-10:]:
        print(f"    {event}")

    trader.disconnect()


# ============================================================
# 单次信号检查 (不下单，只看信号)
# ============================================================

def check_once():
    """
    单次检查当前信号状态

    适用场景:
      - 手动查看当前 MACD 信号
      - 配合定时任务调用
    """
    print("=" * 60)
    print(f"单次信号检查: {STOCK_CODE} ({STOCK_NAME})")
    print("=" * 60)

    print("\n获取行情数据...")
    df = fetch_kline(STOCK_CODE, '1d', '20240101')
    print(f"获取 {len(df)} 条K线")

    dif, dea, macd_bar = calc_macd(df['close'].values,
                                    MACD_SHORT, MACD_LONG, MACD_SIGNAL)

    print(f"\n最近5个交易日:")
    print(f"  {'日期':>12s}  {'收盘价':>8s}  {'DIF':>10s}  {'DEA':>10s}  {'MACD柱':>10s}  信号")
    for i in range(-5, 0):
        date_str = df['date'].iloc[i].strftime('%Y-%m-%d')
        sig = check_signal(dif[:len(dif)+i+1], dea[:len(dea)+i+1])
        sig_text = '金叉' if sig == 'buy' else ('死叉' if sig == 'sell' else '-')
        print(f"  {date_str}  {df['close'].iloc[i]:>8.3f}  "
              f"{dif[i]:>10.6f}  {dea[i]:>10.6f}  {macd_bar[i]:>10.6f}  {sig_text}")

    signal = check_signal(dif, dea)
    print(f"\n当前状态: {'DIF > DEA (多头)' if dif[-1] > dea[-1] else 'DIF < DEA (空头)'}")
    if signal:
        print(f"最新信号: {'金叉 -> 买入' if signal == 'buy' else '死叉 -> 卖出'}")
    else:
        print("最新信号: 无 (持续当前方向)")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print("请选择运行模式:")
    print("  1. 单次信号检查 (查看当前MACD状态，不下单)")
    print("  2. 实盘监控 (持续监控信号，触发时自动下单)")
    print()

    choice = input("输入 1 或 2: ").strip()

    if choice == '1':
        check_once()
    elif choice == '2':
        interval = input("检查间隔 (秒, 默认60): ").strip()
        interval = int(interval) if interval else 60
        run_signal_to_order(check_interval=interval)
    else:
        print("无效选择，执行单次信号检查...\n")
        check_once()
