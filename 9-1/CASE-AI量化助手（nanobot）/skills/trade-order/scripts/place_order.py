# -*- coding: utf-8 -*-
"""
下单/撤单脚本

通过 miniQMT 执行买入、卖出、撤单操作，输出 JSON 结果。
由 Agent 给出建议方案（股票、数量、价格），用户授权后再调用本脚本执行。

用法:
    python place_order.py --action buy --code 513100.SH --volume 100
    python place_order.py --action sell --code 513100.SH --volume 200 --price 1.50
    python place_order.py --action cancel --order_id 12345
"""
import os
import sys
import json
import time
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from dotenv import load_dotenv
from miniqmt_trader import MiniQMTTrader


def main():
    parser = argparse.ArgumentParser(description='交易下单')
    parser.add_argument('--action', required=True,
                        choices=['buy', 'sell', 'cancel'],
                        help='操作类型')
    parser.add_argument('--code', default='', help='股票代码，如 513100.SH')
    parser.add_argument('--volume', type=int, default=100, help='买卖数量')
    parser.add_argument('--price', type=float, default=0.0, help='委托价格，0为市价')
    parser.add_argument('--order_id', type=int, default=0, help='委托编号（撤单用）')
    args = parser.parse_args()

    if args.action in ('buy', 'sell'):
        if not args.code:
            print(json.dumps({"status": "error", "message": "缺少股票代码 --code"},
                              ensure_ascii=False))
            return

        if args.volume <= 0 or args.volume % 100 != 0:
            print(json.dumps({
                "status": "error",
                "message": f"数量必须为100的正整数倍，当前: {args.volume}"
            }, ensure_ascii=False))
            return

    if args.action == 'cancel' and args.order_id <= 0:
        print(json.dumps({"status": "error", "message": "撤单需要 --order_id"},
                          ensure_ascii=False))
        return

    env_path = os.path.join(os.path.dirname(script_dir), '..', '..', '.env')
    if not os.path.exists(env_path):
        env_path = os.path.join(os.path.dirname(script_dir), '..', '.env')
    load_dotenv(env_path)

    qmt_path = os.getenv('QMT_PATH')
    account_id = os.getenv('ACCOUNT_ID')

    if not qmt_path or not account_id:
        print(json.dumps({"status": "error",
                          "message": "缺少配置，请检查 .env 中的 QMT_PATH 和 ACCOUNT_ID"},
                          ensure_ascii=False))
        return

    trader = MiniQMTTrader(qmt_path, account_id)

    try:
        trader.connect()
    except Exception as e:
        print(json.dumps({"status": "error", "message": f"连接失败: {str(e)}"},
                          ensure_ascii=False))
        return

    try:
        if args.action == 'buy':
            order_id, msg = trader.buy(args.code, args.volume, price=args.price)
            time.sleep(1)
            result = {
                "status": "success" if order_id else "error",
                "action": "buy",
                "stock_code": args.code,
                "volume": args.volume,
                "price": args.price if args.price > 0 else "市价",
                "order_id": order_id,
                "message": msg,
                "events": trader.events,
            }

        elif args.action == 'sell':
            order_id, msg = trader.sell(args.code, args.volume, price=args.price)
            time.sleep(1)
            result = {
                "status": "success" if order_id else "error",
                "action": "sell",
                "stock_code": args.code,
                "volume": args.volume,
                "price": args.price if args.price > 0 else "市价",
                "order_id": order_id,
                "message": msg,
                "events": trader.events,
            }

        elif args.action == 'cancel':
            ret_code, msg = trader.cancel(args.order_id)
            time.sleep(1)
            result = {
                "status": "success" if ret_code == 0 else "error",
                "action": "cancel",
                "order_id": args.order_id,
                "message": msg,
                "events": trader.events,
            }

        print(json.dumps(result, ensure_ascii=False, indent=2))

    finally:
        trader.disconnect()


if __name__ == '__main__':
    main()
