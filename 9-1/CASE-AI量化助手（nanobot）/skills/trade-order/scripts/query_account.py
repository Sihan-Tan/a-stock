# -*- coding: utf-8 -*-
"""
账户查询脚本

查询 miniQMT 账户的资产、持仓、委托、成交信息，输出 JSON。

用法:
    python query_account.py --action asset
    python query_account.py --action positions
    python query_account.py --action orders
    python query_account.py --action trades
"""
import os
import sys
import json
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from dotenv import load_dotenv
from miniqmt_trader import MiniQMTTrader


def main():
    parser = argparse.ArgumentParser(description='账户查询')
    parser.add_argument('--action', required=True,
                        choices=['asset', 'positions', 'orders', 'trades'],
                        help='查询类型')
    args = parser.parse_args()

    env_path = os.path.join(os.path.dirname(script_dir), '..', '..', '.env')
    if not os.path.exists(env_path):
        env_path = os.path.join(os.path.dirname(script_dir), '..', '.env')
    load_dotenv(env_path)

    qmt_path = os.getenv('QMT_PATH')
    account_id = os.getenv('ACCOUNT_ID')

    if not qmt_path or not account_id:
        print(json.dumps({"error": "缺少配置，请检查 .env 中的 QMT_PATH 和 ACCOUNT_ID"},
                          ensure_ascii=False))
        return

    trader = MiniQMTTrader(qmt_path, account_id)

    try:
        trader.connect()
    except Exception as e:
        print(json.dumps({"error": f"连接失败: {str(e)}"}, ensure_ascii=False))
        return

    try:
        if args.action == 'asset':
            data = trader.query_asset()
            result = {"action": "asset", "data": data}

        elif args.action == 'positions':
            data = trader.query_positions()
            result = {"action": "positions", "count": len(data), "data": data}

        elif args.action == 'orders':
            data = trader.query_orders()
            result = {"action": "orders", "count": len(data), "data": data}

        elif args.action == 'trades':
            data = trader.query_trades()
            result = {"action": "trades", "count": len(data), "data": data}

        print(json.dumps(result, ensure_ascii=False, indent=2))

    finally:
        trader.disconnect()


if __name__ == '__main__':
    main()
