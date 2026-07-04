# -*- coding: utf-8 -*-
"""
配置文件 - 从 .env 读取数据库连接和回测参数
"""
from pathlib import Path
import pymysql
from dotenv import dotenv_values

_env_path = Path(__file__).parent / '.env'
_env = dotenv_values(_env_path)

# 数据库配置
DB_CONFIG = {
    'host': _env.get('WUCAI_SQL_HOST', 'localhost'),
    'user': _env.get('WUCAI_SQL_USERNAME', 'root'),
    'password': _env.get('WUCAI_SQL_PASSWORD', ''),
    'database': _env.get('WUCAI_SQL_DB', 'wucai_trade'),
    'port': int(_env.get('WUCAI_SQL_PORT', '3306')),
    'charset': 'utf8mb4'
}

# 回测参数（可在 .env 中修改）
INITIAL_CASH = int(_env.get('BACKTEST_INITIAL_CASH', '1000000'))
COMMISSION = float(_env.get('BACKTEST_COMMISSION', '0.0002'))
POSITION_PCT = int(_env.get('BACKTEST_POSITION_PCT', '95'))


def get_connection():
    """获取数据库连接"""
    return pymysql.connect(**DB_CONFIG)


def execute_query(sql, params=None):
    """执行查询，返回字典列表"""
    conn = get_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    cursor.execute(sql, params or ())
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result
