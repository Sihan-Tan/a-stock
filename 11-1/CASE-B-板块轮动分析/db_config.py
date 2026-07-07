# -*- coding: utf-8 -*-
# 21-CASE-B: 数据库配置 (本 CASE 独立副本, 与 CASE-A 同库)
"""
直接连 CASE-A 落库的 MySQL, 共用同一套 .env
"""
from pathlib import Path
import pymysql
from dotenv import dotenv_values

_env_path = Path(__file__).parent / '.env'
_env = dotenv_values(_env_path)


DB_CONFIG = {
    'host':     _env.get('WUCAI_SQL_HOST', 'localhost'),
    'user':     _env.get('WUCAI_SQL_USERNAME', 'root'),
    'password': _env.get('WUCAI_SQL_PASSWORD', ''),
    'database': _env.get('WUCAI_SQL_DB', 'wucai_trade'),
    'port':     int(_env.get('WUCAI_SQL_PORT', '3306')),
    'charset':  'utf8mb4',
}


def get_connection():
    return pymysql.connect(**DB_CONFIG)


def execute_query(sql, params=None):
    conn = get_connection()
    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute(sql, params or ())
        result = cursor.fetchall()
        cursor.close()
        return result
    finally:
        conn.close()
