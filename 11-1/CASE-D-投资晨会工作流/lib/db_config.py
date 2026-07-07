# -*- coding: utf-8 -*-
# 21-CASE-D: 数据库配置 (本 CASE 内嵌副本)
"""
数据库配置 -- 直接读 CASE-A 落库的 wucai_trade.* 表

设计原则:
    本 CASE 是"晨会工作流", 不做跨 CASE import
    跟 CASE-A / CASE-B 用一套 .env (WUCAI_SQL_*), 但代码独立一份

环境变量在外层 .env (CASE-D-投资晨会工作流/.env) 中定义
"""
from pathlib import Path
import pymysql
from dotenv import dotenv_values

_env_path = Path(__file__).resolve().parent.parent / '.env'
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
        return cursor.fetchall()
    finally:
        conn.close()
