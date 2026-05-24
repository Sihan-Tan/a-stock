# -*- coding: utf-8 -*-
"""
配置文件 - 从 .env 读取数据库连接和回测参数

读取项目根目录下的 .env 文件，获取:
  - MySQL 数据库连接信息 (WUCAI_SQL_*)
  - 回测参数 (初始资金、手续费率、仓位比例)
若 .env 中未配置则使用代码中的默认值
"""
from pathlib import Path
from dotenv import dotenv_values
import pymysql

# 定位 .env 文件: 与当前脚本同目录
_env_path = Path(__file__).parent / '.env'
# 加载 .env 中的所有键值对到字典
_env = dotenv_values(_env_path)

# ============================================================
# 数据库配置 - 连接 wucai_trade 数据库
# ============================================================
DB_CONFIG = {
    'host': _env.get('WUCAI_SQL_HOST', 'localhost'),           # 数据库主机地址
    'user': _env.get('WUCAI_SQL_USERNAME', 'root'),            # 数据库用户名
    'password': _env.get('WUCAI_SQL_PASSWORD', ''),            # 数据库密码
    'database': _env.get('WUCAI_SQL_DB', 'wucai_trade'),       # 数据库名称
    'port': int(_env.get('WUCAI_SQL_PORT', '3306')),           # 数据库端口号，默认3306
    'charset': 'utf8mb4'                                        # 字符集，支持emoji等4字节字符
}

# ============================================================
# 回测参数 - 可在 .env 中覆盖默认值
# ============================================================
INITIAL_CASH = int(_env.get('BACKTEST_INITIAL_CASH', '1000000'))   # 初始资金，默认100万
COMMISSION = float(_env.get('BACKTEST_COMMISSION', '0.0002'))      # 手续费率，默认万分之二
POSITION_PCT = int(_env.get('BACKTEST_POSITION_PCT', '95'))        # 仓位比例(%)，默认95%仓位


def get_connection():
    """获取数据库连接（每次调用创建新连接，用完记得关闭）"""
    return pymysql.connect(**DB_CONFIG)


def execute_query(sql, params=None):
    """执行SQL查询，返回字典列表

    参数:
        sql:    SQL查询语句，使用 %s 作为占位符
        params: 查询参数元组，用于替换 %s 占位符（防止SQL注入）

    返回:
        字典列表，每个字典代表一行数据，键为列名

    示例:
        rows = execute_query("SELECT * FROM stock WHERE code=%s", ('600519.SH',))
    """
    conn = get_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)  # DictCursor 让结果以字典形式返回
    cursor.execute(sql, params or ())                  # 参数化查询，防止SQL注入
    result = cursor.fetchall()                         # 获取全部结果行
    cursor.close()
    conn.close()
    return result
