# -*- coding: utf-8 -*-
"""
数据库配置与查询模块
====================

本模块是整个策略体系的数据基础层, 负责:
  1. 从 .env 文件读取数据库连接参数和回测配置
  2. 提供统一的数据库连接和查询接口
  3. 集中管理回测的全局参数 (初始资金/手续费/仓位比例)

设计理念:
  - 所有环境敏感信息集中放在 .env 文件中, 不硬编码在代码里
  - execute_query() 封装了"连接-查询-关闭"的标准流程,
    上层调用方只需关心 SQL 语句本身

使用方式:
  from db_config import execute_query, INITIAL_CASH, COMMISSION
"""
from pathlib import Path          # 跨平台路径处理, 用于定位 .env 文件
import pymysql                    # MySQL 数据库驱动
from dotenv import dotenv_values  # 从 .env 文件读取键值对

# ---- 定位 .env 文件 ----
# __file__ 是当前文件的绝对路径, parent 是其所在目录
# 这样无论从哪个目录运行脚本, 都能找到 .env 文件
_env_path = Path(__file__).parent / '.env'
_env = dotenv_values(_env_path)

# ---- 数据库连接配置 ----
# 每个配置项都提供了默认值, 如果 .env 文件缺失某个配置, 使用默认值
DB_CONFIG = {
    'host': _env.get('WUCAI_SQL_HOST', 'localhost'),      # 数据库主机地址
    'user': _env.get('WUCAI_SQL_USERNAME', 'root'),        # 数据库用户名
    'password': _env.get('WUCAI_SQL_PASSWORD', ''),        # 数据库密码
    'database': _env.get('WUCAI_SQL_DB', 'wucai_trade'),   # 数据库名称
    'port': int(_env.get('WUCAI_SQL_PORT', '3306')),       # 端口号, 默认MySQL标准端口
    'charset': 'utf8mb4'                                   # 字符集, 支持emoji和特殊字符
}

# ---- 回测全局参数 ----
# 这些参数控制回测的初始条件和交易成本
INITIAL_CASH = int(_env.get('BACKTEST_INITIAL_CASH', '1000000'))  # 初始资金, 默认100万
COMMISSION = float(_env.get('BACKTEST_COMMISSION', '0.0002'))     # 手续费率, 默认万分之二
POSITION_PCT = int(_env.get('BACKTEST_POSITION_PCT', '95'))       # 单笔仓位比例, 默认95%


def get_connection():
    """
    获取数据库连接

    返回值:
        pymysql.Connection 对象, 调用方使用完毕后应自行关闭

    典型用法:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT ...')
        ...
        conn.close()
    """
    return pymysql.connect(**DB_CONFIG)


def execute_query(sql, params=None):
    """
    执行SQL查询, 返回字典列表格式的结果

    这是最常用的查询接口, 封装了完整的"获取连接-创建游标-执行-取结果-关闭"流程。
    调用方无需关心连接管理, 该函数会在返回前自动释放资源。

    参数:
        sql: SQL查询语句, 支持 %s 占位符以防止SQL注入
        params: 可选, 占位符参数列表或元组

    返回值:
        list[dict], 每一行是一个字典, key为列名, value为列值
        查询无结果时返回空列表 []

    示例:
        rows = execute_query(
            "SELECT * FROM trade_stock_daily WHERE stock_code = %s AND trade_date >= %s",
            ['600519.SH', '2024-01-01']
        )
    """
    conn = get_connection()
    # DictCursor: 让查询结果以字典形式返回, 方便按列名访问
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    cursor.execute(sql, params or ())
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result
