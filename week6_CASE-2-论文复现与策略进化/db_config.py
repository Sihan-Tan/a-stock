# -*- coding: utf-8 -*-
"""
配置文件 - 从 .env 读取数据库连接和回测参数

作用:
  本文件是整个量化回测系统的数据访问层核心配置。
  它从 .env 文件中读取 MySQL 数据库连接信息以及回测的初始资金、
  手续费率和仓位比例等关键参数，并封装了统一的数据库查询接口。

设计理念:
  - 通过 .env 文件将配置与代码分离，便于在不同环境（开发/生产）之间切换
  - 统一管理数据库连接，避免在多个文件中重复编写连接逻辑
  - 所有模块通过 import DB_CONFIG / get_connection() 来获取配置和连接

安全提醒:
  - .env 文件包含数据库密码，不应提交到版本控制系统
  - .env 已被添加到 .gitignore 以防止误提交
"""
from pathlib import Path          # 跨平台路径处理，优雅地拼接文件路径
import pymysql                    # Python MySQL 客户端库
from dotenv import dotenv_values  # 从 .env 文件读取键值对

# ============================================================
# 数据库配置
# ============================================================

# 定位 .env 文件：获取当前文件所在目录下的 .env
# Path(__file__).parent 表示当前 Python 文件所在的目录
# 这样无论项目在什么路径下运行，都能正确定位到 .env 文件
_env_path = Path(__file__).parent / '.env'
_env = dotenv_values(_env_path)  # 解析 .env 文件，返回字典

DB_CONFIG = {
    'host': _env.get('WUCAI_SQL_HOST', 'localhost'),          # MySQL 主机地址，默认本地
    'user': _env.get('WUCAI_SQL_USERNAME', 'root'),           # 数据库用户名
    'password': _env.get('WUCAI_SQL_PASSWORD', ''),           # 数据库密码
    'database': _env.get('WUCAI_SQL_DB', 'wucai_trade'),      # 数据库名
    'port': int(_env.get('WUCAI_SQL_PORT', '3306')),          # 端口号，MySQL默认3306
    'charset': 'utf8mb4'                                      # 字符集，utf8mb4支持emoji和生僻汉字
}

# ============================================================
# 回测参数
# ============================================================

# 初始资金：默认100万元，模拟A股中小散户或小型私募的起步规模
INITIAL_CASH = int(_env.get('BACKTEST_INITIAL_CASH', '1000000'))

# 手续费率：默认万分之二（0.02%），包含佣金和印花税
# A股实际交易成本约为万分之二至万分之三，这里取保守估计
COMMISSION = float(_env.get('BACKTEST_COMMISSION', '0.0002'))

# 单次交易仓位比例：默认95%，即每次交易使用95%的可用资金
# 不满仓是为了保留部分现金应对滑点和意外情况
POSITION_PCT = int(_env.get('BACKTEST_POSITION_PCT', '95'))


def get_connection():
    """
    获取数据库连接

    每次调用都会创建一个新的 PyMySQL 连接。
    为什么不用连接池？因为回测场景下的查询频率较低（每天一次批量查询），
    创建连接的开销可以接受，且避免了连接池管理的复杂性。

    返回:
        pymysql.connection 对象
    """
    return pymysql.connect(**DB_CONFIG)


def execute_query(sql, params=None):
    """
    执行查询语句，返回字典列表

    为什么用 DictCursor？它让每一行结果以字典形式返回，键是列名，
    这样在代码中可以通过 row['column_name'] 访问数据，比数字索引更可读。

    参数:
        sql: SQL 查询语句，支持 %s 占位符
        params: 可选，SQL 参数列表，用于参数化查询防止 SQL 注入

    返回:
        list[dict]，查询结果的每一行作为一个字典
    """
    conn = get_connection()
    # DictCursor 让 fetchall() 返回字典列表而非元组列表
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    cursor.execute(sql, params or ())   # params or () 确保在没有参数时传入空元组
    result = cursor.fetchall()           # 获取所有结果行
    cursor.close()                       # 关闭游标
    conn.close()                         # 关闭连接（及时释放资源）
    return result
