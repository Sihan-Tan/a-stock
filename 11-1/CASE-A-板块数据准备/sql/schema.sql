-- ============================================================
-- 21-CASE-A 板块数据准备 - 表结构
-- MySQL 8.0 | 字符集 utf8mb4
-- ============================================================
--
-- 3 张表:
--   trade_stock_status   股票综合状态 (含 sector_1/sector_2 申万分类)
--   trade_stock_daily    个股日 K (前复权)
--   trade_sector_daily   板块每日聚合 + 等权合成的板块指数 OHLC
--
-- 申万分类策略:
--   本 CASE 主用 sector_2 (申万二级, 134 个), 因为板块轮动用二级信号更敏感
--   sector_1 (一级, 31 个) 也同时落库
-- ============================================================


-- ------------------------------------------------------------
-- 1. 股票综合状态表 (含申万分类)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trade_stock_status (
    stock_code     VARCHAR(20) NOT NULL PRIMARY KEY,
    stock_name     VARCHAR(50),
    list_date      DATE        COMMENT '上市日期',
    total_shares   BIGINT      COMMENT '总股本(股)',
    float_shares   BIGINT      COMMENT '流通股本(股)',
    sector_1       VARCHAR(30) COMMENT '申万一级行业',
    sector_2       VARCHAR(50) COMMENT '申万二级行业',
    sector_3       VARCHAR(50) COMMENT '申万三级行业',
    updated_at     DATETIME    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    KEY idx_status_sector_1 (sector_1),
    KEY idx_status_sector_2 (sector_2)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='股票综合状态(含申万分类)';


-- ------------------------------------------------------------
-- 2. 个股日 K 线 (前复权)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trade_stock_daily (
    id             INT AUTO_INCREMENT PRIMARY KEY,
    stock_code     VARCHAR(20) NOT NULL,
    trade_date     DATE        NOT NULL,
    open_price     DECIMAL(10,2) COMMENT '开盘价',
    high_price     DECIMAL(10,2),
    low_price      DECIMAL(10,2),
    close_price    DECIMAL(10,2) COMMENT '收盘价(前复权)',
    volume         BIGINT      COMMENT '成交量(股)',
    amount         DECIMAL(20,2) COMMENT '成交额(元)',
    turnover_rate  DECIMAL(10,4) COMMENT '换手率(%)',
    created_at     DATETIME    DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY idx_stock_daily_code_date (stock_code, trade_date),
    KEY idx_stock_daily_code (stock_code),
    KEY idx_stock_daily_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='个股日K线';


-- ------------------------------------------------------------
-- 3. 板块每日聚合 (含合成的指数 OHLC)
-- ------------------------------------------------------------
-- 板块指数合成口径:
--   起始基期 close_idx = 1000.0
--   close_idx_t = close_idx_(t-1) * (1 + 当日成份股 close 等权平均收益率)
--   open_idx_t  = close_idx_(t-1) * (1 + 当日成份股 open  等权平均收益率)
--   high_idx_t  = close_idx_(t-1) * (1 + 当日成份股 high  等权平均收益率)
--   low_idx_t   = close_idx_(t-1) * (1 + 当日成份股 low   等权平均收益率)
--
-- 为什么用累乘而不是 close 直接归一化均值:
--   - 累乘对停牌/退市等成分股变动更鲁棒 (前一日 close_idx 已经吸收了)
--   - 业内多数行业指数 (中证, 申万) 都用类似累乘思路
CREATE TABLE IF NOT EXISTS trade_sector_daily (
    sector_name        VARCHAR(50) NOT NULL COMMENT '板块名称(申万)',
    sector_level       TINYINT     NOT NULL DEFAULT 2 COMMENT '1=一级 2=二级',
    trade_date         DATE        NOT NULL,
    change_pct         DECIMAL(8,4)  COMMENT '当日涨幅(%, 成份股均值)',
    stock_count        INT           COMMENT '成份股数',
    rise_count         INT           COMMENT '上涨家数',
    fall_count         INT           COMMENT '下跌家数',
    flat_count         INT           DEFAULT 0 COMMENT '平盘家数',
    limit_up           INT           DEFAULT 0 COMMENT '涨停家数',
    limit_down         INT           DEFAULT 0 COMMENT '跌停家数',
    top_stock          VARCHAR(20)   COMMENT '领涨股代码',
    top_stock_name     VARCHAR(50)   COMMENT '领涨股名称',
    top_stock_pct      DECIMAL(8,2)  COMMENT '领涨股涨幅(%)',
    open_idx           DECIMAL(12,4) COMMENT '等权合成开盘指数',
    high_idx           DECIMAL(12,4) COMMENT '等权合成最高指数',
    low_idx            DECIMAL(12,4) COMMENT '等权合成最低指数',
    close_idx          DECIMAL(12,4) COMMENT '等权合成收盘指数',
    total_volume       BIGINT        COMMENT '板块总成交量',
    total_amount       DECIMAL(22,2) COMMENT '板块总成交额',
    avg_turnover       DECIMAL(8,4)  COMMENT '平均换手率(%)',
    kline_stock_count  INT           COMMENT '当日参与合成的有效成分股数',
    PRIMARY KEY (sector_name, sector_level, trade_date),
    KEY idx_sector_daily_date  (trade_date),
    KEY idx_sector_daily_level (sector_level, trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='板块每日聚合 + 合成指数';
