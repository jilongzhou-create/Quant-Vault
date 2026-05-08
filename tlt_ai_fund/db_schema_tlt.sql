-- ============================================================
-- TLT AI Fund - 数据库建表 SQL
-- 市场数据表: market_data_tlt
-- ============================================================

-- TLT 市场数据表 (复权收盘价 adjClose 为计算基准)
CREATE TABLE IF NOT EXISTS market_data_tlt (
    symbol          TEXT        NOT NULL,
    timestamp       DATETIME    NOT NULL,
    date            TEXT        NOT NULL,
    open            REAL,
    high            REAL,
    low             REAL,
    close           REAL,
    adj_close       REAL        NOT NULL,
    volume          REAL,
    rsi_14          REAL,
    macd            REAL,
    macd_signal     REAL,
    macd_hist       REAL,
    PRIMARY KEY (symbol, timestamp)
);

-- 创建索引以加速时间序列查询
CREATE INDEX IF NOT EXISTS idx_tlt_timestamp ON market_data_tlt (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_tlt_date ON market_data_tlt (date);
