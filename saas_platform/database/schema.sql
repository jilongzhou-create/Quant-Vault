-- ============================================================
-- SaaS 平台云端数据库 Schema (Supabase / PostgreSQL)
-- ============================================================
-- 设计原则：
--   1. 与本地投研 SQLite 物理隔离，云端独立运行
--   2. 本地通过部署脚本推送数据到云端，推送后本地可关机
--   3. 云端 Production Engine 7x24 读取策略代码 + 因子数据 → 算信号 → 下单
--   4. Streamlit 前端只读展示
-- 注意：所有 CREATE INDEX 使用 IF NOT EXISTS，支持重复执行
-- ============================================================

-- ============================================================
-- 一、数据基座表（支撑云端闭环计算）
-- ============================================================

-- 1. 行情量价数据（对应本地 market_data_crypto / market_data_us_stock / market_data_commodity）
CREATE TABLE IF NOT EXISTS saas_market_data (
    symbol      TEXT        NOT NULL,
    timestamp   TIMESTAMPTZ NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      DOUBLE PRECISION,
    rsi_14      DOUBLE PRECISION,
    macd        DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    macd_hist   DOUBLE PRECISION,
    PRIMARY KEY (symbol, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_saas_market_data_symbol_ts ON saas_market_data (symbol, timestamp DESC);

-- 2. 因子数据（对应本地 factor_data，云端计算信号必需）
CREATE TABLE IF NOT EXISTS saas_factor_data (
    id           BIGSERIAL   PRIMARY KEY,
    symbol       TEXT        NOT NULL,
    timestamp    TIMESTAMPTZ NOT NULL,
    factor_name  TEXT        NOT NULL,
    factor_value DOUBLE PRECISION,
    UNIQUE (symbol, timestamp, factor_name)
);

CREATE INDEX IF NOT EXISTS idx_saas_factor_data_lookup ON saas_factor_data (symbol, timestamp DESC, factor_name);

-- 3. 因子元数据（对应本地 factor_metadata，记录因子来源与获取方式）
CREATE TABLE IF NOT EXISTS saas_factor_metadata (
    factor_name  TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    description  TEXT DEFAULT '',
    source       TEXT DEFAULT '',
    source_type  TEXT DEFAULT 'api',
    fetch_config JSONB DEFAULT '{}',
    unit         TEXT DEFAULT '',
    update_freq  TEXT DEFAULT '',
    PRIMARY KEY (factor_name, symbol)
);

-- ============================================================
-- 二、业务与用户表
-- ============================================================

-- 4. 策略表（核心：本地推送上来的策略源码 + 实时仓位状态）
CREATE TABLE IF NOT EXISTS saas_strategies (
    id                     UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    name                   TEXT        NOT NULL UNIQUE,
    description            TEXT        DEFAULT '',
    target_asset           TEXT        NOT NULL DEFAULT 'crypto',
    target_symbol          TEXT        NOT NULL DEFAULT 'BTC_USDT',
    python_code            TEXT        NOT NULL,
    params_json            JSONB       DEFAULT '{}',
    required_factors       JSONB       DEFAULT '[]',
    timeframe              TEXT        DEFAULT '1d',
    current_target_position DOUBLE PRECISION DEFAULT 0,
    status                 TEXT        NOT NULL DEFAULT 'LIVE',
    backtest_sharpe        DOUBLE PRECISION,
    backtest_annualized_return DOUBLE PRECISION,
    backtest_max_drawdown  DOUBLE PRECISION,
    backtest_start_date    DATE,
    backtest_end_date      DATE,
    published_at           TIMESTAMPTZ DEFAULT NOW(),
    created_at             TIMESTAMPTZ DEFAULT NOW(),
    updated_at             TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_saas_strategies_status ON saas_strategies (status);
CREATE INDEX IF NOT EXISTS idx_saas_strategies_target ON saas_strategies (target_symbol, status);

-- 5. 净值曲线（回测 + 实盘统一存储，is_backtest 区分）
CREATE TABLE IF NOT EXISTS saas_equity_curves (
    id           BIGSERIAL   PRIMARY KEY,
    strategy_id  UUID        NOT NULL REFERENCES saas_strategies(id) ON DELETE CASCADE,
    date         DATE        NOT NULL,
    nav_value    DOUBLE PRECISION NOT NULL,
    is_backtest  BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (strategy_id, date, is_backtest)
);

CREATE INDEX IF NOT EXISTS idx_saas_equity_curves_strategy ON saas_equity_curves (strategy_id, date DESC);

-- 6. 每日 AI 洞察（云端 Production Engine 生成）
CREATE TABLE IF NOT EXISTS saas_daily_insights (
    id               BIGSERIAL   PRIMARY KEY,
    strategy_id      UUID        NOT NULL REFERENCES saas_strategies(id) ON DELETE CASCADE,
    date             DATE        NOT NULL,
    ai_analysis_text TEXT        NOT NULL DEFAULT '',
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (strategy_id, date)
);

-- 7. 用户表（Supabase Auth 扩展，存储加密的交易所 API Key）
CREATE TABLE IF NOT EXISTS saas_users (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    auth_user_id        UUID        UNIQUE,
    username            TEXT        NOT NULL UNIQUE,
    email               TEXT        UNIQUE,
    password_hash       TEXT        DEFAULT '',
    encrypted_api_key   TEXT        DEFAULT '',
    encrypted_api_secret TEXT       DEFAULT '',
    exchange            TEXT        DEFAULT 'binance',
    is_active           BOOLEAN     DEFAULT TRUE,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- 8. 用户订阅表（用户跟单绑定）
CREATE TABLE IF NOT EXISTS saas_subscriptions (
    id                    BIGSERIAL   PRIMARY KEY,
    user_id               UUID        NOT NULL REFERENCES saas_users(id) ON DELETE CASCADE,
    strategy_id           UUID        NOT NULL REFERENCES saas_strategies(id) ON DELETE CASCADE,
    allocated_capital_usdt DOUBLE PRECISION NOT NULL DEFAULT 0,
    is_active             BOOLEAN     DEFAULT TRUE,
    subscribed_at         TIMESTAMPTZ DEFAULT NOW(),
    unsubscribed_at       TIMESTAMPTZ,
    UNIQUE (user_id, strategy_id)
);

CREATE INDEX IF NOT EXISTS idx_saas_subscriptions_user ON saas_subscriptions (user_id, is_active);
CREATE INDEX IF NOT EXISTS idx_saas_subscriptions_strategy ON saas_subscriptions (strategy_id, is_active);

-- 9. 交易订单记录（云端代客下单记录）
CREATE TABLE IF NOT EXISTS saas_orders (
    id              BIGSERIAL   PRIMARY KEY,
    user_id         UUID        NOT NULL REFERENCES saas_users(id),
    strategy_id     UUID        NOT NULL REFERENCES saas_strategies(id),
    subscription_id BIGINT      REFERENCES saas_subscriptions(id),
    symbol          TEXT        NOT NULL,
    side            TEXT        NOT NULL,
    order_type      TEXT        DEFAULT 'market',
    amount          DOUBLE PRECISION,
    price           DOUBLE PRECISION,
    fee             DOUBLE PRECISION DEFAULT 0,
    exchange_order_id TEXT      DEFAULT '',
    status          TEXT        DEFAULT 'PENDING',
    error_message   TEXT        DEFAULT '',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_saas_orders_user ON saas_orders (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_saas_orders_strategy ON saas_orders (strategy_id, created_at DESC);

-- ============================================================
-- 三、自动更新 updated_at 触发器
-- ============================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_saas_strategies_updated_at ON saas_strategies;
CREATE TRIGGER trg_saas_strategies_updated_at
    BEFORE UPDATE ON saas_strategies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS trg_saas_users_updated_at ON saas_users;
CREATE TRIGGER trg_saas_users_updated_at
    BEFORE UPDATE ON saas_users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- 四、RLS (Row Level Security) 策略
-- ============================================================
-- 注意：service_role key 绑定后端操作（绕过 RLS），anon key 绑定前端只读
-- 以下 RLS 策略面向 anon key（前端用户），service_role 自动绕过

ALTER TABLE saas_users ENABLE ROW LEVEL SECURITY;
ALTER TABLE saas_subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE saas_orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE saas_strategies ENABLE ROW LEVEL SECURITY;
ALTER TABLE saas_equity_curves ENABLE ROW LEVEL SECURITY;
ALTER TABLE saas_daily_insights ENABLE ROW LEVEL SECURITY;
ALTER TABLE saas_market_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE saas_factor_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE saas_factor_metadata ENABLE ROW LEVEL SECURITY;

-- saas_users: 用户只能读写自己的记录
DROP POLICY IF EXISTS "Users can read own profile" ON saas_users;
CREATE POLICY "Users can read own profile"
    ON saas_users FOR SELECT
    USING (auth_user_id = auth.uid());

DROP POLICY IF EXISTS "Users can update own profile" ON saas_users;
CREATE POLICY "Users can update own profile"
    ON saas_users FOR UPDATE
    USING (auth_user_id = auth.uid());

-- saas_subscriptions: 用户只能读自己的订阅
DROP POLICY IF EXISTS "Users can read own subscriptions" ON saas_subscriptions;
CREATE POLICY "Users can read own subscriptions"
    ON saas_subscriptions FOR SELECT
    USING (user_id IN (SELECT id FROM saas_users WHERE auth_user_id = auth.uid()));

-- saas_orders: 用户只能读自己的订单
DROP POLICY IF EXISTS "Users can read own orders" ON saas_orders;
CREATE POLICY "Users can read own orders"
    ON saas_orders FOR SELECT
    USING (user_id IN (SELECT id FROM saas_users WHERE auth_user_id = auth.uid()));

-- saas_strategies: 公开只读（LIVE/PAPER 状态）
DROP POLICY IF EXISTS "Public read strategies" ON saas_strategies;
CREATE POLICY "Public read strategies"
    ON saas_strategies FOR SELECT
    USING (status IN ('LIVE', 'PAPER'));

-- saas_equity_curves: 公开只读
DROP POLICY IF EXISTS "Public read equity curves" ON saas_equity_curves;
CREATE POLICY "Public read equity curves"
    ON saas_equity_curves FOR SELECT
    USING (TRUE);

-- saas_daily_insights: 公开只读
DROP POLICY IF EXISTS "Public read daily insights" ON saas_daily_insights;
CREATE POLICY "Public read daily insights"
    ON saas_daily_insights FOR SELECT
    USING (TRUE);

-- saas_market_data: 公开只读（前端展示图表需要）
DROP POLICY IF EXISTS "Public read market data" ON saas_market_data;
CREATE POLICY "Public read market data"
    ON saas_market_data FOR SELECT
    USING (TRUE);

-- saas_factor_data: 公开只读
DROP POLICY IF EXISTS "Public read factor data" ON saas_factor_data;
CREATE POLICY "Public read factor data"
    ON saas_factor_data FOR SELECT
    USING (TRUE);

-- saas_factor_metadata: 公开只读
DROP POLICY IF EXISTS "Public read factor metadata" ON saas_factor_metadata;
CREATE POLICY "Public read factor metadata"
    ON saas_factor_metadata FOR SELECT
    USING (TRUE);
