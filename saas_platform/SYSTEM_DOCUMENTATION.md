# SaaS 平台系统文档

## 目录

1. [系统概述](#系统概述)
2. [目录结构](#目录结构)
3. [架构设计](#架构设计)
4. [数据库设计 (Supabase/PostgreSQL)](#数据库设计-supabasepostgresql)
5. [配置模块 (saas_config)](#配置模块-saas_config)
6. [数据库客户端 (supabase_client)](#数据库客户端-supabase_client)
7. [生产引擎 (Production Engine)](#生产引擎-production-engine)
8. [Web 前端 (Streamlit)](#web-前端-streamlit)
9. [发布桥梁 (publish_strategy_to_saas)](#发布桥梁-publish_strategy_to_saas)
10. [安全机制](#安全机制)
11. [工作流程](#工作流程)
12. [开发进度与待完成项](#开发进度与待完成项)

---

## 系统概述

本 SaaS 平台是 **trading_agent 本地投研系统的云端延伸**，核心使命是将本地挖掘出的优质策略组合"一键上云"，为用户提供：

1. **策略广场** — 公开展示上线策略的回测与模拟实盘净值曲线，用户可浏览策略表现
2. **跟单系统** — 用户绑定 Binance API Key 后，订阅策略并分配资金，系统自动代客下单
3. **云端闭环** — Production Engine 7×24 运行，自动拉取数据 → 计算信号 → 更新净值 → 执行跟单

### 核心设计原则

| 原则 | 说明 |
|------|------|
| 本地/云端物理隔离 | 本地投研系统使用 SQLite，云端 SaaS 使用 Supabase (PostgreSQL)，两者互不干扰 |
| 推送式同步 | 本地通过 `publish_strategy_to_saas.py` 将策略推送到云端，推送后本地可关机 |
| 纯读前端 | Streamlit 前端只从 Supabase 读取数据，禁止重计算 |
| API Key 加密存储 | 用户 Binance API Key 使用 Fernet 对称加密后存储，绝不存明文 |
| 沙盒优先 | 跟单路由默认开启 Binance 测试网沙盒模式，防止误操作 |

---

## 目录结构

```text
saas_platform/
├── __init__.py                   ← 空模块标记
├── saas_config.py                ← 统一配置网关（独立于本地 config.py）
├── SYSTEM_DOCUMENTATION.md       ← 本文档
│
├── database/                     ← 云端数据库层
│   ├── __init__.py
│   ├── schema.sql                ← Supabase DDL（9 张表 + RLS + 触发器）
│   └── supabase_client.py        ← REST API 直连客户端（单例模式）
│
├── production_engine/            ← 云端生产引擎
│   ├── __init__.py
│   ├── daily_job.py              ← 每日定时任务入口（Step1→Step2→Step3）
│   ├── data_fetcher.py           ← 增量数据拉取器（行情+因子→Supabase）
│   ├── signal_engine.py          ← 信号与净值引擎（沙盒执行策略代码）
│   └── copy_trading_router.py    ← 跟单执行路由（代客下单）
│
└── web_frontend/                 ← Streamlit 前端
    ├── __init__.py
    ├── app.py                    ← 单页面应用主入口
    └── crypto_utils.py           ← API Key 加密/解密工具
```

### 外部关联文件

```text
ops_scripts/
└── publish_strategy_to_saas.py   ← 本地→云端发布桥梁（本地投研目录中唯一与 SaaS 交互的脚本）
```

---

## 架构设计

### 整体架构图

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        本地投研系统 (SQLite)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ 策略挖掘  │  │ 回测引擎  │  │ 组合构建  │  │ 内容引擎         │   │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘  └──────────────────┘   │
│        └──────────────┴──────────────┘                              │
│                              │                                      │
│              publish_strategy_to_saas.py                             │
│              (一键推送策略+净值到云端)                                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     云端 SaaS 平台 (Supabase)                        │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                  Production Engine (7×24)                    │    │
│  │                                                              │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │    │
│  │  │ data_fetcher │→│ signal_engine │→│copy_trading_router│  │    │
│  │  │ (拉取数据)    │  │ (计算信号)    │  │ (代客下单)        │  │    │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘  │    │
│  │         ↑                  ↑                    │            │    │
│  │         │                  │                    ▼            │    │
│  │    Binance/FMP/FRED   沙盒执行策略代码    Binance API        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   Supabase (PostgreSQL)                      │    │
│  │  saas_market_data │ saas_factor_data │ saas_strategies       │    │
│  │  saas_equity_curves │ saas_daily_insights │ saas_users       │    │
│  │  saas_subscriptions │ saas_orders │ saas_factor_metadata    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                  Streamlit 前端 (只读展示)                    │    │
│  │  策略广场 │ 用户注册/登录 │ API绑定 │ 策略订阅 │ 订单记录    │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

### 数据流图

```text
每日定时任务 (daily_job.py) 执行流程：

  Step 1: data_fetcher.run_daily_sync()
    ├── Binance REST API → BTC/USDT 日线 OHLCV → saas_market_data
    ├── FMP API → SPY/QQQ/GCUSD/BZUSD 日线 → saas_market_data
    ├── FRED API → 12 个核心宏观指标 → saas_factor_data
    ├── CoinMetrics API → MVRV/链上指标 → saas_factor_data
    ├── Binance Futures API → 资金费率 → saas_factor_data
    └── Alternative.me → 恐惧贪婪指数 → saas_factor_data

  Step 2: signal_engine.run_daily_signal()
    ├── 从 saas_market_data + saas_factor_data 拼装宽表
    ├── 遍历 LIVE 策略 → 沙盒执行 python_code → target_position
    ├── 更新 saas_strategies.current_target_position
    └── 计算模拟净值 → saas_equity_curves (is_backtest=False)

  Step 3: copy_trading_router.run_copy_trading()
    ├── 遍历 LIVE 策略的活跃订阅
    ├── 解密用户 API Key → 连接 Binance
    ├── 计算目标持仓 vs 实际持仓 → 市价单补差
    └── 记录订单 → saas_orders
```

---

## 数据库设计 (Supabase/PostgreSQL)

云端使用 Supabase 托管的 PostgreSQL 数据库，与本地 SQLite 物理隔离。共 9 张业务表。

### 表总览

| # | 表名 | 说明 | 唯一约束 |
|---|------|------|----------|
| 1 | saas_market_data | 行情量价数据（全资产统一） | (symbol, timestamp) |
| 2 | saas_factor_data | 因子数据（云端计算信号必需） | (symbol, timestamp, factor_name) |
| 3 | saas_factor_metadata | 因子元数据（来源、获取方式） | (factor_name, symbol) |
| 4 | saas_strategies | 策略表（核心：源码+仓位+回测指标） | name (UNIQUE) |
| 5 | saas_equity_curves | 净值曲线（回测+实盘统一，is_backtest 区分） | (strategy_id, date, is_backtest) |
| 6 | saas_daily_insights | 每日 AI 洞察 | (strategy_id, date) |
| 7 | saas_users | 用户表（含加密 API Key） | username (UNIQUE) |
| 8 | saas_subscriptions | 用户订阅表（跟单绑定） | (user_id, strategy_id) |
| 9 | saas_orders | 交易订单记录 | id (BIGSERIAL PK) |

### 核心表结构详解

#### 1. saas_market_data — 行情量价数据

| 字段 | 类型 | 说明 |
|------|------|------|
| symbol | TEXT | 交易对符号 (BTC_USDT, SPY, GCUSD 等) |
| timestamp | TIMESTAMPTZ | 时间戳 |
| open/high/low/close | DOUBLE PRECISION | OHLC |
| volume | DOUBLE PRECISION | 成交量 |
| rsi_14 | DOUBLE PRECISION | 14日RSI |
| macd / macd_signal / macd_hist | DOUBLE PRECISION | MACD 指标 |

**主键**: (symbol, timestamp)

#### 2. saas_strategies — 策略表（核心）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 (自动生成) |
| name | TEXT | 策略名称 (UNIQUE) |
| description | TEXT | 策略描述 |
| target_asset | TEXT | 资产大类 (crypto/us_stock/commodity) |
| target_symbol | TEXT | 具体标的 (BTC_USDT/SPY/GCUSD 等) |
| python_code | TEXT | 策略代码（单策略为 Python 代码，组合为 JSON 数组） |
| params_json | JSONB | 参数（组合策略含 weight_mode） |
| required_factors | JSONB | 所需因子列表 |
| timeframe | TEXT | 时间周期 (1d/4h/1h) |
| current_target_position | DOUBLE PRECISION | 当前目标仓位 [-1, 1] |
| status | TEXT | 状态 (LIVE/PAPER) |
| backtest_sharpe / backtest_annualized_return / backtest_max_drawdown | DOUBLE PRECISION | 回测指标 |
| backtest_start_date / backtest_end_date | DATE | 回测日期范围 |
| published_at / created_at / updated_at | TIMESTAMPTZ | 时间戳 |

**python_code 格式说明**：
- 单策略：纯 Python 代码字符串，包含 `generate_signals(df)` 函数
- 组合策略：JSON 数组，每个元素包含 `dir_id`, `name`, `code`, `params`, `sharpe`

#### 3. saas_users — 用户表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| auth_user_id | UUID | Supabase Auth 关联 ID |
| username | TEXT | 用户名 (UNIQUE) |
| email | TEXT | 邮箱 (UNIQUE) |
| encrypted_api_key | TEXT | Fernet 加密后的 Binance API Key |
| encrypted_api_secret | TEXT | Fernet 加密后的 Binance API Secret |
| exchange | TEXT | 交易所名称 (默认 binance) |
| is_active | BOOLEAN | 是否活跃 |

#### 4. saas_subscriptions — 订阅表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGSERIAL | 主键 |
| user_id | UUID | 外键 → saas_users |
| strategy_id | UUID | 外键 → saas_strategies |
| allocated_capital_usdt | DOUBLE PRECISION | 分配资金 (USDT) |
| is_active | BOOLEAN | 是否活跃 |
| subscribed_at / unsubscribed_at | TIMESTAMPTZ | 订阅/退订时间 |

### RLS (Row Level Security) 策略

| 表 | 策略 | 规则 |
|------|------|------|
| saas_users | Users can read/update own profile | auth_user_id = auth.uid() |
| saas_subscriptions | Users can read own subscriptions | 通过子查询匹配 auth_user_id |
| saas_orders | Users can read own orders | 通过子查询匹配 auth_user_id |
| saas_strategies | Public read strategies | status = 'LIVE' 时公开可读 |
| saas_equity_curves | Public read equity curves | 全部公开可读 |
| saas_daily_insights | Public read daily insights | 全部公开可读 |

### 自动更新触发器

`updated_at` 字段在 `saas_strategies` 和 `saas_users` 表上通过 `update_updated_at_column()` 触发器自动维护。

---

## 配置模块 (saas_config)

**文件**: `saas_platform/saas_config.py`

### 设计原则

1. **独立配置网关** — 所有 SaaS 模块禁止直接 `os.getenv`，统一通过本模块读取
2. **双模式兼容** — 本地开发从 `.env` 加载，云端部署直接读环境变量
3. **SAAS_ 前缀映射** — 优先读 `SAAS_` 前缀变量，fallback 到无前缀版本
4. **严格隔离** — 严禁引入本地投研系统的 `DB_PATH` / `BINANCE_API_KEY` 等配置

### 配置项

| 变量 | 环境变量 | 说明 |
|------|---------|------|
| SAAS_SUPABASE_URL | SAAS_SUPABASE_URL | Supabase 项目 URL |
| SAAS_SUPABASE_KEY | SAAS_SUPABASE_KEY | Supabase service_role key |
| FRED_API_KEY | SAAS_FRED_API_KEY / FRED_API_KEY | FRED 宏观数据 API Key |
| FMP_API_KEY | SAAS_FMP_API_KEY / FMP_API_KEY | FMP 美股数据 API Key |
| COINMETRICS_API_KEY | SAAS_COINMETRICS_API_KEY / COINMETRICS_API_KEY | CoinMetrics 链上数据 API Key |
| GEMINI_API_KEY | SAAS_GEMINI_API_KEY / GEMINI_API_KEY | Google Gemini API Key |
| MASTER_KEY | SAAS_MASTER_KEY / MASTER_KEY | Fernet 加密主密钥 |
| PROXY_MODE | SAAS_PROXY_MODE / PROXY_MODE | 代理模式 (direct/proxy) |
| PROXY_URL | SAAS_PROXY_URL / HTTPS_PROXY | 代理地址 |
| SAAS_ENV | SAAS_ENV | 运行环境 (development/production) |

### 工具函数

| 函数 | 说明 |
|------|------|
| `get_saas_config(key, default)` | 通用配置读取（自动 SAAS_ 前缀映射） |
| `is_configured()` | 检查 Supabase 是否已配置 |
| `get_config_summary()` | 返回配置状态摘要（隐藏敏感值），用于启动日志 |

---

## 数据库客户端 (supabase_client)

**文件**: `saas_platform/database/supabase_client.py`

### 设计特点

- **REST API 直连** — 通过 PostgREST API 直接操作 Supabase，无需安装 `supabase-py`
- **单例模式** — `SupabaseClient` 全局唯一实例，通过 `get_client()` 获取
- **仅依赖 requests** — 兼容所有 Python 版本

### 核心类: SupabaseClient

| 方法 | 说明 |
|------|------|
| `select(table, columns, filters, order, limit)` | 查询记录 |
| `insert(table, records)` | 插入记录 |
| `upsert(records, table, on_conflict)` | 插入或更新（支持冲突字段指定） |
| `update(table, filters, data)` | 更新记录 |
| `delete(table, filters)` | 删除记录 |
| `rpc(function_name, params)` | 调用 Supabase RPC 函数 |

### 业务接口函数

#### 行情数据

| 函数 | 说明 |
|------|------|
| `upsert_market_data(records)` | 批量写入行情数据 |
| `get_market_data(symbol, limit, order)` | 查询行情数据 |
| `get_latest_market_timestamp(symbol)` | 获取最新行情时间戳 |

#### 因子数据

| 函数 | 说明 |
|------|------|
| `upsert_factor_data(records)` | 批量写入因子数据 |
| `get_factor_data(symbol, factor_names, limit, order)` | 查询因子数据 |
| `upsert_factor_metadata(records)` | 批量写入因子元数据 |

#### 策略

| 函数 | 说明 |
|------|------|
| `publish_strategy(strategy_data)` | 发布策略（INSERT） |
| `upsert_strategy(strategy_data)` | 插入或更新策略 |
| `get_strategy_code(strategy_id)` | 获取策略代码（仅 LIVE 状态） |
| `get_live_strategies(target_symbol)` | 获取活跃策略列表 |
| `update_strategy_position(strategy_id, position)` | 更新策略目标仓位 |

#### 净值曲线

| 函数 | 说明 |
|------|------|
| `upsert_equity_curve(records)` | 批量写入净值 |
| `bulk_upsert_equity_curves(curves_data_list, batch_size)` | 分批批量写入（默认 500 条/批） |
| `update_daily_nav(strategy_id, date, nav_value, is_backtest)` | 更新单日净值 |
| `get_equity_curve(strategy_id, is_backtest, limit)` | 查询净值曲线 |

#### 用户与订阅

| 函数 | 说明 |
|------|------|
| `create_user(user_data)` | 创建用户 |
| `get_user_by_auth_id(auth_user_id)` | 按 Auth ID 查询用户 |
| `get_user_by_username(username)` | 按用户名查询用户 |
| `update_user_api_keys(user_id, enc_key, enc_secret, exchange)` | 更新加密 API Key |
| `create_subscription(user_id, strategy_id, allocated_capital_usdt)` | 创建订阅 |
| `get_active_subscriptions(strategy_id)` | 获取活跃订阅（含用户加密 Key） |
| `deactivate_subscription(subscription_id)` | 退订 |
| `get_user_subscriptions(user_id)` | 获取用户订阅列表 |

#### 订单

| 函数 | 说明 |
|------|------|
| `create_order(order_data)` | 创建订单记录 |
| `update_order_status(order_id, status, exchange_order_id, error_message)` | 更新订单状态 |
| `get_user_orders(user_id, limit)` | 获取用户订单列表 |

#### 前端展示

| 函数 | 说明 |
|------|------|
| `get_public_strategies()` | 获取公开策略列表（按夏普率降序） |
| `get_strategy_equity_curve(strategy_id, is_backtest, limit)` | 获取策略净值曲线 |

#### AI 洞察

| 函数 | 说明 |
|------|------|
| `upsert_daily_insight(strategy_id, date, ai_analysis_text)` | 写入每日 AI 洞察 |
| `get_daily_insights(strategy_id, limit)` | 获取 AI 洞察列表 |

---

## 生产引擎 (Production Engine)

### 1. daily_job.py — 每日定时任务入口

**执行流程**:

```text
Step 1: data_fetcher.run_daily_sync()     — 增量拉取行情+因子数据
Step 2: signal_engine.run_daily_signal()   — 计算策略信号+模拟净值
Step 3: copy_trading_router.run_copy_trading() — 跟单执行路由
```

**部署方式**:
- Linux cron: `0 8 * * * cd /path/to/project && python -m saas_platform.production_engine.daily_job`
- Windows Task Scheduler: 每日 08:00 执行
- 云函数 / GitHub Actions: 定时触发

### 2. data_fetcher.py — 云端全域数据拉取器

**严格约束**:
- 禁止全量拉取，仅拉取最近 400 天数据
- 绝对禁止写入本地 SQLite，数据落库终点必须是 Supabase
- 复用现有 API 请求逻辑，但下游终点重定向至 Supabase

**数据源一览**:

| 数据源 | 函数 | 写入表 | 说明 |
|--------|------|--------|------|
| Binance REST API | `fetch_crypto_market()` | saas_market_data | BTC/USDT 日线，不依赖 ccxt |
| FMP API | `fetch_fmp_market()` | saas_market_data | SPY/QQQ/GCUSD/BZUSD 日线 |
| FRED API | `fetch_fred_factors()` | saas_factor_data | 12 个核心宏观指标 |
| CoinMetrics API | `fetch_coinmetrics_factors()` | saas_factor_data | MVRV/链上指标 |
| Binance Futures API | `fetch_funding_rate()` | saas_factor_data | BTC 资金费率 |
| Alternative.me | `fetch_fear_greed()` | saas_factor_data | 恐惧贪婪指数 |

**FRED 核心指标 (12个)**:
M2SL, WALCL, FEDFUNDS, CPIAUCSL, UNRATE, T10Y2Y, DGS10, BAMLH0A0HYM2, VIXCLS, WTISPLC, STLFSI4, NFCI

**标的配置 (SYMBOL_CONFIG)**:

| 资产类别 | 标的 | 数据源 |
|---------|------|--------|
| crypto | BTC/USDT | Binance REST API |
| us_stock | SPY, QQQ | FMP API |
| commodity | GCUSD, BZUSD | FMP API |

### 3. signal_engine.py — 云端信号与净值引擎

**核心类**: `CloudSignalEngine`

**执行流程**:

1. **获取活跃策略** — 从 `saas_strategies` 查询 status=LIVE/PAPER 的策略
2. **按标的分组** — 同标的策略复用宽表数据
3. **拼装宽表** — 从 `saas_market_data` + `saas_factor_data` 拉取数据，`merge_asof` 向后对齐
4. **沙盒执行** — `exec()` 执行策略 `python_code`，获取 `target_position`
5. **净值计算** — 基于仓位变化和日收益率，扣除滑点(5bps)+手续费(10bps)
6. **回写 Supabase** — 更新 `current_target_position` 和 `saas_equity_curves`

**策略代码执行**:
- 单策略：执行 `generate_signals(df)` → 返回 `target_position`
- 组合策略：解析 JSON 数组，逐个执行子策略，按 `weight_mode` (equal/sharpe) 加权汇总

**异常隔离**: 单策略报错不崩溃，记录日志跳过

**净值计算参数**:

| 参数 | 值 | 说明 |
|------|---|------|
| SLIPPAGE_BPS | 5 | 滑点 (基点) |
| COMMISSION_BPS | 10 | 手续费 (基点) |
| INITIAL_NAV | 1000.0 | 初始净值 |
| MAX_LOOKBACK_DAYS | 400 | 最大回溯天数 |

### 4. copy_trading_router.py — 云端跟单执行路由

**核心类**: `CopyTradingRouter`

**执行流程**:

1. 遍历所有 LIVE 策略
2. 对每个策略，获取活跃订阅用户列表
3. 对每个用户：解密 API Key → 连接交易所 → 计算目标持仓 → 下单
4. 记录订单到 `saas_orders`

**安全防线**:

| 参数 | 值 | 说明 |
|------|---|------|
| SANDBOX_MODE | True | 默认开启 Binance 测试网 |
| MIN_NOTIONAL_USDT | 11.0 | 最小交易金额 (USDT) |
| WEIGHT_TOLERANCE | 0.03 | 仓位偏离容忍度 (3%) |
| MIN_BTC_AMOUNT | 0.001 | 最小 BTC 精度 |
| MAX_SLIPPAGE_PCT | 0.005 | 最大滑点容忍 |

**交易所连接**:
- 使用 `ccxt` 库创建交易所实例
- 默认使用 Binance 合约 (defaultType='future')
- 强制时钟同步 (`adjustForTimeDifference=True`)
- 3 次重试机制（时钟同步、价格获取）

---

## Web 前端 (Streamlit)

**文件**: `saas_platform/web_frontend/app.py`

### 页面结构

| 页面 | 路由 | 功能 | 权限 |
|------|------|------|------|
| 策略广场 | strategy_square | 浏览所有上线策略的回测+实盘净值曲线 | 公开 |
| 登录 | login | 用户名+密码登录 | 公开 |
| 注册 | register | 用户名+邮箱+密码注册 | 公开 |
| 控制台 | my_console | API 绑定 + 策略订阅 + 持仓概览 | 登录 |
| 订单记录 | my_orders | 查看跟单订单历史 | 登录 |

### 控制台三个 Tab

1. **🔐 API 绑定** — 绑定/更新 Binance API Key（Fernet 加密存储）
2. **📌 策略订阅** — 查看当前订阅 + 新增订阅（选择策略+分配资金）
3. **💼 持仓概览** — 查看各订阅策略的目标仓位和名义敞口

### 图表

- 使用 Plotly 绘制净值曲线（暗色主题 `plotly_dark`）
- 回测净值和模拟实盘净值分 Tab 展示

### 认证系统

- 当前使用简单的 SHA256 密码哈希 + Supabase 存储
- Session state 管理登录状态
- **注意**: 当前认证系统较简陋，生产环境需升级为 Supabase Auth

---

## 发布桥梁 (publish_strategy_to_saas)

**文件**: `ops_scripts/publish_strategy_to_saas.py`

这是本地投研目录中**唯一允许与 SaaS 数据库交互的脚本**。

### 推送内容

| Step | 内容 | 目标表 |
|------|------|--------|
| Step 1 | 提取本地组合元数据 | — |
| Step 2 | 提取子策略源码 | — |
| Step 3 | 策略元数据 + 子策略源码 | saas_strategies |
| Step 4 | 历史回测净值曲线 | saas_equity_curves |
| Step 5 | AI 诊断洞察 | saas_daily_insights (待开发) |

### 使用方式

```bash
cd /path/to/trading_agent
python -m ops_scripts.publish_strategy_to_saas
```

交互式选择要发布的组合，一键推送到云端。

### 数据映射

| 本地 SQLite | 云端 Supabase |
|-------------|---------------|
| portfolios | saas_strategies (组合→策略) |
| portfolio_components + strategy_directions + strategy_versions | saas_strategies.python_code (JSON 数组) |
| portfolio_daily_records (BACKTEST) | saas_equity_curves (is_backtest=True) |

---

## 安全机制

### 1. API Key 加密

- 使用 `cryptography.fernet` 对称加密
- MASTER_KEY 来自环境变量，支持 Fernet 原生 Key 或 SHA256 派生 Key
- 加密后存储到 `saas_users.encrypted_api_key` / `encrypted_api_secret`
- 解密仅在跟单执行时使用，日志中绝不输出明文

### 2. Supabase RLS

- 用户表、订阅表、订单表启用 Row Level Security
- 用户只能读写自己的数据
- 策略和净值公开只读

### 3. 跟单安全

- 默认 Binance 测试网沙盒模式
- 3% 仓位偏离容忍阈值（防止过度交易）
- 最小交易金额检查 (11 USDT)
- 单用户异常不影响其他用户

### 4. 策略沙盒执行

- `exec()` 使用受控 `globals/locals`
- 仅暴露 `pd`, `np` 和 `__builtins__`
- 单策略异常隔离，不影响其他策略

---

## 工作流程

### 策略上云全流程

```text
1. 本地策略挖掘：
   ├── auto_miner / auto_portfolio_agent 挖掘策略 → 组合构建
   ├── portfolio_backtest 组合回测验证
   └── 组合状态提升至 TESTED/PAPER/LIVE

2. 一键发布：
   ├── 运行 publish_strategy_to_saas.py
   ├── 交互式选择组合
   ├── 推送策略源码 + 回测净值到 Supabase
   └── 前端即可展示策略和净值曲线

3. 云端每日运行：
   ├── daily_job.py 每日定时执行
   ├── Step 1: 增量拉取最新数据
   ├── Step 2: 计算信号 + 模拟净值
   └── Step 3: 跟单执行（如有订阅用户）

4. 用户交互：
   ├── 用户注册/登录
   ├── 绑定 Binance API Key
   ├── 浏览策略广场 → 订阅策略 → 分配资金
   └── 系统自动代客下单 → 订单记录可查
```

---

## 开发进度与待完成项

### ✅ 已完成

| 模块 | 状态 | 说明 |
|------|------|------|
| saas_config.py | ✅ 完成 | 统一配置网关，SAAS_ 前缀映射 |
| schema.sql | ✅ 完成 | 9 张表 + RLS + 触发器 |
| supabase_client.py | ✅ 完成 | REST API 直连，完整 CRUD |
| data_fetcher.py | ✅ 完成 | 6 个数据源增量拉取 |
| signal_engine.py | ✅ 完成 | 沙盒执行 + 净值计算 |
| copy_trading_router.py | ✅ 完成 | 跟单路由 + 安全防线 |
| daily_job.py | ✅ 完成 | 三步定时任务入口 |
| app.py (Streamlit) | ✅ 完成 | 策略广场 + 用户系统 + 控制台 |
| crypto_utils.py | ✅ 完成 | Fernet 加密/解密 |
| publish_strategy_to_saas.py | ✅ 完成 | 本地→云端发布桥梁 |

### ⚠️ 待完成 / 需改进

| 模块 | 问题 | 优先级 |
|------|------|--------|
| 用户认证 | 当前使用简单 SHA256 哈希，未接入 Supabase Auth；注册时 `password_hash` 字段不在 schema.sql 中 | 高 |
| schema.sql | `saas_users` 表缺少 `password_hash` 字段，前端注册写入了该字段但数据库无此列 | 高 |
| 前端部署 | 无 Streamlit 部署配置（无 Dockerfile、无 .streamlit/config.toml） | 高 |
| 定时任务部署 | 无 cron / 云函数部署配置 | 高 |
| AI 洞察 | Step 5 (AI 诊断洞察) 标记为"待后续开发"，未实现 | 中 |
| 前端 UI | 订阅面板中策略 ID 截断显示不友好，缺少策略名称展示 | 中 |
| data_fetcher 分页 | `get_market_data` 分页逻辑有 bug（offset 未使用，直接 break） | 中 |
| 环境变量 | .env 中缺少 SAAS_ 前缀的变量模板 | 中 |
| 通知系统 | 无用户通知（下单成功/失败通知） | 低 |
| 风控模块 | 无独立风控层（最大回撤熔断、单日亏损限制等） | 低 |
| 日志监控 | 无集中式日志收集和告警 | 低 |
