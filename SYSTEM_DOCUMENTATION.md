# 交易策略系统文档

## 目录

1. [系统概述](#系统概述)
2. [目录结构](#目录结构)
3. [数据库设计](#数据库设计)
4. [核心模块](#核心模块)
5. [数据管线 (Data Pipeline)](#数据管线-data-pipeline)
6. [交易引擎 (Trading Engine)](#交易引擎-trading-engine)
7. [内容引擎 (Content Engine)](#内容引擎-content-engine)
8. [AI 代理 (Agents)](#ai-代理-agents)
9. [运维脚本 (Ops Scripts)](#运维脚本-ops-scripts)
10. [草稿区 (Draft Zone)](#草稿区-draft-zone)
11. [工作流程](#工作流程)

---

## 系统概述

本系统是一个**全自动、AI驱动的多资产量化对冲基金管理平台**（Multi-Asset AI Hedge Fund）。系统覆盖了从"投研灵感获取"到"实盘资金运作"的完整量化投资生命周期，核心使命是**用 AI 替代传统量化研究员的重复劳动，实现策略研发、数据工程、组合构建与交易执行的全链路自动化**。

### 核心价值

1. **消除量化研究的人力瓶颈**
   传统量化研究高度依赖人工阅读研报、提取因子、编写策略、回测调参。本系统内置多个专门的 AI Agent，能够自主阅读外网研报与新闻，提取市场观点并转化为交易逻辑，自动生成策略代码、执行回测、多轮调参，将优质策略沉淀入库——从"灵感"到"可交易策略"无需人工干预。

2. **多维数据融合的 ELT 数据工厂**
   打破仅依赖"量价数据"的局限。系统采用现代 ELT（Extract-Load-Transform）架构，深度整合六大类另类数据：Binance 高频量价与资金费率、CoinMetrics 链上基本面（MVRV）、恐惧贪婪指数、Deribit 隐含波动率、DeFiLlama 稳定币流向、美联储 200+ 宏观经济指标（FRED）。同时为美股市场建立了独立的数据管线，对接 FMP API 获取 OHLCV、财报电话会议、盈利预期差等深度数据。为大宗商品（原油、黄金）建立了专属管线，对接 FMP API 获取 WTI/布伦特原油与黄金的日线量价数据。所有数据先无损落湖（raw_data），再按需提取为结构化因子（factor_data），支持无限扩展。

3. **多策略组合对冲与智能风控**
   单一策略无法穿越牛熊。系统采用多策略对冲基金架构，支持将不同周期、不同逻辑（趋势跟随、均值回归、跨界套利）的策略组合在一起，通过多策略的弱相关性平滑收益曲线。提供等权分配、夏普加权、满仓缩放等多种资金分配模式，实现稳定、可持续的超额收益。

4. **从回测到实盘的完整闭环**
   提供组合状态流转引擎（DRAFT -> TESTED -> PAPER -> LIVE），具备独立的订单追踪与实盘每日对账单管理能力。实盘执行模块内置多层安全防线：强制时钟同步消灭 -1021 错误、SSL 断连自动重试、3% 仓位偏离度容忍阈值防止过度交易、10 秒防误触倒计时、环境变量控制的实盘开关，确保真金白银运作的绝对安全。

5. **统一多资产数据库架构**
   系统同时支持加密货币（BTC/USDT）、美股（TSLA、PLTR 等）和大宗商品（原油、黄金）三大资产类别，通过统一的数据库管理器（db_manager.py）和动态表名机制（`market_data_<asset_class>`）实现数据隔离，同时合并 raw_data / factor_data 为统一表消除数据孤岛，宏观因子约定 symbol='MACRO' 实现跨资产因子共享。

6. **AI 驱动的宏观归因内容引擎**
   内容引擎不再输出干瘪的数据罗列，而是以"顶级宏观对冲基金经理"视角进行真实宏观归因分析（Macro Attribution Analysis）。系统自动提取每笔交易的 Entry/Exit 时间点与单次收益，结合最优参数，由 LLM 判断策略信号是否真正捕捉到了宏观状态切换（Regime Shift），对成功交易和假信号进行深度历史复盘，输出极具洞察力的结构化分析报告。

7. **DEV/PROD 双环境安全沙盒**
   通过 `.env` 中的 `SYSTEM_ENV` 动态切换数据库。PROD 环境写入生产库（INFO 级日志），DEV 环境写入开发沙盒库（DEBUG 级日志），所有新编写的草稿代码均可直接在 DEV 环境中对正式表结构进行读写测试，无需担忧污染生产数据。

---

## 目录结构

```text
trading_agent/
├── .env                          ← 环境变量配置（API Keys、实盘开关等）
├── SYSTEM_DOCUMENTATION.md       ← 本文档
├── config.py                     ← 全局配置（环境隔离、数据库路径、API Keys）
├── logger_setup.py               ← 统一日志配置
├── requirements.txt              ← Python 依赖
├── vertex_key.json               ← GCP 服务账号密钥
│
├── agents/                       ← AI 代理模块
│   ├── auto_miner.py             ← 策略矿工（自动挖掘策略）
│   ├── auto_portfolio_agent.py   ← FOF 组合代理（自动构建策略组合）
│   ├── base_llm_client.py        ← 大模型客户端（策略生成/调试/调参）
│   ├── data_engineer_agent.py    ← 数据工程师代理（自动生成数据适配器）
│   ├── researcher_agent.py       ← 研究员代理（阅读研报/提取观点）
│
├── content_engine/               ← 内容引擎（AI 宏观归因分析与内容分发）
│   ├── chart_generator.py        ← 极客风资金曲线图生成器（开平仓散点+水印指标）
│   ├── content_director_agent.py ← 内容分发主控（Pipeline A/B 调度 + 交易明细提取）
│   ├── content_writer.py         ← LLM 文案生成器（调用 Prompt 模板）
│   ├── data_extractor.py         ← 策略数据提取器（Pipeline A/B 查询）
│   ├── prompts/                  ← Prompt 模板目录
│   │   ├── xiaohongshu.py        ← 小红书 MacroMind Agent 机器风格模板
│   │   ├── xingqiu.py           ← 知识星球宏观归因分析风格模板
│
├── data/                         ← 数据存储目录
│   ├── trading_system_prod.db    ← PROD 生产库
│   ├── trading_system_dev.db     ← DEV 开发沙盒库
│   ├── fred_metadata.json        ← FRED 宏观指标元数据
│   ├── idea_sources.json         ← 研报/灵感来源配置
│
├── data_pipeline/                ← 数据管线（ELT 架构核心）
│   ├── sync_crypto_data.py       ← 加密货币数据一键同步调度器
│   ├── sync_us_stock_data.py     ← 美股数据一键同步调度器
│   ├── sync_commodities_data.py  ← 大宗商品数据一键同步调度器
│   ├── rss_feeder.py             ← RSS 研报抓取器
│   ├── adapters/                 ← ELT 适配器（Extract → raw_data）
│   │   ├── crypto_binance_adapter.py      ← 币安资金费率
│   │   ├── crypto_coinmetrics_adapter.py  ← MVRV 比率
│   │   ├── crypto_alternative_adapter.py  ← 恐惧贪婪指数
│   │   ├── crypto_defillama_adapter.py    ← 稳定币总市值
│   │   ├── crypto_deribit_adapter.py      ← 隐含波动率 (DVOL)
│   │   ├── crypto_coingecko_adapter.py    ← CoinGecko 市值/板块数据
│   │   ├── crypto_fiat_premium_adapter.py ← 法币溢价 (韩国泡菜溢价等)
│   │   ├── fmp_factor_adapter.py          ← FMP 美股因子数据
│   │   ├── commodities_term_structure_adapter.py ← 大宗商品期限结构
│   │   ├── gold_macro_residual_adapter.py ← 黄金宏观残差因子
│   │   ├── macro_fred_adapter.py          ← 美联储宏观指标 (~200个)
│   ├── fetchers/                 ← 数据获取器（Extract + Transform）
│   │   ├── crypto_market_fetcher.py       ← 加密货币量价数据（Binance 公共数据通道）
│   │   ├── crypto_factor_fetcher.py       ← 加密货币因子数据获取
│   │   ├── us_stock_market_fetcher.py     ← 美股深度历史数据（FMP API）
│   │   ├── commodities_market_fetcher.py  ← 大宗商品量价数据（FMP API）
│
├── database/                     ← 数据库管理层
│   ├── db_manager.py             ← 统一数据库管理器（所有资产类别 + 策略 + 组合）
│
├── draft_zone/                   ← 草稿适配器区（待验证的新数据源）
│   ├── base_draft_adapter.py     ← 草稿适配器基类
│   ├── draft_coingecko_corporate_treasury_adapter.py
│   ├── draft_coingecko_crypto_ai_sector_adapter.py
│   ├── draft_coingecko_lst_peg_adapter.py
│   ├── draft_commodities_term_structure_adapter.py
│   ├── draft_crypto_fiat_premium_adapter.py
│   ├── draft_defillama_dex_volume_adapter.py
│   ├── draft_defillama_fees_adapter.py
│   ├── draft_defillama_yields_convex_adapter.py
│   ├── draft_gold_macro_residual_adapter.py
│   ├── draft_multi_exchange_funding_rate_adapter.py
│   ├── draft_polymarket_odds_adapter.py
│
├── ops_scripts/                  ← 运维管理脚本
│   ├── promote_adapter.py        ← 草稿适配器晋升为正式适配器
│   ├── set_portfolio_status.py   ← 手动设置组合状态
│   ├── resolve_data_requirements.py ← 自动解决数据需求
│   ├── migrate_factor_descriptions.py ← 因子描述迁移脚本
│   ├── cleanup/                  ← 数据清理工具
│   │   ├── clear_data_requirements.py
│   │   ├── clear_factor_data.py
│   │   ├── clear_portfolio_records.py
│   │   ├── clear_portfolio_tables.py
│   │   ├── clear_sandbox_tables.py
│   │   ├── clear_strategy_tables.py
│   │   ├── reset_data_requirements.py
│   ├── migrations/               ← 数据库结构迁移脚本
│   │   ├── migrate_market_data.py
│   │   ├── migrate_portfolio_metrics.py
│   │   ├── refactor_multi_asset_db.py
│   ├── viewers/                  ← 只读查看工具
│   │   ├── check_db_schema.py
│   │   ├── show_data_requirements.py
│   │   ├── show_top_strategies.py
│
├── playground/                   ← 沙盒测试区
│   ├── check_future_leak.py      ← 未来数据泄露检查
│   ├── run_backtest_by_dir_id.py ← 按策略ID运行回测
│   ├── test_binance_market.py    ← 币安连接测试
│   ├── test_draft_adapters.py    ← 草稿适配器测试
│   ├── test_fmp_connection.py    ← FMP API 连接测试
│
├── trading_engine/               ← 交易与回测引擎
│   ├── backtest_engine.py        ← 回测引擎（信号计算 + 绩效评估 + 极速数据加载）
│   ├── execution_agent.py        ← 实盘/模拟盘执行大脑（Daemon 守护进程）
│   ├── main_controller.py        ← 策略研发总控制器（串联 AI 代理 + 回测）
│   ├── portfolio_backtest.py     ← 组合回测器
│   ├── portfolio_manager.py      ← 组合管理工具（交互式创建/管理）
│   ├── portfolio_optimizer.py    ← 组合优化器（资金分配模式）
│
├── scratch/                      ← 临时脚本区
│   ├── clear_dev_db_tables.py
│   ├── debug_pipeline_a.py       ← Pipeline A 调试脚本
│   ├── diag_extractor.py         ← 数据提取器诊断脚本
│   ├── reset_publish_status.py   ← 重置所有策略发布状态为 UNPUBLISHED
│   ├── verify_tables.py
│
├── logs/                         ← 日志目录
│   ├── system.log
```

---

## 数据库设计

系统使用 SQLite 数据库存储所有数据，采用 ELT（Extract-Load-Transform）架构。通过 `SYSTEM_ENV` 环境变量切换 PROD/DEV 数据库。

### 主数据库表（16 张业务表）

| # | 表名 | 说明 | 唯一约束 |
|---|------|------|----------|
| 1 | market_data_crypto | 加密货币行情与指标数据表 | (symbol, timestamp) |
| 2 | factor_data | 外部因子数据表 | (symbol, timestamp, factor_name) |
| 3 | raw_data | 原始数据湖表（ELT 架构核心） | (source_id, event_timestamp) |
| 4 | factor_metadata | 因子元数据表（描述、来源、单位） | (factor_name, symbol) |
| 5 | fred_series_config | FRED 宏观指标配置表 | series_id (PK) |
| 6 | strategy_directions | 策略方向大表 | dir_id (PK) |
| 7 | strategy_versions | 策略版本明细表 | ver_id (PK) |
| 8 | data_requirements | AI 提出的数据需求表 | id (PK) |
| 9 | research_articles | 研究文章记忆表 | url (PK) |
| 10 | portfolios | 组合基本信息表 | name (UNIQUE) |
| 11 | portfolio_components | 组合策略关联表 | id (PK) |
| 12 | portfolio_daily_records | 组合净值每日记录表 | id (PK) |
| 13 | exchange_orders | 交易所订单表 | id (PK) |
| 14 | us_stock_transcripts | 美股电话会议表 | (symbol, fiscal_year, quarter) |
| 15 | market_data_us_stock | 美股行情数据表 | (symbol, timestamp) |
| 16 | market_data_commodity | 大宗商品行情数据表 | (symbol, timestamp) |

### 核心表结构详解

#### 1. market_data_crypto - 加密货币行情与指标数据表

| 字段 | 类型 | 说明 |
|------|------|------|
| symbol | TEXT | 交易对符号 (如 BTC_USDT) |
| timestamp | DATETIME | 时间戳 |
| open | REAL | 开盘价 |
| high | REAL | 最高价 |
| low | REAL | 最低价 |
| close | REAL | 收盘价 |
| volume | REAL | 成交量 |
| rsi_14 | REAL | 14日RSI指标 |
| macd | REAL | MACD值 |
| macd_signal | REAL | MACD信号线 |
| macd_hist | REAL | MACD柱状图 |

**主键**: (symbol, timestamp)

---

#### 2. raw_data - 原始数据湖表（ELT 架构核心）

存储所有数据源的原始 JSON 响应，支持增量更新和无限扩展。

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 (自增) |
| source_id | TEXT | 数据源ID (如 binance_funding, coinmetrics_mvrv, fred_M2SL) |
| event_timestamp | DATETIME | 事件时间戳 |
| fetch_timestamp | DATETIME | 抓取时间戳 |
| raw_content | TEXT | 原始 JSON 内容 |

**唯一约束**: (source_id, event_timestamp)

---

#### 3. factor_data - 外部因子数据表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 (自增) |
| symbol | TEXT | 交易对符号 |
| timestamp | DATETIME | 时间戳 |
| factor_name | TEXT | 因子名称 |
| factor_value | REAL | 因子值 |

**唯一约束**: (symbol, timestamp, factor_name)

---

#### 4. factor_metadata - 因子元数据表

| 字段 | 类型 | 说明 |
|------|------|------|
| factor_name | TEXT | 因子名称 |
| symbol | TEXT | 关联标的 (如 BTC_USDT, MACRO) |
| description | TEXT | 因子描述 |
| source | TEXT | 数据来源 (如 CoinMetrics, FRED, Binance) |
| unit | TEXT | 单位 (默认空字符串) |
| update_freq | TEXT | 更新频率 (默认空字符串) |

**唯一约束**: (factor_name, symbol)

---

#### 5. fred_series_config - FRED 宏观指标配置表

| 字段 | 类型 | 说明 |
|------|------|------|
| series_id | TEXT | 主键 (FRED 序列 ID，如 M2SL, DGS10) |
| title | TEXT | 序列标题 |
| category | TEXT | 分类 (默认 'extension') |
| source_req_id | INTEGER | 来源数据需求 ID (外键) |
| added_at | DATETIME | 添加时间 (默认 CURRENT_TIMESTAMP) |

**主键**: series_id

---

#### 6. strategy_directions - 策略方向大表

| 字段 | 类型 | 说明 |
|------|------|------|
| dir_id | TEXT | 主键 (UUID) |
| name | TEXT | 策略名称 |
| description | TEXT | 策略逻辑描述 |
| status | TEXT | 状态 |
| best_version_id | TEXT | 最佳版本ID (外键) |
| timeframe | TEXT | 时间周期 |
| source | TEXT | 来源 (如 RESEARCHER_AGENT) |
| is_active_ensemble | INTEGER | 入池状态 (0/1)，默认 0 |
| target_asset | TEXT | 目标资产大类 (crypto/us_stock/commodity)，默认 crypto |
| target_symbol | TEXT | 具体交易标的 (BTC_USDT/SPY/QQQ/GCUSD/BZUSD 等)，默认 BTC_USDT |
| publish_status | TEXT | 发布状态 (UNPUBLISHED/PUBLISHED_XHS/PUBLISHED_XQ)，默认 UNPUBLISHED |
| origin_url | TEXT | 原始灵感来源 URL（研报/文章链接，支持全链路追踪） |
| created_at | DATETIME | 创建时间 |

**主键**: dir_id

---

#### 7. strategy_versions - 策略版本明细表

| 字段 | 类型 | 说明 |
|------|------|------|
| ver_id | TEXT | 主键 (UUID) |
| dir_id | TEXT | 策略方向ID (外键) |
| iteration_type | TEXT | 迭代类型 (init/debug/tune) |
| code_content | TEXT | 策略代码 |
| params_json | TEXT | 参数 JSON |
| run_status | TEXT | 运行状态 |
| error_log | TEXT | 错误日志 |
| metric_sharpe | REAL | 夏普率 |
| metric_return | REAL | 总收益率 |
| metric_max_drawdown | REAL | 最大回撤 |
| metric_win_rate | REAL | 胜率 |
| metric_profit_loss_ratio | REAL | 盈亏比 |
| metric_total_trades | INTEGER | 总交易次数 |
| metric_annualized_return | REAL | 年化收益率 |
| metric_excess_return | REAL | 超额收益 |
| metric_total_profit_loss | REAL | 总盈亏 |
| metric_avg_profit_loss_per_trade | REAL | 平均每笔盈亏 |
| metric_excess_annual_return | REAL | 年化超额收益 |
| metric_market_return | REAL | 基准收益率 |
| metric_market_annual_return | REAL | 基准年化收益 |
| metric_start_date | TEXT | 回测起始日 |
| metric_end_date | TEXT | 回测结束日 |
| metric_avg_hold_period | REAL | 平均持仓周期 |
| timeframe | TEXT | 时间周期 |
| created_at | DATETIME | 创建时间 |

**主键**: ver_id  
**外键**: dir_id -> strategy_directions.dir_id

---

#### 8. data_requirements - AI 提出的数据需求表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 (自增) |
| source_name | TEXT | 数据源名称 |
| description | TEXT | 数据源描述 |
| required_reason | TEXT | 需求理由 |
| pending_strategy_name | TEXT | 拟定策略名 |
| pending_strategy_desc | TEXT | 拟定策略逻辑 |
| status | TEXT | 状态 (PENDING/DRAFTING/COMPLETED/UNRESOLVABLE)，默认 PENDING |
| is_awakened | INTEGER | 是否已唤醒 (0/1) |
| source | TEXT | 来源 (如 Researcher_Agent)，默认 Researcher_Agent |
| target_symbol | TEXT | 关联标的 (如 BTC_USDT/SPY)，默认 GLOBAL |
| created_at | DATETIME | 创建时间 |

---

#### 9. research_articles - 研究文章记忆表

| 字段 | 类型 | 说明 |
|------|------|------|
| url | TEXT | 主键 (文章URL) |
| title | TEXT | 文章标题 |
| source_name | TEXT | 数据源名称 |
| content_md | TEXT | Markdown 内容 |
| status | TEXT | 状态 (UNREAD/READ) |
| created_at | DATETIME | 创建时间 |

---

#### 10. portfolios - 组合基本信息表

| 字段 | 类型 | 说明 |
|------|------|------|
| portfolio_id | INTEGER | 主键 (自增) |
| name | TEXT | 组合名称 (唯一) |
| description | TEXT | 组合描述 |
| status | TEXT | 状态 (DRAFT/TESTED/PAPER/LIVE) |
| target_asset | TEXT | 目标资产大类 (crypto/us_stock/commodity)，默认 crypto |
| target_symbol | TEXT | 具体交易标的 (BTC_USDT/SPY/GCUSD 等)，默认 BTC_USDT |
| metric_annualized_return | REAL | 回测年化收益率 |
| metric_sharpe | REAL | 回测夏普率 |
| metric_max_drawdown | REAL | 回测最大回撤 |
| weight_mode | TEXT | 权重模式 (equal/sharpe/scaling/risk_parity) |
| created_at | DATETIME | 创建时间 |

---

#### 11. portfolio_components - 组合策略关联表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 (自增) |
| portfolio_id | INTEGER | 组合ID (外键) |
| dir_id | TEXT | 策略方向ID (外键) |

---

#### 12. portfolio_daily_records - 组合净值每日记录表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 (自增) |
| portfolio_id | INTEGER | 组合ID (外键) |
| date | TEXT | 日期 (格式: YYYY-MM-DD) |
| run_phase | TEXT | 运行阶段 (BACKTEST/PAPER/LIVE) |
| btc_price | REAL | BTC 当日收盘价 |
| combined_signal | REAL | 聚合信号值 [-1, 1] |
| turnover | REAL | 换手率 |
| fee_paid | REAL | 支付的手续费 |
| daily_return | REAL | 当日收益率 |
| nav | REAL | 累计净值 |
| total_equity | REAL | 总权益（PAPER/LIVE 模式写入，通过数据库迁移动态添加） |

**写入规则**:
- BACKTEST 阶段：先清空同组合同阶段旧数据，再写入
- PAPER/LIVE 阶段：使用 INSERT OR REPLACE 追加写入，保留历史记录

---

#### 13. exchange_orders - 交易所订单表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 (自增) |
| portfolio_id | INTEGER | 组合ID |
| symbol | TEXT | 交易对符号 |
| order_id | TEXT | 交易所返回的订单ID |
| side | TEXT | 买卖方向 |
| order_type | TEXT | 订单类型 |
| amount | REAL | 交易数量 |
| price | REAL | 交易价格 |
| fee | REAL | 手续费 |
| status | TEXT | 订单状态 |
| created_at | DATETIME | 创建时间 |

---

#### 美股专属表结构

##### market_data_us_stock - 美股行情数据表

| 字段 | 类型 | 说明 |
|------|------|------|
| symbol | TEXT | 股票代码 (如 TSLA) |
| timestamp | DATETIME | 时间戳 |
| open | REAL | 开盘价 |
| high | REAL | 最高价 |
| low | REAL | 最低价 |
| close | REAL | 收盘价 |
| volume | REAL | 成交量 |
| rsi_14 | REAL | 14日RSI |
| macd | REAL | MACD值 |
| macd_signal | REAL | MACD信号线 |
| macd_hist | REAL | MACD柱状图 |

**主键**: (symbol, timestamp)

##### market_data_commodity - 大宗商品行情数据表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 (自增) |
| symbol | TEXT | 商品代码 (如 CLUSD, BZUSD, GCUSD) |
| timestamp | DATETIME | 时间戳 |
| open | REAL | 开盘价 |
| high | REAL | 最高价 |
| low | REAL | 最低价 |
| close | REAL | 收盘价 |
| volume | REAL | 成交量 |
| rsi_14 | REAL | 14日RSI |
| macd | REAL | MACD值 |
| macd_signal | REAL | MACD信号线 |
| macd_hist | REAL | MACD柱状图 |

**主键**: (symbol, timestamp)

##### us_stock_transcripts - 美股电话会议表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 (自增) |
| symbol | TEXT | 股票代码 |
| fiscal_year | INTEGER | 财年 |
| quarter | INTEGER | 季度 (1-4) |
| publish_date | DATETIME | 发布日期 |
| content | TEXT | 会议内容文本 |
| is_processed | INTEGER | 是否已处理 (0/1) |
| created_at | DATETIME | 创建时间 |

**唯一约束**: (symbol, fiscal_year, quarter)

---

## 核心模块

### 1. config.py - 全局配置模块

**功能**: 系统全局配置，实现 DEV/PROD 双环境隔离

**导出变量**:
- `SYSTEM_ENV`: 当前环境 ('PROD' / 'DEV')
- `DATA_DIR`: 数据目录路径 (project_root/data/)
- `DB_NAME`: 数据库文件名 (根据环境切换)
- `DB_PATH`: 数据库完整路径
- `GOOGLE_APPLICATION_CREDENTIALS`: GCP 服务账号密钥路径
- `GCP_PROJECT_ID`: GCP 项目 ID
- `GCP_LOCATION`: GCP 区域
- `BINANCE_API_KEY`: 币安 API Key
- `BINANCE_SECRET`: 币安 Secret Key
- `LOG_LEVEL`: 日志级别 (PROD=INFO, DEV=DEBUG)

**依赖**: 无 (基础模块)

---

### 2. logger_setup.py - 日志模块

**功能**: 统一的日志记录配置

**日志输出位置**:
- 控制台 (StreamHandler)
- 日志文件 (FileHandler → logs/system.log)

**日志格式**: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

---

## 数据管线 (Data Pipeline)

系统采用 ELT（Extract-Load-Transform）架构，所有数据源先无损落湖（raw_data），再按需提取为结构化因子（factor_data）。

### 加密货币数据管线

**调度器**: `data_pipeline/sync_crypto_data.py`

**数据源与适配器**:

| 适配器 | 数据源 | 写入表 | 频率 |
|--------|--------|--------|------|
| crypto_binance_adapter | 币安资金费率 | raw_data / factor_data | 每日 |
| crypto_coinmetrics_adapter | MVRV 比率 | raw_data / factor_data | 每日 |
| crypto_alternative_adapter | 恐惧贪婪指数 | raw_data / factor_data | 每日 |
| crypto_defillama_adapter | 稳定币总市值 | raw_data / factor_data | 每日 |
| crypto_deribit_adapter | 隐含波动率 (DVOL) | raw_data / factor_data | 每日 |
| macro_fred_adapter | 美联储宏观指标 | raw_data / factor_data | 每日 |

**量价数据获取器**: `crypto_market_fetcher.py`
- 对接 Binance 公共数据通道，获取 1 分钟级 OHLCV 数据
- 自动计算 RSI(14) 和 MACD 指标
- 增量更新逻辑：查询最新时间戳 → 仅拉取增量 → 落库

### 美股数据管线

**调度器**: `data_pipeline/sync_us_stock_data.py`

**数据获取器**: `us_stock_market_fetcher.py`
- 对接 FMP API (`/stable/historical-price-eod/full`)
- 获取日线 OHLCV + 电话会议 + 盈利预期差
- 自动计算 RSI(14) 和 MACD 指标
- 增量更新逻辑

### 大宗商品数据管线

**调度器**: `data_pipeline/sync_commodities_data.py`

**数据获取器**: `data_pipeline/fetchers/commodities_market_fetcher.py`

**CommoditiesMarketFetcher 类**:

| 方法 | 功能 |
|------|------|
| `fetch_daily_data(symbol)` | 请求 FMP API 获取日线 OHLCV，强制时间正序 |
| `calculate_indicators(df)` | 计算 RSI(14) + MACD(12,26,9)，NaN 填 0 |
| `sync_symbol(symbol)` | 增量同步主控：查最新时间戳 → 拉全量 → 算指标 → 截增量 → 落库 |

**支持的商品代码**:

| 代码 | 品种 |
|------|------|
| CLUSD | WTI 原油 |
| BZUSD | 布伦特原油 |
| GCUSD | 黄金 |

**数据库管理器**: `database/db_manager.py`（统一管理器）
- `init_db()`: 创建所有表（含 market_data_commodity）
- `save_market_data('commodity', df)`: 批量写入行情数据（INSERT OR IGNORE）
- `save_raw_data_single(source_id, event_timestamp, data_json)`: 写入原始数据到统一 raw_data 表
- `save_factor_data_single(symbol, timestamp, factor_name, factor_value)`: 写入因子数据到统一 factor_data 表
- `get_latest_market_timestamp('commodity', symbol)`: 查询最新时间戳，无数据返回 '2010-01-01 00:00:00'

---

## 交易引擎 (Trading Engine)

### 1. backtest_engine.py - 回测引擎

**核心功能**:
- 策略代码编译与执行
- 数据加载与降采样
- 绩效指标计算（夏普率、最大回撤、胜率等）

**数据加载函数**:

| 函数 | 用途 | 内存占用 |
|------|------|----------|
| `load_historical_data()` | 加载全量 1 分钟数据 + 因子融合（延迟融合架构） | 高（260万行） |
| `load_resampled_data(symbol, target_asset, timeframe)` | 极速加载指定周期数据，仅读 OHLCV 6 列 → 降采样 → 计算指标 → 融合因子 | 极低（~3000行日线） |
| `load_daily_data_directly(symbol, target_asset)` | `load_resampled_data` 的日线别名 | 极低 |

**关键优化**:
- `load_resampled_data` 相比 `load_historical_data` + `resample_data` 的两步冗余路径，内存占用降低 99%+，是内容引擎的推荐数据加载方式
- 因子数据 pivot 使用字典构建 + `reindex` 方式替代 `pivot_table`，避免 numpy `groupsort_indexer` 内存碎片化问题
- `resample_data` 内部采用分批聚合（每批 20 列），避免 112 列全量 consolidate 导致 OOM

**绩效指标计算** (`calculate_metrics`):
- 状态机终极版：完美处理未平仓、多空反转、复利对齐
- 输出指标：total_return, annual_return, sharpe_ratio, max_drawdown, win_rate, win_loss_ratio, avg_hold_period 等

### 2. execution_agent.py - 实盘执行模块

**安全防线**:
- `LIVE_TRADING` 环境变量控制实盘开关
- 动态切换沙盒/实盘模式
- 10 秒防误触倒计时
- 强制时钟同步（消灭 -1021 Timestamp 错误）
- SSL 断连自动重试
- 3% 仓位偏离度容忍阈值（防止过度交易）
- 最小交易金额检查

### 3. main_controller.py - 策略研发总控制器

串联 AI 代理与回测引擎，实现策略的自动生成、调试、调参全流程。

### 4. portfolio_backtest.py / portfolio_manager.py / portfolio_optimizer.py

- 组合回测、创建管理、资金分配优化
- 支持多资产类别：通过 `target_symbol` 字段指定组合的交易标的（BTC_USDT/SPY/QQQ/GCUSD 等）
- 组合内策略必须与组合的 `target_symbol` 一致，确保数据一致性
- `portfolio_manager.py`：交互式创建组合时选择 `target_symbol`，自动筛选匹配的策略
- `portfolio_optimizer.py`：根据 `target_symbol` 加载对应资产的日线数据，支持 equal/sharpe/scaling/risk_parity 四种权重模式
- `portfolio_backtest.py`：组合回测时按 `target_symbol` 加载对应资产价格数据计算收益率

---

## 内容引擎 (Content Engine)

内容引擎是系统的"最后一公里"——将回测后的策略数据自动转化为可发布的结构化内容。

### 架构概览

```text
strategy_directions (DB)
        │
        ▼
  data_extractor.py ──── 提取策略 + 全量指标
        │
        ├──▶ chart_generator.py ──── 极速加载数据 → 重跑回测 → 生成极客风资金曲线图
        │
        └──▶ content_writer.py ──── 填充 Prompt 模板 → 调用 LLM → 生成文案
                                        │
                                        ▼
                              content_director_agent.py ──── 主控调度 + 交易明细提取
```

### 双管线分发

| 管线 | 目标平台 | 筛选条件 | Prompt 风格 | 发布状态标记 |
|------|---------|----------|------------|-------------|
| Pipeline A | 小红书 | `source=RESEARCHER_AGENT` 且 `annualized_return < 0` 且 `publish_status=UNPUBLISHED` | MacroMind Agent 机器风格（无 Emoji、无情绪、结构化） | PUBLISHED_XHS |
| Pipeline B | 知识星球 | `metric_sharpe > 1.5` 且 `publish_status=UNPUBLISHED` | 顶级宏观对冲基金经理风格（宏观归因分析、Regime Shift 复盘） | PUBLISHED_XQ |

### Pipeline B 宏观归因分析流程

Pipeline B 从干瘪的数据罗列升级为具有极深洞察力的"真实宏观归因分析（Macro Attribution Analysis）"：

1. **交易明细提取** (`extract_trade_history`)
   - 调用 `load_resampled_data` 极速加载策略对应周期的数据
   - 编译策略代码，计算信号与持仓
   - 遍历 `positions.diff()` 提取每一次开仓（Entry）和平仓（Exit）
   - 计算每笔交易的真实收益率（含手续费）
   - 输出格式：`1. 2023-03-10 to 2023-04-15, Return: +25.4%`
   - 同时解析 `params_json`，排除 `commission_rate`，输出核心参数列表

2. **Prompt 模板** (`xingqiu.py`)
   - System 角色：顶级宏观对冲基金经理，极度客观、犀利
   - 输入变量：策略名称、description、总收益、夏普率、最大回撤、胜率、平均持仓周期、code_content、最优参数(params_str)、交易明细(trade_history_str)
   - 强制输出结构（5 段，无 CTA 话术）：
     1. 策略思路 — 精炼提炼宏观博弈逻辑
     2. AI 找到的最优参数 — 只列核心参数
     3. 回测结果与真实宏观归因 — 命中率统计 + 成功交易复盘 + 假信号深度分析
     4. 核心代码切片 — `df['signal'] = ...` 代码块
     5. 风险提示（基于复盘） — 必须基于假信号分析，禁止套话

### 图表生成

`chart_generator.py` 使用 `load_resampled_data` 替代 `load_historical_data` + `resample_data`，大幅降低内存占用。

**图表水印指标**（左上角）：
```
Ann.Return: 45.2% | Sharpe: 2.31
WinRate: 58.3% | MaxDD: -12.4%
```

**图表元素**：
- 绿色资金曲线
- 青色上三角：开仓点（Entry）
- 红色下三角：平仓点（Exit）
- 右下角品牌水印：Powered by MacroMind Agent

---

## AI 代理 (Agents)

### 1. researcher_agent.py - 研究员代理

**功能**: 阅读外网研报与新闻，提取市场观点

**工作流**:
1. 从 RSS 源获取研报链接
2. 使用 LLM 提取核心观点
3. 转化为交易逻辑描述
4. 存入 research_articles 表

### 2. auto_miner.py - 策略矿工

**功能**: 自动挖掘策略

**工作流**:
1. 从 research_articles 获取未读文章
2. 使用 LLM 生成策略代码
3. 自动执行回测
4. 多轮调参优化
5. 沉淀入库

### 3. base_llm_client.py - 大模型客户端

**功能**: 封装 Google Gemini API 调用

**支持的操作**:
- 策略代码生成
- 策略代码调试
- 策略参数调优

### 4. data_engineer_agent.py - 数据工程师代理

**功能**: 自动生成数据适配器代码

### 5. auto_portfolio_agent.py - FOF 组合代理

**功能**: 自动构建策略组合（FOF 模式）

**工作流**:
1. 获取候选策略：按夏普率 ≥ 0.5、交易次数 ≥ 3 筛选，支持按 `target_symbol` 过滤
2. 语义聚类：使用 LLM 对候选策略进行语义分组，识别策略间的逻辑差异
3. 组合构建：从每个聚类中选取最优策略，构建多策略组合
4. 组合回测：自动执行组合回测，计算组合级绩效指标
5. 支持多资产：按 `target_symbol` 分别构建不同资产的组合

---

## 运维脚本 (Ops Scripts)

### promote_adapter.py
将草稿适配器晋升为正式适配器，从 draft_zone/ 移动到 data_pipeline/adapters/。

### set_portfolio_status.py
手动设置组合状态（DRAFT/TESTED/PAPER/LIVE）。

### cleanup/ 目录
| 脚本 | 功能 |
|------|------|
| clear_data_requirements.py | 清空数据需求表 |
| clear_factor_data.py | 清空因子数据表 |
| clear_portfolio_records.py | 清空组合记录 |
| clear_portfolio_tables.py | 清空组合相关表 |
| clear_sandbox_tables.py | 清空沙盒表 |
| clear_strategy_tables.py | 清空策略相关表 |
| reset_data_requirements.py | 重置数据需求状态 |

### viewers/ 目录
| 脚本 | 功能 |
|------|------|
| check_db_schema.py | 查看数据库表结构 |
| show_data_requirements.py | 查看数据需求列表 |
| show_top_strategies.py | 查看顶级策略 |

### scratch/ 目录
| 脚本 | 功能 |
|------|------|
| reset_publish_status.py | 重置所有策略的 publish_status 为 UNPUBLISHED |
| debug_pipeline_a.py | Pipeline A 调试脚本 |
| diag_extractor.py | 数据提取器诊断脚本 |

---

## 草稿区 (Draft Zone)

待验证的新数据源适配器，遵循 `base_draft_adapter.py` 基类接口。

| 适配器 | 数据源 |
|--------|--------|
| draft_coingecko_corporate_treasury_adapter | CoinGecko 企业金库持仓 |
| draft_coingecko_crypto_ai_sector_adapter | CoinGecko AI 板块数据 |
| draft_coingecko_lst_peg_adapter | CoinGecko LST 锚定数据 |
| draft_commodities_term_structure_adapter | 大宗商品期限结构 |
| draft_crypto_fiat_premium_adapter | 法币溢价 (韩国泡菜溢价等) |
| draft_defillama_dex_volume_adapter | DeFiLlama DEX 交易量 |
| draft_defillama_fees_adapter | DeFiLlama 协议费用 |
| draft_defillama_yields_convex_adapter | DeFiLlama Convex 收益 |
| draft_gold_macro_residual_adapter | 黄金宏观残差因子 |
| draft_multi_exchange_funding_rate_adapter | 多交易所资金费率 |
| draft_polymarket_odds_adapter | Polymarket 预测市场赔率 |

---

## 工作流程

### 策略研发全流程

```text
1. 研报获取：
   ├── RSS Feeder 抓取研报链接
   └── researcher_agent 阅读 → 提取观点 → 存入 research_articles

2. 策略生成：
   ├── auto_miner 从文章提取策略逻辑
   ├── base_llm_client 生成策略代码
   └── main_controller 编排多轮迭代（init → debug → tune）

3. 回测评估：
   ├── backtest_engine 编译策略 → 计算信号 → 评估绩效
   ├── 绩效指标落库 strategy_versions
   └── 最佳版本回写 strategy_directions.best_version_id

4. 组合构建：
   ├── auto_portfolio_agent 自动筛选候选策略（夏普率 ≥ 0.5）→ 语义聚类 → 构建组合
   ├── portfolio_manager 创建组合 → 选择 target_symbol → 选择策略
   ├── portfolio_optimizer 优化资金分配
   └── portfolio_backtest 组合回测验证
```

### 内容分发全流程

```text
1. Pipeline A（小红书极差策略）：
   ├── data_extractor → 查询 source=RESEARCHER_AGENT 且 annualized_return < 0 且 UNPUBLISHED
   ├── chart_generator → load_resampled_data → 重跑回测 → 生成极客风资金曲线图
   ├── content_writer → 填充 MacroMind Agent 机器风格 Prompt → 调用 Gemini LLM
   ├── 保存图表 (.png) + 文案 (.md)
   ├── 打印截图提示（origin_url → 手动截图配图）
   └── 更新 publish_status = PUBLISHED_XHS

2. Pipeline B（知识星球宏观归因分析）：
   ├── data_extractor → 查询 metric_sharpe > 1.5 且 UNPUBLISHED
   ├── chart_generator → load_resampled_data → 重跑回测 → 生成极客风资金曲线图
   ├── extract_trade_history → load_resampled_data → 提取交易明细 + 最优参数
   ├── content_writer → 填充宏观归因分析 Prompt（含 trade_history_str + params_str）→ 调用 Gemini LLM
   ├── 保存图表 (.png) + 文案 (.md)
   ├── 打印截图提示（origin_url → 手动截图配图）
   └── 更新 publish_status = PUBLISHED_XQ
```

### 实盘交易全流程

```text
1. 启动前检查：
   ├── .env 中 LIVE_TRADING=true（实盘）/ false（模拟盘）
   ├── exchange 初始化（沙盒/实盘模式切换）
   └── 10 秒防误触倒计时

2. 每日执行：
   ├── 强制时钟同步（与交易所服务器对时）
   ├── 获取当前持仓与目标信号
   ├── 仓位偏离度检查（3% 容忍阈值）
   ├── 最小交易金额检查
   └── 执行再平衡订单

3. 记录落库：
   ├── portfolio_daily_records 写入每日净值
   └── exchange_orders 写入订单记录
```
