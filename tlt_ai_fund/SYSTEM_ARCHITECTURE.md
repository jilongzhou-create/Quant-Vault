# TLT AI Fund - 系统架构文档

---

## 1. 项目概述

### 1.1 项目定位
TLT AI Fund 是一个基于**宏观因子驱动**的美债 ETF (TLT) 中低频趋势策略系统，旨在通过分析三大宏观支柱（通胀、就业、货币政策）来预测 TLT 价格走势。

### 1.2 核心目标
- 构建与 gold_ai_fund 完全解耦的独立策略模块
- 实现基于宏观因子的 SMA 投票机制
- 集成双向趋势霸权风控结构
- 支持现金管理收益（DTB3 无风险利率）

### 1.3 策略理念
TLT (iShares 20+ Year Treasury Bond ETF) 作为长期美债 ETF，对以下因素高度敏感：
- **通胀预期**：高通胀导致债券实际收益率下降，利空美债
- **经济周期**：经济衰退期，美债作为避险资产受到追捧
- **货币政策**：加息周期利空美债，降息周期利多美债

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    TLT AI Fund 系统架构                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────────┐                                          │
│   │   Data Pipeline  │                                          │
│   │  (数据管道层)     │                                          │
│   └────────┬─────────┘                                          │
│            │                                                    │
│            ▼                                                    │
│   ┌──────────────────┐     ┌──────────────────┐                 │
│   │  Market Data     │     │    Macro Data    │                 │
│   │   (TLT 日线)      │     │  (FRED 宏观)     │                 │
│   └────────┬─────────┘     └────────┬─────────┘                 │
│            │                        │                           │
│            └──────────┬─────────────┘                           │
│                       ▼                                         │
│   ┌──────────────────────────────────┐                          │
│   │       Core Framework             │                          │
│   │   (宏观底座 - SMA 投票引擎)        │                          │
│   └──────────────────┬───────────────┘                          │
│                      │                                          │
│                      ▼                                          │
│   ┌──────────────────────────────────┐                          │
│   │       Execution Engine           │                          │
│   │   (双向趋势霸权执行层)            │                          │
│   └──────────────────┬───────────────┘                          │
│                      │                                          │
│                      ▼                                          │
│   ┌──────────────────────────────────┐                          │
│   │      Backtest Engine             │                          │
│   │   (回测引擎 + 现金管理)           │                          │
│   └──────────────────┬───────────────┘                          │
│                      │                                          │
│                      ▼                                          │
│   ┌──────────────────────────────────┐                          │
│   │       Performance Report         │                          │
│   │   (绩效报告 + 年度分解)           │                          │
│   └──────────────────────────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 目录结构

```
tlt_ai_fund/
├── config.py                  # 配置文件 (IS/OOS 日期、参数)
├── db_schema_tlt.sql          # 数据库建表脚本
├── run_backtest.py            # 回测入口脚本
├── sync_tlt_data.py           # 数据同步脚本
├── core/
│   ├── tlt_macro_framework.py # 主框架（整合底座+执行）
│   └── framework/
│       ├── tlt_core_anchor.py # 宏观底座（SMA投票）
│       └── tlt_execution.py   # 趋势霸权执行器
└── engine/
    └── tlt_backtest.py        # 回测引擎（含现金管理）
```

### 2.3 关键文件职责

| 文件 | 职责 | 关键功能 |
|------|------|----------|
| `config.py` | 配置管理 | IS/OOS 周期定义 |
| `tlt_core_anchor.py` | 宏观底座 | 三大因子 SMA 投票 |
| `tlt_execution.py` | 执行层 | 双向趋势霸权 |
| `tlt_macro_framework.py` | 主框架 | 整合数据流 |
| `tlt_backtest.py` | 回测引擎 | 收益计算、风控 |

---

## 3. 核心组件详解

### 3.1 TltCoreAnchor - 宏观底座

#### 3.1.1 三大宏观支柱

| 支柱 | FRED 指标 | 投票逻辑 | 权重 |
|------|-----------|----------|------|
| **通胀锚** | CPIAUCSL | CPI_YoY > SMA60 → -1（利空） | 1/3 |
| **经济锚** | UNRATE | UNRATE > SMA60 → +1（避险利好） | 1/3 |
| **政策锚** | FEDFUNDS | FEDFUNDS > SMA60 → -1（加息利空） | 1/3 |

#### 3.1.2 投票机制

```
vote_cpi = +1 if cpi_yoy < sma_cpi else -1
vote_unrate = +1 if unrate > sma_unrate else -1
vote_fedfunds = +1 if fedfunds < sma_fedfunds else -1

tlt_core_signal = (vote_cpi + vote_unrate + vote_fedfunds) / 3.0
```

**信号值域**：`{-1.0, -0.33, 0.33, 1.0}`

### 3.2 TltExecution - 双向趋势霸权

#### 3.2.1 趋势过滤规则

| 趋势状态 | 条件 | 敞口规则 |
|----------|------|----------|
| **Bear Trap** | Price < SMA50 < SMA200 | 强制空仓 (0.0) |
| **Bull Floor** | Price > SMA50 > SMA200 | 强制 0.25 托底 |
| **Bullish** | 中间态 + signal > 0 | signal（线性映射） |
| **Bearish** | 中间态 + signal ≤ 0 | 空仓 (0.0) |

#### 3.2.2 趋势区间定义

```python
df['bear_trap'] = (price < sma_50) & (sma_50 < sma_200)
df['bull_floor'] = (price > sma_50) & (sma_50 > sma_200)
```

### 3.3 TltBacktestEngine - 回测引擎

#### 3.3.1 核心算法

```python
# 收益计算（杜绝 Look-ahead bias）
strategy_return = (
    position * market_return              # 持仓收益
    + (1 - position) * daily_rf          # 现金利息收益
    - trade_cost                         # 交易成本
)
```

#### 3.3.2 现金管理机制

- 使用 **DTB3**（3个月美债收益率）作为无风险利率
- 日频利息计算：`daily_rf = dtb3 / 100 / 252`
- 闲置资金自动产生利息收益

---

## 4. 数据管道

### 4.1 数据源

| 数据源 | 类型 | 表名 | 关键字段 |
|--------|------|------|----------|
| FMP API | 行情 | `market_data_tlt` | adj_close（计算基准） |
| FRED | 通胀 | `raw_data` (CPIAUCSL) | value |
| FRED | 就业 | `raw_data` (UNRATE) | value |
| FRED | 利率 | `raw_data` (FEDFUNDS) | value |
| FRED | 现金收益 | `raw_data` (DTB3) | value |

### 4.2 数据处理流程

```
FMP API → TLT 日线数据 → adj_close 作为计算基准
                              │
                              ▼
FRED API → CPI/UNRATE/FEDFUNDS → 计算 YoY/SMA → 宏观信号
                              │
                              ▼
                         策略执行 → 回测 → 绩效报告
```

### 4.3 致命业务逻辑 ⚠️

**TLT 作为债券 ETF，每月有大额分红**：
- **必须**使用 `adjClose`（复权收盘价）作为计算基准
- 收益率、SMA 计算均基于 `adj_close`
- 原始 `close` 仅用于参考

---

## 5. 数据库设计

### 5.1 market_data_tlt 表结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `symbol` | TEXT | 标的代码 (TLT) |
| `timestamp` | DATETIME | 时间戳 |
| `date` | TEXT | 日期 |
| `open` | REAL | 开盘价 |
| `high` | REAL | 最高价 |
| `low` | REAL | 最低价 |
| `close` | REAL | 收盘价 |
| `adj_close` | REAL | **复权收盘价（计算基准）** |
| `volume` | REAL | 成交量 |
| `rsi_14` | REAL | RSI 指标 |
| `macd` | REAL | MACD 指标 |
| `macd_signal` | REAL | MACD 信号线 |
| `macd_hist` | REAL | MACD 柱状图 |

### 5.2 索引设计

```sql
CREATE INDEX idx_tlt_timestamp ON market_data_tlt (timestamp DESC);
CREATE INDEX idx_tlt_date ON market_data_tlt (date);
```

---

## 6. 策略参数配置

### 6.1 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sma_window` | 60 | 宏观因子 SMA 窗口 |
| `sma_standard` | 50 | 标准均线（趋势判断） |
| `sma_slow` | 200 | 慢速均线（趋势判断） |
| `bull_floor_value` | 0.25 | 多头托底敞口 |
| `cost_rate` | 0.0002 | 单边交易成本 |

### 6.2 回测周期

| 周期 | 起始 | 结束 | 说明 |
|------|------|------|------|
| IS | 2007-01-01 | 2019-12-31 | 样本内 |
| OOS | 2020-01-01 | 2026-04-30 | 样本外 |

---

## 7. 运行方式

### 7.1 数据同步

```bash
# 全量同步（首次运行）
python tlt_ai_fund/sync_tlt_data.py --full

# 增量同步
python tlt_ai_fund/sync_tlt_data.py
```

### 7.2 回测运行

```bash
# IS 回测
python tlt_ai_fund/run_backtest.py --period is

# OOS 回测
python tlt_ai_fund/run_backtest.py --period oos

# 全周期回测
python tlt_ai_fund/run_backtest.py --period full

# IS + OOS 分别回测
python tlt_ai_fund/run_backtest.py --period both
```

---

## 8. 关键设计原则

### 8.1 模块化设计
- 核心逻辑与执行层解耦
- 数据管道与策略逻辑分离
- 便于独立测试和迭代

### 8.2 防御性编程
- 数据完整性检查（adjClose 必须存在）
- 空值处理（ffill 向前填充）
- Look-ahead bias 杜绝（shift(1)）

### 8.3 可扩展性
- 支持轻松添加新的宏观因子
- 策略参数可配置
- 回测引擎支持多种绩效指标

---

## 9. 与 Gold AI Fund 的对比

| 维度 | Gold AI Fund | TLT AI Fund |
|------|--------------|-------------|
| 标的 | 黄金 (GCUSD) | 美债 ETF (TLT) |
| 宏观锚 | 通胀、M2、VIX | 通胀、就业、利率 |
| 分红处理 | 无 | **必须用 adjClose** |
| 避险属性 | 弱 | 强（经济衰退期） |
| 利率敏感性 | 低 | **高** |

---

## 10. 风险提示

1. **利率风险**：TLT 对利率变动高度敏感
2. **数据质量**：依赖外部 API 数据准确性
3. **过度拟合风险**：参数优化需谨慎
4. **流动性风险**：极端市场环境下的交易成本

---

**文档版本**: v1.0  
**创建日期**: 2026-05-04  
**适用范围**: TLT AI Fund 系统