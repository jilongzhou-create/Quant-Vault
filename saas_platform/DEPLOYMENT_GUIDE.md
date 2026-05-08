# QuantVault SaaS 平台运维手册

> 本文档面向零基础用户，涵盖从本地修改代码到云端部署的完整流程。

---

## 一、项目结构

```
trading_agent/
├── .env                    # 密钥配置（不提交到 Git）
├── .env.example            # 密钥模板（提交到 Git）
├── .gitignore              # Git 忽略规则
├── config.py               # 本地投研系统配置
├── logger_setup.py         # 日志配置
├── deploy_server.sh        # 服务器部署脚本（旧）
├── deploy_step1.sh         # 服务器部署脚本 Step 1
├── deploy_step2.sh         # 服务器部署脚本 Step 2
│
├── saas_platform/          # ★ SaaS 平台核心代码
│   ├── saas_config.py      # 配置网关（读取 .env）
│   ├── database/
│   │   ├── schema.sql      # Supabase 数据库建表 SQL
│   │   └── supabase_client.py  # 数据库操作（增删改查）
│   ├── production_engine/
│   │   ├── signal_engine.py     # 信号计算 + 净值回填引擎
│   │   ├── data_fetcher.py      # 行情/因子数据拉取
│   │   ├── copy_trading_router.py  # 跟单路由
│   │   └── daily_job.py         # 每日定时任务入口
│   └── web_frontend/
│       ├── app.py               # Streamlit 前端主页面
│       └── crypto_utils.py      # API Key 加密工具
│
└── ops_scripts/            # 运维脚本
    ├── publish_strategy_to_saas.py  # 发布策略到云端
    ├── init_saas_data.py            # 一键初始化云端数据
    └── diagnose_saas.py             # 诊断工具
```

---

## 二、关键账号信息

| 服务 | 用途 | 地址 |
|------|------|------|
| GitHub | 代码托管 | https://github.com/jilongzhou-create/Quant-Vault |
| Vultr | 东京 VPS 服务器 | IP: 202.182.125.182 |
| Supabase | 云端数据库 | https://erwgpqvqmiljjndaljap.supabase.co |

---

## 三、日常操作：修改代码并部署到云端

### 3.1 本地修改代码

在 Trae IDE 中正常修改代码即可。

### 3.2 推送代码到 GitHub

在 Trae 终端中运行：

```powershell
git add -A
git commit -m "描述你改了什么"
git push
```

> 如果 `git push` 要求登录，用 GitHub 用户名和 Personal Access Token（不是密码）。

### 3.3 在服务器上更新代码并重启

用 SSH 连接服务器：

```powershell
ssh root@202.182.125.182
```

> 密码输入时屏幕不会显示任何字符，直接粘贴后回车即可。

登录后运行：

```bash
cd /opt/quantvault
git pull
systemctl restart quantvault
```

验证服务是否正常：

```bash
systemctl status quantvault
```

看到 `active (running)` 就成功了。打开浏览器访问 **http://202.182.125.182:8501**

---

## 四、发布新策略到云端

### 4.1 运行发布脚本

在本地 Trae 终端运行：

```powershell
python -m ops_scripts.publish_strategy_to_saas
```

脚本会显示菜单：
1. 查看已有策略
2. 发布新策略
3. 更新已有策略
4. 删除策略

### 4.2 设置实盘切换日期

发布策略后，需要在 Supabase 中设置 `live_start_date`（实盘开始日期）：

1. 打开 https://supabase.com → 你的项目
2. 左侧点击 **SQL Editor**
3. 运行：

```sql
UPDATE saas_strategies
SET live_start_date = '2026-04-07'
WHERE name = '你的策略名称';
```

### 4.3 重新回填实盘净值

在服务器终端运行：

```bash
cd /opt/quantvault
source venv/bin/activate
set -a
source .env
set +a
python /tmp/backfill.py
```

> 如果 `/tmp/backfill.py` 不存在，先创建它（见下方）。

### 4.4 创建 backfill.py 脚本

如果服务器上没有 `/tmp/backfill.py`，在服务器终端**逐行**粘贴：

```bash
cat > /tmp/backfill.py << 'EOF'
import sys
sys.path.insert(0, '/opt/quantvault')
from saas_platform.database.supabase_client import delete_equity_curves, get_public_strategies, get_strategy_equity_curve
from saas_platform.production_engine.signal_engine import CloudSignalEngine

strategies = get_public_strategies()
for s in strategies:
    deleted = delete_equity_curves(s['id'], is_backtest=False)
    print(f'Deleted {s["name"]}: {deleted} records')

engine = CloudSignalEngine()
engine.clear_cache()
result = engine.backfill_historical_nav()
print(f'Backfill result: {result}')

for s in strategies:
    lv = get_strategy_equity_curve(s['id'], is_backtest=False, limit=10000)
    if lv:
        lv_dates = sorted([r.get('date', '?')[:10] for r in lv])
        print(f'{s["name"]}: {len(lv)} live pts, {lv_dates[0]} ~ {lv_dates[-1]}')
EOF
```

---

## 五、数据初始化（完整重置）

如果需要从头初始化所有数据，在服务器终端运行：

```bash
cd /opt/quantvault
source venv/bin/activate
set -a
source .env
set +a
python -m ops_scripts.init_saas_data
```

这会执行：
1. 拉取最新行情+因子数据
2. 清除旧实盘净值并重新回填
3. 计算今日信号
4. 验证数据完整性

---

## 六、服务器管理命令

### 6.1 连接服务器

```powershell
ssh root@202.182.125.182
```

### 6.2 服务管理

| 操作 | 命令 |
|------|------|
| 查看服务状态 | `systemctl status quantvault` |
| 启动服务 | `systemctl start quantvault` |
| 停止服务 | `systemctl stop quantvault` |
| 重启服务 | `systemctl restart quantvault` |
| 查看实时日志 | `journalctl -u quantvault -f` |
| 查看最近20行日志 | `journalctl -u quantvault --no-pager -n 20` |

### 6.3 更新代码

```bash
cd /opt/quantvault
git pull
systemctl restart quantvault
```

### 6.4 手动更新今日数据

```bash
cd /opt/quantvault
source venv/bin/activate
set -a
source .env
set +a
python -m ops_scripts.init_saas_data
```

### 6.5 防火墙

如果网站打不开，检查防火墙：

```bash
ufw allow 8501/tcp
iptables -I INPUT -p tcp --dport 8501 -j ACCEPT
```

### 6.6 每日自动更新

已配置 cron 定时任务，每天 UTC 8:00（日本时间 17:00）自动运行 `init_saas_data`。

查看定时任务：

```bash
cat /etc/cron.d/quantvault-daily
```

---

## 七、Supabase 数据库操作

### 7.1 添加新列

如果代码中新增了数据库字段，需要在 Supabase 中添加：

1. 打开 https://supabase.com → 你的项目
2. 左侧点击 **SQL Editor**
3. 运行 ALTER TABLE 语句，例如：

```sql
ALTER TABLE saas_strategies ADD COLUMN IF NOT EXISTS live_start_date DATE;
```

### 7.2 查看数据

1. 打开 https://supabase.com → 你的项目
2. 左侧点击 **Table Editor**
3. 选择要查看的表

### 7.3 常用表说明

| 表名 | 用途 |
|------|------|
| saas_strategies | 策略信息（名称、代码、指标、live_start_date） |
| saas_equity_curves | 净值曲线（回测+实盘） |
| saas_market_data | 行情数据（OHLCV） |
| saas_factor_data | 因子数据 |
| saas_users | 用户信息 |
| saas_subscriptions | 用户订阅 |

---

## 八、常见问题排查

### Q1: 网站打不开

1. 检查服务是否运行：`systemctl status quantvault`
2. 检查防火墙：`ufw allow 8501/tcp`
3. 检查日志：`journalctl -u quantvault --no-pager -n 20`
4. **注意**：浏览器地址栏必须输入 `http://202.182.125.182:8501`（带 http://，不带 s）

### Q2: 网页显示空白/一直转圈

1. 检查服务器能否连 Supabase：`curl -s -o /dev/null -w "%{http_code}" https://erwgpqvqmiljjndaljap.supabase.co/rest/v1/`
2. 应返回 401（正常，说明网络通）
3. 如果返回 000 或超时，说明服务器网络有问题

### Q3: 图表数据不完整

1. 在服务器上运行数据初始化：`python -m ops_scripts.init_saas_data`
2. 检查 Supabase Table Editor 中 saas_equity_curves 表的数据量

### Q4: git push 失败

1. 检查 GitHub Personal Access Token 是否过期
2. 重新生成：https://github.com/settings/tokens → Generate new token (classic) → 勾选 repo

### Q5: SSH 连不上服务器

1. 在 Vultr 网页上点击服务器 → Console（网页终端）
2. 或者重置密码：Settings → Reset Password

### Q6: 服务器磁盘满了

```bash
df -h
du -sh /opt/quantvault/*
apt clean
```

---

## 九、完整部署流程（从零开始）

如果需要在全新服务器上部署，按以下步骤操作：

### 9.1 创建 Vultr 东京服务器

- Cloud Compute → Tokyo → Ubuntu 24.04 → $6/mo → Deploy

### 9.2 SSH 登录并安装环境

```bash
apt update && apt upgrade -y
apt install -y python3 python3-pip python3-venv git curl
```

### 9.3 拉取代码

```bash
git clone https://github.com/jilongzhou-create/Quant-Vault.git /opt/quantvault
cd /opt/quantvault
python3 -m venv venv
source venv/bin/activate
pip install streamlit pandas numpy requests plotly cryptography python-dotenv
```

### 9.4 配置环境变量

```bash
cp .env.example .env
nano .env
```

填入真实密钥后 Ctrl+O 保存，Ctrl+X 退出。

### 9.5 在 Supabase 执行建表 SQL

打开 Supabase → SQL Editor，粘贴 `saas_platform/database/schema.sql` 的内容并运行。

### 9.6 启动服务

```bash
cat > /etc/systemd/system/quantvault.service << 'EOF'
[Unit]
Description=QuantVault SaaS Platform
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/quantvault
ExecStart=/opt/quantvault/venv/bin/streamlit run saas_platform/web_frontend/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false
Restart=always
RestartSec=5
Environment=PYTHONUTF8=1
EnvironmentFile=/opt/quantvault/.env

[Install]
WantedBy=multi-user.target
EOF
```

```bash
systemctl daemon-reload
systemctl enable quantvault
systemctl start quantvault
```

### 9.7 开放防火墙

```bash
ufw allow 22/tcp
ufw allow 8501/tcp
ufw --force enable
iptables -I INPUT -p tcp --dport 8501 -j ACCEPT
```

### 9.8 初始化数据

```bash
set -a
source .env
set +a
python -m ops_scripts.init_saas_data
```

### 9.9 设置每日自动更新

```bash
cat > /etc/systemd/system/quantvault-daily.service << 'EOF'
[Unit]
Description=QuantVault Daily Data Sync

[Service]
Type=oneshot
WorkingDirectory=/opt/quantvault
ExecStart=/opt/quantvault/venv/bin/python -m ops_scripts.init_saas_data
Environment=PYTHONUTF8=1
EnvironmentFile=/opt/quantvault/.env
EOF
```

```bash
echo "0 8 * * * root /usr/bin/systemctl start quantvault-daily" > /etc/cron.d/quantvault-daily
chmod 644 /etc/cron.d/quantvault-daily
```

---

## 十、安全提醒

1. **永远不要把 `.env` 文件提交到 Git**（已在 .gitignore 中排除）
2. **GitHub 仓库设为 Private**（已设置）
3. **定期更换 Supabase API Key**（如果怀疑泄露）
4. **服务器密码定期更换**（Vultr → Settings → Reset Password）
5. **Binance API Key 只开合约交易权限，禁止提币**
