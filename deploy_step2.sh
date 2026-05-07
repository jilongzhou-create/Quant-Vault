#!/bin/bash
set -e

APP_DIR=/opt/quantvault
cd $APP_DIR

echo "============================================"
echo "  QuantVault SaaS - Step 2: Configure & Start"
echo "============================================"

if [ ! -f .env ]; then
    echo "ERROR: .env file not found! Run deploy_step1.sh first."
    exit 1
fi

cat > /etc/systemd/system/quantvault.service << 'SERVICE'
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
SERVICE

cat > /etc/systemd/system/quantvault-daily.service << 'DAILYSERVICE'
[Unit]
Description=QuantVault Daily Data Sync

[Service]
Type=oneshot
WorkingDirectory=/opt/quantvault
ExecStart=/opt/quantvault/venv/bin/python -m ops_scripts.init_saas_data
Environment=PYTHONUTF8=1
EnvironmentFile=/opt/quantvault/.env
DAILYSERVICE

echo "0 8 * * * root /usr/bin/systemctl start quantvault-daily" > /etc/cron.d/quantvault-daily
chmod 644 /etc/cron.d/quantvault-daily

ufw allow 22/tcp
ufw allow 8501/tcp
ufw --force enable

systemctl daemon-reload
systemctl enable quantvault
systemctl start quantvault

sleep 3

echo ""
echo "============================================"
echo "  Deploy Complete!"
echo "============================================"
echo ""
echo "  Website:  http://202.182.125.182:8501"
echo "  Status:   systemctl status quantvault"
echo "  Logs:     journalctl -u quantvault -f"
echo "  Daily:    Runs at 08:00 UTC (17:00 JST)"
echo ""
echo "  To initialize data, run:"
echo "  cd /opt/quantvault && source venv/bin/activate"
echo "  export \$(cat .env | xargs)"
echo "  python -m ops_scripts.init_saas_data"
echo "============================================"
