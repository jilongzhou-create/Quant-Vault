#!/bin/bash
set -e

echo "============================================"
echo "  QuantVault SaaS - Server Setup Script"
echo "============================================"

apt update && apt upgrade -y
apt install -y python3 python3-pip python3-venv git curl

APP_DIR=/opt/quantvault
mkdir -p $APP_DIR

useradd -m -s /bin/bash quantvault 2>/dev/null || true
chown -R quantvault:quantvault $APP_DIR

sudo -u quantvault bash << 'USER_SCRIPT'
cd /opt/quantvault

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install streamlit pandas numpy requests plotly cryptography supabase

cat > /etc/systemd/system/quantvault.service << 'SERVICE'
[Unit]
Description=QuantVault SaaS Platform
After=network.target

[Service]
Type=simple
User=quantvault
WorkingDirectory=/opt/quantvault
ExecStart=/opt/quantvault/venv/bin/streamlit run saas_platform/web_frontend/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false
Restart=always
RestartSec=5
Environment=PYTHONUTF8=1

[Install]
WantedBy=multi-user.target
SERVICE

cat > /etc/systemd/system/quantvault-daily.service << 'CRON_SERVICE'
[Unit]
Description=QuantVault Daily Data Sync

[Service]
Type=oneshot
User=quantvault
WorkingDirectory=/opt/quantvault
ExecStart=/opt/quantvault/venv/bin/python -m ops_scripts.init_saas_data
Environment=PYTHONUTF8=1
CRON_SERVICE

echo "0 8 * * * quantvault /usr/bin/systemctl start quantvault-daily" > /etc/cron.d/quantvault-daily
chmod 644 /etc/cron.d/quantvault-daily

systemctl daemon-reload
systemctl enable quantvault
systemctl start quantvault

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo "  Website: http://YOUR_SERVER_IP:8501"
echo "  Status:  systemctl status quantvault"
echo "  Logs:    journalctl -u quantvault -f"
echo "  Daily:   Runs at 08:00 UTC daily"
echo "============================================"
USER_SCRIPT
