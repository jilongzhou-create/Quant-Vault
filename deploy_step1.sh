#!/bin/bash
set -e

echo "============================================"
echo "  QuantVault SaaS - One-Click Deploy"
echo "============================================"

apt update && apt upgrade -y
apt install -y python3 python3-pip python3-venv git

APP_DIR=/opt/quantvault
rm -rf $APP_DIR
mkdir -p $APP_DIR

cd $APP_DIR

git clone https://github.com/jilongzhou-create/Quant-Vault.git .

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install streamlit pandas numpy requests plotly cryptography

cp .env.example .env

echo ""
echo "============================================"
echo "  Step 1 Complete! Now edit .env file"
echo "============================================"
echo ""
echo "  Run this command to edit .env:"
echo "  nano /opt/quantvault/.env"
echo ""
echo "  Fill in your actual API keys, then run:"
echo "  bash /opt/quantvault/deploy_step2.sh"
echo "============================================"
