#!/usr/bin/env python3
"""
Gemini Vertex AI 连接测试脚本
支持两种 VPN 模式：
  1. Clash TUN 全局代理 -> 直连（清空代理变量）
  2. 飞鸟 VPN 等独立代理 -> 设置 HTTP/HTTPS 代理
启动时自动检测，也可手动切换
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

PROXY_URL = os.getenv('HTTPS_PROXY') or os.getenv('HTTP_PROXY') or os.getenv('https_proxy') or os.getenv('http_proxy') or ''

from google import genai
from google.genai.types import HttpOptions
from google.oauth2 import service_account

from config import GCP_PROJECT_ID, GCP_LOCATION, GOOGLE_APPLICATION_CREDENTIALS

MODEL_NAME = "gemini-3.1-pro-preview"

current_mode = None


def set_proxy_mode(mode):
    global current_mode
    proxy_keys = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']

    if mode == 'direct':
        for key in proxy_keys:
            os.environ.pop(key, None)
        os.environ['no_proxy'] = '*'
        os.environ['NO_PROXY'] = '*'
        current_mode = 'direct'
        print(f"  Mode: DIRECT (TUN/Clash) - proxy vars cleared, NO_PROXY=*")

    elif mode == 'proxy':
        for key in proxy_keys:
            os.environ.pop(key, None)
        if PROXY_URL:
            os.environ['http_proxy'] = PROXY_URL
            os.environ['https_proxy'] = PROXY_URL
            os.environ['HTTP_PROXY'] = PROXY_URL
            os.environ['HTTPS_PROXY'] = PROXY_URL
        os.environ.pop('no_proxy', None)
        os.environ.pop('NO_PROXY', None)
        current_mode = 'proxy'
        print(f"  Mode: PROXY (飞鸟VPN) - {PROXY_URL}")

    else:
        print(f"  Unknown mode: {mode}")


def init_client():
    print(f"  Project: {GCP_PROJECT_ID}")
    print(f"  Location: {GCP_LOCATION}")
    print(f"  Credentials: {GOOGLE_APPLICATION_CREDENTIALS}")
    print(f"  Key file exists: {os.path.exists(GOOGLE_APPLICATION_CREDENTIALS)}")
    print()

    if os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
        credentials = service_account.Credentials.from_service_account_file(
            GOOGLE_APPLICATION_CREDENTIALS,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        client = genai.Client(
            http_options=HttpOptions(api_version="v1", timeout=300000.0),
            vertexai=True,
            project=GCP_PROJECT_ID,
            location=GCP_LOCATION,
            credentials=credentials
        )
    else:
        client = genai.Client(
            http_options=HttpOptions(api_version="v1", timeout=300000.0),
            vertexai=True,
            project=GCP_PROJECT_ID,
            location=GCP_LOCATION
        )

    print("  Client initialized OK!")
    return client


def chat(client, user_input):
    print("\n  [Gemini thinking...]")
    try:
        response_stream = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=user_input
        )
        full_text = ""
        for chunk in response_stream:
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_text += chunk.text
        print()
        return full_text
    except Exception as e:
        print(f"\n  Error: {e}")
        return None


def main():
    print("=" * 60)
    print("  Gemini Vertex AI Connection Test")
    print("=" * 60)

    print(f"\n  .env proxy config: {PROXY_URL or '(empty)'}")
    print()
    print("  Select proxy mode:")
    print("  [1] DIRECT  - Clash TUN, clear all proxy vars")
    print("  [2] PROXY   - 飞鸟VPN, use .env proxy settings")
    print()

    try:
        choice = input("  Choice [1/2]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Bye!")
        return

    if choice == '2':
        set_proxy_mode('proxy')
    else:
        set_proxy_mode('direct')

    print()
    try:
        client = init_client()
    except Exception as e:
        print(f"\n  Init failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("  Interactive mode:")
    print("    Type message -> chat with Gemini")
    print("    'mode'      -> switch proxy mode")
    print("    'quit'      -> exit")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("  You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("  Bye!")
            break

        if user_input.lower() == 'mode':
            print()
            print("  Switch proxy mode:")
            print("  [1] DIRECT  [2] PROXY")
            try:
                mc = input("  Choice: ").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            new_mode = 'proxy' if mc == '2' else 'direct'
            set_proxy_mode(new_mode)
            try:
                client = init_client()
            except Exception as e:
                print(f"\n  Re-init failed: {e}")
            print()
            continue

        chat(client, user_input)
        print()


if __name__ == "__main__":
    main()
