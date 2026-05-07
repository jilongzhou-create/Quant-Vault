"""
API Key 加密/解密工具模块

使用 cryptography.fernet 对称加密，MASTER_KEY 来自 saas_config
"""

import os
import sys
import base64
import hashlib
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from saas_platform.saas_config import MASTER_KEY

logger = logging.getLogger('saas_platform.web_frontend.crypto_utils')


def _get_fernet():
    from cryptography.fernet import Fernet

    if not MASTER_KEY:
        raise RuntimeError("MASTER_KEY 未配置，无法进行加密操作。请在 .env 中设置 SAAS_MASTER_KEY 或 MASTER_KEY")

    key_bytes = MASTER_KEY.encode('utf-8')
    if len(key_bytes) == 44 and base64.urlsafe_b64encode(base64.urlsafe_b64decode(key_bytes)) == key_bytes:
        fernet_key = key_bytes
    else:
        fernet_key = base64.urlsafe_b64encode(hashlib.sha256(key_bytes).digest())

    return Fernet(fernet_key)


def encrypt_api_key(plaintext: str) -> str:
    if not plaintext:
        return ''
    f = _get_fernet()
    return f.encrypt(plaintext.encode('utf-8')).decode('utf-8')


def decrypt_api_key(ciphertext: str) -> str:
    if not ciphertext:
        return ''
    f = _get_fernet()
    return f.decrypt(ciphertext.encode('utf-8')).decode('utf-8')
