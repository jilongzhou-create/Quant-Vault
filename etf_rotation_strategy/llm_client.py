#!/usr/bin/env python3
"""
LLM API 调用封装 - Google Gemini (Vertex AI)

基于 google-genai 官方库，使用流式请求防断连。
配置来源: 根目录 .env + config.py (GCP_PROJECT_ID, GCP_LOCATION, vertex_key.json)
"""

import os
import json
import re
import time
import logging
from typing import Dict, List, Optional

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

from google import genai
from google.genai.types import HttpOptions
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-3.1-pro-preview"

_GCP_PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT', 'synthmind-social-content')
_GCP_LOCATION = os.environ.get('GCP_LOCATION', 'global')
_GOOGLE_APPLICATION_CREDENTIALS = os.path.join(project_root, 'vertex_key.json')

_MAX_RETRIES = 3
_RETRY_DELAY = 15
_HTTP_TIMEOUT = 180000.0


def _get_gemini_client() -> genai.Client:
    if os.path.exists(_GOOGLE_APPLICATION_CREDENTIALS):
        credentials = service_account.Credentials.from_service_account_file(
            _GOOGLE_APPLICATION_CREDENTIALS,
            scopes=['https://www.googleapis.com/auth/cloud-platform'],
        )
        client = genai.Client(
            http_options=HttpOptions(api_version="v1", timeout=_HTTP_TIMEOUT),
            vertexai=True,
            project=_GCP_PROJECT_ID,
            location=_GCP_LOCATION,
            credentials=credentials,
        )
    else:
        client = genai.Client(
            http_options=HttpOptions(api_version="v1", timeout=_HTTP_TIMEOUT),
            vertexai=True,
            project=_GCP_PROJECT_ID,
            location=_GCP_LOCATION,
        )
    return client


def generate_response(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    调用 Gemini API 生成回复 (流式请求)

    将 OpenAI 风格的 messages 列表转换为 Gemini 的 contents 格式:
      - system 消息合并到第一条 user 消息前缀
      - user/assistant 交替排列

    Args:
        messages:     对话消息列表，格式 [{"role": "system/user/assistant", "content": "..."}]
        model:        模型名称，None 使用默认值
        temperature:  生成温度
        max_tokens:   最大生成 token 数

    Returns:
        str: LLM 生成的文本内容

    Raises:
        RuntimeError: 重试次数耗尽后仍失败
    """
    model = model or MODEL_NAME

    from google.genai import types

    system_parts: List[str] = []
    contents: List[types.Content] = []

    for msg in messages:
        role = msg['role']
        text = msg['content']
        if role == 'system':
            system_parts.append(text)
        elif role == 'user':
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text=text)],
            ))
        elif role == 'assistant':
            contents.append(types.Content(
                role="model",
                parts=[types.Part.from_text(text=text)],
            ))

    config = types.GenerateContentConfig()
    if system_parts:
        config.system_instruction = types.Content(
            parts=[types.Part.from_text(text="\n\n".join(system_parts))],
        )
    if temperature is not None:
        config.temperature = temperature
    if max_tokens is not None:
        config.max_output_tokens = max_tokens

    last_error = None
    for attempt in range(_MAX_RETRIES):
        try:
            logger.info(f"[LLM] 调用模型: {model}, 尝试 {attempt + 1}/{_MAX_RETRIES}")

            client = _get_gemini_client()

            response_stream = client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            )

            response_text = ""
            for chunk in response_stream:
                if chunk.text:
                    response_text += chunk.text

            logger.info(f"[LLM] 生成成功, 长度={len(response_text)}")
            return response_text

        except Exception as e:
            last_error = e
            err_type = type(e).__name__
            err_msg = str(e)
            is_network_error = any(kw in err_type.lower() + err_msg.lower() for kw in
                                   ['ssl', 'timeout', 'eof', 'connection', 'reset', 'broken'])

            if is_network_error:
                logger.warning(f"[LLM] 网络错误 (尝试 {attempt + 1}/{_MAX_RETRIES}): "
                               f"{err_type}: {err_msg[:200]}")
            else:
                logger.warning(f"[LLM] 调用失败 (尝试 {attempt + 1}/{_MAX_RETRIES}): "
                               f"{err_type}: {err_msg[:200]}")

            if attempt < _MAX_RETRIES - 1:
                wait = _RETRY_DELAY * (attempt + 1)
                logger.info(f"[LLM] 等待 {wait}s 后重试...")
                time.sleep(wait)

    raise RuntimeError(f"LLM 调用失败，已达最大重试次数: {last_error}")


def parse_json_response(text: str) -> Dict:
    """
    鲁棒解析 LLM 返回的 JSON

    依次尝试: 直接解析 -> 提取 ```json 块 -> 提取 {} 之间内容

    Args:
        text: LLM 返回的原始文本

    Returns:
        dict: 解析后的字典

    Raises:
        ValueError: 所有解析方式均失败
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        try:
            return json.loads(text[start_idx:end_idx + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"无法解析 LLM 返回的 JSON: {text[:500]}")
