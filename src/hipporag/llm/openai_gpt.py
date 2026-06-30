import functools
import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from typing import List, Tuple
from urllib.parse import urlparse

import httpx
import openai
from filelock import FileLock
from openai import OpenAI
from openai import AzureOpenAI
from packaging import version
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import (
    TextChatMessage
)
from ..utils.logging_utils import get_logger
from .base import BaseLLM, LLMConfig

logger = get_logger(__name__)


def _is_retryable_openai_error(exc: Exception) -> bool:
    if isinstance(
        exc,
        (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
        ),
    ):
        return True
    if isinstance(exc, openai.APIStatusError):
        status_code = getattr(exc, "status_code", None)
        return status_code is not None and int(status_code) >= 500
    return False


def _should_trust_env_for_base_url(base_url: str | None) -> bool:
    proxy_enabled = os.environ.get("OPENAI_PROXY_ENABLED")
    if proxy_enabled is not None and str(proxy_enabled).strip().lower() in {"0", "false", "no", "off"}:
        return False
    if not base_url:
        return True
    hostname = urlparse(base_url).hostname
    if hostname in {"localhost", "127.0.0.1", "::1"}:
        return False
    return True


def _response_content_to_text(response_message: object, message: object) -> str:
    if isinstance(response_message, str):
        return response_message
    if response_message is None:
        refusal = getattr(message, "refusal", None)
        return refusal if isinstance(refusal, str) else ""
    if isinstance(response_message, list):
        parts = []
        for block in response_message:
            if isinstance(block, str):
                parts.append(block)
                continue
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(text, dict) and isinstance(text.get("value"), str):
                    parts.append(text["value"])
                continue
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
            elif isinstance(getattr(text, "value", None), str):
                parts.append(text.value)
        if parts:
            return "\n".join(parts)
    return str(response_message)


def _qwen_thinking_disabled(global_config: BaseConfig) -> bool:
    return bool(getattr(global_config, "qwen_disable_thinking", False))

def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # get messages from args or kwargs
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        # get model, seed and temperature from kwargs or self.llm_config.generate_params
        gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
        model = kwargs.get("model", gen_params.get("model"))
        seed = kwargs.get("seed", gen_params.get("seed"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))
        extra_body = kwargs.get("extra_body", gen_params.get("extra_body"))
        chat_template_kwargs = kwargs.get("chat_template_kwargs", gen_params.get("chat_template_kwargs"))

        # build key data, convert to JSON string and hash to generate key_hash
        key_data = {
            "messages": messages,  # messages requires JSON serializable
            "model": model,
            "seed": seed,
            "temperature": temperature,
            "extra_body": extra_body,
            "chat_template_kwargs": chat_template_kwargs,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        # the file name of lock, ensure mutual exclusion when accessing concurrently
        lock_file = self.cache_file_name + ".lock"

        # Try to read from SQLite cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # if the table does not exist, create it
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()  # commit to save the table creation
            c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message, metadata_str = row
                metadata = json.loads(metadata_str)
                # return cached result and mark as hit
                return message, metadata, True

        # if cache miss, call the original function to get the result
        result = func(self, *args, **kwargs)
        message, metadata = result

        # insert new result into cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # make sure the table exists again (if it doesn't exist, it would be created)
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            metadata_str = json.dumps(metadata)
            c.execute("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                      (key_hash, message, metadata_str))
            conn.commit()
            conn.close()

        return message, metadata, False

    return wrapper

def dynamic_retry_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        max_retries = getattr(self, "max_retries", 5)  
        dynamic_retry = retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception(_is_retryable_openai_error),
        )
        decorated_func = dynamic_retry(func)
        return decorated_func(self, *args, **kwargs)
    return wrapper

class CacheOpenAI(BaseLLM):
    """OpenAI LLM implementation."""
    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "CacheOpenAI":
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(
            cache_dir=cache_dir,
            global_config=global_config,
            max_retries=global_config.max_retry_attempts,
        )

    def __init__(self, cache_dir, global_config, cache_filename: str = None,
                 high_throughput: bool = True,
                 **kwargs) -> None:

        super().__init__()
        self.cache_dir = cache_dir
        self.global_config = global_config

        self.llm_name = global_config.llm_name
        self.llm_base_url = global_config.llm_base_url

        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_suffix = "_no_think" if _qwen_thinking_disabled(global_config) else ""
            cache_filename = f"{self.llm_name.replace('/', '_')}{cache_suffix}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)

        self._init_llm_config()
        if high_throughput:
            limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
            timeout_seconds = float(getattr(global_config, "llm_timeout_seconds", 5 * 60) or (5 * 60))
            read_timeout_seconds = float(
                getattr(global_config, "llm_read_timeout_seconds", timeout_seconds) or timeout_seconds
            )
            client = httpx.Client(
                limits=limits,
                timeout=httpx.Timeout(timeout_seconds, read=read_timeout_seconds),
                trust_env=_should_trust_env_for_base_url(self.llm_base_url),
            )
        else:
            client = None

        self.max_retries = max(1, kwargs.get("max_retries", global_config.max_retry_attempts))

        if self.global_config.azure_endpoint is None:
            self.openai_client = OpenAI(base_url=self.llm_base_url, http_client=client, max_retries=self.max_retries)
        else:
            self.openai_client = AzureOpenAI(api_version=self.global_config.azure_endpoint.split('api-version=')[1],
                                             azure_endpoint=self.global_config.azure_endpoint, max_retries=self.max_retries)

    def _init_llm_config(self) -> None:
        config_dict = self.global_config.__dict__

        config_dict['llm_name'] = self.global_config.llm_name
        config_dict['llm_base_url'] = self.global_config.llm_base_url
        config_dict['generate_params'] = {
                "model": self.global_config.llm_name,
                "max_completion_tokens": config_dict.get("max_new_tokens", 400),
                "n": config_dict.get("num_gen_choices", 1),
                "seed": config_dict.get("seed", 0),
                "temperature": config_dict.get("temperature", 0.0),
            }
        if _qwen_thinking_disabled(self.global_config):
            config_dict['generate_params']["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s llm_config: {self.llm_config}")

    @cache_response
    @dynamic_retry_decorator
    def infer(
        self,
        messages: List[TextChatMessage],
        **kwargs
    ) -> Tuple[List[TextChatMessage], dict]:
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages
        logger.debug(f"Calling OpenAI GPT API with:\n{params}")

        if 'gpt' not in params['model'] or version.parse(openai.__version__) < version.parse("1.45.0"): # if we use vllm to call openai api or if we use openai but the version is too old to use 'max_completion_tokens' argument
            # TODO strange version change in openai protocol, but our current vllm version not changed yet
            params['max_tokens'] = params.pop('max_completion_tokens')

        response = self.openai_client.chat.completions.create(**params)

        choice = response.choices[0]
        response_message = _response_content_to_text(choice.message.content, choice.message)
        
        metadata = {
            "prompt_tokens": response.usage.prompt_tokens, 
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": choice.finish_reason,
        }

        return response_message, metadata
