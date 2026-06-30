from typing import List
import numpy as np
from tqdm import tqdm
import time

from .base import BaseEmbeddingModel, prefix_queries_with_instruction
from ..utils.config_utils import BaseConfig
from ..prompts.linking import get_query_instruction
from ..utils.logging_utils import get_logger
import requests

logger = get_logger(__name__)

class VLLMEmbeddingModel(BaseEmbeddingModel):
    """
    To select this implementation you can initialise HippoRAG with:
        embedding_model_name starts with "VLLM/"
    The embedding base url should contain the v1/embeddings.
    """
    def __init__(self, global_config:BaseConfig, embedding_model_name:str) -> None:
        super().__init__(global_config=global_config)

        self.model_id = embedding_model_name[len("VLLM/"):]
        self.embedding_type = 'float'
        self.batch_size = getattr(global_config, "embedding_batch_size", 32) or 32
        self.max_retry_attempts = max(1, getattr(global_config, "max_retry_attempts", 5) or 5)

        self.base_url = global_config.embedding_base_url
        if self.base_url is None:
            raise ValueError("VLLMEmbeddingModel requires `embedding_base_url` to point to an OpenAI-compatible embeddings endpoint.")

        self.search_query_instr = set([
            get_query_instruction('query_to_fact'),
            get_query_instruction('query_to_hyperedge'),
            get_query_instruction('query_to_passage')
        ])

    def call_model(self, input_text) -> List[np.ndarray]:
        if isinstance(input_text, str):
            input_text = [input_text]
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_id,
            "input": input_text,
        }

        response = requests.post(self.base_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return np.array([result["data"][i]["embedding"] for i in range(len(result["data"]))])

    def encode(self, texts: List[str]) -> np.array:
        response = self.call_model(texts)
        return response

    def _encode_with_retries(self, texts: List[str]) -> np.ndarray:
        delay_seconds = 0.5
        last_exc = None
        for attempt in range(1, self.max_retry_attempts + 1):
            try:
                return self.encode(texts)
            except requests.HTTPError as exc:
                last_exc = exc
                if attempt == self.max_retry_attempts:
                    break
                logger.warning(
                    "Embedding request to %s failed for batch size %s on attempt %s/%s; retrying in %.1fs.",
                    self.base_url,
                    len(texts),
                    attempt,
                    self.max_retry_attempts,
                    delay_seconds,
                )
                time.sleep(delay_seconds)
                delay_seconds = min(delay_seconds * 2, 4.0)
        raise last_exc

    def _encode_with_batch_fallback(self, texts: List[str]) -> np.ndarray:
        try:
            return self._encode_with_retries(texts)
        except requests.HTTPError as exc:
            if len(texts) <= 1:
                raise
            split = max(1, len(texts) // 2)
            logger.warning(
                "Embedding batch of size %s failed against %s; retrying in smaller chunks.",
                len(texts),
                self.base_url,
            )
            left = self._encode_with_batch_fallback(texts[:split])
            right = self._encode_with_batch_fallback(texts[split:])
            return np.concatenate([left, right], axis=0)

    def batch_encode(self, texts: List[str], **kwargs) -> None:
        if isinstance(texts, str):
            texts = [texts]
        texts_to_encode = prefix_queries_with_instruction(list(texts), kwargs.get("instruction", ""))
        if len(texts_to_encode) < self.batch_size:
            return self._encode_with_batch_fallback(texts_to_encode)
        
        results = []
        batch_indexes = list(range(0, len(texts_to_encode), self.batch_size))
        for i in tqdm(batch_indexes, desc="Batch Encoding"):
            results.append(self._encode_with_batch_fallback(texts_to_encode[i:i + self.batch_size]))
        return np.concatenate(results)
