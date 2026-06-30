import os

from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig

from .openai_gpt import CacheOpenAI
from .base import BaseLLM
from .transformers_llm import TransformersLLM


logger = get_logger(__name__)


def _get_llm_class(config: BaseConfig):
    local_base_url = str(config.llm_base_url or "")
    if (
        config.llm_base_url is not None
        and any(host in local_base_url for host in ("localhost", "127.0.0.1", "0.0.0.0"))
        and os.getenv('OPENAI_API_KEY') is None
    ):
        os.environ['OPENAI_API_KEY'] = 'sk-'

    if config.llm_name.startswith('bedrock'):
        from .bedrock_llm import BedrockLLM
        return BedrockLLM(config)
    
    if config.llm_name.startswith('Transformers/'):
        return TransformersLLM(config)
    
    return CacheOpenAI.from_experiment_config(config)
    
