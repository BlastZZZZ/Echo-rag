import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ..prompts import PromptTemplateManager
from ..utils.llm_utils import fix_broken_generated_json
from ..utils.logging_utils import get_logger
from ..utils.misc_utils import NerRawOutput, PropositionRawOutput

logger = get_logger(__name__)


@dataclass
class Proposition:
    text: str
    entities: List[str]


class PropositionExtractor:
    """Extract fully contextualized natural-language propositions from passages."""

    def __init__(self, llm_model):
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        self.llm_model = llm_model

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(str(text).split()).strip()

    @classmethod
    def _normalize_entities(
        cls,
        entities: List[Any],
        allowed_entities: Optional[List[str]] = None,
    ) -> List[str]:
        allowed_lookup = None
        if allowed_entities is not None:
            allowed_lookup = {cls._normalize_text(entity): entity for entity in allowed_entities if cls._normalize_text(entity)}

        normalized_entities: List[str] = []
        for entity in entities or []:
            normalized = cls._normalize_text(entity)
            if not normalized:
                continue
            if allowed_lookup is not None:
                if normalized not in allowed_lookup:
                    continue
                normalized = allowed_lookup[normalized]
            if normalized not in normalized_entities:
                normalized_entities.append(normalized)
        return normalized_entities

    @classmethod
    def _parse_response_payload(
        cls,
        payload: Any,
        allowed_entities: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if isinstance(payload, dict):
            propositions = payload.get("propositions", [])
        elif isinstance(payload, list):
            propositions = payload
        else:
            propositions = []

        parsed: List[Dict[str, Any]] = []
        seen_signatures = set()
        for proposition in propositions:
            if not isinstance(proposition, dict):
                continue
            text = cls._normalize_text(proposition.get("text", ""))
            if not text:
                continue
            entities = cls._normalize_entities(proposition.get("entities", []), allowed_entities=allowed_entities)
            signature = (text.lower(), tuple(sorted(entity.lower() for entity in entities)))
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            parsed.append({"text": text, "entities": entities})
        return parsed

    @classmethod
    def _extract_propositions_from_response(
        cls,
        response: str,
        allowed_entities: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        candidates = [response]
        fixed_response = fix_broken_generated_json(response)
        if fixed_response != response:
            candidates.append(fixed_response)

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
                parsed = cls._parse_response_payload(payload, allowed_entities=allowed_entities)
                if parsed:
                    return parsed
            except (TypeError, json.JSONDecodeError):
                continue

        return []

    def _infer_messages(self, messages, max_completion_tokens: int = 768):
        result = self.llm_model.infer(messages=messages, max_completion_tokens=max_completion_tokens)
        if isinstance(result, tuple) and len(result) == 3:
            response, metadata, cache_hit = result
            metadata = dict(metadata or {})
            metadata["cache_hit"] = cache_hit
            return response, metadata
        if isinstance(result, tuple) and len(result) == 2:
            response, metadata = result
            return response, dict(metadata or {})
        raise TypeError("Unexpected llm_model.infer return format")

    def extract_propositions(
        self,
        chunk_key: str,
        passage: str,
        named_entities: Optional[List[str]] = None,
    ) -> PropositionRawOutput:
        messages = self.prompt_template_manager.render(
            name="proposition_extraction",
            passage=passage,
            named_entities=json.dumps(named_entities or [], ensure_ascii=False),
        )

        raw_response = ""
        metadata: Dict[str, Any] = {}
        try:
            raw_response, metadata = self._infer_messages(messages)
            if metadata.get("finish_reason") == "length":
                raw_response = fix_broken_generated_json(raw_response)
            propositions = self._extract_propositions_from_response(
                raw_response,
                allowed_entities=named_entities,
            )
        except Exception as exc:
            logger.warning(f"Failed proposition extraction for chunk {chunk_key}: {exc}")
            metadata = dict(metadata or {})
            metadata["error"] = str(exc)
            propositions = []

        return PropositionRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            propositions=propositions,
            metadata=metadata,
        )

    def batch_extract_propositions(
        self,
        chunk_passages: Dict[str, Dict[str, Any]],
        ner_results_dict: Dict[str, NerRawOutput],
    ) -> Dict[str, PropositionRawOutput]:
        if not chunk_passages:
            return {}

        proposition_results: List[PropositionRawOutput] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.extract_propositions,
                    chunk_key,
                    row["content"],
                    ner_results_dict.get(chunk_key, NerRawOutput(chunk_key, "", [], {})).unique_entities,
                ): chunk_key
                for chunk_key, row in chunk_passages.items()
            }
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Extracting propositions")
            for future in pbar:
                result = future.result()
                proposition_results.append(result)
                metadata = result.metadata or {}
                total_prompt_tokens += metadata.get("prompt_tokens", 0)
                total_completion_tokens += metadata.get("completion_tokens", 0)
                if metadata.get("cache_hit"):
                    num_cache_hit += 1
                pbar.set_postfix(
                    {
                        "total_prompt_tokens": total_prompt_tokens,
                        "total_completion_tokens": total_completion_tokens,
                        "num_cache_hit": num_cache_hit,
                    }
                )

        return {result.chunk_id: result for result in proposition_results}
