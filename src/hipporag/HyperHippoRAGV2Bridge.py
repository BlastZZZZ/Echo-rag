import hashlib
import importlib.util
import json
import logging
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import MISSING, asdict, fields as dataclass_fields
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from .HippoRAG import HippoRAG
from .evaluation.qa_eval import QAExactMatch, QAF1Score
from .evaluation.retrieval_eval import RetrievalRecall
from .utils.misc_utils import QuerySolution


logger = logging.getLogger(__name__)


DEFAULT_HYPERHIPPO_V2_DIR = os.environ.get("HYPERHIPPO_V2_DIR", "external/hyperhippo_v2")

_EXTERNAL_MODULE_SPECS = [
    ("hyperhippo_v2_external_config", "config.py", "config"),
    ("hyperhippo_v2_external_hypergraph", "hypergraph.py", "hypergraph"),
    ("hyperhippo_v2_external_query_conditioned_ppr", "query_conditioned_ppr.py", "query_conditioned_ppr"),
    ("hyperhippo_v2_external_hypergraph_builder", "hypergraph_builder.py", "hypergraph_builder"),
    ("hyperhippo_v2_external_proposition_extractor", "proposition_extractor.py", "proposition_extractor"),
    ("hyperhippo_v2_external_main", "hyperhippo_v2.py", "hyperhippo_v2"),
]

_V2_INDEX_RELEVANT_PATHS = [
    "config.py",
    "hypergraph.py",
    "hypergraph_builder.py",
    "proposition_extractor.py",
]

_PRESERVE_EXTERNAL_DEFAULT_FIELDS = frozenset({
    "hyperedge_rerank_top_k",
})
_UNSET = object()


def _extract_propositions_with_ordered_persistence(
    *,
    docs: Sequence[str],
    start_idx: int,
    extract_fn: Callable[[str], List[str]],
    persist_fn: Callable[[int, str, List[str]], None],
    desc: str,
    num_workers: int,
    prefetch: Optional[int] = None,
) -> List[List[str]]:
    total = len(docs)
    if start_idx >= total:
        return []

    num_workers = max(1, int(num_workers))
    prefetch = max(num_workers, int(prefetch if prefetch is not None else num_workers * 2))
    new_propositions: List[List[str]] = []

    if num_workers == 1:
        for passage_idx in tqdm(
            range(start_idx, total),
            desc=desc,
            total=total,
            initial=start_idx,
        ):
            passage = docs[passage_idx]
            propositions = extract_fn(passage)
            persist_fn(passage_idx, passage, propositions)
            new_propositions.append(propositions)
        return new_propositions

    logger.info(
        "Running %s with %d workers and prefetch=%d",
        desc.lower(),
        num_workers,
        prefetch,
    )

    next_submit_idx = start_idx
    next_write_idx = start_idx
    pending = {}

    def submit_ready_jobs(executor: ThreadPoolExecutor):
        nonlocal next_submit_idx
        while next_submit_idx < total and len(pending) < prefetch:
            passage_idx = next_submit_idx
            pending[passage_idx] = executor.submit(extract_fn, docs[passage_idx])
            next_submit_idx += 1

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        submit_ready_jobs(executor)
        with tqdm(total=total, initial=start_idx, desc=desc) as progress:
            while next_write_idx < total:
                future = pending.pop(next_write_idx)
                propositions = future.result()
                passage = docs[next_write_idx]
                persist_fn(next_write_idx, passage, propositions)
                new_propositions.append(propositions)
                next_write_idx += 1
                progress.update(1)
                submit_ready_jobs(executor)

    return new_propositions


def _hash_jsonable(payload) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.md5(encoded).hexdigest()


def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _file_sha256(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_external_module(unique_name: str, file_path: str, alias_name: str):
    existing = sys.modules.get(unique_name)
    if existing is not None and getattr(existing, "__file__", None) == file_path:
        return existing

    spec = importlib.util.spec_from_file_location(unique_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create module spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    sys.modules[alias_name] = module
    spec.loader.exec_module(module)
    return module


def _get_dataclass_default(instance, field_name: str):
    dataclass_field_map = getattr(instance, "__dataclass_fields__", None)
    if not isinstance(dataclass_field_map, dict):
        return _UNSET

    field_def = dataclass_field_map.get(field_name)
    if field_def is None:
        return _UNSET
    if field_def.default is not MISSING:
        return field_def.default
    if field_def.default_factory is not MISSING:
        return field_def.default_factory()
    return _UNSET


def _project_external_config_kwargs(config_cls, global_config, preserve_external_default_fields=()):
    projected_kwargs = {}
    for field_def in dataclass_fields(config_cls):
        if not hasattr(global_config, field_def.name):
            continue

        value = getattr(global_config, field_def.name)
        if field_def.name in preserve_external_default_fields:
            default_value = _get_dataclass_default(global_config, field_def.name)
            if default_value is not _UNSET and value == default_value:
                continue

        projected_kwargs[field_def.name] = value

    return projected_kwargs


def load_hyperhippo_v2_modules(source_dir: str) -> Dict[str, object]:
    normalized_source_dir = os.path.abspath(source_dir)
    modules = {}
    for unique_name, relative_path, alias_name in _EXTERNAL_MODULE_SPECS:
        file_path = os.path.join(normalized_source_dir, relative_path)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Missing HyperHippoRAG v2 file: {file_path}")
        modules[alias_name] = _load_external_module(unique_name, file_path, alias_name)
    return modules


class V2LLMAdapter:
    """Adapter that exposes generate(prompt) without changing the external v2 code."""

    def __init__(self, llm_model):
        self._llm_model = llm_model

    def generate(self, prompt: str, **kwargs) -> str:
        infer_kwargs = {"messages": [{"role": "user", "content": prompt}]}
        infer_kwargs.update({key: value for key, value in kwargs.items() if value is not None})
        try:
            result = self._llm_model.infer(**infer_kwargs)
        except Exception:
            if "response_format" not in infer_kwargs:
                raise
            infer_kwargs.pop("response_format", None)
            result = self._llm_model.infer(**infer_kwargs)
        if isinstance(result, tuple):
            if len(result) >= 1:
                return result[0]
        if isinstance(result, str):
            return result
        raise TypeError(f"Unexpected LLM generate result type: {type(result)!r}")


class V2EmbeddingAdapter:
    """Adapter that exposes encode(texts) and normalizes return shapes for the external v2 code."""

    def __init__(self, embedding_model):
        self._embedding_model = embedding_model

    def encode(self, texts):
        is_single = isinstance(texts, str)
        texts_list = [texts] if is_single else list(texts)

        if hasattr(self._embedding_model, "encode"):
            embeddings = self._embedding_model.encode(texts_list)
        else:
            embeddings = self._embedding_model.batch_encode(texts_list)

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            return embeddings
        if is_single:
            return embeddings[0]
        return embeddings


class HyperHippoRAGV2Bridge(HippoRAG):
    """
    Bridge class that runs the external HyperHippoRAG v2 core inside HippoRAG's evaluation shell.

    The external algorithm files are loaded as-is from source_dir. This wrapper only adapts
    interfaces and caching so the system can be evaluated with HippoRAG's existing metrics.
    """

    def __init__(self, *args, source_dir: str = DEFAULT_HYPERHIPPO_V2_DIR, **kwargs):
        self.external_source_dir = os.path.abspath(source_dir)
        super().__init__(*args, **kwargs)
        self._external_modules = load_hyperhippo_v2_modules(self.external_source_dir)
        self._v2_llm_adapter = V2LLMAdapter(self.llm_model)
        self._v2_embedding_adapter = V2EmbeddingAdapter(self.embedding_model)
        self._v2_config = self._build_external_config()
        self._v2_retriever = None
        self._v2_hypergraph = None
        self._indexed_docs: List[str] = []
        self._cache_meta_path = os.path.join(self.working_dir, "hyperhippo_v2_bridge_meta.json")
        self._cache_props_path = os.path.join(self.working_dir, "hyperhippo_v2_bridge_propositions.json")
        self._cache_props_jsonl_path = os.path.join(self.working_dir, "hyperhippo_v2_bridge_propositions.jsonl")
        self._cache_graph_path = os.path.join(self.working_dir, "hyperhippo_v2_bridge_graph.pkl")

    def _source_fingerprint(self) -> str:
        file_hashes = {}
        for _, relative_path, _ in _EXTERNAL_MODULE_SPECS:
            file_path = os.path.join(self.external_source_dir, relative_path)
            file_hashes[relative_path] = _file_sha256(file_path)
        return _hash_jsonable(file_hashes)

    def _index_fingerprint(self) -> str:
        file_hashes = {}
        for relative_path in _V2_INDEX_RELEVANT_PATHS:
            file_path = os.path.join(self.external_source_dir, relative_path)
            file_hashes[relative_path] = _file_sha256(file_path)
        return _hash_jsonable(file_hashes)

    def _build_external_config(self):
        config_cls = self._external_modules["config"].HyperHippoConfig
        projected_kwargs = _project_external_config_kwargs(
            config_cls,
            self.global_config,
            preserve_external_default_fields=_PRESERVE_EXTERNAL_DEFAULT_FIELDS,
        )
        return config_cls(**projected_kwargs)

    def _index_config_payload(self) -> Dict[str, object]:
        field_names = tuple(getattr(self._external_modules["config"], "INDEX_CONFIG_FIELDS", ()))
        config_payload = asdict(self._v2_config)
        if not field_names:
            return config_payload
        return {field_name: config_payload.get(field_name) for field_name in field_names}

    def _doc_signature(self, docs: Sequence[str]) -> str:
        return _hash_jsonable({"docs": list(docs), "num_docs": len(docs)})

    def _current_cache_meta(self, docs: Sequence[str]) -> Dict[str, object]:
        return {
            "doc_signature": self._doc_signature(docs),
            "source_fingerprint": self._source_fingerprint(),
            "index_fingerprint": self._index_fingerprint(),
            "external_source_dir": self.external_source_dir,
            "index_config": self._index_config_payload(),
        }

    def _load_cached_hypergraph(self, docs: Sequence[str]):
        if self.global_config.force_index_from_scratch:
            return None
        if not (os.path.isfile(self._cache_meta_path) and os.path.isfile(self._cache_graph_path)):
            return None
        with open(self._cache_meta_path, "r", encoding="utf-8") as handle:
            cached_meta = json.load(handle)
        if cached_meta != self._current_cache_meta(docs):
            return None
        with open(self._cache_graph_path, "rb") as handle:
            logger.info("Loading cached HyperHippoRAG v2 hypergraph from %s", self._cache_graph_path)
            return pickle.load(handle)

    def _load_cached_propositions(self, docs: Sequence[str]):
        if self.global_config.force_openie_from_scratch:
            return None
        if not (os.path.isfile(self._cache_meta_path) and os.path.isfile(self._cache_props_path)):
            return None
        with open(self._cache_meta_path, "r", encoding="utf-8") as handle:
            cached_meta = json.load(handle)
        if cached_meta != self._current_cache_meta(docs):
            return None
        with open(self._cache_props_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        propositions = payload.get("propositions_per_passage")
        if isinstance(propositions, list) and len(propositions) == len(docs):
            logger.info("Loading cached HyperHippoRAG v2 propositions from %s", self._cache_props_path)
            return propositions
        return None

    def _write_cache_meta(self, docs: Sequence[str]):
        meta = self._current_cache_meta(docs)
        with open(self._cache_meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, ensure_ascii=False)
        return meta

    def _load_incremental_propositions(self, docs: Sequence[str]):
        if self.global_config.force_openie_from_scratch:
            return []
        if not (os.path.isfile(self._cache_meta_path) and os.path.isfile(self._cache_props_jsonl_path)):
            return []

        with open(self._cache_meta_path, "r", encoding="utf-8") as handle:
            cached_meta = json.load(handle)
        if cached_meta != self._current_cache_meta(docs):
            return []

        propositions_per_passage: List[List[str]] = []
        with open(self._cache_props_jsonl_path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Ignoring malformed incremental proposition cache line %d in %s",
                        line_number,
                        self._cache_props_jsonl_path,
                    )
                    break

                expected_idx = len(propositions_per_passage)
                if record.get("passage_idx") != expected_idx:
                    logger.warning(
                        "Stopping incremental proposition cache load at line %d due to non-sequential passage index",
                        line_number,
                    )
                    break

                if expected_idx >= len(docs):
                    break

                if record.get("passage_hash") != _hash_text(docs[expected_idx]):
                    logger.warning(
                        "Stopping incremental proposition cache load at line %d due to passage hash mismatch",
                        line_number,
                    )
                    break

                propositions = record.get("propositions")
                if not isinstance(propositions, list):
                    logger.warning(
                        "Stopping incremental proposition cache load at line %d due to invalid propositions payload",
                        line_number,
                    )
                    break
                propositions_per_passage.append(propositions)

        if propositions_per_passage:
            logger.info(
                "Loaded %d/%d incremental HyperHippoRAG v2 proposition records from %s",
                len(propositions_per_passage),
                len(docs),
                self._cache_props_jsonl_path,
            )
        return propositions_per_passage

    def _append_incremental_proposition(self, passage_idx: int, passage: str, propositions: List[str]):
        record = {
            "passage_idx": passage_idx,
            "passage_hash": _hash_text(passage),
            "propositions": propositions,
        }
        with open(self._cache_props_jsonl_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    def _save_bridge_cache(self, docs: Sequence[str], propositions_per_passage: List[List[str]], hypergraph):
        self._write_cache_meta(docs)
        with open(self._cache_props_path, "w", encoding="utf-8") as handle:
            json.dump({"propositions_per_passage": propositions_per_passage}, handle, indent=2, ensure_ascii=False)
        with open(self._cache_graph_path, "wb") as handle:
            pickle.dump(hypergraph, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _build_v2_retriever(self):
        retriever_cls = self._external_modules["hyperhippo_v2"].HyperHippoRAG
        return retriever_cls(
            hypergraph=self._v2_hypergraph,
            embedding_model=self._v2_embedding_adapter,
            llm_model=self._v2_llm_adapter,
            config=self._v2_config,
        )

    def index(self, docs: List[str]):
        logger.info("Indexing documents with HyperHippoRAG v2 bridge")
        self._indexed_docs = list(docs)
        self._v2_hypergraph = self._load_cached_hypergraph(docs)

        if self._v2_hypergraph is None:
            propositions_per_passage = self._load_cached_propositions(docs)
            if propositions_per_passage is None:
                propositions_per_passage = self._load_incremental_propositions(docs)
                extractor_cls = self._external_modules["proposition_extractor"].PropositionExtractor
                extractor = extractor_cls(
                    self._v2_llm_adapter,
                    self._v2_embedding_adapter,
                    config=self._v2_config,
                )
                start_idx = len(propositions_per_passage)
                if start_idx > 0:
                    logger.info(
                        "Resuming HyperHippoRAG v2 proposition extraction from passage %d/%d",
                        start_idx,
                        len(docs),
                    )
                self._write_cache_meta(docs)
                new_propositions = _extract_propositions_with_ordered_persistence(
                    docs=docs,
                    start_idx=start_idx,
                    extract_fn=extractor.extract_from_passage,
                    persist_fn=self._append_incremental_proposition,
                    desc="HyperHippo v2 Proposition Extraction",
                    num_workers=getattr(self._v2_config, "proposition_extraction_num_workers", 1),
                    prefetch=getattr(self._v2_config, "proposition_extraction_prefetch", None),
                )
                propositions_per_passage.extend(new_propositions)

            builder_cls = self._external_modules["hypergraph_builder"].HypergraphBuilder
            builder = builder_cls(self._v2_embedding_adapter, config=self._v2_config)
            self._v2_hypergraph = builder.build(propositions_per_passage, list(docs))
            self._save_bridge_cache(docs, propositions_per_passage, self._v2_hypergraph)

        self._v2_retriever = self._build_v2_retriever()
        self.ready_to_retrieve = True

    def retrieve(
        self,
        queries: List[str],
        num_to_retrieve: int = None,
        gold_docs: List[List[str]] = None,
    ):
        if not self.ready_to_retrieve:
            raise RuntimeError("HyperHippoRAGV2Bridge index() must be called before retrieve().")

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        retrieval_results = []
        for query in tqdm(queries, desc="Retrieving", total=len(queries)):
            docs = self._v2_retriever.retrieve(query, num_to_retrieve=num_to_retrieve)
            retrieval_results.append(QuerySolution(question=query, docs=list(docs), doc_scores=None))

        if gold_docs is None:
            return retrieval_results

        evaluator = RetrievalRecall(global_config=self.global_config)
        k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
        overall_metrics, _ = evaluator.calculate_metric_scores(
            gold_docs=gold_docs,
            retrieved_docs=[result.docs for result in retrieval_results],
            k_list=k_list,
        )
        return retrieval_results, overall_metrics

    def rag_qa(
        self,
        queries: List[str | QuerySolution],
        gold_docs: List[List[str]] = None,
        gold_answers: List[List[str]] = None,
    ):
        if not queries:
            if gold_docs is not None and gold_answers is not None:
                return [], [], [], {}, {}
            return [], [], []

        if isinstance(queries[0], QuerySolution):
            query_solutions = list(queries)
            retrieval_metrics = None
        else:
            if gold_docs is not None:
                query_solutions, retrieval_metrics = self.retrieve(queries=queries, gold_docs=gold_docs)
            else:
                query_solutions = self.retrieve(queries=queries)
                retrieval_metrics = None

        query_solutions, all_response_message, all_metadata = self.qa(query_solutions)

        if gold_answers is None:
            return query_solutions, all_response_message, all_metadata

        predicted_answers = [query_solution.answer for query_solution in query_solutions]
        em_eval = QAExactMatch(global_config=self.global_config)
        f1_eval = QAF1Score(global_config=self.global_config)
        em_results, _ = em_eval.calculate_metric_scores(gold_answers=gold_answers, predicted_answers=predicted_answers)
        f1_results, _ = f1_eval.calculate_metric_scores(gold_answers=gold_answers, predicted_answers=predicted_answers)
        overall_qa_results = {**em_results, **f1_results}

        for idx, query_solution in enumerate(query_solutions):
            query_solution.gold_answers = list(gold_answers[idx])
            if gold_docs is not None:
                query_solution.gold_docs = gold_docs[idx]

        if gold_docs is not None:
            return query_solutions, all_response_message, all_metadata, retrieval_metrics, overall_qa_results
        return query_solutions, all_response_message, all_metadata

    def bridge_info(self) -> Dict[str, object]:
        return {
            "external_source_dir": self.external_source_dir,
            "working_dir": self.working_dir,
            "global_config": asdict(self.global_config),
            "external_config": asdict(self._v2_config),
            "num_indexed_docs": len(self._indexed_docs),
            "source_fingerprint": self._source_fingerprint(),
            "index_fingerprint": self._index_fingerprint(),
            "cache_paths": {
                "meta": self._cache_meta_path,
                "propositions": self._cache_props_path,
                "propositions_jsonl": self._cache_props_jsonl_path,
                "graph": self._cache_graph_path,
            },
        }
