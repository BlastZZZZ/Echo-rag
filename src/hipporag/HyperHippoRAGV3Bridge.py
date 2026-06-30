import json
import logging
import os
import pickle
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence

from tqdm import tqdm

from .HippoRAG import HippoRAG
from .HyperHippoRAGV2Bridge import (
    DEFAULT_HYPERHIPPO_V2_DIR,
    V2EmbeddingAdapter,
    V2LLMAdapter,
    _extract_propositions_with_ordered_persistence,
    _EXTERNAL_MODULE_SPECS,
    _PRESERVE_EXTERNAL_DEFAULT_FIELDS,
    _V2_INDEX_RELEVANT_PATHS,
    _file_sha256,
    _hash_jsonable,
    _hash_text,
    _load_external_module,
    _project_external_config_kwargs,
    load_hyperhippo_v2_modules,
)
from .evaluation.qa_eval import QAExactMatch, QAF1Score
from .evaluation.retrieval_eval import RetrievalRecall
from .utils.misc_utils import QuerySolution


logger = logging.getLogger(__name__)


_V3_EXTERNAL_MODULE_SPECS = [
    ("hyperhippo_v3_external_config", "config.py", "config"),
    ("hyperhippo_v3_external_hypergraph", "hypergraph.py", "hypergraph"),
    ("hyperhippo_v3_external_query_conditioned_ppr", "query_conditioned_ppr.py", "query_conditioned_ppr"),
    ("hyperhippo_v3_external_hypergraph_builder", "hypergraph_builder.py", "hypergraph_builder"),
    ("hyperhippo_v3_external_proposition_extractor", "proposition_extractor.py", "proposition_extractor"),
    ("hyperhippo_v3_external_main_v2", "hyperhippo_v2.py", "hyperhippo_v2"),
    ("hyperhippo_v3_external_main_v3", "hyperhippo_v3.py", "hyperhippo_v3"),
]

_INDEX_RELEVANT_PATHS = [
    "proposition_extractor.py",
    "hypergraph.py",
    "hypergraph_builder.py",
]

_LOCAL_CACHE_FILENAMES = {
    "meta": "hyperhippo_v3_bridge_meta.json",
    "props": "hyperhippo_v3_bridge_propositions.json",
    "props_jsonl": "hyperhippo_v3_bridge_propositions.jsonl",
    "graph": "hyperhippo_v3_bridge_graph.pkl",
}

_SHARED_V2_CACHE_FILENAMES = {
    "meta": "hyperhippo_v2_bridge_meta.json",
    "props": "hyperhippo_v2_bridge_propositions.json",
    "props_jsonl": "hyperhippo_v2_bridge_propositions.jsonl",
    "graph": "hyperhippo_v2_bridge_graph.pkl",
}


def load_hyperhippo_v3_modules(source_dir: str) -> Dict[str, object]:
    normalized_source_dir = os.path.abspath(source_dir)
    modules = {}
    for unique_name, relative_path, alias_name in _V3_EXTERNAL_MODULE_SPECS:
        file_path = os.path.join(normalized_source_dir, relative_path)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Missing HyperHippoRAG v3 file: {file_path}")
        modules[alias_name] = _load_external_module(unique_name, file_path, alias_name)
    return modules


class HyperHippoRAGV3Bridge(HippoRAG):
    """
    Bridge class that runs the external HyperHippoRAG v3 core inside HippoRAG's evaluation shell.

    V3 reuses the same proposition / hypergraph memory as V2 when shared_index_dir is provided,
    so V2 vs V3 comparisons can isolate retrieval changes instead of rebuilding the index.
    """

    def __init__(
        self,
        *args,
        source_dir: str = DEFAULT_HYPERHIPPO_V2_DIR,
        shared_index_dir: Optional[str] = None,
        **kwargs,
    ):
        self.external_source_dir = os.path.abspath(source_dir)
        self.shared_index_dir = os.path.abspath(shared_index_dir) if shared_index_dir else None
        super().__init__(*args, **kwargs)
        self._external_modules = load_hyperhippo_v3_modules(self.external_source_dir)
        self._v3_llm_adapter = V2LLMAdapter(self.llm_model)
        self._v3_embedding_adapter = V2EmbeddingAdapter(self.embedding_model)
        self._v3_config = self._build_external_config()
        self._v3_retriever = None
        self._v3_hypergraph = None
        self._indexed_docs: List[str] = []

        self._local_cache_meta_path = os.path.join(self.working_dir, _LOCAL_CACHE_FILENAMES["meta"])
        self._local_cache_props_path = os.path.join(self.working_dir, _LOCAL_CACHE_FILENAMES["props"])
        self._local_cache_props_jsonl_path = os.path.join(self.working_dir, _LOCAL_CACHE_FILENAMES["props_jsonl"])
        self._local_cache_graph_path = os.path.join(self.working_dir, _LOCAL_CACHE_FILENAMES["graph"])

    def _source_fingerprint(self) -> str:
        file_hashes = {}
        for _, relative_path, _ in _V3_EXTERNAL_MODULE_SPECS:
            file_path = os.path.join(self.external_source_dir, relative_path)
            file_hashes[relative_path] = _file_sha256(file_path)
        return _hash_jsonable(file_hashes)

    def _index_fingerprint(self) -> str:
        file_hashes = {}
        for relative_path in _INDEX_RELEVANT_PATHS:
            file_path = os.path.join(self.external_source_dir, relative_path)
            file_hashes[relative_path] = _file_sha256(file_path)
        return _hash_jsonable(file_hashes)

    def _shared_v2_source_fingerprint(self) -> str:
        file_hashes = {}
        for _, relative_path, _ in _EXTERNAL_MODULE_SPECS:
            file_path = os.path.join(self.external_source_dir, relative_path)
            file_hashes[relative_path] = _file_sha256(file_path)
        return _hash_jsonable(file_hashes)

    def _shared_v2_index_fingerprint(self) -> str:
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
        config_payload = asdict(self._v3_config)
        if not field_names:
            return config_payload
        return {field_name: config_payload.get(field_name) for field_name in field_names}

    def _doc_signature(self, docs: Sequence[str]) -> str:
        return _hash_jsonable({"docs": list(docs), "num_docs": len(docs)})

    def _current_local_cache_meta(self, docs: Sequence[str]) -> Dict[str, object]:
        return {
            "doc_signature": self._doc_signature(docs),
            "source_fingerprint": self._source_fingerprint(),
            "index_fingerprint": self._index_fingerprint(),
            "external_source_dir": self.external_source_dir,
            "index_config": self._index_config_payload(),
        }

    def _local_cache_is_valid(self, cached_meta: Dict[str, object], docs: Sequence[str]) -> bool:
        return cached_meta == self._current_local_cache_meta(docs)

    def _shared_cache_is_valid(self, cached_meta: Dict[str, object], docs: Sequence[str]) -> bool:
        if cached_meta.get("doc_signature") != self._doc_signature(docs):
            return False
        if cached_meta.get("external_source_dir") != self.external_source_dir:
            return False
        if cached_meta.get("index_config") not in (None, self._index_config_payload()):
            return False

        source_match = cached_meta.get("source_fingerprint") == self._shared_v2_source_fingerprint()
        index_match = cached_meta.get("index_fingerprint") in {
            self._shared_v2_index_fingerprint(),
            self._index_fingerprint(),
        }
        return source_match or index_match

    def _shared_cache_path(self, kind: str) -> Optional[str]:
        if self.shared_index_dir is None:
            return None
        return os.path.join(self.shared_index_dir, _SHARED_V2_CACHE_FILENAMES[kind])

    def _read_meta(self, meta_path: str) -> Optional[Dict[str, object]]:
        if not os.path.isfile(meta_path):
            return None
        with open(meta_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_local_cached_hypergraph(self, docs: Sequence[str]):
        if self.global_config.force_index_from_scratch:
            return None
        if not (os.path.isfile(self._local_cache_meta_path) and os.path.isfile(self._local_cache_graph_path)):
            return None
        cached_meta = self._read_meta(self._local_cache_meta_path)
        if cached_meta is None or not self._local_cache_is_valid(cached_meta, docs):
            return None
        with open(self._local_cache_graph_path, "rb") as handle:
            logger.info("Loading cached HyperHippoRAG v3 hypergraph from %s", self._local_cache_graph_path)
            return pickle.load(handle)

    def _load_shared_cached_hypergraph(self, docs: Sequence[str]):
        if self.global_config.force_index_from_scratch or self.shared_index_dir is None:
            return None
        meta_path = self._shared_cache_path("meta")
        graph_path = self._shared_cache_path("graph")
        if meta_path is None or graph_path is None:
            return None
        if not (os.path.isfile(meta_path) and os.path.isfile(graph_path)):
            return None
        cached_meta = self._read_meta(meta_path)
        if cached_meta is None or not self._shared_cache_is_valid(cached_meta, docs):
            return None
        load_hyperhippo_v2_modules(self.external_source_dir)
        with open(graph_path, "rb") as handle:
            logger.info("Loading shared HyperHippoRAG v2 hypergraph for v3 from %s", graph_path)
            return pickle.load(handle)

    def _load_local_cached_propositions(self, docs: Sequence[str]):
        if self.global_config.force_openie_from_scratch:
            return None
        if not (os.path.isfile(self._local_cache_meta_path) and os.path.isfile(self._local_cache_props_path)):
            return None
        cached_meta = self._read_meta(self._local_cache_meta_path)
        if cached_meta is None or not self._local_cache_is_valid(cached_meta, docs):
            return None
        with open(self._local_cache_props_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        propositions = payload.get("propositions_per_passage")
        if isinstance(propositions, list) and len(propositions) == len(docs):
            logger.info("Loading cached HyperHippoRAG v3 propositions from %s", self._local_cache_props_path)
            return propositions
        return None

    def _load_shared_cached_propositions(self, docs: Sequence[str]):
        if self.global_config.force_openie_from_scratch or self.shared_index_dir is None:
            return None
        meta_path = self._shared_cache_path("meta")
        props_path = self._shared_cache_path("props")
        if meta_path is None or props_path is None:
            return None
        if not (os.path.isfile(meta_path) and os.path.isfile(props_path)):
            return None
        cached_meta = self._read_meta(meta_path)
        if cached_meta is None or not self._shared_cache_is_valid(cached_meta, docs):
            return None
        with open(props_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        propositions = payload.get("propositions_per_passage")
        if isinstance(propositions, list) and len(propositions) == len(docs):
            logger.info("Loading shared HyperHippoRAG v2 propositions for v3 from %s", props_path)
            return propositions
        return None

    def _write_local_cache_meta(self, docs: Sequence[str]):
        meta = self._current_local_cache_meta(docs)
        with open(self._local_cache_meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, ensure_ascii=False)
        return meta

    def _load_incremental_propositions_from_paths(
        self,
        docs: Sequence[str],
        meta_path: str,
        props_jsonl_path: str,
        use_shared_meta: bool,
    ) -> List[List[str]]:
        if self.global_config.force_openie_from_scratch:
            return []
        if not (os.path.isfile(meta_path) and os.path.isfile(props_jsonl_path)):
            return []

        cached_meta = self._read_meta(meta_path)
        if cached_meta is None:
            return []
        validator = self._shared_cache_is_valid if use_shared_meta else self._local_cache_is_valid
        if not validator(cached_meta, docs):
            return []

        propositions_per_passage: List[List[str]] = []
        with open(props_jsonl_path, "r", encoding="utf-8") as handle:
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
                        props_jsonl_path,
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
                "Loaded %d/%d incremental proposition records from %s",
                len(propositions_per_passage),
                len(docs),
                props_jsonl_path,
            )
        return propositions_per_passage

    def _load_incremental_propositions(self, docs: Sequence[str]) -> List[List[str]]:
        local = self._load_incremental_propositions_from_paths(
            docs=docs,
            meta_path=self._local_cache_meta_path,
            props_jsonl_path=self._local_cache_props_jsonl_path,
            use_shared_meta=False,
        )
        if local:
            return local

        if self.shared_index_dir is None:
            return []

        meta_path = self._shared_cache_path("meta")
        props_jsonl_path = self._shared_cache_path("props_jsonl")
        if meta_path is None or props_jsonl_path is None:
            return []
        return self._load_incremental_propositions_from_paths(
            docs=docs,
            meta_path=meta_path,
            props_jsonl_path=props_jsonl_path,
            use_shared_meta=True,
        )

    def _append_incremental_proposition(self, passage_idx: int, passage: str, propositions: List[str]):
        record = {
            "passage_idx": passage_idx,
            "passage_hash": _hash_text(passage),
            "propositions": propositions,
        }
        with open(self._local_cache_props_jsonl_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    def _save_local_cache(self, docs: Sequence[str], propositions_per_passage: List[List[str]], hypergraph):
        self._write_local_cache_meta(docs)
        with open(self._local_cache_props_path, "w", encoding="utf-8") as handle:
            json.dump({"propositions_per_passage": propositions_per_passage}, handle, indent=2, ensure_ascii=False)
        with open(self._local_cache_graph_path, "wb") as handle:
            pickle.dump(hypergraph, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _build_v3_retriever(self):
        retriever_module = self._external_modules["hyperhippo_v3"]
        retriever_cls = getattr(retriever_module, "HyperHippoRAGV3", None)
        if retriever_cls is None:
            retriever_cls = retriever_module.HyperHippoRAG
        return retriever_cls(
            hypergraph=self._v3_hypergraph,
            embedding_model=self._v3_embedding_adapter,
            llm_model=self._v3_llm_adapter,
            config=self._v3_config,
        )

    def index(self, docs: List[str]):
        logger.info("Indexing documents with HyperHippoRAG v3 bridge")
        self._indexed_docs = list(docs)

        self._v3_hypergraph = self._load_local_cached_hypergraph(docs)
        if self._v3_hypergraph is None:
            self._v3_hypergraph = self._load_shared_cached_hypergraph(docs)

        if self._v3_hypergraph is None:
            propositions_per_passage = self._load_local_cached_propositions(docs)
            if propositions_per_passage is None:
                propositions_per_passage = self._load_shared_cached_propositions(docs)
            if propositions_per_passage is None:
                propositions_per_passage = self._load_incremental_propositions(docs)
                extractor_cls = self._external_modules["proposition_extractor"].PropositionExtractor
                extractor = extractor_cls(
                    self._v3_llm_adapter,
                    self._v3_embedding_adapter,
                    config=self._v3_config,
                )
                start_idx = len(propositions_per_passage)
                if start_idx > 0:
                    logger.info(
                        "Resuming HyperHippoRAG v3 proposition extraction from passage %d/%d",
                        start_idx,
                        len(docs),
                    )
                self._write_local_cache_meta(docs)
                new_propositions = _extract_propositions_with_ordered_persistence(
                    docs=docs,
                    start_idx=start_idx,
                    extract_fn=extractor.extract_from_passage,
                    persist_fn=self._append_incremental_proposition,
                    desc="HyperHippo v3 Proposition Extraction",
                    num_workers=getattr(self._v3_config, "proposition_extraction_num_workers", 1),
                    prefetch=getattr(self._v3_config, "proposition_extraction_prefetch", None),
                )
                propositions_per_passage.extend(new_propositions)

            builder_cls = self._external_modules["hypergraph_builder"].HypergraphBuilder
            builder = builder_cls(self._v3_embedding_adapter, config=self._v3_config)
            self._v3_hypergraph = builder.build(propositions_per_passage, list(docs))
            self._save_local_cache(docs, propositions_per_passage, self._v3_hypergraph)

        self._v3_retriever = self._build_v3_retriever()
        self.ready_to_retrieve = True

    def retrieve(
        self,
        queries: List[str],
        num_to_retrieve: int = None,
        gold_docs: List[List[str]] = None,
    ):
        if not self.ready_to_retrieve:
            raise RuntimeError("HyperHippoRAGV3Bridge index() must be called before retrieve().")

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        retrieval_results = []
        for query in tqdm(queries, desc="Retrieving", total=len(queries)):
            docs = self._v3_retriever.retrieve(query, num_to_retrieve=num_to_retrieve)
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
            "shared_index_dir": self.shared_index_dir,
            "working_dir": self.working_dir,
            "global_config": asdict(self.global_config),
            "external_config": asdict(self._v3_config),
            "num_indexed_docs": len(self._indexed_docs),
            "source_fingerprint": self._source_fingerprint(),
            "index_fingerprint": self._index_fingerprint(),
            "cache_paths": {
                "meta": self._local_cache_meta_path,
                "propositions": self._local_cache_props_path,
                "propositions_jsonl": self._local_cache_props_jsonl_path,
                "graph": self._local_cache_graph_path,
            },
        }
