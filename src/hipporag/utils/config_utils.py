import os
from dataclasses import dataclass, field
from typing import (
    Literal,
    Union,
    Optional
)

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BaseConfig:
    """One and only configuration."""
    # LLM specific attributes 
    llm_name: str = field(
        default="gpt-4o-mini",
        metadata={"help": "Class name indicating which LLM model to use."}
    )
    llm_base_url: str = field(
        default=None,
        metadata={"help": "Base URL for the LLM model, if none, means using OPENAI service."}
    )
    embedding_base_url: str = field(
        default=None,
        metadata={"help": "Base URL for an OpenAI compatible embedding model, if none, means using OPENAI service."}
    )
    azure_endpoint: str = field(
        default=None,
        metadata={"help": "Azure Endpoint URI for the LLM model, if none, uses OPENAI service directly."}
    )
    azure_embedding_endpoint: str = field(
        default=None,
        metadata={"help": "Azure Endpoint URI for the OpenAI embedding model, if none, uses OPENAI service directly."}
    )
    max_new_tokens: Union[None, int] = field(
        default=2048,
        metadata={"help": "Max new tokens to generate in each inference."}
    )
    num_gen_choices: int = field(
        default=1,
        metadata={"help": "How many chat completion choices to generate for each input message."}
    )
    seed: Union[None, int] = field(
        default=None,
        metadata={"help": "Random seed."}
    )
    temperature: float = field(
        default=0,
        metadata={"help": "Temperature for sampling in each inference."}
    )
    qwen_disable_thinking: bool = field(
        default=False,
        metadata={"help": "Whether to disable Qwen chat-template thinking mode for OpenAI-compatible servers."}
    )
    response_format: Union[dict, None] = field(
        default_factory=lambda: { "type": "json_object" },
        metadata={"help": "Specifying the format that the model must output."}
    )
    
    ## LLM specific attributes -> Async hyperparameters
    max_retry_attempts: int = field(
        default=5,
        metadata={"help": "Max number of retry attempts for an asynchronous API calling."}
    )
    # Storage specific attributes
    force_openie_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing openie files and rebuild them from scratch."}
    )

    # Storage specific attributes 
    force_index_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing storage files and graph data and will rebuild from scratch."}
    )
    rerank_dspy_file_path: str = field(
        default=None,
        metadata={"help": "Path to the rerank dspy file."}
    )
    passage_node_weight: float = field(
        default=0.05,
        metadata={"help": "Multiplicative factor that modified the passage node weights in PPR."}
    )
    save_openie: bool = field(
        default=True,
        metadata={"help": "If set to True, will save the OpenIE model to disk."}
    )
    
    # Preprocessing specific attributes
    text_preprocessor_class_name: str = field(
        default="TextPreprocessor",
        metadata={"help": "Name of the text-based preprocessor to use in preprocessing."}
    )
    preprocess_encoder_name: str = field(
        default="gpt-4o",
        metadata={"help": "Name of the encoder to use in preprocessing (currently implemented specifically for doc chunking)."}
    )
    preprocess_chunk_overlap_token_size: int = field(
        default=128,
        metadata={"help": "Number of overlap tokens between neighbouring chunks."}
    )
    preprocess_chunk_max_token_size: int = field(
        default=None,
        metadata={"help": "Max number of tokens each chunk can contain. If set to None, the whole doc will treated as a single chunk."}
    )
    preprocess_chunk_func: Literal["by_token", "by_word"] = field(default='by_token')
    
    
    # Information extraction specific attributes
    information_extraction_model_name: Literal["openie_openai_gpt", ] = field(
        default="openie_openai_gpt",
        metadata={"help": "Class name indicating which information extraction model to use."}
    )
    openie_mode: Literal["offline", "online"] = field(
        default="online",
        metadata={"help": "Mode of the OpenIE model to use."}
    )
    skip_graph: bool = field(
        default=False,
        metadata={"help": "Whether to skip graph construction or not. Set it to be true when running vllm offline indexing for the first time."}
    )
    
    
    # Embedding specific attributes
    embedding_model_name: str = field(
        default="nvidia/NV-Embed-v2",
        metadata={"help": "Class name indicating which embedding model to use."}
    )
    embedding_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size of calling embedding model."}
    )
    embedding_return_as_normalized: bool = field(
        default=True,
        metadata={"help": "Whether to normalize encoded embeddings not."}
    )
    embedding_max_seq_len: int = field(
        default=2048,
        metadata={"help": "Max sequence length for the embedding model."}
    )
    embedding_model_dtype: Literal["float16", "float32", "bfloat16", "auto"] = field(
        default="auto",
        metadata={"help": "Data type for local embedding model."}
    )
    
    
    
    # Graph construction specific attributes
    synonymy_edge_topk: int = field(
        default=2047,
        metadata={"help": "k for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_query_batch_size: int = field(
        default=1000,
        metadata={"help": "Batch size for query embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_key_batch_size: int = field(
        default=10000,
        metadata={"help": "Batch size for key embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_sim_threshold: float = field(
        default=0.8,
        metadata={"help": "Similarity threshold to include candidate synonymy nodes."}
    )
    graph_use_canonical_triples: bool = field(
        default=False,
        metadata={"help": "Whether to canonicalize OpenIE triples before graph/fact/entity construction."}
    )
    graph_drop_self_loop_triples: bool = field(
        default=True,
        metadata={"help": "Whether canonical graph build should drop degenerate subject==object triples."}
    )
    is_directed_graph: bool = field(
        default=False,
        metadata={"help": "Whether the graph is directed or not."}
    )
    
    
    
    # Retrieval specific attributes
    linking_top_k: int = field(
        default=5,
        metadata={"help": "The number of linked nodes at each retrieval step"}
    )
    retrieval_top_k: int = field(
        default=200,
        metadata={"help": "Retrieving k documents at each step"}
    )
    damping: float = field(
        default=0.5,
        metadata={"help": "Damping factor for ppr algorithm."}
    )
    retrieval_mode: Literal[
        "hipporag_v2",
        "hyperhippo_strong_v1",
        "hyperhippo_full",
        "hyperhippo_v2_bridge",
        "hyperhippo_v3_bridge",
    ] = field(
        default="hipporag_v2",
        metadata={"help": "Retrieval backbone to run."}
    )
    proposition_merge_threshold: float = field(
        default=0.95,
        metadata={"help": "Merge threshold for canonicalized proposition upserts."}
    )
    entity_seed_top_k: int = field(
        default=8,
        metadata={"help": "Top-k entity nodes used to form the reset vector."}
    )
    hyperedge_seed_top_k: int = field(
        default=12,
        metadata={"help": "Top-k hyperedge nodes used to form the reset vector."}
    )
    passage_seed_top_k: int = field(
        default=12,
        metadata={"help": "Top-k passage nodes used to form the reset vector."}
    )
    diffusion_min_edge_weight: float = field(
        default=1e-4,
        metadata={"help": "Minimum query-conditioned edge weight for diffusion stability."}
    )
    ee_top_k: int = field(
        default=16,
        metadata={"help": "Top-k nearest neighbors considered for entity-entity alias edges."}
    )
    ee_query_batch_size: int = field(
        default=64,
        metadata={"help": "Batch size for entity-entity neighbor search queries."}
    )
    ee_key_batch_size: int = field(
        default=256,
        metadata={"help": "Batch size for entity-entity neighbor search keys."}
    )
    ee_similarity_threshold: float = field(
        default=0.88,
        metadata={"help": "Similarity threshold for entity-entity alias edges."}
    )
    hh_shared_anchor_weight: float = field(
        default=0.35,
        metadata={"help": "Weight of shared-anchor evidence in H-H edge construction."}
    )
    hh_role_compat_weight: float = field(
        default=0.20,
        metadata={"help": "Weight of role compatibility in H-H edge construction."}
    )
    hh_summary_sim_weight: float = field(
        default=0.25,
        metadata={"help": "Weight of summary embedding similarity in H-H edge construction."}
    )
    hh_context_consistency_weight: float = field(
        default=0.20,
        metadata={"help": "Weight of context consistency in H-H edge construction."}
    )
    hh_min_weight_threshold: float = field(
        default=0.40,
        metadata={"help": "Minimum edge weight to keep a hyperedge-hyperedge bridge."}
    )
    hh_top_k_per_node: int = field(
        default=8,
        metadata={"help": "Maximum outgoing H-H neighbors retained per hyperedge node."}
    )
    hyperedge_embedding_mode: Literal["fact_reuse", "prop_only"] = field(
        default="fact_reuse",
        metadata={"help": "How HyperHippo obtains proposition-node embeddings."}
    )
    hyperedge_source_mode: Literal["triple", "proposition"] = field(
        default="triple",
        metadata={"help": "What raw unit HyperHippo turns into hyperedges. triple uses canonical subject-relation-object units; proposition uses natural-language propositions extracted from passages."}
    )
    hyperedge_embedding_text_mode: Literal["summary", "contextual"] = field(
        default="summary",
        metadata={"help": "Text used to embed prop_only hyperedges. summary uses only the proposition summary; contextual augments it with source evidence snippets."}
    )
    hyperedge_embedding_evidence_top_k: int = field(
        default=2,
        metadata={"help": "Maximum number of supporting passage snippets injected into contextual hyperedge embeddings."}
    )
    hyperedge_embedding_evidence_max_chars: int = field(
        default=240,
        metadata={"help": "Maximum characters kept from each supporting passage snippet in contextual hyperedge embeddings."}
    )
    hyperedge_key_view_mode: Literal["single", "multi"] = field(
        default="single",
        metadata={"help": "How many key-side text views each prop_only hyperedge exposes for retrieval. multi enables canonical, structured, object-centric, and contextual variants aggregated into one node score."}
    )
    hyperedge_query_version: Literal["v1", "v2", "v3", "v4"] = field(
        default="v1",
        metadata={"help": "How HyperHippo formulates query-to-hyperedge retrieval. v1 uses the raw question; v2 uses proposition-style rewrites; v3 adds role-aware weighting; v4 uses role-aware rewrites plus role-conditioned diffusion for prop_only."}
    )
    hyperedge_query_max_views: int = field(
        default=4,
        metadata={"help": "Maximum number of hyperedge query views kept per question in query-to-hyperedge v2."}
    )
    hyperedge_query_include_original: bool = field(
        default=True,
        metadata={"help": "Whether HyperHippo query-to-hyperedge v2 keeps the original question as one hyperedge query view."}
    )
    hyperedge_query_score_agg: Literal["max", "mean"] = field(
        default="max",
        metadata={"help": "How to aggregate multi-view hyperedge query scores in query-to-hyperedge v2."}
    )
    hyperedge_query_v3_max_score_weight: float = field(
        default=0.35,
        metadata={"help": "Blend weight on the max view score in hyperedge query v3. The remaining weight is assigned to the role/confidence weighted average."}
    )
    hyperedge_query_v3_confidence_floor: float = field(
        default=0.35,
        metadata={"help": "Minimum confidence used when converting LLM-produced hyperedge query views into v3 aggregation weights."}
    )
    hyperedge_query_v4_role_top_k: int = field(
        default=12,
        metadata={"help": "Top-k role-specific hyperedges retained when building v4 role-conditioned diffusion signals."}
    )
    hyperedge_query_v4_bridge_hh_boost: float = field(
        default=1.10,
        metadata={"help": "Additive boost on hyperedge-hyperedge edges supported by bridge-role hyperedges in v4."}
    )
    hyperedge_query_v4_target_bridge_hh_boost: float = field(
        default=0.70,
        metadata={"help": "Additive boost on target-to-bridge hyperedge transitions in v4."}
    )
    hyperedge_query_v4_target_edge_boost: float = field(
        default=0.55,
        metadata={"help": "Additive boost on target-bearing hyperedge edges such as H-P and E-H in v4."}
    )
    hyperedge_query_v4_bridge_seed_mix: float = field(
        default=0.18,
        metadata={"help": "Mixture weight assigned to bridge-role hyperedge seeds before reranked-fact fusion in v4."}
    )
    hyperedge_query_v4_constraint_edge_boost: float = field(
        default=0.65,
        metadata={"help": "Additive boost on constraint-supported edges in v4."}
    )
    hyperedge_query_v4_passage_score_boost: float = field(
        default=0.65,
        metadata={"help": "Multiplier strength used to inject projected target/constraint support into dense passage scores in v4."}
    )
    hyperedge_query_v4_entity_score_boost: float = field(
        default=0.30,
        metadata={"help": "Multiplier strength used to inject projected target/constraint support into entity scores in v4."}
    )
    hyperedge_query_v4_max_edge_boost: float = field(
        default=3.0,
        metadata={"help": "Maximum multiplicative edge boost allowed in v4 role-conditioned diffusion."}
    )
    hyperedge_rerank_top_k: int = field(
        default=0,
        metadata={"help": "If > 1, rerank the top-k prop_only hyperedge candidates with the LLM before diffusion."}
    )
    hyperedge_rerank_max_chars: int = field(
        default=360,
        metadata={"help": "Maximum characters kept from each candidate hyperedge when building the hyperedge rerank prompt."}
    )
    hyperedge_rerank_score_mix: float = field(
        default=0.60,
        metadata={"help": "Interpolation weight used to blend LLM-reranked hyperedge order back into dense hyperedge scores."}
    )
    hyperedge_rerank_seed_mix: float = field(
        default=0.55,
        metadata={"help": "Interpolation weight used to fuse LLM-reranked hyperedge seeds into the HyperHippo reset component."}
    )
    hyperedge_passage_rerank_top_k: int = field(
        default=0,
        metadata={"help": "If > 1, rerank the top-k retrieved passages with the LLM before returning HyperHippo results."}
    )
    hyperedge_passage_rerank_max_chars: int = field(
        default=480,
        metadata={"help": "Maximum characters kept from each candidate passage when building the HyperHippo passage rerank prompt."}
    )
    use_reranked_fact_hyperedge_seed: bool = field(
        default=False,
        metadata={"help": "Whether to use HippoRAG fact reranking outputs as strong hyperedge seeds in HyperHippo retrieval."}
    )
    reranked_fact_hyperedge_seed_mix: float = field(
        default=0.65,
        metadata={"help": "Interpolation weight used to fuse reranked-fact hyperedge seeds into the HyperHippo hyperedge reset component."}
    )
    freshness_tau: float = field(
        default=5.0,
        metadata={"help": "Decay factor for freshness-aware memory weighting."}
    )
    freshness_as_node_prior: bool = field(
        default=True,
        metadata={"help": "Whether to multiply query-node matches by freshness."}
    )
    
    
    # QA specific attributes
    max_qa_steps: int = field(
        default=1,
        metadata={"help": "For answering a single question, the max steps that we use to interleave retrieval and reasoning."}
    )
    qa_top_k: int = field(
        default=5,
        metadata={"help": "Feeding top k documents to the QA model for reading."}
    )
    
    # Save dir (highest level directory)
    save_dir: str = field(
        default=None,
        metadata={"help": "Directory to save all related information. If it's given, will overwrite all default save_dir setups. If it's not given, then if we're not running specific datasets, default to `outputs`, otherwise, default to a dataset-customized output dir."}
    )
    
    
    
    # Dataset running specific attributes
    ## Dataset running specific attributes -> General
    dataset: Optional[Literal['hotpotqa', 'hotpotqa_train', 'musique', '2wikimultihopqa', 'sample']] = field(
        default=None,
        metadata={"help": "Dataset to use. If specified, it means we will run specific datasets. If not specified, it means we're running freely."}
    )
    ## Dataset running specific attributes -> Graph
    graph_type: Literal[
        'dpr_only', 
        'entity', 
        'passage_entity', 'relation_aware_passage_entity',
        'passage_entity_relation', 
        'facts_and_sim_passage_node_unidirectional',
    ] = field(
        default="facts_and_sim_passage_node_unidirectional",
        metadata={"help": "Type of graph to use in the experiment."}
    )
    corpus_len: Optional[int] = field(
        default=None,
        metadata={"help": "Length of the corpus to use."}
    )
    
    
    def __post_init__(self):
        if self.save_dir is None: # If save_dir not given
            if self.dataset is None: self.save_dir = 'outputs' # running freely
            else: self.save_dir = os.path.join('outputs', self.dataset) # customize your dataset's output dir here
        logger.debug(f"Initializing the highest level of save_dir to be {self.save_dir}")
