# API Reference

---

## config.py — Configuration Dataclasses

### `EpisodicKGConfig`
Configuration for the fast-learning episodic knowledge graph (hippocampus analogue).

**Fields:**
- `max_triples: int = 50_000` — Total triple capacity before consolidation pressure
- `max_nodes: int = 10_000` — Unique entity capacity
- `n_edge_types: int = 16` — Pre-defined edge relation types
- `n_node_types: int = 12` — Pre-defined node entity types
- `temporal_resolution_seconds: int = 3600` — 1-hour binning for temporal queries
- `spatial_grid_size_meters: int = 100` — 100m spatial proximity grid
- `max_temporal_query_horizon_days: int = 730` — 2-year lookback
- `write_admission: Literal["append", "surprise", "dedup"] = "append"` — Fast-write policy
- `dedup_window_seconds: int = 300` — Dedup within 5 min window
- `enable_crop_cycle_objects: bool = True` — CropCycle as first-class object
- `enable_disease_front_objects: bool = True` — DiseaseEvent with spatial spread
- `enable_treatment_log: bool = True` — TreatmentAction with temporal ordering
- `temporal_path_max_depth: int = 5` — Max hops in temporal path queries
- `spatial_proximity_radius_m: float = 500.0` — Default spatial query radius
- `kg_backend: Literal["in_memory", "neo4j", "duckdb"] = "in_memory"` — Persistence backend
- `checkpoint_interval_minutes: int = 60` — Checkpoint frequency
- `node_embed_dim: int = 64` — Per-node feature dimension
- `edge_embed_dim: int = 16` — Per-edge feature dimension

### `SemanticMLConfig`
Configuration for the slow-learning semantic layer (neocortex analogue).

**Fields:**
- `encoder_type: Literal["gcn", "gat", "graph_transformer"] = "gcn"` — GNN architecture
- `hidden_dim: int = 256` — Hidden dimension
- `n_layers: int = 3` — Number of GCN layers
- `n_heads: int = 4` — Number of attention heads
- `d_ff: int = 1024` — Feed-forward dimension
- `node_embed_dim: int = 64` — Per-node feature dimension
- `edge_embed_dim: int = 16` — Per-edge feature dimension
- `pattern_embed_dim: int = 128` — Compressed pattern representation dimension
- `n_pattern_slots: int = 64` — Number of learned pattern prototypes
- `consolidation_batch_size: int = 256` — Minimum subgraphs before consolidation
- `consolidation_lr: float = 1e-4` — Learning rate for consolidation
- `consolidation_frequency_minutes: int = 1440` — Daily offline consolidation
- `consolidation_warmup_hours: int = 48` — Wait before first consolidation
- `consolidation_n_epochs: int = 10` — Epochs per consolidation round
- `contrastive_margin: float = 1.0` — Margin for contrastive loss
- `inference_mode: Literal["embedding_similarity", "prototype_match", "gcn_classify"] = "embedding_similarity"`
- `confidence_threshold: float = 0.7` — Min confidence for semantic-only response
- `few_shot_k: int = 5` — K episodes for few-shot adaptation
- `few_shot_lr: float = 1e-3` — Learning rate for few-shot
- `dropout: float = 0.1` — Dropout rate
- `dtype: str = "float32"` — Default dtype

### `AgentControllerConfig`
Configuration for the LLM-based agent controller orchestrator.

**Fields:**
- `llm_model: str = "gpt-4o-mini"` — LLM backend model
- `llm_temperature: float = 0.1` — Low temperature for deterministic reasoning
- `llm_max_tokens: int = 2048` — Max output tokens
- `llm_context_window: int = 128_000` — LLM context window
- `max_iterative_cycles: int = 5` — Max reconciliation iterations
- `early_exit_confidence: float = 0.9` — Exit threshold for reconciliation
- `iteration_timeout_ms: int = 1000` — Per-iteration timeout
- `parallel_initial_query: bool = True` — Query both systems in parallel
- `enable_semantic_prior_routing: bool = True` — Use semantic priors to refine KG queries
- `enable_episodic_revision: bool = True` — Use episodic findings to revise semantic beliefs
- `reconciliation_method: Literal["llm_judge", "weighted_vote", "confidence_max"] = "llm_judge"`
- `provenance_tracking: bool = True` — Tag every fact with source layer
- `working_memory_max_tokens: int = 16_000` — Active session context limit
- `working_memory_eviction: Literal["lru", "token_count", "semantic_similarity"] = "lru"`

### `CLSMemorySystemConfig`
Top-level configuration aggregating all subsystem configs.

**Fields:**
- `episodic_kg: EpisodicKGConfig` — Subsystem 1 config
- `semantic_ml: SemanticMLConfig` — Subsystem 2 config
- `agent_controller: AgentControllerConfig` — Subsystem 3 config
- `debug: bool = False`
- `log_level: str = "INFO"`
- `seed: int = 42` — Random seed for reproducibility
- `version: str = "0.1.0"`

---

## data_model.py — Data Model

### Enums

- `CropStage` — `PLANTING`, `VEGETATIVE`, `FLOWERING`, `FRUITING`, `MATURATION`, `HARVEST`, `FALLOW`
- `DiseaseStatus` — `SUSPECTED`, `CONFIRMED`, `ACTIVE`, `CONTAINED`, `RESOLVED`, `RECURRENT`
- `TreatmentType` — `CHEMICAL_FUNGICIDE`, `CHEMICAL_PESTICIDE`, `BIOLOGICAL`, `CULTURAL`, `REMOVAL`, `PREVENTATIVE`, `IRRIGATION_ADJUSTMENT`, `NUTRITIONAL`
- `EdgeRelation` — `OCCURRED_DURING`, `TREATED_WITH`, `FOLLOWED_BY`, `SPATIALLY_NEAR`, `SAME_FIELD`, `PRECEDED_BY`, `SAME_CROP`, `CAUSED_BY`, `RELATED_TO`, `OBSERVED_IN`

### Domain objects

- `CropCycle` — First-class episodic object for a complete crop growing cycle
- `DiseaseEvent` — First-class episodic object for disease occurrence with spatial spread
- `TreatmentAction` — First-class episodic object for treatment applied to a disease event

### Graph data structures

- `KGNode` — Node in the episodic knowledge graph with `node_id`, `node_type`, `timestamp`, `features`, `attributes`, `spatial_x/y`
- `KGEdge` — Directed edge with `source`, `target`, `relation`, `timestamp`, `features`
- `KGSubgraph` — Standardized subgraph returned by KG queries. Contains `nodes`, `edges`, `node_features` ((N, D_node)), `edge_index` ((2, E)), `edge_features` ((E, D_edge)), `node_mask` ((N,)), `label`
- `SubgraphBatch` — Batched collection of KGSubgraphs. Tensors padded to `(B, max_N, D)` shapes.

### Query primitives

- `TemporalPathQuery` — Extracts ordered temporal sequences with `start_node_id`, `relation_sequence`, `max_hops`, `from_date`, `to_date`
- `SpatialProximityQuery` — Finds disease events within spatial proximity with `center_field_id`, `radius_m`, `from_date`, `to_date`, `disease_filter`

### Diagnostic types

- `DiagnosticContext` — Session context: `field_id`, `crop_type`, `season_start/end`
- `SemanticInferenceResult` — `pattern_embed` (pattern_embed_dim,), `matched_prototype_idx`, `confidence`, `provenance`
- `ReconciliationResult` — `consistency_score`, `gaps`, `contradictions`, `semantic_prior_needed`, `episodic_refinement_needed`, `confidence`
- `DiagnosticResponse` — `answer` (str), `provenance` (list[dict]), `num_iterations`, `confidence`, `evidence` (list[dict])

---

## base.py — Abstract Interfaces

### `BaseEpisodicMemory(ABC, nn.Module)`
Abstract base for fast-learning episodic memory.

**Methods:**
- `fast_write(event) -> str` — Ingest event immediately, return node ID
- `temporal_path_query(query: TemporalPathQuery) -> list[KGSubgraph]` — Extract ordered temporal sequences
- `spatial_proximity_query(query: SpatialProximityQuery) -> list[KGSubgraph]` — Find events in spatial proximity
- `extract_consolidation_batch(max_samples=256) -> list[KGSubgraph]` — Return matured subgraphs
- `prewarm_semantic_query(pattern_embedding: Tensor, k=5) -> list[KGSubgraph]` — Semantic → Episodic: K most similar subgraphs
- `to_subgraph_batch(subgraphs) -> SubgraphBatch` — Convert subgraphs to batched tensors

### `BaseSemanticMemory(ABC, nn.Module)`
Abstract base for slow-learning semantic memory.

**Methods:**
- `infer_pattern(query: str | KGSubgraph) -> SemanticInferenceResult` — Match query to semantic pattern
- `consolidate(episodic_subgraphs) -> None` — Offline contrastive consolidation
- `few_shot_adapt(support_subgraphs, query) -> dict` — Prototypical network few-shot adaptation
- `query_episodic_via_semantic_prior(semantic_result, episodic_memory, k=5) -> list[KGSubgraph]` — Semantic priors guide KG search

### `BaseAgentController(ABC)`
Abstract base for the agent controller orchestrator.

**Methods:**
- `diagnose(query: str, context: DiagnosticContext) -> DiagnosticResponse` — Main entry point for diagnostics
- `_reconcile(episodic_facts, semantic_pattern, iteration) -> ReconciliationResult` — Compare episodic vs. semantic
- `_generate_response(query, reconciliation_log, kg_results, semantic_results) -> dict` — Generate response with provenance

### `count_params(model: nn.Module) -> None`
Print total and trainable parameter counts for a PyTorch model.

---

## kg.py — Episodic Knowledge Graph

### `TemporalIndex`
Timestamp-sorted index for efficient time-range queries. O(log N + K) range queries via binary search on quantized time bins.

**Methods:**
- `insert(node_id, timestamp)` — Insert at quantized bin. O(1) amortized.
- `query_range(from_date, to_date) -> list[str]` — All node IDs in range. O(K).
- `remove(node_id)` — Remove from index.
- `__len__()` — Total entries.

### `SpatialIndex`
Geohash-based spatial proximity index. Grid cell approach for efficient radius queries.

**Methods:**
- `insert(node_id, x, y)` — Insert at coordinates. O(1).
- `radius_query(center_x, center_y, radius_m) -> list[str]` — All nodes within radius. O(K).
- `lookup(node_id) -> tuple[float, float]` — Get coordinates. O(1).
- `remove(node_id)` — Remove from index.

### `DedupCache`
Time-windowed deduplication cache for fast-write admission. Prevents duplicate ingestion within configurable window.

**Methods:**
- `contains(dedup_key) -> bool` — Check if key exists and is within window.
- `get(dedup_key) -> Optional[str]` — Get node ID or None if expired.
- `put(dedup_key, node_id)` — Insert entry.
- `evict_expired() -> int` — Remove expired entries; return count.

### `EpisodicKnowledgeGraph(BaseEpisodicMemory)`
Fast-learning episodic memory backed by a spatio-temporal knowledge graph.

**Constructor:** `EpisodicKnowledgeGraph(config: EpisodicKGConfig)`

**Methods:**
- `fast_write(event) -> str` — Ingest domain event. O(log N) amortized.
- `temporal_path_query(query: TemporalPathQuery) -> list[KGSubgraph]` — BFS over edges filtered by relation sequence, sorted by timestamp.
- `spatial_proximity_query(query: SpatialProximityQuery) -> list[KGSubgraph]` — Spatial index radius query with temporal + disease filters.
- `extract_consolidation_batch(max_samples=256) -> list[KGSubgraph]` — Matured subgraphs (age > checkpoint_interval, repeat_count >= 2).
- `prewarm_semantic_query(pattern_embedding: Tensor, k=5) -> list[KGSubgraph]` — Top-K cosine similarity search against cached subgraph embeddings.
- `to_subgraph_batch(subgraphs) -> SubgraphBatch` — Delegates to SubgraphBatch constructor.
- `update_subgraph_embeddings(embeddings)` — Update cached subgraph embeddings after consolidation.
- `clear_pending_consolidation()` — Clear consolidation buffer.
- `get_statistics() -> dict` — Nodes, edges, writes, queries, pending consolidation.

---

## layers.py — Neural Network Layers

### `GCNLayer(nn.Module)`
Edge-feature-aware Graph Convolutional Layer with learned heterophily gating.

**Constructor:** `GCNLayer(in_dim, out_dim, edge_dim=16, dropout=0.1, use_layer_norm=True, use_residual=True)`

**Forward:** `(x: (N, in_dim), edge_index: (2, E), edge_attr: (E, edge_dim)) -> (N, out_dim)`

### `EdgeAwareGCN(nn.Module)`
Stacked GCN with residual connections and layer normalization.

**Constructor:** `EdgeAwareGCN(in_dim, hidden_dim, out_dim, n_layers=3, edge_dim=16, dropout=0.1)`

**Forward:** `(x: (N, in_dim), edge_index: (2, E), edge_attr: (E, edge_dim)) -> (N, out_dim)`

### `PrototypeAttention(nn.Module)`
Multihead cross-attention over learned prototype pattern slots.

**Constructor:** `PrototypeAttention(pattern_dim, n_prototypes, n_heads=4, dropout=0.1)`

**Forward:** `(query: (B, 1, P)) -> ((B, P), (B, 1, S))` — attended pattern + attention weights

### `MLP(nn.Module)`
Configurable multi-layer perceptron with GELU activations.

**Constructor:** `MLP(in_dim, hidden_dim, out_dim, n_layers=2, dropout=0.1)`

**Forward:** `(x: (..., in_dim)) -> (..., out_dim)`

### `SubgraphPooling(nn.Module)`
Pool subgraph node embeddings to a single graph-level vector. Supports mean and max pooling with padding masks.

**Constructor:** `SubgraphPooling(pooling="mean")`

**Forward:** `(x: (B, N, D), mask: (B, N)) -> (B, D)`

### `ContrastiveLoss(nn.Module)`
NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss for contrastive consolidation. Includes margin-based hinge loss for additional separation.

**Constructor:** `ContrastiveLoss(temperature=0.1, margin=1.0)`

**Forward:** `(embeddings: (B, D), labels: (B,) or None) -> scalar loss`

### `contrastive_loss(embeddings, labels, margin, temperature)`
Functional alias for ContrastiveLoss.

---

## semantic.py — Semantic ML Layer

### `SemanticPatternExtractor(nn.Module)`
Slow-learning semantic layer (neocortex analogue). Architecture: GCN encoder → prototype attention → confidence head.

**Constructor:** `SemanticPatternExtractor(config: SemanticMLConfig)`

**Methods:**
- `forward(batch: SubgraphBatch, use_checkpoint=False) -> dict` — Encode batched subgraphs into `{pattern_embed: (B, P), prototype_weights: (B, S), confidence: (B,)}`. Gradient checkpointing supported.
- `consolidate(episodic_subgraphs, n_epochs=10, lr=1e-4) -> dict` — Offline contrastive consolidation training. Returns `{epochs, final_loss, losses}`.
- `few_shot_adapt(support_subgraphs, query) -> dict` — Prototypical network few-shot adaptation. Returns `{pattern_embed, matched_prototype, similarity, confidence}`.
- `get_prototype(idx) -> Tensor` — Return prototype vector at index. (P,)
- `get_all_prototypes() -> Tensor` — Return all prototypes. (S, P)
- `compute_prototype_similarity(embedding) -> Tensor` — Cosine similarity to all prototypes. (S,)

### `SemanticMemoryManager(BaseSemanticMemory)`
Manages semantic ML layer lifecycle: training, inference, consolidation schedule, and cross-layer query paths.

**Constructor:** `SemanticMemoryManager(config: SemanticMLConfig)`

**Methods:**
- `infer_pattern(query: str | KGSubgraph) -> SemanticInferenceResult` — Text or subgraph → pattern embedding + confidence + provenance
- `consolidate(episodic_subgraphs) -> dict` — Train extractor on matured subgraphs; rebuild pattern cache
- `should_consolidate() -> bool` — Check warmup + schedule
- `query_episodic_via_semantic_prior(semantic_result, episodic_memory, k=5) -> list[KGSubgraph]` — Semantic priors → KG refinement
- `few_shot_adapt(support_subgraphs, query) -> dict` — Delegate to extractor

---

## controller.py — Agent Controller

### `WorkingMemory`
Short-term buffer for active diagnostic session. Eviction prevents context overflow.

**Constructor:** `WorkingMemory(max_tokens=16_000, eviction_policy="lru")`

**Methods:**
- `add(key, value)` — Add item, evicting if over capacity
- `get(key) -> Any` — Retrieve item, updating LRU access log
- `remove(key)` — Delete item
- `clear()` — Reset for new session
- `get_all() -> dict` — All items (for LLM context)

### `ConsolidationScheduler`
Manages offline consolidation lifecycle: scheduled, threshold-based, and on-demand triggers.

**Constructor:** `ConsolidationScheduler(config, episodic_kg, semantic_memory)`

**Methods:**
- `should_consolidate() -> bool` — Check warmup + schedule
- `step() -> dict` — Run consolidation if scheduled; return stats
- `force_consolidation() -> dict` — Immediate consolidation

### `LLMInterface`
Lightweight interface for LLM calls. Stub returns deterministic JSON responses for reconciliation prompts and template responses for other prompts.

**Constructor:** `LLMInterface(model="gpt-4o-mini", temperature=0.1, max_tokens=2048)`

**Methods:**
- `complete(prompt, temperature=None) -> str` — Send completion request. In stub, returns deterministic responses.

### `CLSAgentController(BaseAgentController)`
LLM-based orchestration layer managing iterative bidirectional querying.

**Constructor:** `CLSAgentController(config: AgentControllerConfig, episodic_kg, semantic_memory)`

**Methods:**
- `diagnose(query: str, context: DiagnosticContext = None) -> DiagnosticResponse` — Main entry point. Implements the iterative bidirectional querying protocol.
- `_reconcile(episodic_facts, semantic_pattern, iteration) -> ReconciliationResult` — Route to llm_judge / weighted_vote / confidence_max
- `_generate_response(query, reconciliation_log, kg_results, semantic_results) -> dict` — Response with provenance
- `get_statistics() -> dict` — Diagnosis count, total iterations, avg iterations/diagnosis

---

## model.py — Top-Level System

### `CLSMemorySystem(nn.Module)`
Top-level CLS bicameral memory architecture tying together all three subsystems.

**Constructor:** `CLSMemorySystem(config: CLSMemorySystemConfig)`

**Methods:**
- `fast_write(event) -> str` — Fast-write event into episodic KG
- `diagnose(query: str, context: DiagnosticContext = None) -> DiagnosticResponse` — Full diagnostic query through iterative bidirectional protocol
- `diagnose_batch(queries: list[tuple]) -> list[DiagnosticResponse]` — Sequential batch diagnose
- `consolidate(force=False) -> dict` — Run consolidation cycle
- `forward(subgraph_batch: SubgraphBatch, use_checkpoint=False) -> dict` — Forward through semantic ML only
- `set_ablation_mode(mode: str)` — Switch between "full", "kg_only", "no_iterate", "generic_kg"
- `get_system_state() -> dict` — Full system state summary
- `reset()` — Reset all subsystems to initial state
