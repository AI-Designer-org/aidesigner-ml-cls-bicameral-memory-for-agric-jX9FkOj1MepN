# Architecture Blueprint: Iterative Bidirectional CLS Memory for Agricultural Diagnostic Agents

> **Generated:** 2026-05-26
> **Architect stage output** — consumes research lifecycle contract from `ml-research`
> **Next stage:** `ml-coder` (implementation), `ml-validator` (verification)
> **Domain:** LM (Memory Systems / Agent Orchestration), Graph ML, Scientific ML (Agriculture)

---

## Table of Contents

1. [Domain Identification](#1-domain-identification)
2. [Design Constraints Summary](#2-design-constraints-summary)
3. [ModelConfig Dataclasses](#3-modelconfig-dataclasses)
4. [Architecture Overview](#4-architecture-overview)
5. [Core Component Pseudocode](#5-core-component-pseudocode)
6. [ASCII Architecture Diagram](#6-ascii-architecture-diagram)
7. [Inductive Bias Justifications](#7-inductive-bias-justifications)
8. [Research-to-Architecture Traceability](#8-research-to-architecture-traceability)
9. [Domain-Specific Considerations](#9-domain-specific-considerations)
10. [Implementation Risk Flags](#10-implementation-risk-flags)
11. [Baseline & Evaluation Requirements](#11-baseline--evaluation-requirements)
12. [Suggested Ablations](#12-suggested-ablations)

---

## 1. Domain Identification

| Domain | Role in Design | Primary Concern |
|---|---|---|
| **LM — Memory Systems** | Primary | CLS-inspired bicameral memory, iterative query protocol, agent controller orchestration, fast/slow learning separation |
| **Graph ML** | Primary | Episodic spatio-temporal knowledge graph, graph query operations, temporal path queries, subgraph extraction |
| **Scientific ML (Agriculture)** | Secondary | Domain-tailored node types (CropCycle, DiseaseEvent), spatial-temporal constructs for disease fronts, crop cycles |
| **GenAI / LLM** | Secondary | Agent controller as LLM orchestrator, natural language query parsing, response generation with provenance |

**Design stance:** This is a multi-system architecture — not a single end-to-end model. The three subsystems (Episodic KG, Semantic ML, Agent Controller) are independently designed but must share well-defined interfaces and a common data model.

---

## 2. Design Constraints Summary

Extracted from the upstream research lifecycle contract:

| Constraint | Source | Implication |
|---|---|---|
| Stability Gap proves monolithic collapse at N=5, ρ>0.6 | arXiv:2601.15313 (grounded) | Architecture MUST separate episodic from semantic storage; no single-store design is viable |
| Iterative bidirectional querying is the primary novelty claim | Research contract claim #1 (hypothesis) | Agent controller MUST support multi-turn episodic↔semantic querying, not just one-directional consolidation |
| Agricultural domain has high semantic density | Research contract (grounded) | Domain-specific episodic constructs MUST be designed for ρ>0.6 regime from day one |
| Zep/Graphiti is NOT monolithic (it's a temporal KG) | Research contract correction | Baseline comparison MUST be fair — compare against Zep's temporal KG as an alternative episodic design, not a "monolithic" strawman |
| Multiple ag neuro-symbolic systems exist (OpenAg, NeuroCausal-FusionNet) | Research contract correction | Must cite and differentiate from existing ag neuro-symbolic work; cannot claim "first to apply CLS to agriculture" |
| AOI and All-Mem consolidate one-directionally | Research contract | The iterative bidirectional protocol is the key differentiator — must be falsifiable |
| Latency concern: iterative cycles may negate accuracy gains | Blocking unknown | Must include latency budget and iteration count limit in config |

### Target Scale Assumptions

| Parameter | Value | Rationale |
|---|---|---|
| Episodic KG capacity | 10K–100K triples | Covers 50+ turn conversations × multiple crop cycles per agricultural season |
| Semantic ML hidden dim | 256 | Lightweight enough for inference on CPU; should not require GPU for pattern extraction |
| Agent controller LLM | GPT-4o-mini or equivalent | Production-viable cost profile; 128K context window sufficient for working memory |
| Max iterative cycles | 5 | Bounded iterations prevent latency explosion; derived from early-exit confidence threshold |
| Max query latency (p99) | < 5 s | Real-time diagnostic setting; iterative cycles must fit within this budget |
| Storage footprint | < 500 MB at 50K facts | Must run on edge or cloud-lean deployment |

---

## 3. ModelConfig Dataclasses

### 3.1 Top-Level System Config

```python
from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class EpisodicKGConfig:
    """Configuration for the fast-learning episodic knowledge graph (hippocampus analogue)."""
    # Graph dimensions
    max_triples: int = 50_000          # Total triple capacity before consolidation pressure
    max_nodes: int = 10_000            # Unique entity capacity
    n_edge_types: int = 16             # Pre-defined edge relation types
    n_node_types: int = 12             # Pre-defined node entity types

    # Spatio-temporal indexing
    temporal_resolution_seconds: int = 3600  # 1-hour binning for temporal queries
    spatial_grid_size_meters: int = 100      # 100m spatial proximity grid
    max_temporal_query_horizon_days: int = 730  # 2-year lookback

    # Fast-write admission policy
    write_admission: str = "append"    # "append" | "surprise" | "dedup"
    dedup_window_seconds: int = 300    # dedup within 5 min window

    # Domain-specific node type flags
    enable_crop_cycle_objects: bool = True     # CropCycle as first-class object
    enable_disease_front_objects: bool = True  # DiseaseEvent with spatial spread
    enable_treatment_log: bool = True          # TreatmentAction with temporal ordering

    # Query routing
    temporal_path_max_depth: int = 5   # Max hops in temporal path queries
    spatial_proximity_radius_m: float = 500.0  # Default spatial query radius

    # Persistence
    kg_backend: str = "in_memory"      # "in_memory" | "neo4j" | "duckdb"
    checkpoint_interval_minutes: int = 60


@dataclass
class SemanticMLConfig:
    """Configuration for the slow-learning semantic layer (neocortex analogue)."""
    # Architecture
    encoder_type: str = "gcn"          # "gcn" | "gat" | "graph_transformer"
    hidden_dim: int = 256
    n_layers: int = 3
    n_heads: int = 4                   # For GAT or Graph Transformer variants
    d_ff: int = 1024                   # Feed-forward dimension

    # Input / output
    node_embed_dim: int = 64           # Per-node feature dimension
    edge_embed_dim: int = 16           # Per-edge feature dimension
    pattern_embed_dim: int = 128       # Compressed pattern representation
    n_pattern_slots: int = 64          # Number of learned pattern prototypes

    # Consolidation
    consolidation_batch_size: int = 256
    consolidation_lr: float = 1e-4
    consolidation_frequency_minutes: int = 1440  # Daily offline consolidation
    consolidation_warmup_hours: int = 48         # Wait 48h before first consolidation

    # Inference
    inference_mode: str = "embedding_similarity"  # "embedding_similarity" | "prototype_match" | "gcn_classify"
    confidence_threshold: float = 0.7              # Min confidence for semantic response

    # Few-shot adaptation
    few_shot_k: int = 5                # K episodes for few-shot pattern adaptation
    few_shot_lr: float = 1e-3

    # Training
    dropout: float = 0.1
    dtype: str = "float32"


@dataclass
class AgentControllerConfig:
    """Configuration for the LLM-based agent controller orchestrator."""
    # LLM backend
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1       # Low temperature for deterministic diagnostic reasoning
    llm_max_tokens: int = 2048
    llm_context_window: int = 128_000

    # Iterative querying protocol
    max_iterative_cycles: int = 5
    early_exit_confidence: float = 0.9  # Exit if reconciliation confidence exceeds this
    iteration_timeout_ms: int = 1000    # Per-iteration timeout

    # Query routing
    parallel_initial_query: bool = True  # Query both systems in parallel on first pass
    enable_semantic_prior_routing: bool = True  # Use semantic priors to refine KG queries
    enable_episodic_revision: bool = True       # Use episodic findings to revise semantic beliefs

    # Reconciliation
    reconciliation_method: str = "llm_judge"  # "llm_judge" | "weighted_vote" | "confidence_max"
    provenance_tracking: bool = True          # Tag every fact with source layer

    # Working memory
    working_memory_max_tokens: int = 16_000  # Active session context limit
    working_memory_eviction: str = "lru"     # "lru" | "token_count" | "semantic_similarity"


@dataclass
class CLSMemorySystemConfig:
    """Top-level configuration for the CLS bicameral memory architecture."""
    # Subsystem configs
    episodic_kg: EpisodicKGConfig = field(default_factory=EpisodicKGConfig)
    semantic_ml: SemanticMLConfig = field(default_factory=SemanticMLConfig)
    agent_controller: AgentControllerConfig = field(default_factory=AgentControllerConfig)

    # System-wide settings
    debug: bool = False
    log_level: str = "INFO"
    seed: int = 42
    version: str = "0.1.0"
```

### 3.2 Data Model — First-Class Spatio-Temporal Objects

```python
from dataclasses import dataclass
from typing import Optional
from enum import Enum


# ── Enums for domain-specific types ──

class CropStage(Enum):
    PLANTING = "planting"
    VEGETATIVE = "vegetative"
    FLOWERING = "flowering"
    FRUITING = "fruiting"
    MATURATION = "maturation"
    HARVEST = "harvest"
    FALLOW = "fallow"

class DiseaseStatus(Enum):
    SUSPECTED = "suspected"
    CONFIRMED = "confirmed"
    ACTIVE = "active"
    CONTAINED = "contained"
    RESOLVED = "resolved"
    RECURRENT = "recurrent"

class TreatmentType(Enum):
    CHEMICAL_FUNGICIDE = "chemical_fungicide"
    BIOLOGICAL = "biological"
    CULTURAL = "cultural"
    REMOVAL = "removal"
    PREVENTATIVE = "preventative"


# ── First-class KG Node Objects ──

@dataclass
class CropCycle:
    """A first-class episodic object representing a complete crop growing cycle."""
    cycle_id: str
    field_id: str
    crop_type: str
    variety: str
    planting_date: str              # ISO 8601
    harvest_date: Optional[str]
    stages: list[tuple[CropStage, str]]  # (stage, ISO timestamp)
    yield_kg: Optional[float]
    notes: list[str]

@dataclass
class DiseaseEvent:
    """A first-class episodic object representing a disease occurrence with spatial spread."""
    event_id: str
    field_id: str
    crop_cycle_id: str
    disease_name: str
    first_observed: str             # ISO 8601
    status: DiseaseStatus
    severity: float                 # 0.0–1.0
    affected_area_m2: float
    spread_direction: Optional[str] # compass direction or "none"
    spread_rate: Optional[float]    # m²/day
    symptoms: list[str]
    confirmed_by: Optional[str]     # lab test ID or observation

@dataclass
class TreatmentAction:
    """A first-class episodic object representing a treatment applied to a disease event."""
    treatment_id: str
    disease_event_id: str
    treatment_type: TreatmentType
    agent: str                      # active ingredient / method
    dosage: str
    application_date: str           # ISO 8601
    effectiveness: Optional[float]  # 0.0–1.0 (retrospective)
    follow_up_required: bool
    notes: str


# ── Query Language Primitives ──

@dataclass
class TemporalPathQuery:
    """Query for extracting ordered sequences across time."""
    start_node_id: str
    end_node_id: Optional[str]
    relation_sequence: list[str]    # e.g. ["treated_with", "followed_by"]
    max_hops: int = 5
    from_date: Optional[str] = None
    to_date: Optional[str] = None

@dataclass
class SpatialProximityQuery:
    """Query for finding disease events within spatial proximity."""
    center_field_id: str
    radius_m: float = 500.0
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    disease_filter: Optional[list[str]] = None
```

---

## 4. Architecture Overview

The architecture implements a **Complementary Learning Systems (CLS)** paradigm adapted for autonomous agricultural diagnostic agents. It comprises three interacting subsystems:

### Subsystem 1: Episodic KG ("Hippocampus")
- **Fast-write:** New observations, disease reports, treatment actions, and crop cycle events are ingested immediately into a spatio-temporal knowledge graph
- **Rich relational structure:** Nodes have typed attributes (temporal, spatial, domain-specific); edges encode causal, temporal, and spatial relationships
- **Query capabilities:** Temporal path queries (ordered sequences), spatial proximity queries, subgraph extraction
- **Consolidation interface:** Exports matured episodic patterns to the semantic ML layer

### Subsystem 2: Semantic ML Layer ("Neocortex")
- **Slow-learning:** Trained offline on consolidated episodic data
- **Pattern extraction:** Learns generalized disease progression models, treatment efficacy patterns, seasonal correlations
- **Inference:** Given a query or episodic subgraph, produces compressed pattern embeddings and confidence scores
- **Few-shot adaptation:** Can adapt to novel disease-crop combinations from K episodic examples

### Subsystem 3: Agent Controller ("Prefrontal Cortex")
- **Orchestration:** Routes queries, manages the iterative bidirectional protocol between episodic and semantic layers
- **Reconciliation:** Resolves conflicts between episodic specifics and semantic generalizations
- **Response generation:** Produces final diagnostic output with provenance tracking

### Iterative Bidirectional Querying Protocol

```
                    ┌────────────────────────────────────────────┐
                    │          Agent Controller (LLM)            │
                    │                                            │
  User Query ──────►│ 1. Parse query                            │
                    │ 2. Route to episodic + semantic (parallel) │
                    │ 3. Reconcile results                      │
                    │ 4. If gap/conflict: refine and loop       │
                    │ 5. Generate response with provenance       │──► Response
                    └────┬───────────────┬───────────────────────┘
                         │               │
              ┌──────────▼─────┐  ┌──────▼──────────┐
              │  Episodic KG   │  │  Semantic ML     │
              │  (Fast, graph) │  │  (Slow, learned) │
              │                │  │                  │
              │  Timestamped   │  │  GNN encoder     │
              │  Spatial nodes │  │  Pattern slots   │
              │  Causal edges  │  │  Confidence      │
              └────────────────┘  └──────────────────┘
```

### Protocol Steps (Detailed)

1. **Initial parallel query:** Both subsystems queried simultaneously with the user's natural language query
2. **Episodic response:** KG returns specific facts (e.g., "Field A-42 had powdery mildew on May 15-30, treated with sulfur on May 18")
3. **Semantic response:** ML layer returns generalized patterns (e.g., "Powdery mildew in wheat typically appears 7-14 days after humidity > 80%, responds well to sulfur within 72h of first observation")
4. **Reconciliation check:** Controller compares — are they consistent? Does the general pattern apply to this specific case? Are there episodic anomalies that contradict the semantic model?
5. **Refinement pass 1 (semantic → episodic):** If semantic pattern suggests specific risk factors, re-query KG with refined constraints (e.g., "check humidity readings for field A-42 in the week before May 8")
6. **Refinement pass 2 (episodic → semantic):** If episodic findings reveal anomalous disease progression, request semantic layer to update its pattern or provide alternative explanations
7. **Convergence:** Either confidence threshold is met, or max iterations reached
8. **Response:** Final answer with provenance tags per claim (source: episodic/semantic/both, confidence score)

---

## 5. Core Component Pseudocode

### 5.1 Episodic KG — Graph Operations

```python
class EpisodicKnowledgeGraph:
    """Fast-learning episodic memory backed by a spatio-temporal knowledge graph."""

    def __init__(self, config: EpisodicKGConfig):
        self.config = config
        self.nodes: dict[str, KGNode] = {}
        self.edges: list[KGEdge] = []
        self.temporal_index: TemporalIndex = TemporalIndex(
            resolution_seconds=config.temporal_resolution_seconds
        )
        self.spatial_index: SpatialIndex = SpatialIndex(
            grid_size_meters=config.spatial_grid_size_meters
        )
        self.dedup_cache: DedupCache = DedupCache(
            window_seconds=config.dedup_window_seconds
        )
        self.pending_consolidation: list[KGSubgraph] = []

    def fast_write(self, event: DomainEvent) -> NodeID:
        """
        Ingests a new observation/disease/treatment event immediately.
        O(log N) write via temporal + spatial index.

        Steps:
        1. Map domain event to typed KG node + edges
        2. Deduplicate if similar event exists within window_seconds
        3. Add to temporal index (sorted by timestamp)
        4. Add to spatial index (geohash grid)
        5. Append to pending_consolidation buffer for eventual semantic layer training
        """
        # Deduplication check
        dedup_key = self._compute_dedup_key(event)
        if self.dedup_cache.contains(dedup_key):
            existing_id = self.dedup_cache.get(dedup_key)
            # Merge: append notes, update severity if higher
            self._merge_event(existing_id, event)
            return existing_id

        node = self._event_to_node(event)
        node_id = node.node_id
        self.nodes[node_id] = node

        # Create edges to related nodes
        for relation, target_id in self._extract_relations(event):
            edge = KGEdge(
                source=node_id,
                target=target_id,
                relation=relation,
                timestamp=event.timestamp,
            )
            self.edges.append(edge)

        # Update indices
        self.temporal_index.insert(node_id, event.timestamp)
        if event.has_spatial:
            self.spatial_index.insert(node_id, event.location, event.spatial_extent)

        # Buffer for consolidation
        subgraph = self._extract_local_subgraph(node_id, radius=2)
        self.pending_consolidation.append(subgraph)

        return node_id

    def temporal_path_query(self, query: TemporalPathQuery) -> list[KGSubgraph]:
        """
        Extracts ordered temporal sequences from the KG.
        e.g. "Show me the treatment sequence for field A-42 this season"

        Uses BFS over edges filtered by relation_sequence,
        sorted by timestamp at each hop.

        Returns ordered list of subgraphs (each hop = one subgraph).
        """
        results = []
        frontier = [(query.start_node_id, 0)]
        visited = set()

        while frontier and len(results[0].hops) < query.max_hops:
            current_id, depth = frontier.pop(0)

            if current_id in visited:
                continue
            visited.add(current_id)

            # Filter edges by relation for this hop
            expected_rel = query.relation_sequence[depth] if depth < len(query.relation_sequence) else None
            neighbors = self._get_temporal_neighbors(
                current_id,
                relation_filter=expected_rel,
                from_date=query.from_date,
                to_date=query.to_date,
            )

            for neighbor_id, edge in neighbors:
                subgraph = self._extract_subgraph_between(current_id, neighbor_id)
                results.append(subgraph)
                frontier.append((neighbor_id, depth + 1))

        return results

    def spatial_proximity_query(self, query: SpatialProximityQuery) -> list[KGSubgraph]:
        """
        Finds all disease events within spatial proximity of a field.
        e.g. "Are there disease events near field B-17?"

        Uses spatial index for initial candidate filtering,
        then temporal filter and disease-type filter.
        """
        center_geohash = self.spatial_index.lookup(query.center_field_id)
        candidates = self.spatial_index.radius_query(
            center_geohash, query.radius_m
        )
        # Apply temporal and disease filters
        results = []
        for node_id in candidates:
            node = self.nodes[node_id]
            if not isinstance(node, DiseaseEvent):
                continue
            if query.disease_filter and node.disease_name not in query.disease_filter:
                continue
            if query.from_date and node.first_observed < query.from_date:
                continue
            if query.to_date and node.first_observed > query.to_date:
                continue
            subgraph = self._extract_local_subgraph(node_id, radius=1)
            results.append(subgraph)

        return sorted(results, key=lambda r: r.timestamp)

    def extract_consolidation_batch(self, max_samples: int = 256) -> list[KGSubgraph]:
        """
        Returns the oldest pending episodic subgraphs for semantic layer training.

        Maturity heuristic: subgraphs older than consolidation_frequency_minutes
        and with at least one repeat observation are considered "matured."
        """
        now = datetime.now()
        matured = [
            sg for sg in self.pending_consolidation
            if (now - sg.timestamp).total_seconds() > self.config.consolidation_frequency_minutes
            and sg.repeat_count >= 2
        ]
        return matured[:max_samples]

    def prewarm_semantic_query(self, pattern_embedding: torch.Tensor, k: int = 5) -> list[KGSubgraph]:
        """
        Semantic → Episodic: Given a semantic pattern embedding, find the K most
        similar episodic subgraphs. This is how semantic priors guide KG query refinement.

        Uses cosine similarity between pattern_embedding and precomputed
        subgraph embeddings.
        """
        if not hasattr(self, '_subgraph_embeddings'):
            return []
        similarities = cosine_similarity(
            pattern_embedding.unsqueeze(0),
            self._subgraph_embeddings
        )
        top_k_indices = similarities[0].topk(k).indices
        return [self.pending_consolidation[i] for i in top_k_indices]
```

### 5.2 Semantic ML Layer — Pattern Extraction and Inference

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticPatternExtractor(nn.Module):
    """
    Slow-learning semantic layer (neocortex analogue).

    Encodes episodic KG subgraphs into compressed pattern embeddings,
    learns shared prototype patterns across episodes, and provides
    confidence-scored inference for diagnostic generalization.

    Architecture: GCN encoder → prototype attention → pattern embedding
    """

    def __init__(self, config: SemanticMLConfig):
        super().__init__()
        self.config = config

        # Node and edge embedding layers
        self.node_embed = nn.Linear(config.node_embed_dim, config.hidden_dim)
        self.edge_embed = nn.Linear(config.edge_embed_dim, config.hidden_dim)

        # GCN layers for subgraph encoding
        self.gcn_layers = nn.ModuleList()
        for i in range(config.n_layers):
            in_dim = config.hidden_dim if i > 0 else config.hidden_dim
            self.gcn_layers.append(
                GCNLayer(in_dim, config.hidden_dim, dropout=config.dropout)
            )

        # Learnable prototype patterns (semantic memory slots)
        self.prototype_vectors = nn.Parameter(
            torch.randn(config.n_pattern_slots, config.pattern_embed_dim)
        )

        # Prototype attention: attends over prototypes given a subgraph encoding
        self.prototype_attention = nn.MultiheadAttention(
            embed_dim=config.pattern_embed_dim,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Output projector from attended prototypes → pattern embedding
        self.pattern_projector = nn.Sequential(
            nn.Linear(config.pattern_embed_dim, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.pattern_embed_dim),
        )

        # Confidence head: predicts reliability of pattern match
        self.confidence_head = nn.Sequential(
            nn.Linear(config.pattern_embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, subgraph_batch: SubgraphBatch) -> dict:
        """
        Args:
            subgraph_batch: batched episodic KG subgraphs
                - node_features: (B, max_nodes, node_embed_dim)
                - edge_indices: (B, 2, max_edges)
                - edge_features: (B, max_edges, edge_embed_dim)
                - node_mask: (B, max_nodes)

        Returns:
            dict with:
                - pattern_embed: (B, pattern_embed_dim) compressed pattern per subgraph
                - prototype_weights: (B, n_pattern_slots) attention over prototypes
                - confidence: (B,) predicted reliability per prediction
        """
        B, N, _ = subgraph_batch.node_features.shape

        # Embed nodes and edges
        h = self.node_embed(subgraph_batch.node_features)  # (B, N, H)

        # GCN message passing over subgraph structure
        for gcn in self.gcn_layers:
            h = gcn(h, subgraph_batch.edge_indices, subgraph_batch.edge_features)

        # Mask padding nodes
        h = h * subgraph_batch.node_mask.unsqueeze(-1)

        # Pool subgraph nodes to single graph-level vector
        # Mean pooling over valid nodes (masked)
        graph_embed = h.sum(dim=1) / subgraph_batch.node_mask.sum(dim=1, keepdim=True).clamp(min=1)
        # graph_embed: (B, hidden_dim)

        # Project to pattern space
        query = self.pattern_projector(graph_embed).unsqueeze(1)  # (B, 1, P)

        # Attend over prototype vectors
        prototypes = self.prototype_vectors.unsqueeze(0).expand(B, -1, -1)  # (B, S, P)
        attended_pattern, attn_weights = self.prototype_attention(
            query, prototypes, prototypes
        )
        # attended_pattern: (B, 1, P) → squeeze to (B, P)
        pattern_embed = attended_pattern.squeeze(1)

        # Confidence estimation
        confidence = self.confidence_head(pattern_embed).squeeze(-1)

        return {
            "pattern_embed": pattern_embed,
            "prototype_weights": attn_weights,  # (B, 1, S)
            "confidence": confidence,
        }

    def consolidate(self, episodic_subgraphs: list[KGSubgraph]):
        """
        Offline consolidation: train on matured episodic subgraphs.

        This is called periodically (every consolidation_frequency_minutes).
        Loss = contrastive loss: positive pairs (same disease, same stage)
        should be close in pattern space; negative pairs should be far.
        """
        self.train()
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.consolidation_lr
        )

        for epoch in range(10):  # Few epochs — slow but not excessive
            batch_loader = self._build_consolidation_batches(
                episodic_subgraphs,
                batch_size=self.config.consolidation_batch_size,
            )
            for batch in batch_loader:
                output = self.forward(batch)
                # Contrastive loss: subgraphs from same disease class are positive pairs
                loss = contrastive_loss(
                    output["pattern_embed"],
                    batch.labels,
                    margin=1.0,
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

    def few_shot_adapt(self, support_subgraphs: list[KGSubgraph], query: KGSubgraph) -> dict:
        """
        Few-shot adaptation: given K episodic examples of a novel disease-crop
        combination, produce a pattern embedding for the query subgraph.

        Uses prototypical network approach:
        1. Encode support subgraphs → prototype per class
        2. Encode query subgraph
        3. Return closest prototype match + confidence
        """
        self.eval()
        with torch.no_grad():
            support_embeds = []
            for sg in support_subgraphs[:self.config.few_shot_k]:
                batch = self._subgraph_to_batch(sg)
                output = self.forward(batch)
                support_embeds.append(output["pattern_embed"])

            # Prototype = mean of support embeddings
            prototype = torch.stack(support_embeds).mean(dim=0)

            # Encode query
            query_batch = self._subgraph_to_batch(query)
            query_output = self.forward(query_batch)

            # Similarity
            similarity = F.cosine_similarity(
                query_output["pattern_embed"], prototype.unsqueeze(0)
            )

        return {
            "pattern_embed": query_output["pattern_embed"],
            "matched_prototype": prototype,
            "similarity": similarity.item(),
            "confidence": query_output["confidence"].item(),
        }


class SemanticMemoryManager:
    """
    Manages the semantic ML layer lifecycle: training, inference, consolidation schedule,
    and the Episodic → Semantic and Semantic → Episodic query paths.
    """

    def __init__(self, config: SemanticMLConfig):
        self.config = config
        self.extractor = SemanticPatternExtractor(config)
        self.last_consolidation_time: Optional[datetime] = None
        self.pattern_cache: dict[str, torch.Tensor] = {}  # disease → pattern_embed

    def infer_pattern(self, query: str | KGSubgraph) -> SemanticInferenceResult:
        """
        Given a text query or episodic subgraph, returns the best matching
        semantic pattern with confidence score.

        For text queries: first embed the query, then match against prototype vectors.
        For subgraphs: run the GCN encoder and match against prototypes.
        """
        if isinstance(query, str):
            # Text-to-pattern: embed query text and find closest prototype
            query_embed = self._embed_text(query)
            similarities = F.cosine_similarity(
                query_embed.unsqueeze(0),
                self.extractor.prototype_vectors
            )
            best_idx = similarities.argmax()
            confidence = self.extractor.confidence_head(
                self.extractor.prototype_vectors[best_idx]
            )
            return SemanticInferenceResult(
                pattern_embed=self.extractor.prototype_vectors[best_idx],
                matched_prototype_idx=best_idx.item(),
                confidence=confidence.item(),
                provenance="semantic_prototype",
            )
        else:
            # KG subgraph → pattern
            batch = self._subgraph_to_batch(query)
            output = self.extractor.forward(batch)
            return SemanticInferenceResult(
                pattern_embed=output["pattern_embed"],
                confidence=output["confidence"].item(),
                provenance="gcn_encoder",
                prototype_weights=output["prototype_weights"],
            )

    def query_episodic_via_semantic_prior(
        self,
        semantic_result: SemanticInferenceResult,
        episodic_kg: EpisodicKnowledgeGraph,
        k: int = 5,
    ) -> list[KGSubgraph]:
        """
        Semantic → Episodic query refinement.

        Takes a semantic pattern embedding and retrieves the K most similar
        episodic subgraphs from the KG. This enables semantic priors to
        guide KG search (e.g., "this pattern suggests early-stage mildew —
        check observations within 7 days of high humidity").
        """
        return episodic_kg.prewarm_semantic_query(
            semantic_result.pattern_embed, k=k
        )

    def consolidate(self, episodic_kg: EpisodicKnowledgeGraph):
        """
        Scheduled offline consolidation: extract matured episodic subgraphs,
        train the semantic extractor on them.
        """
        batch = episodic_kg.extract_consolidation_batch(
            max_samples=self.config.consolidation_batch_size * 10
        )
        if len(batch) < self.config.consolidation_batch_size:
            logger.info("Not enough matured episodes for consolidation; skipping.")
            return

        logger.info(f"Consolidating {len(batch)} episodic subgraphs...")
        self.extractor.consolidate(batch)
        self.last_consolidation_time = datetime.now()
        logger.info("Consolidation complete.")

        # Warm the pattern cache for common disease types
        self._rebuild_pattern_cache()
```

### 5.3 Agent Controller — Orchestrating Iterative Bidirectional Querying

```python
class CLSAgentController:
    """
    LLM-based orchestration layer for CLS bicameral memory.

    Manages the iterative bidirectional querying protocol between episodic KG
    and semantic ML layer. Generates final responses with provenance tracking.
    """

    def __init__(
        self,
        config: AgentControllerConfig,
        episodic_kg: EpisodicKnowledgeGraph,
        semantic_memory: SemanticMemoryManager,
    ):
        self.config = config
        self.episodic_kg = episodic_kg
        self.semantic_memory = semantic_memory
        self.llm = self._init_llm(config)
        self.working_memory: WorkingMemory = WorkingMemory(
            max_tokens=config.working_memory_max_tokens,
            eviction_policy=config.working_memory_eviction,
        )

    def diagnose(self, query: str, context: DiagnosticContext) -> DiagnosticResponse:
        """
        Main entry point for agricultural diagnostics.

        Implements the iterative bidirectional querying protocol.

        Returns:
            DiagnosticResponse with:
            - answer: natural language diagnosis
            - provenance: per-claim source tags
            - num_iterations: cycles used
            - confidence: overall confidence score
            - evidence: list of supporting facts with source
        """
        self.working_memory.clear()
        self.working_memory.add("query", query)
        self.working_memory.add("context", context)

        # ── Step 1: Parse query ──
        parsed = self._parse_diagnostic_query(query, context)

        # ── Step 2: Initial parallel query ──
        iteration = 0
        episode_results = self.episodic_kg.fast_write(parsed.to_event())
        # Query KG for relevant history
        kg_results = self.episodic_kg.temporal_path_query(
            TemporalPathQuery(
                start_node_id=context.field_id,
                relation_sequence=["occurred_during", "treated_with", "followed_by"],
                from_date=context.season_start,
                to_date=context.season_end,
            )
        )
        semantic_results = self.semantic_memory.infer_pattern(query)

        # ── Step 3-6: Iterative reconciliation loop ──
        reconciliation_log = []
        while iteration < self.config.max_iterative_cycles:
            iteration += 1
            self.working_memory.add(f"iteration_{iteration}_kg", kg_results)
            self.working_memory.add(f"iteration_{iteration}_semantic", semantic_results)

            # Reconciliation: compare episodic and semantic findings
            reconciliation = self._reconcile(
                episodic_facts=kg_results,
                semantic_pattern=semantic_results,
                iteration=iteration,
            )
            reconciliation_log.append(reconciliation)

            # Check early exit
            if reconciliation.confidence >= self.config.early_exit_confidence:
                break

            # Decide refinement direction based on reconciliation gaps
            if reconciliation.semantic_prior_needed:
                # Episodic → Semantic refinement: KG results suggest a pattern
                # that doesn't match any known prototype → request semantic revision
                refined_pattern = self.semantic_memory.few_shot_adapt(
                    support_subgraphs=kg_results[:self.semantic_memory.config.few_shot_k],
                    query=self._build_query_subgraph(query, context),
                )
                semantic_results = SemanticInferenceResult(
                    pattern_embed=refined_pattern["pattern_embed"],
                    confidence=refined_pattern["confidence"],
                    provenance="few_shot_adapted",
                )

            if reconciliation.episodic_refinement_needed:
                # Semantic → Episodic refinement: semantic pattern suggests
                # specific risk factors → re-query KG with refined constraints
                kg_results = self.semantic_memory.query_episodic_via_semantic_prior(
                    semantic_results,
                    self.episodic_kg,
                    k=self.semantic_memory.config.few_shot_k,
                )

        # ── Step 8: Generate response with provenance ──
        response = self._generate_response(
            query=query,
            reconciliation_log=reconciliation_log,
            kg_results=kg_results,
            semantic_results=semantic_results,
        )

        return DiagnosticResponse(
            answer=response["answer"],
            provenance=response["provenance"],
            num_iterations=iteration,
            confidence=reconciliation.confidence if reconciliation_log else 0.0,
            evidence=response["evidence"],
        )

    def _reconcile(
        self,
        episodic_facts: list[KGSubgraph],
        semantic_pattern: SemanticInferenceResult,
        iteration: int,
    ) -> ReconciliationResult:
        """
        Compares episodic facts against semantic pattern to identify:
        - Consistency: do the facts match the pattern?
        - Gaps: does the pattern suggest factors not in the KG?
        - Contradictions: do facts contradict the pattern?

        Returns a structured reconciliation result with flags for
        which refinement direction is needed.
        """
        # Build prompt for LLM judge
        if self.config.reconciliation_method == "llm_judge":
            prompt = f"""Compare the following episodic facts against the semantic pattern.
Assess consistency, identify gaps, and flag contradictions.

Episodic facts:
{self._format_subgraphs(episodic_facts)}

Semantic pattern (confidence: {semantic_pattern.confidence:.2f}):
{self._format_pattern(semantic_pattern)}

Output JSON:
{{
    "consistency_score": <0.0-1.0>,
    "gaps": [<list of missing information>],
    "contradictions": [<list of contradictions>],
    "semantic_prior_needed": <true/false>,
    "episodic_refinement_needed": <true/false>,
    "refined_query_suggestion": "<optional revised query>"
}}"""
            result = self.llm.complete(prompt, temperature=0.0)
            return ReconciliationResult.parse(result)
        else:
            # Fallback: confidence-based reconciliation
            return ReconciliationResult(
                consistency_score=semantic_pattern.confidence,
                gaps=[],
                contradictions=[],
                semantic_prior_needed=(
                    semantic_pattern.confidence < self.config.early_exit_confidence
                    and len(episodic_facts) > 0
                ),
                episodic_refinement_needed=(
                    semantic_pattern.confidence < self.config.early_exit_confidence
                    and len(episodic_facts) > 0
                ),
                confidence=semantic_pattern.confidence,
            )

    def _generate_response(
        self,
        query: str,
        reconciliation_log: list[ReconciliationResult],
        kg_results: list[KGSubgraph],
        semantic_results: SemanticInferenceResult,
    ) -> dict:
        """
        Generates final diagnostic response using LLM with full context.

        Provenance tracking: every claim in the output is tagged with
        whether it came from episodic KG, semantic ML, or both.

        Response structure:
        - answer: natural language diagnosis
        - provenance: list of {claim, source, confidence} triples
        - evidence: supporting evidence from each layer
        """
        provenance_items = []

        # Tag episodic facts
        for sg in kg_results:
            provenance_items.append({
                "claim": sg.summary,
                "source": "episodic_kg",
                "confidence": sg.confidence,
                "timestamp": sg.timestamp.isoformat(),
            })

        # Tag semantic pattern
        provenance_items.append({
            "claim": f"Semantic pattern match: {semantic_results.matched_prototype_idx}",
            "source": "semantic_ml",
            "confidence": semantic_results.confidence,
            "provenance": semantic_results.provenance,
        })

        prompt = f"""You are an agricultural diagnostic agent with a bicameral memory system.
Given the user's query, the episodic facts (specific observations), and the semantic
pattern (generalized agricultural knowledge), produce a diagnosis.

User query: {query}

Episodic facts (specific, timestamped):
{self._format_subgraphs(kg_results)}

Semantic pattern (generalized, confidence {semantic_results.confidence:.2f}):
{self._format_pattern(semantic_results)}

Reconciliation history ({len(reconciliation_log)} iterations):
{self._format_reconciliation_log(reconciliation_log)}

Provide:
1. A clear diagnosis
2. Confidence level
3. Supporting evidence from each memory system
4. Any caveats or alternative explanations"""
        return self.llm.complete(prompt, temperature=self.config.llm_temperature)
```

### 5.4 Consolidation and Working Memory

```python
class ConsolidationScheduler:
    """
    Manages the offline consolidation lifecycle.

    Consolidation frequency: configurable (default: daily).
    Warmup period: no consolidation before warmup_hours has elapsed (allows
    sufficient episodic data to accumulate).

    Consolidation triggers:
    - Scheduled: every consolidation_frequency_minutes
    - Threshold-based: when pending_consolidation buffer exceeds 80% capacity
    - On-demand: explicitly triggered by agent controller (e.g., after a
      high-value diagnostic session)
    """

    def __init__(self, config: SemanticMLConfig, episodic_kg, semantic_memory):
        self.config = config
        self.episodic_kg = episodic_kg
        self.semantic_memory = semantic_memory
        self.system_start_time = datetime.now()
        self.consolidation_count = 0

    def should_consolidate(self) -> bool:
        """Check whether consolidation should run."""
        # Warmup period check
        hours_elapsed = (datetime.now() - self.system_start_time).total_seconds() / 3600
        if hours_elapsed < self.config.consolidation_warmup_hours:
            return False

        # Time-based check
        if self.semantic_memory.last_consolidation_time:
            minutes_since = (
                datetime.now() - self.semantic_memory.last_consolidation_time
            ).total_seconds() / 60
            if minutes_since < self.config.consolidation_frequency_minutes:
                return False

        return True

    def step(self):
        """Called periodically (e.g., every minute in a background loop)."""
        if self.should_consolidate():
            self.semantic_memory.consolidate(self.episodic_kg)
            self.consolidation_count += 1


class WorkingMemory:
    """
    Short-term working memory buffer for active diagnostic session.

    Holds:
    - Current query context
    - Iteration-level results (episodic + semantic per cycle)
    - Reconciliation log
    - Intermediate reasoning state

    Eviction policy prevents context overflow.
    Session-scoped: cleared at end of each diagnostic session.
    """

    def __init__(self, max_tokens: int = 16_000, eviction_policy: str = "lru"):
        self.max_tokens = max_tokens
        self.eviction_policy = eviction_policy
        self.items: dict[str, Any] = {}
        self.access_log: list[tuple[str, datetime]] = []

    def add(self, key: str, value: Any):
        """Add item, evicting if over capacity."""
        tokens = estimate_tokens(str(value))
        while self._total_tokens() + tokens > self.max_tokens:
            self._evict_one()
        self.items[key] = value
        self.access_log.append((key, datetime.now()))

    def get(self, key: str) -> Optional[Any]:
        """Retrieve item, updating access log for LRU eviction."""
        self.access_log.append((key, datetime.now()))
        return self.items.get(key)

    def clear(self):
        """Clear working memory for new session."""
        self.items.clear()
        self.access_log.clear()
```

---

## 6. ASCII Architecture Diagram

```
                                ┌─────────────────────────────────────┐
                                │         User / Agent Interface      │
                                │  "Why is my wheat showing powdery   │
                                │   mildew in field A-42 this season?" │
                                └──────────────┬──────────────────────┘
                                               │
                                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    AGENT CONTROLLER (LLM Orchestrator)                   │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                    Working Memory (Session)                        │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────────┐ │  │
│  │  │ Query    │ │ Iter 1   │ │ Iter 2   │ │ Reconciliation Log   │ │  │
│  │  │ Context  │ │ Results  │ │ Results  │ │ (3-5 entries)        │ │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌─────────────────┐   ┌────────────────────────────┐   ┌─────────────┐ │
│  │ Query Parser    │──▶│ Iterative Bidirectional    │──▶│ Response    │ │
│  │ (NL → structured│   │ Querying Protocol           │   │ Generator   │ │
│  │  query)         │   │                            │   │ (with       │ │
│  └─────────────────┘   │ ┌──────┐ ┌──────┐ ┌─────┐ │   │  provenance)│ │
│                         │ │Init  │ │Recon-│ │Refi-│ │   └─────────────┘ │
│                         │ │Query │ │cile  │ │ne   │ │                   │
│                         │ └──────┘ └──────┘ └─────┘ │                   │
│                         └────────────────────────────┘                   │
└──────────────────────────┬───────────────────────────────────────────────┘
                           │                           │
              ┌────────────▼────────────┐    ┌─────────▼──────────────┐
              │    EPISODIC KG          │    │    SEMANTIC ML LAYER   │
              │    (Hippocampus)        │    │    (Neocortex)         │
              │    Fast-write / Fast-read│    │    Slow-learn /       │
              │                         │    │    Fast-infer         │
              │                         │    │                        │
              │  ┌─────────────────┐    │    │  ┌──────────────────┐  │
              │  │ Temporal Index  │    │    │  │ GCN Subgraph     │  │
              │  │ (sorted by ts)  │    │    │  │ Encoder          │  │
              │  └────────┬────────┘    │    │  └────────┬─────────┘  │
              │           │            │    │           │              │
              │  ┌────────▼────────┐   │    │  ┌────────▼─────────┐  │
              │  │ Spatial Index   │   │    │  │ Prototype Slots  │  │
              │  │ (geohash grid)  │   │    │  │ (64 learned      │  │
              │  └────────┬────────┘   │    │  │  patterns)       │  │
              │           │            │    │  └────────┬─────────┘  │
              │  ┌────────▼────────┐   │    │           │              │
              │  │ Graph Store     │   │    │  ┌────────▼─────────┐  │
              │  │ 10K-50K triples │   │    │  │ Confidence Head  │  │
              │  │ Nodes + Edges   │   │    │  │ + Few-Shot Adapt │  │
              │  └─────────────────┘   │    │  └──────────────────┘  │
              │                         │    │                        │
              │  Domain Objects:        │    │  Consolidation:       │
              │  ├─ CropCycle (stage†)  │    │  ├─ Daily offline     │
              │  ├─ DiseaseEvent (sp†)  │    │  ├─ Contrastive loss  │
              │  ├─ TreatmentAction (†) │    │  └─ Prototype update  │
              │  └─ Observation         │    │                        │
              └─────────────────────────┘    └────────────────────────┘
                           │                           │
                           └───────────┬───────────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │ Consolidation    │
                              │ Scheduler        │
                              │ (Daily / on-demand) │
                              │                  │
                              │ Episodic → Semantic  │
                              │ (matured patterns)   │
                              └─────────────────┘


┌────────────────────────────────────────────────────────────────────────────┐
│                        DATA FLOW SEQUENCE                                │
│                                                                            │
│  QUERY: "Why is my wheat showing powdery mildew in field A-42?"          │
│                                                                            │
│  1. [PARSE]        Agent Controller → extracts {field, crop, symptom}     │
│  2. [WRITE]        Agent Controller → Episodic KG: fast_write(obs)        │
│  3. [QUERY]        Agent Controller → Episodic KG: temporal_path_query()  │
│                    Returns: [May 8: high humidity, May 15: first spots,   │
│                              May 18: sulfur treatment]                   │
│  4. [INFER]        Agent Controller → Semantic ML: infer_pattern()        │
│                    Returns: "Powdery mildew in wheat: humidity>80%,       │
│                              sulfur effective within 72h (conf: 0.85)"    │
│  5. [RECONCILE]    Agent Controller: compare facts vs pattern             │
│                    Gap: "Did humidity exceed 80% before May 8?"           │
│  6. [REFINE]       Agent Controller → Episodic KG: spatial_proximity()    │
│                    → checks weather station near A-42 → humidity 85%      │
│  7. [RECONCILE 2]  Pattern confirmed; confidence → 0.92 → early exit     │
│  8. [RESPOND]      Generated with provenance tags                        │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Inductive Bias Justifications

### 7.1 Episodic KG Design Choices

| Choice | Justification | Evidence Status |
|---|---|---|
| **Spatio-temporal KG over vector store** | Agriculture inherently requires temporal ordering (crop cycles, treatment sequences) and spatial relationships (disease spread). Vector stores flatten these into embedding space, losing explicit relational structure. Graph models preserve them as first-class citizens. | Grounded: Zep/Graphiti (arXiv:2501.13956) achieves 94.8% on DMR via temporal KG |
| **Temporal+spatial dual indexing** | Disease queries split evenly between "when did this happen?" and "where did this happen?" — separate indices prevent either axis from being a sequential scan bottleneck. | Hypothesis: Expected latency improvement over single-index approaches |
| **First-class CropCycle/DiseaseEvent/TreatmentAction objects** | These are not generic KG nodes — they carry structured fields (growth stage enum, disease status FSM, spatial spread vector) that enable domain-specific query optimizations (e.g., "find all fields in flowering stage within 500m of a confirmed rust case"). | Hypothesis: Expected to reduce query complexity vs. generic property graph |
| **Fast-write with dedup window** | Agricultural sensors and field reports can produce duplicate observations within minutes. A 5-minute dedup window prevents KG bloat without losing genuine repeated observations (which may signal worsening conditions). | Grounded: Common pattern in time-series data pipelines |
| **Append-only admission (no surprise filtering)** | Unlike Titans' surprise-based admission, we admit all events — in agriculture, a "boring" observation (e.g., "no disease found") is as informative as a surprising one for ruling out conditions. | Hypothesis: Surprise-based filtering would discard diagnostically useful negative evidence |
| **Temporal path query max depth = 5** | A typical agricultural episode (observation → diagnosis → treatment → follow-up → resolution) spans 5 hops. Deeper queries indicate disconnected episodes. | Grounded: Domain analysis of agricultural diagnostic workflows |
| **Spatial proximity default = 500m** | Disease spread between fields is most correlated within 500m (wind-borne spores, shared irrigation). Larger radii return too many false positives. | Hypothesis: Based on typical field sizes; requires empirical validation |

### 7.2 Semantic ML Layer Design Choices

| Choice | Justification | Evidence Status |
|---|---|---|
| **GCN encoder over transformer** | Episodic KG subgraphs are small (10-50 nodes), irregularly structured, and benefit from localized message passing. Transformers assume dense, fixed-size inputs. GCNs match the graph structure naturally. | Grounded: GCNs are the standard encoder for KG subgraphs in neuro-symbolic systems |
| **Prototype attention (64 learned slots)** | Agricultural disease patterns cluster naturally — e.g., "powdery mildew presentation in wheat" is a recurring prototype. Learned slots allow the model to discover these clusters without explicit labeling. 64 slots covers common disease × crop combinations while leaving room for novel patterns. | Hypothesis: 64 slots derived from ~20 common crops × ~3 major disease types each |
| **Contrastive consolidation loss** | The semantic layer should learn that the same disease across different fields produces similar pattern embeddings, while different diseases produce distinct embeddings. Contrastive loss achieves this without requiring exhaustive labeled data. | Grounded: Standard approach for representation learning in limited-label regimes |
| **Few-shot adaptation via prototypical network** | Novel disease-crop combinations (e.g., a new fungal strain in a region) will have few episodic examples. Prototypical networks are the simplest well-studied approach for K-shot generalization. | Grounded: Prototypical Networks (Snell et al., NeurIPS 2017) |
| **Daily consolidation frequency** | Agricultural disease progression operates on 3-14 day cycles. Daily consolidation ensures the semantic layer updates faster than the disease can spread, but not so fast that it churns on noisy single-day observations. | Hypothesis: Domain-specific; may need adjustment after empirical evaluation |
| **48-hour warmup period** | The semantic layer needs enough episodic data to discover meaningful patterns. Training on the first few hours yields degenerate prototypes. | Grounded: Common best practice for contrastive representation learning |

### 7.3 Agent Controller Design Choices

| Choice | Justification | Evidence Status |
|---|---|---|
| **LLM-based orchestration over hardcoded rules** | Agricultural diagnostic queries are diverse and nuanced (e.g., "is this related to the rust we saw last year?"). An LLM provides the natural language understanding and reasoning flexibility that rules cannot match. | Grounded: LLM-based agents are the standard for flexible task decomposition |
| **Low temperature (0.1) for diagnostic reasoning** | Diagnostic accuracy requires determinism and reproducibility. Stochastic responses would undermine user trust in a decision-support system. | Grounded: Standard practice for factual/medical reasoning tasks |
| **Max 5 iterative cycles** | Agricultural diagnostics need real-time response (farmer in the field). Each iteration adds ~200-500ms latency. 5 cycles at <1s each keeps total under 5s. Beyond 5 cycles, either the architecture is missing information or the query is unanswerable. | Hypothesis: Bounded by latency requirement; expected to converge in 2-3 cycles empirically |
| **Parallel initial query (not sequential)** | In the first pass, neither episodic nor semantic has an information advantage. Parallelizing the initial query reduces latency by 1 round-trip. | Grounded: Obvious latency optimization |
| **LLM-based reconciliation ("llm_judge")** | Reconciling episodic specifics with semantic generalizations requires nuanced comparison (e.g., "this season's unusual weather pattern may override the historical norm"). An LLM judge provides this reasoning. | Hypothesis: Weighted-vote or confidence-max alternatives may be faster but less accurate |
| **Provenance tracking per claim** | In diagnostic decision support, the user must know which facts come from direct observation (episodic) vs. learned patterns (semantic) to assess trustworthiness. Provenance enables this. | Grounded: Requirement for interpretable AI in high-stakes domains |

### 7.4 Overall System Design

| Choice | Justification | Evidence Status |
|---|---|---|
| **Bicameral separation (not monolithic)** | The Stability Gap (arXiv:2601.15313) proves that monolithic neural memory collapses at N=5 facts with ρ>0.6 semantic density. Agricultural disease data exceeds this density. Separation is a geometric necessity, not an empirical preference. | Grounded: Formal proof in arXiv:2601.15313 |
| **Iterative bidirectional querying (not one-directional consolidation)** | AOI, All-Mem, and Dual-System all consolidate one-directionally (episodic→semantic). Iterative bidirectional querying allows semantic priors to refine KG search and episodic anomalies to update semantic beliefs. This is the primary architectural novelty. | Hypothesis: No existing work evaluates this pattern in a CLS memory framework |
| **Agent controller as separate orchestrator (not embedded in either memory system)** | Neither the KG nor the ML layer should own the query logic — the controller provides a clean separation of concerns, making each subsystem independently testable and replaceable. | Grounded: Standard architectural principle (separation of concerns) |

---

## 8. Research-to-Architecture Traceability

| Research Contract Item | Architecture Decision | Evidence Status | Validation Hook |
|---|---|---|---|
| **Novelty Claim 1:** Iterative bidirectional querying outperforms one-directional consolidation | AgentController.max_iterative_cycles >= 2, reconciliation_method = "llm_judge", enable_semantic_prior_routing + enable_episodic_revision both True | Hypothesis | Ablation A: set enable_semantic_prior_routing=False AND enable_episodic_revision=False → compare accuracy on counterfactual queries |
| **Novelty Claim 2:** Domain-specific spatio-temporal episodic objects for agriculture | EpisodicKGConfig.enable_crop_cycle_objects=True, enable_disease_front_objects=True; first-class CropCycle/DiseaseEvent/TreatmentAction dataclasses with temporal and spatial fields | TODO: unverified | Ablation B: replace with generic KG nodes (no typed fields) → compare temporal reasoning accuracy |
| **Novelty Claim 3:** Monolithic memory insufficient for ag diagnostics (Stability Gap) | Bicameral architecture by construction: Episodic KG + Semantic ML are separate subsystems with independent storage | Grounded | Ablation C: merge into single KG store (no semantic layer) → measure accuracy drop at ρ>0.6 |
| **Baseline: Zep/Graphiti** | EpisodicKGConfig uses temporal KG with indexed timestamps (similar to Graphiti's core mechanism) | Must evaluate | Run same LongMemEval-style benchmark on Zep vs. proposed architecture |
| **Baseline: Mem0** | AgentController uses multi-signal retrieval (semantic embedding + entity match) as alternative reconciliation method | Must evaluate | Compare Mem0 (Apr 2026) on agricultural benchmark vs. proposed |
| **Baseline: AOI three-layer** | Architecture has 3 tiers: Working Memory (session) + Episodic KG + Semantic ML, mirroring AOI's Working→Episodic→Semantic | Must evaluate | Compare against AOI if reproducible; else compare against one-directional ablation (our AOI-equivalent) |
| **Evaluation: LongMemEval / LOCOMO for agriculture** | AgentController interface is benchmark-agnostic; DiagnosticContext supports agricultural facts (crop cycles, disease events) | Must implement | Adapt LongMemEval: replace general facts with agricultural domain facts |
| **Evaluation: Fact recall precision/recall** | EpisodicKG.temporal_path_query() returns ordered sequences; provenance tracking tags per-fact source | Grounded requirement | Unit test: write 50 facts, query each, measure recall@k |
| **Evaluation: Generalization accuracy (semantic)** | SemanticPatternExtractor.consolidate() with contrastive loss; few_shot_adapt() for novel patterns | Grounded requirement | Hold-out disease × crop combinations; measure semantic accuracy on held-out patterns |
| **Evaluation: Counterfactual reasoning** | Iterative bidirectional querying protocol enables "what-if" by combining episodic specifics with semantic generalizations | Hypothesis | Evaluation: query "what if we had treated on day 3 instead of day 7?" — compare accuracy vs. monolithic |
| **Evaluation: Temporal reasoning** | TemporalIndex enables timestamp-sorted queries; TemporalPathQuery returns ordered sequences | Grounded requirement | Unit test: temporal ordering queries with shuffled inputs |
| **Evaluation: Semantic Density Robustness** | ρ sweep is an evaluation protocol, not an architectural parameter — but the architecture must handle all densities gracefully | Evaluation requirement | Sweep ρ from 0.1 to 0.9 while keeping fact count constant; measure recall degradation |
| **Evaluation: Latency budget** | AgentControllerConfig.max_iterative_cycles=5, iteration_timeout_ms=1000, parallel_initial_query=True | Must measure | Profile end-to-end latency per query; check p99 < 5s |
| **Blocking unknown: Stability Gap at realistic density** | No architectural change needed — this is an empirical test. The architecture is motivated by this theory. | TODO: unverified | Evaluate recall degradation at N={1,3,5,10,20} with ρ=0.7 on agricultural facts |
| **Blocking unknown: Iterative querying latency** | AgentControllerConfig.max_iterative_cycles is configurable; recommend starting at 3, increasing only if accuracy gains justify latency cost | Must measure | Sweep max_iterative_cycles={1,2,3,5}; measure accuracy vs. latency tradeoff curve |
| **Blocking unknown: Agricultural KG bootstrapping** | EpisodicKGConfig.kg_backend supports "in_memory" for prototyping; the node/edge type schema must be populated | TODO: unverified | Feasibility: populate CropCycle + DiseaseEvent schemas from existing ontologies (AgroPortal, Crop Ontology) |
| **Existing work: OpenAg, NeuroCausal-FusionNet** | Differentiated by CLS framing + iterative bidirectional querying; not trying to compete on agricultural feature set | Must cite | Compare only on the specific metric of CLS-grounded diagnostic accuracy (not on general ag tasks) |

---

## 9. Domain-Specific Considerations

### 9.1 LM / Memory Systems

- **Position / order scheme:** The Episodic KG uses **explicit temporal indexing** (sorted by timestamp) rather than positional embeddings. This is the natural choice for a KG where every edge has a timestamp attribute. There is no sequence length limit beyond the KG capacity.
- **Causal contract:** The temporal index provides strict causality — queries can be constrained to `from_date`/`to_date` ranges, and temporal path queries respect edge direction + timestamp ordering. No information leakage from future to past.
- **Fast/slow separation:** The Stability Gap (arXiv:2601.15313) mandates that fast (episodic) and slow (semantic) memory use **different storage representations** — a graph store and a neural network, respectively. Simply separating into two vector databases would not avoid the Stability Gap because both would use the same underlying representation.

### 9.2 Graph ML

- **Expressiveness ceiling:** The episodic KG is a **property graph** with typed nodes and edges, which is equivalent to 1-WL (Weisfeiler-Lehman) expressiveness. This is sufficient for agricultural diagnostics because:
  - Disease spread is fundamentally a local phenomenon (neighbor fields, shared equipment)
  - Temporal sequences are linear paths, not complex graph structures
  - The semantic ML layer provides the expressiveness beyond 1-WL via its learned prototype attention
- **Permutation invariance:** Node-level queries (e.g., "what happened at field A-42?") are permutation-invariant by construction (nodes are identified by ID). Graph-level queries (e.g., "what's the disease pattern across the farm?") are invariant because the KG stores all edges explicitly and queries traverse them systematically.
- **Positional encoding:** Not needed — the KG has explicit spatial coordinates and timestamps. Positional encodings (like Laplacian eigenvectors) are for graphs without a canonical coordinate system, which we have.
- **Edge features:** Used extensively — every KG edge carries a timestamp and relation type. Edge features are critical for temporal path queries.
- **Scalability:** Full-graph traversal is avoided. All queries use indexed access (temporal index, spatial index, node ID lookup). Maximum subgraph extraction radius is bounded (max_depth=5, radius=500m).
- **Heterophily:** Agricultural disease KGs are inherently heterophilic — a healthy field and a diseased field may be neighbors, and the KG must capture this contrast (not smooth over it). GCN message passing could smooth away this signal. **Mitigation:** The semantic ML layer uses edge features (including "is_infected" and temporal deltas) to prevent over-smoothing across class boundaries.

### 9.3 Scientific ML (Agriculture)

- **Physics constraints:** Not enforced as hard constraints in the architecture. Agricultural disease progression is stochastic enough that hard PDE constraints would be misleading. Soft constraints could be added via a disease progression model in the semantic ML layer (e.g., "infection probability given temperature, humidity, and proximity to known cases").
- **Function vs. operator learning:** The semantic ML layer is a **function** (maps episodic subgraph → pattern embedding), not a neural operator. This is appropriate because we're classifying/retrieving patterns, not solving PDE families.
- **Symmetry/equivariance:** Not required — agricultural fields have fixed geographic coordinates. There's no rotational or translational symmetry to exploit.
- **Mesh type:** The KG is a **graph on arbitrary point cloud** (fields are discrete locations with coordinates). No regular grid or mesh is assumed.
- **Rollout stability:** Not applicable — the architecture does not do multi-step time-stepping. Each query is independent. Temporal reasoning is done via graph queries, not autoregressive prediction.

### 9.4 GenAI / LLM (Agent Controller)

- **Conditioning interface:** The agent controller LLM is conditioned on working memory content (query + episodic results + semantic results + reconciliation history). No special conditioning architecture (cross-attention, AdaLN) is needed — standard prompt-based conditioning suffices for an orchestrator.
- **Sampling efficiency:** A single LLM call per iteration (reconciliation) + one final call (response generation). With max 5 iterations, worst case is 6 LLM calls per diagnostic query. Each call is <500 tokens output.
- **Tool use:** The LLM does not generate KG queries via tool calls (to avoid latency and parsing errors). Instead, the controller pre-computes KG and ML results and passes them as structured context. The LLM's role is reconciliation, not query generation.

---

## 10. Implementation Risks

### Risk 1: LLM-Based Reconciliation Latency

| Severity | Medium |
|---|---|
| **Impact** | Each reconciliation step requires an LLM call. At 5 iterations × ~1s per LLM call, the reconciliation loop alone takes 5s — before KG queries, ML inference, and response generation. Total could exceed 10s. |
| **Mitigation** | (a) Start with `max_iterative_cycles=3`; increase only if accuracy gains justify. (b) Implement fallback to `confidence_max` reconciliation (no LLM) for latency-critical queries. (c) Cache reconciliation results for similar query patterns. |
| **Falsification** | Latency ablation: when `reconciliation_method=confidence_max` (no LLM), accuracy must not drop below 90% of LLM-based reconciliation accuracy. |

### Risk 2: GCN Over-Smoothing in Semantic ML Layer

| Severity | Medium-High |
|---|---|
| **Impact** | The GCN encoder may over-smooth node features across disease boundaries (healthy vs. infected fields are neighbors but should remain distinct). This would collapse the pattern embedding space and destroy semantic discriminability. |
| **Mitigation** | (a) Use edge features (including disease status, temporal difference) in message passing — GCNLayer must be edge-feature-aware. (b) Keep n_layers=3 or lower (deeper GCNs over-smooth more). (c) Use residual connections + layer normalization after each GCN layer. (d) Monitor prototype slot utilization — if most queries collapse to 2-3 prototypes, over-smoothing is confirmed. |
| **Falsification** | Prototype utilization test: after consolidation, measure entropy of prototype_weights distribution across a held-out evaluation set. If entropy < 0.5 (i.e., most queries map to <3 prototypes), the model is over-smoothing. |

### Risk 3: Consolidation Catastrophic Forgetting

| Severity | Medium |
|---|---|
| **Impact** | Daily consolidation retrains the semantic extractor on new episodic data. Without replay of old episodes, the model may forget earlier-season disease patterns (e.g., early blight patterns from May may be forgotten by August when late blight dominates). |
| **Mitigation** | (a) Maintain a replay buffer of exemplar episodic subgraphs from each disease × season combination. (b) Use elastic weight consolidation (EWC) or similar regularization to protect prototype vectors for less-frequent diseases. (c) Fallback: if confidence < 0.3, defer to episodic KG directly (skip semantic layer). |
| **Falsification** | Temporal forgetting test: query a disease pattern from 3 months ago. If semantic accuracy drops >15% compared to the original evaluation, forgetting is confirmed. |

### Risk 4: Episodic KG Write Contention Under High-Volume Sensor Ingestion

| Severity | Low-Medium |
|---|---|
| **Impact** | If the architecture ingests data from IoT sensors (e.g., hourly soil moisture, daily drone imagery), the KG may experience write contention that blocks read queries. |
| **Mitigation** | (a) Implement read-write separation: writes go to a write-ahead log (WAL), reads go to the indexed snapshot. (b) Batch sensor writes every 5 minutes rather than per-reading. (c) The `write_admission` policy can switch to "batch" mode under high volume. |
| **Falsification** | Write contention test: simulate 1000 sensor readings/minute while running concurrent diagnostic queries. If p99 read latency exceeds 2s, mitigation is needed. |

---

## 11. Baseline & Evaluation Requirements

### 11.1 Baseline Systems (Carried Forward from Research Contract)

| Baseline | What it tests | Comparison dimension |
|---|---|---|
| **Zep / Graphiti** | Temporal KG without CLS bicameral separation | Episodic KG quality; temporal reasoning |
| **Mem0 (Apr 2026)** | Vector+KG hybrid without explicit episodic/semantic separation | Multi-signal retrieval effectiveness |
| **Letta / MemGPT** | Agentic self-editing memory without KG | Agentic memory management |
| **AOI (if reproducible)** | Existing CLS-inspired 3-tier with one-directional consolidation | Iterative vs. one-directional querying |
| **Proposed - iterative querying** | Ours with one-directional consolidation only | Value of bidirectional iteration |
| **Proposed - KG-only** | Ours with episodic KG but no semantic ML layer | Value of learned semantic patterns |
| **Proposed - vector-only** | Ours with vector episodic (no KG structure) | Value of structured graph representation |

### 11.2 Evaluation Metrics

| Metric | Measurement | Target | Critical? |
|---|---|---|---|
| Fact recall precision | Correct facts returned / total facts returned | > 0.85 | Yes |
| Fact recall recall | Correct facts returned / total relevant facts | > 0.85 | Yes |
| Generalization accuracy | Correct on unseen disease × crop combinations | > 0.75 | Yes |
| Counterfactual accuracy | Correct on "what if" scenario queries | > 0.70 | Hypothesis test |
| Temporal ordering accuracy | Correct sequence/duration ordering | > 0.90 | Yes |
| Semantic Density Robustness | Recall at ρ = 0.9 | < 20% drop from ρ = 0.1 | Novel metric |
| End-to-end latency | p99 response time | < 5s | Deployment requirement |
| Storage footprint | MB at 50K facts | < 500 MB | Deployment requirement |

### 11.3 Agricultural LongMemEval Benchmark (Required)

The standard LongMemEval must be adapted for agriculture. Suggested conversion:

| Original LongMemEval category | Agricultural adaptation |
|---|---|
| Personal facts (birthdays, preferences) | Crop cycle facts (planting dates, variety names, field IDs) |
| Event sequences (travel itinerary) | Disease progression timeline (symptoms → diagnosis → treatment → outcome) |
| Temporal ordering (what happened first?) | Treatment sequence ordering ("which treatment was applied first to field A-12?") |
| Update tracking (changed preferences) | Treatment adjustments ("the fungicide was changed from sulfur to potassium bicarbonate on May 20") |
| Cross-episode reasoning (schedule conflicts) | Cross-field disease spread ("did the rust in field B-17 spread from field B-14?") |

Generate 50+ turn conversations covering 2-3 full crop cycles with interleaved disease events.

---

## 12. Suggested Ablations

Each ablation is a single `ModelConfig` field change tied to a specific hypothesis.

| # | Ablation Name | Config Field | Baseline Value | Ablated Value | Hypothesis Tested | Expected Metric Movement | Failure Interpretation | Owning Stage |
|---|---|---|---|---|---|---|---|---|
| 1 | **No iterative querying** | `AgentControllerConfig.max_iterative_cycles` | 5 | 1 | Iterative bidirectional querying improves diagnostic accuracy over one-directional consolidation | Accuracy drops 5-10% on counterfactuals; latency drops 60% | Hypothesis falsified → iterative querying adds complexity without benefit | `ml-research` |
| 2 | **No semantic ML layer** | N/A (remove subsystem) | Full architecture | Episodic KG only | Semantic ML layer provides generalization that KG alone cannot | Generalization accuracy drops >15%; fact recall unchanged | Hypothesis falsified → KG-only is sufficient; semantic layer is unnecessary overhead | `ml-architect` |
| 3 | **Generic KG (no typed objects)** | `EpisodicKGConfig.enable_crop_cycle_objects` | True | False | Domain-specific typed objects improve temporal/spatial reasoning | Temporal accuracy drops 5-10%; spatial accuracy drops 5-10% | Hypothesis falsified → generic KG is sufficient; typed objects are over-engineering | `ml-architect` |
| 4 | **LLM reconciliation → confidence_max** | `AgentControllerConfig.reconciliation_method` | "llm_judge" | "confidence_max" | LLM-based reconciliation improves over simple confidence comparison | Counterfactual accuracy drops 5-10%; latency drops 40% | Hypothesis falsified → simple confidence comparison is sufficient; LLM overhead not justified | `ml-architect` |
| 5 | **Parallel → sequential initial query** | `AgentControllerConfig.parallel_initial_query` | True | False | Parallel initial query reduces latency | Latency increases ~1 round-trip; accuracy unchanged | Trivial → keep parallel | `ml-coder` |
| 6 | **Consolidation frequency sweep** | `SemanticMLConfig.consolidation_frequency_minutes` | 1440 (daily) | {60, 360, 4320} | Daily consolidation is optimal for agricultural disease dynamics | Accuracy vs. staleness tradeoff curve | If 60-min matches daily accuracy → consolidate more often; if 4320-min matches daily → reduce consolidation overhead | `ml-validator` |
| 7 | **Prototype slot count sweep** | `SemanticMLConfig.n_pattern_slots` | 64 | {8, 16, 32, 128} | 64 slots covers agricultural disease patterns without over/under-fitting | Prototype utilization entropy changes | If 16 slots achieve same accuracy → reduce; if 128 needed → increase | `ml-validator` |
| 8 | **GCN → MLP encoder** | `SemanticMLConfig.encoder_type` | "gcn" | "mlp" (no graph structure) | GCN's graph structure awareness improves semantic pattern extraction | Pattern accuracy drops 10-15% on structurally complex queries | Hypothesis falsified → graph structure doesn't help semantic patterns; simpler MLP suffices | `ml-architect` |

### Ablation Ordering

Recommended implementation order:

1. **Ablation 1** (no iterative querying) — First and most critical: validates the primary novelty claim
2. **Ablation 2** (no semantic ML) — Second: validates whether the semantic layer is needed
3. **Ablation 5** (parallel → sequential) — Third: easy implementation, confirms latency optimization
4. **Ablation 4** (confidence_max reconciliation) — Fourth: validates LLM overhead vs. benefit
5. **Ablation 3** (generic KG) — Fifth: validates domain-specific objects
6. **Ablations 6-8** — Exploratory hyperparameter sweeps

---

## Appendix: Interface Contract Between Subsystems

```python
# ── Episodic KG → Agent Controller Interface ──

@dataclass
class KGSubgraph:
    """Standardized subgraph returned by KG queries."""
    nodes: list[KGNode]
    edges: list[KGEdge]
    root_node_id: str
    query_type: str               # "temporal_path" | "spatial_proximity" | "subgraph"
    confidence: float              # 0.0-1.0 (completeness of returned subgraph)
    timestamp: datetime
    summary: str                  # Human-readable summary for LLM consumption
    metadata: dict


# ── Semantic ML → Agent Controller Interface ──

@dataclass
class SemanticInferenceResult:
    """Standardized result from semantic ML layer inference."""
    pattern_embed: torch.Tensor   # Compressed pattern embedding
    matched_prototype_idx: Optional[int]
    confidence: float              # 0.0-1.0
    provenance: str                # "semantic_prototype" | "gcn_encoder" | "few_shot_adapted"
    prototype_weights: Optional[torch.Tensor]


# ── Agent Controller → Response ──

@dataclass
class DiagnosticResponse:
    """Final response from the CLS memory system."""
    answer: str                    # Natural language diagnosis
    provenance: list[dict]         # Per-claim source tags
    num_iterations: int            # Cycles used in iterative protocol
    confidence: float              # Overall system confidence
    evidence: list[dict]           # Supporting evidence with sources


@dataclass
class ReconciliationResult:
    """Structured output of the episodic↔semantic reconciliation step."""
    consistency_score: float
    gaps: list[str]
    contradictions: list[str]
    semantic_prior_needed: bool
    episodic_refinement_needed: bool
    refined_query_suggestion: Optional[str]
    confidence: float
```

---

*End of Architecture Blueprint. Next stage: `ml-coder` for implementation, `ml-validator` for verification.*
