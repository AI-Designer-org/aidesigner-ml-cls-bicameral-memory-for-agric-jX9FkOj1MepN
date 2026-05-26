# Architecture

## 1. Motivation

### The Stability Gap Problem

Monolithic neural memory — whether a single vector store, a single knowledge graph, or an end-to-end neural network — is geometrically doomed under high semantic density. The "Mind the Gap" paper (arXiv:2601.15313, Jan 2026) formally proves that a monolithic neural memory collapses within **N=5 semantically related facts** when the semantic density ρ exceeds 0.6. Agricultural disease data operates in precisely this regime: disease → symptom → treatment → season → field are tightly correlated (ρ likely > 0.6 by any reasonable measure).

This is not a tuning or scaling issue — it is a geometric necessity. The proof shows that as density increases, the representation space becomes saturated, and new facts overwrite or conflate with existing ones. The required architectural response is **bicameral separation**: the brain's Complementary Learning Systems (CLS) theory, which separates fast-learning episodic memory (hippocampus) from slow-learning semantic memory (neocortex).

### What Prior CLS Systems Do — and Don't Do

Multiple production-adjacent systems already implement CLS-inspired memory:

- **AOI Three-Layer Memory** (arXiv:2512.13956, Dec 2025): Working → Episodic → Semantic hierarchy with LLM-based compression at layer boundaries (72.4% compression, 92.8% critical info preservation). Consolidates one-directionally: episodic → semantic.
- **All-Mem** (arXiv:2603.19595, Mar 2026): Online fast-path writes + offline consolidation with SPLIT/MERGE/UPDATE topology edits. Original evidence always preserved. Consolidates one-directionally.
- **Dual-System Memory for LLMs** (Feb 2026): MEMIT for fast hippocampal encoding + LoRA for slow neocortical consolidation during simulated "sleep."
- **Zep / Graphiti** (arXiv:2501.13956, Jan 2025): Temporal knowledge graph engine for agent memory, achieving 94.8% on DMR benchmark. Not framed as CLS; no episodic/semantic separation.

All of these consolidate **one-directionally** (episodic → semantic). None implement **iterative bidirectional querying** where the semantic layer's priors refine the episodic KG search, and the episodic findings update the semantic beliefs, in a loop until convergence.

### The Agricultural Gap

Agricultural neuro-symbolic systems exist (OpenAg, arXiv:2506.04571; NeuroCausal-FusionNet, EPJ Web Conf. 328, 2025; AgriSensNet, ICAISDA-25, Mar 2026) but none frame their KG as a CLS episodic analogue with explicit fast-write/slow-consolidation separation, and none evaluate the Stability Gap hypothesis in the agricultural domain.

### Core Hypothesis

> An autonomous agricultural diagnostic agent with a CLS-inspired bicameral memory architecture — comprising a spatio-temporal knowledge graph (episodic hippocampus analogue) and a compressed ML pattern layer (semantic neocortex analogue) with iterative bidirectional querying — will achieve higher diagnostic accuracy on long-horizon agricultural tasks (crop cycle tracking, disease progression prediction) than equivalent monolithic memory models, because the Stability Gap (arXiv:2601.15313) renders monolithic storage provably insufficient under the semantic density of agricultural domain knowledge.

---

## 2. At a glance

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
```

| Property | Value |
|---|---|
| Parameter count (default config) | 1,369,920 (SemanticPatternExtractor) |
| Time complexity (KG write) | O(log N) amortized via temporal + spatial index |
| Time complexity (KG query) | O(log N + K) temporal range; O(K) spatial radius |
| Time complexity (semantic forward) | O(B·N·H² + B·S·P) for batch B, nodes N, hidden H, slots S, pattern P |
| Space complexity | O(N + E) for KG; O(S·P) for semantic prototypes |
| Hardware requirements | CPU-inference capable (semantic ML: ~2.8M FLOPs forward); GPU optional for training |

---

## 3. The core component: Iterative Bidirectional Querying Protocol

### 3.1 Intuition

The agent controller works like a diagnostician consulting two specialists: a field notebook (episodic KG) that records every specific observation with timestamps and locations, and a medical textbook (semantic ML) that encodes generalized knowledge about disease patterns. Rather than just taking notes and periodically updating the textbook (one-directional consolidation), the controller actively mediates a conversation between them:

1. Both specialists are consulted in parallel with the same query.
2. The field notebook returns specific facts: "Field A-42 had powdery mildew on May 15, treated with sulfur on May 18."
3. The textbook returns a generalized pattern: "Powdery mildew in wheat typically follows 7-14 days of humidity > 80%; sulfur is effective within 72h."
4. The controller compares them. If the textbook suggests checking humidity, the controller asks the field notebook: "Was humidity > 80% in the week before May 8?"
5. If the field notebook shows an unusual progression, the controller asks the textbook: "Can you adapt your pattern for this anomaly?"
6. This loop continues until the controller is confident enough to produce a diagnosis, or a maximum iteration limit is reached.

This bidirectional iteration — where semantic priors refine episodic search and episodic anomalies update semantic beliefs — is the primary architectural novelty.

### 3.2 Equations

**Episodic KG temporal index query:**

$$\text{results} = \{n \in N \mid t_{\text{from}} \leq t_n \leq t_{\text{to}}\}$$

where $N$ is the set of KG nodes, $t_n$ is the quantized timestamp of node $n$, and bins are computed as $\text{bin}(t) = \lfloor t / \tau \rfloor$ with $\tau = 3600$s resolution.

**Semantic pattern extraction (GCN encoder):**

$$h^{(l+1)}_i = \sigma\left( W_{\text{self}} \, h^{(l)}_i + \sum_{j \in \mathcal{N}(i)} g_{ij} \cdot W_{\text{msg}} \, [h^{(l)}_j \,\|\, e_{ij}] \right)$$

where $g_{ij} = \text{Sigmoid}(W_{\text{gate}} \, [h^{(l)}_j \,\|\, e_{ij}])$ is a learned heterophily gate controlling how much neighbor $j$ influences node $i$, and $e_{ij}$ are edge features.

**Graph-level pooling:**

$$h_{\text{graph}} = \frac{1}{|\mathcal{V}_{\text{valid}}|} \sum_{i \in \mathcal{V}_{\text{valid}}} h^{(L)}_i$$

**Prototype attention:**

$$\mathbf{a} = \text{Softmax}\left( \frac{Q K^\top}{\sqrt{d_k}} \right), \quad \mathbf{p}_{\text{attended}} = \mathbf{a} V$$

where $Q$ is the projected graph embedding, $K=V$ are the $S=64$ learned prototype vectors, and $\mathbf{p}_{\text{attended}} \in \mathbb{R}^{P}$ is the compressed pattern embedding.

**Confidence estimation:**

$$c = \text{Sigmoid}\left( W_c \, \mathbf{p}_{\text{attended}} + b_c \right), \quad c \in [0, 1]$$

**Contrastive consolidation loss:**

$$\mathcal{L}_{\text{contrast}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\sum_{j \in \mathcal{P}(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_j / \tau)}{\sum_{k \neq i} \exp(\mathbf{z}_i \cdot \mathbf{z}_k / \tau)}$$

where $\mathcal{P}(i)$ are positive pairs (same disease class), $\tau = 0.1$ is the temperature, and $\mathbf{z} = \text{L2Normalize}(\mathbf{p}_{\text{attended}})$.

**Reconciliation confidence (after $k$ iterations):**

$$c_{\text{final}}^{(k)} = \begin{cases} 
c_{\text{LLM}}(\text{episodic\_facts}, \text{semantic\_pattern}) & \text{if method = llm\_judge} \\
\max(c_{\text{ep}}, c_{\text{sem}}) & \text{if method = confidence\_max} \\
0.5(c_{\text{ep}} + c_{\text{sem}}) & \text{if method = weighted\_vote}
\end{cases}$$

Early exit when $c_{\text{final}}^{(k)} \geq \theta_{\text{exit}} = 0.9$ or $k \geq K_{\max} = 5$.

### 3.3 Reference implementation walk-through

Below is the core reconciliation loop from `controller.py` (simplified). The full implementation is at `coder/controller.py`, method `CLSAgentController.diagnose()` lines 351-499.

```python
# ── Steps 3-6: Iterative reconciliation loop ──
while iteration < self.config.max_iterative_cycles:
    iteration += 1

    # Store in working memory
    self.working_memory.add(f"kg_results_iter_{iteration}", kg_results)
    self.working_memory.add(f"semantic_results_iter_{iteration}", semantic_results)

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

    # Semantic → Episodic refinement (semantic prior guides KG search)
    if (reconciliation.semantic_prior_needed
            and self.config.enable_semantic_prior_routing):
        refined_kg = self.semantic_memory.query_episodic_via_semantic_prior(
            semantic_results, self.episodic_kg, k=5)
        if refined_kg:
            kg_results = refined_kg

    # Episodic → Semantic refinement (anomalous facts update semantic beliefs)
    if (reconciliation.episodic_refinement_needed
            and self.config.enable_episodic_revision):
        refined_semantic = self.semantic_memory.few_shot_adapt(
            support_subgraphs=kg_results[:5],
            query=self._build_query_subgraph(query, context),
        )
        if refined_semantic["confidence"] > semantic_results.confidence:
            semantic_results = SemanticInferenceResult(
                pattern_embed=refined_semantic["pattern_embed"],
                confidence=refined_semantic["confidence"],
                provenance="few_shot_adapted",
            )
```

Key design points:
- **Early exit** prevents unbounded latency — most queries converge in 2-3 cycles
- **Two refinement directions** are independent flags (`enable_semantic_prior_routing`, `enable_episodic_revision`) for clean ablation
- **Working memory** stores iteration history for LLM context in final response generation

---

## 4. Tensor shape evolution

Shape conventions for a forward pass through the SemanticPatternExtractor with a batch of 4 subgraphs, each with up to 10 nodes and 8 edges.

| Stage | Shape | Notes |
|---|---|---|
| Input node_features | (4, 10, 64) | (B, max_nodes, node_embed_dim), float32 |
| Input edge_index | (4, 2, 8) | (B, 2, max_edges), int64 |
| Input edge_features | (4, 8, 16) | (B, max_edges, edge_embed_dim), float32 |
| Input node_mask | (4, 10) | bool, True = valid node |
| Input edge_mask | (4, 8) | bool, True = valid edge |
| After node_embed (Linear) | (4, 10, 256) | (B, N, hidden_dim) |
| After edge_embed (Linear) | (4, 8, 256) | (B, E, hidden_dim) |
| After GCN encoder (3 layers) | (4, 10, 256) | (B, N, hidden_dim) — per-node |
| After mean pooling | (4, 256) | (B, hidden_dim) — graph-level |
| After pattern projector | (4, 128) | (B, pattern_embed_dim) — query for attention |
| After prototype attention | (4, 128) | (B, pattern_embed_dim) — attended pattern |
| After confidence head | (4,) | (B,) — per-sample confidence in [0, 1] |

---

## 5. Design decisions

### Episodic KG

| Decision | Alternative considered | Why we chose this | Trade-off accepted |
|---|---|---|---|
| Spatio-temporal KG over vector store | Pure vector embedding | Agriculture requires explicit temporal ordering (crop cycles) and spatial relationships (disease spread) that vector stores flatten | Higher storage cost; requires dual indexing |
| Append-only admission (no surprise filtering) | Titans' surprise-based admission | "Boring" observations (e.g., "no disease found") are diagnostically useful negative evidence | Higher KG storage; may include redundant data |
| Temporal + spatial dual indexing | Single sequential scan | Disease queries split evenly between "when?" and "where?" — separate indices prevent sequential scans | Index maintenance overhead |
| First-class domain objects (CropCycle, DiseaseEvent, TreatmentAction) | Generic property graph | Typed fields enable domain-specific query optimizations (e.g., "find flowering-stage fields within 500m of rust") | Less flexible for unforeseen query types |
| Dedup window (5 min) | No dedup | Agricultural sensors produce near-duplicate observations within minutes | May miss genuine rapid changes |

### Semantic ML Layer

| Decision | Alternative considered | Why we chose this | Trade-off accepted |
|---|---|---|---|
| GCN encoder | Transformer | KG subgraphs are small (10-50 nodes), irregularly structured — GCNs match naturally | 1-WL expressiveness ceiling |
| Prototype attention (64 slots) | Flat embedding | Agricultural disease patterns cluster naturally; 64 slots ~20 crops × ~3 diseases | May not cover rare disease × crop combinations |
| Heterophily gate in GCN | Standard GCN message passing | Healthy vs. infected fields are spatial neighbors but must remain distinct | Extra parameters; learned gate may not always activate correctly |
| Contrastive consolidation loss | Cross-entropy with labeled data | Limited-label regime; contrastive loss works without exhaustive labels | Requires careful negative sampling |
| Few-shot adaptation via prototypical networks | Fine-tuning | Prototypical networks work in K-shot without per-task fine-tuning | Less expressive than fine-tuning for very novel patterns |

### Agent Controller

| Decision | Alternative considered | Why we chose this | Trade-off accepted |
|---|---|---|---|
| LLM-based orchestration | Hardcoded rules | Agricultural queries are diverse and nuanced; LLM provides flexibility | Latency cost per reconciliation step |
| Max 5 iterative cycles | Unbounded iteration | Agricultural diagnostics need real-time response (< 5s); 5 cycles at ~1s each fits budget | May exit before full convergence on complex queries |
| LLM-based reconciliation ("llm_judge") | Weighted vote / confidence max | Reconciling episodic specifics with semantic generalizations requires nuanced comparison | Higher latency; requires LLM API access |
| Provenance tracking per claim | No provenance | In decision support, user must know source of each fact to assess trust | Storage and prompt token overhead |

---

## 6. Domain-specific considerations

### 6.1 LM / Memory Systems

- **Position / order scheme:** The Episodic KG uses **explicit temporal indexing** (sorted by timestamp) rather than positional embeddings. Every edge carries a timestamp attribute, so temporal ordering is a first-class property.
- **Causal contract:** The temporal index provides strict causality — queries can be constrained to `from_date`/`to_date` ranges, and temporal path queries respect edge direction + timestamp ordering. No information leakage from future to past.
- **Fast/slow separation:** The Stability Gap (arXiv:2601.15313) mandates that fast (episodic) and slow (semantic) memory use **different storage representations** — a graph store and a neural network, respectively. Simply separating into two vector databases would not avoid the Stability Gap because both would use the same underlying representation.

### 6.2 Graph ML

- **Expressiveness ceiling:** The episodic KG is a **property graph** with typed nodes and edges, which is equivalent to 1-WL (Weisfeiler-Lehman) expressiveness. This is sufficient for agricultural diagnostics because disease spread is fundamentally local (neighbor fields, shared equipment), and temporal sequences are linear paths. The semantic ML layer provides additional expressiveness beyond 1-WL via prototype attention.
- **Permutation invariance:** Node-level queries (e.g., "what happened at field A-42?") are permutation-invariant by construction. Graph-level queries are invariant because the KG stores all edges explicitly and the semantic layer uses mean pooling.
- **Positional encoding:** Not needed — the KG has explicit spatial coordinates and timestamps.
- **Scalability:** Full-graph traversal is avoided. All queries use indexed access (temporal index, spatial index, node ID lookup). Maximum subgraph extraction radius is bounded (max_depth=5, radius=500m).
- **Heterophily:** Agricultural disease KGs are inherently heterophilic — healthy and diseased fields may be neighbors. The GCN layer uses a learned gating mechanism to prevent over-smoothing across class boundaries.

### 6.3 Scientific ML (Agriculture)

- **Physics constraints:** Not enforced as hard constraints — agricultural disease progression is stochastic enough that hard PDE constraints would be misleading. Soft constraints could be added via a disease progression model in the semantic ML layer.
- **Function vs. operator learning:** The semantic ML layer is a **function** (maps episodic subgraph → pattern embedding), not a neural operator. This is appropriate for classification/retrieval, not PDE families.
- **Symmetry/equivariance:** Not required — agricultural fields have fixed geographic coordinates. No rotational or translational symmetry to exploit.
- **Mesh type:** The KG is a **graph on arbitrary point cloud** (fields are discrete locations). No regular grid or mesh.

### 6.4 GenAI / LLM (Agent Controller)

- **Conditioning interface:** The agent controller LLM is conditioned on working memory content via standard prompt-based conditioning. No cross-attention or AdaLN is needed.
- **Tool use:** The LLM does not generate KG queries directly (to avoid latency and parsing errors). Instead, the controller pre-computes KG and ML results and passes them as structured context. The LLM's role is reconciliation, not query generation.

---

## 7. Known limitations

- **External baselines not run** — The falsification condition (research contract §4) requires comparison against Zep, Mem0, Letta, and AOI on the same agricultural benchmark. None of these have been integrated. The claim of outperforming monolithic models is **TODO: unverified**.
- **Iterative querying benefit unmeasured** — The primary novelty claim (iterative bidirectional querying outperforms one-directional consolidation) has ablation infrastructure (`ablation_runner.py --ablation no_iterate`) but no results. The hypothesis could be falsified if one-directional consolidation matches full-iteration accuracy.
- **GCN → MLP encoder ablation not implemented** — Ablation 8 is registered in the ablation runner but `SemanticPatternExtractor` does not support `encoder_type="mlp"`. The claim that GCN's graph structure awareness improves pattern extraction cannot be tested.
- **LLM stub** — The agent controller uses a deterministic stub for LLM responses. Production deployment requires integration with OpenAI, Anthropic, or equivalent API, which will introduce non-determinism and latency variance.
- **Simple text embedding** — The semantic layer's text-to-prototype matching uses bag-of-ngrams hashing, not a proper text encoder (e.g., Sentence-BERT). This is sufficient for prototype matching but inadequate for nuanced natural language queries.
- **No convergence guarantee** — The iterative protocol is bounded by `max_iterative_cycles` and an early-exit confidence threshold, but there is no formal proof of monotonic improvement or convergence rate.
- **Counterfactual reasoning benchmark missing** — The research contract requires counterfactual "what-if" accuracy evaluation. The current benchmark suite covers fact recall and temporal reasoning but not counterfactuals.
- **Storage footprint at scale unmeasured** — Target is < 500 MB at 50K facts (architecture blueprint §2). Current profiling is at small scale only.
- **Not evaluated on real agricultural data** — All benchmarks use synthetic data. Real-world agricultural ontologies (AgroPortal, Crop Ontology) have not been integrated.
