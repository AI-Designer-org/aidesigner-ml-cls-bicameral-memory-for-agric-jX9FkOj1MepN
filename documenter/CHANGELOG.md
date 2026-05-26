# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-05-26

### Added

**Core architecture — CLS Bicameral Memory System**
- `EpisodicKnowledgeGraph` — Fast-learning episodic memory backed by a spatio-temporal knowledge graph with temporal indexing (1-hour bins), spatial proximity indexing (100m grid), and time-windowed dedup cache (5-minute window). Supports fast-write ingestion, temporal path BFS queries, spatial radius queries, and consolidation batch extraction.
- `SemanticPatternExtractor` — Slow-learning semantic memory (neocortex analogue): GCN encoder (+ mean pooling) → prototype cross-attention (64 learned slots) → confidence head. Trained via contrastive consolidation (NT-Xent + hinge margin).
- `GCNLayer` — Edge-feature-aware graph convolution with learned heterophily gating (sigmoid-gated message passing) to mitigate over-smoothing.
- `PrototypeAttention` — Multihead cross-attention over learned prototype pattern slots with float32 numerical safety promotions.
- `CLSAgentController` — LLM-based orchestration layer implementing iterative bidirectional querying protocol: initial parallel query → reconciliation (llm_judge / weighted_vote / confidence_max) → episodic refinement → semantic refinement → loop until convergence (max 5 cycles).
- `CLSMemorySystem` — Top-level module tying all three subsystems together with 4 configurable ablation modes (`full`, `kg_only`, `no_iterate`, `generic_kg`).

**Domain data model**
- First-class episodic objects: `CropCycle`, `DiseaseEvent`, `TreatmentAction` with typed enums (`CropStage`, `DiseaseStatus`, `TreatmentType`, `EdgeRelation`).
- Graph data structures: `KGNode`, `KGEdge`, `KGSubgraph`, `SubgraphBatch` (padded batch with pin_memory/device transfer).
- Query primitives: `TemporalPathQuery`, `SpatialProximityQuery`.
- Diagnostic types: `DiagnosticContext`, `SemanticInferenceResult`, `ReconciliationResult`, `DiagnosticResponse` (with provenance tracking).

**Configuration system**
- `EpisodicKGConfig`, `SemanticMLConfig`, `AgentControllerConfig`, `CLSMemorySystemConfig` — frozen dataclasses with Literal type constraints and cross-field validation for embedding dimension consistency.

**Agent infrastructure**
- `WorkingMemory` — Short-term session buffer with LRU/token_count/semantic_similarity eviction policies.
- `ConsolidationScheduler` — Manages offline consolidation lifecycle with scheduled (daily), threshold-based (256+ subgraphs), and on-demand triggers.
- `LLMInterface` — Lightweight LLM wrapper with stub mode returning deterministic JSON for reconciliation.

**Test suite (38 tests across 7 test classes)**
- `TestShapes` (10 tests) — Tensor shape correctness for every forward pass.
- `TestGradients` (4 tests) — Gradient flow verification for all learnable components.
- `TestCorrectness` (8 tests) — KG query correctness, dedup behavior, spatial proximity, temporal ordering.
- `TestNumerics` (6 tests) — float32, float16, bfloat16, bf16 mixed-precision, NaN/Inf guards.
- `TestPermutationInvariance` (1 test) — GCN permutation invariance property.
- `TestAblationModes` (5 tests) — All 4 ablation modes + counterfactual diagnosis tests.
- `TestInterfaceContracts` (4 tests) — Abstract base class contract enforcement.

**Benchmark infrastructure (9 benchmarks across 3 domains)**
- LM / Memory Systems: fact recall (precision/recall/F1), temporal reasoning (ordering accuracy), semantic density robustness (ρ sweep 0.1–0.9), write throughput (events/s).
- Graph ML: expressiveness probe (1-WL hard pairs), oversmoothing check (collapse ratio by depth).
- Agricultural / Scientific ML: crop cycle memory (multi-turn accuracy), disease progression tracking (status updates), spatial proximity retrieval (radius correctness).

**Ablation framework (12 configurations)**
- 12 ablation configs: baseline, no_iterate, kg_only, generic_kg, confidence_max, sequential_query, freq_60min, freq_360min, freq_4320min, slots_8, slots_32, slots_128.
- CLI runner with `--all`, `--sweep`, `--ablation` flags. Results saved to `research_eval/ablation_results.json`.

**Profiling infrastructure (4 profilers)**
- Semantic ML: FLOPs estimate, latency, peak CUDA memory, optional `torch.profiler` trace.
- Episodic KG: write throughput, query latency, memory estimate.
- End-to-end: avg/P99/min/max diagnosis latency.
- Parameter scaling across 4 config sizes (tiny/small/medium/large).

**Research-quality evaluation**
- `scorecard.json` — 6-dimension scoring (novelty 3/5, experimental comprehensiveness 4/5, theoretical foundation 4/5, result analysis 2/5, implementation reproducibility 4/5, writing readiness 3/5) with 5 blocking gaps and 7 recommended next experiments.
- `claim_grounding.md` — Every claim mapped to code location or flagged `TODO: unverified`.
- `experiment_coverage.md` — Evaluation requirements vs. implementation status.
- `claim_grounding_rubric.md` — Scoring rubric with blocking gap definitions.

**Documentation**
- `README.md` — Project overview, highlights, quick start, repository layout, citation.
- `docs/ARCHITECTURE.md` — Design rationale, ASCII diagram, equations, shape evolution table, 13 design decisions, domain-specific considerations, 10 known limitations.
- `docs/TRAINING.md` — Environment setup, hyperparameters (3 tables), consolidation lifecycle, training recipe, 8-row troubleshooting table.
- `docs/BENCHMARKS.md` — 9 benchmark tables, 12-row ablation study with hypothesis tests, 4 profiling tables, 6-dimension research-quality scorecard, blocking gaps, 3 P0 experiments.
- `docs/API.md` — Complete API reference covering 8 public modules with constructor signatures, method signatures, and shape contracts.
- `CHANGELOG.md` — This file.

### Fixed
- **HIGH** `GCNLayer.forward` — LayerNorm mixed-dtype error in bf16 mode: replaced `self.norm(out.float())` with explicit `F.layer_norm(out.float(), ..., weight.float(), bias.float(), ...)` to prevent dtype mismatch.
- **HIGH** `SemanticPatternExtractor` — `batch.node_mask.float()` forced float32 promotion in bf16 context: changed to `.to(dtype=h_out.dtype)` for dtype consistency.
- **MEDIUM** `ContrastiveLoss.forward` — Empty tensor `.mean()` produced NaN when all labels were the same class: added `if neg_sim.numel() > 0:` guard.
- **MEDIUM** `PrototypeAttention.forward` — float32 input cast + bf16 attention weights caused `mat1/mat2 dtype mismatch`: added temporary float32 promotion for attention module, restored after forward.
- **MEDIUM** `SemanticPatternExtractor` — confidence_head float32 cast + bf16 weights: same float32 promotion pattern applied.

---

## [Unreleased]

### Planned
- [ ] **Run full ablation suite** (`python validator/ablation_runner.py --all`) to populate accuracy × latency Pareto front.
- [ ] **Integrate external baselines** (Zep, Mem0, Letta, AOI) for monolithic comparison — required to test falsification condition.
- [ ] **Run density robustness at scale** (N=50+) with seed-fixed data to test Stability Gap hypothesis in agriculture.
- [ ] **Implement counterfactual benchmark** for testing semantic-episodic conflict resolution.
- [ ] **Storage scaling benchmark** at 10K–100K events for KG memory profiling.
- [ ] **Implement `encoder_type="mlp"`** in `SemanticPatternExtractor` to complete ablation 8 (gcn_mlp comparison).
- [ ] **Seed-version random data generation** for reproducible benchmark results across runs.
- [ ] **Paper draft** with publication-grade tables from benchmark results.
