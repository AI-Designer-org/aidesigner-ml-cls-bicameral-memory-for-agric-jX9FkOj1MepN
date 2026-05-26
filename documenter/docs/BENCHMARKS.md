# Benchmarks

All numbers are reproducible with the commands shown. Numbers marked `TODO` have not been measured — do not cite them.

> **Current status:** Benchmark infrastructure is complete (9 benchmarks, 12 ablation configs, profiling suite). Experimental results are **TODO: unverified** — the ablation runner and benchmarks must be executed to produce numbers.

---

## LM / Memory Systems benchmarks

### Fact recall

| Metric | Value | Command | Notes |
|---|---|---|---|
| Precision | TODO | `python benchmarks.py --benchmark fact_recall` | 30 events, 10 queries |
| Recall | TODO | same | |
| F1 | TODO | same | |
| Avg latency | TODO | same | |
| P99 latency | TODO | same | |

Measure: precision, recall, and F1 for fact retrieval across conversation turns with 30 synthetic agricultural events (crop cycles, disease events, treatments) across 3 fields.

Reproduce: `python validator/benchmarks.py --benchmark fact_recall`

### Temporal reasoning

| Metric | Value | Command | Notes |
|---|---|---|---|
| Temporal ordering accuracy | TODO | `python benchmarks.py --benchmark temporal_reasoning` | 20 sequences |

Measure: accuracy of temporal ordering queries (e.g., "which treatment came first?") across 20 event sequences with known timestamps.

Reproduce: `python validator/benchmarks.py --benchmark temporal_reasoning`

### Semantic Density Robustness

| ρ | Accuracy | Command | Notes |
|---|---|---|---|
| 0.1 | TODO | `python benchmarks.py --benchmark density_robustness` | 20 events |
| 0.3 | TODO | same | |
| 0.5 | TODO | same | |
| 0.7 | TODO | same | |
| 0.9 | TODO | same | |
| Robustness drop | TODO | same | acc@0.1 − acc@0.9 |

Measure: accuracy degradation as semantic density ρ sweeps from 0.1 to 0.9 at constant fact count (N=20). This is the novel metric from the research contract that directly tests the Stability Gap hypothesis.

> **TODO: unverified** — this is a critical experiment. The Stability Gap theorem (arXiv:2601.15313) predicts collapse at ρ > 0.6 with N ≥ 5. The robustness drop should be < 20% if the bicameral architecture mitigates the gap.

Reproduce: `python validator/benchmarks.py --benchmark density_robustness`

### Write throughput

| Metric | Value | Command | Notes |
|---|---|---|---|
| Events/s | TODO | `python benchmarks.py --benchmark write_throughput` | 100 disease events |
| Total time | TODO | same | |
| Nodes after | TODO | same | |
| Edges after | TODO | same | |

Measure: KG write throughput for bulk ingestion of 100 synthetic disease events.

Reproduce: `python validator/benchmarks.py --benchmark write_throughput`

---

## Graph ML benchmarks

### Expressiveness probe

| Metric | Value | Command | Notes |
|---|---|---|---|
| Can distinguish non-isomorphic graphs | TODO | `python benchmarks.py --benchmark expressiveness` | Cosine similarity between 1-WL hard pairs |
| Cosine similarity | TODO | same | |

Measure: whether the semantic layer can distinguish two non-isomorphic 3-regular graphs on 6 nodes that 1-WL cannot distinguish. The GCN encoder is 1-WL by default; additional expressiveness comes from prototype attention.

> Expected: GCN alone cannot distinguish these graphs (cosine similarity ~1.0). The prototype attention layer may provide marginal separability.

Reproduce: `python validator/benchmarks.py --benchmark expressiveness`

### Oversmoothing check

| Depth | Pattern std | Confidence | Command |
|---|---|---|---|
| 1 | TODO | TODO | `python benchmarks.py --benchmark oversmoothing` |
| 2 | TODO | TODO | same |
| 3 | TODO | TODO | same |
| 4 | TODO | TODO | same |
| 5 | TODO | TODO | same |
| 6 | TODO | TODO | same |
| Collapse ratio | TODO | TODO | std@6 / std@1 |

Measure: whether node features collapse as GCN depth increases. A collapse ratio < 0.1 indicates severe over-smoothing. The heterophily gate in `GCNLayer` is designed to mitigate this.

Reproduce: `python validator/benchmarks.py --benchmark oversmoothing`

---

## Agricultural (Scientific ML) benchmarks

### Crop cycle memory

| Metric | Value | Command | Notes |
|---|---|---|---|
| Crop cycle accuracy | TODO | `python benchmarks.py --benchmark crop_cycle_memory` | 5 fields, 10 queries |

Measure: multi-turn diagnostic accuracy across 5 distinct crop cycles with different crops (wheat, corn, rice, soybean, tomato) and planting dates.

Reproduce: `python validator/benchmarks.py --benchmark crop_cycle_memory`

### Disease progression tracking

| Metric | Value | Command | Notes |
|---|---|---|---|
| Disease nodes | TODO | `python benchmarks.py --benchmark disease_progression` | After status updates |
| Total writes | TODO | same | |
| Total nodes | TODO | same | |

Measure: ability to track disease progression (suspected → confirmed → contained) through event updates, verifying dedup and merge behavior.

Reproduce: `python validator/benchmarks.py --benchmark disease_progression`

### Spatial proximity retrieval

| Radius | Returned | Expected | Command |
|---|---|---|---|
| 100m | TODO | TODO | `python benchmarks.py --benchmark spatial_proximity` |
| 300m | TODO | TODO | same |
| 600m | TODO | TODO | same |

Measure: spatial proximity query correctness across 8 disease events at known distances (0-700m) from a center field.

Reproduce: `python validator/benchmarks.py --benchmark spatial_proximity`

---

## Ablation study

| Ablation | Config delta | Accuracy | Δ vs. baseline | Latency (s) | Command |
|---|---|---|---|---|---|
| baseline | Full architecture | TODO | — | TODO | `python ablation_runner.py --ablation baseline` |
| no_iterate | `max_iterative_cycles=1` | TODO | TODO | TODO | `python ablation_runner.py --ablation no_iterate` |
| kg_only | Semantic ML disabled | TODO | TODO | TODO | `python ablation_runner.py --ablation kg_only` |
| generic_kg | Typed objects disabled | TODO | TODO | TODO | `python ablation_runner.py --ablation generic_kg` |
| confidence_max | `reconciliation_method=confidence_max` | TODO | TODO | TODO | `python ablation_runner.py --ablation confidence_max` |
| sequential_query | `parallel_initial_query=False` | TODO | TODO | TODO | `python ablation_runner.py --ablation sequential_query` |
| freq_60min | `consolidation_frequency=60` | TODO | TODO | TODO | `python ablation_runner.py --ablation freq_60min` |
| freq_360min | `consolidation_frequency=360` | TODO | TODO | TODO | `python ablation_runner.py --ablation freq_360min` |
| freq_4320min | `consolidation_frequency=4320` | TODO | TODO | TODO | `python ablation_runner.py --ablation freq_4320min` |
| slots_8 | `n_pattern_slots=8` | TODO | TODO | TODO | `python ablation_runner.py --ablation slots_8` |
| slots_32 | `n_pattern_slots=32` | TODO | TODO | TODO | `python ablation_runner.py --ablation slots_32` |
| slots_128 | `n_pattern_slots=128` | TODO | TODO | TODO | `python ablation_runner.py --ablation slots_128` |
| gcn_mlp | `encoder_type=mlp` | ⚠️ **NOT FUNCTIONAL** | — | — | `python ablation_runner.py --ablation gcn_mlp` |

> **TODO: unverified** — all ablation results are pending execution. Run `python validator/ablation_runner.py --all` to populate.

### Hypothesis tests per ablation

| Ablation | Hypothesis tested | Expected movement |
|---|---|---|
| no_iterate | Iterative bidirectional querying improves diagnostic accuracy over one-directional consolidation | Accuracy drops 5-10% on counterfactuals; latency drops 60% |
| kg_only | Semantic ML layer provides generalization that KG alone cannot | Generalization accuracy drops >15%; fact recall unchanged |
| generic_kg | Domain-specific typed objects improve temporal/spatial reasoning | Temporal accuracy drops 5-10% |
| confidence_max | LLM-based reconciliation improves over simple confidence comparison | Counterfactual accuracy drops 5-10%; latency drops 40% |
| sequential_query | Parallel initial query reduces latency | Latency increases ~1 round-trip; accuracy unchanged |
| gcn_mlp | GCN's graph structure awareness improves pattern extraction | Pattern accuracy drops 10-15% on structurally complex queries |

---

## Profiling

GPU: CPU only (tested); CUDA-capable. Default config (hidden_dim=256, n_layers=3, n_pattern_slots=64, pattern_embed_dim=128).

### SemanticPatternExtractor

| Metric | Value | Command |
|---|---|---|
| Total params | 1,369,920 | `python profiling.py --subsystem semantic` |
| Trainable params | 1,369,920 | same |
| Est. forward FLOPs | 2.8M | same |
| Avg forward time (CPU) | TODO | same |
| Peak CUDA memory | TODO | same (CUDA only) |

### EpisodicKnowledgeGraph

| Metric | Value | Command |
|---|---|---|
| Events/s write throughput | TODO | `python profiling.py --subsystem kg` |
| Avg query time | TODO | same |
| Est. memory (500 events) | TODO | same |

### Full system (end-to-end)

| Metric | Value | Command |
|---|---|---|
| Avg latency per diagnosis | TODO | `python profiling.py --subsystem end_to_end` |
| P99 latency | TODO | same |
| Semantic params | 1,369,920 | same |

### Parameter scaling

| Config | Params | Config details | Command |
|---|---|---|---|
| tiny | TODO | hidden=32, layers=1, slots=4, pattern=16 | `python profiling.py --subsystem scaling` |
| small | TODO | hidden=64, layers=2, slots=8, pattern=32 | same |
| medium | TODO | hidden=128, layers=3, slots=32, pattern=64 | same |
| large (default) | 1,369,920 | hidden=256, layers=3, slots=64, pattern=128 | same |

Reproduce:
```bash
python validator/profiling.py --subsystem semantic    # Semantic ML profile
python validator/profiling.py --subsystem kg           # KG profile
python validator/profiling.py --subsystem end_to_end   # End-to-end profile
python validator/profiling.py --subsystem scaling      # Parameter scaling
```

---

## Research-quality evaluation

| Dimension | Score (0-5) | Evidence | Gaps |
|---|---|---|---|
| **Novelty** | 3/5 | Iterative bidirectional querying protocol implemented; domain-specific spatio-temporal KG objects; CLS bicameral separation | CLS pattern itself not novel (AOI, All-Mem); iterative querying benefit unmeasured; external baselines not run |
| **Experimental comprehensiveness** | 4/5 | 9 benchmarks, 12 ablation configs, 38 tests across 7 classes, novel density robustness metric | External baselines missing; counterfactual benchmark missing; storage at scale unmeasured |
| **Theoretical foundation** | 4/5 | Stability Gap theorem motivates architecture; GCN heterophily gate theoretically grounded; prototypical networks for few-shot | No convergence guarantee for iterative protocol; 1-WL ceiling unproven to be exceeded; text embedding is heuristic |
| **Result analysis** | 2/5 | Test infrastructure validates code correctness; profiling measures latency/FLOPs | **No experimental results exist** — all metrics are infrastructure-only; no statistical significance |
| **Implementation reproducibility** | 4/5 | Seeded initialization; frozen configs; CLI-driven evaluation scripts; device-portable | LLM stub non-deterministic in production; random data not seed-versioned; `id()` in node IDs not restart-deterministic |
| **Writing readiness** | 3/5 | Shape comments throughout code; comprehensive docstrings; provenance tracking; claim grounding documented | No paper draft; benchmark results not formatted as publication tables; no README for evaluation pipeline |

### Blocking gaps

| Code | Severity | Description |
|---|---|---|
| `baseline_not_beaten` | **BLOCKING** | No external baseline comparison (Zep, Mem0, Letta, AOI). The falsification condition requires monolithic comparison — without it, "outperforms monoliths" is unverifiable. |
| `claim_not_grounded` | **BLOCKING** | Iterative bidirectional querying benefit is unmeasured. The primary novelty claim has test infrastructure but no results. |
| `benchmark_not_executable` | WARNING | Random data generation without versioning → results not reproducible across runs. |
| `ablation_missing` | WARNING | GCN → MLP encoder ablation (ablation 8) registered but `SemanticPatternExtractor` does not support `encoder_type="mlp"`. |

### Required next experiments (P0)

1. **Run full ablation suite:** `python validator/ablation_runner.py --all` → accuracy × latency Pareto for all config variants. This is the minimum bar for validating the architecture's design choices.
2. **Integrate and compare against external baselines:** Port the agricultural benchmark to Zep/Graphiti or Mem0 API for monolithic comparison. Without this, the falsification condition from the research contract cannot be tested.
3. **Run density robustness at scale:** `python validator/benchmarks.py --benchmark density_robustness` with N=50+ and seed-fixed data. This directly tests the Stability Gap hypothesis in agriculture.
