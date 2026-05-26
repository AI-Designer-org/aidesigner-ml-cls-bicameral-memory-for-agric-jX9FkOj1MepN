# Experiment Coverage — CLS Bicameral Memory for Agricultural Agents

> Maps every evaluation requirement, baseline, and ablation from the research contract
> to implemented artifacts. Identifies gaps and TODO items.

---

## 1. Baselines Required by `ml-research`

| Baseline | Status | Location | Notes |
|---|---|---|---|
| Zep / Graphiti (temporal KG) | ❌ **MISSING** | — | Requires external system integration; not implemented in this codebase |
| Mem0 (vector+KG, Apr 2026) | ❌ **MISSING** | — | Requires external system integration; not implemented in this codebase |
| Letta / MemGPT (filesystem tiered) | ❌ **MISSING** | — | Requires external system integration; not implemented in this codebase |
| AOI three-layer (if reproducible) | ❌ **MISSING** | — | Requires external system integration; not implemented in this codebase |
| Proposed - no_iterate | ✅ **IMPLEMENTED** | `validator/ablation_runner.py` — `no_iterate` ablation | Sets `max_iterative_cycles=1`, disables semantic_prior_routing and episodic_revision |
| Proposed - kg_only | ✅ **IMPLEMENTED** | `model.py` `set_ablation_mode("kg_only")` | Disables semantic ML layer at runtime |
| Proposed - vector_only | ⚠️ **PARTIAL** | — | Requires alternative episodic implementation (vector store); not yet built |

### Gap Analysis
All four external baselines are missing. This is the single largest gap — the falsification condition (research contract §4) explicitly requires monolithic comparison. Without it, the primary claim ("outperforms monolithic models") cannot be evaluated.

---

## 2. Evaluation Requirements from `ml-research`

| Requirement | Status | Implementation | Notes |
|---|---|---|---|
| LongMemEval/LOCOMO adapted for agriculture (50+ turns) | ⚠️ **PARTIAL** | `benchmarks.py` `benchmark_fact_recall` — 30 events, 10 queries | Coarse adaptation; not at full 50-turn scale |
| Fact recall precision/recall | ✅ **IMPLEMENTED** | `benchmarks.py` `benchmark_fact_recall` | Measures precision, recall, F1 |
| Generalization accuracy (unseen patterns) | ⚠️ **PARTIAL** | `benchmarks.py` `benchmark_crop_cycle_memory` | Indirectly tested via novel cycle queries |
| Counterfactual reasoning accuracy | ❌ **MISSING** | — | Not explicitly benchmarked |
| Temporal reasoning accuracy | ✅ **IMPLEMENTED** | `benchmarks.py` `benchmark_temporal_reasoning` | Temporal ordering accuracy metric |
| Semantic Density Robustness (rho sweep) | ✅ **IMPLEMENTED** | `benchmarks.py` `benchmark_semantic_density_robustness` | Sweeps rho from 0.1 to 0.9 |
| End-to-end latency (p99 < 5s) | ✅ **IMPLEMENTED** | `profiling.py` `profile_cls_end_to_end` | Measures avg/p99/min/max latency |
| Storage footprint (< 500 MB at 50K facts) | ❌ **MISSING** | — | Not measured at target scale |

### Gap Analysis
- **Counterfactual reasoning** is the most significant missing benchmark. The research contract (§4) calls it out specifically as a hypothesis test for iterative bidirectional querying.
- **LongMemEval adaptation** is coarse — the research contract requests 50+ turn conversations with interleaved crop cycles and disease events. Current benchmark uses 30 events and 10 queries.
- **Storage footprint** at target scale (50K facts, <500 MB) is not measured.

---

## 3. Single-Field Ablations from `ml-architect`

| # | Ablation | Status | Implementation | Files |
|---|---|---|---|---|
| 1 | No iterative querying | ✅ **IMPLEMENTED** | `max_iterative_cycles=1`, `enable_semantic_prior_routing=False`, `enable_episodic_revision=False` | `ablation_runner.py` — `no_iterate` |
| 2 | No semantic ML layer | ✅ **IMPLEMENTED** | `set_ablation_mode("kg_only")` | `ablation_runner.py` — `kg_only` |
| 3 | Generic KG (no typed objects) | ✅ **IMPLEMENTED** | `set_ablation_mode("generic_kg")` | `ablation_runner.py` — `generic_kg` |
| 4 | LLM → confidence_max reconciliation | ✅ **IMPLEMENTED** | `reconciliation_method="confidence_max"` | `ablation_runner.py` — `confidence_max` |
| 5 | Parallel → sequential initial query | ✅ **IMPLEMENTED** | `parallel_initial_query=False` | `ablation_runner.py` — `sequential_query` |
| 6 | Consolidation frequency sweep | ✅ **IMPLEMENTED** | Sweep 60/360/4320 min | `ablation_runner.py` — `freq_60min`, `freq_360min`, `freq_4320min` |
| 7 | Prototype slot count sweep | ✅ **IMPLEMENTED** | Sweep 8/32/128 slots | `ablation_runner.py` — `slots_8`, `slots_32`, `slots_128` |
| 8 | GCN → MLP encoder | ⚠️ **REGISTERED, NOT IMPLEMENTED** | `encoder_type="mlp"` set in config | `ablation_runner.py` — `gcn_mlp`; `SemanticPatternExtractor` does not support MLP mode |

### Gap Analysis
Ablations 1-7 are fully implemented and runnable. Ablation 8 is registered in the ablation runner but will fail at runtime because `SemanticPatternExtractor` uses `EdgeAwareGCN` regardless of `encoder_type`. This requires adding an MLP-based encoder path.

---

## 4. Synthetic Benchmarks Implemented

| Benchmark | Domain | File | Module |
|---|---|---|---|
| Fact recall precision/recall | LM/Memory Systems | `benchmarks.py:benchmark_fact_recall` | Domain A |
| Temporal reasoning accuracy | LM/Memory Systems | `benchmarks.py:benchmark_temporal_reasoning` | Domain A |
| Semantic Density Robustness | LM/Memory Systems (novel) | `benchmarks.py:benchmark_semantic_density_robustness` | Domain A |
| Write throughput | LM/Memory Systems | `benchmarks.py:benchmark_fact_write_throughput` | Domain A |
| Expressiveness probe | Graph ML | `benchmarks.py:benchmark_expressiveness_probe` | Domain B |
| Oversmoothing check | Graph ML | `benchmarks.py:benchmark_oversmoothing_check` | Domain B |
| Crop cycle memory | Scientific ML (Agriculture) | `benchmarks.py:benchmark_crop_cycle_memory` | Domain C |
| Disease progression tracking | Scientific ML (Agriculture) | `benchmarks.py:benchmark_disease_progression_tracking` | Domain C |
| Spatial proximity retrieval | Scientific ML (Agriculture) | `benchmarks.py:benchmark_spatial_proximity_retrieval` | Domain C |

---

## 5. Ablations Implemented

| Ablation | Config Field | Baseline → Ablated | Script |
|---|---|---|---|
| no_iterate | `max_iterative_cycles` + routing flags | 5 → 1 | `ablation_runner.py` |
| kg_only | N/A (runtime mode) | full → kg_only | `ablation_runner.py` |
| generic_kg | N/A (runtime mode) | full → generic_kg | `ablation_runner.py` |
| confidence_max | `reconciliation_method` | llm_judge → confidence_max | `ablation_runner.py` |
| sequential_query | `parallel_initial_query` | True → False | `ablation_runner.py` |
| freq_60min | `consolidation_frequency_minutes` | 1440 → 60 | `ablation_runner.py` |
| freq_360min | `consolidation_frequency_minutes` | 1440 → 360 | `ablation_runner.py` |
| freq_4320min | `consolidation_frequency_minutes` | 1440 → 4320 | `ablation_runner.py` |
| slots_8 | `n_pattern_slots` | 64 → 8 | `ablation_runner.py` |
| slots_32 | `n_pattern_slots` | 64 → 32 | `ablation_runner.py` |
| slots_128 | `n_pattern_slots` | 64 → 128 | `ablation_runner.py` |
| gcn_mlp (STUB) | `encoder_type` | gcn → mlp | `ablation_runner.py` — **not functional** |

---

## 6. Metrics Reported

| Metric | Implementation | When Reported |
|---|---|---|
| Fact recall precision/recall/F1 | `benchmark_fact_recall` | On benchmark run |
| Temporal ordering accuracy | `benchmark_temporal_reasoning` | On benchmark run |
| Robustness drop (acc@0.1 - acc@0.9) | `benchmark_semantic_density_robustness` | On benchmark run |
| Write throughput (events/s) | `benchmark_fact_write_throughput` | On benchmark run |
| Expressiveness (can distinguish) | `benchmark_expressiveness_probe` | On benchmark run |
| Oversmoothing collapse ratio | `benchmark_oversmoothing_check` | On benchmark run |
| Crop cycle accuracy | `benchmark_crop_cycle_memory` | On benchmark run |
| Spatial retrieval counts | `benchmark_spatial_proximity_retrieval` | On benchmark run |
| Diagnostic accuracy | `ablation_runner.py` | On ablation run |
| Average/P99 latency | `profiling.py` | On profiling run |
| Parameter count | `base.py:count_params` | On model init |
| Est. forward FLOPs | `profiling.py:profile_semantic_ml` | On profiling run |
| Peak CUDA memory | `profiling.py:profile_semantic_ml` | On profiling run (CUDA only) |
| KG node/edge counts | `kg.py:get_statistics` | On query |

---

## 7. Results Still TODO: Unverified

| Result | Why Unverified | Required Action |
|---|---|---|
| Iterative querying improves accuracy over one-directional | Ablation infrastructure exists but not run | `python ablation_runner.py --all` |
| External baselines are worse on ag diagnostics | No external system integration | Implement Zep/Mem0/Letta/AOI comparison |
| Semantic ML layer improves generalization | kg_only ablation infrastructure exists | `python ablation_runner.py --ablation kg_only` |
| Domain-specific typed objects help | generic_kg ablation infrastructure exists | `python ablation_runner.py --ablation generic_kg` |
| LLM reconciliation is worth the overhead | confidence_max ablation exists | `python ablation_runner.py --ablation confidence_max` |
| Storage footprint < 500 MB at 50K facts | No measurement at scale | Profile with 50K events |
| Latency budget p99 < 5s at full scale | Small-scale profiling only | Profile with full KG + real LLM |
| GCN outperforms MLP encoder | MLP encoder not implemented | Add MLP support to SemanticPatternExtractor |

---

## 8. Can the Benchmark Suite Distinguish the Architecture from a Trivial Baseline?

**Partially.** The benchmarks focus on:
- **Within-architecture comparisons** (ablation A vs. ablation B) — well-supported via ablation_runner.py
- **Absolute metrics** (latency, throughput, recall) — supported via benchmarks.py and profiling.py
- **Theoretical grounding validation** (density robustness, oversmoothing) — supported

**Not supported:**
- **Cross-architecture comparison** — there is no "baseline architecture" runner for Zep, Mem0, Letta, or AOI
- **Random baseline** — there is no test that compares accuracy against random guessing or heuristic rules
- **Ablated-architecture vs. trivial-baseline** — the ablation runner compares variants of the full architecture but not against, e.g., a simple rule-based diagnostic system

**Recommendation:** Add a `--baseline random` mode to the benchmark runner that replaces the CLS system with random or heuristic responses, providing a lower bound for accuracy comparison.

---

## 9. Validation Results

### Test Suite Status

| Layer | Test Class | Tests | Status |
|---|---|---|---|
| 1a — Shape | `TestShapes` | 10 | ✅ All pass |
| 1b — Gradient Flow | `TestGradients` | 4 | ✅ All pass |
| 1c — Correctness | `TestCorrectness` | 8 | ✅ All pass |
| 1d — Numerical Stability | `TestNumerics` | 6 | ✅ All pass |
| 1e — Invariance | `TestPermutationInvariance` | 1 | ✅ Passes |
| 1f — Ablations | `TestAblationModes` | 5 | ✅ All pass |
| 1g — Interfaces | `TestInterfaceContracts` | 4 | ✅ All pass |
| **Total** | **7 classes** | **38** | **✅ All pass** |

### Real Bugs Found During Validation

The validator discovered **5 real bugs** in the coder implementation through its test suite (not test-logic errors):

| # | Severity | File | Bug | Fix |
|---|---|---|---|---|
| 1 | 🔴 HIGH | `coder/layers.py:GCNLayer` | LayerNorm mixed-dtype error when model in bf16/fp16 mode | Use `F.layer_norm` with explicit `.float()` on norm params |
| 2 | 🔴 HIGH | `coder/semantic.py:173` | `batch.node_mask.float()` forces float32 even in bf16 mode | Use `.to(dtype=h_out.dtype)` |
| 3 | 🟡 MEDIUM | `coder/layers.py:ContrastiveLoss` | Empty tensor `.mean()` → NaN when all labels share a class | Add `if neg_sim.numel() > 0:` guard |
| 4 | 🟡 MEDIUM | `coder/layers.py:PrototypeAttention` | float32 input cast + bf16 attention weights → dtype mismatch | Temporarily promote attention module to float32 |
| 5 | 🟡 MEDIUM | `coder/semantic.py:190-191` | confidence_head float32 cast + bf16 weights → dtype mismatch | Same float32 promotion pattern |

### Verifier Components Smoke Test

| Component | Status | Notes |
|---|---|---|
| `pytest test_model.py -v` | ✅ 38/38 pass | All unit tests pass |
| `benchmarks.py --benchmark expressiveness` | ✅ Runs correctly | GCN correctly identified as 1-WL bounded |
| `ablation_runner.py --ablation baseline` | ✅ Runs correctly | Accuracy 0/8 (expected for untrained model) |
| `profiling.py --subsystem semantic` | ✅ Runs correctly | Reports 1.37M params, 2.8M FLOPs, 5.83ms avg |

---

*Generated by ml-validator. See also: `claim_grounding.md`, `scorecard.json`, `rubric.md`*
