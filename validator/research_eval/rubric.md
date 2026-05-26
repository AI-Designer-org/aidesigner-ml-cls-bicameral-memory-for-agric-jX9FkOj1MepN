# Research Evaluation Rubric — CLS Bicameral Memory for Agricultural Agents

> **Validator Layer 5 — Research-Quality Scorecard**
> Generated: 2026-05-26

---

## Scoring Scale

| Score | Meaning |
|---|---|
| 0 | Not addressed or no artifact exists |
| 1 | Mentioned but unsupported |
| 2 | Partially supported with major gaps |
| 3 | Plausible and minimally supported |
| 4 | Strong, with clear evidence and reproducible checks |
| 5 | Publication-ready for this scaffold's scope |

---

## Dimension 1: Novelty (Score: 3/5)

### What's Being Scored
Whether the architecture represents a genuine advance over the existing literature identified in the research contract (AOI, All-Mem, Zep/Graphiti, Dual-System Memory, "Mind the Gap").

### Evidence
- **Iterative bidirectional querying protocol** is the primary novelty claim. The protocol is implemented in `controller.py` (`CLSAgentController.diagnose()`) with a clear loop structure: initial parallel query → reconciliation → semantic→episodic refinement → episodic→semantic refinement → loop until convergence. This pattern is **not present** in AOI (one-directional compression) or All-Mem (one-directional consolidation).
- **Domain-specific spatio-temporal KG objects** (`CropCycle`, `DiseaseEvent`, `TreatmentAction` in `data_model.py`) are first-class entities with typed enums, temporal fields, and spatial coordinates. This tailors the CLS episodic memory to agricultural diagnostics beyond what generic KG frameworks provide.
- **Stability Gap grounding** is correctly cited (arXiv:2601.15313) and the architecture responds to it with bicameral separation using **different storage representations** (graph store vs. neural network), not just separated vector databases which would share the same underlying representation.

### Concerns
- The CLS bicameral design itself is **not novel** — multiple production systems implement it. The novelty resides entirely in (a) the iterative querying protocol, (b) agricultural domain tailoring, and (c) the empirics of evaluating on agricultural diagnostics.
- The iterative bidirectional protocol is **implemented but its effectiveness is unmeasured**. It could be that one-directional consolidation suffices, which would collapse the primary novelty claim.
- **External baselines not run** — without comparing against Zep, Mem0, Letta, or AOI on the same benchmark, the architecture's relative improvement cannot be assessed.

### Domain-Specific Questions
- **LM/Memory Systems:** Does the benchmark suite test the iterative querying protocol? → Yes, `ablation_runner.py --ablation no_iterate` isolates this. Does it compare against one-directional consolidation? → Yes, the `no_iterate` ablation is specifically this comparison.
- **Graph ML:** Does it test expressiveness beyond 1-WL? → Partially (`benchmark_expressiveness_probe`), but the GCN used is 1-WL by default. Extra expressiveness comes from prototype attention, which is not directly tested for expressiveness.
- **Scientific ML (Agriculture):** Does it test domain-specific episodic objects? → Yes (`benchmark_crop_cycle_memory`, `benchmark_disease_progression_tracking`, `benchmark_spatial_proximity_retrieval`).

---

## Dimension 2: Experimental Comprehensiveness (Score: 4/5)

### What's Being Scored
Whether the validator suite covers all evaluation requirements from the research contract, including baselines, ablations, benchmarks, and metrics.

### Evidence
- **9 domain-specific benchmarks** cover the primary evaluation requirements from the research contract: fact recall (`benchmark_fact_recall`), temporal reasoning (`benchmark_temporal_reasoning`), semantic density robustness (`benchmark_semantic_density_robustness`), write throughput (`benchmark_fact_write_throughput`), expressiveness (`benchmark_expressiveness_probe`), oversmoothing (`benchmark_oversmoothing_check`), crop cycle memory (`benchmark_crop_cycle_memory`), disease progression (`benchmark_disease_progression_tracking`), spatial proximity (`benchmark_spatial_proximity_retrieval`).
- **8 ablation variants** covering all single-field config changes from the architecture blueprint §12, including: no_iterate, kg_only, generic_kg, confidence_max, sequential_query, frequency sweeps (60/360/4320 min), prototype slot sweeps (8/32/128), and GCN→MLP.
- **Novel metric implemented:** Semantic Density Robustness (rho sweep 0.1-0.9) — this is a unique contribution of the research contract.
- **30+ pytest tests** across 7 test classes covering shapes, gradients, correctness, numerics, permutation invariance, ablation modes, and interface contracts.
- **Profiling script** measures latency, FLOPs, parameter counts, and memory for each subsystem.

### Concerns
- **External baselines are missing.** The research contract explicitly requires comparison against Zep, Mem0, Letta, and AOI. These require external system integration beyond the current codebase, but their absence means the falsification condition from the research contract cannot be tested.
- **Counterfactual reasoning** is listed as an evaluation requirement but is only indirectly covered by the fact_recall and temporal_reasoning benchmarks. A dedicated counterfactual benchmark is missing.
- **Storage footprint at target scale** (50K facts, <500 MB) is not measured.
- **Latency-accuracy Pareto frontier** for the iterative cycle count vs. accuracy tradeoff is not plotted.

---

## Dimension 3: Theoretical Foundation (Score: 4/5)

### What's Being Scored
Whether the architecture's design choices are grounded in established theory, and whether the theoretical claims made in the research contract are accurately reflected in the implementation.

### Evidence
- **Stability Gap theorem (arXiv:2601.15313)** drives the core architectural choice: bicameral separation using different storage representations (graph + neural network), not just partitioned vector stores.
- **GCN with heterophily gate** is theoretically motivated for agricultural disease KGs where healthy and infected fields are adjacent but must remain distinct in representation space — standard GCNs would smooth this boundary.
- **Prototypical networks** for few-shot adaptation is a well-studied approach (Snell et al., NeurIPS 2017).
- **Contrastive loss** for consolidation training is standard for representation learning in limited-label regimes.
- **Temporal + spatial dual indexing** is motivated by the observation that agricultural queries split evenly between "when?" and "where?" questions.

### Concerns
- **No convergence guarantee** for the iterative bidirectional protocol. The loop terminates on `max_iterative_cycles` or a confidence threshold, but there's no proof of monotonic improvement or convergence rate.
- **Expressiveness ceiling** is acknowledged (1-WL for property graphs) but the claim that prototype attention lifts this ceiling is unproven.
- **Text embedding** uses a simple bag-of-ngrams hash — there's no theoretical justification for why this is sufficient for matching agricultural queries to prototype vectors. In production, this would need to be replaced with Sentence-BERT or similar.

---

## Dimension 4: Result Analysis (Score: 2/5)

### What's Being Scored
Whether the validator produces actual experimental results or just infrastructure for results.

### Evidence
- Tests validate code **correctness** (shapes, gradients, numerical stability) but not research **hypotheses**.
- Ablation runner and benchmarks produce metrics when run, but no pre-computed results exist.
- Profiling script measures latency and FLOPs per subsystem.

### Concerns
- **No experimental results exist.** The ablation runner and benchmarks must be executed before any research claims can be supported. The current state is "infrastructure complete, results pending."
- **No external baseline comparisons.** The falsification condition from the research contract (§4) requires comparison against monolithic architectures — this has not been done.
- **No statistical significance.** All benchmarks use a single random seed and small sample sizes.
- **No learning curves, convergence plots, or confidence intervals.**

---

## Dimension 5: Implementation Reproducibility (Score: 4/5)

### What's Being Scored
Whether the implementation is deterministic, well-configured, and can be reproduced by an independent researcher.

### Evidence
- **Seeded initialization** (`torch.manual_seed(config.seed)`) ensures deterministic model instantiation.
- **Frozen config dataclasses** (`@dataclass(frozen=True)`) prevent configuration drift.
- **CLI-driven scripts** for benchmarks, ablations, and profiling enable exact reproduction: `python benchmarks.py`, `python ablation_runner.py`, `python profiling.py`.
- **Comprehensive pytest suite** validates unit-level correctness before system-level evaluation.
- **Device-portable** — all tensors can be moved between CPU and CUDA via `.to(device)`.
- **Interface contracts** (`BaseEpisodicMemory`, `BaseSemanticMemory`, `BaseAgentController`) ensure subsystem compatibility.

### Concerns
- **LLM stub** returns deterministic responses — production runs with real LLMs will differ.
- **Random data generation** in benchmarks is not seed-saved for exact reproduction of eval sets.
- **`id()` in node IDs** (`kg.py` line 576: `f"{type(event).__name__.lower()}_{id(event)}_{now.timestamp()}"`) is not deterministic across process restarts.
- **No environment specification** (Dockerfile, conda environment.yml, or requirements.txt beyond stdlib + torch).

---

## Dimension 6: Writing Readiness (Score: 3/5)

### What's Being Scored
Whether the codebase, benchmarks, and documentation are structured for paper writing, including shape comments, provenance tracking, and clear claim-to-code mapping.

### Evidence
- **Shape comments** are present throughout `layers.py`, `semantic.py`, and `kg.py` — verified by `test_tensor_shape_comments()` in the smoke test (30+ shape annotations).
- **Comprehensive docstrings** on every class and method follow a consistent format with Args/Returns sections.
- **Provenance tracking** is implemented in the agent controller — every diagnostic response includes per-claim source tags (`episodic_kg` or `semantic_ml`).
- **Claim grounding** is documented in `claim_grounding.md` — every research claim maps to file paths and test commands.

### Concerns
- **No paper draft, abstract, or figure generation** infrastructure exists.
- **Benchmark results** are printed to stdout but not formatted as LaTeX tables or publication-ready figures.
- **No README** explaining the end-to-end evaluation pipeline.
- **Provenance visualization** is not implemented — provenance is stored in structured dicts but not rendered for human consumption.

---

## Blocking Gaps

| Code | Severity | Description |
|---|---|---|
| `baseline_not_beaten` | **BLOCKING** | No external baseline comparison (Zep, Mem0, Letta, AOI) — primary falsification condition untestable |
| `claim_not_grounded` | **BLOCKING** | Iterative bidirectional querying benefit unmeasured — primary novelty claim lacks evidence |
| `benchmark_not_executable` | **WARNING** | Random data generation without versioning → results not reproducible across runs |
| `ablation_missing` | **WARNING** | GCN→MLP encoder ablation registered but unimplemented (encoder_type="mlp" unsupported) |
| `novelty_unverified` | **BLOCKING** | Core novelty claim has test infrastructure but no results |

---

## Recommended Next Experiments (Priority Order)

1. **P0 — Full ablation run:** `python ablation_runner.py --all` → accuracy × latency Pareto for all configs
2. **P0 — External baseline integration:** Port agricultural benchmark to Zep/Mem0/Letta API for comparison
3. **P1 — Density robustness at scale:** `python benchmarks.py --benchmark density_robustness` with N=50, seed-fixed data
4. **P1 — Iteration count sweep:** Evaluate accuracy vs. `max_iterative_cycles={1,2,3,5}` at fixed event count
5. **P1 — Storage measurement:** Profile memory at 10K, 25K, 50K facts
6. **P2 — MLP encoder ablation:** Add MLP support to SemanticPatternExtractor, then run ablation
7. **P2 — Prototype utilization analysis:** Measure entropy of prototype_weights after consolidation to detect oversmoothing
