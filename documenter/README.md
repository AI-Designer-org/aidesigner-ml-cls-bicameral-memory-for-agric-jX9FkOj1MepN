# CLS Bicameral Memory for Agricultural Diagnostic Agents

A Complementary Learning Systems (CLS) memory architecture that decouples agent memory into a fast-learning episodic knowledge graph and a slow-learning semantic ML layer, orchestrated by an LLM-based controller with iterative bidirectional querying — designed for autonomous agricultural diagnostics.

This architecture addresses a provable limitation of monolithic neural memory: the Stability Gap (arXiv:2601.15313) proves that single-representation memory collapses under the semantic density of agricultural domain knowledge (disease-symptom-treatment-season correlations at ρ > 0.6). By mimicking the brain's separation of hippocampus (episodic) and neocortex (semantic), the architecture preserves both specific timestamped observations and generalized disease patterns. The primary novelty is an **iterative bidirectional querying protocol** — episodic and semantic layers query each other in a loop until convergence, rather than consolidating in only one direction (episodic → semantic) as in prior CLS systems (AOI, All-Mem).

> **Status:** Proof-of-concept implementation v0.1.0. Code correctness validated (38/38 tests pass). Research claims — including the marginal benefit of iterative bidirectional querying — are infrastructure-ready but results are **TODO: unverified**.

## Highlights

- **Bicameral CLS separation** — Episodic KG (fast-write spatio-temporal graph) + Semantic ML (slow-learning GCN-based pattern extractor) use different storage representations to avoid the Stability Gap; see [ARCHITECTURE.md](docs/ARCHITECTURE.md#3-the-core-component)
- **Iterative bidirectional querying** — Agent controller orchestrates a reconciliation loop between episodic and semantic layers, enabling semantic priors to refine KG search and episodic anomalies to update semantic beliefs; see [ARCHITECTURE.md](docs/ARCHITECTURE.md#5-iterative-bidirectional-querying-protocol)
- **Domain-tailored spatio-temporal objects** — First-class `CropCycle`, `DiseaseEvent`, and `TreatmentAction` nodes with typed enums, temporal coordinates, and spatial indexing for agricultural diagnostics; see [ARCHITECTURE.md](docs/ARCHITECTURE.md#6-domain-specific-considerations)
- **Ablation-ready design** — Four configurable ablation modes (`full`, `kg_only`, `no_iterate`, `generic_kg`) and 10+ single-field config variants for hypothesis testing; see [BENCHMARKS.md](docs/BENCHMARKS.md#ablation-study)
- **bf16/fp16 safe** — All layers pass numerical stability tests in float16, bfloat16, and float32; see [TRAINING.md](docs/TRAINING.md#recommended-training-recipe)

## Quick start

```bash
# Smoke test — instantiates full system, ingests synthetic events, validates shapes
python coder/smoke_test.py

# Unit test suite — 38 tests across 7 test classes
python -m pytest validator/test_model.py -v

# Run domain benchmarks
python validator/benchmarks.py --benchmark fact_recall

# Run ablation suite
python validator/ablation_runner.py --all
```

## Repository layout

```
coder/                          # PyTorch implementation
├── config.py                   # Frozen dataclasses for all subsystem configs
├── data_model.py               # Domain objects (CropCycle, DiseaseEvent, etc.)
├── base.py                     # Abstract base classes (interfaces)
├── kg.py                       # EpisodicKnowledgeGraph (hippocampus analogue)
├── layers.py                   # Neural network layers (GCN, attention, contrastive loss)
├── semantic.py                 # SemanticPatternExtractor + SemanticMemoryManager
├── controller.py               # CLSAgentController + WorkingMemory + ConsolidationScheduler
├── model.py                    # Top-level CLSMemorySystem (ties subsystems together)
├── smoke_test.py               # End-to-end smoke test
└── __init__.py                 # Public API exports
validator/                      # Validation and evaluation
├── test_model.py               # 38 unit tests (7 test classes)
├── benchmarks.py               # 9 domain-specific benchmarks
├── ablation_runner.py          # 12 ablation configurations
├── profiling.py                # Latency, FLOPs, and memory profiling
└── research_eval/              # Research-quality scoring
    ├── scorecard.json          # Quantitative scores (6 dimensions)
    ├── claim_grounding.md      # Every claim mapped to code or "TODO: unverified"
    ├── experiment_coverage.md  # Evaluation requirements vs. implementation
    ├── ablation_results.json   # Ablation run data
    └── rubric.md               # Scoring rubric and blocking gaps
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — Design rationale, equations, inductive biases, known limitations
- [docs/TRAINING.md](docs/TRAINING.md) — Environment setup, hyperparameters, training recipe, troubleshooting
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — Benchmark results, ablation study, profiling, research-quality evaluation
- [docs/API.md](docs/API.md) — Module-level API reference for all public classes and functions

## Related work

This architecture builds on the Stability Gap theorem (arXiv:2601.15313) and differentiates from AOI (arXiv:2512.13956), All-Mem (arXiv:2603.19595), and Zep/Graphiti (arXiv:2501.13956) through its iterative bidirectional querying protocol and agricultural domain tailoring. See [ARCHITECTURE.md](docs/ARCHITECTURE.md#1-motivation) for full positioning.

## Citation

```bibtex
@misc{cls-ag-memory-2026,
  title  = {CLS Bicameral Memory for Agricultural Diagnostic Agents:
            Iterative Bidirectional Querying in a Neuro-Symbolic
            Complementary Learning Systems Architecture},
  author = {<TODO>},
  year   = {2026},
  note   = {Generated via ml-designer pipeline}
}
```
