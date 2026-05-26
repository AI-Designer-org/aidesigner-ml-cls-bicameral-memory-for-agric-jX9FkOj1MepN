# Training & Reproduction

## Environment

The implementation uses a multi-paradigm architecture — the episodic KG is not a learned component, and the semantic ML layer is a standalone PyTorch module trained via contrastive consolidation. The agent controller uses an LLM interface (stub by default).

- Python: 3.12+
- PyTorch: 2.2+ (tested with 2.2–2.5)
- CUDA: Optional; CPU inference is sufficient for the semantic ML layer (~2.8M FLOPs forward)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu  # or cu118/cu121
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> **No additional dependencies** — the core implementation uses only Python standard library + PyTorch. The LLM stub in `controller.py` returns deterministic responses; for production use, replace `LLMInterface.complete()` with the OpenAI/Anthropic SDK.

## Default hyperparameters

### EpisodicKnowledgeGraph

| Field | Default | Rationale |
|---|---|---|
| `max_triples` | 50,000 | Covers 50+ turn conversations × multiple crop cycles per agricultural season |
| `max_nodes` | 10,000 | Unique entity capacity (fields, crops, disease events, treatments) |
| `temporal_resolution_seconds` | 3600 | 1-hour binning — sufficient for agricultural events (daily to weekly resolution) |
| `spatial_grid_size_meters` | 100 | 100m grid — matches typical field size granularity |
| `write_admission` | "append" | All events admitted (negative evidence is informative for diagnosis) |
| `dedup_window_seconds` | 300 | 5-min dedup for sensor/field-report duplicates |
| `temporal_path_max_depth` | 5 | Typical ag episode spans 5 hops (observation → diagnosis → treatment → follow-up → resolution) |
| `spatial_proximity_radius_m` | 500 | Disease spread is most correlated within 500m (wind-borne spores, shared irrigation) |

### SemanticPatternExtractor

| Field | Default | Rationale |
|---|---|---|
| `hidden_dim` | 256 | Sufficient capacity for agricultural disease patterns |
| `n_layers` | 3 | Deep enough for meaningful message passing; shallow enough to avoid over-smoothing |
| `n_pattern_slots` | 64 | ~20 common crops × ~3 major disease types each, with margin for novel patterns |
| `pattern_embed_dim` | 128 | Compressed pattern representation for prototype matching |
| `n_heads` | 4 | Reasonable for prototype attention over 64 slots |
| `d_ff` | 1024 | Feed-forward dimension for pattern projector |
| `dropout` | 0.1 | Light regularization for contrastive consolidation |
| `consolidation_batch_size` | 256 | Minimum subgraphs before consolidation triggers |
| `consolidation_lr` | 1e-4 | Conservative learning rate for offline consolidation |
| `consolidation_n_epochs` | 10 | Few epochs per consolidation round (contrastive loss converges quickly) |
| `contrastive_margin` | 1.0 | Margin for hinge component in contrastive loss |
| `consolidation_frequency_minutes` | 1440 | Daily consolidation — disease cycles operate on 3-14 day timescales |
| `consolidation_warmup_hours` | 48 | Wait before first consolidation to accumulate sufficient episodic data |
| `confidence_threshold` | 0.7 | Minimum confidence for semantic-only response; below this, refine |
| `few_shot_k` | 5 | K episodes for few-shot pattern adaptation |

### AgentController

| Field | Default | Rationale |
|---|---|---|
| `llm_model` | "gpt-4o-mini" | Production-viable cost profile |
| `llm_temperature` | 0.1 | Deterministic diagnostic reasoning (low temperature) |
| `max_iterative_cycles` | 5 | Bounded latency; typical convergence in 2-3 cycles |
| `early_exit_confidence` | 0.9 | Exit reconciliation loop when confidence exceeds this threshold |
| `parallel_initial_query` | True | Query both memory systems in parallel on first pass |
| `enable_semantic_prior_routing` | True | Use semantic priors to refine KG queries |
| `enable_episodic_revision` | True | Use episodic findings to revise semantic beliefs |
| `reconciliation_method` | "llm_judge" | LLM-based reconciliation (alternatives: "confidence_max", "weighted_vote") |
| `working_memory_max_tokens` | 16,000 | Active session context limit |
| `working_memory_eviction` | "lru" | Least-recently-used eviction for context overflow |

## Recommended training recipe

> **Note:** The semantic ML layer is trained via **offline consolidation**, not end-to-end supervised learning. There is no "training run" in the traditional sense — training is triggered periodically by the `ConsolidationScheduler`.

| Setting | Value | Notes |
|---|---|---|
| Optimizer | AdamW | β1=0.9, β2=0.999 |
| Peak LR | 1e-4 | Used in consolidation training |
| Batch size (consolidation) | 256 | Min subgraphs before consolidation |
| Weight decay | Not applied | Contrastive loss only; parameters are small (1.4M) |
| Grad clip | 1.0 | Global norm, applied in `extractor.consolidate()` |
| Precision | float32 (default) | bf16/fp16 supported and tested |

### Training lifecycle

1. **Episodic accumulation phase** (0–48 hr): The system ingests observations, disease reports, and treatment events via `fast_write()`. No semantic training occurs during warmup.
2. **First consolidation** (48+ hr): `ConsolidationScheduler.should_consolidate()` returns True → `SemanticMemoryManager.consolidate()` trains the `SemanticPatternExtractor` on matured episodic subgraphs using contrastive loss.
3. **Ongoing cycling**: Consolidation triggers every `consolidation_frequency_minutes` (default: daily). After each consolidation, the KG's subgraph embeddings are rebuilt for semantic-prior retrieval.
4. **On-demand consolidation**: Can be triggered explicitly via `system.consolidate(force=True)` after high-value diagnostic sessions.

### Consolidation training loop (from `semantic.py:SemanticPatternExtractor.consolidate()`)

```python
for epoch in range(n_epochs):  # default: 10
    optimizer.zero_grad()
    output = self._forward_impl(batch)
    loss = contrastive_loss(
        output["pattern_embed"],
        labels=batch.labels,
        temperature=0.1,
        margin=config.contrastive_margin,
    )
    loss.backward()
    clip_grad_norm_(self.parameters(), 1.0)
    optimizer.step()
```

### Domain-specific notes

- **Semantic Density**: Agricultural disease data is semantically dense (disease → symptom → treatment → season are correlated at ρ > 0.6). The contrastive loss must separate these correlated patterns into distinct prototype slots.
- **Few-shot adaptation**: After consolidation, the extractor supports K-shot adaptation for novel disease-crop combinations via `few_shot_adapt()`. This uses a prototypical network approach (Snell et al., NeurIPS 2017) and does not require gradient updates.
- **Order of consolidation**: The `extract_consolidation_batch()` method returns subgraphs in chronological order (oldest first). This naturally implements a curriculum — earlier-season patterns (e.g., early blight) are learned before later-season patterns (e.g., late blight).

### Expected behavior

> No reference training run exists. The consolidation infrastructure produces a training loss that decreases over epochs, but no benchmark accuracy targets have been validated experimentally. Run `python benchmarks.py --benchmark fact_recall` after consolidation to measure fact recall.

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Consolidation skipped | Fewer than `consolidation_batch_size` matured subgraphs | Wait longer or reduce `consolidation_batch_size` |
| Consolidation loss is NaN | Empty tensor in contrastive loss (all-same-class labels) | Check batch labels have at least 2 distinct classes |
| bf16 forward pass fails with mixed-dtype error | `batch.node_mask.float()` promotes tensor to float32 | Fixed in v0.1.0 — uses `.to(dtype=h_out.dtype)` |
| Prototype attention mat1/mat2 dtype mismatch | bf16 model weights with float32 input | Fixed in v0.1.0 — attention module temporarily promoted to float32 |
| LLM reconciliation returns malformed JSON | LLM stub provides deterministic responses; production LLM may produce parse errors | The `_llm_reconciliation()` method falls back to `confidence_max` on `json.JSONDecodeError` |
| Few-shot adaptation returns low similarity | Support subgraphs may not match query subgraph | Increase `few_shot_k` or ensure support set covers relevant disease |
| Semantic Density Robustness benchmark shows no density sensitivity | Small-scale benchmark (20 events) may be below Stability Gap threshold | Increase to N=50+ events with controlled ρ values |
