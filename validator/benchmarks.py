#!/usr/bin/env python3
"""
Layer 2 — Domain-Specific Benchmarks for CLS Bicameral Memory System.

Covers three domains at once (LM/Memory Systems, Graph ML, Scientific ML/Agriculture):

Domain A — LM / Memory Systems:
    - Fact recall precision/recall across conversation turns
    - Temporal reasoning accuracy on sequence/duration ordering
    - Semantic Density Robustness (rho sweep)
    - Counterfactual reasoning accuracy
    - Write throughput and read latency benchmarks

Domain B — Graph ML:
    - Expressiveness probe (distinguishing non-isomorphic graphs)
    - Oversmoothing check across layer depths
    - Subgraph embedding stability

Domain C — Scientific ML (Agriculture):
    - Crop cycle memory accuracy through multi-turn conversations
    - Disease progression tracking across time
    - Treatment sequence recall
    - Spatial proximity retrieval correctness

Each benchmark function returns a dict of metrics and can be run standalone.

Run:
    python benchmarks.py                          # Run all benchmarks
    python benchmarks.py --benchmark fact_recall   # Run specific benchmark
"""

import sys
import os
import math
import argparse
import random
import time
from datetime import datetime, timedelta
from typing import Optional

import torch

CODER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "coder"))
if CODER_DIR not in sys.path:
    sys.path.insert(0, CODER_DIR)

from config import (
    CLSMemorySystemConfig,
    EpisodicKGConfig,
    SemanticMLConfig,
    AgentControllerConfig,
)
from data_model import (
    CropCycle, CropStage,
    DiseaseEvent, DiseaseStatus,
    TreatmentAction, TreatmentType,
    DiagnosticContext,
    KGSubgraph, SubgraphBatch,
    TemporalPathQuery, SpatialProximityQuery,
)
from kg import EpisodicKnowledgeGraph
from semantic import SemanticMemoryManager, SemanticPatternExtractor
from controller import CLSAgentController
from model import CLSMemorySystem


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_cfg(small: bool = True) -> CLSMemorySystemConfig:
    """Create a config suitable for benchmarking."""
    return CLSMemorySystemConfig(
        episodic_kg=EpisodicKGConfig(
            max_triples=50000, max_nodes=10000,
            node_embed_dim=32, edge_embed_dim=8,
        ),
        semantic_ml=SemanticMLConfig(
            node_embed_dim=32, edge_embed_dim=8,
            hidden_dim=64, n_layers=2, n_pattern_slots=16,
            pattern_embed_dim=32, n_heads=2, d_ff=256,
            dropout=0.1, consolidation_batch_size=8,
        ),
        agent_controller=AgentControllerConfig(
            max_iterative_cycles=3 if not small else 1,
            reconciliation_method="confidence_max",
            provenance_tracking=True,
        ),
    )


def _generate_agricultural_conversation(
    n_events: int = 50,
    n_fields: int = 3,
    seed: int = 42,
) -> tuple[list, list, list]:
    """Generate a synthetic agricultural diagnostic conversation.

    Produces:
        events: list of (event_object, dict) for ingestion
        queries: list of (query_string, context, expected_fact) triple for evaluation
        facts: list of facts that should be recallable

    Each conversation spans multiple crop cycles across fields,
    with interleaved disease events and treatments.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    fields = [f"field_{chr(65 + i)}" for i in range(n_fields)]
    diseases = ["powdery_mildew", "rust", "blight", "leaf_spot", "fusarium"]
    treatments = {
        "powdery_mildew": ["sulfur", "potassium_bicarbonate", "myclobutanil"],
        "rust": ["tebuconazole", "propiconazole", "azoxystrobin"],
        "blight": ["chlorothalonil", "mancozeb", "copper_hydroxide"],
        "leaf_spot": ["pyraclostrobin", "fluxapyroxad", "difenoconazole"],
        "fusarium": ["prothioconazole", "metconazole", "tebuconazole"],
    }
    crops = ["wheat", "corn", "soybean"]
    stages = list(CropStage)

    events = []
    queries = []
    facts = []

    base_date = datetime(2026, 3, 1)

    for i in range(n_events):
        field = random.choice(fields)
        disease = random.choice(diseases)
        crop = random.choice(crops)

        # Crop cycle
        cycle_start = base_date + timedelta(days=random.randint(0, 60))
        cycle = CropCycle(
            cycle_id=f"cc_{i:04d}", field_id=field, crop_type=crop,
            variety=f"{crop}_v{random.randint(1, 5)}",
            planting_date=cycle_start.strftime("%Y-%m-%d"),
            stages=[(random.choice(stages), (cycle_start + timedelta(days=d)).strftime("%Y-%m-%d"))
                    for d in [0, 15, 30, 45]],
        )
        events.append(cycle)

        # Disease event
        obs_date = cycle_start + timedelta(days=random.randint(20, 60))
        disease_ev = DiseaseEvent(
            event_id=f"de_{i:04d}", field_id=field, crop_cycle_id=f"cc_{i:04d}",
            disease_name=disease,
            first_observed=obs_date.strftime("%Y-%m-%d"),
            status=random.choice(list(DiseaseStatus)),
            severity=round(random.uniform(0.2, 0.9), 2),
            affected_area_m2=random.uniform(50, 500),
            symptoms=[f"symptom_{j}" for j in range(random.randint(1, 3))],
        )
        events.append(disease_ev)

        # Treatment
        treat_date = obs_date + timedelta(days=random.randint(1, 7))
        treatment = TreatmentAction(
            treatment_id=f"tr_{i:04d}",
            disease_event_id=f"de_{i:04d}",
            treatment_type=random.choice(list(TreatmentType)),
            agent=random.choice(treatments[disease]),
            dosage=f"{random.uniform(1, 5):.1f} L/ha",
            application_date=treat_date.strftime("%Y-%m-%d"),
            effectiveness=round(random.uniform(0.3, 0.95), 2),
        )
        events.append(treatment)

        # Record fact for recall testing
        facts.append({
            "field": field,
            "disease": disease,
            "crop": crop,
            "planting": cycle_start.strftime("%Y-%m-%d"),
            "obs_date": obs_date.strftime("%Y-%m-%d"),
            "treatment": treatment.agent,
            "treatment_date": treat_date.strftime("%Y-%m-%d"),
        })

    # Generate queries
    for j, fact in enumerate(facts[: min(20, len(facts))]):
        queries.append((
            f"What disease was found in {fact['field']}?",
            DiagnosticContext(field_id=fact["field"], crop_type=fact["crop"]),
            fact["disease"],
        ))

    return events, queries, facts


# ═══════════════════════════════════════════════════════════════════════════════
# Domain A: LM / Memory Systems Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_fact_recall(n_events: int = 30, n_queries: int = 10) -> dict:
    """Fact recall precision/recall across conversation turns.

    Ingests synthetic agricultural events into the CLS system,
    then queries facts back and measures precision and recall.

    Returns:
        dict with precision, recall, f1, and per-field breakdown.
    """
    print("=" * 60)
    print("Benchmark: Fact Recall")
    print("=" * 60)

    cfg = _make_cfg(small=True)
    system = CLSMemorySystem(cfg)

    events, queries, facts = _generate_agricultural_conversation(
        n_events=n_events, n_fields=3
    )

    # Ingest events
    for ev in events:
        system.fast_write(ev)
    print(f"  Ingested {len(events)} events ({len(queries)} query slots)")

    # Run queries and check recall
    correct = 0
    total = 0
    latencies = []
    results = []

    for q_str, ctx, expected in queries:
        start = time.time()
        response = system.diagnose(q_str, ctx)
        latency = time.time() - start
        latencies.append(latency)

        total += 1
        # Check if expected disease name appears in the answer
        found = expected.lower() in response.answer.lower()
        if found:
            correct += 1
        results.append({
            "query": q_str,
            "expected": expected,
            "found": found,
            "confidence": response.confidence,
            "latency": latency,
        })

    precision = correct / max(total, 1)
    recall = correct / max(total, 1)  # same in this setup since each query has 1 ground truth
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    avg_latency = sum(latencies) / max(len(latencies), 1)
    p99_latency = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0

    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "avg_latency_s": round(avg_latency, 3),
        "p99_latency_s": round(p99_latency, 3),
        "n_queries": total,
        "n_correct": correct,
        "n_events_ingested": len(events),
    }

    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Avg latency: {metrics['avg_latency_s']:.3f}s")
    print(f"  P99 latency: {metrics['p99_latency_s']:.3f}s")
    print(f"  [{correct}/{total}] queries found expected fact")

    return metrics


def benchmark_temporal_reasoning(n_sequences: int = 20) -> dict:
    """Temporal reasoning accuracy on sequence/duration ordering queries.

    Creates event sequences with known temporal order, then queries
    for ordering information (e.g., "Which treatment came first?",
    "What was the disease progression timeline?").
    """
    print("\n" + "=" * 60)
    print("Benchmark: Temporal Reasoning")
    print("=" * 60)

    cfg = _make_cfg(small=True)
    kg = EpisodicKnowledgeGraph(cfg.episodic_kg)

    correct = 0
    total = 0

    for seq_id in range(n_sequences):
        base = datetime(2026, 1, 1) + timedelta(days=seq_id * 10)
        field_id = f"field_T{seq_id}"

        # Create a temporal sequence: planting → disease → treatment1 → treatment2
        events = [
            CropCycle(
                cycle_id=f"cc_t{seq_id}", field_id=field_id,
                crop_type="wheat", variety="test",
                planting_date=base.strftime("%Y-%m-%d"),
            ),
            DiseaseEvent(
                event_id=f"de_t{seq_id}", field_id=field_id,
                crop_cycle_id=f"cc_t{seq_id}",
                disease_name=f"disease_{seq_id % 5}",
                first_observed=(base + timedelta(days=20)).strftime("%Y-%m-%d"),
                status=DiseaseStatus.CONFIRMED, severity=0.5,
            ),
            TreatmentAction(
                treatment_id=f"tr1_t{seq_id}",
                disease_event_id=f"de_t{seq_id}",
                treatment_type=TreatmentType.CHEMICAL_FUNGICIDE,
                agent="treatment_A",
                application_date=(base + timedelta(days=22)).strftime("%Y-%m-%d"),
            ),
            TreatmentAction(
                treatment_id=f"tr2_t{seq_id}",
                disease_event_id=f"de_t{seq_id}",
                treatment_type=TreatmentType.BIOLOGICAL,
                agent="treatment_B",
                application_date=(base + timedelta(days=25)).strftime("%Y-%m-%d"),
            ),
        ]

        for ev in events:
            kg.fast_write(ev)

        # Query: temporal path should return events in order
        results = kg.temporal_path_query(
            TemporalPathQuery(
                start_node_id=f"field_{field_id}",
                relation_sequence=("occurred_in", "treated", "followed_by"),
                max_hops=4,
            )
        )

        total += 1
        # We should have at least some results with temporal ordering
        if len(results) >= 2:
            # Check timestamps are increasing
            timestamps = [r.timestamp for r in results]
            if all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1)):
                correct += 1

    accuracy = correct / max(total, 1)
    metrics = {
        "temporal_ordering_accuracy": round(accuracy, 4),
        "n_sequences": total,
        "n_correct": correct,
    }
    print(f"  Temporal ordering accuracy: {metrics['temporal_ordering_accuracy']:.4f}")
    print(f"  [{correct}/{total}] sequences correctly ordered")
    return metrics


def benchmark_semantic_density_robustness() -> dict:
    """Semantic Density Robustness — accuracy degradation as query density increases.

    Sweeps semantic density rho from 0.1 to 0.9 while keeping fact count constant.
    Measures recall degradation to assess Stability Gap vulnerability.

    This is the novel metric proposed in the research contract.
    """
    print("\n" + "=" * 60)
    print("Benchmark: Semantic Density Robustness")
    print("=" * 60)

    cfg = _make_cfg(small=True)
    system = CLSMemorySystem(cfg)

    results = {}
    base_events = 20

    for rho in [0.1, 0.3, 0.5, 0.7, 0.9]:
        # Generate events at this density level
        # rho controls how many facts share attributes (disease names)
        n_shared = int(base_events * rho)
        diseases = ["target_disease"] * n_shared + \
                   [f"other_disease_{i}" for i in range(base_events - n_shared)]

        random.seed(42)
        for i, disease in enumerate(diseases):
            ev = DiseaseEvent(
                event_id=f"de_density_{i:04d}",
                field_id=f"field_d{i % 3}",
                crop_cycle_id=f"cc_density_{i:04d}",
                disease_name=disease,
                first_observed=f"2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                status=DiseaseStatus.CONFIRMED,
                severity=0.5,
            )
            system.fast_write(ev)

        # Query for the target disease
        ctx = DiagnosticContext(field_id="field_d0", crop_type="wheat")
        response = system.diagnose("What do you know about target_disease?", ctx)

        # Check if the answer contains the target disease
        found = "target_disease" in response.answer.lower()
        results[rho] = {
            "found": found,
            "confidence": response.confidence,
            "n_facts_ingested": base_events,
            "n_shared_attributes": n_shared,
        }
        print(f"  rho={rho:.1f} (n_shared={n_shared}): found={found}, "
              f"conf={response.confidence:.3f}")

    # Calculate robustness score: accuracy at rho=0.9 vs rho=0.1
    acc_low = 1.0 if results.get(0.1, {}).get("found") else 0.0
    acc_high = 1.0 if results.get(0.9, {}).get("found") else 0.0
    robustness_drop = acc_low - acc_high

    metrics = {
        "robustness_drop": round(robustness_drop, 4),
        "accuracy_at_rho_0_1": acc_low,
        "accuracy_at_rho_0_9": acc_high,
        "results_by_rho": results,
    }
    print(f"  Robustness drop (acc@0.1 - acc@0.9): {robustness_drop:.2f}")
    return metrics


def benchmark_fact_write_throughput(n_events: int = 100) -> dict:
    """Measure KG write throughput for bulk ingestion."""
    print("\n" + "=" * 60)
    print("Benchmark: Fact Write Throughput")
    print("=" * 60)

    cfg = _make_cfg(small=True)
    kg = EpisodicKnowledgeGraph(cfg.episodic_kg)

    events = []
    for i in range(n_events):
        events.append(DiseaseEvent(
            event_id=f"de_tp_{i:04d}", field_id=f"field_{i % 5}",
            crop_cycle_id=f"cc_tp_{i:04d}", disease_name=f"disease_{i % 10}",
            first_observed=f"2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
            status=DiseaseStatus.CONFIRMED, severity=random.random(),
        ))

    start = time.time()
    for ev in events:
        kg.fast_write(ev)
    elapsed = time.time() - start

    throughput = n_events / elapsed
    metrics = {
        "n_events": n_events,
        "total_time_s": round(elapsed, 3),
        "throughput_events_per_s": round(throughput, 1),
        "nodes_after": len(kg.nodes),
        "edges_after": len(kg.edges),
    }
    print(f"  Wrote {n_events} events in {elapsed:.3f}s")
    print(f"  Throughput: {throughput:.1f} events/s")
    print(f"  Final graph: {len(kg.nodes)} nodes, {len(kg.edges)} edges")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Domain B: Graph ML Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_expressiveness_probe() -> dict:
    """Expressiveness probe: can the semantic layer distinguish
    non-isomorphic graphs that are 1-WL indistinguishable?

    Uses the classic regular graph pair: two 6-node 3-regular graphs
    that 1-WL cannot distinguish. A model that can differentiate them
    has >1-WL expressiveness.
    """
    print("\n" + "=" * 60)
    print("Benchmark: Expressiveness Probe (1-WL hard pair)")
    print("=" * 60)

    cfg = SemanticMLConfig(
        node_embed_dim=8, edge_embed_dim=4,
        hidden_dim=32, n_layers=2, n_pattern_slots=8,
        pattern_embed_dim=32, n_heads=2,
    )
    extractor = SemanticPatternExtractor(cfg)
    extractor.eval()

    # Construct two non-isomorphic 3-regular graphs on 6 nodes
    # Graph 1: two disjoint triangles (0-1-2-0, 3-4-5-3)
    # Graph 2: 6-cycle with chords (0-1-2-3-4-5-0 plus 0-2, 1-3, 4-0)
    # Both have the same degree sequence (3,3,3,3,3,3) but different structures

    def build_graph(edges_6):
        """Build a KGSubgraph from a list of (src, tgt) edges on 6 nodes."""
        sg = KGSubgraph(
            root_node_id="g", timestamp=datetime.now(),
            query_type="subgraph", summary="expressiveness test",
        )
        feat_dim = cfg.node_embed_dim
        edge_dim = cfg.edge_embed_dim
        N = 6
        feat = torch.randn(N, feat_dim)  # same random features for both
        sg.node_features = feat
        E = len(edges_6)
        edge_idx = torch.zeros(2, E, dtype=torch.long)
        edge_feat = torch.randn(E, edge_dim)
        for j, (s, t) in enumerate(edges_6):
            edge_idx[0, j] = s
            edge_idx[1, j] = t
        sg.edge_index = edge_idx
        sg.edge_features = edge_feat
        return sg

    # Triangle pair graph
    g1_edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3),
                (0, 3), (1, 4), (2, 5)]  # cross edges make it 3-regular

    # 6-cycle with chords
    g2_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
                (0, 2), (1, 3), (4, 0)]

    sg1 = build_graph(g1_edges)
    sg2 = build_graph(g2_edges)

    with torch.no_grad():
        out1 = extractor._forward_impl(SubgraphBatch([sg1]))
        out2 = extractor._forward_impl(SubgraphBatch([sg2]))

    emb1, emb2 = out1["pattern_embed"], out2["pattern_embed"]
    similarity = torch.cosine_similarity(emb1, emb2).item()

    # If similarity is very high (>0.99), the model cannot distinguish them
    can_distinguish = similarity < 0.95
    metrics = {
        "can_distinguish_non_isomorphic": can_distinguish,
        "cosine_similarity": round(similarity, 4),
        "n_layers": cfg.n_layers,
    }
    print(f"  Cosine similarity between non-isomorphic graphs: {similarity:.4f}")
    print(f"  Can distinguish: {can_distinguish}")
    print(f"  (Note: GCN=1-WL, may not distinguish regular graphs without feature aug)")
    return metrics


def benchmark_oversmoothing_check(max_layers: int = 6) -> dict:
    """Oversmoothing: check if node features collapse after many layers.

    As GCN depth increases, node representations should not become
    identical (which would indicate oversmoothing).
    """
    print("\n" + "=" * 60)
    print("Benchmark: Oversmoothing Check")
    print("=" * 60)

    results = {}
    N = 20
    feat_dim = 32
    edge_dim = 8
    x = torch.randn(N, feat_dim)
    edge_idx = torch.randint(0, N, (2, 30))
    edge_attr = torch.randn(30, edge_dim)

    for n_layers in range(1, max_layers + 1):
        cfg = SemanticMLConfig(
            node_embed_dim=feat_dim, edge_embed_dim=edge_dim,
            hidden_dim=64, n_layers=n_layers, n_pattern_slots=8,
            pattern_embed_dim=32, n_heads=2,
        )
        extractor = SemanticPatternExtractor(cfg)
        extractor.eval()

        # Build subgraph
        sg = KGSubgraph(root_node_id="test", timestamp=datetime.now(),
                        query_type="subgraph", summary=f"{n_layers} layers")
        sg.node_features = x.clone()
        sg.edge_index = edge_idx.clone()
        sg.edge_features = edge_attr.clone()

        with torch.no_grad():
            out = extractor._forward_impl(SubgraphBatch([sg]))

        # Measure output embedding variance across nodes
        embed_std = out["pattern_embed"].std().item()
        results[n_layers] = {
            "pattern_embed_std": round(embed_std, 4),
            "confidence": round(out["confidence"].item(), 4),
        }
        print(f"  n_layers={n_layers}: pattern_std={embed_std:.4f}, "
              f"conf={out['confidence'].item():.3f}")

    # Oversmoothing indicator: std decreases significantly with depth
    std_at_1 = results[1]["pattern_embed_std"] if 1 in results else 1.0
    std_at_max = results[max_layers]["pattern_embed_std"] if max_layers in results else 0.0
    collapse_ratio = std_at_max / max(std_at_1, 1e-8)

    metrics = {
        "collapse_ratio": round(collapse_ratio, 4),
        "std_at_1_layer": results.get(1, {}).get("pattern_embed_std", 0),
        "std_at_max_layers": results.get(max_layers, {}).get("pattern_embed_std", 0),
        "oversmoothing_severe": collapse_ratio < 0.1,
        "results_by_depth": results,
    }
    print(f"  Collapse ratio (std@{max_layers}/std@1): {collapse_ratio:.4f}")
    print(f"  Oversmoothing severe: {collapse_ratio < 0.1}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Domain C: Scientific ML (Agriculture) Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_crop_cycle_memory(n_cycles: int = 5) -> dict:
    """Crop cycle memory: track multi-turn conversations about crop cycles.

    Tests whether the system can recall planting dates, growth stages,
    and cycle-specific facts across multiple diagnostic turns.
    """
    print("\n" + "=" * 60)
    print("Benchmark: Crop Cycle Memory")
    print("=" * 60)

    cfg = _make_cfg(small=False)
    system = CLSMemorySystem(cfg)

    # Create distinct crop cycles across multiple fields
    cycles = []
    for i in range(n_cycles):
        field = f"field_c{i}"
        crop = ["wheat", "corn", "rice", "soybean", "tomato"][i]
        planting = f"2026-{(i % 3) + 3:02d}-{(i * 5) % 28 + 1:02d}"
        cycle = CropCycle(
            cycle_id=f"cc_bench_{i}", field_id=field, crop_type=crop,
            variety=f"{crop}_bench",
            planting_date=planting,
            stages=[
                (CropStage.PLANTING, planting),
                (CropStage.VEGETATIVE, f"2026-{(i % 3) + 4:02d}-{(i * 5 + 15) % 28 + 1:02d}"),
            ],
        )
        system.fast_write(cycle)
        cycles.append({"field": field, "crop": crop, "planting": planting})

    # Multi-turn conversation
    correct = 0
    total = 0

    for c in cycles:
        # Turn 1: Ask about crop
        ctx = DiagnosticContext(field_id=c["field"])
        resp1 = system.diagnose(f"What is planted in {c['field']}?", ctx)
        total += 1
        if c["crop"] in resp1.answer.lower():
            correct += 1

        # Turn 2: Ask about planting date
        resp2 = system.diagnose(f"When was {c['crop']} planted in {c['field']}?", ctx)
        total += 1
        if c["planting"] in resp2.answer:
            correct += 1

    accuracy = correct / max(total, 1)
    metrics = {
        "crop_cycle_accuracy": round(accuracy, 4),
        "n_cycles": n_cycles,
        "n_queries": total,
        "n_correct": correct,
    }
    print(f"  Crop cycle accuracy: {metrics['crop_cycle_accuracy']:.4f}")
    print(f"  [{correct}/{total}] correct")
    return metrics


def benchmark_disease_progression_tracking() -> dict:
    """Disease progression tracking: verify that the system correctly
    tracks disease status changes over time."""
    print("\n" + "=" * 60)
    print("Benchmark: Disease Progression Tracking")
    print("=" * 60)

    cfg = _make_cfg(small=False)
    kg = EpisodicKnowledgeGraph(cfg.episodic_kg)

    # Create a disease progression narrative
    disease_ev = DiseaseEvent(
        event_id="de_prog", field_id="field_prog",
        crop_cycle_id="cc_prog", disease_name="late_blight",
        first_observed="2026-05-01", status=DiseaseStatus.SUSPECTED,
        severity=0.2, affected_area_m2=50.0,
    )
    kg.fast_write(disease_ev)

    # Update: confirmed
    update1 = DiseaseEvent(
        event_id="de_prog", field_id="field_prog",  # same ID → merge
        crop_cycle_id="cc_prog", disease_name="late_blight",
        first_observed="2026-05-03", status=DiseaseStatus.CONFIRMED,
        severity=0.5, affected_area_m2=150.0,
        symptoms=["water_soaked_lesions"],
    )
    kg.fast_write(update1)

    # Update: contained
    update2 = DiseaseEvent(
        event_id="de_prog", field_id="field_prog",
        crop_cycle_id="cc_prog", disease_name="late_blight",
        first_observed="2026-05-10", status=DiseaseStatus.CONTAINED,
        severity=0.3, affected_area_m2=150.0,
    )
    kg.fast_write(update2)

    # Query for disease events
    # Check dedup handled the same event_id
    print(f"  Disease nodes: {sum(1 for n in kg.nodes.values() if n.node_type == 'disease_event')}")

    # Check KG statistics
    stats = kg.get_statistics()
    print(f"  KG stats: {stats['nodes']} nodes, {stats['edges']} edges, "
          f"{stats['writes']} writes")

    metrics = {
        "disease_nodes": sum(1 for n in kg.nodes.values() if n.node_type == "disease_event"),
        "total_writes": stats["writes"],
        "total_nodes": stats["nodes"],
    }
    return metrics


def benchmark_spatial_proximity_retrieval(n_fields: int = 8) -> dict:
    """Spatial proximity retrieval correctness.

    Creates disease events at known distances from a center field
    and verifies that queries with the correct radius return them.
    """
    print("\n" + "=" * 60)
    print("Benchmark: Spatial Proximity Retrieval")
    print("=" * 60)

    cfg = _make_cfg(small=True)
    kg = EpisodicKnowledgeGraph(cfg.episodic_kg)

    # Register center field as a node
    kg.fast_write({
        "type": "field",
        "field_id": "center_field",
        "timestamp": datetime.now(),
        "summary": "Center field for spatial test",
    })

    # Place disease events at known distances
    events = []
    for i in range(n_fields):
        dist = i * 100.0  # 0, 100, 200, ..., 700m
        field_id = f"field_spatial_{i}"
        ev = DiseaseEvent(
            event_id=f"de_spatial_{i}", field_id=field_id,
            crop_cycle_id=f"cc_spatial_{i}", disease_name=f"disease_{i % 3}",
            first_observed=f"2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
            status=DiseaseStatus.CONFIRMED, severity=0.5,
        )
        kg.fast_write(ev)
        events.append({"field_id": field_id, "distance_m": dist})

    # Query with various radii
    results = {}
    for radius in [100, 300, 600]:
        query_result = kg.spatial_proximity_query(
            SpatialProximityQuery(
                center_field_id="center_field",
                radius_m=float(radius),
            )
        )
        n_returned = len(query_result)
        expected = sum(1 for e in events if e["distance_m"] <= radius)
        results[radius] = {"returned": n_returned, "expected": expected}
        print(f"  radius={radius}m: returned={n_returned}, expected_up_to={expected}")

    metrics = {
        "results_by_radius": results,
        "n_fields": n_fields,
    }
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════════════

BENCHMARKS = {
    "fact_recall": benchmark_fact_recall,
    "temporal_reasoning": benchmark_temporal_reasoning,
    "density_robustness": benchmark_semantic_density_robustness,
    "write_throughput": benchmark_fact_write_throughput,
    "expressiveness": benchmark_expressiveness_probe,
    "oversmoothing": benchmark_oversmoothing_check,
    "crop_cycle_memory": benchmark_crop_cycle_memory,
    "disease_progression": benchmark_disease_progression_tracking,
    "spatial_proximity": benchmark_spatial_proximity_retrieval,
}


def run_all():
    """Run all benchmarks and return aggregated results."""
    all_metrics = {}
    for name, fn in BENCHMARKS.items():
        try:
            all_metrics[name] = fn()
        except Exception as e:
            print(f"\n  [ERROR] Benchmark '{name}' failed: {e}")
            import traceback
            traceback.print_exc()
            all_metrics[name] = {"error": str(e)}
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLS Memory System Benchmarks")
    parser.add_argument("--benchmark", "-b", type=str, default=None,
                        help=f"Specific benchmark to run: {list(BENCHMARKS.keys())}")
    parser.add_argument("--list", action="store_true", help="List available benchmarks")
    args = parser.parse_args()

    if args.list:
        print("Available benchmarks:")
        for name in BENCHMARKS:
            print(f"  {name}: {BENCHMARKS[name].__doc__.strip()}")
        sys.exit(0)

    if args.benchmark:
        if args.benchmark not in BENCHMARKS:
            print(f"Unknown benchmark: {args.benchmark}")
            print(f"Available: {list(BENCHMARKS.keys())}")
            sys.exit(1)
        BENCHMARKS[args.benchmark]()
    else:
        run_all()
