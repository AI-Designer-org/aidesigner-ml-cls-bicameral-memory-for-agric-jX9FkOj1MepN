#!/usr/bin/env python3
"""
Layer 4 — Profiling Script for CLS Bicameral Memory System.

Measures memory usage, latency, and FLOP estimates for each subsystem:
    1. Episodic KG: write throughput, query latency, memory footprint
    2. Semantic ML Layer: forward pass time, parameter count, peak memory
    3. Agent Controller: diagnose end-to-end latency, iteration cost
    4. Full system: end-to-end benchmark

Usage:
    python profiling.py                         # Quick profile (CPU, small config)
    python profiling.py --device cuda           # Profile on GPU
    python profiling.py --detailed              # Detailed torch.profiler trace
    python profiling.py --subsystem semantic    # Profile a specific subsystem
"""

import sys
import os
import time
import math
import argparse
from datetime import datetime
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
)
from kg import EpisodicKnowledgeGraph
from semantic import SemanticPatternExtractor, SemanticMemoryManager
from controller import CLSAgentController
from model import CLSMemorySystem
from base import count_params


def build_sample_input(cfg: SemanticMLConfig, batch_size: int = 4, device: str = "cpu"):
    """Build a sample SubgraphBatch for profiling the semantic ML layer."""
    B, N, E = batch_size, 10, 8
    node_features = torch.randn(B, N, cfg.node_embed_dim, device=device)
    edge_index = torch.randint(0, N, (B, 2, E), device=device)
    edge_features = torch.randn(B, E, cfg.edge_embed_dim, device=device)
    node_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    edge_mask = torch.ones(B, E, dtype=torch.bool, device=device)

    # Build SubgraphBatch
    subgraphs = []
    for b in range(B):
        sg = KGSubgraph(
            root_node_id=f"prof_{b}", timestamp=datetime.now(),
            query_type="subgraph", summary=f"Profile subgraph {b}",
        )
        sg.node_features = node_features[b].cpu()
        sg.edge_index = edge_index[b].cpu()
        sg.edge_features = edge_features[b].cpu()
        subgraphs.append(sg)

    return SubgraphBatch(subgraphs)


# ═══════════════════════════════════════════════════════════════════════════════
# Parameter & FLOP Counting
# ═══════════════════════════════════════════════════════════════════════════════

def profile_semantic_ml(cfg: Optional[SemanticMLConfig] = None, device: str = "cpu",
                         use_torch_profiler: bool = False, steps: int = 20):
    """Profile the SemanticPatternExtractor: params, FLOPs, latency, memory."""
    print("\n" + "=" * 60)
    print("Profiling: SemanticPatternExtractor")
    print("=" * 60)

    if cfg is None:
        cfg = SemanticMLConfig(hidden_dim=256, n_layers=3, n_pattern_slots=64,
                               pattern_embed_dim=128, n_heads=4, d_ff=1024)

    extractor = SemanticPatternExtractor(cfg).to(device)
    extractor.eval()
    count_params(extractor)

    # Estimate FLOPs per forward pass
    total_params = sum(p.numel() for p in extractor.parameters())
    # Rough FLOP estimate: ~2 * params per forward (multiply-add),
    # plus attention O(B * N * S * P) for prototype attention
    B, N, S, P = 4, 10, cfg.n_pattern_slots, cfg.pattern_embed_dim
    gcn_flops = 2 * total_params  # Approximate
    attn_flops = 2 * B * 1 * S * P  # Query × prototype dot products
    total_flops = gcn_flops + attn_flops
    print(f"  Est. forward FLOPs: {total_flops / 1e6:.1f}M")

    # Warmup
    batch = build_sample_input(cfg, batch_size=B, device=device)
    for _ in range(5):
        with torch.no_grad():
            extractor._forward_impl(batch)

    # Latency benchmark
    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(steps):
            with torch.no_grad():
                extractor._forward_impl(batch)
        end_event.record()
        torch.cuda.synchronize()
        avg_ms = start_event.elapsed_time(end_event) / steps
    else:
        start = time.time()
        for _ in range(steps):
            with torch.no_grad():
                extractor._forward_impl(batch)
        avg_ms = (time.time() - start) * 1000 / steps

    print(f"  Avg forward time: {avg_ms:.2f}ms (over {steps} runs)")

    # Peak memory (if CUDA)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            out = extractor._forward_impl(batch)
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"  Peak CUDA memory: {peak_mb:.1f}MB")

    # Torch profiler
    if use_torch_profiler and device == "cuda":
        from torch.profiler import profile, ProfilerActivity
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            for _ in range(10):
                with torch.no_grad():
                    extractor._forward_impl(batch)

        print("\n  Profiler top operations by CUDA time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    return {
        "subsystem": "semantic_ml",
        "params": total_params,
        "flops_millions": round(total_flops / 1e6, 1),
        "avg_forward_ms": round(avg_ms, 2),
        "hidden_dim": cfg.hidden_dim,
        "n_layers": cfg.n_layers,
        "n_pattern_slots": cfg.n_pattern_slots,
    }


def profile_episodic_kg(cfg: Optional[EpisodicKGConfig] = None,
                         n_events: int = 500):
    """Profile the EpisodicKnowledgeGraph: write throughput, memory."""
    print("\n" + "=" * 60)
    print("Profiling: EpisodicKnowledgeGraph")
    print("=" * 60)

    if cfg is None:
        cfg = EpisodicKGConfig()

    kg = EpisodicKnowledgeGraph(cfg)

    # Generate synthetic events
    events = []
    for i in range(n_events):
        events.append(DiseaseEvent(
            event_id=f"de_prof_{i:04d}", field_id=f"field_{i % 10}",
            crop_cycle_id=f"cc_prof_{i:04d}", disease_name=f"disease_{i % 20}",
            first_observed=f"2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
            status=DiseaseStatus.CONFIRMED, severity=0.5,
        ))

    # Write throughput
    start = time.time()
    for ev in events:
        kg.fast_write(ev)
    write_time = time.time() - start

    # Query latency
    query = TemporalPathQuery(
        start_node_id="field_field_0",
        relation_sequence=("occurred_in", "treated"),
        max_hops=3,
    )
    start = time.time()
    n_queries = 100
    for _ in range(n_queries):
        kg.temporal_path_query(query)
    query_time = (time.time() - start) / n_queries

    # Memory estimate (approximate by string lengths)
    node_mem = sum(sys.getsizeof(n) for n in kg.nodes.values())
    edge_mem = sum(sys.getsizeof(e) for e in kg.edges.values())
    total_mem_bytes = node_mem + edge_mem

    stats = kg.get_statistics()
    print(f"  Wrote {n_events} events in {write_time:.3f}s "
          f"({n_events / write_time:.0f} ev/s)")
    print(f"  Avg query time: {query_time * 1000:.2f}ms")
    print(f"  Graph: {stats['nodes']} nodes, {stats['edges']} edges")
    print(f"  Est. memory: {total_mem_bytes / 1024:.1f}KB")

    return {
        "subsystem": "episodic_kg",
        "n_events": n_events,
        "write_throughput_ev_s": round(n_events / write_time, 1),
        "avg_query_ms": round(query_time * 1000, 2),
        "n_nodes": stats["nodes"],
        "n_edges": stats["edges"],
        "est_memory_kb": round(total_mem_bytes / 1024, 1),
    }


def profile_cls_end_to_end(cfg: Optional[CLSMemorySystemConfig] = None,
                           n_diagnoses: int = 20):
    """Profile the full CLSMemorySystem end-to-end."""
    print("\n" + "=" * 60)
    print("Profiling: CLSMemorySystem (End-to-End)")
    print("=" * 60)

    if cfg is None:
        cfg = CLSMemorySystemConfig(
            episodic_kg=EpisodicKGConfig(max_triples=50000, max_nodes=10000,
                                          node_embed_dim=32, edge_embed_dim=8),
            semantic_ml=SemanticMLConfig(hidden_dim=64, n_layers=2,
                                          n_pattern_slots=16, pattern_embed_dim=32,
                                          node_embed_dim=32, edge_embed_dim=8),
            agent_controller=AgentControllerConfig(max_iterative_cycles=2,
                                                    reconciliation_method="confidence_max"),
        )

    system = CLSMemorySystem(cfg)

    # Pre-populate KG
    for i in range(30):
        ev = DiseaseEvent(
            event_id=f"de_e2e_{i:04d}", field_id=f"field_{i % 3}",
            crop_cycle_id=f"cc_e2e_{i:04d}", disease_name=f"disease_{i % 5}",
            first_observed=f"2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
            status=DiseaseStatus.CONFIRMED, severity=0.5,
        )
        system.fast_write(ev)

    contexts = [
        DiagnosticContext(field_id=f"field_{i % 3}", crop_type="wheat",
                          season_start=datetime(2026, 1, 1),
                          season_end=datetime(2026, 12, 31))
        for i in range(n_diagnoses)
    ]
    queries = [f"What diseases were found in field_{i % 3}?" for i in range(n_diagnoses)]

    # Warmup
    system.diagnose("warmup", contexts[0])

    # Benchmark
    latencies = []
    for i in range(n_diagnoses):
        start = time.time()
        resp = system.diagnose(queries[i], contexts[i])
        latencies.append(time.time() - start)

    avg_lat = sum(latencies) / len(latencies)
    p99_lat = sorted(latencies)[int(len(latencies) * 0.99)]
    min_lat = min(latencies)
    max_lat = max(latencies)

    state = system.get_system_state()
    params = sum(p.numel() for p in system.semantic_memory.parameters())

    print(f"  Diagnoses: {n_diagnoses}")
    print(f"  Avg latency: {avg_lat * 1000:.0f}ms")
    print(f"  P99 latency: {p99_lat * 1000:.0f}ms")
    print(f"  Min/Max: {min_lat * 1000:.0f}ms / {max_lat * 1000:.0f}ms")
    print(f"  Semantic params: {params:,}")
    print(f"  KG state: {state['statistics']['kg_node_count']} nodes")

    return {
        "subsystem": "cls_end_to_end",
        "n_diagnoses": n_diagnoses,
        "avg_latency_ms": round(avg_lat * 1000, 1),
        "p99_latency_ms": round(p99_lat * 1000, 1),
        "min_latency_ms": round(min_lat * 1000, 1),
        "max_latency_ms": round(max_lat * 1000, 1),
        "semantic_params": params,
    }


def profile_param_scaling():
    """Profile how parameter count scales with config choices."""
    print("\n" + "=" * 60)
    print("Scaling: Parameter Count by Config")
    print("=" * 60)

    configs = [
        ("tiny",   SemanticMLConfig(hidden_dim=32, n_layers=1, n_pattern_slots=4,
                                    pattern_embed_dim=16, n_heads=1, d_ff=64)),
        ("small",  SemanticMLConfig(hidden_dim=64, n_layers=2, n_pattern_slots=8,
                                    pattern_embed_dim=32, n_heads=2, d_ff=128)),
        ("medium", SemanticMLConfig(hidden_dim=128, n_layers=3, n_pattern_slots=32,
                                    pattern_embed_dim=64, n_heads=4, d_ff=256)),
        ("large",  SemanticMLConfig(hidden_dim=256, n_layers=3, n_pattern_slots=64,
                                    pattern_embed_dim=128, n_heads=4, d_ff=512)),
    ]

    for name, cfg in configs:
        extractor = SemanticPatternExtractor(cfg)
        total = sum(p.numel() for p in extractor.parameters())
        trainable = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
        print(f"  {name:<8} total={total:>8,}  trainable={trainable:>8,}  "
              f"config=({cfg.hidden_dim}hd, {cfg.n_layers}L, {cfg.n_pattern_slots}slots)")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLS Memory System Profiling")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--detailed", action="store_true", help="Use torch.profiler")
    parser.add_argument("--subsystem", type=str, default=None,
                        choices=["semantic", "kg", "end_to_end", "scaling"])
    args = parser.parse_args()

    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")

    print("=" * 60)
    print("CLS Bicameral Memory System — Profiling Suite")
    print(f"Device: {device}  |  PyTorch: {torch.__version__}")
    print("=" * 60)

    if args.subsystem == "semantic":
        profile_semantic_ml(device=device, use_torch_profiler=args.detailed)
    elif args.subsystem == "kg":
        profile_episodic_kg()
    elif args.subsystem == "end_to_end":
        profile_cls_end_to_end()
    elif args.subsystem == "scaling":
        profile_param_scaling()
    else:
        # Run all
        profile_semantic_ml(device=device, use_torch_profiler=args.detailed)
        profile_episodic_kg()
        profile_cls_end_to_end()
        profile_param_scaling()

    print("\nProfiling complete.")
