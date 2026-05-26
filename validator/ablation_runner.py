#!/usr/bin/env python3
"""
Layer 3 — Ablation Runner for CLS Bicameral Memory System.

Runs single-field config changes (ablations) from the architecture blueprint §12
and compares results against the baseline configuration.

Each ablation is a single ModelConfig field change tied to a specific hypothesis
from the research contract.

Ablation catalog (from architecture blueprint §12):
    1. No iterative querying   (max_iterative_cycles: 5 → 1)
    2. No semantic ML layer    (remove subsystem — kg_only mode)
    3. Generic KG (no typed objects)   (enable_crop_cycle_objects: True → False)
    4. LLM → confidence_max reconciliation   (method: "llm_judge" → "confidence_max")
    5. Parallel → sequential initial query   (parallel_initial_query: True → False)
    6. Consolidation frequency sweep          (1440 → 60/360/4320 min)
    7. Prototype slot count sweep             (64 → 8/16/32/128)
    8. GCN → MLP encoder                      (encoder_type: "gcn" → "mlp")

Usage:
    python ablation_runner.py                        # Run primary ablations (1-5)
    python ablation_runner.py --all                  # Run all ablations (1-8)
    python ablation_runner.py --sweep                # Run hyperparameter sweeps (6-8)
    python ablation_runner.py --ablation no_iterate  # Run single ablation

Output:
    CSV results printed to stdout
    JSON results written to research_eval/ablation_results.json
"""

import sys
import os
import json
import time
import math
import argparse
from copy import deepcopy
from datetime import datetime
from dataclasses import replace
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
from model import CLSMemorySystem

# Results directory
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "research_eval")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Standard Evaluation Setup
# ═══════════════════════════════════════════════════════════════════════════════

BASE_CONFIG = CLSMemorySystemConfig(
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
        max_iterative_cycles=3,
        early_exit_confidence=0.9,
        reconciliation_method="confidence_max",
        provenance_tracking=True,
        parallel_initial_query=True,
    ),
)


def _make_eval_data(n_events: int = 20):
    """Create standard evaluation events + queries."""
    events = []
    for i in range(n_events):
        field = f"field_{chr(65 + i % 3)}"
        disease = ["powdery_mildew", "rust", "blight", "leaf_spot"][i % 4]
        events.append(DiseaseEvent(
            event_id=f"de_eval_{i:04d}", field_id=field,
            crop_cycle_id=f"cc_eval_{i:04d}", disease_name=disease,
            first_observed=f"2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
            status=DiseaseStatus.CONFIRMED, severity=0.5,
        ))
    queries = [
        (f"What happened in field_{chr(65 + i % 3)}?",
         DiagnosticContext(field_id=f"field_{chr(65 + i % 3)}"),
         ["powdery_mildew", "rust", "blight", "leaf_spot"][i % 4])
        for i in range(min(8, n_events))
    ]
    return events, queries


def evaluate_system(system: CLSMemorySystem, events: list, queries: list) -> dict:
    """Run a standard evaluation and return metrics."""
    # Ingest events
    for ev in events:
        system.fast_write(ev)

    # Run queries
    correct = 0
    total_latency = 0.0
    n_queries = len(queries)

    for q_str, ctx, expected_disease in queries:
        start = time.time()
        response = system.diagnose(q_str, ctx)
        total_latency += time.time() - start

        found = expected_disease.lower() in response.answer.lower()
        if found:
            correct += 1

    metrics = {
        "accuracy": correct / max(n_queries, 1),
        "correct": correct,
        "total_queries": n_queries,
        "avg_latency_s": total_latency / max(n_queries, 1),
        "events_ingested": len(events),
    }
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Ablation Definitions
# ═══════════════════════════════════════════════════════════════════════════════

ABLATIONS = {
    # ── Primary ablations (1-5) ──

    "baseline": {
        "description": "Full architecture with default config",
        "setup": lambda cfg: cfg,
        "config_type": "config",
    },

    "no_iterate": {
        "description": "No iterative querying (max_iterative_cycles=1, one-directional)",
        "setup": lambda cfg: replace(
            cfg,
            agent_controller=replace(
                cfg.agent_controller,
                max_iterative_cycles=1,
                enable_semantic_prior_routing=False,
                enable_episodic_revision=False,
            )
        ),
        "config_type": "config",
        "hypothesis": "Iterative bidirectional querying improves diagnostic accuracy over one-directional consolidation",
        "expected_change": "Accuracy drops 5-10% on counterfactuals; latency drops 60%",
    },

    "kg_only": {
        "description": "No semantic ML layer (episodic KG only)",
        "setup": lambda system: system.set_ablation_mode("kg_only"),
        "config_type": "runtime",
        "hypothesis": "Semantic ML layer provides generalization that KG alone cannot",
        "expected_change": "Generalization accuracy drops >15%; fact recall unchanged",
    },

    "generic_kg": {
        "description": "Generic KG without domain-specific typed objects",
        "setup": lambda system: system.set_ablation_mode("generic_kg"),
        "config_type": "runtime",
        "hypothesis": "Domain-specific typed objects improve temporal/spatial reasoning",
        "expected_change": "Temporal accuracy drops 5-10%",
    },

    "confidence_max": {
        "description": "LLM reconciliation → confidence_max reconciliation",
        "setup": lambda cfg: replace(
            cfg,
            agent_controller=replace(
                cfg.agent_controller,
                reconciliation_method="confidence_max",
            )
        ),
        "config_type": "config",
        "hypothesis": "LLM-based reconciliation improves over simple confidence comparison",
        "expected_change": "Counterfactual accuracy drops 5-10%; latency drops 40%",
    },

    "sequential_query": {
        "description": "Parallel → sequential initial query",
        "setup": lambda cfg: replace(
            cfg,
            agent_controller=replace(
                cfg.agent_controller,
                parallel_initial_query=False,
            )
        ),
        "config_type": "config",
        "hypothesis": "Parallel initial query reduces latency",
        "expected_change": "Latency increases ~1 round-trip; accuracy unchanged",
    },

    # ── Hyperparameter sweeps (6-8) ──

    "freq_60min": {
        "description": "Consolidation frequency = 60 minutes",
        "setup": lambda cfg: replace(
            cfg,
            semantic_ml=replace(
                cfg.semantic_ml,
                consolidation_frequency_minutes=60,
            )
        ),
        "config_type": "config",
        "hypothesis": "More frequent consolidation improves accuracy",
    },

    "freq_360min": {
        "description": "Consolidation frequency = 360 minutes (6 hours)",
        "setup": lambda cfg: replace(
            cfg,
            semantic_ml=replace(
                cfg.semantic_ml,
                consolidation_frequency_minutes=360,
            )
        ),
        "config_type": "config",
    },

    "freq_4320min": {
        "description": "Consolidation frequency = 4320 minutes (3 days)",
        "setup": lambda cfg: replace(
            cfg,
            semantic_ml=replace(
                cfg.semantic_ml,
                consolidation_frequency_minutes=4320,
            )
        ),
        "config_type": "config",
    },

    "slots_8": {
        "description": "Prototype slots = 8",
        "setup": lambda cfg: replace(
            cfg,
            semantic_ml=replace(
                cfg.semantic_ml,
                n_pattern_slots=8,
            )
        ),
        "config_type": "config",
        "hypothesis": "Fewer prototype slots may underfit agricultural patterns",
    },

    "slots_32": {
        "description": "Prototype slots = 32",
        "setup": lambda cfg: replace(
            cfg,
            semantic_ml=replace(
                cfg.semantic_ml,
                n_pattern_slots=32,
            )
        ),
        "config_type": "config",
    },

    "slots_128": {
        "description": "Prototype slots = 128",
        "setup": lambda cfg: replace(
            cfg,
            semantic_ml=replace(
                cfg.semantic_ml,
                n_pattern_slots=128,
            )
        ),
        "config_type": "config",
        "hypothesis": "More prototype slots may overfit; may improve capacity",
    },

    "gcn_mlp": {
        "description": "GCN → MLP encoder (no graph structure awareness)",
        "setup": lambda cfg: replace(
            cfg,
            semantic_ml=replace(
                cfg.semantic_ml,
                encoder_type="mlp",
            )
        ),
        "config_type": "config",
        "hypothesis": "GCN's graph structure awareness improves semantic pattern extraction",
        "expected_change": "Pattern accuracy drops 10-15% on structurally complex queries",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_ablation(
    name: str,
    spec: dict,
    events: list,
    queries: list,
) -> dict:
    """Run a single ablation and return metrics."""
    print(f"\n  [{name}] {spec.get('description', '')}")
    print(f"         Hypothesis: {spec.get('hypothesis', 'N/A')}")

    cfg = deepcopy(BASE_CONFIG)

    # Apply config-level changes
    if spec.get("config_type") == "config":
        cfg = spec["setup"](cfg)

    system = CLSMemorySystem(cfg)

    # Apply runtime-level changes
    if spec.get("config_type") == "runtime":
        spec["setup"](system)

    # Evaluate
    metrics = evaluate_system(system, events, queries)
    print(f"         Accuracy: {metrics['accuracy']:.3f} "
          f"({metrics['correct']}/{metrics['total_queries']}), "
          f"Latency: {metrics['avg_latency_s']:.3f}s")

    return {
        "name": name,
        "description": spec.get("description", ""),
        "hypothesis": spec.get("hypothesis", ""),
        "expected_change": spec.get("expected_change", ""),
        "metrics": metrics,
    }


def run_ablations(ablation_names: Optional[list[str]] = None) -> list[dict]:
    """Run specified ablations (or all if None)."""
    events, queries = _make_eval_data(n_events=20)

    if ablation_names:
        names = [n for n in ablation_names if n in ABLATIONS]
    else:
        names = list(ABLATIONS.keys())

    results = []
    for name in names:
        try:
            result = run_ablation(name, ABLATIONS[name], events, queries)
            results.append(result)
        except Exception as e:
            print(f"  [ERROR] Ablation '{name}' failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "name": name,
                "error": str(e),
            })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(results: list[dict]):
    """Print formatted results table."""
    print("\n" + "=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)
    print(f"{'Ablation':<20} {'Accuracy':<10} {'Correct':<10} {'Latency(s)':<12} {'Expected Change'}")
    print("-" * 70)

    baseline_acc = None
    for r in results:
        if r["name"] == "baseline":
            m = r.get("metrics", {})
            baseline_acc = m.get("accuracy", 0)
            break

    for r in results:
        m = r.get("metrics", {})
        acc = m.get("accuracy", 0)
        correct = f"{m.get('correct', 0)}/{m.get('total_queries', 0)}"
        latency = m.get("avg_latency_s", 0)
        expected = r.get("expected_change", "")

        # Show delta from baseline
        if baseline_acc is not None and r["name"] != "baseline":
            delta = acc - baseline_acc
            acc_str = f"{acc:.3f} ({delta:+.3f})"
        else:
            acc_str = f"{acc:.3f}"

        print(f"{r['name']:<20} {acc_str:<10} {correct:<10} {latency:.3f}s      {expected[:40]}")


def save_results(results: list[dict]):
    """Save results to JSON."""
    path = os.path.join(RESULTS_DIR, "ablation_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

PRIMARY_ABLATIONS = ["baseline", "no_iterate", "kg_only", "generic_kg",
                     "confidence_max", "sequential_query"]
SWEEP_ABLATIONS = ["freq_60min", "freq_360min", "freq_4320min",
                   "slots_8", "slots_32", "slots_128",
                   "gcn_mlp"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLS Memory System Ablation Runner")
    parser.add_argument("--all", action="store_true", help="Run all ablations including sweeps")
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweeps only (6-8)")
    parser.add_argument("--ablation", type=str, default=None,
                        help=f"Run specific ablation. Options: {list(ABLATIONS.keys())}")
    parser.add_argument("--list", action="store_true", help="List available ablations")
    args = parser.parse_args()

    if args.list:
        print("Available ablations:")
        for name, spec in ABLATIONS.items():
            print(f"  {name}: {spec['description']}")
        sys.exit(0)

    if args.ablation:
        names = [args.ablation]
    elif args.sweep:
        names = SWEEP_ABLATIONS
    elif args.all:
        names = list(ABLATIONS.keys())
    else:
        names = PRIMARY_ABLATIONS

    print("=" * 70)
    print("CLS Bicameral Memory System — Ablation Runner")
    print(f"Running {len(names)} ablation(s): {names}")
    print("=" * 70)

    results = run_ablations(names)
    print_report(results)
    save_results(results)
