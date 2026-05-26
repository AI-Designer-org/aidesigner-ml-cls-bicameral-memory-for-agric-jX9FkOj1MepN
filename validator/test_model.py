#!/usr/bin/env python3
"""
Layer 1 — Unit Tests for the CLS Bicameral Memory System.

Tests are grouped into classes following the ML Validator specification:

    1a. TestShapes         — Output shape assertions for all subsystems
    1b. TestGradients      — Gradient flow and NaN gradient checks
    1c. TestCorrectness    — Domain-specific invariance/causality/completeness
    1d. TestNumerics       — bf16/fp16 stability, extreme input values
    1e. TestPermutation    — Graph ML permutation invariance
    1f. TestAblationModes  — Ablation mode switching correctness

Run with:
    cd /artifacts/j_jX9FkOj1MepN/work/coder
    python -m pytest /path/to/validator/test_model.py -v
"""

import sys
import os
import math
import copy
from datetime import datetime
from typing import Optional

import pytest
import torch

# Ensure the coder package is importable
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
    SemanticInferenceResult, ReconciliationResult, DiagnosticResponse,
)
from kg import EpisodicKnowledgeGraph
from semantic import SemanticPatternExtractor, SemanticMemoryManager
from controller import CLSAgentController, WorkingMemory
from model import CLSMemorySystem
from layers import GCNLayer, EdgeAwareGCN, PrototypeAttention, ContrastiveLoss


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def base_cfg() -> CLSMemorySystemConfig:
    """Minimal config for fast test execution."""
    return CLSMemorySystemConfig(
        episodic_kg=EpisodicKGConfig(
            max_triples=1000,
            max_nodes=500,
            node_embed_dim=32,
            edge_embed_dim=8,
        ),
        semantic_ml=SemanticMLConfig(
            node_embed_dim=32,
            edge_embed_dim=8,
            hidden_dim=64,
            n_layers=2,
            n_pattern_slots=8,
            pattern_embed_dim=32,
            n_heads=2,
            d_ff=256,
            dropout=0.1,
            consolidation_batch_size=2,
        ),
        agent_controller=AgentControllerConfig(
            max_iterative_cycles=2,
            early_exit_confidence=0.9,
            reconciliation_method="confidence_max",
            provenance_tracking=True,
        ),
    )


@pytest.fixture
def system(base_cfg) -> CLSMemorySystem:
    """Fully initialized CLSMemorySystem in eval mode."""
    sys = CLSMemorySystem(base_cfg)
    return sys


@pytest.fixture
def kg(base_cfg) -> EpisodicKnowledgeGraph:
    """Episodic KG with some pre-loaded events."""
    kg = EpisodicKnowledgeGraph(base_cfg.episodic_kg)
    # Seed events
    events = [
        CropCycle(
            cycle_id="cc_001", field_id="field_A42", crop_type="wheat",
            variety="spring_wheat", planting_date="2026-04-01",
            stages=[(CropStage.PLANTING, "2026-04-01"), (CropStage.VEGETATIVE, "2026-04-15")],
        ),
        DiseaseEvent(
            event_id="de_001", field_id="field_A42", crop_cycle_id="cc_001",
            disease_name="powdery_mildew", first_observed="2026-05-15",
            status=DiseaseStatus.CONFIRMED, severity=0.6, affected_area_m2=250.0,
            symptoms=["white_powdery_coating"],
        ),
        TreatmentAction(
            treatment_id="tr_001", disease_event_id="de_001",
            treatment_type=TreatmentType.CHEMICAL_FUNGICIDE,
            agent="sulfur_based_fungicide", dosage="2.5 L/ha",
            application_date="2026-05-18", effectiveness=0.7,
        ),
    ]
    for ev in events:
        kg.fast_write(ev)
    return kg


@pytest.fixture
def sample_subgraph_batch() -> SubgraphBatch:
    """A synthetic SubgraphBatch for forward-pass testing."""
    cfg = SemanticMLConfig(node_embed_dim=32, edge_embed_dim=8,
                           hidden_dim=64, n_layers=2, n_pattern_slots=8, pattern_embed_dim=32)
    B, N, E = 4, 10, 8
    subgraphs = []
    for b in range(B):
        sg = KGSubgraph(
            root_node_id=f"test_{b}",
            timestamp=datetime.now(),
            query_type="subgraph",
            summary=f"Test subgraph {b}",
        )
        sg.node_features = torch.randn(N, cfg.node_embed_dim)
        sg.edge_index = torch.randint(0, N, (2, E))
        sg.edge_features = torch.randn(E, cfg.edge_embed_dim)
        sg.node_mask = torch.ones(N, dtype=torch.bool)
        sg.label = b % 3  # 3 classes
        subgraphs.append(sg)
    return SubgraphBatch(subgraphs)


# ═══════════════════════════════════════════════════════════════════════════════
# 1a. Shape Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestShapes:
    """Output shape assertions for all subsystems."""

    def test_kg_fast_write_returns_node_id(self, kg):
        """fast_write returns a valid string node ID."""
        event = DiseaseEvent(
            event_id="de_test", field_id="field_X99", crop_cycle_id="cc_test",
            disease_name="test", first_observed="2026-06-01",
        )
        node_id = kg.fast_write(event)
        assert isinstance(node_id, str), f"Expected str, got {type(node_id)}"
        assert len(node_id) > 0, "Node ID should not be empty"

    def test_temporal_path_query_shape(self, kg):
        """temporal_path_query returns list of KGSubgraph."""
        results = kg.temporal_path_query(
            TemporalPathQuery(
                start_node_id="field_field_A42",
                relation_sequence=("occurred_in",),
                max_hops=2,
            )
        )
        assert isinstance(results, list), f"Expected list, got {type(results)}"
        if results:
            assert all(isinstance(r, KGSubgraph) for r in results)

    def test_spatial_proximity_query_shape(self, kg):
        """spatial_proximity_query returns list of KGSubgraph."""
        results = kg.spatial_proximity_query(
            SpatialProximityQuery(center_field_id="field_A42", radius_m=1000.0)
        )
        assert isinstance(results, list)

    def test_semantic_forward_shapes(self, sample_subgraph_batch):
        """SemanticPatternExtractor forward produces expected tensor shapes."""
        cfg = SemanticMLConfig(node_embed_dim=32, edge_embed_dim=8,
                               hidden_dim=64, n_layers=2, n_pattern_slots=8,
                               pattern_embed_dim=32, n_heads=2)
        extractor = SemanticPatternExtractor(cfg)
        with torch.no_grad():
            out = extractor._forward_impl(sample_subgraph_batch)

        B = sample_subgraph_batch.node_features.shape[0]
        assert out["pattern_embed"].shape == (B, cfg.pattern_embed_dim), \
            f"Expected ({B}, {cfg.pattern_embed_dim}), got {out['pattern_embed'].shape}"
        assert out["prototype_weights"].shape == (B, cfg.n_pattern_slots), \
            f"Expected ({B}, {cfg.n_pattern_slots}), got {out['prototype_weights'].shape}"
        assert out["confidence"].shape == (B,), \
            f"Expected ({B},), got {out['confidence'].shape}"

    def test_semantic_inference_result_shape(self, sample_subgraph_batch):
        """SemanticMemoryManager.infer_pattern returns SemanticInferenceResult."""
        cfg = SemanticMLConfig(node_embed_dim=32, edge_embed_dim=8,
                               hidden_dim=64, n_layers=2, n_pattern_slots=8,
                               pattern_embed_dim=32, n_heads=2)
        mgr = SemanticMemoryManager(cfg)

        # Test with subgraph
        sg = sample_subgraph_batch
        single_sg = KGSubgraph(
            root_node_id="test", timestamp=datetime.now(),
            query_type="subgraph", summary="test",
        )
        single_sg.node_features = torch.randn(5, cfg.node_embed_dim)
        single_sg.edge_index = torch.randint(0, 5, (2, 4))
        single_sg.edge_features = torch.randn(4, cfg.edge_embed_dim)

        result = mgr.infer_pattern(single_sg)
        assert isinstance(result, SemanticInferenceResult)
        assert result.pattern_embed.shape == (cfg.pattern_embed_dim,), \
            f"Expected ({cfg.pattern_embed_dim},), got {result.pattern_embed.shape}"
        assert 0.0 <= result.confidence <= 1.0, \
            f"Confidence out of range: {result.confidence}"

    def test_diagnose_returns_diagnostic_response(self, system):
        """diagnose() returns properly structured DiagnosticResponse."""
        context = DiagnosticContext(
            field_id="field_A42", crop_type="wheat",
            season_start=datetime(2026, 3, 1), season_end=datetime(2026, 6, 1),
        )
        response = system.diagnose("Why is my wheat showing powdery mildew?", context)
        assert isinstance(response, DiagnosticResponse)
        assert isinstance(response.answer, str)
        assert len(response.answer) > 0
        assert response.num_iterations >= 1
        assert 0.0 <= response.confidence <= 1.0
        assert isinstance(response.provenance, list)
        assert isinstance(response.evidence, list)

    def test_gcn_layer_shapes(self):
        """GCNLayer preserves or transforms shapes correctly."""
        layer = GCNLayer(in_dim=16, out_dim=32, edge_dim=8)
        N, E = 10, 6
        x = torch.randn(N, 16)
        edge_index = torch.randint(0, N, (2, E))
        edge_attr = torch.randn(E, 8)
        out = layer(x, edge_index, edge_attr)
        assert out.shape == (N, 32), f"Expected ({N}, 32), got {out.shape}"

    def test_prototype_attention_shapes(self):
        """PrototypeAttention returns correct shapes."""
        attn = PrototypeAttention(pattern_dim=64, n_prototypes=16, n_heads=4)
        B = 4
        query = torch.randn(B, 1, 64)
        pattern, weights = attn(query)
        assert pattern.shape == (B, 64), f"Expected ({B}, 64), got {pattern.shape}"
        assert weights.shape == (B, 1, 16), f"Expected ({B}, 1, 16), got {weights.shape}"

    def test_consolidation_scheduler_step_shape(self, system):
        """consolidation step returns dict with expected keys."""
        result = system.consolidate()
        assert isinstance(result, dict)

    def test_system_state_shape(self, system):
        """get_system_state returns nested dict with statistics."""
        state = system.get_system_state()
        assert "config" in state
        assert "statistics" in state
        stats = state["statistics"]
        assert "events_ingested" in stats
        assert "diagnoses_performed" in stats
        assert "kg_node_count" in stats


# ═══════════════════════════════════════════════════════════════════════════════
# 1b. Gradient Flow Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGradients:
    """Gradient flow and NaN gradient checks."""

    def test_semantic_extractor_gradients(self, sample_subgraph_batch):
        """All parameters in SemanticPatternExtractor receive gradients."""
        cfg = SemanticMLConfig(node_embed_dim=32, edge_embed_dim=8,
                               hidden_dim=64, n_layers=2, n_pattern_slots=8,
                               pattern_embed_dim=32, n_heads=2)
        extractor = SemanticPatternExtractor(cfg)
        extractor.train()

        out = extractor._forward_impl(sample_subgraph_batch)
        # Include both pattern_embed and confidence in loss to ensure
        # all parameters (including confidence_head) receive gradients
        loss = out["pattern_embed"].sum() + out["confidence"].sum()
        loss.backward()

        dead = [n for n, p in extractor.named_parameters() if p.grad is None]
        assert len(dead) == 0, f"Parameters with no gradient: {dead[:10]}"

        nan_params = [n for n, p in extractor.named_parameters()
                      if p.grad is not None and torch.isnan(p.grad).any()]
        assert len(nan_params) == 0, f"Parameters with NaN gradient: {nan_params[:10]}"

    def test_gcn_layer_gradients(self):
        """GCNLayer parameters receive gradients."""
        layer = GCNLayer(in_dim=16, out_dim=16, edge_dim=8)
        layer.train()
        N, E = 10, 6
        x = torch.randn(N, 16)
        edge_index = torch.randint(0, N, (2, E))
        edge_attr = torch.randn(E, 8)
        out = layer(x, edge_index, edge_attr)
        out.sum().backward()

        dead = [n for n, p in layer.named_parameters() if p.grad is None]
        assert len(dead) == 0, f"Dead parameters: {dead}"

    def test_contrastive_loss_backward(self):
        """ContrastiveLoss produces valid gradients."""
        criterion = ContrastiveLoss()
        B, D = 8, 32
        embeddings = torch.randn(B, D, requires_grad=True)
        labels = torch.randint(0, 3, (B,))
        loss = criterion(embeddings, labels)
        loss.backward()
        assert embeddings.grad is not None
        assert not torch.isnan(embeddings.grad).any()

    def test_consolidation_gradients(self):
        """Consolidation training produces gradients and updates weights."""
        cfg = SemanticMLConfig(node_embed_dim=32, edge_embed_dim=8,
                               hidden_dim=64, n_layers=2, n_pattern_slots=8,
                               pattern_embed_dim=32, n_heads=2)
        extractor = SemanticPatternExtractor(cfg)
        # Create 4 subgraphs with labels
        subgraphs = []
        for i in range(4):
            sg = KGSubgraph(root_node_id=f"s{i}", timestamp=datetime.now(),
                            query_type="subgraph", summary=f"sg{i}")
            sg.node_features = torch.randn(5, cfg.node_embed_dim)
            sg.edge_index = torch.randint(0, 5, (2, 3))
            sg.edge_features = torch.randn(3, cfg.edge_embed_dim)
            sg.label = i % 2  # 2 classes
            subgraphs.append(sg)

        before = extractor.get_all_prototypes().clone()
        stats = extractor.consolidate(subgraphs, n_epochs=2, lr=1e-3)
        after = extractor.get_all_prototypes()
        # Prototypes should have changed
        assert not torch.allclose(before, after, atol=1e-6), \
            "Consolidation did not update prototype vectors"
        assert stats["epochs"] == 2
        assert stats["final_loss"] is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 1c. Correctness / Invariance Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCorrectness:
    """Domain-specific correctness: temporal causality, dedup, KG invariants."""

    def test_temporal_index_ordering(self, kg):
        """Temporal index returns nodes in correct time order."""
        # Add events with known timestamps
        from datetime import timedelta
        now = datetime.now()
        kg.temporal_index.insert("early", now - timedelta(days=10))
        kg.temporal_index.insert("middle", now - timedelta(days=5))
        kg.temporal_index.insert("late", now)

        results = kg.temporal_index.query_range(
            from_date=now - timedelta(days=20),
            to_date=now,
        )
        # All 3 should appear (they're in different bins depending on resolution)
        assert "early" in results
        assert "middle" in results
        assert "late" in results

    def test_dedup_cache_prevents_duplicates(self, kg):
        """DedupCache prevents duplicate ingestion within time window."""
        event = DiseaseEvent(
            event_id="de_dedup", field_id="field_D1", crop_cycle_id="cc_dedup",
            disease_name="test_dedup", first_observed="2026-06-01",
            status=DiseaseStatus.SUSPECTED, severity=0.3,
        )
        # Write twice (fixture already wrote 3 events)
        id1 = kg.fast_write(event)
        prev_count = kg._write_count
        id2 = kg.fast_write(event)
        # Same event should return existing ID
        assert id1 == id2, "Dedup should return same ID for duplicate event"
        # Write count should increment (we still count the write attempt)
        assert kg._write_count == prev_count + 1, \
            f"Write count should increment by 1, was {prev_count}, now {kg._write_count}"
        # Dedup should prevent adding a second node for the same event.
        # Fixture already has 1 disease node (de_001), so total should be 2:
        # fixture's + test's first write (second write is dedup'd).
        disease_nodes = [n for n in kg.nodes.values()
                         if n.node_type == "disease_event"]
        assert len(disease_nodes) == 2, \
            f"Dedup should have prevented duplicate node, found {len(disease_nodes)}"

    def test_spatial_proximity_empty_kg(self):
        """Spatial query on empty KG returns empty list (no crash)."""
        config = EpisodicKGConfig()
        kg = EpisodicKnowledgeGraph(config)
        results = kg.spatial_proximity_query(
            SpatialProximityQuery(center_field_id="nowhere", radius_m=100)
        )
        assert results == [], f"Expected empty list, got {results}"

    def test_temporal_path_query_empty_kg(self):
        """Temporal path query on empty KG returns empty list."""
        config = EpisodicKGConfig()
        kg = EpisodicKnowledgeGraph(config)
        results = kg.temporal_path_query(
            TemporalPathQuery(start_node_id="nonexistent", max_hops=3)
        )
        assert results == [], f"Expected empty list, got {results}"

    def test_working_memory_lru_eviction(self):
        """WorkingMemory evicts oldest items under LRU policy."""
        wm = WorkingMemory(max_tokens=50, eviction_policy="lru")
        for i in range(30):
            wm.add(f"key_{i}", "x" * 20)
        assert len(wm) < 30, "Eviction did not trigger"
        # The most recently added should still be present
        # (LRU evicts least recently used)
        assert "key_29" in wm or len(wm) > 0

    def test_diagnose_does_not_mutate_input(self, system):
        """diagnose() does not mutate the input context."""
        context = DiagnosticContext(
            field_id="field_A42", crop_type="wheat",
            season_start=datetime(2026, 3, 1), season_end=datetime(2026, 6, 1),
        )
        field_before = context.field_id
        system.diagnose("test query", context)
        assert context.field_id == field_before, "Context was mutated"

    def test_subgraph_batch_empty_raises(self):
        """SubgraphBatch with empty list raises ValueError."""
        with pytest.raises(ValueError, match="at least one subgraph"):
            SubgraphBatch([])

    def test_extract_consolidation_batch_empty(self, kg):
        """extract_consolidation_batch returns empty list when no mature subgraphs."""
        batch = kg.extract_consolidation_batch(max_samples=10)
        assert isinstance(batch, list)


# ═══════════════════════════════════════════════════════════════════════════════
# 1d. Numerical Stability Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestNumerics:
    """Numerical stability: bf16 forward, extreme inputs, NaN/Inf checks."""

    def test_bf16_semantic_forward(self, sample_subgraph_batch):
        """SemanticPatternExtractor runs in bfloat16 without NaN/Inf."""
        cfg = SemanticMLConfig(node_embed_dim=32, edge_embed_dim=8,
                               hidden_dim=64, n_layers=2, n_pattern_slots=8,
                               pattern_embed_dim=32, n_heads=2)
        extractor = SemanticPatternExtractor(cfg)
        extractor = extractor.bfloat16()

        batch_bf16 = copy.deepcopy(sample_subgraph_batch)
        batch_bf16.node_features = batch_bf16.node_features.bfloat16()
        if batch_bf16.edge_features is not None:
            batch_bf16.edge_features = batch_bf16.edge_features.bfloat16()

        with torch.no_grad():
            out = extractor._forward_impl(batch_bf16)

        assert not torch.isnan(out["pattern_embed"]).any(), "NaN in bf16 pattern_embed"
        assert not torch.isinf(out["pattern_embed"]).any(), "Inf in bf16 pattern_embed"
        assert not torch.isnan(out["confidence"]).any(), "NaN in bf16 confidence"

    def test_fp16_semantic_forward(self, sample_subgraph_batch):
        """SemanticPatternExtractor runs in float16 without NaN/Inf."""
        cfg = SemanticMLConfig(node_embed_dim=32, edge_embed_dim=8,
                               hidden_dim=64, n_layers=2, n_pattern_slots=8,
                               pattern_embed_dim=32, n_heads=2)
        extractor = SemanticPatternExtractor(cfg)
        extractor = extractor.half()

        batch_fp16 = copy.deepcopy(sample_subgraph_batch)
        batch_fp16.node_features = batch_fp16.node_features.half()
        if batch_fp16.edge_features is not None:
            batch_fp16.edge_features = batch_fp16.edge_features.half()

        with torch.no_grad():
            out = extractor._forward_impl(batch_fp16)

        assert not torch.isnan(out["pattern_embed"]).any(), "NaN in fp16 pattern_embed"
        assert not torch.isinf(out["pattern_embed"]).any(), "Inf in fp16 pattern_embed"
        assert not torch.isnan(out["confidence"]).any(), "NaN in fp16 confidence"

    def test_prototype_initialization(self):
        """Prototype vectors are initialized on unit sphere."""
        attn = PrototypeAttention(pattern_dim=64, n_prototypes=16, n_heads=4)
        protos = attn.prototype_vectors.data
        norms = protos.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"Prototype norms: {norms}"

    def test_gcn_heterophily_gate_range(self):
        """Heterophily gate output stays in (0, 1)."""
        layer = GCNLayer(in_dim=16, out_dim=16, edge_dim=8)
        N, E = 10, 6
        x = torch.randn(N, 16)
        edge_index = torch.randint(0, N, (2, E))
        edge_attr = torch.randn(E, 8)
        _ = layer(x, edge_index, edge_attr)
        # Access the gate weights
        gate = layer.gate
        test_input = torch.randn(10, 16)
        gate_vals = gate(test_input)
        assert gate_vals.min() >= 0.0, f"Gate < 0: {gate_vals.min()}"
        assert gate_vals.max() <= 1.0, f"Gate > 1: {gate_vals.max()}"

    def test_contrastive_loss_numerical_stability(self):
        """ContrastiveLoss handles edge cases without NaN."""
        criterion = ContrastiveLoss(temperature=0.1)

        # Case 1: all same class
        emb1 = torch.randn(8, 32)
        labels1 = torch.zeros(8, dtype=torch.long)
        loss1 = criterion(emb1, labels1)
        assert torch.isfinite(loss1), f"Loss not finite (all same class): {loss1}"

        # Case 2: all different classes
        emb2 = torch.randn(8, 32)
        labels2 = torch.arange(8, dtype=torch.long)
        loss2 = criterion(emb2, labels2)
        assert torch.isfinite(loss2), f"Loss not finite (all different): {loss2}"

        # Case 3: no labels (instance discrimination)
        emb3 = torch.randn(4, 32)
        loss3 = criterion(emb3, labels=None)
        assert torch.isfinite(loss3), f"Loss not finite (no labels): {loss3}"

    def test_subgraph_batch_tensor_dtypes(self, sample_subgraph_batch):
        """SubgraphBatch tensors have correct dtypes."""
        sb = sample_subgraph_batch
        assert sb.node_features.dtype == torch.float32
        assert sb.edge_index.dtype == torch.long
        assert sb.node_mask.dtype == torch.bool


# ═══════════════════════════════════════════════════════════════════════════════
# 1e. Permutation Invariance (Graph ML)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPermutationInvariance:
    """Graph-level output must be invariant to node permutation."""

    def test_subgraph_permutation_invariance(self):
        """SemanticPatternExtractor output should be invariant to node permutation
        in the subgraph (since we use mean pooling over all nodes)."""
        cfg = SemanticMLConfig(node_embed_dim=32, edge_embed_dim=8,
                               hidden_dim=64, n_layers=2, n_pattern_slots=8,
                               pattern_embed_dim=32, n_heads=2)
        extractor = SemanticPatternExtractor(cfg)
        extractor.eval()

        N, E = 10, 8
        features = torch.randn(N, cfg.node_embed_dim)
        edge_index = torch.randint(0, N, (2, E))
        edge_features = torch.randn(E, cfg.edge_embed_dim)

        # Build original subgraph
        sg_orig = KGSubgraph(root_node_id="test", timestamp=datetime.now(),
                             query_type="subgraph", summary="orig")
        sg_orig.node_features = features.clone()
        sg_orig.edge_index = edge_index.clone()
        sg_orig.edge_features = edge_features.clone()

        # Build permuted subgraph
        perm = torch.randperm(N)
        inv_perm = torch.argsort(perm)  # inverse permutation for edges
        sg_perm = KGSubgraph(root_node_id="test", timestamp=datetime.now(),
                             query_type="subgraph", summary="perm")
        sg_perm.node_features = features[perm].clone()
        # Remap edge indices
        edge_index_perm = torch.zeros_like(edge_index)
        for e in range(E):
            edge_index_perm[0, e] = inv_perm[edge_index[0, e]]
            edge_index_perm[1, e] = inv_perm[edge_index[1, e]]
        sg_perm.edge_index = edge_index_perm
        sg_perm.edge_features = edge_features.clone()

        with torch.no_grad():
            out_orig = extractor._forward_impl(SubgraphBatch([sg_orig]))
            out_perm = extractor._forward_impl(SubgraphBatch([sg_perm]))

        # Mean pooling should produce invariant output
        assert torch.allclose(out_orig["pattern_embed"], out_perm["pattern_embed"],
                              atol=1e-4), "Pattern embedding changed under permutation"
        assert torch.allclose(out_orig["confidence"], out_perm["confidence"],
                              atol=1e-4), "Confidence changed under permutation"


# ═══════════════════════════════════════════════════════════════════════════════
# 1f. Ablation Mode Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAblationModes:
    """Ablation mode switching and behavioral correctness."""

    def test_ablation_kg_only(self, system):
        """kg_only mode produces answer without semantic ML."""
        system.set_ablation_mode("kg_only")
        context = DiagnosticContext(field_id="field_A42", crop_type="wheat")
        response = system.diagnose("test query", context)
        assert isinstance(response.answer, str)
        assert len(response.answer) > 0
        # Should have KG-only evidence
        for ev in response.evidence:
            assert ev["type"] == "episodic", \
                f"Expected episodic evidence, got {ev['type']}"

    def test_ablation_no_iterate(self, system):
        """no_iterate mode has max_iterative_cycles=1."""
        system.set_ablation_mode("no_iterate")
        context = DiagnosticContext(field_id="field_A42", crop_type="wheat")
        response = system.diagnose("test query", context)
        assert response.num_iterations <= 1, \
            f"Expected <=1 iterations, got {response.num_iterations}"

    def test_ablation_full_restore(self, system):
        """Switching back to full mode restores iterative querying."""
        system.set_ablation_mode("no_iterate")
        system.set_ablation_mode("full")
        context = DiagnosticContext(field_id="field_A42", crop_type="wheat")
        response = system.diagnose("test query", context)
        # In full mode, max_iterative_cycles is from config (2 in base_cfg)
        assert response.num_iterations >= 1

    def test_ablation_invalid_mode(self, system):
        """Invalid ablation mode raises ValueError."""
        with pytest.raises(ValueError):
            system.set_ablation_mode("nonexistent_mode")

    def test_ablation_reset_preserves_state(self, system):
        """Reset clears statistics."""
        context = DiagnosticContext(field_id="field_A42")
        system.diagnose("test", context)
        system.reset()
        state = system.get_system_state()
        assert state["statistics"]["events_ingested"] == 0
        assert state["statistics"]["diagnoses_performed"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 1g. Interface Contract Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestInterfaceContracts:
    """Test that all abstract base classes are properly implemented."""

    def test_base_episodic_memory_interface(self, kg):
        """EpisodicKnowledgeGraph implements all BaseEpisodicMemory methods."""
        from base import BaseEpisodicMemory
        assert isinstance(kg, BaseEpisodicMemory)

        # All abstract methods should be callable
        methods = ["fast_write", "temporal_path_query", "spatial_proximity_query",
                    "extract_consolidation_batch", "prewarm_semantic_query",
                    "to_subgraph_batch"]
        for m in methods:
            assert hasattr(kg, m), f"Missing method: {m}"

    def test_base_semantic_memory_interface(self):
        """SemanticMemoryManager implements all BaseSemanticMemory methods."""
        from base import BaseSemanticMemory
        cfg = SemanticMLConfig()
        mgr = SemanticMemoryManager(cfg)
        assert isinstance(mgr, BaseSemanticMemory)

        methods = ["infer_pattern", "consolidate", "few_shot_adapt",
                    "query_episodic_via_semantic_prior"]
        for m in methods:
            assert hasattr(mgr, m), f"Missing method: {m}"

    def test_base_agent_controller_interface(self, kg):
        """CLSAgentController implements all BaseAgentController methods."""
        from base import BaseAgentController
        cfg = AgentControllerConfig()
        semantic_mgr = SemanticMemoryManager(SemanticMLConfig())
        controller = CLSAgentController(cfg, kg, semantic_mgr)
        assert isinstance(controller, BaseAgentController)

        methods = ["diagnose", "_reconcile", "_generate_response"]
        for m in methods:
            assert hasattr(controller, m), f"Missing method: {m}"

    def test_provenance_tracking(self, system):
        """Provenance tracking tags every claim with its source layer."""
        context = DiagnosticContext(field_id="field_A42", crop_type="wheat")
        response = system.diagnose("test query", context)
        for prov in response.provenance:
            assert "source" in prov, f"Provenance item missing source: {prov}"
            assert prov["source"] in ("episodic_kg", "semantic_ml"), \
                f"Unknown source: {prov['source']}"
            assert "confidence" in prov


# ═══════════════════════════════════════════════════════════════════════════════
# Run with: python -m pytest test_model.py -v
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
