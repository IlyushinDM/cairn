"""Тесты верификатора причинного графа (пять аксиом)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from cairn.reasoning.graph_verifier import CausalGraphVerifier, AxiomStatus


def make_verifier():
    return CausalGraphVerifier(
        temporal_tolerance_sec=15.0,
        transitivity_threshold=0.3,
        monotonicity_epsilon=0.05,
    )


def test_acyclicity_ok():
    v = make_verifier()
    result = v.verify(
        edges=[(0, 1), (1, 2)],
        causal_effects={0: 0.9, 1: 0.5, 2: 0.1},
        anomaly_times={0: 0.0, 1: 5.0, 2: 10.0},
        physical_edges={(0, 1), (1, 2)},
        root_candidate=0,
    )
    acyc = next(r for r in result.axiom_results if r.name == "Ацикличность")
    assert acyc.status == AxiomStatus.OK


def test_acyclicity_cycle_detected():
    v = make_verifier()
    result = v.verify(
        edges=[(0, 1), (1, 2), (2, 0)],  # цикл
        causal_effects={0: 0.9, 1: 0.5, 2: 0.1},
        anomaly_times={0: 0.0, 1: 1.0, 2: 2.0},
        physical_edges={(0, 1), (1, 2), (2, 0)},
        root_candidate=0,
    )
    acyc = next(r for r in result.axiom_results if r.name == "Ацикличность")
    assert acyc.status == AxiomStatus.VIOLATED


def test_temporal_violation():
    v = make_verifier()
    result = v.verify(
        edges=[(0, 1)],
        causal_effects={0: 0.9, 1: 0.1},
        anomaly_times={0: 100.0, 1: 0.0},  # причина началась ПОСЛЕ следствия
        physical_edges={(0, 1)},
        root_candidate=0,
    )
    temp = next(r for r in result.axiom_results if r.name == "Темпоральная согласованность")
    assert temp.status == AxiomStatus.VIOLATED


def test_confidence_all_ok():
    v = make_verifier()
    result = v.verify(
        edges=[(0, 1)],
        causal_effects={0: 0.9, 1: 0.1},
        anomaly_times={0: 0.0, 1: 5.0},
        physical_edges={(0, 1)},
        root_candidate=0,
    )
    # Доверие ≈ 1.0 при отсутствии нарушений
    assert result.confidence >= 0.8
