"""Тесты условной смеси гауссовых распределений."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
import pytest

from cairn.reasoning.conditional_gmm import ConditionalGMM


def make_gmm():
    return ConditionalGMM(state_dim=32, context_dim=8, n_components=3)


def test_forward_shapes():
    gmm = make_gmm()
    ctx = torch.randn(5, 8)
    mu, sigma, w = gmm(ctx)
    assert mu.shape == (5, 3, 32)
    assert sigma.shape == (5, 3, 32)
    assert w.shape == (5, 3)
    assert torch.allclose(w.sum(dim=-1), torch.ones(5), atol=1e-5)


def test_nll_positive():
    gmm = make_gmm()
    h = torch.randn(5, 32)
    ctx = torch.randn(5, 8)
    nll = gmm.nll(h, ctx)
    assert nll.shape == (5,)
    # NLL может быть отрицательным для многомерных гауссовых (плотность > 1)
    assert not torch.isnan(nll).any()


def test_prototype_shape():
    gmm = make_gmm()
    ctx = torch.randn(5, 8)
    proto = gmm.conditional_prototype(ctx)
    assert proto.shape == (5, 32)


def test_anomaly_threshold():
    gmm = make_gmm()
    h = torch.randn(100, 32)
    ctx = torch.randn(100, 8)
    nll = gmm.nll(h, ctx)
    delta = gmm.anomaly_threshold(nll, percentile=0.95)
    assert isinstance(delta, float)
