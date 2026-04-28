"""Тесты двухветвевого кодировщика метрик."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
import pytest

from cairn.perception.metric_encoder import MetricEncoder, StablePatternBranch, BreakpointBranch


def test_breakpoint_branch_shape():
    branch = BreakpointBranch(input_features=8, window=30, out_dim=32)
    x = torch.randn(4, 90, 8)  # batch=4, T=90, F=8
    out = branch(x)
    assert out.shape == (4, 32)


def test_metric_encoder_shape():
    enc = MetricEncoder(input_features=8, window=30, d1=32, d2=32, out_dim=64)
    x = torch.randn(4, 90, 8)
    out = enc(x)
    assert out.shape == (4, 64)


def test_metric_encoder_gradient_flows():
    enc = MetricEncoder(input_features=4, window=20, d1=16, d2=16, out_dim=32)
    x = torch.randn(2, 60, 4, requires_grad=True)
    out = enc(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
