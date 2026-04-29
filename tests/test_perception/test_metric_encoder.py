"""Тесты двухветвевого кодировщика метрик."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
import pytest

from cairn.perception.metric_encoder import DualBranchMetricEncoder, MetricEncoder, StablePatternBranch, BreakpointBranch


def test_breakpoint_branch_shape():
    branch = BreakpointBranch(n_metrics=8, window=30, d_out=32)
    x = torch.randn(4, 90, 8)
    assert branch(x).shape == (4, 32)

def test_metric_encoder_shape():
    enc = DualBranchMetricEncoder(n_metrics=8, d_ssm=32, d_brk=32, d_out=64, window=30)
    x = torch.randn(4, 90, 8)
    assert enc(x).shape == (4, 64)

def test_metric_encoder_gradient_flows():
    enc = DualBranchMetricEncoder(n_metrics=4, d_ssm=16, d_brk=16, d_out=32, window=20)
    x = torch.randn(2, 60, 4, requires_grad=True)
    enc(x).sum().backward()
    assert x.grad is not None
