"""Общие фикстуры pytest для CAIRN.

Запуск:
    pytest                          # все тесты
    pytest -m "not slow"            # без медленных
    pytest -m "not gpu"             # без GPU
    pytest --cov=cairn --cov-report=term-missing
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Базовые фикстуры ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def device():
    """CPU для всех тестов (GPU – только с маркером @pytest.mark.gpu)."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def small_arch():
    """Минимальная архитектура для быстрых тестов."""
    return {
        "state_dim":     16,
        "context_dim":   4,
        "n_metrics":     4,
        "n_components":  3,
        "n_confounders": 1,
        "confounder_dim": 4,
        "d_met":         16,  # must match context_builder input dim
        "d_log":         4,
        "d_tr":          4,
        "d_ssm":         4,
        "d_brk":         4,
        "ssm_state_dim": 8,
        "window":        8,
        "log_vocab_size": 50,
        "n_conv_layers": 1,
    }


@pytest.fixture(scope="session")
def cairn_model(small_arch, device):
    """Инициализированная CAIRN модель (не обученная)."""
    from cairn.perception import StateBuilder
    from cairn.reasoning import (
        ConditionalGMM, ConfoundedVGAE, CounterfactualModule,
    )
    from cairn.training import CAIRNModel

    A = small_arch
    model = CAIRNModel(
        state_builder=StateBuilder(
            n_metrics=A["n_metrics"],
            log_vocab_size=A["log_vocab_size"],
            state_dim=A["state_dim"],
            context_dim=A["context_dim"],
            d_met=A["d_met"], d_log=A["d_log"],
            d_tr=A["d_tr"],   d_ssm=A["d_ssm"],
            d_brk=A["d_brk"], ssm_state_dim=A["ssm_state_dim"],
            window=A["window"],
        ),
        gmm=ConditionalGMM(
            A["state_dim"], A["context_dim"], A["n_components"],
        ),
        vgae=ConfoundedVGAE(
            A["state_dim"], A["n_confounders"], A["confounder_dim"],
        ),
        cf_module=CounterfactualModule(
            A["state_dim"], A["n_conv_layers"],
        ),
    ).to(device)
    model.eval()
    return model


@pytest.fixture
def sample_metric_data(small_arch):
    """Синтетические метрики: (T=20, N=3, F=4)."""
    from cairn.connectors.base import MetricData

    T, N, F = 20, 3, small_arch["n_metrics"]
    timestamps = np.linspace(0, 100, T)
    values     = np.random.randn(T, N, F).astype(np.float32)
    names      = [f"service-{i}" for i in range(N)]
    metrics    = ["cpu", "memory", "latency", "rps"]
    return MetricData(timestamps, values, names, metrics)


@pytest.fixture(scope="session")
def tiny_hypergraph():
    """Минимальный гиперграф: 3 узла, 2 рёбра."""
    from pathlib import Path as _Path
    import tempfile, yaml, os
    from cairn.connectors.csv_file import YAMLTopologyConnector
    from cairn.perception import HypergraphBuilder

    topo_data = {
        "instances": [
            {"name": "svc-0", "service": "a", "host": "h",
             "cpu_limit": 1.0, "memory_limit": 256, "version": "1"},
            {"name": "svc-1", "service": "b", "host": "h",
             "cpu_limit": 1.0, "memory_limit": 256, "version": "1"},
            {"name": "svc-2", "service": "c", "host": "h",
             "cpu_limit": 1.0, "memory_limit": 256, "version": "1"},
        ],
        "call_edges": [
            ["svc-0", "svc-1"],
            ["svc-1", "svc-2"],
        ],
        "colocation_groups":   [],
        "load_balancer_groups": [],
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(topo_data, f, allow_unicode=True)
        tmp_path = f.name

    try:
        topo = YAMLTopologyConnector(tmp_path).fetch()
        return HypergraphBuilder.from_topology_data(topo)
    finally:
        os.unlink(tmp_path)