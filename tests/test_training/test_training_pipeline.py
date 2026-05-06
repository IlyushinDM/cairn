"""Тесты системы обучения CAIRN.

Классы:
  TestCAIRNLoss          — unit-тесты функции потерь (все 6 компонент)
  TestCAIRNDataset       — unit-тесты датасета и IncidentBuilder
  TestCAIRNModel         — unit-тест forward-pass модели
  TestCAIRNTrainer       — один проход обучения (1 эпоха каждого этапа)
  TestComputeMetrics     — unit-тесты метрик AC@k, Avg@5, F1
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
import torch
import torch.nn as nn

from cairn.training.loss import CAIRNLoss, LossWeights
from cairn.training.data_loader import (
    Incident, CAIRNDataset, IncidentBuilder, create_demo_dataset,
)
from cairn.training.trainer import (
    CAIRNModel, CAIRNTrainer, TrainerConfig, compute_metrics,
)
from cairn.perception import StateBuilder, HypergraphBuilder
from cairn.reasoning import ConditionalGMM, ConfoundedVGAE, CounterfactualModule

SAMPLE_DIR = Path(__file__).parent.parent.parent / "data" / "sample"

# ---------------------------------------------------------------------------
# Константы (уменьшены для скорости тестов)
# ---------------------------------------------------------------------------
N   = 5     # число сервисов
D   = 32    # state_dim
CTX = 8     # context_dim
F   = 4     # число метрик
T   = 30    # длина ряда
L   = 10    # длина журнала


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture
def hypergraph():
    builder = HypergraphBuilder(N)
    return builder.from_topology(
        call_paths=[[0, 1], [1, 2], [1, 3]],
        colocated_groups=[[1, 4]],
        lb_groups=[],
        instance_names=[f"svc-{i}" for i in range(N)],
    )

@pytest.fixture
def fake_incident(is_anomaly=True):
    """Синтетический инцидент для unit-тестов (без реальных данных)."""
    return Incident(
        metric_data=torch.randn(N, T, F),
        log_data=torch.randint(0, 50, (N, L)),
        trace_data=torch.randint(0, 5, (N,)),
        context=torch.zeros(N, CTX),
        root_cause=2 if is_anomaly else -1,
        fault_type="cpu_exhaustion" if is_anomaly else "normal",
        instance_names=[f"svc-{i}" for i in range(N)],
        is_anomaly=is_anomaly,
    )

@pytest.fixture
def normal_incident():
    return Incident(
        metric_data=torch.randn(N, T, F),
        log_data=torch.randint(0, 50, (N, L)),
        trace_data=torch.zeros(N, dtype=torch.long),
        context=torch.zeros(N, CTX),
        root_cause=-1,
        fault_type="normal",
        instance_names=[f"svc-{i}" for i in range(N)],
        is_anomaly=False,
    )

@pytest.fixture
def model(hypergraph):
    return CAIRNModel(
        state_builder=StateBuilder(
            n_metrics=F, log_vocab_size=100,
            state_dim=D, context_dim=CTX,
            d_met=16, d_log=8, d_tr=8,
            d_ssm=8, d_brk=8, ssm_state_dim=16, window=10,
            context_raw_dim=CTX,
        ),
        gmm=ConditionalGMM(state_dim=D, context_dim=CTX, n_components=3),
        vgae=ConfoundedVGAE(state_dim=D, n_confounders=2, confounder_dim=8),
        cf_module=CounterfactualModule(state_dim=D, n_conv_layers=1),
    )

@pytest.fixture
def loss_fn():
    return CAIRNLoss(adaptive=False)

@pytest.fixture
def dataset(fake_incident, normal_incident):
    incidents = [normal_incident] * 3 + [fake_incident] * 2
    return CAIRNDataset(incidents)


# ===========================================================================
# TestCAIRNLoss
# ===========================================================================

class TestCAIRNLoss:

    def test_init_default_weights(self):
        loss = CAIRNLoss()
        assert loss.w.lambda_pe == 1.0
        assert loss.w.lambda_reb == 0.1

    def test_init_initial_weights_list(self):
        loss = CAIRNLoss(initial_weights=[2.0, 2.0, 2.0, 1.0, 1.0, 0.5])
        assert loss.w.lambda_pe == 2.0
        assert loss.w.lambda_reb == 0.5

    def test_loss_pe_root_highest(self):
        """L_PE = 0 если root имеет наивысший PE."""
        loss = CAIRNLoss()
        pe = torch.tensor([1.0, 5.0, 2.0, 3.0])  # root=1 — максимальный
        val = loss.loss_pe(pe, root_idx=1)
        assert val.item() == pytest.approx(0.0, abs=0.2)

    def test_loss_pe_root_not_highest(self):
        """L_PE > 0 если другие выше root."""
        loss = CAIRNLoss(margin=0.0)
        pe = torch.tensor([5.0, 1.0, 3.0])  # root=1 — НЕ максимальный
        val = loss.loss_pe(pe, root_idx=1)
        assert val.item() > 0.0

    def test_loss_um_shape(self):
        loss = CAIRNLoss()
        nll = torch.randn(N).abs()
        result = loss.loss_um(nll, [])
        assert result.shape == ()

    def test_loss_vak_reconstruction(self):
        loss = CAIRNLoss()
        h = torch.randn(N, D)
        result = loss.loss_vak(h, h.clone(), torch.zeros(N, D), torch.zeros(N, D), [])
        assert result.shape == ()
        assert result.item() >= 0

    def test_loss_nez_range(self):
        """L_нез ∈ [0, 1] для нормализованных векторов."""
        loss = CAIRNLoss()
        u = torch.randn(N, D)
        val = loss.loss_nez(u)
        assert 0.0 <= val.item() <= 1.0 + 1e-5

    def test_loss_nez_single_node(self):
        loss = CAIRNLoss()
        u = torch.randn(1, D)
        assert loss.loss_nez(u).item() == 0.0

    def test_loss_kr_correct_direction(self):
        """L_КР должна штрафовать, если sim(root_anom, root_norm) > sim(others_anom, others_norm)."""
        loss = CAIRNLoss(tcd_margin=0.0)
        h_root_anom  = torch.tensor([1.0, 0.0])
        h_root_norm  = torch.tensor([0.0, 1.0])   # противоположны
        h_oth_anom   = torch.tensor([[1.0, 0.0]])
        h_oth_norm   = torch.tensor([[1.0, 0.0]])  # одинаковы
        val = loss.loss_kr(h_root_anom, h_root_norm, h_oth_anom, h_oth_norm)
        assert val.shape == ()

    def test_pretrain_loss_returns_tuple(self):
        loss = CAIRNLoss()
        nll = torch.randn(N).abs()
        h   = torch.randn(N, D)
        total, comps = loss.pretrain_loss(nll, h, h.clone(), torch.zeros(N, D), torch.zeros(N, D))
        assert isinstance(total, torch.Tensor)
        assert "L_um" in comps and "L_vak" in comps

    def test_main_loss_returns_tuple(self):
        loss = CAIRNLoss()
        pe  = torch.randn(N)
        u   = torch.randn(N, D)
        h   = torch.randn(N, D)
        p   = torch.randn(N, D)
        total, comps = loss.main_loss(pe, 0, u, h[0], p[0], h[1:], p[1:])
        assert isinstance(total, torch.Tensor)
        assert "L_pe" in comps

    def test_forward_returns_total_and_components(self):
        loss = CAIRNLoss()
        outputs = {
            "pe_scores":     torch.randn(N),
            "nll_normal":    torch.randn(N).abs(),
            "cov_matrices":  [],
            "h":             torch.randn(N, D),
            "h_recon":       torch.randn(N, D),
            "mu_u":          torch.zeros(N, D),
            "log_var_u":     torch.zeros(N, D),
            "kl_z_terms":    [],
            "u_hat":         torch.randn(N, D),
            "h_root_anom":   torch.randn(D),
            "h_root_norm":   torch.randn(D),
            "h_others_anom": torch.randn(N-1, D),
            "h_others_norm": torch.randn(N-1, D),
            "edge_weights":  None,
            "edge_cf_stats": None,
        }
        targets = {"root_idx": 0}
        total, comps = loss(outputs, targets)
        assert total.shape == ()
        assert "L_pe" in comps and "L_um" in comps

    def test_update_weights_adaptive(self):
        loss = CAIRNLoss(adaptive=True)
        old_pe = loss.w.lambda_pe
        loss.update_weights({"L_pe": 10.0, "L_um": 0.1, "L_vak": 0.1, "L_nez": 0.1, "L_kr": 0.1, "L_reb": 0.1})
        # L_pe доминирует → её вес должен вырасти
        assert loss.w.lambda_pe > old_pe

    def test_update_weights_sum_preserved(self):
        """Сумма весов сохраняется после перевзвешивания."""
        loss = CAIRNLoss(adaptive=True)
        init_sum = sum(vars(loss.w).values())
        loss.update_weights({"L_pe": 5.0, "L_um": 1.0, "L_vak": 1.0, "L_nez": 1.0, "L_kr": 1.0, "L_reb": 1.0})
        new_sum = sum(vars(loss.w).values())
        assert new_sum == pytest.approx(init_sum, rel=1e-3)

    def test_gradient_flows_through_loss(self):
        loss = CAIRNLoss()
        h = torch.randn(N, D, requires_grad=True)
        nll = torch.randn(N).abs()
        total, _ = loss.pretrain_loss(nll, h, h + 0.1, torch.zeros(N, D), torch.zeros(N, D))
        total.backward()
        assert h.grad is not None


# ===========================================================================
# TestCAIRNDataset
# ===========================================================================

class TestCAIRNDataset:

    def test_len(self, dataset):
        assert len(dataset) == 5

    def test_getitem_returns_incident(self, dataset):
        inc = dataset[0]
        assert isinstance(inc, Incident)

    def test_normal_subset(self, dataset):
        ns = dataset.normal_subset()
        assert ns.n_anomaly == 0
        assert ns.n_normal > 0

    def test_anomaly_subset(self, dataset):
        as_ = dataset.anomaly_subset()
        assert as_.n_normal == 0
        assert as_.n_anomaly > 0

    def test_summary_string(self, dataset):
        s = dataset.summary()
        assert "CAIRNDataset" in s
        assert "нормальных" in s

    def test_n_normal_and_n_anomaly(self, dataset):
        assert dataset.n_normal + dataset.n_anomaly == len(dataset)

    def test_incident_shapes(self, fake_incident):
        assert fake_incident.metric_data.shape == (N, T, F)
        assert fake_incident.log_data.shape == (N, L)
        assert fake_incident.trace_data.shape == (N,)

    def test_incident_n_instances(self, fake_incident):
        assert fake_incident.n_instances == N

    def test_incident_n_timesteps(self, fake_incident):
        assert fake_incident.n_timesteps == T

    def test_anomaly_has_root_cause(self, fake_incident):
        assert fake_incident.is_anomaly
        assert fake_incident.root_cause >= 0

    def test_normal_has_no_root_cause(self, normal_incident):
        assert not normal_incident.is_anomaly
        assert normal_incident.root_cause == -1


# ===========================================================================
# TestCAIRNModel
# ===========================================================================

class TestCAIRNModel:

    def test_forward_returns_dict(self, model, fake_incident, hypergraph):
        model.eval()
        with torch.no_grad():
            outputs = model(fake_incident, hypergraph)
        assert isinstance(outputs, dict)

    def test_forward_required_keys(self, model, fake_incident, hypergraph):
        model.eval()
        with torch.no_grad():
            out = model(fake_incident, hypergraph)
        required = {"H", "C", "nll_normal", "h", "h_recon", "mu_u", "log_var_u", "u_hat"}
        assert required <= set(out.keys())

    def test_forward_H_shape(self, model, fake_incident, hypergraph):
        model.eval()
        with torch.no_grad():
            out = model(fake_incident, hypergraph)
        assert out["H"].shape == (N, D)

    def test_forward_nll_shape(self, model, fake_incident, hypergraph):
        model.eval()
        with torch.no_grad():
            out = model(fake_incident, hypergraph)
        assert out["nll_normal"].shape == (N,)

    def test_forward_no_nan(self, model, fake_incident, hypergraph):
        model.eval()
        with torch.no_grad():
            out = model(fake_incident, hypergraph)
        for key in ("H", "C", "nll_normal", "h_recon"):
            assert not torch.isnan(out[key]).any(), f"NaN в {key}"

    def test_forward_normal_incident(self, model, normal_incident, hypergraph):
        """Работает и без root_cause."""
        model.eval()
        with torch.no_grad():
            out = model(normal_incident, hypergraph)
        assert out["pe_scores"].shape == (N,)


# ===========================================================================
# TestComputeMetrics
# ===========================================================================

class TestComputeMetrics:

    def test_ac1_found_at_top(self):
        ranked = [(2, 5.0), (0, 3.0), (1, 1.0)]
        m = compute_metrics(ranked, root_cause=2)
        assert m["AC@1"] == 1.0

    def test_ac1_not_found_at_top(self):
        ranked = [(0, 5.0), (1, 3.0), (2, 1.0)]
        m = compute_metrics(ranked, root_cause=2)
        assert m["AC@1"] == 0.0

    def test_ac3_found_at_3(self):
        ranked = [(0, 5.0), (1, 3.0), (2, 1.0)]
        m = compute_metrics(ranked, root_cause=2)
        assert m["AC@3"] == 1.0

    def test_avg5_rank_based(self):
        ranked = [(2, 5.0), (0, 3.0)]
        m = compute_metrics(ranked, root_cause=2)
        assert m["Avg@5"] == pytest.approx(1.0)   # rank=0 → 1/(0+1)=1

    def test_f1_perfect(self):
        ranked = [(2, 5.0)]
        m = compute_metrics(ranked, root_cause=2)
        assert m["F1"] > 0.9

    def test_f1_zero_miss(self):
        ranked = [(0, 5.0), (1, 3.0)]
        m = compute_metrics(ranked, root_cause=2)
        assert m["F1"] == pytest.approx(0.0, abs=0.01)

    def test_all_keys_present(self):
        ranked = [(0, 1.0)]
        m = compute_metrics(ranked, root_cause=0)
        assert set(m.keys()) == {"AC@1", "AC@3", "AC@5", "Avg@5", "F1"}


# ===========================================================================
# TestCAIRNTrainer — один проход (1 эпоха каждого этапа)
# ===========================================================================

class TestCAIRNTrainer:
    """Быстрые тесты тренера: 1 эпоха каждого этапа на синтетических данных."""

    @pytest.fixture
    def trainer(self, model, loss_fn, hypergraph):
        cfg = TrainerConfig(
            pretrain_epochs=1,
            main_epochs=1,
            finetune_epochs=1,
            freeze_epochs=0,
            patience=100,         # отключаем early stopping
            log_every=1,
            device="cpu",
            checkpoint_dir="/tmp/cairn_test_ckpt",
            save_every=999,
        )
        return CAIRNTrainer(model, loss_fn, hypergraph, cfg)

    @pytest.fixture
    def small_dataset(self, fake_incident, normal_incident):
        return CAIRNDataset(
            [normal_incident, normal_incident, fake_incident, fake_incident]
        )

    def test_train_returns_history(self, trainer, small_dataset):
        history = trainer.train(small_dataset)
        assert isinstance(history, dict)
        assert "pretrain_loss" in history
        assert "main_loss" in history
        assert "finetune_loss" in history

    def test_pretrain_loss_recorded(self, trainer, small_dataset):
        history = trainer.train(small_dataset)
        assert len(history["pretrain_loss"]) >= 1

    def test_main_loss_recorded(self, trainer, small_dataset):
        history = trainer.train(small_dataset)
        assert len(history["main_loss"]) >= 1

    def test_finetune_loss_recorded(self, trainer, small_dataset):
        history = trainer.train(small_dataset)
        assert len(history["finetune_loss"]) >= 1

    def test_evaluate_returns_metrics(self, trainer, small_dataset):
        trainer.train(small_dataset)
        metrics = trainer.evaluate(small_dataset)
        assert set(metrics.keys()) == {"AC@1", "AC@3", "AC@5", "Avg@5", "F1"}

    def test_metrics_in_range(self, trainer, small_dataset):
        trainer.train(small_dataset)
        metrics = trainer.evaluate(small_dataset)
        for k, v in metrics.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} вне диапазона [0,1]"

    def test_save_and_load(self, trainer, small_dataset, tmp_path):
        trainer.train(small_dataset)
        ckpt = tmp_path / "test.pt"
        trainer.save(ckpt)
        assert ckpt.exists()
        # Загружаем обратно
        trainer.load(ckpt)

    def test_model_parameters_changed_after_train(self, trainer, small_dataset):
        """Параметры модели должны измениться после обучения."""
        params_before = {k: v.clone() for k, v in trainer.model.named_parameters()}
        trainer.train(small_dataset)
        params_after = dict(trainer.model.named_parameters())
        changed = any(
            not torch.allclose(params_before[k], params_after[k])
            for k in params_before
        )
        assert changed, "Параметры модели не изменились после обучения"


# ===========================================================================
# TestTrainingOnDemoData — интеграционный тест на реальных данных
# ===========================================================================

@pytest.mark.skipif(
    not (SAMPLE_DIR / "metrics.csv").exists(),
    reason="Демо-данные отсутствуют. Запустите: python scripts/generate_demo_data.py",
)
class TestTrainingOnDemoData:
    """Один проход обучения на демо-данных (1 эпоха каждого этапа)."""

    @pytest.fixture(scope="class")
    def demo_dataset(self):
        return create_demo_dataset(SAMPLE_DIR, window_size=30, stride=10)

    @pytest.fixture(scope="class")
    def demo_hypergraph(self):
        from cairn.connectors.csv_file import YAMLTopologyConnector
        topo = YAMLTopologyConnector(SAMPLE_DIR / "topology.yaml").fetch()
        return HypergraphBuilder.from_topology_data(topo)

    @pytest.fixture(scope="class")
    def demo_trainer(self, demo_dataset, demo_hypergraph):
        N_inst = demo_hypergraph.n_nodes
        F_inst = demo_dataset[0].metric_data.shape[2]
        D = 32
        CTX = 8
        model = CAIRNModel(
            state_builder=StateBuilder(
                n_metrics=F_inst, log_vocab_size=300,
                state_dim=D, context_dim=CTX,
                d_met=16, d_log=8, d_tr=8,
                d_ssm=8, d_brk=8, ssm_state_dim=16, window=15,
            ),
            gmm=ConditionalGMM(state_dim=D, context_dim=CTX, n_components=3),
            vgae=ConfoundedVGAE(state_dim=D, n_confounders=2, confounder_dim=8),
            cf_module=CounterfactualModule(state_dim=D, n_conv_layers=1),
        )
        cfg = TrainerConfig(
            pretrain_epochs=1, main_epochs=1, finetune_epochs=1,
            freeze_epochs=0, patience=100, log_every=1, device="cpu",
            checkpoint_dir="/tmp/cairn_demo_ckpt", save_every=999,
        )
        return CAIRNTrainer(model, CAIRNLoss(), demo_hypergraph, cfg)

    def test_dataset_created(self, demo_dataset):
        assert len(demo_dataset) > 0
        assert demo_dataset.n_anomaly > 0
        assert demo_dataset.n_normal > 0

    def test_dataset_incident_shape(self, demo_dataset, demo_hypergraph):
        inc = demo_dataset[0]
        assert inc.metric_data.shape[0] == demo_hypergraph.n_nodes

    def test_train_one_epoch(self, demo_trainer, demo_dataset):
        history = demo_trainer.train(demo_dataset)
        assert len(history["pretrain_loss"]) == 1
        assert len(history["main_loss"]) == 1
        assert len(history["finetune_loss"]) == 1

    def test_all_losses_finite(self, demo_trainer, demo_dataset):
        history = demo_trainer.train(demo_dataset)
        for stage in ("pretrain_loss", "main_loss", "finetune_loss"):
            for v in history[stage]:
                assert not (v != v), f"{stage} содержит NaN"  # NaN check

    def test_evaluate_after_train(self, demo_trainer, demo_dataset):
        demo_trainer.train(demo_dataset)
        metrics = demo_trainer.evaluate(demo_dataset)
        assert "AC@1" in metrics
        for v in metrics.values():
            assert 0.0 <= v <= 1.0
