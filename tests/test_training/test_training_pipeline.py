"""Тесты системы обучения CAIRN."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
import pytest, torch
from cairn.training import (
    CAIRNLoss, LossWeights,
    Incident, CAIRNDataset, collate_incidents,
    CAIRNModel, CAIRNTrainer, TrainerConfig, compute_metrics,
)
SAMPLE_DIR = Path(__file__).parent.parent.parent / "data" / "sample"
N, D, CTX, T_WIN, F, L = 5, 32, 16, 20, 4, 10

@pytest.fixture
def loss_fn():
    return CAIRNLoss(weights=LossWeights(), margin=0.1, tcd_margin=0.3)

@pytest.fixture
def dummy_incident():
    return Incident(
        metric_data=torch.randn(N, T_WIN, F), log_data=torch.randint(0,50,(N,L)),
        trace_data=torch.zeros(N, dtype=torch.long), context=torch.randn(N, CTX),
        root_cause=1, fault_type="cpu_exhaustion",
        instance_names=[f"svc-{i}" for i in range(N)], is_anomaly=True,
    )

@pytest.fixture
def dummy_normal():
    return Incident(
        metric_data=torch.randn(N, T_WIN, F), log_data=torch.zeros(N,L,dtype=torch.long),
        trace_data=torch.zeros(N, dtype=torch.long), context=torch.randn(N, CTX),
        root_cause=-1, fault_type="normal",
        instance_names=[f"svc-{i}" for i in range(N)], is_anomaly=False,
    )

@pytest.fixture
def dataset(dummy_incident, dummy_normal):
    return CAIRNDataset([dummy_normal]*5 + [dummy_incident]*3)

@pytest.fixture
def hypergraph():
    from cairn.perception import HypergraphBuilder
    return HypergraphBuilder(N).from_topology(
        call_paths=[[0,1],[1,2],[1,3]], colocated_groups=[[1,4]], lb_groups=[],
        instance_names=[f"svc-{i}" for i in range(N)],
    )

@pytest.fixture
def model():
    from cairn.perception import StateBuilder
    from cairn.reasoning import ConditionalGMM, ConfoundedVGAE, CounterfactualModule
    return CAIRNModel(
        state_builder=StateBuilder(n_metrics=F, log_vocab_size=100, state_dim=D, context_dim=CTX,
            d_met=16, d_log=8, d_tr=8, d_ssm=8, d_brk=8, ssm_state_dim=16, window=10),
        gmm=ConditionalGMM(state_dim=D, context_dim=CTX, n_components=2),
        vgae=ConfoundedVGAE(state_dim=D, n_confounders=2, confounder_dim=8),
        cf_module=CounterfactualModule(state_dim=D, n_conv_layers=1),
    )

# --- CAIRNLoss ---
class TestCAIRNLoss:
    def test_loss_pe_zero_when_root_highest(self, loss_fn):
        pe = torch.tensor([10.0, 1.0, 2.0, 3.0, 4.0])
        assert loss_fn.loss_pe(pe, 0).item() < 1e-5

    def test_loss_pe_positive_when_root_not_highest(self, loss_fn):
        pe = torch.tensor([1.0, 5.0, 2.0, 3.0, 4.0])
        assert loss_fn.loss_pe(pe, 0).item() > 0.0

    def test_loss_um_shape(self, loss_fn):
        assert loss_fn.loss_um(torch.randn(20).abs(), []).shape == ()

    def test_loss_vak_shape(self, loss_fn):
        h = torch.randn(N, D)
        assert loss_fn.loss_vak(h, h, torch.randn(N,D), torch.randn(N,D), []).shape == ()

    def test_loss_nez_range(self, loss_fn):
        l = loss_fn.loss_nez(torch.randn(N, D))
        assert 0.0 <= l.item() <= 1.0 + 1e-5

    def test_loss_nez_single(self, loss_fn):
        assert loss_fn.loss_nez(torch.randn(1, D)).item() == 0.0

    def test_loss_kr_shape(self, loss_fn):
        assert loss_fn.loss_kr(
            torch.randn(D), torch.randn(D),
            torch.randn(N-1,D), torch.randn(N-1,D)
        ).shape == ()

    def test_forward_returns_tuple(self, loss_fn):
        outputs = dict(
            pe_scores=torch.randn(N), nll_normal=torch.randn(N).abs(),
            cov_matrices=[], h=torch.randn(N,D), h_recon=torch.randn(N,D),
            mu_u=torch.randn(N,D), log_var_u=torch.randn(N,D), kl_z_terms=[],
            u_hat=torch.randn(N,D), h_root_anom=torch.randn(D), h_root_norm=torch.randn(D),
            h_others_anom=torch.randn(N-1,D), h_others_norm=torch.randn(N-1,D),
            edge_weights=None, edge_cf_stats=None,
        )
        total, comps = loss_fn(outputs, {"root_idx": 0})
        assert isinstance(total, torch.Tensor) and isinstance(comps, dict)
        assert all(k in comps for k in ("L_pe","L_um","L_vak","L_nez","L_kr","L_reb","loss"))

    def test_update_weights_adaptive(self):
        loss = CAIRNLoss(adaptive=True)
        before = loss.w.lambda_pe
        loss.update_weights({k: 10.0 if k=="L_pe" else 0.1 for k in CAIRNLoss.COMPONENT_NAMES})
        assert loss.w.lambda_pe != before

    def test_update_weights_no_op_when_not_adaptive(self, loss_fn):
        before = loss_fn.w.lambda_pe
        loss_fn.update_weights({"L_pe": 999.0})
        assert loss_fn.w.lambda_pe == before

    def test_pretrain_loss(self, loss_fn):
        total, comps = loss_fn.pretrain_loss(
            torch.randn(N).abs(), torch.randn(N,D), torch.randn(N,D),
            torch.randn(N,D), torch.randn(N,D),
        )
        assert torch.isfinite(total) and "L_um" in comps

    def test_main_loss(self, loss_fn):
        total, comps = loss_fn.main_loss(
            pe_scores=torch.randn(N), root_idx=0, u_hat=torch.randn(N,D),
            h_root_anom=torch.randn(D), h_root_norm=torch.randn(D),
            h_others_anom=torch.randn(N-1,D), h_others_norm=torch.randn(N-1,D),
        )
        assert torch.isfinite(total) and "L_pe" in comps

# --- DataLoader ---
class TestDataLoader:
    def test_incident_shapes(self, dummy_incident):
        assert dummy_incident.metric_data.shape == (N, T_WIN, F)
        assert dummy_incident.log_data.shape    == (N, L)
        assert dummy_incident.root_cause == 1 and dummy_incident.is_anomaly

    def test_dataset_subsets(self, dataset):
        assert len(dataset.normal_subset())  == 5
        assert len(dataset.anomaly_subset()) == 3

    def test_collate(self, dummy_incident):
        c = collate_incidents([dummy_incident, dummy_incident])
        assert c["metric_data"].shape == (2, N, T_WIN, F)
        assert c["root_cause"].shape  == (2,)

    def test_compute_metrics_hit(self):
        m = compute_metrics([(2,5.),(0,3.),(1,2.)], root_cause=2)
        assert m["AC@1"] == 1.0 and m["AC@3"] == 1.0

    def test_compute_metrics_miss(self):
        m = compute_metrics([(0,5.),(1,3.),(2,2.)], root_cause=4)
        assert m["AC@1"] == 0.0 and m["F1"] == 0.0

    def test_compute_metrics_avg5(self):
        m = compute_metrics([(4,5.),(2,4.),(1,3.)], root_cause=2)
        assert m["Avg@5"] == pytest.approx(0.5)

# --- Trainer ---
class TestCAIRNTrainer:
    @pytest.fixture
    def trainer(self, model, loss_fn, hypergraph):
        cfg = TrainerConfig(pretrain_epochs=1, main_epochs=1, finetune_epochs=1,
                            freeze_epochs=0, patience=100, device="cpu",
                            checkpoint_dir="/tmp/cairn_test")
        return CAIRNTrainer(model, loss_fn, hypergraph, cfg)

    def test_model_forward_shapes(self, model, dummy_incident, hypergraph):
        model.eval()
        with torch.no_grad():
            out = model(dummy_incident, hypergraph)
        assert out["H"].shape == (N, D)
        assert out["nll_normal"].shape == (N,)
        assert not torch.isnan(out["H"]).any()

    def test_full_train_one_epoch(self, trainer, dataset):
        history = trainer.train(dataset)
        assert len(history["pretrain_loss"]) >= 1
        assert len(history["main_loss"])     >= 1
        assert len(history["finetune_loss"]) >= 1

    def test_evaluate_returns_valid_metrics(self, trainer, dataset):
        metrics = trainer.evaluate(dataset)
        for k in ("AC@1","AC@3","AC@5","Avg@5","F1"):
            assert k in metrics and 0.0 <= metrics[k] <= 1.0

    def test_save_load(self, trainer, dataset, tmp_path):
        trainer.train(dataset)
        p = tmp_path / "ckpt.pt"
        trainer.save(p)
        assert p.exists()
        trainer.load(p)

# --- Demo pipeline ---
@pytest.mark.skipif(
    not (SAMPLE_DIR/"metrics.csv").exists(),
    reason="Нет демо-данных. Запустите: python scripts/generate_demo_data.py",
)
class TestTrainingPipeline:
    @pytest.fixture(scope="class")
    def demo_ds(self):
        from cairn.training import create_demo_dataset
        return create_demo_dataset(SAMPLE_DIR, window_size=30, stride=20)

    @pytest.fixture(scope="class")
    def demo_hg(self):
        from cairn.connectors.csv_file import YAMLTopologyConnector
        from cairn.perception import HypergraphBuilder
        topo = YAMLTopologyConnector(SAMPLE_DIR/"topology.yaml").fetch()
        return HypergraphBuilder.from_topology_data(topo)

    def test_dataset_has_normal_and_anomaly(self, demo_ds):
        assert demo_ds.n_normal > 0 and demo_ds.n_anomaly > 0

    def test_incident_shapes(self, demo_ds):
        inc = demo_ds[0]
        assert inc.metric_data.ndim == 3
        assert len(inc.instance_names) == inc.n_instances

    def test_one_epoch_all_stages(self, demo_ds, demo_hg):
        from cairn.perception import StateBuilder
        from cairn.reasoning import ConditionalGMM, ConfoundedVGAE, CounterfactualModule
        inc = demo_ds[0]; _, _, n_metrics = inc.metric_data.shape
        mdl = CAIRNModel(
            state_builder=StateBuilder(n_metrics=n_metrics, log_vocab_size=300, state_dim=32,
                context_dim=CTX, d_met=16, d_log=8, d_tr=8, d_ssm=8, d_brk=8,
                ssm_state_dim=16, window=10),
            gmm=ConditionalGMM(state_dim=32, context_dim=CTX, n_components=2),
            vgae=ConfoundedVGAE(state_dim=32, n_confounders=2, confounder_dim=8),
            cf_module=CounterfactualModule(state_dim=32, n_conv_layers=1),
        )
        trainer = CAIRNTrainer(mdl, CAIRNLoss(), demo_hg,
                               TrainerConfig(pretrain_epochs=1, main_epochs=1, finetune_epochs=1,
                                             freeze_epochs=0, patience=100, checkpoint_dir="/tmp/ct"))
        h = trainer.train(demo_ds)
        assert len(h["pretrain_loss"]) >= 1

    def test_evaluate_on_demo(self, demo_ds, demo_hg):
        from cairn.perception import StateBuilder
        from cairn.reasoning import ConditionalGMM, ConfoundedVGAE, CounterfactualModule
        inc = demo_ds[0]; _, _, n_metrics = inc.metric_data.shape
        mdl = CAIRNModel(
            state_builder=StateBuilder(n_metrics=n_metrics, log_vocab_size=300, state_dim=32,
                context_dim=CTX, d_met=16, d_log=8, d_tr=8, d_ssm=8, d_brk=8,
                ssm_state_dim=16, window=10),
            gmm=ConditionalGMM(state_dim=32, context_dim=CTX, n_components=2),
            vgae=ConfoundedVGAE(state_dim=32, n_confounders=2, confounder_dim=8),
            cf_module=CounterfactualModule(state_dim=32, n_conv_layers=1),
        )
        trainer = CAIRNTrainer(mdl, CAIRNLoss(), demo_hg,
                               TrainerConfig(checkpoint_dir="/tmp/ce"))
        m = trainer.evaluate(demo_ds)
        assert all(k in m for k in ("AC@1","F1"))
