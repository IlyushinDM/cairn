"""Трёхэтапный тренер CAIRN (раздел 5.2).

Этап 1 – Претрейн (normal_only):    L_УМ + L_ВАК
Этап 2 – Основное (anomaly_only):   L_ПЭ + L_нез + L_КР + L_реб
Этап 3 – Файнтюн (all):             L = λ₁·L_ПЭ + … + λ₆·L_реб

Функции:
  train()    – полный цикл обучения
  evaluate() – метрики AC@1, AC@3, Avg@5, F1
  save() / load() – чекпоинты
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from cairn.training.data_loader import CAIRNDataset, Incident, collate_incidents
from cairn.training.loss import CAIRNLoss, LossWeights


# ---------------------------------------------------------------------------
# Конфигурация тренера
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """Гиперпараметры обучения."""
    # Этапы
    pretrain_epochs:  int   = 50
    main_epochs:      int   = 100
    finetune_epochs:  int   = 30
    freeze_epochs:    int   = 10      # эпох заморозки в начале этапа 2
    # Оптимизация
    lr:               float = 1e-3
    weight_decay:     float = 1e-4
    batch_size:       int   = 1       # батч = 1 инцидент (N узлов)
    # Ранняя остановка
    patience:         int   = 10
    min_delta:        float = 1e-4
    # Чекпоинты
    checkpoint_dir:   str   = "checkpoints"
    save_every:       int   = 10
    # Устройство
    device:           str   = "cpu"
    # Логирование
    log_every:        int   = 5


# ---------------------------------------------------------------------------
# Вспомогательная модель-обёртка
# ---------------------------------------------------------------------------

class CAIRNModel(nn.Module):
    """Объединяет все обучаемые компоненты CAIRN.

    Параметры
    ----------
    state_builder  : StateBuilder
    gmm            : ConditionalGMM
    vgae           : ConfoundedVGAE
    cf_module      : CounterfactualModule
    hypergraph     : CausalHypergraph (фиксированный для датасета)
    """

    def __init__(self, state_builder, gmm, vgae, cf_module) -> None:
        super().__init__()
        self.state_builder = state_builder
        self.gmm           = gmm
        self.vgae          = vgae
        self.cf_module     = cf_module

    def forward(self, incident: Incident, hypergraph) -> Dict[str, torch.Tensor]:
        """Полный forward-pass на одном инциденте.

        Возвращает словарь outputs для CAIRNLoss.forward().
        """
        dev = next(self.parameters()).device
        metrics = incident.metric_data.to(dev)    # (N, T, F)
        log_ids = incident.log_data.to(dev)       # (N, L)
        depths  = incident.trace_data.to(dev)     # (N,)
        ctx_raw = incident.context.to(dev)        # (N, C)

        # Фаза восприятия
        H, C = self.state_builder(metrics, log_ids, depths, context_raw=ctx_raw)  # (N, d), (N, 16)

        # Нормальное состояние
        nll_normal = self.gmm.nll(H, C)     # (N,) – аномальность каждого узла

        # Абдукция
        inc_mat   = hypergraph.incidence_matrix().to(dev)
        edge_wts  = hypergraph.edge_weights().to(dev)
        edge_idx  = self._incidence_to_edge_index(inc_mat, dev)
        edge_type = torch.zeros(edge_idx.shape[1], dtype=torch.long, device=dev)

        exog, _, _, kl_loss = self.vgae.encode(H, edge_idx, edge_type)
        h_recon = self.vgae.decode(exog)

        # Прототипы для каждого узла
        prototypes = self.gmm.prototype(C)   # (N, d)

        # Причинные эффекты через контрфактику
        root_idx = incident.root_cause
        if root_idx >= 0 and root_idx < H.shape[0]:
            H_cf   = self.cf_module.intervene(H, root_idx, prototypes[root_idx], hypergraph)
            nll_cf = self.gmm.nll(H_cf, C)
            pe_scores = (nll_normal - nll_cf)            # (N,) – градиент сохранён

            # Для L_КР: первопричина vs остальные
            others = [i for i in range(H.shape[0]) if i != root_idx]
            h_root_anom   = H[root_idx]
            h_root_norm   = prototypes[root_idx]
            h_others_anom = H[others] if others else H[:0]
            h_others_norm = prototypes[others] if others else prototypes[:0]
        else:
            N = H.shape[0]
            pe_scores     = torch.zeros(N, device=dev)
            h_root_anom   = H[0]
            h_root_norm   = prototypes[0]
            h_others_anom = H[1:] if N > 1 else H[:0]
            h_others_norm = prototypes[1:] if N > 1 else prototypes[:0]

        # Ковариационные матрицы GMM (диагональные – возвращаем как identity для простоты)
        _, _, log_vars = self.gmm(C)        # (N, D, d)
        cov_matrices = [torch.diag(log_vars[0, k].exp()) for k in range(log_vars.shape[1])]

        # mu_u и log_var_u из VGAE encode
        _, mu_u, log_var_u = self.vgae.exogenous_enc(
            H,
            torch.zeros(H.shape[0], dtype=torch.long, device=dev),
            edge_idx,
            self.vgae._edge_type_to_feats(edge_type, dev, H.dtype),
        )

        return {
            "H":             H,
            "C":             C,
            "nll_normal":    nll_normal,
            "pe_scores":     pe_scores,
            "h":             H,
            "h_recon":       h_recon,
            "mu_u":          mu_u,
            "log_var_u":     log_var_u,
            "kl_z_terms":    [],
            "u_hat":         exog,
            "h_root_anom":   h_root_anom,
            "h_root_norm":   h_root_norm,
            "h_others_anom": h_others_anom,
            "h_others_norm": h_others_norm,
            "cov_matrices":  cov_matrices,
            "edge_weights":  None,
            "edge_cf_stats": None,
        }

    @staticmethod
    def _incidence_to_edge_index(H: torch.Tensor, device) -> torch.Tensor:
        """Преобразует матрицу инцидентности в формат edge_index (2, E)."""
        nz = H.nonzero(as_tuple=True)
        if len(nz[0]) == 0:
            return torch.zeros(2, 0, dtype=torch.long, device=device)
        # node→hyperedge → превращаем в попарные рёбра через общее гиперребро
        node_idx, edge_idx = nz
        return torch.stack([node_idx, edge_idx])


# ---------------------------------------------------------------------------
# Метрики оценки
# ---------------------------------------------------------------------------

def compute_metrics(
    ranked_results: List[Tuple[int, float]],   # [(node_idx, ce), ...]
    root_cause: int,
) -> Dict[str, float]:
    """AC@k, Avg@5, F1 для одного инцидента."""
    ranked_idxs = [idx for idx, _ in ranked_results]
    k = len(ranked_idxs)

    ac1 = float(root_cause in ranked_idxs[:1]) if k >= 1 else 0.0
    ac3 = float(root_cause in ranked_idxs[:3]) if k >= 3 else float(root_cause in ranked_idxs)
    ac5 = float(root_cause in ranked_idxs[:5]) if k >= 5 else float(root_cause in ranked_idxs)

    # Средний ранг в top-5 (avg@5 = 1/(rank+1) если найден, иначе 0)
    avg5 = 0.0
    if root_cause in ranked_idxs[:5]:
        rank = ranked_idxs[:5].index(root_cause)
        avg5 = 1.0 / (rank + 1)

    # F1: precision=AC@1, recall=1.0 (предполагаем 1 первопричину)
    prec = ac1
    rec  = 1.0 if root_cause in ranked_idxs else 0.0
    f1   = 2 * prec * rec / (prec + rec + 1e-8)

    return {"AC@1": ac1, "AC@3": ac3, "AC@5": ac5, "Avg@5": avg5, "F1": f1}


# ---------------------------------------------------------------------------
# CAIRNTrainer
# ---------------------------------------------------------------------------

class CAIRNTrainer:
    """Трёхэтапный тренер CAIRN.

    Параметры
    ----------
    model : CAIRNModel
    loss_fn : CAIRNLoss
    hypergraph : CausalHypergraph – фиксированный для всего датасета
    config : TrainerConfig
    """

    def __init__(
        self,
        model: CAIRNModel,
        loss_fn: CAIRNLoss,
        hypergraph,
        config: Optional[TrainerConfig] = None,
    ) -> None:
        self.model     = model
        self.loss_fn   = loss_fn
        self.hypergraph = hypergraph
        self.cfg       = config or TrainerConfig()
        self.device    = torch.device(self.cfg.device)

        self.model.to(self.device)
        self.history: Dict[str, List] = {
            "pretrain_loss": [], "main_loss": [], "finetune_loss": [],
            "eval_ac1": [], "eval_ac3": [], "eval_f1": [],
        }

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def train(self, dataset: CAIRNDataset) -> Dict[str, List]:
        """Полный трёхэтапный цикл обучения.

        Возвращает
        ----------
        history : dict с историей потерь и метрик по эпохам.
        """
        from loguru import logger
        logger.info(f"Устройство: {self.device}")
        logger.info(dataset.summary())

        # Этап 1: Претрейн
        logger.info("=== Этап 1: Претрейн (L_УМ + L_ВАК) ===")
        self._pretrain(dataset.normal_subset())

        # Этап 2: Основное обучение
        logger.info("=== Этап 2: Основное обучение ===")
        self._main_train(dataset.anomaly_subset(), dataset.normal_subset())

        # Этап 3: Файнтюн
        logger.info("=== Этап 3: Файнтюн (все компоненты) ===")
        self._finetune(dataset)

        return self.history

    def evaluate(self, dataset: CAIRNDataset) -> Dict[str, float]:
        """Оценка на тестовом наборе аномальных инцидентов.

        Метрики: AC@1, AC@3, AC@5, Avg@5, F1.
        """
        from cairn.reasoning import ConditionalGMM, CounterfactualModule, CascadeFunnel

        self.model.eval()
        anom_data = dataset.anomaly_subset()
        if len(anom_data) == 0:
            return {"AC@1": 0.0, "AC@3": 0.0, "AC@5": 0.0, "Avg@5": 0.0, "F1": 0.0}

        funnel = CascadeFunnel(
            l0_top_k=30, 
            l1_top_k=5, 
            l2_top_k=5
        )
        all_metrics: List[Dict[str, float]] = []

        with torch.no_grad():
            for incident in anom_data:
                if incident.root_cause < 0:
                    continue
                outputs = self.model(incident, self.hypergraph)
                H, C    = outputs["H"], outputs["C"]
                nll     = self.model.gmm.nll(H, C)
                adj     = self.hypergraph.adjacency_matrix().to(self.device)
                adj_norm= adj / adj.sum(1, keepdim=True).clamp(min=1)

                ranked = funnel.run(
                    nll, H, adj_norm,
                    self.model.cf_module,
                    self.model.gmm,
                    C, self.hypergraph,
                )
                m = compute_metrics(ranked, incident.root_cause)
                all_metrics.append(m)

        if not all_metrics:
            return {"AC@1": 0.0, "AC@3": 0.0, "AC@5": 0.0, "Avg@5": 0.0, "F1": 0.0}

        avg = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]}
        return avg

    def save(self, path: str | Path) -> None:
        """Сохраняет чекпоинт модели и конфигурацию тренера."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "history": self.history,
            "loss_weights": vars(self.loss_fn.w),
        }, path)

    def load(self, path: str | Path) -> None:
        """Загружает чекпоинт."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.history = ckpt.get("history", self.history)

    # ------------------------------------------------------------------
    # Внутренние методы этапов
    # ------------------------------------------------------------------

    def _make_optimizer(self, params=None) -> Adam:
        params = params or self.model.parameters()
        return Adam(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    def _pretrain(self, normal_ds: CAIRNDataset) -> None:
        """Этап 1: нормальные данные, L_УМ + L_ВАК."""
        if len(normal_ds) == 0:
            return
        opt = self._make_optimizer()
        best_loss, patience_cnt = float("inf"), 0

        for epoch in range(self.cfg.pretrain_epochs):
            self.model.train()
            epoch_loss = self._run_epoch_pretrain(normal_ds, opt)
            self.history["pretrain_loss"].append(epoch_loss)

            # Early stopping
            if epoch_loss < best_loss - self.cfg.min_delta:
                best_loss, patience_cnt = epoch_loss, 0
            else:
                patience_cnt += 1
                if patience_cnt >= self.cfg.patience:
                    break

            if (epoch + 1) % self.cfg.log_every == 0:
                self._log(f"Претрейн эп.{epoch+1}: loss={epoch_loss:.4f}")

    def _main_train(self, anom_ds: CAIRNDataset, normal_ds: CAIRNDataset) -> None:
        """Этап 2: аномальные данные, L_ПЭ + L_нез + L_КР + L_реб."""
        if len(anom_ds) == 0:
            return
        opt = self._make_optimizer()
        best_loss, patience_cnt = float("inf"), 0

        # Заморозка StateBuilder на первых freeze_epochs
        freeze = self.cfg.freeze_epochs
        self._set_frozen(self.model.state_builder, True)

        for epoch in range(self.cfg.main_epochs):
            if epoch == freeze:
                self._set_frozen(self.model.state_builder, False)
                opt = self._make_optimizer()

            self.model.train()
            epoch_loss = self._run_epoch_main(anom_ds, opt)
            self.history["main_loss"].append(epoch_loss)

            if epoch_loss < best_loss - self.cfg.min_delta:
                best_loss, patience_cnt = epoch_loss, 0
            else:
                patience_cnt += 1
                if patience_cnt >= self.cfg.patience:
                    self._set_frozen(self.model.state_builder, False)
                    break

            if (epoch + 1) % self.cfg.log_every == 0:
                self._log(f"Основное эп.{epoch+1}: loss={epoch_loss:.4f}")

        self._set_frozen(self.model.state_builder, False)

    def _finetune(self, dataset: CAIRNDataset) -> None:
        """Этап 3: все данные, все компоненты потерь."""
        if len(dataset) == 0:
            return
        opt = self._make_optimizer()
        sched = CosineAnnealingLR(opt, T_max=self.cfg.finetune_epochs)
        best_loss, patience_cnt = float("inf"), 0

        for epoch in range(self.cfg.finetune_epochs):
            self.model.train()
            epoch_loss = self._run_epoch_finetune(dataset, opt)
            if epoch > 0:
                sched.step()
            self.history["finetune_loss"].append(epoch_loss)

            if epoch_loss < best_loss - self.cfg.min_delta:
                best_loss, patience_cnt = epoch_loss, 0
                # Сохраняем лучший чекпоинт
                ckpt_dir = Path(self.cfg.checkpoint_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                self.save(ckpt_dir / "best.pt")
            else:
                patience_cnt += 1
                if patience_cnt >= self.cfg.patience:
                    break

            if (epoch + 1) % self.cfg.log_every == 0:
                self._log(f"Файнтюн эп.{epoch+1}: loss={epoch_loss:.4f}")

    # ------------------------------------------------------------------
    # Прогоны эпох
    # ------------------------------------------------------------------

    def _run_epoch_pretrain(self, ds: CAIRNDataset, opt) -> float:
        total, count = 0.0, 0
        for incident in ds:
            opt.zero_grad()
            try:
                outputs = self.model(incident, self.hypergraph)
                loss, comps = self.loss_fn.pretrain_loss(
                    nll_normal=outputs["nll_normal"],
                    h=outputs["h"],
                    h_recon=outputs["h_recon"],
                    mu_u=outputs["mu_u"],
                    log_var_u=outputs["log_var_u"],
                    kl_z_terms=outputs.get("kl_z_terms", []),
                    cov_matrices=outputs.get("cov_matrices", []),
                )
                if torch.isfinite(loss):
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()
                    total += loss.item()
                    count += 1
            except Exception:
                continue
        return total / max(count, 1)

    def _run_epoch_main(self, ds: CAIRNDataset, opt) -> float:
        total, count = 0.0, 0
        for incident in ds:
            if not incident.is_anomaly or incident.root_cause < 0:
                continue
            opt.zero_grad()
            try:
                outputs = self.model(incident, self.hypergraph)
                loss, comps = self.loss_fn.main_loss(
                    pe_scores=outputs["pe_scores"],
                    root_idx=incident.root_cause,
                    u_hat=outputs["u_hat"],
                    h_root_anom=outputs["h_root_anom"],
                    h_root_norm=outputs["h_root_norm"],
                    h_others_anom=outputs["h_others_anom"],
                    h_others_norm=outputs["h_others_norm"],
                )
                if torch.isfinite(loss):
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()
                    if self.loss_fn.adaptive:
                        self.loss_fn.update_weights({k: v.item() for k, v in comps.items()})
                    total += loss.item()
                    count += 1
            except Exception:
                continue
        return total / max(count, 1)

    def _run_epoch_finetune(self, ds: CAIRNDataset, opt) -> float:
        total, count = 0.0, 0
        for incident in ds:
            opt.zero_grad()
            try:
                outputs = self.model(incident, self.hypergraph)
                targets = {"root_idx": max(incident.root_cause, 0)}

                if incident.is_anomaly and incident.root_cause >= 0:
                    loss, comps = self.loss_fn(outputs, targets)
                else:
                    # Нормальный пример: только L_УМ + L_ВАК
                    loss, comps = self.loss_fn.pretrain_loss(
                        outputs["nll_normal"], outputs["h"], outputs["h_recon"],
                        outputs["mu_u"], outputs["log_var_u"],
                    )

                if torch.isfinite(loss):
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()
                    total += loss.item()
                    count += 1
            except Exception:
                continue
        return total / max(count, 1)

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    @staticmethod
    def _set_frozen(module: nn.Module, frozen: bool) -> None:
        for p in module.parameters():
            p.requires_grad = not frozen

    @staticmethod
    def _log(msg: str) -> None:
        try:
            from loguru import logger
            logger.info(msg)
        except ImportError:
            print(msg)
