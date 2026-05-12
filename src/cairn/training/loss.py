"""Композитная функция потерь CAIRN (раздел 5.1, формула 5.1).

L = λ₁·L_ПЭ + λ₂·L_УМ + λ₃·L_ВАК + λ₄·L_нез + λ₅·L_КР + λ₆·L_реб

Адаптивное перевзвешивание: λ_k обновляются через экспоненциальное
скользящее среднее потерь – компоненты с высокой потерей получают больший вес.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    """Начальные веса λ₁–λ₆ (формула 5.1)."""
    lambda_pe:  float = 1.0   # L_ПЭ – ранжирование причинных эффектов
    lambda_um:  float = 1.0   # L_УМ – условная модель нормального состояния
    lambda_vak: float = 1.0   # L_ВАК – вариационный автокодировщик
    lambda_nez: float = 0.5   # L_нез – ограничение независимости
    lambda_kr:  float = 0.5   # L_КР – контрастное разделение
    lambda_reb: float = 0.1   # L_реб – штраф за необоснованные рёбра


class CAIRNLoss(nn.Module):
    """Композитная функция потерь CAIRN.

    Поддерживает два режима вызова:
      1. Полный forward(outputs, targets) – для тренера
      2. Отдельные методы loss_pe(), loss_um() и т.д. – для ступенчатого обучения

    Параметры
    ----------
    weights : LossWeights
        Начальные веса λ₁–λ₆.
    margin : float
        Отступ для L_ПЭ (формула 5.2).
    tcd_margin : float
        δ_KR для L_КР (формула 5.6).
    beta_kl : float
        β – KL для экзогенных переменных (формула 5.4).
    beta_kl_z : float
        β_z – KL для скрытых факторов.
    cov_reg : float
        λ_рег – регуляризация ковариации (формула 5.3).
    adaptive : bool
        Включить адаптивное перевзвешивание через EMA.
    ema_decay : float
        Коэффициент затухания EMA (по умолчанию 0.9).
    """

    COMPONENT_NAMES = ("L_pe", "L_um", "L_vak", "L_nez", "L_kr", "L_reb")
    WEIGHT_ATTRS    = ("lambda_pe", "lambda_um", "lambda_vak", "lambda_nez", "lambda_kr", "lambda_reb")

    def __init__(
        self,
        weights: Optional[LossWeights] = None,
        initial_weights: Optional[List[float]] = None,
        margin: float = 0.1,
        tcd_margin: float = 0.3,
        beta_kl: float = 1.0,
        beta_kl_z: float = 0.1,
        cov_reg: float = 1e-4,
        adaptive: bool = False,
        ema_decay: float = 0.9,
    ) -> None:
        super().__init__()

        # Поддержка обоих форматов задания весов
        if initial_weights is not None and weights is None:
            fields = ("lambda_pe", "lambda_um", "lambda_vak", "lambda_nez", "lambda_kr", "lambda_reb")
            kw = dict(zip(fields, initial_weights))
            weights = LossWeights(**kw)

        self.w           = weights or LossWeights()
        self.margin      = margin
        self.tcd_margin  = tcd_margin
        self.beta_kl     = beta_kl
        self.beta_kl_z   = beta_kl_z
        self.cov_reg     = cov_reg
        self.adaptive    = adaptive
        self.ema_decay   = ema_decay

        # EMA буферы для адаптивного перевзвешивания (не обучаемые параметры)
        self._ema: Dict[str, float] = {k: 1.0 for k in self.COMPONENT_NAMES}

    # ------------------------------------------------------------------
    # Публичный API: forward(outputs, targets) → (total, components)
    # ------------------------------------------------------------------

    def forward(
        self, outputs: dict, targets: dict
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Вычисляет все компоненты потерь и взвешенную сумму.

        Параметры
        ----------
        outputs : dict – выходы модели:
            pe_scores    : (N,) причинные эффекты
            nll_normal   : (M,) NLL нормальных наблюдений
            cov_matrices : list[Tensor] ковариационные матрицы GMM
            h            : (N, d) исходные состояния
            h_recon      : (N, d) восстановленные состояния
            mu_u         : (N, d) μ экзогенных
            log_var_u    : (N, d) log σ² экзогенных
            kl_z_terms   : list[Tensor] KL конфаундеров
            u_hat        : (N, d) экзогенные переменные
            h_root_anom  : (d,) вектор первопричины (аномальный)
            h_root_norm  : (d,) прототип первопричины (нормальный)
            h_others_anom: (N-1, d)
            h_others_norm: (N-1, d)
            edge_weights : (E,) | None
            edge_cf_stats: (E,) | None
        targets : dict:
            root_idx : int – индекс первопричины

        Возвращает
        ----------
        (total_loss, {component_name: value})
        """
        root_idx = targets["root_idx"]
        w = self.w

        L_pe  = self.loss_pe(outputs["pe_scores"], root_idx)
        L_um  = self.loss_um(outputs["nll_normal"], outputs.get("cov_matrices", []))
        L_vak = self.loss_vak(
            outputs["h"], outputs["h_recon"],
            outputs["mu_u"], outputs["log_var_u"],
            outputs.get("kl_z_terms", []),
        )
        L_nez = self.loss_nez(outputs["u_hat"])
        L_kr  = self.loss_kr(
            outputs["h_root_anom"], outputs["h_root_norm"],
            outputs["h_others_anom"], outputs["h_others_norm"],
        )
        L_reb = (
            self.loss_reb(outputs["edge_weights"], outputs["edge_cf_stats"])
            if outputs.get("edge_weights") is not None
            else torch.tensor(0.0, device=outputs["h"].device)
        )

        components = {
            "L_pe":  L_pe,
            "L_um":  L_um,
            "L_vak": L_vak,
            "L_nez": L_nez,
            "L_kr":  L_kr,
            "L_reb": L_reb,
        }

        total = (
            w.lambda_pe  * L_pe
            + w.lambda_um  * L_um
            + w.lambda_vak * L_vak
            + w.lambda_nez * L_nez
            + w.lambda_kr  * L_kr
            + w.lambda_reb * L_reb
        )

        components_detached = {k: v.detach() for k, v in components.items()}
        components_detached["loss"] = total.detach()

        return total, components_detached

    # ------------------------------------------------------------------
    # Адаптивное перевзвешивание (EMA)
    # ------------------------------------------------------------------

    def update_weights(self, component_losses: Dict[str, float]) -> None:
        """Обновляет веса λ_k через экспоненциальное перевзвешивание.

        Стратегия: компоненты с высоким EMA-значением потери получают больший вес,
        чтобы стимулировать их улучшение. Веса нормируются так, что их сумма
        остаётся равной сумме начальных весов.

        Параметры
        ----------
        component_losses : dict[str, float]
            Текущие значения компонент потерь.
        """
        if not self.adaptive:
            return

        decay = self.ema_decay
        total_ema = 0.0
        for name in self.COMPONENT_NAMES:
            val = component_losses.get(name, 0.0)
            self._ema[name] = decay * self._ema[name] + (1 - decay) * abs(float(val))
            total_ema += self._ema[name]

        if total_ema < 1e-8:
            return

        # Начальная сумма весов
        init_sum = sum(getattr(self.w, attr) for attr in self.WEIGHT_ATTRS)
        # Перераспределяем: lambda_k ∝ EMA_k, сумма = init_sum
        for name, attr in zip(self.COMPONENT_NAMES, self.WEIGHT_ATTRS):
            new_w = (self._ema[name] / total_ema) * init_sum
            setattr(self.w, attr, float(new_w))

    # ------------------------------------------------------------------
    # Отдельные компоненты (для ступенчатого обучения)
    # ------------------------------------------------------------------

    def loss_pe(
        self, pe_scores: torch.Tensor, root_idx: int
    ) -> torch.Tensor:
        """L_ПЭ – ранжирование причинных эффектов (формула 5.2).

        Штраф за то, что CE других узлов выше CE первопричины.
        """
        pe_root   = pe_scores[root_idx]
        others    = [pe_scores[:root_idx], pe_scores[root_idx + 1:]]
        pe_others = torch.cat([o for o in others if o.numel() > 0])
        if pe_others.numel() == 0:
            return torch.tensor(0.0, device=pe_scores.device)
        return F.relu(pe_others - pe_root + self.margin).mean()

    def loss_um(
        self,
        nll_normal: torch.Tensor,
        cov_matrices: List[torch.Tensor],
    ) -> torch.Tensor:
        """L_УМ – условная модель нормального состояния (формула 5.3)."""
        nll_mean = nll_normal.mean()
        frob_reg = (
            sum(torch.norm(c, "fro") for c in cov_matrices)
            if cov_matrices
            else torch.tensor(0.0, device=nll_normal.device)
        )
        return nll_mean + self.cov_reg * frob_reg

    def loss_vak(
        self,
        h: torch.Tensor,
        h_recon: torch.Tensor,
        mu_u: torch.Tensor,
        log_var_u: torch.Tensor,
        kl_z_terms: List[torch.Tensor],
    ) -> torch.Tensor:
        """L_ВАК – вариационный автокодировщик (формула 5.4)."""
        mse   = F.mse_loss(h_recon, h)
        kl_u  = -0.5 * (1 + log_var_u - mu_u.pow(2) - log_var_u.exp()).mean()
        kl_z  = sum(kl_z_terms) if kl_z_terms else torch.tensor(0.0, device=h.device)
        return mse + self.beta_kl * kl_u + self.beta_kl_z * kl_z

    def loss_nez(self, u_hat: torch.Tensor) -> torch.Tensor:
        """L_нез – ограничение независимости (формула 5.5)."""
        N = u_hat.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=u_hat.device)
        u_c    = u_hat - u_hat.mean(dim=1, keepdim=True)
        std    = u_c.std(dim=1, keepdim=True).clamp(min=1e-8)
        u_norm = u_c / std
        corr   = (u_norm @ u_norm.T) / u_norm.shape[1]
        mask   = 1 - torch.eye(N, device=u_hat.device)
        return (corr.abs() * mask).sum() / mask.sum()

    def loss_kr(
        self,
        h_root_anom: torch.Tensor,
        h_root_norm: torch.Tensor,
        h_others_anom: torch.Tensor,
        h_others_norm: torch.Tensor,
    ) -> torch.Tensor:
        """L_КР – контрастное разделение (формула 5.6)."""
        cos      = nn.CosineSimilarity(dim=-1)
        sim_root = cos(h_root_anom.unsqueeze(0), h_root_norm.unsqueeze(0))
        if h_others_anom.shape[0] == 0:
            return torch.tensor(0.0, device=h_root_anom.device)
        sim_others = cos(h_others_anom, h_others_norm)
        return F.relu(self.tcd_margin - sim_others + sim_root.detach()).mean()

    def loss_reb(
        self,
        edge_weights: torch.Tensor,
        edge_cf_stats: torch.Tensor,
        threshold: float = 0.05,
    ) -> torch.Tensor:
        """L_реб – штраф за необоснованные адаптивные рёбра (формула 5.7)."""
        return (edge_weights * F.relu(threshold - edge_cf_stats)).sum()

    # ------------------------------------------------------------------
    # Претрейн-хелперы: подмножества компонент для каждого этапа
    # ------------------------------------------------------------------

    def pretrain_loss(
        self,
        nll_normal: torch.Tensor,
        h: torch.Tensor,
        h_recon: torch.Tensor,
        mu_u: torch.Tensor,
        log_var_u: torch.Tensor,
        kl_z_terms: Optional[List[torch.Tensor]] = None,
        cov_matrices: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Этап 1: L_УМ + L_ВАК (только нормальные данные)."""
        L_um  = self.loss_um(nll_normal, cov_matrices or [])
        L_vak = self.loss_vak(h, h_recon, mu_u, log_var_u, kl_z_terms or [])
        total = self.w.lambda_um * L_um + self.w.lambda_vak * L_vak
        return total, {"L_um": L_um.detach(), "L_vak": L_vak.detach(), "loss": total.detach()}

    def main_loss(
        self,
        pe_scores: torch.Tensor,
        root_idx: int,
        u_hat: torch.Tensor,
        h_root_anom: torch.Tensor,
        h_root_norm: torch.Tensor,
        h_others_anom: torch.Tensor,
        h_others_norm: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
        edge_cf_stats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Этап 2: L_ПЭ + L_нез + L_КР + L_реб (аномальные данные)."""
        L_pe  = self.loss_pe(pe_scores, root_idx)
        L_nez = self.loss_nez(u_hat)
        L_kr  = self.loss_kr(h_root_anom, h_root_norm, h_others_anom, h_others_norm)
        L_reb = (
            self.loss_reb(edge_weights, edge_cf_stats)
            if edge_weights is not None
            else torch.tensor(0.0, device=pe_scores.device)
        )
        total = (
            self.w.lambda_pe  * L_pe
            + self.w.lambda_nez * L_nez
            + self.w.lambda_kr  * L_kr
            + self.w.lambda_reb * L_reb
        )
        return total, {
            "L_pe": L_pe.detach(), "L_nez": L_nez.detach(),
            "L_kr": L_kr.detach(), "L_reb": L_reb.detach(),
            "loss": total.detach(),
        }
