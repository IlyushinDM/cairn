"""Композитная функция потерь CAIRN (раздел 5.1, формула 5.1).

L = λ₁·L_ПЭ + λ₂·L_УМ + λ₃·L_ВАК + λ₄·L_нез + λ₅·L_КР + λ₆·L_реб
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    lambda_pe: float = 1.0
    lambda_um: float = 1.0
    lambda_vak: float = 1.0
    lambda_nez: float = 0.5
    lambda_kr: float = 0.5
    lambda_reb: float = 0.1


class CAIRNLoss(nn.Module):
    """Составная функция потерь.

    Параметры
    ----------
    weights : LossWeights
        Коэффициенты λ₁–λ₆.
    margin : float
        Отступ для L_ПЭ (формула 5.2).
    tcd_margin : float
        δ_KR для L_КР (формула 5.6).
    beta_kl : float
        β — коэффициент KL для экзогенных переменных (формула 5.4).
    beta_kl_z : float
        β_z — KL для скрытых факторов.
    cov_reg : float
        λ_рег — регуляризация ковариации (формула 5.3).
    """

    def __init__(
        self,
        weights: LossWeights | None = None,
        margin: float = 0.1,
        tcd_margin: float = 0.3,
        beta_kl: float = 1.0,
        beta_kl_z: float = 0.1,
        cov_reg: float = 1e-4,
    ) -> None:
        super().__init__()
        self.w = weights or LossWeights()
        self.margin = margin
        self.tcd_margin = tcd_margin
        self.beta_kl = beta_kl
        self.beta_kl_z = beta_kl_z
        self.cov_reg = cov_reg

    # ------------------------------------------------------------------
    # L_ПЭ: ранжирование причинных эффектов (формула 5.2)
    # ------------------------------------------------------------------
    def loss_pe(
        self,
        pe_scores: torch.Tensor,   # (N,) — ПЭ(i) для каждого узла
        root_idx: int,
    ) -> torch.Tensor:
        pe_root = pe_scores[root_idx]
        pe_others = torch.cat([pe_scores[:root_idx], pe_scores[root_idx + 1:]])
        losses = F.relu(pe_others - pe_root + self.margin)  # формула 5.2
        return losses.mean()

    # ------------------------------------------------------------------
    # L_УМ: условная модель нормального состояния (формула 5.3)
    # ------------------------------------------------------------------
    def loss_um(
        self,
        nll_normal: torch.Tensor,              # (M_н,) NLL нормальных наблюдений
        cov_matrices: list[torch.Tensor],      # [Σ_k] ковариационные матрицы компонент
    ) -> torch.Tensor:
        nll_mean = nll_normal.mean()
        frob_reg = sum(torch.norm(cov, "fro") for cov in cov_matrices)
        return nll_mean + self.cov_reg * frob_reg

    # ------------------------------------------------------------------
    # L_ВАК: вариационный автокодировщик (формула 5.4)
    # ------------------------------------------------------------------
    def loss_vak(
        self,
        h: torch.Tensor,              # (N, d) оригинальные состояния
        h_recon: torch.Tensor,        # (N, d) восстановленные состояния
        mu_u: torch.Tensor,           # (N, d//2) — μ экзогенных
        log_var_u: torch.Tensor,      # (N, d//2) — log σ² экзогенных
        kl_z_terms: list[torch.Tensor],  # K скаляров KL для ẑ_k
    ) -> torch.Tensor:
        mse = F.mse_loss(h_recon, h)
        kl_u = -0.5 * (1 + log_var_u - mu_u.pow(2) - log_var_u.exp()).mean()
        kl_z = sum(kl_z_terms)
        return mse + self.beta_kl * kl_u + self.beta_kl_z * kl_z

    # ------------------------------------------------------------------
    # L_нез: ограничение независимости экзогенных переменных (формула 5.5)
    # ------------------------------------------------------------------
    def loss_nez(self, u_hat: torch.Tensor) -> torch.Tensor:
        """
        u_hat : Tensor, shape (N, d)
        Среднее |corr(ûᵢ, ûⱼ)| по всем парам i≠j.
        """
        N = u_hat.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=u_hat.device)
        # Нормализуем строки по std
        u_centered = u_hat - u_hat.mean(dim=1, keepdim=True)
        std = u_centered.std(dim=1, keepdim=True).clamp(min=1e-8)
        u_norm = u_centered / std                          # (N, d)

        # Матрица корреляций (косинусное подобие нормализованных строк)
        corr_matrix = (u_norm @ u_norm.T) / u_norm.shape[1]  # (N, N)
        # Маска внедиагональных элементов
        mask = 1 - torch.eye(N, device=u_hat.device)
        mean_corr = (corr_matrix.abs() * mask).sum() / mask.sum()
        return mean_corr

    # ------------------------------------------------------------------
    # L_КР: контрастное разделение (формула 5.6)
    # ------------------------------------------------------------------
    def loss_kr(
        self,
        h_root_anom: torch.Tensor,    # (d,) — вектор первопричины в аномальном состоянии
        h_root_norm: torch.Tensor,    # (d,) — условный прототип первопричины
        h_others_anom: torch.Tensor,  # (N-1, d) — остальные узлы аномальные
        h_others_norm: torch.Tensor,  # (N-1, d) — остальные узлы нормальные
    ) -> torch.Tensor:
        cos = nn.CosineSimilarity(dim=-1)
        # Для первопричины: аномальное ↔ нормальное должны быть ДАЛЕКО (cos → -1)
        sim_root = cos(h_root_anom.unsqueeze(0), h_root_norm.unsqueeze(0))

        # Для остальных: аномальное ↔ нормальное должны быть БЛИЗКО (cos → +1)
        sim_others = cos(h_others_anom, h_others_norm)   # (N-1,)

        # Отступ: sim_others ≥ sim_root + δ_KR (штрафуем нарушения)
        losses = F.relu(self.tcd_margin - sim_others + sim_root.detach())
        return losses.mean()

    # ------------------------------------------------------------------
    # L_реб: штраф за необоснованные адаптивные рёбра (формула 5.7)
    # ------------------------------------------------------------------
    def loss_reb(
        self,
        edge_weights: torch.Tensor,          # (E_адд,) — веса адаптивных рёбер
        edge_cf_stats: torch.Tensor,         # (E_адд,) — контрфактическая значимость τ(e)
        threshold: float = 0.05,
    ) -> torch.Tensor:
        penalty = edge_weights * F.relu(threshold - edge_cf_stats)
        return penalty.sum()

    # ------------------------------------------------------------------
    # Итоговая функция потерь
    # ------------------------------------------------------------------
    def forward(
        self,
        pe_scores: torch.Tensor,
        root_idx: int,
        nll_normal: torch.Tensor,
        cov_matrices: list[torch.Tensor],
        h: torch.Tensor,
        h_recon: torch.Tensor,
        mu_u: torch.Tensor,
        log_var_u: torch.Tensor,
        kl_z_terms: list[torch.Tensor],
        u_hat: torch.Tensor,
        h_root_anom: torch.Tensor,
        h_root_norm: torch.Tensor,
        h_others_anom: torch.Tensor,
        h_others_norm: torch.Tensor,
        edge_weights: torch.Tensor | None = None,
        edge_cf_stats: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        w = self.w
        L_pe = self.loss_pe(pe_scores, root_idx)
        L_um = self.loss_um(nll_normal, cov_matrices)
        L_vak = self.loss_vak(h, h_recon, mu_u, log_var_u, kl_z_terms)
        L_nez = self.loss_nez(u_hat)
        L_kr = self.loss_kr(h_root_anom, h_root_norm, h_others_anom, h_others_norm)
        L_reb = (
            self.loss_reb(edge_weights, edge_cf_stats)
            if (edge_weights is not None and edge_cf_stats is not None)
            else torch.tensor(0.0)
        )

        total = (
            w.lambda_pe * L_pe
            + w.lambda_um * L_um
            + w.lambda_vak * L_vak
            + w.lambda_nez * L_nez
            + w.lambda_kr * L_kr
            + w.lambda_reb * L_reb
        )

        return {
            "loss": total,
            "L_pe": L_pe.detach(),
            "L_um": L_um.detach(),
            "L_vak": L_vak.detach(),
            "L_nez": L_nez.detach(),
            "L_kr": L_kr.detach(),
            "L_reb": L_reb.detach(),
        }
