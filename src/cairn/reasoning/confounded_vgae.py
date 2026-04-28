"""Вариационный графовый автокодировщик с латентными конфаундерами (раздел 3.2).

Реализует абдукцию: вычисление экзогенных переменных ûᵢ (формулы 3.9–3.14)
и латентных переменных скрытых общих факторов ẑ_k (формулы 3.15–3.19).
Ограничение независимости C_u вычисляется в loss.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExogenousEncoder(nn.Module):
    """Вычисляет экзогенные переменные ûᵢ через внимание на причинных предшественниках.

    Параметры
    ----------
    state_dim : int
        d — размерность вектора состояния.
    edge_dim : int
        Размерность вектора признаков ребра.
    n_types : int
        Число типов узлов (сервис, БД, кеш и т.д.).
    """

    def __init__(self, state_dim: int = 128, edge_dim: int = 16, n_types: int = 4) -> None:
        super().__init__()
        self.state_dim = state_dim

        # Тип-зависимая проекция h → h^пр (формула 3.9)
        self.type_proj = nn.ModuleList([
            nn.Linear(state_dim, state_dim) for _ in range(n_types)
        ])

        # Матрица внимания W_A (формула 3.10)
        # Вход: h^пр_i || h^пр_j || e^пр_ij
        self.attn = nn.Linear(2 * state_dim + edge_dim, 1)

        # Матрица сообщений W_M (формула 3.12)
        self.msg_proj = nn.Linear(state_dim + edge_dim, state_dim)

        # Вариационный сплит: 2d → μ + log σ² (формула 3.13)
        self.split_proj = nn.Linear(state_dim, 2 * state_dim)

    def forward(
        self,
        h: torch.Tensor,              # (N, d)
        node_types: torch.Tensor,     # (N,) int — тип каждого узла
        edge_index: torch.Tensor,     # (2, E) — j→i (источник→цель)
        edge_feats: torch.Tensor,     # (E, edge_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Возвращает
        ----------
        u_hat : Tensor, shape (N, d//2)  — μ выборки
        mu_u  : Tensor, shape (N, d//2)
        log_var_u : Tensor, shape (N, d//2)
        """
        N = h.shape[0]

        # Тип-зависимая проекция (формула 3.9)
        h_pr = torch.zeros_like(h)
        for t, proj in enumerate(self.type_proj):
            mask = node_types == t
            if mask.any():
                h_pr[mask] = proj(h[mask])

        src, dst = edge_index[0], edge_index[1]  # j, i

        # Внимание (формулы 3.10–3.11)
        attn_in = torch.cat([h_pr[dst], h_pr[src], edge_feats], dim=-1)  # (E, 2d+e)
        a = F.leaky_relu(self.attn(attn_in).squeeze(-1))                  # (E,)

        # Нормализация softmax по предшественникам dst[i]
        alpha = torch.zeros(N, dtype=torch.float, device=h.device)
        # scatter softmax
        from torch_scatter import scatter_softmax
        alpha_e = scatter_softmax(a, dst, dim=0)  # (E,)

        # Агрегация сообщений (формула 3.12)
        msg_in = torch.cat([h_pr[src], edge_feats], dim=-1)  # (E, d+e)
        msg = self.msg_proj(msg_in)                            # (E, d)

        agg = torch.zeros(N, self.state_dim, device=h.device)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), alpha_e.unsqueeze(1) * msg)

        u = h_pr + agg  # (N, d) — экзогенная переменная до выборки

        # Вариационная выборка (формулы 3.13–3.14)
        u_params = self.split_proj(u)  # (N, 2d)
        half = self.state_dim
        mu_u = u_params[:, :half]
        log_var_u = u_params[:, half:]
        sigma_u = torch.exp(0.5 * log_var_u.clamp(-4, 4))
        eps = torch.randn_like(mu_u)
        u_hat = mu_u + sigma_u * eps  # (N, half)
        return u_hat, mu_u, log_var_u


class LatentConfounderModule(nn.Module):
    """Модуль латентных переменных скрытых общих факторов (формулы 3.15–3.19).

    Параметры
    ----------
    state_dim : int
        d.
    n_confounders : int
        K = 3.
    confounder_dim : int
        d_z = 32.
    """

    def __init__(
        self,
        state_dim: int = 128,
        n_confounders: int = 3,
        confounder_dim: int = 32,
    ) -> None:
        super().__init__()
        self.K = n_confounders
        self.dz = confounder_dim

        # q(z_k | H, A): μ_k = W_μ * mean(H) + b (формула 3.16)
        self.z_mu = nn.ModuleList([
            nn.Linear(state_dim, confounder_dim) for _ in range(n_confounders)
        ])
        self.z_log_sigma = nn.ModuleList([
            nn.Linear(state_dim, confounder_dim) for _ in range(n_confounders)
        ])

        # Маска m_ki (формула 3.18)
        self.mask_proj = nn.Linear(confounder_dim + state_dim, 1)

        # Декодирование ẑ_k → вклад в ûᵢ (формула 3.19)
        self.z_dec = nn.Linear(confounder_dim, state_dim // 2)

    def forward(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """
        Параметры
        ----------
        h : Tensor, shape (N, state_dim)

        Возвращает
        ----------
        correction   : Tensor, shape (N, state_dim//2)  — суммарная поправка Σ m_ki * W_dec * ẑ_k
        z_hats       : list of K tensors, shape (confounder_dim,) — выборки ẑ_k
        kl_terms     : list of K tensors, shape () — KL-дивергенции
        """
        N = h.shape[0]
        h_mean = h.mean(dim=0, keepdim=True)  # (1, d) — глобальное среднее (формула 3.16)

        z_hats, kl_terms = [], []
        correction = torch.zeros(N, self.state_dim_half(h), device=h.device)

        for k in range(self.K):
            mu_k = self.z_mu[k](h_mean)                          # (1, dz)
            log_sigma_k = self.z_log_sigma[k](h_mean)            # (1, dz)
            sigma_k = torch.exp(0.5 * log_sigma_k.clamp(-4, 4))
            eps = torch.randn_like(mu_k)
            z_hat = (mu_k + sigma_k * eps).squeeze(0)            # (dz,)

            # KL: q(z_k) || N(0,I)
            kl = -0.5 * (1 + log_sigma_k - mu_k ** 2 - torch.exp(log_sigma_k)).sum()
            kl_terms.append(kl)
            z_hats.append(z_hat)

            # Маска m_ki для каждого узла (формула 3.18)
            z_exp = z_hat.unsqueeze(0).expand(N, -1)             # (N, dz)
            mask_in = torch.cat([z_exp, h], dim=-1)              # (N, dz+d)
            m_ki = torch.sigmoid(self.mask_proj(mask_in))        # (N, 1)

            # Вклад в поправку (формула 3.19)
            z_dec = self.z_dec(z_hat).unsqueeze(0).expand(N, -1)  # (N, d//2)
            correction = correction + m_ki * z_dec

        return correction, z_hats, kl_terms

    def state_dim_half(self, h: torch.Tensor) -> int:
        return self.z_dec.out_features
