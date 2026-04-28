"""Условная смесь гауссовых распределений нормального состояния (раздел 3.1).

Формулы 3.1–3.8. Параметры каждой компоненты являются функцией контекстного вектора cᵢ.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalGMM(nn.Module):
    """Условная смесь гауссовых распределений P(hᵢ | cᵢ, Θ) (формула 3.1).

    Параметры
    ----------
    state_dim : int
        d — размерность вектора состояния.
    context_dim : int
        d_c = 16 — размерность контекстного вектора.
    n_components : int
        D = 5 — число гауссовых компонент.
    hidden_dim : int
        Ширина скрытого слоя МСП.
    """

    def __init__(
        self,
        state_dim: int = 128,
        context_dim: int = 16,
        n_components: int = 5,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.n_components = n_components

        # МСП_μ: c → (D × d) — центры компонент (формула 3.2)
        self.mlp_mu = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_components * state_dim),
        )
        # МСП_σ: c → (D × d) — log-дисперсии компонент (формула 3.3)
        self.mlp_log_sigma = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_components * state_dim),
        )
        # МСП_ω: c → D — логиты весов компонент (формула 3.4)
        self.mlp_omega = nn.Sequential(
            nn.Linear(context_dim, 32), nn.ReLU(),
            nn.Linear(32, n_components),
        )

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Вычисляет параметры смеси для заданного контекста.

        Параметры
        ----------
        context : Tensor, shape (batch, context_dim)

        Возвращает
        ----------
        mu      : Tensor, shape (batch, D, d)
        sigma   : Tensor, shape (batch, D, d)  — стандартные отклонения > 0
        weights : Tensor, shape (batch, D)      — веса компонент (сумма = 1)
        """
        batch = context.shape[0]
        D, d = self.n_components, self.state_dim

        mu = self.mlp_mu(context).view(batch, D, d)
        log_sigma = self.mlp_log_sigma(context).view(batch, D, d)
        sigma = torch.exp(log_sigma.clamp(-4, 4))  # стабилизация
        weights = F.softmax(self.mlp_omega(context), dim=-1)  # (batch, D)
        return mu, sigma, weights

    def log_prob(self, h: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """log P(hᵢ | cᵢ, Θ) — формула 3.1.

        Параметры
        ----------
        h       : Tensor, shape (batch, d)
        context : Tensor, shape (batch, context_dim)

        Возвращает
        ----------
        log_p : Tensor, shape (batch,)
        """
        mu, sigma, weights = self.forward(context)   # (batch, D, d)

        # log N(h | μ_k, diag(σ_k²)) для каждой компоненты
        h_exp = h.unsqueeze(1).expand_as(mu)         # (batch, D, d)
        var = sigma ** 2 + 1e-8
        log_gauss = -0.5 * (
            ((h_exp - mu) ** 2 / var).sum(dim=-1)
            + torch.log(var).sum(dim=-1)
            + self.state_dim * math.log(2 * math.pi)
        )  # (batch, D)

        log_weights = torch.log(weights + 1e-8)      # (batch, D)
        log_p = torch.logsumexp(log_gauss + log_weights, dim=-1)  # (batch,)
        return log_p

    def nll(self, h: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Отрицательный логарифм правдоподобия — мера аномальности NLLᵢ (формула 3.7)."""
        return -self.log_prob(h, context)

    def conditional_prototype(self, context: torch.Tensor) -> torch.Tensor:
        """Условный прототип μ*(cᵢ) — центр наиболее вероятной компоненты (формулы 3.5–3.6).

        Параметры
        ----------
        context : Tensor, shape (batch, context_dim)

        Возвращает
        ----------
        proto : Tensor, shape (batch, d)
        """
        mu, _, weights = self.forward(context)       # mu: (batch, D, d)
        k_star = weights.argmax(dim=-1)              # (batch,)
        # Собираем μ_{k*} для каждого элемента батча
        batch = context.shape[0]
        idx = k_star.view(batch, 1, 1).expand(batch, 1, self.state_dim)
        proto = mu.gather(1, idx).squeeze(1)         # (batch, d)
        return proto

    def anomaly_threshold(
        self, normal_nll: torch.Tensor, percentile: float = 0.99
    ) -> float:
        """Вычисляет порог δ по ε-му процентилю нормальных наблюдений (формула 3.8)."""
        q = torch.quantile(normal_nll, percentile)
        return q.item()
