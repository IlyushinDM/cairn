"""Условная смесь гауссовых распределений нормального состояния (раздел 3.1).

P(h | c, Θ) = Σ_k ω_k(c) · N(h | μ_k(c), diag(σ_k²(c)))

Параметры каждой компоненты – функция контекстного вектора c:
  MLP_μ  : c → (D × d)
  MLP_σ  : c → (D × d) – log-дисперсии
  MLP_ω  : c → D        – логиты весов
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOG_2PI = math.log(2 * math.pi)   # Python-константа, не тензор


class ConditionalGMM(nn.Module):
    """Условная смесь гауссовых распределений (раздел 3.1, формулы 3.1–3.8).

    Параметры
    ----------
    state_dim : int
        d – размерность вектора состояния.
    context_dim : int
        d_c = 16 – размерность контекстного вектора.
    n_components : int
        D = 5 – число гауссовых компонент.
    hidden_dim : int
        Ширина скрытого слоя MLP.
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

        # MLP_μ: c → (D × d)   – условные центры (формула 3.2)
        self.mlp_mu = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_components * state_dim),
        )
        # MLP_σ: c → (D × d)   – log-дисперсии (формула 3.3)
        self.mlp_log_sigma = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_components * state_dim),
        )
        # MLP_ω: c → D          – логиты весов (формула 3.4)
        self.mlp_omega = nn.Sequential(
            nn.Linear(context_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_components),
        )

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def forward(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Параметры смеси для заданного контекста.

        Параметры
        ----------
        context : Tensor, shape (batch, context_dim)

        Возвращает
        ----------
        weights : Tensor, shape (batch, D)       – суммируются в 1
        means   : Tensor, shape (batch, D, d)
        log_vars: Tensor, shape (batch, D, d)    – log σ²
        """
        batch = context.shape[0]
        D, d = self.n_components, self.state_dim

        means    = self.mlp_mu(context).view(batch, D, d)
        log_vars = self.mlp_log_sigma(context).view(batch, D, d).clamp(-6, 4)
        weights  = F.softmax(self.mlp_omega(context), dim=-1)   # (batch, D)
        return weights, means, log_vars

    def log_prob(self, h: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """log P(h | c, Θ) – формула 3.1.

        Параметры
        ----------
        h       : Tensor, shape (batch, d)
        context : Tensor, shape (batch, context_dim) или (1, context_dim)

        Возвращает
        ----------
        log_p : Tensor, shape (batch,)
        """
        # Поддержка единственного контекста для всего батча
        if context.shape[0] == 1 and h.shape[0] > 1:
            context = context.expand(h.shape[0], -1)

        weights, means, log_vars = self.forward(context)    # (B, D, d)
        h_exp = h.unsqueeze(1).expand_as(means)             # (B, D, d)

        # log N(h | μ_k, diag(σ_k²)) – формула 3.1 (диагональная ковариация)
        log_gauss = -0.5 * (
            ((h_exp - means) ** 2 / (log_vars.exp() + 1e-8)).sum(-1)
            + log_vars.sum(-1)
            + self.state_dim * _LOG_2PI
        )                                                    # (B, D)

        log_p = torch.logsumexp(
            log_gauss + torch.log(weights + 1e-8), dim=-1
        )                                                    # (B,)
        return log_p

    def nll(self, h: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Отрицательный логарифм правдоподобия – мера аномальности NLL_i (формула 3.7)."""
        return -self.log_prob(h, context)

    def prototype(self, context: torch.Tensor) -> torch.Tensor:
        """Условный прототип μ*(c) – центр компоненты с максимальным весом (формула 3.5–3.6).

        Параметры
        ----------
        context : Tensor, shape (batch, context_dim)

        Возвращает
        ----------
        proto : Tensor, shape (batch, d)
        """
        weights, means, _ = self.forward(context)
        k_star = weights.argmax(dim=-1)                      # (batch,)
        idx = k_star.view(-1, 1, 1).expand(-1, 1, self.state_dim)
        return means.gather(1, idx).squeeze(1)               # (batch, d)

    # Alias for backward compatibility with older code
    def conditional_prototype(self, context: torch.Tensor) -> torch.Tensor:
        return self.prototype(context)

    def anomaly_threshold(
        self, normal_nll: torch.Tensor, percentile: float = 0.99
    ) -> float:
        """Порог δ по ε-му процентилю NLL нормальных наблюдений (формула 3.8)."""
        return torch.quantile(normal_nll, percentile).item()

    def detect_drift(
        self, nll_history: torch.Tensor, threshold: float, window: int = 20
    ) -> bool:
        """Обнаруживает дрейф распределения (раздел 3.1.5).

        Дрейф считается обнаруженным, если скользящее среднее NLL систематически
        превышает порог δ более чем в половине последних ``window`` шагов.

        Параметры
        ----------
        nll_history : Tensor, shape (T,)
            История значений NLL по времени.
        threshold : float
            Порог δ (из anomaly_threshold).
        window : int
            Ширина скользящего окна.

        Возвращает
        ----------
        bool – True если дрейф обнаружен.
        """
        if nll_history.numel() < window:
            return False
        recent = nll_history[-window:]
        frac_above = (recent > threshold).float().mean().item()
        return frac_above > 0.5
