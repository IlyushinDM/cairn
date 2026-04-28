"""Двухветвевой кодировщик метрик CAIRN.

Раздел 2.2 описания архитектуры.

Ветвь 1 (StablePatternBranch): модель пространства состояний, переведённая в
  частотную область через Z-преобразование (формулы 2.1–2.4).
Ветвь 2 (BreakpointBranch): нормированный вектор разрыва + Conv1D (формулы 2.5–2.9).
Объединение (MetricEncoder): конкатенация + линейная проекция (формула 2.10).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.fft


class StablePatternBranch(nn.Module):
    """Ветвь устойчивых закономерностей (SSM в частотной области).

    Параметры
    ----------
    input_features : int
        Число метрик F (каналов входного ряда).
    hidden_dim : int
        Размерность скрытого состояния D.
    out_dim : int
        Размерность выхода d₁.
    """

    def __init__(self, input_features: int, hidden_dim: int = 64, out_dim: int = 32) -> None:
        super().__init__()
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Матрицы SSM: A, B, C — обучаемые параметры (формулы 2.1–2.3)
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(hidden_dim, input_features) * 0.01)
        self.C = nn.Parameter(torch.randn(out_dim, hidden_dim) * 0.01)

        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Параметры
        ----------
        x : Tensor, shape (batch, T, F)
            Временной ряд метрик.

        Возвращает
        ----------
        Tensor, shape (batch, out_dim)
        """
        # Вычисляем передаточную функцию Φ(ω) через FFT (формула 2.3)
        T = x.shape[1]
        freqs = torch.fft.rfftfreq(T, device=x.device)  # (T//2+1,)
        omega = 2 * torch.pi * freqs  # (num_freqs,)

        # Спектр входного сигнала: (batch, num_freqs, F)
        X_f = torch.fft.rfft(x, dim=1)

        # Аппроксимация Φ(ω)B для каждой частоты:
        # Φ(ω) = C (I - A e^{-jω})^{-1} B
        # Для эффективности: усредняем по частотам, используем диагональное приближение
        # (полный инверс A слишком дорог; вместо него — диагональное представление)
        diag_A = torch.diagonal(self.A)  # (D,)
        # (num_freqs, D): I - A*e^{-jω} — диагональная часть
        e_jw = torch.exp(-1j * omega.unsqueeze(1) * diag_A.unsqueeze(0))  # (freq, D)
        inv_diag = 1.0 / (1.0 - e_jw + 1e-8)  # (freq, D)

        # Φ(ω) ≈ C * inv_diag * B: (freq, out_dim, F) через einsum
        phi = torch.einsum("od,fd,df->fof", self.C, inv_diag.real, self.B)  # упрощение

        # Свёртка в частотной области (формула 2.4): Φ(ω) * X_f
        # X_f: (batch, freq, F), используем среднее по частотам как агрегацию
        out_f = (X_f.unsqueeze(-2) * phi.unsqueeze(0)).mean(dim=(1, 3))  # (batch, out_dim)

        return self.proj(out_f.real)  # (batch, out_dim)


class BreakpointBranch(nn.Module):
    """Ветвь обнаружения разрывов.

    Вычисляет нормированный вектор разрыва Δᵢ (формулы 2.5–2.8),
    затем обрабатывает его Conv1D (формула 2.9).

    Параметры
    ----------
    input_features : int
        Число метрик F.
    window : int
        Размер окна W (шаги).
    out_dim : int
        Размерность выхода d₂.
    """

    def __init__(self, input_features: int, window: int = 60, out_dim: int = 32) -> None:
        super().__init__()
        self.window = window
        self.conv = nn.Sequential(
            nn.Conv1d(input_features, out_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Параметры
        ----------
        x : Tensor, shape (batch, T, F)

        Возвращает
        ----------
        Tensor, shape (batch, out_dim)
        """
        W = self.window
        T = x.shape[1]
        W = min(W, T // 3)  # защита от коротких рядов

        # Референсное окно: t-2W .. t-W
        x_ref = x[:, :W, :]                   # (batch, W, F)
        # Текущее окно: t-W .. t
        x_cur = x[:, W: 2 * W, :] if T >= 2 * W else x[:, -W:, :]

        mu_ref = x_ref.mean(dim=1)             # (batch, F)
        sigma_ref = x_ref.std(dim=1) + 1e-6   # (batch, F)
        mu_cur = x_cur.mean(dim=1)             # (batch, F)

        delta = (mu_cur - mu_ref) / sigma_ref  # (batch, F) — формула 2.7
        # delta расширяем обратно в «временной» форме для Conv1D
        delta_seq = delta.unsqueeze(2)         # (batch, F, 1)
        delta_seq = delta_seq.expand(-1, -1, W)  # (batch, F, W)

        h = self.conv(delta_seq)               # (batch, out_dim, W)
        h = self.pool(h).squeeze(-1)           # (batch, out_dim)
        return h


class MetricEncoder(nn.Module):
    """Двухветвевой кодировщик метрик (раздел 2.2).

    Параметры
    ----------
    input_features : int
        F — число метрик.
    window : int
        W — размер окна разрыва.
    d1 : int
        Размерность выхода ветви устойчивых закономерностей.
    d2 : int
        Размерность выхода ветви разрывов.
    out_dim : int
        d_met — итоговая размерность (формула 2.10).
    """

    def __init__(
        self,
        input_features: int,
        window: int = 60,
        d1: int = 32,
        d2: int = 32,
        out_dim: int = 64,
    ) -> None:
        super().__init__()
        self.stable = StablePatternBranch(input_features, out_dim=d1)
        self.breakpoint = BreakpointBranch(input_features, window=window, out_dim=d2)
        # Проекция конкатенации (формула 2.10)
        self.proj = nn.Linear(d1 + d2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Параметры
        ----------
        x : Tensor, shape (batch, T, F)

        Возвращает
        ----------
        h_met : Tensor, shape (batch, out_dim)
        """
        h_stable = self.stable(x)        # (batch, d1)
        h_break = self.breakpoint(x)     # (batch, d2)
        h_cat = torch.cat([h_stable, h_break], dim=-1)  # (batch, d1+d2)
        return self.proj(h_cat)          # (batch, out_dim) — h_met
