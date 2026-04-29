"""Двухветвевой кодировщик метрик CAIRN (раздел 2.2).

Ветвь A (SSMBranch): модель пространства состояний в частотной области.
    s_k = A·s_{k-1} + B·x_k
    y_k = C·s_k
    Φ(ω) = C·(e^{jω}I - A)^{-1}·B  — передаточная функция
    Свёртка: Y = IRFFT(Φ(ω) · X_f(ω))

Ветвь B (BreakpointBranch): нормированный вектор разрыва.
    μ_ref = mean(x[t-2W:t-W]),  σ_ref = std(x[t-2W:t-W])
    Δ = (μ_cur - μ_ref) / (σ_ref + ε)
    h_brk = ReLU(Conv1D(Δ, kernel_size=7))

Объединение: h_met = W_proj · [h_ssm ∥ h_brk] + b_proj
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.fft


class SSMBranch(nn.Module):
    """Ветвь A — SSM в частотной области (формулы 2.1–2.4).

    Параметры
    ----------
    n_metrics : int
        F — число входных каналов.
    ssm_state_dim : int
        D — размерность скрытого состояния SSM.
    d_out : int
        d_ssm — размерность выхода ветви.
    """

    def __init__(self, n_metrics: int, ssm_state_dim: int = 64, d_out: int = 32) -> None:
        super().__init__()
        self.ssm_state_dim = ssm_state_dim
        self.d_out = d_out

        # Обучаемые матрицы SSM
        self.A = nn.Parameter(torch.randn(ssm_state_dim) * 0.01)       # (D,) — диагональ
        self.B = nn.Parameter(torch.randn(ssm_state_dim, n_metrics) * 0.01)  # (D, F)
        self.C = nn.Parameter(torch.randn(d_out, ssm_state_dim) * 0.01)      # (d_out, D)

        self.proj = nn.Linear(d_out, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Параметры
        ----------
        x : Tensor, shape (batch, T, F)

        Возвращает
        ----------
        h_ssm : Tensor, shape (batch, d_out)
        """
        batch, T, F = x.shape
        device = x.device

        # Спектр входного сигнала X(ω): (batch, n_freqs, F)
        X_f = torch.fft.rfft(x, dim=1)                        # complex
        n_freqs = X_f.shape[1]

        # Частоты ω = 2π·k/T
        freqs = torch.fft.rfftfreq(T, device=device)          # (n_freqs,)
        omega = 2 * math.pi * freqs                            # (n_freqs,)

        # Передаточная функция Φ(ω) = C · diag(1/(e^{jω} - a_d)) · B
        # диагональное приближение для скорости (A → diag(A))
        e_jw = torch.exp(1j * omega.unsqueeze(1).to(torch.cfloat))   # (n_freqs, 1)
        a_d = self.A.to(torch.cfloat).unsqueeze(0)                   # (1, D)
        inv_denom = 1.0 / (e_jw - a_d + 1e-8)                       # (n_freqs, D)

        # Φ(ω)[f, o, e] = Σ_d C[o,d] * inv_denom[f,d] * B[d,e]
        # → shape (n_freqs, d_out, F)
        C_c = self.C.to(torch.cfloat)   # (d_out, D)
        B_c = self.B.to(torch.cfloat)   # (D, F)
        phi = torch.einsum("od,fd,de->foe", C_c, inv_denom, B_c)

        # Свёртка в частотной области: Y_f[b,f,o] = Σ_e Φ[f,o,e] * X_f[b,f,e]
        Y_f = torch.einsum("foe,bfe->bfo", phi, X_f)          # (batch, n_freqs, d_out)

        # Перевод в временно́е пространство и пулинг по времени
        y = torch.fft.irfft(Y_f.permute(0, 2, 1), n=T, dim=2)  # (batch, d_out, T)
        h_ssm = y.mean(dim=2)                                    # (batch, d_out)

        return self.proj(h_ssm)


class BreakpointBranch(nn.Module):
    """Ветвь B — обнаружение разрывов (формулы 2.5–2.9).

    Параметры
    ----------
    n_metrics : int
        F — число каналов.
    window : int
        W — размер окна.
    d_out : int
        d_brk — размерность выхода.
    """

    def __init__(self, n_metrics: int, window: int = 60, d_out: int = 32) -> None:
        super().__init__()
        self.window = window
        # kernel_size=7 захватывает форму переходного процесса разрыва
        self.conv = nn.Sequential(
            nn.Conv1d(n_metrics, d_out, kernel_size=7, padding=3),
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
        h_brk : Tensor, shape (batch, d_out)
        """
        T = x.shape[1]
        W = min(self.window, max(T // 3, 1))

        # Референсное окно [0, W) и текущее окно [W, 2W)
        x_ref = x[:, :W, :]
        x_cur = x[:, -W:, :]

        mu_ref = x_ref.mean(dim=1)             # (batch, F)
        sigma_ref = x_ref.std(dim=1) + 1e-6   # (batch, F)
        mu_cur = x_cur.mean(dim=1)             # (batch, F)

        delta = (mu_cur - mu_ref) / sigma_ref  # (batch, F) — нормированный разрыв

        # Разворачиваем в «временну́ю» последовательность для Conv1d
        delta_seq = delta.unsqueeze(2).expand(-1, -1, W)  # (batch, F, W)

        h = self.conv(delta_seq)    # (batch, d_out, W)
        return self.pool(h).squeeze(-1)  # (batch, d_out)


class DualBranchMetricEncoder(nn.Module):
    """Двухветвевой кодировщик метрик (основной класс, раздел 2.2).

    Параметры
    ----------
    n_metrics : int
        F — число входных метрик.
    d_ssm : int
        Размерность выхода SSM-ветви.
    d_brk : int
        Размерность выхода ветви разрыва.
    d_out : int
        d_met — итоговая размерность после проекции.
    ssm_state_dim : int
        D — размерность скрытого состояния SSM.
    window : int
        W — размер окна для ветви разрыва.
    """

    def __init__(
        self,
        n_metrics: int,
        d_ssm: int = 32,
        d_brk: int = 32,
        d_out: int = 64,
        ssm_state_dim: int = 64,
        window: int = 60,
    ) -> None:
        super().__init__()
        self.ssm_branch = SSMBranch(n_metrics, ssm_state_dim=ssm_state_dim, d_out=d_ssm)
        self.brk_branch = BreakpointBranch(n_metrics, window=window, d_out=d_brk)
        self.proj = nn.Linear(d_ssm + d_brk, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Параметры
        ----------
        x : Tensor, shape (batch, T, F)

        Возвращает
        ----------
        h_met : Tensor, shape (batch, d_out)
        """
        h_ssm = self.ssm_branch(x)               # (batch, d_ssm)
        h_brk = self.brk_branch(x)               # (batch, d_brk)
        h_cat = torch.cat([h_ssm, h_brk], dim=-1)
        return self.norm(self.proj(h_cat))        # (batch, d_out)


# Обратная совместимость с тестами предыдущих сессий
MetricEncoder = DualBranchMetricEncoder
# Для прямого импорта по старым именам
StablePatternBranch = SSMBranch
