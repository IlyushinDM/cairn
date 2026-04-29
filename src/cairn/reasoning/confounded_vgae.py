"""Вариационный графовый автокодировщик со скрытыми конфаундерами (раздел 3.2).

Конвейер абдукции:
  1. ExogenousEncoder  — вычисляет û_i через attention-агрегацию предшественников
  2. LatentConfounderModule — K скрытых общих факторов ẑ_k, каждый влияет на узлы через маску
  3. ConfoundedVGAE     — объединяет (1) и (2), добавляет декодер и функцию независимости

Намеренно не используется torch_scatter — scatter softmax реализован через
стандартные операции PyTorch (>= 2.0).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Вспомогательная функция: scatter softmax без torch_scatter
# ---------------------------------------------------------------------------

def _scatter_softmax(
    src: torch.Tensor,     # (E,) — веса рёбер
    index: torch.Tensor,   # (E,) long — индекс целевого узла
    num_nodes: int,
) -> torch.Tensor:
    """Численно стабильный scatter softmax (без torch_scatter).

    Для каждого целевого узла i: α_{j→i} = exp(a_{j→i}) / Σ_{j'} exp(a_{j'→i})

    Требует PyTorch >= 2.0 для scatter_reduce_ с reduce='amax'.
    """
    # Максимум per-узел для стабильности
    max_per_node = torch.full((num_nodes,), float('-inf'),
                              device=src.device, dtype=src.dtype)
    max_per_node.scatter_reduce_(0, index, src, reduce='amax', include_self=True)
    # Узлы без входящих рёбер получают -inf → заменяем на 0
    max_per_node = torch.nan_to_num(max_per_node, neginf=0.0)

    exp_src = torch.exp(src - max_per_node[index])

    denom = torch.zeros(num_nodes, device=src.device, dtype=src.dtype)
    denom.scatter_add_(0, index, exp_src)

    return exp_src / (denom[index] + 1e-8)


# ---------------------------------------------------------------------------
# ExogenousEncoder
# ---------------------------------------------------------------------------

class ExogenousEncoder(nn.Module):
    """Вычисляет экзогенные переменные û_i через attention-агрегацию (формулы 3.9–3.14).

    Параметры
    ----------
    state_dim : int
        d — размерность вектора состояния.
    edge_dim : int
        Размерность признаков рёбер.
    n_node_types : int
        Число типов узлов (service / database / cache / ...).
    """

    def __init__(
        self,
        state_dim: int = 128,
        edge_dim: int = 16,
        n_node_types: int = 4,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim

        # Тип-зависимая проекция (формула 3.9): h → h^пр
        self.type_proj = nn.ModuleList(
            [nn.Linear(state_dim, state_dim) for _ in range(n_node_types)]
        )
        # Attention: [h_dst || h_src || e_ij] → скаляр (формула 3.10)
        self.attn = nn.Linear(2 * state_dim + edge_dim, 1)
        # Сообщение: [h_src || e_ij] → d (формула 3.12)
        self.msg_proj = nn.Linear(state_dim + edge_dim, state_dim)
        # Вариационный сплит: d → 2d (формула 3.13)
        self.split_proj = nn.Linear(state_dim, 2 * state_dim)

    def forward(
        self,
        h: torch.Tensor,                  # (N, d)
        node_types: torch.Tensor,         # (N,) long
        edge_index: torch.Tensor,         # (2, E): [src_j, dst_i]
        edge_feats: torch.Tensor,         # (E, edge_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Возвращает
        ----------
        u_hat     : (N, d) — экзогенная переменная (reparameterised sample)
        mu_u      : (N, d)
        log_var_u : (N, d)
        """
        N = h.shape[0]

        # Тип-зависимая проекция (формула 3.9)
        h_pr = torch.zeros_like(h)
        for t, proj in enumerate(self.type_proj):
            mask = node_types == t
            if mask.any():
                h_pr[mask] = proj(h[mask])

        src_idx, dst_idx = edge_index[0], edge_index[1]   # (E,)

        # Attention-веса (формулы 3.10–3.11)
        attn_in = torch.cat([h_pr[dst_idx], h_pr[src_idx], edge_feats], dim=-1)
        a = F.leaky_relu(self.attn(attn_in).squeeze(-1), negative_slope=0.2)  # (E,)
        alpha = _scatter_softmax(a, dst_idx, N)                                # (E,)

        # Агрегация сообщений (формула 3.12)
        msg = self.msg_proj(torch.cat([h_pr[src_idx], edge_feats], dim=-1))    # (E, d)
        agg = torch.zeros(N, self.state_dim, device=h.device, dtype=h.dtype)
        agg.scatter_add_(0, dst_idx.unsqueeze(1).expand_as(msg), alpha.unsqueeze(1) * msg)

        u_pre = h_pr + agg                                 # (N, d)

        # Вариационная выборка (формулы 3.13–3.14)
        params = self.split_proj(u_pre)                    # (N, 2d)
        mu_u, log_var_u = params.chunk(2, dim=-1)          # (N, d) each
        log_var_u = log_var_u.clamp(-6, 4)
        sigma_u = torch.exp(0.5 * log_var_u)
        u_hat = mu_u + sigma_u * torch.randn_like(mu_u)   # (N, d)
        return u_hat, mu_u, log_var_u


# ---------------------------------------------------------------------------
# LatentConfounderModule
# ---------------------------------------------------------------------------

class LatentConfounderModule(nn.Module):
    """K скрытых общих факторов ẑ_k (формулы 3.15–3.19).

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
        self.state_dim = state_dim

        # q(z_k | H): глобальный энкодер (формула 3.16)
        self.z_mu = nn.ModuleList(
            [nn.Linear(state_dim, confounder_dim) for _ in range(n_confounders)]
        )
        self.z_log_sigma = nn.ModuleList(
            [nn.Linear(state_dim, confounder_dim) for _ in range(n_confounders)]
        )
        # Маска m_ki (формула 3.18): [z_k || h_i] → (0, 1)
        self.mask_proj = nn.Linear(confounder_dim + state_dim, 1)
        # Декодирование ẑ_k → вклад (формула 3.19): dz → d
        self.z_dec = nn.Linear(confounder_dim, state_dim)

    def forward(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """
        Параметры
        ----------
        h : (N, d)

        Возвращает
        ----------
        correction : (N, d)    — суммарная поправка Σ_k m_ki · W_dec · ẑ_k
        z_hats     : list[K]   — выборки (dz,) каждого фактора
        kl_terms   : list[K]   — KL-дивергенции
        """
        N = h.shape[0]
        h_mean = h.mean(0, keepdim=True)               # (1, d) — глобальный пул

        correction = torch.zeros(N, self.state_dim, device=h.device, dtype=h.dtype)
        z_hats, kl_terms = [], []

        for k in range(self.K):
            mu_k       = self.z_mu[k](h_mean)             # (1, dz)
            log_sig_k  = self.z_log_sigma[k](h_mean).clamp(-6, 4)
            sigma_k    = torch.exp(0.5 * log_sig_k)
            z_hat_k    = (mu_k + sigma_k * torch.randn_like(mu_k)).squeeze(0)  # (dz,)

            kl_k = -0.5 * (1 + log_sig_k - mu_k.pow(2) - log_sig_k.exp()).sum()
            z_hats.append(z_hat_k)
            kl_terms.append(kl_k)

            # Маска m_ki (формула 3.18)
            z_exp = z_hat_k.unsqueeze(0).expand(N, -1)    # (N, dz)
            m_ki = torch.sigmoid(self.mask_proj(torch.cat([z_exp, h], dim=-1)))  # (N, 1)

            # Вклад (формула 3.19)
            dec = self.z_dec(z_hat_k).unsqueeze(0).expand(N, -1)  # (N, d)
            correction = correction + m_ki * dec

        return correction, z_hats, kl_terms


# ---------------------------------------------------------------------------
# ConfoundedVGAE — объединяющий класс
# ---------------------------------------------------------------------------

class ConfoundedVGAE(nn.Module):
    """Вариационный графовый АКЭ со скрытыми конфаундерами (раздел 3.2).

    Реализует полный конвейер абдукции:
        û_i = u_i + Σ_k m_ki · W_dec · ẑ_k

    Параметры
    ----------
    state_dim : int
        d = 128.
    n_confounders : int
        K = 3.
    confounder_dim : int
        d_z = 32.
    n_node_types : int
        Число типов узлов.
    edge_dim : int
        Размерность признаков рёбер.
    """

    def __init__(
        self,
        state_dim: int = 128,
        n_confounders: int = 3,
        confounder_dim: int = 32,
        n_node_types: int = 4,
        edge_dim: int = 16,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim

        self.exogenous_enc = ExogenousEncoder(state_dim, edge_dim, n_node_types)
        self.confounder_mod = LatentConfounderModule(state_dim, n_confounders, confounder_dim)
        # Декодер: û_i → ĥ_i
        self.decoder = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2),
            nn.ReLU(),
            nn.Linear(state_dim * 2, state_dim),
        )

    def encode(
        self,
        h: torch.Tensor,                         # (N, d)
        edge_index: torch.Tensor,                # (2, E)
        edge_type: torch.Tensor,                 # (E,) long — тип ребра
        node_types: Optional[torch.Tensor] = None,  # (N,) long
    ) -> tuple[torch.Tensor, list, list, torch.Tensor]:
        """Абдукция: вычисляет модифицированные экзогенные переменные û.

        Возвращает
        ----------
        exogenous  : (N, d)          — û_i
        conf_latents : list[K]       — ẑ_k (dz,) каждый
        masks      : list[K]         — пустой список (маски внутри LatentConfounder)
        kl_loss    : scalar Tensor   — KL_u + KL_z
        """
        N = h.shape[0]
        if node_types is None:
            node_types = torch.zeros(N, dtype=torch.long, device=h.device)

        # Признаки рёбер из типа (one-hot размерности 4, проецируем в edge_dim)
        edge_feats = self._edge_type_to_feats(edge_type, h.device, h.dtype)

        u_hat, mu_u, log_var_u = self.exogenous_enc(h, node_types, edge_index, edge_feats)
        kl_u = -0.5 * (1 + log_var_u - mu_u.pow(2) - log_var_u.exp()).sum(dim=-1).mean()

        correction, z_hats, kl_terms = self.confounder_mod(h)
        kl_z = sum(kl_terms)

        exogenous = u_hat + correction           # û_i = u_i + Σ m_ki·W_dec·ẑ_k
        kl_loss = kl_u + kl_z

        return exogenous, z_hats, [], kl_loss

    def decode(
        self,
        exogenous_modified: torch.Tensor,        # (N, d) — û
        edge_index: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Реконструкция ĥ из экзогенных переменных.

        Возвращает
        ----------
        h_recon : (N, d)
        """
        return self.decoder(exogenous_modified)

    def independence_loss(self, exogenous: torch.Tensor) -> torch.Tensor:
        """Среднее |corr(û_i, û_j)| по всем парам i≠j (формула 5.5).

        Параметры
        ----------
        exogenous : (N, d)

        Возвращает
        ----------
        scalar Tensor
        """
        N = exogenous.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=exogenous.device)

        # Нормализуем каждую строку
        u = exogenous - exogenous.mean(dim=1, keepdim=True)
        std = u.std(dim=1, keepdim=True).clamp(min=1e-8)
        u_norm = u / std                         # (N, d)

        # Корреляционная матрица строк (по признакам d)
        corr = (u_norm @ u_norm.T) / u_norm.shape[1]   # (N, N)

        # Средний |corr| по внедиагональным элементам
        mask = 1 - torch.eye(N, device=exogenous.device)
        return (corr.abs() * mask).sum() / mask.sum()

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _edge_type_to_feats(
        self, edge_type: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Конвертирует int-тип ребра в вещественный признак (N_e, edge_dim)."""
        n_types = 4
        edge_dim = self.exogenous_enc.attn.in_features - 2 * self.state_dim
        if edge_type.numel() == 0:
            return torch.zeros(0, edge_dim, device=device, dtype=dtype)
        # Sinusoidal encoding типа ребра (0-индексированного) в edge_dim
        pos = edge_type.float().unsqueeze(1)                       # (E, 1)
        freqs = torch.arange(edge_dim, device=device, dtype=dtype) # (edge_dim,)
        feats = torch.sin(pos * (freqs + 1) / (n_types + 1))      # (E, edge_dim)
        return feats
