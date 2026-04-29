"""Фаза рассуждения CAIRN — причинно-следственное ядро.

Публичный API:
    ConditionalGMM           — условная GMM нормального состояния
    ConfoundedVGAE           — VGAE со скрытыми конфаундерами
    ExogenousEncoder         — attention-агрегация предшественников
    LatentConfounderModule   — K скрытых общих факторов
    CounterfactualModule     — дифференцируемое вмешательство do(i)
    HypergraphConv           — нормализованная гиперграфовая свёртка
    MultiRootCauseDecomposition — трёхрежимная декомпозиция
    DecompositionMode        — режимы декомпозиции
    CascadeFunnel            — каскадная воронка 500→30→5→1
    CausalGraphVerifier      — верификатор по 5 аксиомам
    AxiomStatus, AxiomResult, VerificationReport — результаты верификации
"""

from cairn.reasoning.conditional_gmm import ConditionalGMM
from cairn.reasoning.confounded_vgae import (
    ConfoundedVGAE,
    ExogenousEncoder,
    LatentConfounderModule,
)
from cairn.reasoning.counterfactual import (
    CounterfactualModule,
    CounterfactualInterventionModule,
    HypergraphConv,
)
from cairn.reasoning.decomposition import (
    MultiRootCauseDecomposition,
    DecompositionMode,
    decompose_multiple_roots,
    additivity_ratio,
    joint_causal_effect,
)
from cairn.reasoning.funnel import CascadeFunnel
from cairn.reasoning.graph_verifier import (
    CausalGraphVerifier,
    AxiomStatus,
    AxiomResult,
    VerificationReport,
)

__all__ = [
    "ConditionalGMM",
    "ConfoundedVGAE",
    "ExogenousEncoder",
    "LatentConfounderModule",
    "CounterfactualModule",
    "CounterfactualInterventionModule",
    "HypergraphConv",
    "MultiRootCauseDecomposition",
    "DecompositionMode",
    "decompose_multiple_roots",
    "additivity_ratio",
    "joint_causal_effect",
    "CascadeFunnel",
    "CausalGraphVerifier",
    "AxiomStatus",
    "AxiomResult",
    "VerificationReport",
]
