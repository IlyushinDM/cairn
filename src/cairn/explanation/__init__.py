"""Фаза объяснения CAIRN — цепочки доказательств, верификация, отчёты.

Публичный API:
    EvidenceChain           — структура данных цепочки доказательств
    NodeAnnotation          — аннотация узла
    EdgeAnnotation          — аннотация ребра
    EvidenceChainBuilder    — строит цепочку из результатов рассуждения
    TemplateTextGenerator   — шаблонный генератор объяснений (MVP)
    TextExplanationGenerator — фасад с выбором уровня генерации
    ALPVerifier             — верификатор по 5 правилам целостности
    ALPVerificationResult   — результат верификации
    MediationDiagnostic     — медиационная диагностика вклада компонентов
    MediationReport         — отчёт медиации
    LayerContribution       — вклад слоя свёртки
    EdgeContribution        — вклад гиперребра
"""

from cairn.explanation.evidence_chain import (
    EvidenceChain,
    NodeAnnotation,
    EdgeAnnotation,
    EvidenceChainBuilder,
)
from cairn.explanation.text_generator import (
    TemplateTextGenerator,
    TextExplanationGenerator,
    TemplateGenerator,   # alias
)
from cairn.explanation.alp_verifier import (
    ALPVerifier,
    ALPVerificationResult,
)
from cairn.explanation.mediation import (
    MediationDiagnostic,
    MediationReport,
    LayerContribution,
    EdgeContribution,
)

__all__ = [
    # Цепочка доказательств
    "EvidenceChain",
    "NodeAnnotation",
    "EdgeAnnotation",
    "EvidenceChainBuilder",
    # Генераторы текста
    "TemplateTextGenerator",
    "TextExplanationGenerator",
    "TemplateGenerator",
    # Верификатор
    "ALPVerifier",
    "ALPVerificationResult",
    # Медиация
    "MediationDiagnostic",
    "MediationReport",
    "LayerContribution",
    "EdgeContribution",
]
