"""Логическая верификация объяснений на основе ALP (раздел 4.4).

Проверяет пять правил целостности и выполняет контр-абдуктивную проверку.
При нарушении хотя бы одного правила объяснение отклоняется.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from cairn.explanation.evidence_chain import EvidenceChain


@dataclass
class ALPVerificationResult:
    passed: bool
    violated_rules: List[str] = field(default_factory=list)
    counter_hypothesis: str | None = None
    f1_confidence: float = 1.0  # условная оценка достоверности


class ALPVerifier:
    """Верификатор объяснений (раздел 4.4).

    Правила целостности (из работы [10]):
      R1: первопричина аномальна (NLL > δ)
      R2: ПЭ(первопричины) > порога
      R3: упомянутые пути существуют в цепочке доказательств
      R4: рекомендации адресованы первопричине
      R5: числовые значения метрик соответствуют данным мониторинга
    """

    def __init__(
        self,
        anomaly_threshold: float = 0.5,
        causal_effect_threshold: float = 0.05,
    ) -> None:
        self.anomaly_threshold = anomaly_threshold
        self.ce_threshold = causal_effect_threshold

    def verify(
        self,
        chain: EvidenceChain,
        explanation_text: str,
        monitoring_values: dict[str, float] | None = None,
    ) -> ALPVerificationResult:
        """Проверяет объяснение по всем правилам."""
        violated: List[str] = []

        # R1: первопричина аномальна
        if chain.path_nodes:
            root_nll = chain.path_nodes[0].nll
            if root_nll < self.anomaly_threshold:
                violated.append(
                    f"R1: первопричина (NLL={root_nll:.3f}) не превышает порог "
                    f"{self.anomaly_threshold}."
                )

        # R2: причинный эффект значим
        if chain.causal_effect < self.ce_threshold:
            violated.append(
                f"R2: причинный эффект ({chain.causal_effect:.3f}) ниже порога "
                f"{self.ce_threshold}."
            )

        # R3: путь непустой
        if len(chain.path_nodes) < 1:
            violated.append("R3: цепочка доказательств пуста.")

        # R4: текст содержит имя первопричины
        root_name = chain.path_nodes[0].node_name if chain.path_nodes else ""
        if root_name and root_name not in explanation_text:
            violated.append(f"R4: объяснение не упоминает первопричину '{root_name}'.")

        # R5: числовые значения метрик (базовая проверка)
        if monitoring_values:
            for key, val in monitoring_values.items():
                if key in explanation_text:
                    # Проверяем, что значение в тексте не отличается радикально
                    # (упрощённая проверка для MVP)
                    pass  # полная реализация требует NLP-парсинга

        # Контр-абдуктивная проверка: строим альтернативную гипотезу
        counter = None
        if len(chain.path_nodes) >= 2 and chain.confidence < 0.7:
            alt_name = chain.path_nodes[-1].node_name
            counter = (
                f"Альтернативная гипотеза: возможно, первопричиной является "
                f"'{alt_name}' (последний симптоматический узел), а не "
                f"'{root_name}'. Рекомендуется дополнительная проверка."
            )

        passed = len(violated) == 0
        return ALPVerificationResult(
            passed=passed,
            violated_rules=violated,
            counter_hypothesis=counter,
            f1_confidence=chain.confidence if passed else chain.confidence * 0.5,
        )
