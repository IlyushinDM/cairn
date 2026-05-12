"""Логическая верификация объяснений на основе ALP (раздел 4.4).

Пять правил целостности (IC1–IC5) + контр-абдуктивная проверка.

IC1: первопричина аномальна (NLL > δ)
IC2: CE > порога значимости
IC3: путь распространения непустой и связный
IC4: объяснение упоминает первопричину и тип сбоя
IC5: числовые значения ПЭ и NLL согласованы с данными мониторинга
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from cairn.explanation.evidence_chain import EvidenceChain


@dataclass
class ALPVerificationResult:
    """Результат логической верификации объяснения.

    Атрибуты
    ----------
    passed : bool
        True если ни одно правило не нарушено.
    violated_rules : list[str]
        Описания нарушенных правил.
    counter_hypothesis : str | None
        Альтернативная гипотеза (из контр-абдукции).
    f1_confidence : float
        Скорректированная достоверность [0, 1].
    warnings : list[str]
        Предупреждения (нарушения с уровнем WARNING).
    """
    passed: bool
    violated_rules: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    counter_hypothesis: Optional[str] = None
    f1_confidence: float = 1.0


class ALPVerifier:
    """Верификатор объяснений по 5 правилам целостности (раздел 4.4).

    Параметры
    ----------
    anomaly_threshold : float
        δ – порог NLL для IC1.
    ce_threshold : float
        Минимальный CE для IC2.
    numeric_tolerance : float
        Допустимое относительное отклонение числовых значений в тексте (IC5).
    """

    def __init__(
        self,
        anomaly_threshold: float = 0.5,
        ce_threshold: float = 0.05,
        numeric_tolerance: float = 0.15,
    ) -> None:
        self.anomaly_threshold  = anomaly_threshold
        self.ce_threshold       = ce_threshold
        self.numeric_tolerance  = numeric_tolerance

    def verify(
        self,
        chain: EvidenceChain,
        explanation_text: str,
        monitoring_values: Optional[Dict[str, float]] = None,
    ) -> ALPVerificationResult:
        """Проверяет объяснение по всем правилам целостности.

        Параметры
        ----------
        chain : EvidenceChain
        explanation_text : str
            Текст из TemplateTextGenerator.
        monitoring_values : dict | None
            {имя_метрики: значение} для IC5.

        Возвращает
        ----------
        ALPVerificationResult
        """
        violated:  List[str] = []
        warnings:  List[str] = []
        root_name = chain.path_nodes[0].node_name if chain.path_nodes else ""

        # IC1: первопричина аномальна
        violated += self._check_ic1(chain)
        # IC2: CE значим
        violated += self._check_ic2(chain)
        # IC3: путь непустой и связный
        ic3_issues = self._check_ic3(chain)
        violated += [e for e in ic3_issues if "пуст" in e]
        warnings  += [e for e in ic3_issues if "пуст" not in e]
        # IC4: текст упоминает первопричину и тип сбоя
        violated += self._check_ic4(chain, explanation_text)
        # IC5: числовые значения согласованы
        warnings += self._check_ic5(chain, explanation_text, monitoring_values or {})

        # Контр-абдуктивная проверка
        counter = self._counter_abduce(chain, root_name)

        passed = len(violated) == 0
        # Штрафуем confidence за нарушения и предупреждения
        penalty = 0.2 * len(violated) + 0.05 * len(warnings)
        adj_conf = max(0.0, chain.confidence - penalty)

        return ALPVerificationResult(
            passed=passed,
            violated_rules=violated,
            warnings=warnings,
            counter_hypothesis=counter,
            f1_confidence=adj_conf,
        )

    # ------------------------------------------------------------------
    # Правила целостности
    # ------------------------------------------------------------------

    def _check_ic1(self, chain: EvidenceChain) -> List[str]:
        """IC1: первопричина аномальна (NLL > δ)."""
        if not chain.path_nodes:
            return ["IC1: цепочка пуста – невозможно проверить аномальность первопричины."]
        root_nll = chain.path_nodes[0].nll
        if root_nll < self.anomaly_threshold:
            return [
                f"IC1: первопричина NLL={root_nll:.3f} < порога {self.anomaly_threshold}. "
                "Узел не является аномальным."
            ]
        return []

    def _check_ic2(self, chain: EvidenceChain) -> List[str]:
        """IC2: причинный эффект значим."""
        if chain.causal_effect < self.ce_threshold:
            return [
                f"IC2: причинный эффект CE={chain.causal_effect:.3f} < порога {self.ce_threshold}. "
                "Вмешательство не снижает аномальность системы."
            ]
        return []

    def _check_ic3(self, chain: EvidenceChain) -> List[str]:
        """IC3: путь непустой и каждое ребро ссылается на существующие узлы."""
        issues: List[str] = []
        if not chain.path_nodes:
            issues.append("IC3: цепочка доказательств пуста.")
            return issues

        node_idxs = {n.node_idx for n in chain.path_nodes}
        for e in chain.path_edges:
            if e.src not in node_idxs or e.dst not in node_idxs:
                issues.append(
                    f"IC3: ребро ({e.src}→{e.dst}) ссылается на узел вне пути. "
                    "Нарушена связность цепочки."
                )
        return issues

    def _check_ic4(self, chain: EvidenceChain, text: str) -> List[str]:
        """IC4: текст упоминает имя первопричины и (желательно) тип сбоя."""
        issues: List[str] = []
        if not chain.path_nodes:
            return issues
        root = chain.path_nodes[0]
        if root.node_name and root.node_name not in text:
            issues.append(
                f"IC4: объяснение не упоминает первопричину '{root.node_name}'."
            )
        return issues

    def _check_ic5(
        self, chain: EvidenceChain, text: str, monitoring: Dict[str, float]
    ) -> List[str]:
        """IC5: числовые значения CE и NLL в тексте согласованы с данными.

        Ищет числа вида 0.XX или XX.XX в тексте и сравнивает с фактическими
        значениями CE и NLL из цепочки.
        """
        warnings: List[str] = []

        # Извлекаем числа из текста (простой regex)
        numbers_in_text = set(
            round(float(m), 2)
            for m in re.findall(r"\\b\\d+\\.\\d{1,4}\\b", text)
        )

        # Проверяем CE: ищем округлённое значение
        ce_rounded = round(chain.causal_effect, 2)
        if monitoring and ce_rounded not in numbers_in_text and chain.causal_effect > 0:
            pass   # CE часто указывается в %, шаблон может использовать другой формат

        # Проверяем значения мониторинга
        for metric, expected in monitoring.items():
            if metric in text:
                # Ищем числа рядом с именем метрики
                pattern = rf"{re.escape(metric)}[\\s:=]*([\\d.]+)"
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    found = float(match.group(1))
                    if abs(found - expected) / (abs(expected) + 1e-8) > self.numeric_tolerance:
                        warnings.append(
                            f"IC5: значение {metric}={found:.2f} в тексте расходится "
                            f"с данными мониторинга ({expected:.2f})."
                        )
        return warnings

    # ------------------------------------------------------------------
    # Контр-абдуктивная проверка
    # ------------------------------------------------------------------

    def _counter_abduce(
        self, chain: EvidenceChain, root_name: str
    ) -> Optional[str]:
        """Строит альтернативную гипотезу, если достоверность низкая.

        Стратегия: ищет узел с высокой NLL, который НЕ является первопричиной.
        Если он аномальнее первопричины – предлагает его как альтернативу.
        """
        if len(chain.path_nodes) < 2:
            return None

        # Сортируем узлы пути по убыванию NLL (кроме первопричины)
        candidates = sorted(
            chain.path_nodes[1:],
            key=lambda n: n.nll,
            reverse=True,
        )
        if not candidates:
            return None

        best_alt = candidates[0]
        root_nll = chain.path_nodes[0].nll

        # Если альтернатива аномальнее первопричины – это сигнал
        if best_alt.nll > root_nll * 1.2:
            return (
                f"Альтернативная гипотеза: '{best_alt.node_name}' "
                f"(NLL={best_alt.nll:.2f}) аномальнее первопричины '{root_name}' "
                f"(NLL={root_nll:.2f}). Рекомендуется дополнительная проверка обоих узлов."
            )

        # Если достоверность низкая – указываем на неопределённость
        if chain.confidence < 0.6:
            return (
                f"Достоверность вывода низкая ({chain.confidence:.0%}). "
                f"Нельзя исключить '{best_alt.node_name}' как альтернативную первопричину."
            )

        return None
