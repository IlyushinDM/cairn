"""Генератор текстовых объяснений (раздел 4.3).

Три уровня с нарастающим качеством:
  1. template   — шаблонный генератор (мгновенно)
  2. local_llm  — локальная языковая модель (2-3 с)
  3. cloud_llm  — облачная языковая модель с RAG (5-8 с, опционально)
"""

from __future__ import annotations

from typing import Literal

from cairn.explanation.evidence_chain import EvidenceChain


class TemplateGenerator:
    """Шаблонный генератор объяснений (уровень 1)."""

    TEMPLATE_RU = (
        "Обнаружена первопричина сбоя: {root_name}.\n"
        "При нормализации этого компонента аномальность системы снизится на {effect:.0%}.\n"
        "Путь распространения сбоя: {path}.\n"
        "Рекомендация: проверьте и восстановите работоспособность {root_name}.\n"
        "Достоверность вывода: {confidence:.0%}."
    )

    def generate(self, chain: EvidenceChain) -> str:
        root_name = (
            chain.path_nodes[0].node_name
            if chain.path_nodes
            else str(chain.root_cause_idx)
        )
        path = " → ".join(n.node_name for n in chain.path_nodes)
        text = self.TEMPLATE_RU.format(
            root_name=root_name,
            effect=chain.causal_effect,
            path=path,
            confidence=chain.confidence,
        )
        if chain.confounder_warnings:
            text += "\n⚠ " + "\n⚠ ".join(chain.confounder_warnings)
        if chain.drift_warning:
            text += "\n⚠ Обнаружен дрейф распределения — результат требует дополнительной проверки."
        return text


class TextExplanationGenerator:
    """Фасад для трёх уровней генерации объяснений (раздел 4.3).

    Параметры
    ----------
    level : str
        "template" | "local_llm" | "cloud_llm"
    local_llm_model : str | None
        Путь к локальной LLM (для уровня "local_llm").
    """

    def __init__(
        self,
        level: Literal["template", "local_llm", "cloud_llm"] = "template",
        local_llm_model: str | None = None,
    ) -> None:
        self.level = level
        self.template_gen = TemplateGenerator()
        self._local_llm = None

        if level == "local_llm" and local_llm_model:
            try:
                # Lazy import — не обязательная зависимость
                from transformers import pipeline
                self._local_llm = pipeline("text-generation", model=local_llm_model)
            except ImportError:
                pass  # Фоллбек на шаблонный генератор

    def generate(self, chain: EvidenceChain) -> str:
        """Генерирует текстовое объяснение на основе цепочки доказательств."""
        if self.level == "template" or self._local_llm is None:
            return self.template_gen.generate(chain)

        if self.level == "local_llm":
            return self._local_llm_generate(chain)

        # cloud_llm — заглушка для MVP
        return self.template_gen.generate(chain)

    def _local_llm_generate(self, chain: EvidenceChain) -> str:
        """Генерация через локальную LLM с данными из цепочки доказательств."""
        context = chain.summary()
        prompt = (
            f"На основе следующих данных о сбое:\n{context}\n\n"
            "Сформулируй развёрнутое объяснение первопричины и рекомендации для инженера:"
        )
        try:
            result = self._local_llm(prompt, max_new_tokens=256, do_sample=False)
            return result[0]["generated_text"].replace(prompt, "").strip()
        except Exception:
            return self.template_gen.generate(chain)
