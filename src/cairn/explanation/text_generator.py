"""Генератор текстовых объяснений (раздел 4.3).

Три уровня с нарастающим качеством:
  TemplateTextGenerator  — шаблонный генератор (MVP, мгновенно)
  TextExplanationGenerator — фасад с выбором уровня (template | local_llm | cloud_llm)
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

from cairn.explanation.evidence_chain import EvidenceChain

# Рекомендации по типу сбоя
_RECOMMENDATIONS: Dict[str, str] = {
    "cpu_exhaustion":  "Горизонтально масштабируйте сервис или оптимизируйте CPU-intensive операции.",
    "memory_pressure": "Проверьте утечки памяти и увеличьте memory limit контейнера.",
    "latency_spike":   "Проверьте зависимости upstream, очереди и тайм-ауты.",
    "overload":        "Включите circuit breaker и добавьте rate limiting.",
    "unknown":         "Проверьте метрики и журналы сервиса для определения причины.",
}

# Описания типов рёбер для текста
_EDGE_DESCRIPTIONS: Dict[str, str] = {
    "call":        "вызывает",
    "colocation":  "размещён совместно с",
    "loadbalance": "является репликой сервиса",
    "adaptive":    "коррелирует с",
}


class TemplateTextGenerator:
    """Шаблонный генератор объяснений (уровень 0, MVP).

    Шаблон:
        «Первопричина: {service} ({fault_type}, ПЭ={ce:.2f}).
        Путь распространения: {chain}.
        Рекомендация: {recommendation}.»
    """

    # Основной шаблон отчёта
    _TEMPLATE = (
        "Первопричина: {root_name} ({fault_type}, ПЭ={ce:.2f}).\n"
        "Путь распространения: {path}.\n"
        "Рекомендация: {recommendation}\n"
        "Достоверность вывода: {confidence:.0%}."
    )

    def generate(self, chain: EvidenceChain) -> str:
        """Генерирует текстовый отчёт по цепочке доказательств."""
        if not chain.path_nodes:
            return f"Первопричина (индекс {chain.root_cause_idx}): нет данных для объяснения."

        root = chain.path_nodes[0]
        root_name  = root.node_name
        fault_type = root.failure_type or "unknown"
        recommendation = _RECOMMENDATIONS.get(fault_type, _RECOMMENDATIONS["unknown"])

        # 1.2: Добавляем доминантную метрику если известна
        dominant = getattr(root, "dominant_metric", None)
        if dominant:
            _METRIC_LABELS = {
                "cpu":        "CPU",
                "memory":     "Memory",
                "latency_ms": "Latency",
                "rps":        "RPS",
            }
            fault_type = f"{fault_type}, доминирует {_METRIC_LABELS.get(dominant, dominant)}"

        # Путь распространения с типами рёбер
        path = self._format_path(chain)

        text = self._TEMPLATE.format(
            root_name=root_name,
            fault_type=fault_type,
            ce=chain.causal_effect,
            path=path,
            recommendation=recommendation,
            confidence=chain.confidence,
        )

        # Предупреждения
        if chain.confounder_warnings:
            text += "\n" + "\n".join(f"⚠  {w}" for w in chain.confounder_warnings)
        if chain.drift_warning:
            text += "\n⚠  Обнаружен дрейф распределения — результат требует дополнительной проверки."

        return text

    def _format_path(self, chain: EvidenceChain) -> str:
        """Форматирует путь распространения с аннотацией типов рёбер."""
        if len(chain.path_nodes) == 1:
            return chain.path_nodes[0].node_name

        edge_map = {(e.src, e.dst): e for e in chain.path_edges}
        parts = [chain.path_nodes[0].node_name]
        for i in range(1, len(chain.path_nodes)):
            src_idx = chain.path_nodes[i - 1].node_idx
            dst_idx = chain.path_nodes[i].node_idx
            edge = edge_map.get((src_idx, dst_idx))
            if edge:
                verb = _EDGE_DESCRIPTIONS.get(edge.edge_type, "→")
                parts.append(f"--[{verb}]--> {chain.path_nodes[i].node_name}")
            else:
                parts.append(f"→ {chain.path_nodes[i].node_name}")

        return " ".join(parts)


class TextExplanationGenerator:
    """Фасад для выбора уровня генерации объяснений (раздел 4.3).

    Параметры
    ----------
    level : str
        "template" | "local_llm" | "cloud_llm"
    local_llm_model : str | None
        Путь к локальной LLM.
    """

    def __init__(
        self,
        level: Literal["template", "local_llm", "cloud_llm"] = "template",
        local_llm_model: Optional[str] = None,
    ) -> None:
        self.level = level
        self._template = TemplateTextGenerator()
        self._local_llm = None

        if level == "local_llm" and local_llm_model:
            try:
                from transformers import pipeline  # type: ignore[import-untyped]
                self._local_llm = pipeline("text-generation", model=local_llm_model)
            except (ImportError, ModuleNotFoundError):
                pass  # Fallback на шаблонный генератор

    def generate(self, chain: EvidenceChain) -> str:
        if self.level == "template" or self._local_llm is None:
            return self._template.generate(chain)
        if self.level == "local_llm":
            return self._local_llm_generate(chain)
        # cloud_llm — заглушка для MVP
        return self._template.generate(chain)

    def _local_llm_generate(self, chain: EvidenceChain) -> str:
        context = chain.summary()
        prompt = (
            f"На основе следующих данных о сбое:\n{context}\n\n"
            "Сформулируй развёрнутое объяснение первопричины и рекомендации:"
        )
        try:
            result = self._local_llm(prompt, max_new_tokens=256, do_sample=False)
            return result[0]["generated_text"].replace(prompt, "").strip()
        except Exception:
            return self._template.generate(chain)


# Alias: старое имя для обратной совместимости
TemplateGenerator = TemplateTextGenerator