"""Демонстрационный запуск CAIRN на синтетических данных.

Использование:
    python scripts/demo.py --config configs/demo.yaml
"""

from __future__ import annotations

import argparse

import torch
import numpy as np

# Теперь импорты из вашего пакета работают, потому что пакет установлен
from cairn.config import load_config
from cairn.utils.logging import setup_logging
from cairn.perception.hypergraph_builder import HypergraphBuilder
from cairn.reasoning.conditional_gmm import ConditionalGMM
from cairn.reasoning.counterfactual import CounterfactualInterventionModule
from cairn.reasoning.funnel import CascadeFunnel
from cairn.reasoning.graph_verifier import CausalGraphVerifier
from cairn.explanation.evidence_chain import EvidenceChain, NodeAnnotation, EdgeAnnotation
from cairn.explanation.text_generator import TextExplanationGenerator
from cairn.explanation.alp_verifier import ALPVerifier


def make_synthetic_data(n_services: int, T: int, F: int, state_dim: int):
    """Генерирует синтетические данные для демонстрации."""
    metrics = torch.randn(n_services, T, F)
    # Вносим аномалию в сервис #2
    metrics[2, T // 2:, :] += 5.0
    states = torch.randn(n_services, state_dim)
    states[2] += 3.0
    contexts = torch.randn(n_services, 16)
    return metrics, states, contexts


def main():
    parser = argparse.ArgumentParser(description="CAIRN Demo")
    parser.add_argument("--config", default="configs/demo.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level)

    from loguru import logger
    logger.info("Запуск CAIRN Demo")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Синтетические данные
    N, T, F = 10, 120, 8
    d = cfg.model.state_dim
    metrics, states, contexts = make_synthetic_data(N, T, F, d)

    # Условная GMM для нормального состояния
    gmm = ConditionalGMM(
        state_dim=d,
        context_dim=cfg.model.context_dim,
        n_components=cfg.model.gmm_components,
    )
    nll = gmm.nll(states, contexts)                       # (N,)
    prototypes = gmm.conditional_prototype(contexts) # (N, d)                                              # (N, d)

    # Гиперграф (простая топология для демо)
    builder = HypergraphBuilder(N)
    hg = builder.from_topology(
        call_paths=[[0, 1, 2], [3, 4, 2], [5, 6, 7]],
        colocated_groups=[[0, 1, 3], [2, 4]],
        lb_groups=[[2, 8, 9]],
    )
    incidence = hg.incidence_matrix()
    edge_weights = hg.edge_weights()

    # Контрфактический анализ
    cf_module = CounterfactualInterventionModule(state_dim=d)
    funnel = CascadeFunnel(
        l0_top_k=cfg.funnel.l0_top_k,
        l1_top_k=cfg.funnel.l1_top_k,
        l2_top_k=cfg.funnel.l2_top_k,
    )

    adjacency = (incidence @ incidence.T).clamp(0, 1).fill_diagonal_(0)
    adj_norm = adjacency / adjacency.sum(dim=1, keepdim=True).clamp(min=1)


    def nll_fn(s, c=None):
        curr_context = c if c is not None else contexts[:s.size(0)]
        return gmm.nll(s, curr_context)

    ranked = funnel.run(
        nll, states, adj_norm, cf_module, nll_fn, prototypes, incidence, edge_weights
    )
    root_idx, root_pe = ranked[0]
    logger.info(f"Первопричина: сервис #{root_idx}, ПЭ = {root_pe:.3f}")

    # Верификация графа
    verifier = CausalGraphVerifier(**{
        k: v for k, v in cfg.verifier.__dict__.items()
        if k in ("temporal_tolerance_sec", "transitivity_threshold", "monotonicity_epsilon")
    })
    edges_list = [(0, 2), (3, 2), (2, 7)]
    report = verifier.verify(
        edges=edges_list,
        causal_effects={i: nll[i].item() for i in range(N)},
        anomaly_times={i: float(i) for i in range(N)},
        physical_edges={(s, d) for s, d in edges_list},
        root_candidate=root_idx,
    )
    logger.info(f"\n{report.summary()}")

    # Цепочка доказательств
    chain = EvidenceChain(
        root_cause_idx=root_idx,
        path_nodes=[
            NodeAnnotation(root_idx, f"service-{root_idx}", nll[root_idx].item(), "cpu_usage"),
            NodeAnnotation(7, "service-7", nll[7].item(), "latency"),
        ],
        path_edges=[
            EdgeAnnotation(root_idx, 7, "call", strength=root_pe),
        ],
        causal_effect=root_pe,
        confidence=report.confidence,
    )

    # Генерация объяснения
    gen = TextExplanationGenerator(level=cfg.explanation.generator_level)
    explanation = gen.generate(chain)
    logger.info(f"\n--- Объяснение ---\n{explanation}")

    # Логическая верификация
    alp = ALPVerifier(anomaly_threshold=0.0)
    alp_result = alp.verify(chain, explanation)
    status = "✅ Прошла" if alp_result.passed else "❌ Не прошла"
    logger.info(f"Логическая верификация: {status}")
    if not alp_result.passed:
        for r in alp_result.violated_rules:
            logger.warning(f"  {r}")

    logger.success("Демонстрация завершена.")


if __name__ == "__main__":
    main()
