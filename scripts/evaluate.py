"""Скрипт оценки обученной модели CAIRN.

Использование:
    # Честная оценка на тестовом сценарии (unseen data):
    python scripts/evaluate.py --checkpoint data/sample/demo_model.pt \
                               --data-dir data/sample/scenario_3

    # Оценка на всём demo-датасете (для сравнения):
    python scripts/evaluate.py --checkpoint data/sample/demo_model.pt \
                               --data-dir data/sample
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def compute_ndcg(ranked: list[tuple[int, float]], root_cause: int, k: int) -> float:
    """NDCG@k — нормализованный дисконтированный выигрыш.

    Для задачи RCA: релевантность 1 только у root_cause, остальные 0.
    NDCG@k = DCG@k / IDCG@k, где IDCG@k = 1 (если root в топ-k).
    """
    import math
    for rank, (node_idx, _) in enumerate(ranked[:k], start=1):
        if node_idx == root_cause:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def compute_mrr(ranked: list[tuple[int, float]], root_cause: int) -> float:
    """MRR — Mean Reciprocal Rank.

    1/rank для первого корректного ответа.
    """
    for rank, (node_idx, _) in enumerate(ranked, start=1):
        if node_idx == root_cause:
            return 1.0 / rank
    return 0.0


def compute_precision_at_k(
    ranked: list[tuple[int, float]], root_cause: int, k: int
) -> float:
    """Precision@k — доля правильных ответов в топ-k."""
    top_k = [idx for idx, _ in ranked[:k]]
    return 1.0 if root_cause in top_k else 0.0


def compute_extended_metrics(
    ranked: list[tuple[int, float]], root_cause: int
) -> dict[str, float]:
    """Полный набор метрик для одного инцидента."""
    return {
        "AC@1":    compute_precision_at_k(ranked, root_cause, 1),
        "AC@3":    compute_precision_at_k(ranked, root_cause, 3),
        "AC@5":    compute_precision_at_k(ranked, root_cause, 5),
        "NDCG@1":  compute_ndcg(ranked, root_cause, 1),
        "NDCG@3":  compute_ndcg(ranked, root_cause, 3),
        "NDCG@5":  compute_ndcg(ranked, root_cause, 5),
        "MRR":     compute_mrr(ranked, root_cause),
    }


def _load_arch_config(ckpt: dict, checkpoint_path: Path) -> dict:
    """Восстанавливает параметры архитектуры из трёх источников по приоритету:

    1. arch_config внутри чекпоинта  (после pretrain_demo.py — всегда)
    2. demo_config.yaml рядом с .pt  (резервная копия)
    3. Shape-inference из весов      (крайний fallback)
    """
    from loguru import logger

    # ── Уровень 1 ─────────────────────────────────────────────────────────
    if "arch_config" in ckpt:
        logger.info("Архитектура из чекпоинта (arch_config) ✓")
        return ckpt["arch_config"]

    # ── Уровень 2 ─────────────────────────────────────────────────────────
    yaml_path = checkpoint_path.parent / "demo_config.yaml"
    if yaml_path.exists():
        try:
            import yaml
            with yaml_path.open(encoding="utf-8") as f:
                demo_cfg = yaml.safe_load(f)
            if "model" in demo_cfg:
                logger.info(f"Архитектура из {yaml_path.name} ✓")
                return demo_cfg["model"]
        except Exception as e:
            logger.warning(f"Не удалось прочитать {yaml_path}: {e}")

    # ── Уровень 3 ─────────────────────────────────────────────────────────
    logger.warning(
        "arch_config не найден. "
        "Восстанавливаем архитектуру из shape весов — возможны неточности."
    )
    state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))

    D              = state["state_builder.norm.weight"].shape[0]
    CTX            = state["state_builder.context_builder.proj.2.weight"].shape[0]
    F              = state["state_builder.metric_enc.ssm_branch.B"].shape[1]
    d_ssm          = state["state_builder.metric_enc.ssm_branch.proj.weight"].shape[0]
    n_components   = state["gmm.mlp_omega.2.bias"].shape[0]
    confounder_dim = state["vgae.confounder_mod.z_mu.0.weight"].shape[0]
    n_confounders  = sum(
        1 for k in state
        if k.startswith("vgae.confounder_mod.z_mu.") and k.endswith(".weight")
    ) or 1

    return {
        "state_dim":      D,
        "context_dim":    CTX,
        "n_metrics":      F,
        "d_ssm":          d_ssm,
        "d_met":          d_ssm * 2,
        "d_log":          d_ssm,
        "d_tr":           d_ssm,
        "d_brk":          d_ssm,
        "ssm_state_dim":  d_ssm * 2,
        "n_components":   n_components,
        "confounder_dim": confounder_dim,
        "n_confounders":  n_confounders,
        "log_vocab_size": 300,
        "window":         15,
        "n_conv_layers":  1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CAIRN Evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",   default="configs/default.yaml")
    parser.add_argument("--dataset",  default="demo",
                        choices=["demo", "gaia", "trainticket"])
    parser.add_argument("--data-dir", default="data/sample",
                        help="Путь к датасету. Для честной оценки укажите "
                             "data/sample/scenario_3 (тестовый сценарий).")
    parser.add_argument("--device",   default="cpu")
    parser.add_argument("--ablation", action="store_true",
                        help="Запустить ablation study (сравнение конфигураций)")
    parser.add_argument("--baseline", action="store_true",
                        help="Сравнить с baseline методами (Random, NLL-only)")
    args = parser.parse_args()

    from cairn.config import load_config
    from cairn.utils.logging import setup_logging

    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level)

    from loguru import logger
    logger.info(f"Чекпоинт: {args.checkpoint}")
    logger.info(f"Данные:    {args.data_dir}")

    # ── Данные ───────────────────────────────────────────────────────────
    import torch
    from cairn.training.data_loader import create_demo_dataset

    data_dir = Path(args.data_dir)
    dataset  = create_demo_dataset(data_dir)
    logger.info(dataset.summary())

    # ── Предупреждение если тестируемся на обучающих данных ──────────────
    ckpt_dir = Path(args.checkpoint).parent
    try:
        import yaml
        yaml_path = ckpt_dir / "demo_config.yaml"
        if yaml_path.exists():
            split_cfg = yaml.safe_load(yaml_path.open(encoding="utf-8")).get("split", {})
            train_scenarios = split_cfg.get("train", [])
            test_scenario   = split_cfg.get("test", "")
            data_name       = data_dir.name

            if data_name in train_scenarios:
                logger.warning(
                    f"⚠ '{data_name}' входил в TRAIN — метрики будут завышены! "
                    f"Для честной оценки используйте --data-dir .../{ test_scenario}"
                )
            elif data_name == test_scenario or str(data_dir).endswith(test_scenario):
                logger.info(f"Датасет '{data_name}' — тестовый сценарий ✓ (честная оценка)")
    except Exception:
        pass  # не критично, просто предупреждение

    # ── Топология ────────────────────────────────────────────────────────
    from cairn.connectors.csv_file import YAMLTopologyConnector
    from cairn.perception import HypergraphBuilder

    topo_path = data_dir / "topology.yaml"
    if not topo_path.exists():
        # Ищем топологию в родительской директории (для scenario_N)
        topo_path = data_dir.parent / "scenario_1" / "topology.yaml"
    topo       = YAMLTopologyConnector(topo_path).fetch()
    hypergraph = HypergraphBuilder.from_topology_data(topo)

    # ── Загружаем чекпоинт ───────────────────────────────────────────────
    from cairn.perception import StateBuilder
    from cairn.reasoning import ConditionalGMM, ConfoundedVGAE, CounterfactualModule
    from cairn.training  import CAIRNModel, CAIRNLoss, CAIRNTrainer, TrainerConfig

    ckpt  = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt.get("model_state", {}))

    A = _load_arch_config(ckpt, Path(args.checkpoint))

    D              = A["state_dim"]
    CTX            = A["context_dim"]
    # context_raw_dim читаем из весов — он не всегда хранится в конфиге
    CTX_RAW        = state["state_builder.context_builder.proj.0.weight"].shape[1]
    F              = A["n_metrics"]
    d_ssm          = A["d_ssm"]
    n_components   = A["n_components"]
    n_confounders  = A["n_confounders"]
    confounder_dim = A["confounder_dim"]

    logger.info(
        f"Архитектура: D={D}, CTX={CTX}, F={F}, d_ssm={d_ssm}, "
        f"n_components={n_components}, n_confounders={n_confounders}, "
        f"confounder_dim={confounder_dim}"
    )

    model = CAIRNModel(
        state_builder=StateBuilder(
            n_metrics=F,
            log_vocab_size=A.get("log_vocab_size", 300),
            state_dim=D,
            context_dim=CTX,
            d_met=A.get("d_met", d_ssm * 2),
            d_log=A.get("d_log", d_ssm),
            d_tr=A.get("d_tr",  d_ssm),
            d_ssm=d_ssm,
            d_brk=A.get("d_brk", d_ssm),
            ssm_state_dim=A.get("ssm_state_dim", d_ssm * 2),
            window=A.get("window", 15),
            context_raw_dim=CTX_RAW,
        ),
        gmm=ConditionalGMM(
            state_dim=D,
            context_dim=CTX,
            n_components=n_components,
        ),
        vgae=ConfoundedVGAE(
            state_dim=D,
            n_confounders=n_confounders,
            confounder_dim=confounder_dim,
        ),
        cf_module=CounterfactualModule(
            state_dim=D,
            n_conv_layers=A.get("n_conv_layers", 1),
        ),
    )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Отсутствующие ключи ({len(missing)}): {missing[:3]}...")
    if unexpected:
        logger.warning(f"Лишние ключи ({len(unexpected)}): {unexpected[:3]}...")
    if not missing and not unexpected:
        logger.info("Чекпоинт загружен без расхождений ✓")
    else:
        logger.info("Чекпоинт загружен")

    # ── Оценка ───────────────────────────────────────────────────────────
    from cairn.reasoning import CascadeFunnel
    from cairn.training.trainer import compute_metrics

    model.eval()
    anom_subset = dataset.anomaly_subset()
    if len(anom_subset) == 0:
        logger.error("Нет аномальных инцидентов в датасете.")
        return

    all_results: list[dict] = []
    with torch.no_grad():
        for i in range(len(anom_subset)):
            incident = anom_subset[i]
            outputs  = model(incident, hypergraph)
            H, C     = outputs["H"], outputs["C"]
            nll      = model.gmm.nll(H, C)

            adj      = hypergraph.adjacency_matrix()
            adj_norm = adj / adj.sum(1, keepdim=True).clamp(min=1)

            funnel = CascadeFunnel(
                l0_top_k=len(hypergraph.instance_names),
                l1_top_k=5,
                l2_top_k=5,
            )
            ranked = funnel.run(
                nll, H, adj_norm,
                model.cf_module, model.gmm, C, hypergraph,
            )
            all_results.append({
                "ranked":     ranked,
                "root_cause": incident.root_cause,
            })

    # ── Метрики ──────────────────────────────────────────────────────────
    # ── Основные метрики ─────────────────────────────────────────────────
    all_metrics = [
        compute_extended_metrics(res["ranked"], res["root_cause"])
        for res in all_results
    ]
    metrics = {
        key: sum(m[key] for m in all_metrics) / len(all_metrics)
        for key in all_metrics[0]
    }

    # ── Вывод основных метрик ─────────────────────────────────────────────
    print("\n" + "═" * 52)
    print(f"  CAIRN — Метрики качества локализации первопричин")
    print("═" * 52)
    print(f"  Чекпоинт: {args.checkpoint}")
    print(f"  Данные:   {args.data_dir}  ({len(anom_subset)} инцидентов)")
    print("─" * 52)
    groups = [
        ("Accuracy@k",  ["AC@1",   "AC@3",   "AC@5"]),
        ("NDCG@k",      ["NDCG@1", "NDCG@3", "NDCG@5"]),
        ("Ranking",     ["MRR"]),
    ]
    for group_name, keys in groups:
        print(f"\n  {group_name}:")
        for key in keys:
            val  = metrics.get(key, 0.0)
            bar  = "█" * int(val * 20)
            print(f"    {key:<8}  {val:.4f}  {bar}")
    print("─" * 52)

    # ── Ablation Study ────────────────────────────────────────────────────
    if args.ablation:
        print("\n" + "═" * 52)
        print("  ABLATION STUDY — Вклад компонентов")
        print("═" * 52)
        print(f"  {'Конфигурация':<30} {'AC@1':>6} {'NDCG@3':>7} {'MRR':>6}")
        print("─" * 52)

        ablation_configs = [
            ("CAIRN (полная)", True,  True),
            ("- graph_verifier", False, True),
            ("- cf_module",    True,  False),
        ]

        for cfg_name, use_graph, use_cf in ablation_configs:
            abl_results = []
            with torch.no_grad():
                for i in range(len(anom_subset)):
                    incident = anom_subset[i]
                    outputs  = model(incident, hypergraph)
                    H, C     = outputs["H"], outputs["C"]
                    nll      = model.gmm.nll(H, C)
                    adj      = hypergraph.adjacency_matrix()
                    adj_norm = adj / adj.sum(1, keepdim=True).clamp(min=1)

                    funnel = CascadeFunnel(
                        l0_top_k=len(hypergraph.instance_names),
                        l1_top_k=5, l2_top_k=5,
                    )
                    ranked_abl = funnel.run(
                        nll, H, adj_norm,
                        model.cf_module if use_cf else None,
                        model.gmm, C, hypergraph,
                    )

                    # Топологическая корректировка
                    if use_graph:
                        import numpy as _np
                        ce_scores  = dict(ranked_abl)
                        called_by: dict[int, int] = {}
                        callee_map: dict[int, list] = {}
                        for edge in hypergraph.edges:
                            if edge.edge_type == "call" and len(edge.members) >= 2:
                                s, d = edge.members[0], edge.members[1]
                                callee_map.setdefault(s, []).append(d)
                                called_by[d] = called_by.get(d, 0) + 1
                        adjusted = {}
                        for idx, score in ce_scores.items():
                            cs  = [ce_scores.get(c, 0.0) for c in callee_map.get(idx, [])
                                   if c in ce_scores]
                            cascade = float(_np.mean(cs)) if cs else 0.0
                            adjusted[idx] = score / (1.0 + cascade) / (
                                1.0 + called_by.get(idx, 0) * 0.5)
                        ranked_abl = sorted(adjusted.items(),
                                            key=lambda x: x[1], reverse=True)

                    abl_results.append({
                        "ranked": ranked_abl,
                        "root_cause": incident.root_cause,
                    })

            abl_metrics = [
                compute_extended_metrics(r["ranked"], r["root_cause"])
                for r in abl_results
            ]
            abl_avg = {k: sum(m[k] for m in abl_metrics) / len(abl_metrics)
                       for k in abl_metrics[0]}
            print(f"  {cfg_name:<30} "
                  f"{abl_avg['AC@1']:>6.3f} "
                  f"{abl_avg['NDCG@3']:>7.3f} "
                  f"{abl_avg['MRR']:>6.3f}")

        print("─" * 52)

    # ── Baseline сравнение ────────────────────────────────────────────────
    if args.baseline:
        import random, math
        n_nodes = len(hypergraph.instance_names)
        print("\n" + "═" * 52)
        print("  BASELINE — Сравнение с базовыми методами")
        print("═" * 52)
        print(f"  {'Метод':<25} {'AC@1':>6} {'NDCG@3':>7} {'MRR':>6}")
        print("─" * 52)

        # CAIRN
        print(f"  {'CAIRN (полная)':<25} "
              f"{metrics['AC@1']:>6.3f} "
              f"{metrics['NDCG@3']:>7.3f} "
              f"{metrics['MRR']:>6.3f}")

        # NLL-only (без CascadeFunnel и топологии)
        nll_results = []
        with torch.no_grad():
            for i in range(len(anom_subset)):
                incident = anom_subset[i]
                outputs  = model(incident, hypergraph)
                H, C     = outputs["H"], outputs["C"]
                nll_vals = model.gmm.nll(H, C)
                ranked_nll = sorted(
                    [(j, float(nll_vals[j])) for j in range(len(nll_vals))],
                    key=lambda x: x[1], reverse=True,
                )
                nll_results.append({
                    "ranked": ranked_nll,
                    "root_cause": incident.root_cause,
                })
        nll_m = [compute_extended_metrics(r["ranked"], r["root_cause"])
                 for r in nll_results]
        nll_avg = {k: sum(m[k] for m in nll_m) / len(nll_m) for k in nll_m[0]}
        print(f"  {'NLL-only (без графа)':<25} "
              f"{nll_avg['AC@1']:>6.3f} "
              f"{nll_avg['NDCG@3']:>7.3f} "
              f"{nll_avg['MRR']:>6.3f}")

        # Random baseline
        random.seed(42)
        rand_results = []
        for i in range(len(anom_subset)):
            incident = anom_subset[i]
            n = len(hypergraph.instance_names)
            perm = list(range(n))
            random.shuffle(perm)
            rand_results.append({
                "ranked": [(j, 0.0) for j in perm],
                "root_cause": incident.root_cause,
            })
        rand_m = [compute_extended_metrics(r["ranked"], r["root_cause"])
                  for r in rand_results]
        rand_avg = {k: sum(m[k] for m in rand_m) / len(rand_m) for k in rand_m[0]}
        print(f"  {'Random baseline':<25} "
              f"{rand_avg['AC@1']:>6.3f} "
              f"{rand_avg['NDCG@3']:>7.3f} "
              f"{rand_avg['MRR']:>6.3f}")
        print("─" * 52)

    print(f"  Аномальных инцидентов: {len(anom_subset)}")
    print(f"  Данные: {args.data_dir}")

    ac1 = metrics.get("AC@1", 0.0)
    if ac1 >= 0.8:
        verdict = "[OK] Отлично (AC@1 >= 80%)"
    elif ac1 >= 0.6:
        verdict = "[~~] Удовлетворительно (AC@1 >= 60%)"
    else:
        verdict = "[!!] Требует дообучения (AC@1 < 60%)"
    print(f"\n{verdict}\n")


if __name__ == "__main__":
    main()
