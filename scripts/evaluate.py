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
    all_metrics = [
        compute_metrics(res["ranked"], res["root_cause"])
        for res in all_results
    ]
    metrics = {
        key: sum(m[key] for m in all_metrics) / len(all_metrics)
        for key in all_metrics[0]
    }

    # ── Вывод ────────────────────────────────────────────────────────────
    print("\n" + "─" * 40)
    print(f"{'Метрика':<12} {'Значение':>10}")
    print("─" * 40)
    for name, val in metrics.items():
        print(f"{name:<12} {val:>10.4f}")
    print("─" * 40)
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