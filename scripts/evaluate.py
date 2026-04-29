"""Скрипт оценки обученной модели CAIRN.

Использование:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="CAIRN Evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",  default="configs/default.yaml")
    parser.add_argument("--dataset", default="demo", choices=["demo", "gaia", "trainticket"])
    parser.add_argument("--data-dir", default="data/sample")
    parser.add_argument("--device",  default="cpu")
    args = parser.parse_args()

    from cairn.config import load_config
    from cairn.utils.logging import setup_logging

    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level)

    from loguru import logger
    logger.info(f"Чекпоинт: {args.checkpoint}")
    logger.info(f"Датасет: {args.dataset}")

    # --- Данные ---
    from cairn.training.data_loader import create_demo_dataset

    data_dir = Path(args.data_dir)
    dataset  = create_demo_dataset(data_dir)
    logger.info(dataset.summary())

    # --- Топология ---
    from cairn.connectors.csv_file import YAMLTopologyConnector
    from cairn.perception import HypergraphBuilder

    topo = YAMLTopologyConnector(data_dir / "topology.yaml").fetch()
    hypergraph = HypergraphBuilder.from_topology_data(topo)

    # --- Модель (восстанавливаем архитектуру из конфига) ---
    import torch
    from cairn.perception import StateBuilder
    from cairn.reasoning import ConditionalGMM, ConfoundedVGAE, CounterfactualModule
    from cairn.training import CAIRNModel, CAIRNLoss, CAIRNTrainer, TrainerConfig

    mc = cfg.model
    model = CAIRNModel(
        state_builder=StateBuilder(
            n_metrics=dataset[0].metric_data.shape[2],
            log_vocab_size=300,
            state_dim=mc.state_dim,
            context_dim=mc.context_dim,
            d_met=mc.metric_dim,
            d_log=mc.log_dim,
            d_tr=mc.trace_dim,
            d_ssm=mc.metric_dim // 2,
            d_brk=mc.metric_dim // 2,
            ssm_state_dim=mc.state_dim // 2,
            window=mc.breakpoint_window,
        ),
        gmm=ConditionalGMM(
            state_dim=mc.state_dim,
            context_dim=mc.context_dim,
            n_components=mc.gmm_components,
        ),
        vgae=ConfoundedVGAE(
            state_dim=mc.state_dim,
            n_confounders=mc.latent_confounders,
            confounder_dim=mc.confounder_dim,
        ),
        cf_module=CounterfactualModule(
            state_dim=mc.state_dim,
            n_conv_layers=mc.hypergraph_layers,
        ),
    )

    trainer = CAIRNTrainer(
        model, CAIRNLoss(), hypergraph,
        TrainerConfig(device=args.device),
    )
    trainer.load(args.checkpoint)
    logger.info("Чекпоинт загружен")

    # --- Оценка ---
    metrics = trainer.evaluate(dataset)

    # --- Таблица результатов ---
    print("\n" + "─" * 40)
    print(f"{'Метрика':<12} {'Значение':>10}")
    print("─" * 40)
    for name, val in metrics.items():
        print(f"{name:<12} {val:>10.4f}")
    print("─" * 40)

    # Итоговая оценка
    ac1 = metrics.get("AC@1", 0.0)
    if ac1 >= 0.8:
        verdict = "✅ Отлично (AC@1 ≥ 80%)"
    elif ac1 >= 0.6:
        verdict = "⚠  Удовлетворительно (AC@1 ≥ 60%)"
    else:
        verdict = "❌ Требует дообучения (AC@1 < 60%)"
    print(f"\n{verdict}\n")


if __name__ == "__main__":
    main()
