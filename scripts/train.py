"""Скрипт полного цикла обучения CAIRN.

Использование:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/demo.yaml --epochs 1 --no-save
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="CAIRN Training")
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="Возобновить с чекпоинта")
    parser.add_argument("--pretrain-epochs",  type=int, default=None)
    parser.add_argument("--main-epochs",      type=int, default=None)
    parser.add_argument("--finetune-epochs",  type=int, default=None)
    parser.add_argument("--no-save",    action="store_true", help="Не сохранять чекпоинты")
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    from cairn.config import load_config
    from cairn.utils.logging import setup_logging

    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level)

    from loguru import logger
    logger.info(f"Конфигурация: {args.config}")

    # --- Данные ---
    from cairn.training.data_loader import create_demo_dataset

    data_dir = Path("data/sample")
    if not (data_dir / "metrics.csv").exists():
        logger.info("Демо-данные не найдены, генерируем...")
        import subprocess
        subprocess.run([sys.executable, "scripts/generate_demo_data.py"], check=True)

    logger.info("Загрузка датасета...")
    dataset = create_demo_dataset(
        sample_dir=data_dir,
        window_size=min(60, cfg.model.breakpoint_window * 2),
        stride=10,
    )
    logger.info(dataset.summary())

    if len(dataset) == 0:
        logger.error("Датасет пуст – проверьте данные в data/sample/")
        sys.exit(1)

    # --- Топология ---
    from cairn.connectors.csv_file import YAMLTopologyConnector
    from cairn.perception import HypergraphBuilder

    topo = YAMLTopologyConnector(data_dir / "topology.yaml").fetch()
    hypergraph = HypergraphBuilder.from_topology_data(topo)
    N_inst = hypergraph.n_nodes
    logger.info(f"Топология: {N_inst} экземпляров, {len(hypergraph.edges)} гиперрёбер")

    # --- Модель ---
    import torch
    from cairn.perception import StateBuilder
    from cairn.reasoning import ConditionalGMM, ConfoundedVGAE, CounterfactualModule
    from cairn.training import CAIRNModel, CAIRNLoss, CAIRNTrainer, TrainerConfig, LossWeights

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

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Параметров модели: {n_params:,}")

    # --- Функция потерь ---
    tc = cfg.training
    loss_fn = CAIRNLoss(
        weights=LossWeights(
            lambda_pe=tc.loss_weights.lambda_pe,
            lambda_um=tc.loss_weights.lambda_um,
            lambda_vak=tc.loss_weights.lambda_vak,
            lambda_nez=tc.loss_weights.lambda_nez,
            lambda_kr=tc.loss_weights.lambda_kr,
            lambda_reb=tc.loss_weights.lambda_reb,
        ),
        margin=tc.margin,
        tcd_margin=tc.tcd_margin,
        beta_kl=tc.beta_kl,
        beta_kl_z=tc.beta_kl_z,
        cov_reg=tc.cov_reg,
        adaptive=True,
    )

    # --- Тренер ---
    trainer_cfg = TrainerConfig(
        pretrain_epochs  = args.pretrain_epochs  or tc.pretrain_epochs,
        main_epochs      = args.main_epochs      or tc.main_epochs,
        finetune_epochs  = args.finetune_epochs  or tc.finetune_epochs,
        freeze_epochs    = tc.freeze_epochs,
        lr               = tc.lr,
        batch_size       = tc.batch_size,
        device           = args.device,
        checkpoint_dir   = "checkpoints" if not args.no_save else "/tmp/cairn_ckpt",
        save_every       = 10,
        patience         = 10,
    )

    trainer = CAIRNTrainer(model, loss_fn, hypergraph, trainer_cfg)

    # Возобновление с чекпоинта
    if args.checkpoint and Path(args.checkpoint).exists():
        logger.info(f"Загрузка чекпоинта: {args.checkpoint}")
        trainer.load(args.checkpoint)

    # --- Обучение ---
    t0 = time.time()
    history = trainer.train(dataset)
    elapsed = time.time() - t0
    logger.success(f"Обучение завершено за {elapsed:.1f}с")

    # --- Оценка ---
    eval_metrics = trainer.evaluate(dataset)
    logger.info("Метрики на обучающем наборе:")
    for k, v in eval_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # --- Сохранение финальной модели ---
    if not args.no_save:
        out = Path("checkpoints") / "final.pt"
        trainer.save(out)
        logger.success(f"Модель сохранена: {out}")


if __name__ == "__main__":
    main()
