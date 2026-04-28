"""Точка входа для обучения CAIRN.

Использование:
    python scripts/train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cairn.config import load_config
from cairn.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="CAIRN Training")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="Возобновить с чекпоинта")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level)

    from loguru import logger
    logger.info(f"Конфигурация загружена из {args.config}")
    logger.info(
        f"Параметры модели: state_dim={cfg.model.state_dim}, "
        f"gmm_components={cfg.model.gmm_components}, "
        f"latent_confounders={cfg.model.latent_confounders}"
    )
    logger.info("Реализация тренера (cairn.training.trainer) будет добавлена на следующем этапе.")


if __name__ == "__main__":
    main()
