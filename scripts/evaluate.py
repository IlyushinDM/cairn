"""Точка входа для оценки качества CAIRN.

Использование:
    python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cairn.config import load_config
from cairn.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="CAIRN Evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="gaia", choices=["gaia", "trainticket", "rcaeval"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level)

    from loguru import logger
    logger.info(f"Оценка на датасете: {args.dataset}")
    logger.info(f"Чекпоинт: {args.checkpoint}")
    logger.info("Реализация оценщика будет добавлена на следующем этапе.")


if __name__ == "__main__":
    main()
