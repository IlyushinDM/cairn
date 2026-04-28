"""Настройка логирования через loguru."""

from __future__ import annotations

import sys
from loguru import logger


def setup_logging(
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
    log_file: str | None = "logs/cairn.log",
) -> None:
    """Конфигурирует loguru для консоли и (опционально) файла.

    Параметры
    ----------
    level : str
        Минимальный уровень логирования.
    rotation : str
        Условие ротации файла (например, "10 MB" или "1 day").
    retention : str
        Срок хранения логов.
    log_file : str | None
        Путь к файлу. Если None — только консоль.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan> — <level>{message}</level>",
        colorize=True,
    )
    if log_file:
        logger.add(
            log_file,
            level=level,
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} — {message}",
        )
