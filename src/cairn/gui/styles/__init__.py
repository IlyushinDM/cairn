"""Темы CAIRN GUI."""
from pathlib import Path

STYLES_DIR = Path(__file__).parent

def load_theme(name: str = "dark") -> str:
    path = STYLES_DIR / f"{name}_theme.qss"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""
