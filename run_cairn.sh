#!/bin/bash
# CAIRN – запуск GUI (Linux/macOS)
cd "$(dirname "$0")"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

python -m cairn "$@"
