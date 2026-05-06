# CAIRN — Causal Attentive Intervention Reasoning Network

> Программная система прогнозирования и устранения причин возникновения нештатных событий в микросервисных системах.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## О проекте

**CAIRN** (от англ. *cairn* — каменная пирамида-ориентир) — система поиска первопричин сбоев в микросервисных системах. Подобно ориентиру на горной тропе, система указывает путь от наблюдаемых симптомов к первопричине сбоя.

Архитектура решает две фундаментальные проблемы существующих подходов:

| Проблема | Решение в CAIRN |
|---|---|
| **Непрозрачность вывода** | Трёхслойная система интерпретируемости: проверяемая цепочка доказательств, медиационная диагностика, логическая верификация объяснений |
| **Смешение корреляции с причинностью** | Дифференцируемое контрфактическое вмешательство внутри обучения + обнаружение скрытых общих факторов + формальная верификация причинного графа по 5 аксиомам |

---

## Архитектура

```
Данные мониторинга
(метрики / журналы / трассировки)
        │
        ▼
┌─────────────────────┐
│   Фаза восприятия   │  Двухветвевой кодировщик метрик (SSM + разрыв),
│  (Perception Phase) │  кодировщик журналов, трассировок,
└────────┬────────────┘  построение причинного гиперграфа
         │
         ▼
┌─────────────────────────┐
│   Фаза рассуждения      │  Условная GMM нормального состояния,
│  (Reasoning Phase)      │  VGAE с латентными конфаундерами,
└────────┬────────────────┘  контрфактический модуль, каскадная воронка,
         │                   верификатор причинного графа (5 аксиом)
         ▼
┌─────────────────────────┐
│   Фаза объяснения       │  Проверяемая цепочка доказательств,
│  (Explanation Phase)    │  медиационная диагностика, генерация текста,
└────────┬────────────────┘  логическая верификация (ALP + контр-абдукция)
         │
         ▼
   Отчёт + GUI (PySide6)
```

---

## Установка

```bash
git clone https://github.com/<your-org>/cairn.git
cd cairn

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Базовые зависимости
pip install -e .

# Опциональные коннекторы (Prometheus, Elasticsearch)
pip install -e ".[connectors]"

# Зависимости для разработки (линтеры, тесты)
pip install -e ".[dev]"
```

> **Примечание.** PyTorch Geometric требует совместимого колеса с вашей версией CUDA.  
> Установите вручную: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

---

## Быстрый старт

```bash
# Демонстрационный запуск на синтетических данных
python scripts/demo.py --config configs/demo.yaml

# Обучение на собственных данных
python scripts/train.py --config configs/default.yaml

# Оценка качества
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt

# Запуск GUI
python -m cairn.gui.main_window
```

---

## Структура проекта

```
cairn/
├── src/cairn/
│   ├── config.py              # Pydantic-модели конфигурации
│   ├── perception/            # Кодировщики метрик, журналов, трассировок;
│   │                          # построение гиперграфа
│   ├── reasoning/             # Условная GMM, VGAE+конфаундеры,
│   │                          # контрфактический модуль, воронка, верификатор
│   ├── explanation/           # Цепочка доказательств, медиация,
│   │                          # генерация текста, ALP-верификатор
│   ├── training/              # Функция потерь, тренер, загрузчик данных
│   ├── connectors/            # CSV, Prometheus, Elasticsearch, Jaeger
│   └── gui/                   # PySide6 интерфейс
├── tests/                     # Unit-тесты по модулям
├── data/sample/               # Демо-данные
├── configs/                   # YAML-конфигурации
├── scripts/                   # train.py / evaluate.py / demo.py
└── docs/architecture.md       # Детальное описание архитектуры
```

---

## Ключевые гиперпараметры

| Параметр | Значение по умолчанию | Обоснование |
|---|---|---|
| `state_dim` | 128 | Оптимум по CHASE [5] |
| `gmm_components` | 5 | Охватывает типичные режимы нагрузки |
| `latent_confounders` | 3 | Подбирается ablation-тестом |
| `attention_heads` | 8 | Стандарт для трансформеров |
| `hypergraph_layers` | 1 | Увеличение ухудшает качество [5] |

Полная таблица — в `configs/default.yaml`.

---

## Наборы данных для валидации

- **GAIA** — публичный benchmark микросервисных сбоев
- **TrainTicket** — система продажи билетов с искусственно внесёнными сбоями
- **RCAEval-RE2-OB** — мультимодальный benchmark с метками первопричин

Размещайте в `data/benchmarks/` (папка добавлена в `.gitignore`).

---

## Воспроизводимость

Все эксперименты запускаются с фиксированным сидом (`seed: 42` в конфигурации). Обученные веса сохраняются в `checkpoints/`.

---

## Цитирование

Если вы используете CAIRN в исследованиях, пожалуйста, ссылайтесь на:

```
Ткач Д.М. CAIRN: Causal Attentive Intervention Reasoning Network.
Дипломная работа, СПбГУТ, 2026.
```

---

## Лицензия

MIT — подробнее в файле [LICENSE](LICENSE).



python scripts/generate_demo_data.py --out data/sample --seed 42
python scripts/pretrain_demo.py --epochs 10
python scripts/diagnose.py --scenario 3 --model data/sample/demo_model.pt
python scripts/evaluate.py --checkpoint data/sample/demo_model.pt --data-dir data/sample/scenario_3