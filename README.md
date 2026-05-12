# CAIRN – Causal AI Root cause Identification for Networks

> Система автоматической локализации первопричин сбоев в микросервисных приложениях.  
> Объединяет причинно-следственный вывод, гиперграфовые нейронные сети и интерпретируемые объяснения.

---

## Содержание

- [Описание](#описание)
- [Быстрый старт](#быстрый-старт)
- [Подключение Online Boutique](#подключение-online-boutique)
- [Архитектура](#архитектура)
- [Метрики качества](#метрики-качества)
- [Структура проекта](#структура-проекта)
- [Разработка и тесты](#разработка-и-тесты)

---

## Описание

CAIRN решает задачу **Root Cause Analysis (RCA)** – автоматического определения сервиса, который стал причиной деградации в распределённой системе. В отличие от традиционных подходов, система:

- использует **причинно-следственный вывод** (Conditional GMM + VGAE) вместо корреляционного анализа;
- строит **гиперграф** микросервисов с учётом топологии вызовов;
- предоставляет **интерпретируемые объяснения** с доказательной цепочкой и контрфактическим анализом воздействия;
- работает в **реактивном и проактивном** режимах – обнаруживает аномалии автоматически.

### Результаты на Online Boutique (Google Microservices Demo)

| Сценарий | Инжектировано | Ранг CAIRN | Score |
|---|---|---|---|
| CPU stress | `redis-cart` | **#1** | 2.418 |
| Service pause | `cartservice` | **#1** | 10.675 |

---

## Быстрый старт

### Требования

- Python 3.11+
- Docker Desktop (для live-режима)
- 4 GB RAM

### Установка

```bash
git clone https://github.com/<your-org>/cairn-v2.git
cd cairn-v2

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### Демо-режим (без Docker)

```bash
python scripts/run_gui.py
```

В открывшемся окне выберите **Файл → Демо-режим** и запустите любой из 5 сценариев. Анализ займёт несколько секунд.

### Оценка качества на демо-данных

```bash
# Основные метрики
python scripts/evaluate.py \
  --checkpoint data/sample/demo_model.pt \
  --data-dir   data/sample/scenario_5

# С ablation study и сравнением baseline
python scripts/evaluate.py \
  --checkpoint data/sample/demo_model.pt \
  --data-dir   data/sample/scenario_5 \
  --ablation --baseline
```

---

## Подключение Online Boutique

### 1. Запустить стек

```bash
docker compose -f docker/docker-compose.yml up -d
```

Сервисы будут доступны:
- Boutique UI: http://localhost:8081
- Prometheus:  http://localhost:9090
- cAdvisor:    http://localhost:8080

### 2. Подключить в GUI

1. Запустить `python scripts/run_gui.py`
2. Нажать иконку **Connect** на боковой панели
3. Выбрать `Online Boutique` → **Проверить** → **Подключить**
4. Подождать ~5 минут для установки baseline (статусбар покажет прогресс)

### 3. Инъекция сбоя

```bash
# CPU stress в redis (в отдельном терминале)
python scripts/inject_fault.py \
  --type cpu --target redis-cart \
  --duration 60 --warmup 30 --live
```

CAIRN обнаружит аномалию автоматически и переключится на вкладку **Результаты**.

---

## Архитектура

```
Входные данные
  ├── Метрики (docker stats / Prometheus)
  ├── Журналы контейнеров (docker logs)
  └── Latency-трассировки (Locust loadgenerator)
         │
         ▼
  ┌─────────────────────────────────────────┐
  │           Perception Layer              │
  │  DualBranchMetricEncoder                │
  │  ├── SSM Branch (временные зависимости) │
  │  └── Breakpoint Branch (смена режима)   │
  │  LogEncoder · TraceEncoder              │
  │  StateBuilder → H (state), C (context)  │
  └──────────────────┬──────────────────────┘
                     │
         ▼
  ┌─────────────────────────────────────────┐
  │           Reasoning Layer               │
  │  ConditionalGMM  → NLL (аномальность)  │
  │  ConfoundedVGAE  → скрытые факторы     │
  │  CounterfactualModule → вмешательство  │
  │  CascadeFunnel   → ранжирование        │
  │  GraphVerifier   → топол. корректировка│
  └──────────────────┬──────────────────────┘
                     │
         ▼
  ┌─────────────────────────────────────────┐
  │           Explanation Layer             │
  │  EvidenceChainBuilder → цепочка         │
  │  ALPVerifier → логическая верификация  │
  │  TemplateTextGenerator → объяснение    │
  └─────────────────────────────────────────┘
```

### Ключевые компоненты

| Компонент | Назначение |
|---|---|
| `StateBuilder` | Многомодальный энкодер: метрики + логи + трассировки → state вектор |
| `ConditionalGMM` | Моделирует нормальное поведение; NLL = степень аномальности |
| `ConfoundedVGAE` | Выявляет скрытые факторы и причинно-следственные связи |
| `CounterfactualModule` | «Что было бы, если бы сервис работал нормально?» |
| `CascadeFunnel` | Трёхуровневое ранжирование кандидатов |
| `GraphVerifier` | Топологическая корректировка: учитывает каскадные эффекты |
| `AnomalyMonitor` | Фоновый мониторинг с адаптивным порогом (baseline + 2σ) |

---

## Метрики качества

Оценка на `scenario_5` (unseen test set, 4 аномальных инцидента):

| Метрика | CAIRN | NLL-only | Random |
|---|---|---|---|
| **AC@1** | **1.000** | 1.000 | 0.250 |
| **AC@3** | **1.000** | 1.000 | 0.500 |
| **NDCG@3** | **1.000** | 1.000 | 0.565 |
| **MRR** | **1.000** | 1.000 | 0.550 |

Вклад `graph_verifier` на реальных данных (Online Boutique):

| Конфигурация | redis ранг | Score |
|---|---|---|
| CAIRN (полная) | **#1** | 2.418 |
| Без `graph_verifier` | #3 | 1.082 |

> Топологическая корректировка повышает точность ранжирования на +78% на реальных данных.

---

## Структура проекта

```
cairn-v2/
├── src/cairn/
│   ├── connectors/          # Источники данных
│   │   ├── live_connector.py       # Live-система (Docker/Prometheus)
│   │   ├── csv_file.py             # CSV/YAML файлы
│   │   ├── docker_log_connector.py # Журналы контейнеров
│   │   └── latency_trace_connector.py # Latency из loadgenerator
│   ├── perception/          # Слой восприятия
│   │   ├── state_builder.py        # Многомодальный энкодер
│   │   ├── metric_encoder.py       # SSM + Breakpoint ветви
│   │   └── hypergraph.py           # Построение гиперграфа
│   ├── reasoning/           # Слой рассуждений
│   │   ├── gmm.py                  # Conditional GMM
│   │   ├── vgae.py                 # Confounded VGAE
│   │   ├── counterfactual.py       # CF-вмешательство
│   │   └── funnel.py               # Cascade Funnel
│   ├── explanation/         # Слой объяснений
│   │   ├── evidence_chain.py       # Цепочка доказательств
│   │   ├── alp_verifier.py         # Логическая верификация
│   │   └── text_generator.py       # Генерация объяснений
│   ├── evaluation/          # Метрики качества
│   │   └── metrics.py              # AC@k, NDCG@k, MRR
│   ├── training/            # Обучение
│   │   ├── trainer.py
│   │   └── data_loader.py
│   └── gui/                 # Графический интерфейс
│       ├── main_window.py
│       ├── anomaly_monitor.py      # Фоновый мониторинг
│       ├── controller.py
│       ├── icons/                  # SVG-иконки
│       ├── styles/                 # QSS-темы
│       └── widgets/
├── scripts/
│   ├── run_gui.py           # Запуск GUI
│   ├── evaluate.py          # Оценка качества
│   ├── inject_fault.py      # Инъекция сбоев
│   └── train.py             # Обучение модели
├── tests/                   # pytest (59 тестов)
├── data/sample/             # Демо-данные и чекпоинт
├── configs/connectors/      # Конфиги подключений
├── docker/                  # Online Boutique стек
└── docs/                    # Документация
```

---

## Разработка и тесты

### Запуск тестов

```bash
# Быстрые тесты (без GPU и slow)
pytest tests/ -m "not slow and not gpu" -q

# С отчётом покрытия
pytest tests/ -m "not slow and not gpu" \
  --cov=src/cairn --cov-report=term-missing

# Все тесты включая интеграционные
pytest tests/ -v
```

### Линтинг

```bash
flake8 src/ scripts/
isort src/ scripts/ --check-only
```

### Структура веток

| Ветка | Содержание |
|---|---|
| `main` | Стабильная версия |
| `feat/multimodal-observability` | Логи, трассировки, активные модули |
| `feat/autonomous-monitoring` | Автономный мониторинг аномалий |
| `chore/testing-and-ci` | Тесты и CI |
| `docs/readme-and-finalization` | Документация |

---

## Требования

Основные зависимости:

```
torch>=2.0.0
PySide6>=6.5.0
numpy>=1.24.0
matplotlib>=3.7.0
networkx>=3.0
pyyaml>=6.0
loguru>=0.7.0
```

Полный список: `requirements.txt`

---

*Дипломная работа. СПбГУТ, 2026.*
