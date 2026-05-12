"""Генератор синтетических демо-данных для CAIRN.

Создаёт три сценария в data/sample/:
  scenario_1 – CPU Exhaustion    (первопричина: order-service-1)
  scenario_2 – Memory Leak       (первопричина: cache-service-1)
  scenario_3 – Network Delay     (первопричина: frontend-1)

Каждый сценарий содержит:
  metrics.csv   – временные ряды метрик (300 шагов: 200 норм + 100 аном)
  logs.txt      – лог-сообщения
  traces.json   – трассировки вызовов
  topology.yaml – топология микросервисов
  labels.json   – разметка первопричины и временных периодов

Использование:
    python scripts/generate_demo_data.py
    python scripts/generate_demo_data.py --out data/sample --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import yaml


# ── Константы ────────────────────────────────────────────────────────────────

BASE_TS      = 1_700_000_000.0   # начало временного ряда (unix timestamp)
N_NORMAL     = 200               # шагов нормального периода
N_ANOMALY    = 100               # шагов аномального периода
STEP         = 1.0               # секунд на шаг

INSTANCES = [
    "cache-service-1",
    "database-1",
    "frontend-1",
    "order-service-1",
    "payment-service-1",
]

METRICS = ["cpu", "memory", "latency_ms", "rps"]

# Базовые (нормальные) значения метрик для каждого сервиса
# Подобраны по фактическим значениям из diagnose.py
BASELINE = {
    "cache-service-1":   {"cpu": 0.10, "memory": 0.70, "latency_ms":  2.0, "rps": 500.0},
    "database-1":        {"cpu": 0.25, "memory": 0.60, "latency_ms":  8.0, "rps": 300.0},
    "frontend-1":        {"cpu": 0.15, "memory": 0.40, "latency_ms": 12.0, "rps": 200.0},
    "order-service-1":   {"cpu": 0.30, "memory": 0.55, "latency_ms": 25.0, "rps": 180.0},
    "payment-service-1": {"cpu": 0.20, "memory": 0.45, "latency_ms": 18.0, "rps": 160.0},
}

# Шум (std) для каждой метрики как доля от базового значения
NOISE_FRAC = {"cpu": 0.05, "memory": 0.02, "latency_ms": 0.10, "rps": 0.03}

# Топология: рёбра call-графа (направленные: source → target)
# database-1 является центральным хабом
CALL_EDGES = [
    ("cache-service-1",   "database-1"),
    ("frontend-1",        "database-1"),
    ("order-service-1",   "database-1"),
    ("payment-service-1", "database-1"),
]

# ── Параметры аномалий ────────────────────────────────────────────────────────

SCENARIOS = {
    "scenario_1": {
        "name":       "CPU Exhaustion",
        "root_cause": {"instance": "order-service-1", "type": "cpu_exhaustion"},
        # 1.1: Усиленные множители – root выделяется по ВСЕМ метрикам
        # Цель: ratio root/neighbor > 5x, чтобы GMM дал положительный CE
        "anomaly": {
            "order-service-1": {
                "cpu":        {"mult": 8.0},   # +700% (было +300%)
                "memory":     {"mult": 2.5},   # +150% (было +40%)
                "latency_ms": {"mult": 10.0},  # +900% (было +200%)
                "rps":        {"mult": 0.35},  # -65%  (было -40%)
            },
            # Каскад: сильно ослаблен чтобы root доминировал
            "database-1": {
                "cpu":        {"mult": 1.15},  # +15%  (было +30%)
                "latency_ms": {"mult": 1.3},   # +30%  (было +50%)
            },
            "payment-service-1": {
                "latency_ms": {"mult": 1.1},   # +10%  (было +20%)
            },
        },
    },
    "scenario_2": {
        "name":       "Memory Leak",
        "root_cause": {"instance": "cache-service-1", "type": "memory_pressure"},
        # 1.1: Усиленные множители для memory_pressure
        "anomaly": {
            "cache-service-1": {
                "cpu":        {"mult": 3.5},   # +250% (было +80%)
                "memory":     {"mult": 8.0},   # +700% (было +250%)
                "latency_ms": {"mult": 12.0},  # +1100% (было +150%)
                "rps":        {"mult": 0.30},  # -70%   (было -30%)
            },
            # Каскад: минимальный эффект
            "database-1": {
                "memory":     {"mult": 1.1},   # +10%  (было +20%)
                "latency_ms": {"mult": 1.2},   # +20%  (было +30%)
            },
        },
    },
    "scenario_3": {
        "name":       "Network Delay",
        "root_cause": {"instance": "frontend-1", "type": "latency_spike"},
        "anomaly": {
            "frontend-1": {
                "cpu":        {"mult": 3.5},
                "memory":     {"mult": 2.0},
                "latency_ms": {"mult": 20.0},
                "rps":        {"mult": 0.40},
            },
            "database-1": {
                "latency_ms": {"mult": 1.8},
                "cpu":        {"mult": 1.1},
            },
            "order-service-1": {
                "latency_ms": {"mult": 1.6},
            },
            "payment-service-1": {
                "latency_ms": {"mult": 1.4},
            },
        },
    },

    # ── 1.3: Новые тестовые сценарии ─────────────────────────────────────────

    "scenario_4": {
        "name":       "Payment Service Overload",
        "root_cause": {"instance": "payment-service-1", "type": "overload"},
        # Перегрузка payment-service-1 – новый корень, новый тип сбоя
        "anomaly": {
            "payment-service-1": {
                "cpu":        {"mult": 7.0},   # +600%
                "memory":     {"mult": 2.8},   # +180%
                "latency_ms": {"mult": 15.0},  # +1400%
                "rps":        {"mult": 0.25},  # -75%
            },
            # Минимальный каскад
            "database-1": {
                "latency_ms": {"mult": 1.3},   # +30%
                "cpu":        {"mult": 1.1},   # +10%
            },
        },
    },

    "scenario_5": {
        "name":       "Database Bottleneck",
        "root_cause": {"instance": "database-1", "type": "cpu_exhaustion"},
        # database-1 – центральный хаб, его деградация затрагивает всех
        # Важно: database доминирует СИЛЬНО чтобы выделиться на фоне каскада
        "anomaly": {
            "database-1": {
                "cpu":        {"mult": 9.0},   # +800%
                "memory":     {"mult": 3.0},   # +200%
                "latency_ms": {"mult": 18.0},  # +1700%
                "rps":        {"mult": 0.20},  # -80%
            },
            # Все соседи получают умеренный каскадный эффект
            "cache-service-1": {
                "latency_ms": {"mult": 1.5},   # +50%
            },
            "frontend-1": {
                "latency_ms": {"mult": 1.4},   # +40%
            },
            "order-service-1": {
                "latency_ms": {"mult": 1.6},   # +60%
            },
            "payment-service-1": {
                "latency_ms": {"mult": 1.5},   # +50%
            },
        },
    },
}

# ── Топология YAML ────────────────────────────────────────────────────────────

TOPOLOGY_YAML = """\
# Топология микросервисной системы CAIRN Demo
# Генерируется автоматически generate_demo_data.py

instances:
  - name: cache-service-1
    service: cache
    host: node-1
    cpu_limit: 2.0
    memory_limit: 512
    version: "1.0"
  - name: database-1
    service: database
    host: node-2
    cpu_limit: 4.0
    memory_limit: 2048
    version: "1.0"
  - name: frontend-1
    service: frontend
    host: node-1
    cpu_limit: 2.0
    memory_limit: 512
    version: "1.0"
  - name: order-service-1
    service: order
    host: node-3
    cpu_limit: 2.0
    memory_limit: 512
    version: "1.0"
  - name: payment-service-1
    service: payment
    host: node-3
    cpu_limit: 2.0
    memory_limit: 512
    version: "1.0"

call_edges:
  - [cache-service-1, database-1]
  - [frontend-1, database-1]
  - [order-service-1, database-1]
  - [payment-service-1, database-1]

colocation_groups:
  - [order-service-1, cache-service-1]

load_balancer_groups: []
"""

# ── Вспомогательные функции ───────────────────────────────────────────────────

def _add_noise(value: float, metric: str, rng: np.random.Generator) -> float:
    """Добавляет гауссов шум к значению метрики."""
    std = abs(value) * NOISE_FRAC.get(metric, 0.05)
    return float(np.clip(value + rng.normal(0, std), 0, None))


def _apply_anomaly(base: float, metric: str, params: dict, rng: np.random.Generator) -> float:
    """Применяет аномальный эффект к базовому значению."""
    result = base
    if "mult" in params:
        result *= params["mult"]
    if "add" in params:
        result += params["add"]
    return _add_noise(result, metric, rng)


def _generate_metrics(scenario: dict, rng: np.random.Generator) -> list[dict]:
    """Генерирует временной ряд метрик для сценария.

    Возвращает список строк для CSV:
    [{"timestamp": float, "instance": str, "cpu": float, ...}, ...]
    """
    rows = []
    anomaly_params = scenario["anomaly"]

    for step in range(N_NORMAL + N_ANOMALY):
        ts        = BASE_TS + step * STEP
        is_anom   = (step >= N_NORMAL)

        for inst in INSTANCES:
            base   = BASELINE[inst]
            inst_anom = anomaly_params.get(inst, {}) if is_anom else {}

            row = {"timestamp": ts, "instance": inst}
            for metric in METRICS:
                bval = base[metric]
                if inst_anom and metric in inst_anom:
                    val = _apply_anomaly(bval, metric, inst_anom[metric], rng)
                else:
                    val = _add_noise(bval, metric, rng)
                row[metric] = round(val, 4)
            rows.append(row)

    return rows


def _write_metrics_csv(rows: list[dict], path: Path) -> None:
    """Записывает метрики в CSV файл."""
    header = "timestamp,instance," + ",".join(METRICS)
    lines  = [header]
    for row in rows:
        vals = ",".join(str(row[m]) for m in METRICS)
        lines.append(f"{row['timestamp']},{row['instance']},{vals}")
    path.write_text("\n".join(lines), encoding="utf-8")


# Шаблоны лог-сообщений для каждого типа состояния
LOG_TEMPLATES_NORMAL = [
    "{inst} processed request in {lat:.1f}ms",
    "{inst} health check OK",
    "{inst} cache hit ratio 0.{r:02d}",
    "{inst} connection pool size {p}",
    "{inst} completed batch job",
]

LOG_TEMPLATES_ANOMALY = {
    "cpu_exhaustion": [
        "{inst} CPU usage critical: {cpu:.0f}%",
        "{inst} thread pool exhausted",
        "{inst} request queue depth: {q}",
        "{inst} response time degraded: {lat:.0f}ms",
        "{inst} GC pressure high",
    ],
    "memory_pressure": [
        "{inst} memory usage high: {mem:.0f}MB",
        "{inst} OOM warning: {mem:.0f}MB used",
        "{inst} cache eviction rate elevated",
        "{inst} heap dump triggered",
        "{inst} memory leak suspected in module",
    ],
    "latency_spike": [
        "{inst} upstream timeout after {lat:.0f}ms",
        "{inst} network latency spike detected",
        "{inst} connection timeout to downstream",
        "{inst} retry attempt {r} for request",
        "{inst} circuit breaker OPEN",
    ],
    "overload": [
        "{inst} request rate exceeded capacity: {q} queued",
        "{inst} rate limiter activated",
        "{inst} service degraded under high load",
        "{inst} response time {lat:.0f}ms exceeds SLA",
        "{inst} shed load: {r} requests dropped",
    ],
}


def _generate_logs(scenario: dict, rng: np.random.Generator) -> list[str]:
    """Генерирует лог-файл для сценария."""
    root_inst  = scenario["root_cause"]["instance"]
    fault_type = scenario["root_cause"]["type"]
    lines      = []

    for step in range(N_NORMAL + N_ANOMALY):
        ts      = BASE_TS + step * STEP
        is_anom = (step >= N_NORMAL)
        n_logs  = rng.integers(1, 4)

        for _ in range(n_logs):
            inst = rng.choice(INSTANCES)
            if is_anom and inst == root_inst:
                tmpl = rng.choice(LOG_TEMPLATES_ANOMALY[fault_type])
                msg  = tmpl.format(
                    inst=inst,
                    lat=BASELINE[inst]["latency_ms"] * 15,
                    cpu=BASELINE[inst]["cpu"] * 400,
                    mem=BASELINE[inst]["memory"] * 4000,
                    r=int(rng.integers(1, 5)),
                    q=int(rng.integers(50, 200)),
                )
            else:
                tmpl = rng.choice(LOG_TEMPLATES_NORMAL)
                msg  = tmpl.format(
                    inst=inst,
                    lat=BASELINE[inst]["latency_ms"] * (1 + rng.normal(0, 0.1)),
                    r=int(rng.integers(70, 99)),
                    p=int(rng.integers(5, 20)),
                )
            lines.append(f"{ts:.3f} | {inst} | INFO | {msg}")

    return lines


def _generate_traces(rng: np.random.Generator) -> list[dict]:
    """Генерирует трассировки вызовов в формате JSONTraceConnector."""
    traces = []
    for step in range(0, N_NORMAL + N_ANOMALY, 10):
        ts = BASE_TS + step * STEP
        for src, tgt in CALL_EDGES:
            trace_id = f"trace-{step:04d}-{src[:3]}-{tgt[:3]}"
            dur_root  = float(rng.uniform(50, 300))
            dur_child = float(rng.uniform(10, 100))
            # service = имя без суффикса "-1"
            src_svc = src.rsplit("-", 1)[0]
            tgt_svc = tgt.rsplit("-", 1)[0]
            traces.append({
                "trace_id": trace_id,
                "spans": [
                    {
                        "span_id":        f"{trace_id}-root",
                        "parent_span_id": None,
                        "service":        src_svc,
                        "instance":       src,
                        "operation":      f"call_{tgt_svc}",
                        "start_time":     ts,
                        "duration_ms":    dur_root,
                        "status":         "OK",
                        "attributes":     {},
                    },
                    {
                        "span_id":        f"{trace_id}-child",
                        "parent_span_id": f"{trace_id}-root",
                        "service":        tgt_svc,
                        "instance":       tgt,
                        "operation":      "handle_request",
                        "start_time":     ts + dur_root * 0.1 / 1000,
                        "duration_ms":    dur_child,
                        "status":         "OK",
                        "attributes":     {},
                    },
                ],
            })
    return traces


def _generate_labels(scenario: dict) -> dict:
    """Генерирует labels.json для сценария."""
    return {
        "scenario":    scenario["name"],
        "root_cause":  scenario["root_cause"],
        "normal_period": {
            "start": BASE_TS,
            "end":   BASE_TS + N_NORMAL - 1,
        },
        "anomaly_period": {
            "start": BASE_TS + N_NORMAL,
            "end":   BASE_TS + N_NORMAL + N_ANOMALY - 1,
        },
    }


# ── Основная функция ──────────────────────────────────────────────────────────

def generate_scenario(
    sc_name: str,
    scenario: dict,
    out_dir: Path,
    seed: int,
) -> None:
    """Генерирует один сценарий в директорию out_dir/sc_name/."""
    sc_dir = out_dir / sc_name
    sc_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    print(f"  {sc_name}: {scenario['name']} "
          f"(root={scenario['root_cause']['instance']}, "
          f"type={scenario['root_cause']['type']})")

    # metrics.csv
    rows = _generate_metrics(scenario, rng)
    _write_metrics_csv(rows, sc_dir / "metrics.csv")

    # logs.txt
    log_lines = _generate_logs(scenario, rng)
    (sc_dir / "logs.txt").write_text("\n".join(log_lines), encoding="utf-8")

    # traces.json
    traces = _generate_traces(rng)
    (sc_dir / "traces.json").write_text(
        json.dumps(traces, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # topology.yaml (одинакова для всех сценариев)
    (sc_dir / "topology.yaml").write_text(TOPOLOGY_YAML, encoding="utf-8")

    # labels.json
    labels = _generate_labels(scenario)
    (sc_dir / "labels.json").write_text(
        json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    n_rows = len(rows) // len(INSTANCES)
    print(f"    Метрик: {len(rows)} строк ({N_NORMAL} норм + {N_ANOMALY} аном шагов × {len(INSTANCES)} сервисов)")
    print(f"    Логов:  {len(log_lines)} строк")


def main() -> None:
    parser = argparse.ArgumentParser(description="Генератор демо-данных CAIRN")
    parser.add_argument("--out",  default="data/sample", help="Директория вывода")
    parser.add_argument("--seed", type=int, default=42,  help="Random seed")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Генерируем демо-данные в {out_dir}/")
    print(f"  Период:  {N_NORMAL} норм + {N_ANOMALY} аном шагов")
    print(f"  Сервисы: {', '.join(INSTANCES)}")
    print(f"  Метрики: {', '.join(METRICS)}\n")

    for i, (sc_name, scenario) in enumerate(SCENARIOS.items(), 1):
        generate_scenario(sc_name, scenario, out_dir, seed=args.seed + i)

    print(f"\n✅ Данные сгенерированы:")
    for sc_name in SCENARIOS:
        sc_dir = out_dir / sc_name
        files  = [f.name for f in sc_dir.iterdir()]
        print(f"  {sc_dir}/  →  {', '.join(sorted(files))}")

    print("\nСледующий шаг:")
    print(f"  python scripts/pretrain_demo.py --epochs 10")


if __name__ == "__main__":
    main()