"""Скрипт генерации демонстрационного набора данных CAIRN.

Система из 5 сервисов:
  frontend → order-service → payment-service
                           → database
  order-service и cache-service размещены на одном хосте (node-2)

Сценарий:
  - 200 отсчётов нормальной работы (t = 0..199 с)
  - 100 отсчётов аномалии (t = 200..299 с)
  - Первопричина: order-service-1 (cpu_exhaustion)

Использование:
    python scripts/generate_demo_data.py
    python scripts/generate_demo_data.py --out data/sample --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Конфигурация топологии
# ---------------------------------------------------------------------------

INSTANCES = [
    {"name": "frontend-1",        "service": "frontend",        "host": "node-1", "cpu_limit": 2.0,  "memory_limit": 512,  "version": "1.2"},
    {"name": "order-service-1",   "service": "order-service",   "host": "node-2", "cpu_limit": 4.0,  "memory_limit": 1024, "version": "2.1"},
    {"name": "payment-service-1", "service": "payment-service", "host": "node-3", "cpu_limit": 2.0,  "memory_limit": 512,  "version": "1.5"},
    {"name": "cache-service-1",   "service": "cache-service",   "host": "node-2", "cpu_limit": 1.0,  "memory_limit": 2048, "version": "3.0"},
    {"name": "database-1",        "service": "database",        "host": "node-4", "cpu_limit": 8.0,  "memory_limit": 4096, "version": "14.2"},
]

CALL_EDGES = [
    ["frontend-1",      "order-service-1"],
    ["order-service-1", "payment-service-1"],
    ["order-service-1", "database-1"],
]

COLOCATION_GROUPS = [
    ["order-service-1", "cache-service-1"],  # оба на node-2
]

LB_GROUPS: list = []  # В демо по одному экземпляру каждого сервиса

# ---------------------------------------------------------------------------
# Профили нормального поведения
# ---------------------------------------------------------------------------
# {instance: {metric: (base_value, noise_std)}}
NORMAL_PROFILE = {
    "frontend-1":        {"cpu": (0.15, 0.02), "memory": (0.40, 0.03), "latency_ms": (12.0, 1.5),  "rps": (200.0, 10.0)},
    "order-service-1":   {"cpu": (0.30, 0.03), "memory": (0.55, 0.04), "latency_ms": (25.0, 3.0),  "rps": (180.0, 8.0)},
    "payment-service-1": {"cpu": (0.20, 0.02), "memory": (0.45, 0.03), "latency_ms": (18.0, 2.0),  "rps": (160.0, 7.0)},
    "cache-service-1":   {"cpu": (0.10, 0.01), "memory": (0.70, 0.05), "latency_ms": (2.0,  0.5),  "rps": (500.0, 20.0)},
    "database-1":        {"cpu": (0.25, 0.03), "memory": (0.60, 0.04), "latency_ms": (8.0,  1.0),  "rps": (300.0, 12.0)},
}

# Аномальный профиль (только отклонения от нормы для затронутых инстансов)
ANOMALY_DELTA = {
    "order-service-1":   {"cpu": +0.65, "latency_ms": +150.0, "rps": -40.0},  # первопричина: cpu_exhaustion
    "frontend-1":        {"latency_ms": +80.0,  "rps": -20.0},                 # downstream effect
    "payment-service-1": {"latency_ms": +60.0,  "rps": -15.0},                 # downstream effect
    "database-1":        {"cpu": +0.10, "latency_ms": +20.0},                  # слабый эффект
    "cache-service-1":   {"cpu": +0.05},                                        # совместное размещение
}

METRICS = ["cpu", "memory", "latency_ms", "rps"]
NORMAL_STEPS = 200
ANOMALY_STEPS = 100
BASE_TS = 1_700_000_000.0  # начальная метка


# ---------------------------------------------------------------------------
# Генерация
# ---------------------------------------------------------------------------

def generate_metrics(rng: np.random.Generator, out_dir: Path) -> None:
    rows = []
    inst_names = [i["name"] for i in INSTANCES]

    for step in range(NORMAL_STEPS + ANOMALY_STEPS):
        ts = BASE_TS + step
        is_anomaly = step >= NORMAL_STEPS

        for inst in inst_names:
            row = {"timestamp": ts, "instance": inst}
            profile = NORMAL_PROFILE[inst]
            delta = ANOMALY_DELTA.get(inst, {}) if is_anomaly else {}

            for metric in METRICS:
                base, noise = profile.get(metric, (0.0, 0.01))
                d = delta.get(metric, 0.0)
                # Аномалия нарастает постепенно в первые 20 шагов
                ramp = min(1.0, (step - NORMAL_STEPS) / 20.0) if is_anomaly else 0.0
                val = base + d * ramp + rng.normal(0, noise)
                # Ограничиваем cpu в [0, 1]
                if metric == "cpu":
                    val = float(np.clip(val, 0.0, 1.0))
                elif metric == "memory":
                    val = float(np.clip(val, 0.0, 1.0))
                else:
                    val = max(0.0, float(val))
                row[metric] = round(val, 4)

            rows.append(row)

    out_path = out_dir / "metrics.csv"
    import csv
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "instance"] + METRICS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  ✓ metrics.csv  ({len(rows)} строк)")


def generate_logs(rng: np.random.Generator, out_dir: Path) -> None:
    lines = []
    normal_msgs = {
        "frontend-1":        ["Запрос обработан успешно", "GET /api/orders: 200", "Соединение установлено"],
        "order-service-1":   ["Заказ создан", "Обработка платежа начата", "DB запрос выполнен за {ms}мс"],
        "payment-service-1": ["Платёж авторизован", "Транзакция подтверждена"],
        "cache-service-1":   ["Cache hit", "Cache miss: перезагрузка", "TTL обновлён"],
        "database-1":        ["SELECT выполнен за {ms}мс", "Индекс использован", "Соединение из пула"],
    }
    anomaly_msgs = {
        "order-service-1":   [
            "WARN: CPU throttling detected",
            "ERROR: Request timeout after 5000ms",
            "WARN: Thread pool exhausted (queue={q})",
            "ERROR: DB connection timeout",
        ],
        "frontend-1":        ["WARN: Upstream latency high ({ms}ms)", "ERROR: Gateway timeout"],
        "payment-service-1": ["WARN: Response time degraded"],
        "database-1":        ["WARN: Slow query detected ({ms}ms)"],
        "cache-service-1":   ["WARN: High CPU on colocated node"],
    }

    for step in range(NORMAL_STEPS + ANOMALY_STEPS):
        ts = BASE_TS + step
        is_anomaly = step >= NORMAL_STEPS

        for inst in [i["name"] for i in INSTANCES]:
            # Нормальные сообщения (каждый 10-й шаг)
            if step % 10 == 0:
                msgs = normal_msgs.get(inst, ["Операция выполнена"])
                msg = rng.choice(msgs).format(ms=int(rng.integers(1, 50)), q=int(rng.integers(1, 100)))
                lines.append(f"{ts:.1f} | {inst} | INFO | {msg}")

            # Аномальные сообщения
            if is_anomaly and inst in anomaly_msgs:
                ramp = min(1.0, (step - NORMAL_STEPS) / 20.0)
                if rng.random() < 0.3 * ramp:
                    msgs = anomaly_msgs[inst]
                    msg = rng.choice(msgs).format(ms=int(rng.integers(100, 5000)), q=int(rng.integers(10, 50)))
                    level = "ERROR" if "ERROR" in msg else "WARN"
                    lines.append(f"{ts:.1f} | {inst} | {level} | {msg}")

    out_path = out_dir / "logs.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  ✓ logs.txt     ({len(lines)} строк)")


def generate_traces(rng: np.random.Generator, out_dir: Path) -> None:
    traces = []
    # Генерируем ~2 трассировки на секунду
    for step in range(0, NORMAL_STEPS + ANOMALY_STEPS, 2):
        ts = BASE_TS + step
        is_anomaly = step >= NORMAL_STEPS
        ramp = min(1.0, (step - NORMAL_STEPS) / 20.0) if is_anomaly else 0.0

        trace_id = uuid.uuid4().hex[:16]
        span_root_id = uuid.uuid4().hex[:8]
        span_order_id = uuid.uuid4().hex[:8]
        span_pay_id = uuid.uuid4().hex[:8]
        span_db_id = uuid.uuid4().hex[:8]

        base_latency = 12.0 + rng.normal(0, 1.5) + 80.0 * ramp
        order_latency = 25.0 + rng.normal(0, 3.0) + 150.0 * ramp
        pay_latency = 18.0 + rng.normal(0, 2.0) + 60.0 * ramp
        db_latency = 8.0 + rng.normal(0, 1.0) + 20.0 * ramp

        status_order = "ERROR" if is_anomaly and ramp > 0.7 and rng.random() < 0.3 else "OK"

        spans = [
            {
                "span_id": span_root_id,
                "parent_span_id": None,
                "service": "frontend",
                "instance": "frontend-1",
                "operation": "GET /api/orders",
                "start_time": ts,
                "duration_ms": round(base_latency, 2),
                "status": "OK",
            },
            {
                "span_id": span_order_id,
                "parent_span_id": span_root_id,
                "service": "order-service",
                "instance": "order-service-1",
                "operation": "processOrder",
                "start_time": ts + 0.002,
                "duration_ms": round(order_latency, 2),
                "status": status_order,
            },
            {
                "span_id": span_pay_id,
                "parent_span_id": span_order_id,
                "service": "payment-service",
                "instance": "payment-service-1",
                "operation": "authorizePayment",
                "start_time": ts + 0.010,
                "duration_ms": round(pay_latency, 2),
                "status": "OK",
            },
            {
                "span_id": span_db_id,
                "parent_span_id": span_order_id,
                "service": "database",
                "instance": "database-1",
                "operation": "SELECT orders",
                "start_time": ts + 0.015,
                "duration_ms": round(db_latency, 2),
                "status": "OK",
            },
        ]
        traces.append({"trace_id": trace_id, "spans": spans})

    out_path = out_dir / "traces.json"
    out_path.write_text(json.dumps(traces, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✓ traces.json  ({len(traces)} трассировок)")


def generate_topology(out_dir: Path) -> None:
    topology = {
        "instances": INSTANCES,
        "call_edges": CALL_EDGES,
        "colocation_groups": COLOCATION_GROUPS,
        "load_balancer_groups": LB_GROUPS,
    }
    out_path = out_dir / "topology.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(topology, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"  ✓ topology.yaml ({len(INSTANCES)} экземпляров)")


def generate_labels(out_dir: Path) -> None:
    """Метаданные аномалии — для валидации и обучения."""
    labels = {
        "dataset": "cairn-demo-v1",
        "normal_period": {"start": BASE_TS, "end": BASE_TS + NORMAL_STEPS - 1, "steps": NORMAL_STEPS},
        "anomaly_period": {"start": BASE_TS + NORMAL_STEPS, "end": BASE_TS + NORMAL_STEPS + ANOMALY_STEPS - 1, "steps": ANOMALY_STEPS},
        "root_cause": {
            "instance": "order-service-1",
            "service": "order-service",
            "type": "cpu_exhaustion",
            "onset_ts": BASE_TS + NORMAL_STEPS,
            "description": "CPU загрузка order-service-1 достигает 95%, вызывая деградацию задержки вниз по цепочке вызовов",
        },
        "affected_instances": ["order-service-1", "frontend-1", "payment-service-1", "database-1", "cache-service-1"],
    }
    out_path = out_dir / "labels.json"
    out_path.write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✓ labels.json  (аннотации аномалии)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация демо-данных CAIRN")
    parser.add_argument("--out", default="data/sample", help="Директория для вывода")
    parser.add_argument("--seed", type=int, default=42, help="Seed генератора случайных чисел")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    print(f"Генерация демо-данных CAIRN → {out_dir}/")
    print(f"  Сервисов: {len(INSTANCES)}, Нормальных шагов: {NORMAL_STEPS}, Аномальных шагов: {ANOMALY_STEPS}")
    generate_metrics(rng, out_dir)
    generate_logs(rng, out_dir)
    generate_traces(rng, out_dir)
    generate_topology(out_dir)
    generate_labels(out_dir)
    print("Готово.")


if __name__ == "__main__":
    main()
