"""Диагностика Prometheus – показывает доступные метрики и labels."""

import subprocess
import json
import requests

PROMETHEUS_URL = "http://localhost:9090"

# Шаг 1: маппинг ID → имя контейнера через docker ps
print("=" * 60)
print("Шаг 1: Контейнеры Docker")
print("=" * 60)
result = subprocess.run(
    ["docker", "ps", "--format", "{{.ID}}\t{{.Names}}"],
    capture_output=True, text=True
)
id_to_name = {}
for line in result.stdout.strip().split("\n"):
    if "\t" in line:
        short_id, name = line.split("\t", 1)
        id_to_name[short_id] = name
        print(f"  {short_id}  →  {name}")

# Шаг 2: примеры labels в Prometheus
print("\n" + "=" * 60)
print("Шаг 2: Labels в Prometheus")
print("=" * 60)
resp = requests.get(
    f"{PROMETHEUS_URL}/api/v1/query",
    params={"query": "container_cpu_usage_seconds_total"},
    timeout=10,
)
data = resp.json().get("data", {}).get("result", [])
print(f"Всего серий: {len(data)}")
if data:
    print("Labels первой серии:", json.dumps(data[0]["metric"], indent=2))
    print("Labels второй серии:", json.dumps(data[1]["metric"], indent=2) if len(data) > 1 else "-")

# Шаг 3: ищем cairn-контейнеры в метриках
print("\n" + "=" * 60)
print("Шаг 3: Cairn-контейнеры в Prometheus")
print("=" * 60)
found = 0
for series in data:
    metric = series["metric"]
    id_field = metric.get("id", "")
    # Берём последние 12 символов пути /docker/<hash>
    if "/docker/" in id_field:
        short = id_field.split("/docker/")[-1][:12]
        name = id_to_name.get(short, "unknown")
        if name.startswith("cairn-"):
            print(f"  {short}  →  {name}  (labels: {list(metric.keys())})")
            found += 1

if found == 0:
    print("  [!!] Cairn-контейнеры не найдены в метриках!")
    print("  Все доступные id-пути:")
    for series in data[:8]:
        print(f"    {series['metric'].get('id', '?')}")