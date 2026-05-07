"""Диагностика cAdvisor — без фильтров."""
import requests

URL = "http://localhost:9090"

# Проверяем без id-фильтра
print("=== Все доступные метрики container_* ===")
r = requests.get(f"{URL}/api/v1/label/__name__/values", timeout=5)
names = r.json().get("data", [])
container_metrics = [n for n in names if "container" in n]
print(f"Найдено container_* метрик: {len(container_metrics)}")
for m in container_metrics[:15]:
    print(f"  {m}")

print("\n=== container_cpu_usage_seconds_total без фильтра ===")
r = requests.get(f"{URL}/api/v1/query",
                 params={"query": "container_cpu_usage_seconds_total"},
                 timeout=5)
result = r.json().get("data", {}).get("result", [])
print(f"Всего серий: {len(result)}")
for s in result[:5]:
    print(f"  id={s['metric'].get('id','?')} name={s['metric'].get('name','?')} val={s['value'][1]}")

print("\n=== container_memory_usage_bytes без фильтра ===")
r = requests.get(f"{URL}/api/v1/query",
                 params={"query": "container_memory_usage_bytes"},
                 timeout=5)
result = r.json().get("data", {}).get("result", [])
print(f"Всего серий: {len(result)}")
for s in result[:5]:
    print(f"  id={s['metric'].get('id','?')} name={s['metric'].get('name','?')} val={s['value'][1]}")

print("\n=== cAdvisor статус ===")
try:
    r = requests.get("http://localhost:8080/metrics", timeout=5)
    lines = [l for l in r.text.split("\n") if "container_cpu" in l and not l.startswith("#")]
    print(f"Строк с container_cpu в /metrics: {len(lines)}")
    if lines:
        print(f"  Пример: {lines[0][:100]}")
except Exception as e:
    print(f"cAdvisor /metrics: {e}")
