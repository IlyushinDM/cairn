"""Ищем источники latency/trace данных в контейнерах Boutique."""
import subprocess, re

CONTAINERS = [
    "cairn-loadgenerator",
    "cairn-frontend",
    "cairn-checkoutservice",
    "cairn-cartservice",
]

TIME_RE = re.compile(
    r'(\d+\.?\d*)\s*(ms|s|sec|milliseconds?|seconds?|duration|latency|took|elapsed)',
    re.IGNORECASE
)

print("=== Анализ логов на наличие timing данных ===\n")
for container in CONTAINERS:
    try:
        result = subprocess.run(
            ["docker", "logs", container, "--since", "60s", "--tail", "50"],
            capture_output=True, text=True, timeout=5
        )
        lines = (result.stdout + result.stderr).strip().split("\n")
        timing_lines = [l for l in lines if TIME_RE.search(l)]
        print(f"{container}: {len(lines)} строк, {len(timing_lines)} с timing")
        for l in timing_lines[:3]:
            print(f"  {l[:120]}")
        if not timing_lines and lines and lines[0]:
            print(f"  Пример: {lines[0][:120]}")
    except Exception as e:
        print(f"{container}: {e}")
    print()

print("=== Метрики cAdvisor с network ===")
import requests
URL = "http://localhost:9090"
for q in [
    'container_network_receive_bytes_total{id=~"/docker/.+"}',
    'container_network_transmit_bytes_total{id=~"/docker/.+"}',
]:
    r = requests.get(f"{URL}/api/v1/query", params={"query": q}, timeout=5)
    n = len(r.json().get("data",{}).get("result",[]))
    print(f"  {q[:50]}: {n} серий")
