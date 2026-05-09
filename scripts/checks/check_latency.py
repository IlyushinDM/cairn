"""Проверяем доступные latency метрики в Prometheus."""
import requests

URL = "http://localhost:9090"

queries = {
    "http_request_duration p99":
        'histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[1m])) by (le, service_name))',
    "http_request_duration p50":
        'histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket[1m])) by (le, service_name))',
    "http_req_count":
        'sum(rate(http_requests_total[1m])) by (service_name)',
    "grpc_duration p99":
        'histogram_quantile(0.99, sum(rate(grpc_server_handling_seconds_bucket[1m])) by (le, grpc_service))',
    "app latency":
        'histogram_quantile(0.99, sum(rate(app_request_latencies_bucket[1m])) by (le, service_name))',
}

print(f"Prometheus: {URL}\n")
found = []
for name, q in queries.items():
    try:
        r = requests.get(f"{URL}/api/v1/query", params={"query": q}, timeout=5)
        result = r.json().get("data", {}).get("result", [])
        n = len(result)
        print(f"  {'OK' if n > 0 else '--'}  {name}: {n} серий")
        if n > 0:
            found.append((name, q, result))
            for s in result[:2]:
                svc = s["metric"].get("service_name") or s["metric"].get("grpc_service", "?")
                val = float(s["value"][1])
                print(f"       {svc}: {val:.3f}s")
    except Exception as e:
        print(f"  !!  {name}: {e}")

# Также проверим все доступные метрики с latency/duration в имени
print("\n=== Все метрики с 'latency' или 'duration' ===")
r = requests.get(f"{URL}/api/v1/label/__name__/values", timeout=5)
names = r.json().get("data", [])
latency_metrics = [n for n in names if any(k in n for k in
                   ['latency', 'duration', 'response_time', 'request_time'])]
for m in latency_metrics[:20]:
    print(f"  {m}")
