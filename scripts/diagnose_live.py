"""Диагностика подключения LiveSystemConnector.
Запуск: python diagnose_live.py configs/connectors/online_boutique.yaml
"""
import sys, time
sys.path.insert(0, "src")

cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/connectors/online_boutique.yaml"
print(f"Конфиг: {cfg}\n")

from cairn.connectors.live_connector import LiveSystemConnector
conn = LiveSystemConnector(cfg)

print(f"Система: {conn.system_name}")
ok, msg = conn.is_available()
print(f"Статус:  {'OK' if ok else 'FAIL'} – {msg}\n")

print("Запрашиваем метрики (последние 5 минут)...")
now = time.time()
try:
    md = conn.fetch_metrics(now - 300, now)
    print(f"Экземпляров: {md.n_instances}")
    print(f"Метрик:      {md.n_metrics} {md.metric_names}")
    print(f"Точек:       {len(md.timestamps)}")
    print(f"Экземпляры:  {md.instance_names[:10]}")
    if md.n_instances == 0:
        print("\n[!!] Нет данных – смотрим raw запрос:")
        import requests, yaml
        with open(cfg) as f:
            c = yaml.safe_load(f)
        url = c["metrics"]["url"]
        q   = list(c["metrics"]["metric_queries"].values())[0]
        r   = requests.get(f"{url}/api/v1/query_range",
                           params={"query": q, "start": now-300, "end": now, "step": 15},
                           timeout=10)
        results = r.json().get("data",{}).get("result",[])
        print(f"  Prometheus вернул {len(results)} серий")
        if results:
            print(f"  Первая серия labels: {results[0]['metric']}")
            print(f"  Первая серия точек:  {len(results[0].get('values',[]))}")
except Exception as e:
    print(f"ОШИБКА: {e}")
    import traceback; traceback.print_exc()
