"""Инъекция сбоев в Online Boutique и запуск CAIRN-анализа.

Три типа сбоев:
  cpu       — CPU stress в целевом контейнере (требует stress-ng)
  pause     — приостановка контейнера (имитирует network partition)
  memory    — утечка памяти через stress-ng

Использование:
    # Инъекция CPU в cartservice, 60 секунд
    python scripts/inject_fault.py --type cpu --target cartservice --duration 60

    # Инъекция с автоматическим анализом CAIRN после сбоя
    python scripts/inject_fault.py --type cpu --target cartservice --duration 60 --analyze

    # Проверить что стек запущен
    python scripts/inject_fault.py --check
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── Маппинг сервисов Online Boutique → имена контейнеров ─────────────────────
SERVICE_MAP = {
    "frontend":              "cairn-frontend",
    "cartservice":           "cairn-cartservice",
    "productcatalogservice": "cairn-productcatalog",
    "currencyservice":       "cairn-currencyservice",
    "recommendationservice": "cairn-recommendationservice",
    "checkoutservice":       "cairn-checkoutservice",
    "shippingservice":       "cairn-shippingservice",
    "paymentservice":        "cairn-paymentservice",
    "emailservice":          "cairn-emailservice",
    "adservice":             "cairn-adservice",
    "redis-cart":            "cairn-redis",
}

PROMETHEUS_URL = "http://localhost:9090"
CAIRN_METRICS  = [
    "container_cpu_usage_seconds_total",
    "container_memory_usage_bytes",
    "container_network_receive_bytes_total",
    "container_network_transmit_bytes_total",
]


def check_stack() -> bool:
    """Проверяет что Docker-стек запущен."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=True
        )
        running = result.stdout.strip().split("\n")
        needed  = ["cairn-prometheus", "cairn-cadvisor", "cairn-frontend"]
        missing = [n for n in needed if n not in running]
        if missing:
            print(f"[!!] Не запущены контейнеры: {missing}")
            print("     Запустите: docker compose -f docker/docker-compose.yml up -d")
            return False
        print("[OK] Стек запущен:")
        for name in running:
            if name.startswith("cairn-"):
                print(f"     {name}")
        print(f"\n[OK] Boutique:    http://localhost:8081")
        print(f"[OK] Prometheus:  http://localhost:9090")
        print(f"[OK] cAdvisor:    http://localhost:8080")
        return True
    except Exception as e:
        print(f"[!!] Ошибка Docker: {e}")
        return False


def _has_shell(container: str) -> bool:
    """Проверяет наличие sh (distroless-образы его не имеют)."""
    r = subprocess.run(["docker", "exec", container, "sh", "-c", "echo ok"],
                       capture_output=True)
    return r.returncode == 0


def inject_cpu(container: str, duration: int) -> None:
    """CPU stress. Для distroless-образов использует pause/unpause cycle."""
    print(f"\n[->] CPU stress в {container} на {duration}с...")
    import threading

    if _has_shell(container):
        cmds = [
            f"stress-ng --cpu 0 --timeout {duration}s &",
            f"python3 -c 'import time,math; t=time.time(); [math.factorial(9999) for _ in iter(int,1) if time.time()-t<{duration}]' &",
        ]
        for cmd in cmds:
            r = subprocess.run(["docker", "exec", container, "sh", "-c", cmd],
                               capture_output=True)
            if r.returncode == 0:
                print(f"[OK] stress запущен")
                return

    # Distroless — pause/unpause cycle
    print("[~~] Distroless-образ — использую pause/unpause cycle...")
    stop = threading.Event()

    def _cycle():
        while not stop.is_set():
            subprocess.run(["docker", "pause",   container], capture_output=True)
            stop.wait(0.5)
            subprocess.run(["docker", "unpause", container], capture_output=True)
            stop.wait(0.3)

    t = threading.Thread(target=_cycle, daemon=True)
    t.start()
    stop.wait(duration)
    stop.set()
    subprocess.run(["docker", "unpause", container], capture_output=True)
    print(f"[OK] Pause/unpause cycle завершён ({duration}с)")


def inject_pause(container: str, duration: int) -> None:
    """Приостановка контейнера — имитирует network partition."""
    print(f"\n[→] Pause {container} на {duration}с...")
    subprocess.run(["docker", "pause", container], check=True)
    print(f"[OK] Контейнер приостановлен")
    time.sleep(duration)
    subprocess.run(["docker", "unpause", container], check=True)
    print(f"[OK] Контейнер возобновлён")


def inject_memory(container: str, duration: int) -> None:
    """Memory stress. Для distroless — fallback на pause/unpause."""
    print(f"\n[->] Memory stress в {container} на {duration}с...")
    if _has_shell(container):
        cmds = [
            f"stress-ng --vm 1 --vm-bytes 256M --timeout {duration}s &",
            f"python3 -c 'import time; d=[bytearray(1024*1024) for _ in range(256)]; time.sleep({duration})' &",
        ]
        for cmd in cmds:
            r = subprocess.run(["docker", "exec", container, "sh", "-c", cmd],
                               capture_output=True)
            if r.returncode == 0:
                print("[OK] memory stress запущен")
                return
    print("[~~] Distroless — fallback на pause/unpause cycle")
    inject_cpu(container, duration)


# Метрики которые запрашиваем из Prometheus (cAdvisor)
# Используем rate(60s) — надёжнее irate после перезапуска стека
PROM_QUERIES = {
    "cpu":        'irate(container_cpu_usage_seconds_total{id=~"/docker/.+"}[60s])',
    "memory_mb":  'container_memory_usage_bytes{id=~"/docker/.+"} / 1048576',
    "net_rx_kbps":'rate(container_network_receive_bytes_total{id=~"/docker/.+"}[60s]) / 1024',
    "net_tx_kbps":'rate(container_network_transmit_bytes_total{id=~"/docker/.+"}[60s]) / 1024',
}
def _build_id_to_name() -> dict[str, str]:
    """Строит маппинг short_docker_id -> container_name через docker ps."""
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.ID}}	{{.Names}}"],
        capture_output=True, text=True
    )
    mapping = {}
    for line in result.stdout.strip().split("\n"):
        if "\t" in line:
            short_id, name = line.split("\t", 1)
            mapping[short_id.strip()] = name.strip()
    return mapping


def _fetch_prometheus_direct(url: str, metric_names: list,
                             start: float, end: float, step: int = 5):
    """Получает метрики из Prometheus через query_range API.

    cAdvisor хранит имя контейнера в label id=/docker/<hash>.
    Маппим hash[:12] → cairn-имя через docker ps.
    """
    import requests
    import numpy as np

    hash_to_name = {k: v for k, v in _build_id_to_name().items()
                    if v.startswith("cairn-")}
    print(f"     Cairn-контейнеров найдено: {len(hash_to_name)}")

    query_map        = PROM_QUERIES
    metric_names_out = list(query_map.keys())
    all_data: dict[str, dict[str, list]] = {}

    for metric_key, query in query_map.items():
        try:
            resp = requests.get(
                f"{url}/api/v1/query_range",
                params={"query": query, "start": start, "end": end, "step": step},
                timeout=15,
            )
            resp.raise_for_status()
            result = resp.json().get("data", {}).get("result", [])
        except Exception as e:
            print(f"     [!!] Ошибка запроса {metric_key}: {e}")
            continue

        for series in result:
            id_field = series.get("metric", {}).get("id", "")
            if "/docker/" not in id_field:
                continue
            short = id_field.split("/docker/")[-1][:12]
            inst  = hash_to_name.get(short)
            if not inst:
                continue
            all_data.setdefault(inst, {})
            all_data[inst][metric_key] = [
                (float(ts), float(val))
                for ts, val in series.get("values", [])
                if val not in ("NaN", "Inf", "-Inf")
            ]

    if not all_data:
        print("[!!] Нет данных из Prometheus.")
        print(f"     Найдено cairn-контейнеров: {len(hash_to_name)}")
        print("     Возможно временной диапазон слишком мал — увеличьте --warmup.")
        return None

    inst_names   = sorted(all_data.keys())
    print(f"     Контейнеры с данными: {inst_names}")

    first_inst   = inst_names[0]
    first_metric = next((m for m in metric_names_out if m in all_data[first_inst]), None)
    if not first_metric:
        return None

    ts_list    = [ts for ts, _ in all_data[first_inst][first_metric]]
    timestamps = np.array(ts_list, dtype=np.float64)
    T          = len(timestamps)
    ts_idx     = {ts: i for i, ts in enumerate(ts_list)}

    values = np.full((T, len(inst_names), len(metric_names_out)), np.nan)
    for ni, inst in enumerate(inst_names):
        for mi, met in enumerate(metric_names_out):
            for ts, val in all_data.get(inst, {}).get(met, []):
                if ts in ts_idx:
                    values[ts_idx[ts], ni, mi] = val

    from cairn.connectors.base import MetricData
    return MetricData(timestamps, values, inst_names, metric_names_out)


def _write_boutique_topology(path: Path) -> None:
    """Создаёт topology.yaml для Online Boutique."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = """\
# Топология Online Boutique для CAIRN
instances:
  - name: cairn-frontend
    service: frontend
    host: docker-host
    cpu_limit: 2.0
    memory_limit: 256
    version: "v0.10.1"
  - name: cairn-cartservice
    service: cartservice
    host: docker-host
    cpu_limit: 2.0
    memory_limit: 256
    version: "v0.10.1"
  - name: cairn-productcatalog
    service: productcatalogservice
    host: docker-host
    cpu_limit: 2.0
    memory_limit: 256
    version: "v0.10.1"
  - name: cairn-checkoutservice
    service: checkoutservice
    host: docker-host
    cpu_limit: 2.0
    memory_limit: 256
    version: "v0.10.1"
  - name: cairn-paymentservice
    service: paymentservice
    host: docker-host
    cpu_limit: 2.0
    memory_limit: 256
    version: "v0.10.1"
  - name: cairn-shippingservice
    service: shippingservice
    host: docker-host
    cpu_limit: 2.0
    memory_limit: 256
    version: "v0.10.1"
  - name: cairn-currencyservice
    service: currencyservice
    host: docker-host
    cpu_limit: 2.0
    memory_limit: 256
    version: "v0.10.1"
  - name: cairn-recommendationservice
    service: recommendationservice
    host: docker-host
    cpu_limit: 2.0
    memory_limit: 256
    version: "v0.10.1"
  - name: cairn-redis
    service: redis-cart
    host: docker-host
    cpu_limit: 1.0
    memory_limit: 128
    version: "7.2"

call_edges:
  - [cairn-frontend,             cairn-cartservice]
  - [cairn-frontend,             cairn-productcatalog]
  - [cairn-frontend,             cairn-currencyservice]
  - [cairn-frontend,             cairn-recommendationservice]
  - [cairn-frontend,             cairn-shippingservice]
  - [cairn-frontend,             cairn-checkoutservice]
  - [cairn-checkoutservice,      cairn-cartservice]
  - [cairn-checkoutservice,      cairn-paymentservice]
  - [cairn-checkoutservice,      cairn-shippingservice]
  - [cairn-checkoutservice,      cairn-currencyservice]
  - [cairn-checkoutservice,      cairn-productcatalog]
  - [cairn-cartservice,          cairn-redis]
  - [cairn-recommendationservice, cairn-productcatalog]

colocation_groups: []
load_balancer_groups: []
"""
    path.write_text(content, encoding="utf-8")


def _prometheus_has_per_container_data() -> bool:
    """Проверяет что Prometheus видит отдельные контейнеры (не только агрегаты)."""
    try:
        import requests as _req
        r = _req.get(f"{PROMETHEUS_URL}/api/v1/query",
                     params={"query": 'container_memory_usage_bytes{id=~"/docker/.{12,}"}'},
                     timeout=5)
        return len(r.json().get("data", {}).get("result", [])) > 0
    except Exception:
        return False


def _collect_docker_stats_period(duration_sec: int,
                                  interval: float = 2.0) -> dict:
    """Собирает метрики через docker stats за указанный период.

    Возвращает: {container_name: {"cpu_pct": [...], "mem_mb": [...]}}
    """
    import numpy as np
    data: dict[str, dict[str, list]] = {}
    t_end = time.time() + duration_sec
    n = 0
    while time.time() < t_end:
        try:
            result = subprocess.run(
                ["docker", "stats", "--no-stream", "--format",
                 "{{.Name}}	{{.CPUPerc}}	{{.MemUsage}}"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                name = parts[0].strip()
                if not name.startswith("cairn-"):
                    continue
                cpu_s, mem_s = parts[1].strip(), parts[2].strip()
                cpu = float(cpu_s.replace("%", "") or 0)
                mem_part = mem_s.split("/")[0].strip()
                if "GiB" in mem_part:
                    mem = float(mem_part.replace("GiB","").strip()) * 1024
                elif "MiB" in mem_part:
                    mem = float(mem_part.replace("MiB","").strip())
                elif "KiB" in mem_part:
                    mem = float(mem_part.replace("KiB","").strip()) / 1024
                else:
                    mem = 0.0
                data.setdefault(name, {"cpu_pct": [], "mem_mb": []})
                data[name]["cpu_pct"].append(cpu)
                data[name]["mem_mb"].append(mem)
            n += 1
        except Exception:
            pass
        time.sleep(interval)
    print(f"     docker stats: {n} снимков, {len(data)} контейнеров")
    return data


def _analyze_stats(norm_data: dict, anom_data: dict,
                   target_container: str, fault_type: str) -> None:
    """Ранжирует контейнеры по аномальному скору из docker stats."""
    import numpy as np

    # Фильтруем: только сервисы из boutique_topology.yaml
    # Инфраструктурные (prometheus, cadvisor) исключаем — они не часть приложения
    BOUTIQUE_SERVICES = {
        "cairn-frontend", "cairn-cartservice", "cairn-checkoutservice",
        "cairn-paymentservice", "cairn-shippingservice", "cairn-currencyservice",
        "cairn-recommendationservice", "cairn-productcatalog",
        "cairn-emailservice", "cairn-adservice", "cairn-redis",
    }
    all_containers = sorted(
        (set(norm_data) | set(anom_data)) & BOUTIQUE_SERVICES
    )

    scores = {}
    for name in all_containers:
        n_vals = norm_data.get(name, {})
        a_vals = anom_data.get(name, {})
        if not n_vals or not a_vals:
            continue
        container_scores = []
        for metric in ("cpu_pct", "mem_mb"):
            n_arr  = np.array(n_vals.get(metric, [0.0]))
            a_arr  = np.array(a_vals.get(metric, [0.0]))
            mu_n, std_n = n_arr.mean(), n_arr.std()
            mu_a, std_a = a_arr.mean(), a_arr.std()
            pooled = np.sqrt((std_n**2 + std_a**2) / 2.0 + 1e-9)
            effect     = abs(mu_a - mu_n) / (pooled + 1e-9)
            var_change = np.log1p(std_a / (std_n + 1e-6))
            container_scores.append(effect + var_change)
        scores[name] = float(np.mean(container_scores))

    # Топологическая корректировка из boutique_topology.yaml
    try:
        from cairn.connectors.csv_file import YAMLTopologyConnector
        from cairn.perception import HypergraphBuilder
        topo_path = Path("docker/boutique_topology.yaml")
        if not topo_path.exists():
            _write_boutique_topology(topo_path)
        topo       = YAMLTopologyConnector(topo_path).fetch()
        hypergraph = HypergraphBuilder.from_topology_data(topo)

        # Для каждого сервиса: сколько других вызывает его (n_callers)?
        # Downstream (много вызывающих) — жертва, штраф к скору
        called_by: dict[str, int] = {}
        callee_map: dict[str, list] = {}
        for edge in hypergraph.edges:
            if edge.edge_type == "call" and len(edge.members) >= 2:
                src = hypergraph.instance_names[edge.members[0]]
                dst = hypergraph.instance_names[edge.members[1]]
                callee_map.setdefault(src, []).append(dst)
                called_by[dst] = called_by.get(dst, 0) + 1

        import numpy as _np
        adjusted = {}
        for name, score in scores.items():
            callees     = callee_map.get(name, [])
            cs          = [scores.get(c, 0.0) for c in callees if c in scores]
            cascade_avg = float(_np.mean(cs)) if cs else 0.0
            n_callers   = called_by.get(name, 0)
            # upstream score: высокий если нет cascade + мало вызывающих
            adjusted[name] = score / (1.0 + cascade_avg) / (1.0 + n_callers * 0.5)

        top_orig = max(scores,    key=scores.get)
        top_adj  = max(adjusted,  key=adjusted.get)
        if top_adj != top_orig:
            print(f"     [топология] скор скорректирован: {top_orig} -> {top_adj}")
        scores = adjusted
    except Exception as e:
        pass  # топология недоступна — используем исходные скоры

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'─'*52}")
    print(f"{'Контейнер':<35} {'Score':>8}  Ранг")
    print(f"{'─'*52}")
    for rank, (name, score) in enumerate(ranked, 1):
        marker = "  <- ROOT CANDIDATE" if rank == 1 else ""
        print(f"{name:<35} {score:>8.3f}  #{rank}{marker}")
    print(f"{'─'*52}")

    if not ranked:
        print("[!!] Нет данных для анализа"); return

    root_name, root_score = ranked[0]
    target_rank = next((r for r,(n,_) in enumerate(ranked,1)
                        if n == target_container), None)

    print(f"\nПервопричина по CAIRN:    {root_name}  (score={root_score:.3f})")
    print(f"Инжектированный сбой:     {target_container} ({fault_type})")

    if root_name == target_container:
        print("Результат: ✓ ВЕРНО — первопричина определена точно")
    elif target_rank and target_rank <= 3:
        print(f"Результат: ~ ЧАСТИЧНО — {target_container} на ранге #{target_rank}")
        print(f"           Каскадный эффект через топологию")
    else:
        print(f"Результат: ✗ {target_container} на ранге #{target_rank}")

    print(f"\nТоп-3 каузальной цепочки:")
    for rank, (name, score) in enumerate(ranked[:3], 1):
        marker = " <- инжектировано" if name == target_container else ""
        print(f"  #{rank}  {name:<35} score={score:.3f}{marker}")


def collect_and_analyze(target_service: str, fault_type: str,
                        normal_duration: int, anomaly_duration: int) -> None:
    """Собирает метрики и запускает CAIRN-анализ.

    Автоматически выбирает источник данных:
    - Prometheus (если cAdvisor видит отдельные контейнеры)
    - docker stats (fallback для Windows/WSL2)
    """
    target_container = SERVICE_MAP.get(target_service, target_service)

    if _prometheus_has_per_container_data():
        # ── Путь 1: Prometheus ────────────────────────────────────────────
        print(f"\n[->] Сбор данных из Prometheus ({PROMETHEUS_URL})...")
        try:
            from cairn.connectors.csv_file import YAMLTopologyConnector
            from cairn.perception import HypergraphBuilder
        except ImportError as e:
            print(f"[!!] Ошибка импорта CAIRN: {e}"); return

        now      = time.time()
        norm_end = now - anomaly_duration
        md = _fetch_prometheus_direct(
            PROMETHEUS_URL, list(PROM_QUERIES.keys()),
            norm_end - normal_duration, now, step=5,
        )
        if md is None:
            return
        print(f"[OK] Получено: {md.n_instances} контейнеров, {len(md.timestamps)} точек")

        topo_path = Path("docker/boutique_topology.yaml")
        if not topo_path.exists():
            _write_boutique_topology(topo_path)
        topo       = YAMLTopologyConnector(topo_path).fetch()
        hypergraph = HypergraphBuilder.from_topology_data(topo)
        _run_cairn_on_live_data(md, hypergraph, target_service, fault_type,
                                norm_end, now)

    else:
        # ── Путь 2: docker stats (Windows/WSL2 fallback) ──────────────────
        print(f"\n[~~] cAdvisor не видит отдельные контейнеры.")
        print(f"     Используем docker stats (CPU + Memory в реальном времени).")
        print(f"\n[->] Сбор нормального периода ({normal_duration}с)...")
        norm_data = _collect_docker_stats_period(normal_duration)
        print(f"[->] Сбор аномального периода ({anomaly_duration}с, инъекция активна)...")
        anom_data = _collect_docker_stats_period(anomaly_duration)
        print(f"\n[->] CAIRN-анализ (docker stats)...")
        _analyze_stats(norm_data, anom_data, target_container, fault_type)


def _run_cairn_on_live_data(md, hypergraph, target_service, fault_type,
                             norm_end, anom_end):
    """Запускает CAIRN-анализ на живых данных.

    Метод: сравниваем variance ratio метрик в нормальный и аномальный периоды.
    Pause/unpause создаёт высокую дисперсию в аномальный период — это
    физически корректный детектор для live-данных любого формата.
    """
    import numpy as np

    topo_names = set(hypergraph.instance_names)
    avail      = [n for n in md.instance_names if n in topo_names]
    if len(avail) < 2:
        print(f"[!!] Мало контейнеров в данных Prometheus: {avail}")
        print("     Подождите 1–2 минуты и повторите.")
        return

    print(f"     Контейнеры: {avail}")
    T = len(md.timestamps)

    # Разбиваем на нормальный и аномальный периоды по timestamp
    norm_mask = md.timestamps <= norm_end
    anom_mask = md.timestamps >  norm_end
    n_norm, n_anom = int(norm_mask.sum()), int(anom_mask.sum())
    print(f"     Нормальных точек: {n_norm}, аномальных: {n_anom}")

    if n_norm < 3 or n_anom < 3:
        split = T // 2
        norm_mask = np.zeros(T, dtype=bool); norm_mask[:split] = True
        anom_mask = ~norm_mask
        print(f"     [~~] Делим поровну: {split} / {T-split}")

    ranked_scores = {}
    for name in avail:
        ni   = md.instance_names.index(name)
        vals = np.nan_to_num(md.values[:, ni, :], nan=0.0)   # (T, F)
        scores = []
        for fi in range(vals.shape[1]):
            col      = vals[:, fi]
            mu_norm  = col[norm_mask].mean()
            mu_anom  = col[anom_mask].mean()
            std_norm = col[norm_mask].std()
            std_anom = col[anom_mask].std()

            # Pooled std (Cohen's d style) — не взрывается при std_norm → 0
            pooled = np.sqrt((std_norm**2 + std_anom**2) / 2.0 + 1e-9)

            # Effect size: насколько изменилось среднее относительно типичного разброса
            effect = abs(mu_anom - mu_norm) / pooled

            # Variance change: насколько выросла нестабильность (логарифм чтобы сгладить)
            var_change = np.log1p(std_anom / (std_norm + 1e-6))

            scores.append(effect + var_change)
        ranked_scores[name] = float(np.mean(scores))
    

    # ── Топологическая корректировка ─────────────────────────────────────
    try:
        called_by: dict[str, int] = {}
        callee_map: dict[str, list] = {}
        for edge in hypergraph.edges:
            if edge.edge_type == "call" and len(edge.members) >= 2:
                src = hypergraph.instance_names[edge.members[0]]
                dst = hypergraph.instance_names[edge.members[1]]
                callee_map.setdefault(src, []).append(dst)
                called_by[dst] = called_by.get(dst, 0) + 1

        adjusted = {}
        for name, score in ranked_scores.items():
            callees     = callee_map.get(name, [])
            cs          = [ranked_scores.get(c, 0.0) for c in callees if c in ranked_scores]
            cascade_avg = float(np.mean(cs)) if cs else 0.0
            n_callers   = called_by.get(name, 0)
            adjusted[name] = score / (1.0 + cascade_avg) / (1.0 + n_callers * 0.3)

        top_orig = max(ranked_scores, key=ranked_scores.get)
        top_adj  = max(adjusted, key=adjusted.get)
        if top_adj != top_orig:
            print(f"     [топология] скор: {top_orig} -> {top_adj}")
        ranked_scores = adjusted
    except Exception:
        pass

    ranked = sorted(ranked_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'─'*50}")
    print(f"{'Контейнер':<30} {'NLL':>10}  Ранг")
    print(f"{'─'*50}")
    for rank, (name, score) in enumerate(ranked, 1):
        marker = "  ← ROOT CANDIDATE" if rank == 1 else ""
        print(f"{name:<30} {score:>10.3f}  #{rank}{marker}")
    print(f"{'─'*50}")

    root_name, root_score = ranked[0]
    target_container = SERVICE_MAP.get(target_service, target_service)
    correct = root_name == target_container
    target_rank = next((r for r,(n,_) in enumerate(ranked,1) if n==target_container), None)

    print(f"\nПервопричина по CAIRN:    {root_name}  (score={root_score:.3f})")
    print(f"Инжектированный сбой:     {target_container} ({fault_type})")
    if correct:
        print("Результат: ✓ ВЕРНО — первопричина определена точно")
    elif target_rank and target_rank <= 3:
        print(f"Результат: ~ ЧАСТИЧНО — {target_container} на ранге #{target_rank}")
        print(f"           Каскадный эффект: {root_name} показал наибольшую аномалию")
        print(f"           как downstream-зависимость инжектированного сбоя")
    else:
        print(f"Результат: ✗ {target_container} на ранге #{target_rank}")
        print("           Попробуйте --target redis-cart или --type pause")

    print("\nТоп-3 каузальной цепочки:")
    for rank, (name, score) in enumerate(ranked[:3], 1):
        marker = " ← инжектировано" if name == target_container else ""
        print(f"  #{rank}  {name:<35} score={score:.3f}{marker}")

def main() -> None:
    parser = argparse.ArgumentParser(description="CAIRN Fault Injector — Online Boutique")
    parser.add_argument("--type",     default="cpu",
                        choices=["cpu", "pause", "memory"],
                        help="Тип инъекции")
    parser.add_argument("--target",   default="cartservice",
                        choices=list(SERVICE_MAP.keys()),
                        help="Целевой сервис")
    parser.add_argument("--duration", type=int, default=60,
                        help="Длительность инъекции (сек)")
    parser.add_argument("--warmup",   type=int, default=30,
                        help="Период нормальной работы до инъекции (сек)")
    parser.add_argument("--analyze",  action="store_true",
                        help="Запустить CAIRN-анализ после инъекции")
    parser.add_argument("--live",     action="store_true",
                        help="Live-режим: собирать docker stats параллельно с инъекцией")
    parser.add_argument("--check",    action="store_true",
                        help="Проверить что стек запущен")
    args = parser.parse_args()

    if args.check:
        check_stack()
        return

    # --live: docker stats режим — собираем норму ДО, аномалию ВО ВРЕМЯ инъекции
    if args.live or (args.analyze and not _prometheus_has_per_container_data()):
        print(f"╔══════════════════════════════════════════╗")
        print(f"║     CAIRN Live Analysis (docker stats)   ║")
        print(f"╠══════════════════════════════════════════╣")
        print(f"║  Тип:     {args.type:<30} ║")
        print(f"║  Цель:    {args.target:<30} ║")
        print(f"╚══════════════════════════════════════════╝")
        if not check_stack():
            sys.exit(1)

        container = SERVICE_MAP.get(args.target, args.target)

        print(f"\n[->] Нормальный период ({args.warmup}с)...")
        norm_data = _collect_docker_stats_period(args.warmup, interval=2.0)

        print(f"\n[->] Инъекция + аномальный период ({args.duration}с)...")
        if args.type == "cpu":
            inject_cpu(container, args.duration)
        elif args.type == "pause":
            pass  # pause сам по себе занимает duration
        elif args.type == "memory":
            inject_memory(container, args.duration)

        anom_data = _collect_docker_stats_period(args.duration, interval=2.0)

        print(f"\n[->] CAIRN-анализ...")
        target_container = SERVICE_MAP.get(args.target, args.target)
        _analyze_stats(norm_data, anom_data, target_container, args.type)
        return

    print(f"╔══════════════════════════════════════════╗")
    print(f"║     CAIRN Fault Injection Demo           ║")
    print(f"╠══════════════════════════════════════════╣")
    print(f"║  Тип:     {args.type:<30} ║")
    print(f"║  Цель:    {args.target:<30} ║")
    print(f"║  Длит.:   {args.duration:<30} ║")
    print(f"╚══════════════════════════════════════════╝")

    if not check_stack():
        sys.exit(1)

    container = SERVICE_MAP.get(args.target, args.target)

    # Прогрев — нормальная работа
    if args.warmup > 0:
        print(f"\n[→] Прогрев {args.warmup}с (нормальная работа)...")
        time.sleep(args.warmup)
        print(f"[OK] Прогрев завершён")

    # Инъекция сбоя
    t_start = time.time()
    if args.type == "cpu":
        inject_cpu(container, args.duration)
    elif args.type == "pause":
        inject_pause(container, args.duration)
    elif args.type == "memory":
        inject_memory(container, args.duration)

    if args.type != "pause":
        print(f"[→] Ждём {args.duration}с пока сбой проявится в метриках...")
        time.sleep(args.duration)

    elapsed = time.time() - t_start
    print(f"[OK] Инъекция завершена ({elapsed:.0f}с)")

    # Анализ
    if args.analyze:
        if not _prometheus_has_per_container_data():
            # docker stats: собираем нормальный период ДО и аномальный ВО ВРЕМЯ инъекции
            # Поэтому запускаем collect_and_analyze параллельно с инъекцией
            print(f"\n[~~] Windows/WSL2 режим: docker stats собирается параллельно с инъекцией.")
            print(f"     Запустите заново с флагом --analyze — "
                  f"сбор нормы ({args.warmup}с) и аномалии ({args.duration}с).")
            print(f"\n     Команда для полного теста:")
            print(f"     python scripts/inject_fault.py "
                  f"--type {args.type} --target {args.target} "
                  f"--duration {args.duration} --warmup {args.warmup} --analyze --live")
        else:
            collect_and_analyze(
                target_service=args.target,
                fault_type=args.type,
                normal_duration=args.warmup or 30,
                anomaly_duration=int(elapsed),
            )
    else:
        print(f"\n[→] Для запуска анализа:")
        print(f"    python scripts/inject_fault.py --type {args.type} "
              f"--target {args.target} --duration {args.duration} --analyze")


if __name__ == "__main__":
    main()