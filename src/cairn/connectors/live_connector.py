"""LiveSystemConnector – универсальный коннектор живых систем для CAIRN.

Читает конфиг-файл формата YAML и создаёт нужные коннекторы.
CAIRN не знает о конкретных системах – только об этом интерфейсе.

Пример использования:
    conn = LiveSystemConnector.from_config("configs/connectors/online_boutique.yaml")
    topology = conn.fetch_topology()
    metrics  = conn.fetch_metrics(start_time, end_time)

Схема конфига (configs/connectors/<system>.yaml):
    system:
      name: "Online Boutique"
      description: "Google microservices demo"

    topology:
      source: yaml_file
      path: docker/boutique_topology.yaml

    metrics:
      source: prometheus          # или: csv_file, json_file
      url: http://localhost:9090
      step: 15s
      id_label: id               # label из которого берём имя экземпляра
      id_mapping: docker_ps      # docker_ps | static | label
      metric_queries:
        cpu:       'irate(container_cpu_usage_seconds_total{id=~"/docker/.+"}[60s])'
        memory_mb: 'container_memory_usage_bytes{id=~"/docker/.+"} / 1048576'
        net_rx_kbps: 'irate(container_network_receive_bytes_total{id=~"/docker/.+"}[60s]) / 1024'
        net_tx_kbps: 'irate(container_network_transmit_bytes_total{id=~"/docker/.+"}[60s]) / 1024'

    # Только для csv_file:
    # metrics:
    #   source: csv_file
    #   path: data/my_system/metrics.csv
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from cairn.connectors.base import (
    ConnectorConfigError,
    MetricData,
    TopologyData,
)


def _write_boutique_topology_yaml(path: "Path") -> None:
    """Автоматически создаёт topology.yaml для Online Boutique."""
    yaml_content = """instances:
  - {name: cairn-frontend,              service: frontend,              host: docker-host, cpu_limit: 2.0, memory_limit: 256, version: "v0.10.1"}
  - {name: cairn-cartservice,           service: cartservice,           host: docker-host, cpu_limit: 2.0, memory_limit: 256, version: "v0.10.1"}
  - {name: cairn-productcatalog,        service: productcatalogservice, host: docker-host, cpu_limit: 2.0, memory_limit: 256, version: "v0.10.1"}
  - {name: cairn-checkoutservice,       service: checkoutservice,       host: docker-host, cpu_limit: 2.0, memory_limit: 256, version: "v0.10.1"}
  - {name: cairn-paymentservice,        service: paymentservice,        host: docker-host, cpu_limit: 2.0, memory_limit: 256, version: "v0.10.1"}
  - {name: cairn-shippingservice,       service: shippingservice,       host: docker-host, cpu_limit: 2.0, memory_limit: 256, version: "v0.10.1"}
  - {name: cairn-currencyservice,       service: currencyservice,       host: docker-host, cpu_limit: 2.0, memory_limit: 256, version: "v0.10.1"}
  - {name: cairn-recommendationservice, service: recommendationservice, host: docker-host, cpu_limit: 2.0, memory_limit: 256, version: "v0.10.1"}
  - {name: cairn-redis,                 service: redis-cart,            host: docker-host, cpu_limit: 1.0, memory_limit: 128, version: "7.2"}
call_edges:
  - [cairn-frontend, cairn-cartservice]
  - [cairn-frontend, cairn-productcatalog]
  - [cairn-frontend, cairn-currencyservice]
  - [cairn-frontend, cairn-recommendationservice]
  - [cairn-frontend, cairn-shippingservice]
  - [cairn-frontend, cairn-checkoutservice]
  - [cairn-checkoutservice, cairn-cartservice]
  - [cairn-checkoutservice, cairn-paymentservice]
  - [cairn-checkoutservice, cairn-shippingservice]
  - [cairn-checkoutservice, cairn-currencyservice]
  - [cairn-checkoutservice, cairn-productcatalog]
  - [cairn-cartservice, cairn-redis]
  - [cairn-recommendationservice, cairn-productcatalog]
colocation_groups: []
load_balancer_groups: []
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_content, encoding="utf-8")



class LiveSystemConnector:
    """Универсальный коннектор живой системы.

    Читает конфиг YAML и делегирует работу
    специализированным коннекторам (Prometheus, CSV и т.д.).
    """

    def __init__(self, config_path: str | Path) -> None:
        self._config_path = Path(config_path)
        if not self._config_path.exists():
            raise ConnectorConfigError(
                f"Конфиг коннектора не найден: {config_path}\n"
                f"Создайте файл или выберите существующий."
            )
        with self._config_path.open(encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f)

        self._system_name = self._cfg.get("system", {}).get("name", "Unknown System")
        self._metric_source = self._cfg.get("metrics", {}).get("source", "csv_file")

    @classmethod
    def from_config(cls, config_path: str | Path) -> "LiveSystemConnector":
        """Фабричный метод – создаёт коннектор из файла конфига."""
        return cls(config_path)

    @property
    def system_name(self) -> str:
        return self._system_name

    # ── Топология ─────────────────────────────────────────────────────────────

    def fetch_topology(self) -> TopologyData:
        """Загружает топологию согласно конфигу."""
        topo_cfg = self._cfg.get("topology", {})
        source   = topo_cfg.get("source", "yaml_file")

        if source == "yaml_file":
            from cairn.connectors.csv_file import YAMLTopologyConnector
            topo_path = Path(topo_cfg["path"])
            if not topo_path.is_absolute():
                # Пробуем относительно CWD (корень проекта) – первый приоритет
                from_cwd = Path.cwd() / topo_path
                # Затем относительно корня проекта (3 уровня от конфига)
                from_cfg = self._config_path.parent.parent.parent / topo_path
                if from_cwd.exists():
                    topo_path = from_cwd
                elif from_cfg.exists():
                    topo_path = from_cfg
                else:
                    # Последняя попытка – сгенерировать топологию автоматически
                    topo_path.parent.mkdir(parents=True, exist_ok=True)
                    _write_boutique_topology_yaml(topo_path)
            return YAMLTopologyConnector(topo_path).fetch()

        raise ConnectorConfigError(f"Неизвестный источник топологии: {source}")

    # ── Метрики ───────────────────────────────────────────────────────────────

    def fetch_metrics(
        self,
        start_time: float,
        end_time: float,
    ) -> MetricData:
        """Загружает метрики согласно конфигу."""
        if self._metric_source == "prometheus":
            return self._fetch_prometheus(start_time, end_time)
        elif self._metric_source == "csv_file":
            return self._fetch_csv(start_time, end_time)
        elif self._metric_source == "docker_stats":
            return self._fetch_docker_stats()
        else:
            raise ConnectorConfigError(
                f"Неизвестный источник метрик: {self._metric_source}\n"
                f"Поддерживаются: prometheus, csv_file, docker_stats"
            )

    def is_available(self) -> tuple[bool, str]:
        """Проверяет доступность системы. Возвращает (ok, message)."""
        if self._metric_source == "prometheus":
            return self._check_prometheus()
        elif self._metric_source == "csv_file":
            metrics_cfg = self._cfg.get("metrics", {})
            path = Path(metrics_cfg.get("path", ""))
            if path.exists():
                return True, f"CSV: {path}"
            return False, f"Файл не найден: {path}"
        elif self._metric_source == "docker_stats":
            return self._check_docker()
        return False, "Неизвестный источник"

    def _check_docker(self) -> tuple[bool, str]:
        try:
            result = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=3
            )
            names = [n for n in result.stdout.strip().split("\n") if n]
            inst_filter = self._cfg.get("metrics", {}).get("instance_filter", [])
            found = [n for n in names if n in inst_filter] if inst_filter else names
            if found:
                return True, f"Docker: {len(found)} контейнеров найдено"
            return False, "Docker: контейнеры не запущены"
        except Exception as e:
            return False, f"Docker недоступен: {e}"

    # ── Docker Stats ──────────────────────────────────────────────────────────

    def _fetch_docker_stats(self) -> MetricData:
        """Собирает метрики через docker stats.

        Делает несколько снимков за window_sec секунд и строит временной ряд.
        Метрики: cpu_pct, memory_mb – доступны на всех платформах.
        """
        metrics_cfg   = self._cfg.get("metrics", {})
        window_sec    = float(metrics_cfg.get("window_sec", 60))
        interval_sec  = float(metrics_cfg.get("interval_sec", 2.0))
        inst_filter   = set(metrics_cfg.get("instance_filter", []))

        snapshots: list[tuple[float, dict[str, dict[str, float]]]] = []
        t_end = time.time() + window_sec

        while time.time() < t_end:
            ts = time.time()
            try:
                result = subprocess.run(
                    ["docker", "stats", "--no-stream", "--format",
                     "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"],
                    capture_output=True, text=True, timeout=5
                )
                snap: dict[str, dict[str, float]] = {}
                for line in result.stdout.strip().split("\n"):
                    parts = line.split("\t")
                    if len(parts) < 3:
                        continue
                    name = parts[0].strip()
                    if inst_filter and name not in inst_filter:
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
                    snap[name] = {"cpu_pct": cpu, "memory_mb": mem}
                if snap:
                    snapshots.append((ts, snap))
            except Exception:
                pass
            time.sleep(interval_sec)

        if not snapshots:
            return MetricData(np.array([]), np.zeros((0,0,0)), [], [])

        metric_names = ["cpu_pct", "memory_mb"]
        # Берём все экземпляры из первого снимка
        inst_names = sorted(snapshots[0][1].keys())
        timestamps = np.array([ts for ts, _ in snapshots], dtype=np.float64)
        T = len(timestamps)
        values = np.full((T, len(inst_names), len(metric_names)), np.nan)

        for ti, (_, snap) in enumerate(snapshots):
            for ni, inst in enumerate(inst_names):
                if inst in snap:
                    for mi, met in enumerate(metric_names):
                        values[ti, ni, mi] = snap[inst].get(met, np.nan)

        return MetricData(timestamps, values, inst_names, metric_names)

    # ── Prometheus ────────────────────────────────────────────────────────────

    def _check_prometheus(self) -> tuple[bool, str]:
        try:
            import requests
            url = self._cfg["metrics"]["url"]
            r   = requests.get(f"{url}/api/v1/query",
                               params={"query": "up"}, timeout=3)
            r.raise_for_status()
            return True, f"Prometheus: {url}"
        except Exception as e:
            return False, f"Prometheus недоступен: {e}"

    def _build_id_mapping(self) -> dict[str, str]:
        """Строит маппинг prometheus_id → instance_name."""
        metrics_cfg = self._cfg.get("metrics", {})
        mapping_type = metrics_cfg.get("id_mapping", "docker_ps")

        if mapping_type == "docker_ps":
            # Маппинг через docker ps (для контейнеризованных систем)
            result = subprocess.run(
                ["docker", "ps", "--format", "{{.ID}}\t{{.Names}}"],
                capture_output=True, text=True, timeout=5
            )
            id_map = {}
            for line in result.stdout.strip().split("\n"):
                if "\t" in line:
                    short_id, name = line.split("\t", 1)
                    id_map[short_id.strip()] = name.strip()
            return id_map

        elif mapping_type == "static":
            # Статический маппинг из конфига
            return metrics_cfg.get("id_map", {})

        elif mapping_type == "label":
            # Имя берётся напрямую из указанного label
            return {}  # без маппинга, используем label как есть

        return {}

    def _fetch_prometheus(self, start_time: float, end_time: float) -> MetricData:
        """Запрашивает метрики из Prometheus по PromQL-запросам из конфига."""
        import requests

        metrics_cfg    = self._cfg.get("metrics", {})
        url            = metrics_cfg["url"]
        step           = metrics_cfg.get("step", "15s")
        # step может быть "15s" или числом секунд
        step_sec = int(step.replace("s","").replace("m","")) if isinstance(step, str) else int(step)
        id_label       = metrics_cfg.get("id_label", "id")
        mapping_type   = metrics_cfg.get("id_mapping", "docker_ps")
        queries        = metrics_cfg.get("metric_queries", {})

        if not queries:
            raise ConnectorConfigError("Не заданы metric_queries в конфиге")

        # Строим маппинг id → имя
        id_to_name = self._build_id_mapping()

        all_data: dict[str, dict[str, list]] = {}
        metric_names = list(queries.keys())

        for metric_key, query in queries.items():
            try:
                resp = requests.get(
                    f"{url}/api/v1/query_range",
                    params={"query": query, "start": start_time,
                            "end": end_time, "step": step_sec},
                    timeout=15,
                )
                resp.raise_for_status()
                result = resp.json().get("data", {}).get("result", [])
            except Exception as e:
                raise ConnectorConfigError(
                    f"Ошибка запроса '{metric_key}': {e}"
                ) from e

            for series in result:
                labels   = series.get("metric", {})
                id_field = labels.get(id_label, "")
                inst     = None

                if mapping_type == "docker_ps" and "/docker/" in id_field:
                    short = id_field.split("/docker/")[-1][:12]
                    inst  = id_to_name.get(short)
                elif mapping_type == "label":
                    inst  = id_field or labels.get("name") or labels.get("container")
                elif mapping_type == "static":
                    inst  = id_to_name.get(id_field, id_field)

                if not inst:
                    continue

                all_data.setdefault(inst, {})
                all_data[inst][metric_key] = [
                    (float(ts), float(val))
                    for ts, val in series.get("values", [])
                    if val not in ("NaN", "Inf", "-Inf")
                ]

        # Применяем instance_filter из конфига
        instance_filter = metrics_cfg.get("instance_filter", None)
        if instance_filter and all_data:
            all_data = {k: v for k, v in all_data.items() if k in instance_filter}

        if not all_data:
            return MetricData(
                np.array([], dtype=np.float64),
                np.zeros((0, 0, len(metric_names))),
                [], metric_names,
            )

        inst_names   = sorted(all_data.keys())
        first_inst   = inst_names[0]
        first_metric = next((m for m in metric_names if m in all_data[first_inst]), None)
        if not first_metric:
            return MetricData(np.array([]), np.zeros((0, 0, 0)), [], metric_names)

        ts_list    = [ts for ts, _ in all_data[first_inst][first_metric]]
        timestamps = np.array(ts_list, dtype=np.float64)
        T          = len(timestamps)
        ts_idx     = {ts: i for i, ts in enumerate(ts_list)}

        values = np.full((T, len(inst_names), len(metric_names)), np.nan)
        for ni, inst in enumerate(inst_names):
            for mi, met in enumerate(metric_names):
                for ts, val in all_data.get(inst, {}).get(met, []):
                    if ts in ts_idx:
                        values[ts_idx[ts], ni, mi] = val

        return MetricData(timestamps, values, inst_names, metric_names)

    # ── CSV ───────────────────────────────────────────────────────────────────

    def _fetch_csv(self, start_time: float, end_time: float) -> MetricData:
        """Читает метрики из CSV-файла."""
        from cairn.connectors.csv_file import CSVMetricConnector
        metrics_cfg = self._cfg.get("metrics", {})
        path        = Path(metrics_cfg["path"])
        if not path.is_absolute():
            path = self._config_path.parent.parent / path
        return CSVMetricConnector(path).fetch(start_time, end_time)


# ── Реестр доступных конфигов ─────────────────────────────────────────────────

def discover_connector_configs(configs_dir: str | Path = "configs/connectors") -> list[Path]:
    """Возвращает список доступных конфиг-файлов коннекторов."""
    d = Path(configs_dir)
    if not d.exists():
        return []
    return sorted(d.glob("*.yaml"))
