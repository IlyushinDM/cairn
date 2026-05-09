"""Контроллер CAIRN — связывает GUI-события с вычислительным ядром.

Паттерн: CAIRNMainWindow владеет CAIRNController.
Контроллер держит всё состояние модели/данных и эмитирует Qt-сигналы.
Main window только подключает сигналы к слотам виджетов.

Сигналы:
    data_loaded()                      — данные успешно загружены
    training_progress(ep, total, stage, losses)
    training_finished(history)
    analysis_complete(results)         — список dict с результатами
    error(title, message)
"""

from __future__ import annotations

import json
import torch
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal, Slot


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class AnalysisWorker(QThread):
    """QThread для фонового запуска анализа первопричин.

    Эмитирует finished(list[dict]) или error(str).
    """

    finished = Signal(list)   # list of result dicts
    error    = Signal(str)

    def __init__(self, controller: "CAIRNController", parent=None):
        super().__init__(parent)
        self._ctrl = controller

    def run(self) -> None:
        try:
            results = self._ctrl._run_analysis_core()
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# ModuleConfig — конфигурация включённых модулей
# ---------------------------------------------------------------------------

class ModuleConfig:
    """Отслеживает состояние модулей CAIRN.

    Активные модули: реально влияют на анализ.
    Опциональные: включаются для расширенного режима.
    """

    def __init__(self):
        # ── Активные (включены по умолчанию) ──────────────────────────
        self.graph_verifier = True    # топологическая корректировка скоров
        self.cf_module      = True    # контрфактический модуль VGAE
        self.alp_verifier   = True    # логическая верификация цепочки

        # ── Опциональные ───────────────────────────────────────────────
        self.ssm_branch    = False   # спектральная ветвь SSM (эксп.)
        self.drift_detect  = False   # обнаружение дрейфа распределения
        self.indep_loss    = False   # ограничение независимости L_нез

    def apply(self, key: str, enabled: bool) -> None:
        if hasattr(self, key):
            setattr(self, key, enabled)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in vars(self) if not k.startswith("_")}

    def summary(self) -> str:
        active = [k for k, v in self.to_dict().items() if v]
        return f"Активных модулей: {len(active)}: {', '.join(active)}"


# ---------------------------------------------------------------------------
# CAIRNController
# ---------------------------------------------------------------------------

class CAIRNController(QObject):
    """Контроллер: связывает GUI с ядром CAIRN.

    Параметры
    ----------
    config_path : Path | None
        Путь к YAML-конфигурации.
    """

    # ── Сигналы ──────────────────────────────────────────────────────────
    data_loaded          = Signal()              # данные загружены
    training_progress    = Signal(int, int, int, str)  # epoch, total, stage, name
    training_loss        = Signal(dict)          # {component: value}
    training_finished    = Signal(dict)          # history
    analysis_complete    = Signal(list)          # list[dict]
    error                = Signal(str, str)      # title, message

    def __init__(self, config_path: Optional[Path] = None, parent=None):
        super().__init__(parent)
        self._config_path   = config_path or Path("configs/demo.yaml")
        self._config        = None
        self._hypergraph    = None
        self._metric_data   = None
        self._is_live_mode  = False
        self._dataset       = None
        self._model         = None
        self._trainer       = None
        self._training_worker: Optional[QThread] = None
        self._analysis_worker: Optional[AnalysisWorker] = None
        self._last_results:  list[dict] = []
        self._last_chain     = None
        self._modules        = ModuleConfig()

        self._load_config()

    # ------------------------------------------------------------------
    # Конфигурация
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        try:
            from cairn.config import load_config
            if self._config_path.exists():
                self._config = load_config(self._config_path)
        except Exception:
            self._config = None

    @Slot(dict)
    def save_config(self, cfg_dict: dict) -> None:
        try:
            import yaml
            with open(self._config_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg_dict, f, allow_unicode=True)
            self._load_config()
        except Exception as e:
            self.error.emit("Ошибка сохранения конфигурации", str(e))

    @Slot(str, bool)
    def on_module_toggled(self, key: str, enabled: bool) -> None:
        """Реакция на чекбокс модуля — обновляет ModuleConfig и применяет к модели."""
        self._modules.apply(key, enabled)

        model = self._model
        if model is None:
            return

        # Применяем изменения к компонентам модели
        if key == "ssm_branch":
            enc = model.state_builder.metric_enc
            if hasattr(enc, "use_ssm"):
                enc.use_ssm = enabled  # type: ignore[assignment]

        elif key == "indep_loss":
            if self._trainer and hasattr(self._trainer, "loss_fn"):
                self._trainer.loss_fn.w.lambda_nez = 0.5 if enabled else 0.0

        elif key == "drift_detect":
            gmm = model.gmm
            object.__setattr__(gmm, "_drift_enabled", enabled)

        elif key == "graph_verifier":
            # Флаг читается в _run_analysis_core напрямую
            pass

        elif key == "cf_module":
            # Флаг читается в _run_analysis_core напрямую
            pass

        elif key == "alp_verifier":
            # Флаг читается в _run_analysis_core напрямую
            pass

    # ------------------------------------------------------------------
    # Загрузка данных
    # ------------------------------------------------------------------

    def load_live_data(self, metric_data, topo_data, hypergraph) -> None:
        """Загружает живые данные от LiveSystemConnector."""
        self._metric_data = metric_data
        self._topo_data   = topo_data
        self._hypergraph  = hypergraph
        self._is_live_mode = True

        # Авто-загрузка pre-trained модели если ещё не загружена
        if self._model is None:
            import torch
            from pathlib import Path as _Path
            model_path = _Path("data/sample/demo_model.pt")
            if model_path.exists():
                try:
                    self._load_checkpoint_silent(str(model_path))
                    self._log(f"Модель авто-загружена: {model_path}")
                except Exception as e:
                    self._log(f"Авто-загрузка модели не удалась: {e}")

        self.data_loaded.emit()

    def _load_checkpoint_silent(self, path: str) -> None:
        """Загружает чекпоинт без диалогов."""
        import torch
        from cairn.perception import StateBuilder
        from cairn.reasoning import ConditionalGMM, ConfoundedVGAE, CounterfactualModule
        from cairn.training import CAIRNModel

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        A    = ckpt.get("arch_config", {})
        D    = A.get("state_dim", 32);  CTX = A.get("context_dim", 8)
        F    = A.get("n_metrics", 4)

        model = CAIRNModel(
            state_builder=StateBuilder(
                n_metrics=F, log_vocab_size=300,
                state_dim=D, context_dim=CTX,
                d_met=A.get("d_met",16), d_log=A.get("d_log",8),
                d_tr=A.get("d_tr",8),   d_ssm=A.get("d_ssm",8),
                d_brk=A.get("d_brk",8), ssm_state_dim=A.get("ssm_state_dim",16),
                window=A.get("window",15),
            ),
            gmm=ConditionalGMM(D, CTX, A.get("n_components",7)),
            vgae=ConfoundedVGAE(D, A.get("n_confounders",2), A.get("confounder_dim",8)),
            cf_module=CounterfactualModule(D, A.get("n_conv_layers",1)),
        )
        state = ckpt.get("model_state_dict", ckpt.get("model_state", {}))
        model.load_state_dict(state, strict=False)
        model.eval()
        self._model = model
        # Обновляем статус в GUI
        self.model_status_changed.emit(True) if hasattr(self, "model_status_changed") else None

    @Slot()
    def load_demo_data(self, sample_dir: Path = Path("data/sample")) -> None:
        """Загружает демо-данные и строит гиперграф."""
        try:
            from cairn.connectors.csv_file import (
                CSVMetricConnector, FileLogConnector,
                JSONTraceConnector, YAMLTopologyConnector,
            )
            from cairn.perception import HypergraphBuilder

            import json as _json
            _labels_path = sample_dir / "labels.json"
            if _labels_path.exists():
                _lab    = _json.loads(_labels_path.read_text(encoding="utf-8"))
                BASE_TS = float(_lab.get("normal_period",  {}).get("start", 1_700_000_000.0))
                END_TS  = float(_lab.get("anomaly_period", {}).get("end",   BASE_TS + 299))
            else:
                BASE_TS, END_TS = 1_700_000_000.0, 1_700_000_299.0

            self._metric_data = CSVMetricConnector(sample_dir / "metrics.csv").fetch(BASE_TS, END_TS)
            self._log_data    = FileLogConnector(sample_dir / "logs.txt").fetch(BASE_TS, END_TS)
            self._trace_data  = JSONTraceConnector(sample_dir / "traces.json").fetch(BASE_TS, END_TS)
            self._topo_data   = YAMLTopologyConnector(sample_dir / "topology.yaml").fetch()
            self._hypergraph  = HypergraphBuilder.from_topology_data(self._topo_data)

            self.data_loaded.emit()
        except Exception as e:
            self.error.emit("Ошибка загрузки данных", str(e))

    def get_metric_data(self):
        return getattr(self, "_metric_data", None)

    def get_topology(self):
        return getattr(self, "_topo_data", None)

    # ------------------------------------------------------------------
    # Обучение
    # ------------------------------------------------------------------

    @Slot()
    def start_training(self) -> None:
        """Строит модель и запускает обучение в QThread."""
        if self._training_worker and self._training_worker.isRunning():
            return

        try:
            trainer, dataset = self._build_trainer()
        except Exception as e:
            self.error.emit("Ошибка инициализации", str(e))
            return

        from cairn.gui.widgets.training_tab import TrainingWorker
        from cairn.training import TrainerConfig

        tc = getattr(self._config, "training", None)
        cfg = TrainerConfig(
            pretrain_epochs=getattr(tc, "pretrain_epochs", 5),
            main_epochs=getattr(tc, "main_epochs", 5),
            finetune_epochs=getattr(tc, "finetune_epochs", 5),
            checkpoint_dir="checkpoints",
            device="cpu",
        )

        worker = TrainingWorker(trainer, dataset, cfg, parent=self)
        worker.progress.connect(self.training_progress)
        worker.loss_updated.connect(self.training_loss)
        worker.finished.connect(self.training_finished)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(lambda msg: self.error.emit("Ошибка обучения", msg))

        self._training_worker = worker
        self._trainer = trainer
        self._model   = trainer.model
        self._dataset = dataset

        worker.start()

    @Slot()
    def stop_training(self) -> None:
        if self._training_worker and self._training_worker.isRunning():
            from cairn.gui.widgets.training_tab import TrainingWorker as _TW
            if isinstance(self._training_worker, _TW):
                self._training_worker.stop()
            self._training_worker.wait(3000)

    def _build_trainer(self):
        from cairn.training import (
            create_demo_dataset, CAIRNModel, CAIRNLoss, CAIRNTrainer,
        )
        from cairn.perception import StateBuilder
        from cairn.reasoning import ConditionalGMM, ConfoundedVGAE, CounterfactualModule

        sc_dir = getattr(self, "_demo_sc_dir", None) or "data/sample"
        dataset = create_demo_dataset(sc_dir, window_size=30, stride=10)
        if len(dataset) == 0:
            raise RuntimeError("Датасет пуст. Проверьте data/sample/.")

        mc  = getattr(self._config, "model", None)
        D   = getattr(mc, "state_dim",   64)
        CTX = getattr(mc, "context_dim", 16)
        F   = dataset[0].metric_data.shape[2]

        loss_fn = CAIRNLoss(adaptive=True)
        # Применяем текущее состояние модулей
        if not self._modules.indep_loss:
            loss_fn.w.lambda_nez = 0.0

        model = CAIRNModel(
            state_builder=StateBuilder(
                n_metrics=F, log_vocab_size=300,
                state_dim=D, context_dim=CTX,
                d_met=32, d_log=16, d_tr=16,
                d_ssm=16, d_brk=16, ssm_state_dim=32, window=15,
            ),
            gmm=ConditionalGMM(state_dim=D, context_dim=CTX, n_components=3),
            vgae=ConfoundedVGAE(state_dim=D, n_confounders=2, confounder_dim=16),
            cf_module=CounterfactualModule(state_dim=D, n_conv_layers=1),
        )

        # Применяем настройки SSM-ветви
        if not self._modules.ssm_branch:
            enc = model.state_builder.metric_enc
            if hasattr(enc, "use_ssm"):
                enc.use_ssm = False

        trainer = CAIRNTrainer(model, loss_fn, self._hypergraph)
        return trainer, dataset

    # ------------------------------------------------------------------
    # Анализ (асинхронный через AnalysisWorker)
    # ------------------------------------------------------------------

    @Slot()
    def start_analysis(self) -> None:
        """Запускает анализ в фоновом потоке."""
        if self._model is None:
            self.error.emit("Модель не готова",
                            "Сначала обучите модель или загрузите чекпоинт.")
            return

        # Если старый воркер ещё жив — ждём завершения (макс 3 сек)
        if self._analysis_worker is not None:
            if self._analysis_worker.isRunning():
                self._analysis_worker.quit()
                self._analysis_worker.wait(3000)
            self._analysis_worker = None

        # parent=None — управляем временем жизни вручную
        worker = AnalysisWorker(self, parent=None)
        worker.finished.connect(self._on_analysis_finished)
        worker.finished.connect(self._cleanup_analysis_worker)
        worker.error.connect(lambda msg: self.error.emit("Ошибка анализа", msg))
        self._analysis_worker = worker
        worker.start()

    @Slot()
    def _cleanup_analysis_worker(self) -> None:
        """Безопасно удаляет воркер после завершения."""
        if self._analysis_worker is not None:
            self._analysis_worker.deleteLater()
            self._analysis_worker = None

    def _run_analysis_core(self) -> list[dict]:
        """Ядро анализа — работает на живых данных или демо."""
        import torch
        import numpy as np
        from cairn.reasoning import CascadeFunnel

        model     = self._model
        hypergraph = self._hypergraph
        if model is None or hypergraph is None:
            raise RuntimeError("Модель или гиперграф не инициализированы.")

        is_live = getattr(self, "_is_live_mode", False)
        md      = getattr(self, "_metric_data", None)

        if is_live and md is not None and md.n_instances > 0:
            # ── Живые данные ─────────────────────────────────────────────
            W = 30
            T = len(md.timestamps)
            window = min(W, T)
            vals   = md.values[-window:, :, :]
            vals   = np.nan_to_num(vals, nan=0.0)

            # Дополняем до 4 метрик
            expected_F = getattr(model.state_builder, "n_metrics", 4)
            F_cur = vals.shape[2]
            if F_cur < expected_F:
                pad  = np.zeros((vals.shape[0], vals.shape[1], expected_F - F_cur))
                vals = np.concatenate([vals, pad], axis=2)
            elif F_cur > expected_F:
                vals = vals[:, :, :expected_F]

            # Нормализуем
            for fi in range(vals.shape[2]):
                col_max = vals[:, :, fi].max()
                if col_max > 1e-6:
                    vals[:, :, fi] /= col_max

            F = vals.shape[2]
            H_list, C_list = [], []
            with torch.no_grad():
                for ni in range(vals.shape[1]):
                    chunk = vals[:, ni, :]
                    if chunk.shape[0] < W:
                        pad   = np.zeros((W - chunk.shape[0], F))
                        chunk = np.vstack([pad, chunk])
                    m_t     = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)
                    log_ids = torch.zeros(1, 1, dtype=torch.long)
                    log_len = torch.ones(1, dtype=torch.long)
                    dummy_d = torch.zeros(1, 16, dtype=torch.float32)
                    H_i, C_i = model.state_builder(m_t, log_ids, log_len, dummy_d)
                    H_list.append(H_i)
                    C_list.append(C_i)

            H   = torch.cat(H_list, dim=0)
            C   = torch.cat(C_list, dim=0)
            nll = model.gmm.nll(H, C)

        else:
            # ── Демо-данные ───────────────────────────────────────────────
            from cairn.training.data_loader import create_demo_dataset
            sc_dir  = getattr(self, "_demo_sc_dir", None) or "data/sample"
            dataset = create_demo_dataset(sc_dir, window_size=30, stride=10)
            anom    = dataset.anomaly_subset()
            if len(anom) == 0:
                raise RuntimeError("Нет аномальных инцидентов в датасете.")
            incident = anom[0]
            with torch.no_grad():
                outputs = model(incident, hypergraph)
            H, C = outputs["H"], outputs["C"]
            nll  = model.gmm.nll(H, C)

        # ── Общий путь: ранжирование ──────────────────────────────────────
        N_inst = len(hypergraph.instance_names)
        funnel = CascadeFunnel(
            l0_top_k=N_inst, l1_top_k=N_inst, l2_top_k=N_inst
        )
        adj      = hypergraph.adjacency_matrix()
        adj_norm = adj / adj.sum(1, keepdim=True).clamp(min=1)

        # Модуль: cf_module — использовать CounterfactualModule или нет
        use_cf = self._modules.cf_module

        # NLL-ранжирование: GMM корректно оценивает аномальность каждого узла
        # Передаём cf_module только если он активен
        ranked = funnel.run(
            nll, H, adj_norm,
            model.cf_module if use_cf else None,
            model.gmm, C, hypergraph,
        )

        # Модуль: graph_verifier — топологическая корректировка скоров
        if self._modules.graph_verifier:
            try:
                import numpy as _np
                names     = hypergraph.instance_names
                ce_scores = dict(ranked)
                # Строим граф вызовов
                called_by:  dict[int, int]       = {}
                callee_map: dict[int, list[int]]  = {}
                for edge in hypergraph.edges:
                    if edge.edge_type == "call" and len(edge.members) >= 2:
                        src, dst = edge.members[0], edge.members[1]
                        callee_map.setdefault(src, []).append(dst)
                        called_by[dst] = called_by.get(dst, 0) + 1

                adjusted = {}
                for idx, score in ce_scores.items():
                    callees     = callee_map.get(idx, [])
                    cs          = [ce_scores.get(c, 0.0) for c in callees
                                   if c in ce_scores]
                    cascade_avg = float(_np.mean(cs)) if cs else 0.0
                    n_callers   = called_by.get(idx, 0)
                    adjusted[idx] = score / (1.0 + cascade_avg) / (
                        1.0 + n_callers * 0.5)

                ranked = sorted(adjusted.items(), key=lambda x: x[1], reverse=True)
            except Exception:
                pass  # если топология недоступна — используем исходный ранг

        # Строим объяснение
        from cairn.explanation import EvidenceChainBuilder, TemplateTextGenerator, ALPVerifier
        names = hypergraph.instance_names
        nll_scores = {i: nll[i].item() for i in range(H.shape[0])}
        ce_scores  = dict(ranked)
        root_idx   = ranked[0][0] if ranked else 0

        # 1.2: Вычисляем доминантную метрику для каждого узла
        _incident_for_metrics = locals().get("incident", None)
        dominant_metrics = (
            self._compute_dominant_metrics(_incident_for_metrics, names)
            if _incident_for_metrics is not None
            else {}
        )

        # Передаём все оценки — builder строит путь через граф по убыванию NLL
        chain = EvidenceChainBuilder().build(
            root_cause=root_idx,
            causal_graph=self._hypergraph,
            ce_scores=ce_scores,
            nll_scores=nll_scores,
            anomaly_threshold=float(sorted(nll_scores.values())[len(nll_scores) // 3]),
        )
        # Устанавливаем тип сбоя и доминантную метрику на узлы пути
        for node in chain.path_nodes:
            node.dominant_metric = dominant_metrics.get(node.node_idx)
        if chain.path_nodes:
            chain.path_nodes[0].failure_type = getattr(self, "_demo_fault_hint", None)

        text = TemplateTextGenerator().generate(chain)
        # Улучшаем рекомендации если они generic
        if "Проверьте метрики и журналы" in text and chain.path_nodes:
            root = chain.path_nodes[0]
            ft   = getattr(root, "failure_type", "resource_anomaly") or "resource_anomaly"
            dm   = getattr(root, "dominant_metric", None)
            recs = {
                "cpu_exhaustion":    "Ограничьте CPU (cgroups/limits), проверьте CPU-интенсивные операции, рассмотрите горизонтальное масштабирование.",
                "memory_leak":       "Проверьте утечки памяти, перезапустите сервис, увеличьте memory limit или оптимизируйте кэш.",
                "latency_spike":     "Проверьте сетевые задержки, время ответа зависимостей, наличие long GC паузы.",
                "traffic_surge":     "Проверьте количество входящих запросов, рассмотрите rate limiting или auto-scaling.",
                "network_saturation":"Проверьте пропускную способность сети, наличие packet loss, DNS-резолюцию.",
                "resource_anomaly":  f"Доминирующая метрика: {dm or 'неизвестна'}. Проверьте нагрузку на сервис и его зависимости.",
            }
            rec = recs.get(ft, recs["resource_anomaly"])
            text = text.replace(
                "Рекомендация: Проверьте метрики и журналы сервиса для определения причины.",
                f"Рекомендация: {rec}"
            )

        # Относительные пороги: root должен быть выше медианы, а не абсолютного нуля
        # Это корректно для отрицательных NLL (GMM после обучения)
        nll_sorted = sorted(nll_scores.values())
        ce_sorted  = sorted(ce_scores.values())
        nll_median = nll_sorted[len(nll_sorted) // 2]   # медиана NLL
        ce_median  = ce_sorted[len(ce_sorted) // 2]     # медиана CE
        # Модуль: alp_verifier — логическая верификация
        if self._modules.alp_verifier:
            result = ALPVerifier(
                anomaly_threshold=nll_median,
                ce_threshold=ce_median,
            ).verify(chain, text)
        else:
            result = None

        self._last_chain  = chain
        self._last_alp    = result

        return [
            {
                "rank":       i + 1,
                "idx":        idx,
                "name":       names[idx] if idx < len(names) else f"node-{idx}",
                "ce":         round(ce, 4),
                "nll":        round(nll[idx].item(), 4),
                "fault_type": self._infer_fault_type(
                    getattr(self, "_demo_fault_hint", None),
                    dominant_metrics.get(idx)
                ) if i == 0 else "—",
                "confidence": max(0.0, 0.8 - i * 0.15),
            }
            for i, (idx, ce) in enumerate(ranked)
        ]

    def _infer_fault_type(self, hint: str | None, dominant_metric: str | None) -> str:
        """Определяет тип сбоя по подсказке и доминантной метрике."""
        if hint and hint != "unknown":
            return hint
        if dominant_metric:
            mapping = {
                "cpu":         "cpu_exhaustion",
                "cpu_pct":     "cpu_exhaustion",
                "memory":      "memory_leak",
                "memory_mb":   "memory_leak",
                "latency_ms":  "latency_spike",
                "rps":         "traffic_surge",
                "net_rx_kbps": "network_saturation",
                "net_tx_kbps": "network_saturation",
            }
            return mapping.get(dominant_metric, "resource_anomaly")
        return "resource_anomaly"


    def _compute_dominant_metrics(self, incident, instance_names: list[str]) -> dict[int, str]:
        """1.2: Вычисляет наиболее отклонившуюся метрику для каждого узла.

        Сравнивает первую и последнюю треть временного окна инцидента.
        Возвращает {node_idx: metric_name} для узлов с явным доминирующим сигналом.
        """
        import torch
        METRIC_NAMES = ["cpu", "memory", "latency_ms", "rps"]
        result: dict[int, str] = {}

        try:
            m = incident.metric_data  # (N, T, F)
            N, T, F = m.shape
            third = max(1, T // 3)

            # Базовый период — первая треть окна
            base = m[:, :third, :].mean(dim=1)          # (N, F)
            # Аномальный период — последняя треть
            anom = m[:, -third:, :].mean(dim=1)         # (N, F)

            # Относительное отклонение |Δ| / (|base| + ε)
            delta = (anom - base).abs() / (base.abs() + 1e-6)  # (N, F)

            for i in range(min(N, len(instance_names))):
                best_f = int(delta[i].argmax().item())
                best_delta = float(delta[i, best_f])
                # Считаем метрику доминантной если отклонение > 20%
                if best_delta > 0.20 and best_f < len(METRIC_NAMES):
                    result[i] = METRIC_NAMES[best_f]
        except Exception:
            pass  # не критично — dominant_metric остаётся null

        return result

    @Slot(list)
    def _on_analysis_finished(self, results: list[dict]) -> None:
        self._last_results = results
        self.analysis_complete.emit(results)

    # ------------------------------------------------------------------
    # Экспорт
    # ------------------------------------------------------------------

    def export_json(self, path: str | Path) -> None:
        """Экспортирует полный отчёт в JSON."""
        report = {
            "cairn_version": "0.1",
            "results": self._last_results,
            "chain":   self._last_chain.to_dict() if self._last_chain else {},
            "modules": self._modules.to_dict(),
            "config":  str(self._config_path),
        }
        Path(path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def export_graph_png(self, path: str | Path, graph_ax) -> None:
        """Сохраняет matplotlib-граф в PNG."""
        if graph_ax is None:
            return
        graph_ax.get_figure().savefig(
            str(path), dpi=150, bbox_inches="tight",
            facecolor="#161922", edgecolor="none",
        )

    # ------------------------------------------------------------------
    # Состояние
    # ------------------------------------------------------------------

    @property
    def has_data(self) -> bool:
        return self._hypergraph is not None

    @property
    def has_model(self) -> bool:
        return self._model is not None

    @property
    def hypergraph(self):
        return self._hypergraph

    @property
    def model(self):
        return self._model

    @property
    def last_chain(self):
        return self._last_chain

    @property
    def last_alp(self):
        return getattr(self, "_last_alp", None)
