"""Диалог сравнения режимов анализа (Ablation Study в GUI).

Запускает анализ в нескольких конфигурациях и показывает сравнение.
Помогает понять вклад каждого модуля в результат.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView,
    QLabel, QProgressBar, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)


class ComparisonWorker(QThread):
    """Фоновый поток для запуска анализа в разных конфигурациях."""

    result_ready = Signal(str, list)   # config_name, ranked_results
    finished_all = Signal()
    error        = Signal(str)

    def __init__(self, ctrl, configs: list[tuple[str, dict]]):
        super().__init__()
        self._ctrl    = ctrl
        self._configs = configs

    def run(self) -> None:
        import torch
        import numpy as np

        model      = self._ctrl._model
        hypergraph = self._ctrl._hypergraph
        if model is None or hypergraph is None:
            self.error.emit("Модель или гиперграф не загружены")
            return

        md = getattr(self._ctrl, "_metric_data", None)

        for cfg_name, cfg_flags in self._configs:
            try:
                ranked = self._run_single(
                    model, hypergraph, md, cfg_flags
                )
                self.result_ready.emit(cfg_name, ranked)
            except Exception as e:
                self.error.emit(f"{cfg_name}: {e}")

        self.finished_all.emit()

    def _run_single(
        self, model, hypergraph, md, flags: dict
    ) -> list[tuple[int, float, str]]:
        """Один прогон анализа с заданными флагами.

        Возвращает: [(node_idx, score, node_name), ...]
        """
        import torch, numpy as np
        from cairn.reasoning import CascadeFunnel

        names = hypergraph.instance_names

        # Строим H, C из живых данных или демо
        is_live = getattr(self._ctrl, "_is_live_mode", False)
        if is_live and md is not None and md.n_instances > 0:
            W = 30; F_exp = 4
            vals = np.nan_to_num(md.values[-min(W, len(md.timestamps)):, :, :], nan=0.0)
            if vals.shape[2] < F_exp:
                pad  = np.zeros((vals.shape[0], vals.shape[1], F_exp - vals.shape[2]))
                vals = np.concatenate([vals, pad], axis=2)
            elif vals.shape[2] > F_exp:
                vals = vals[:, :, :F_exp]
            for fi in range(vals.shape[2]):
                mx = vals[:, :, fi].max()
                if mx > 1e-6:
                    vals[:, :, fi] /= mx

            H_list, C_list = [], []
            with torch.no_grad():
                for ni in range(vals.shape[1]):
                    chunk = vals[:, ni, :]
                    if chunk.shape[0] < W:
                        chunk = np.vstack([np.zeros((W-chunk.shape[0], chunk.shape[1])), chunk])
                    m_t     = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)
                    log_ids = torch.zeros(1, 1, dtype=torch.long)
                    log_len = torch.ones(1,    dtype=torch.long)
                    dummy_d = torch.zeros(1, 16, dtype=torch.float32)
                    Hi, Ci  = model.state_builder(m_t, log_ids, log_len, dummy_d)
                    H_list.append(Hi); C_list.append(Ci)
            H   = torch.cat(H_list, dim=0)
            C   = torch.cat(C_list, dim=0)
            nll = model.gmm.nll(H, C)
        else:
            from cairn.training.data_loader import create_demo_dataset
            sc_dir  = getattr(self._ctrl, "_demo_sc_dir", None) or "data/sample"
            dataset = create_demo_dataset(sc_dir, window_size=30, stride=10)
            anom    = dataset.anomaly_subset()
            if len(anom) == 0:
                raise RuntimeError("Нет аномальных инцидентов")
            with torch.no_grad():
                outputs = model(anom[0], hypergraph)
            H, C = outputs["H"], outputs["C"]
            nll  = model.gmm.nll(H, C)

        N = len(names)
        adj      = hypergraph.adjacency_matrix()
        adj_norm = adj / adj.sum(1, keepdim=True).clamp(min=1)

        # cf_module
        cf_mod = model.cf_module if flags.get("cf_module", True) else None
        funnel = CascadeFunnel(l0_top_k=N, l1_top_k=N, l2_top_k=N)
        with torch.no_grad():
            ranked = funnel.run(nll, H, adj_norm, cf_mod, model.gmm, C, hypergraph)

        # graph_verifier
        if flags.get("graph_verifier", True):
            try:
                ce_scores  = dict(ranked)
                called_by: dict[int, int]  = {}
                callee_map: dict[int, list] = {}
                for edge in hypergraph.edges:
                    if edge.edge_type == "call" and len(edge.members) >= 2:
                        s, d = edge.members[0], edge.members[1]
                        callee_map.setdefault(s, []).append(d)
                        called_by[d] = called_by.get(d, 0) + 1
                adj_sc = {}
                for idx, score in ce_scores.items():
                    cs      = [ce_scores.get(c, 0.0) for c in callee_map.get(idx, [])
                               if c in ce_scores]
                    cascade = float(np.mean(cs)) if cs else 0.0
                    adj_sc[idx] = score / (1.0 + cascade) / (
                        1.0 + called_by.get(idx, 0) * 0.5)
                ranked = sorted(adj_sc.items(), key=lambda x: x[1], reverse=True)
            except Exception:
                pass

        return [
            (idx, float(score), names[idx] if idx < len(names) else f"node-{idx}")
            for idx, score in ranked
        ]


class ComparisonDialog(QDialog):
    """Диалог сравнения режимов анализа."""

    def __init__(self, ctrl, parent=None):
        super().__init__(parent)
        self._ctrl    = ctrl
        self._results: dict[str, list] = {}
        self.setWindowTitle("Сравнение режимов анализа")
        self.setMinimumSize(860, 500)
        self._build_ui()
        self._run_comparison()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Заголовок
        hdr = QLabel("Ablation Study — влияние модулей на результат анализа")
        hdr.setStyleSheet("font-size:13px; font-weight:600; color:#4a9eff;")
        layout.addWidget(hdr)

        hint = QLabel(
            "Каждый столбец — отдельный прогон с отключённым модулем. "
            "Выделены позиции где ранг изменился."
        )
        hint.setStyleSheet("color:#858585; font-size:11px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # Прогресс
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setFixedHeight(6)
        layout.addWidget(self._progress)

        # Таблица
        self._table = QTableWidget(0, 0)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(
            self._table.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._table)

        # Легенда
        legend_row = QHBoxLayout()
        for color, text in [
            ("#3ecf8e", "Ранг улучшился"),
            ("#f44747", "Ранг ухудшился"),
            ("#2d3348", "Без изменений"),
        ]:
            lbl = QLabel(f"■ {text}")
            lbl.setStyleSheet(f"color:{color}; font-size:10px;")
            legend_row.addWidget(lbl)
        legend_row.addStretch()
        layout.addLayout(legend_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _run_comparison(self) -> None:
        configs = [
            ("CAIRN (полная)",       {"graph_verifier": True,  "cf_module": True}),
            ("− graph_verifier",     {"graph_verifier": False, "cf_module": True}),
            ("− cf_module",          {"graph_verifier": True,  "cf_module": False}),
            ("− оба модуля",         {"graph_verifier": False, "cf_module": False}),
        ]

        self._worker = ComparisonWorker(self._ctrl, configs)
        self._worker.result_ready.connect(self._on_result)
        self._worker.finished_all.connect(self._on_all_done)
        self._worker.error.connect(
            lambda msg: self._table.setItem(
                0, 0, QTableWidgetItem(f"Ошибка: {msg}")))
        self._worker.start()

    def _on_result(self, cfg_name: str, ranked: list) -> None:
        self._results[cfg_name] = ranked
        self._rebuild_table()

    def _on_all_done(self) -> None:
        self._progress.setVisible(False)

    def _rebuild_table(self) -> None:
        if not self._results:
            return

        configs = list(self._results.keys())
        # Собираем все сервисы в порядке первого прогона
        first_ranked = list(self._results.values())[0]
        services = [name for _, _, name in first_ranked]

        n_rows = len(services)
        n_cols = 1 + len(configs)   # "Сервис" + по колонке на конфиг

        self._table.setRowCount(n_rows)
        self._table.setColumnCount(n_cols)

        headers = ["Сервис"] + configs
        self._table.setHorizontalHeaderLabels(headers)

        # Базовый ранг (первая конфигурация)
        base_ranks: dict[str, int] = {}
        if configs:
            for rank, (_, _, name) in enumerate(self._results[configs[0]], 1):
                base_ranks[name] = rank

        # Заполняем строки
        for row, svc_name in enumerate(services):
            # Колонка "Сервис"
            svc_item = QTableWidgetItem(svc_name)
            svc_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._table.setItem(row, 0, svc_item)

            # Колонки конфигураций
            for col, cfg_name in enumerate(configs, 1):
                ranked = self._results.get(cfg_name, [])
                # Найти ранг данного сервиса
                rank = next(
                    (r for r, (_, _, n) in enumerate(ranked, 1) if n == svc_name),
                    None
                )
                if rank is None:
                    item = QTableWidgetItem("—")
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self._table.setItem(row, col, item)
                    continue

                score = ranked[rank-1][1]
                base  = base_ranks.get(svc_name, rank)

                text = f"#{rank}  ({score:.3f})"
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)

                # Цвет по изменению ранга
                if col == 1:
                    # Базовая конфигурация
                    if rank == 1:
                        item.setForeground(QColor("#4a9eff"))
                        font = QFont(); font.setBold(True)
                        item.setFont(font)
                elif rank < base:
                    item.setForeground(QColor("#3ecf8e"))  # лучше
                elif rank > base:
                    item.setForeground(QColor("#f44747"))  # хуже
                    item.setBackground(QColor("#2d1a1a"))

                self._table.setItem(row, col, item)
