"""Диагностика контрфактического модуля CAIRN – 6 гипотез.

Последовательно проверяет все возможные причины некорректного ранжирования:
  H1: Прототип слишком близок к аномальному состоянию
  H2: Гиперграф слишком связный – вмешательство размазывается
  H3: CE агрегируется mean() – эффект разбавляется
  H4: Свёртка перезатирает вмешательство
  H5: L_CE не обучила модель различать причины
  H6: Данные слишком слабые / аномалия не выражена

Использование:
    python scripts/diagnose.py --scenario 1 --model data/sample/demo_model.pt
    python scripts/diagnose.py --scenario 2 --epochs 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

# Принудительно UTF-8 для Windows-терминалов
import io
import sys
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

GRN = "\033[92m"; RED = "\033[91m"; YLW = "\033[93m"
RST = "\033[0m";  BLD = "\033[1m";  CYN = "\033[96m"

def ok(msg):   print(f"  [OK] {msg}")
def fail(msg): print(f"  [!!] {msg}")
def warn(msg): print(f"  [~~] {msg}")
def hdr(msg):  print(f"\n{BLD}{CYN}{'='*55}\n  {msg}\n{'='*55}{RST}")
def sub(msg):  print(f"\n{BLD}  {msg}{RST}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="1", choices=["1","2","3"])
    parser.add_argument("--model",    default=None)
    parser.add_argument("--epochs",   type=int, default=None)
    args = parser.parse_args()

    sc_dir = Path(f"data/sample/scenario_{args.scenario}")
    labels = __import__("json").loads((sc_dir / "labels.json").read_text(encoding="utf-8"))
    true_root  = labels["root_cause"]["instance"]
    fault      = labels["root_cause"]["type"]

    print(f"\n{BLD}CAIRN Deep Diagnostics – Сценарий {args.scenario}: {labels['scenario']}{RST}")
    print(f"  Истинная первопричина: {BLD}{true_root}{RST}  [{fault}]")

    # ── Загрузка данных ───────────────────────────────────────────────────────
    from cairn.connectors.csv_file import (
        CSVMetricConnector, FileLogConnector, YAMLTopologyConnector,
    )
    from cairn.perception import StateBuilder, DrainTokenizer, HypergraphBuilder

    metric_conn = CSVMetricConnector(sc_dir / "metrics.csv")
    topo = YAMLTopologyConnector(sc_dir / "topology.yaml").fetch()
    hg   = HypergraphBuilder.from_topology_data(topo)

    model_path = Path(args.model) if args.model else Path("data/sample/demo_model.pt")

    # ── Загрузка модели ───────────────────────────────────────────────────────
    from cairn.training import CAIRNModel, CAIRNLoss, CAIRNTrainer, TrainerConfig, create_demo_dataset
    from cairn.reasoning import ConditionalGMM, ConfoundedVGAE, CounterfactualModule

    # Читаем arch_config из чекпоинта (уровни: arch_config → shape-inference → defaults)
    D, CTX, F = 32, 8, 4
    n_components   = 3
    n_confounders  = 2
    confounder_dim = 8
    context_raw_dim = CTX

    if model_path.exists():
        _ckpt = torch.load(str(model_path), map_location="cpu", weights_only=True)
        _key  = "model_state" if "model_state" in _ckpt else "model_state_dict"
        _st   = _ckpt.get(_key, {})

        # Уровень 1: arch_config в чекпоинте
        if "arch_config" in _ckpt:
            A = _ckpt["arch_config"]
            D              = A.get("state_dim",      D)
            CTX            = A.get("context_dim",    CTX)
            F              = A.get("n_metrics",      F)
            n_components   = A.get("n_components",   n_components)
            n_confounders  = A.get("n_confounders",  n_confounders)
            confounder_dim = A.get("confounder_dim", confounder_dim)
        # Уровень 2: shape-inference из весов
        elif _st:
            _w = _st.get("gmm.mlp_omega.2.bias")
            if _w is not None:
                n_components = _w.shape[0]
            _w2 = _st.get("vgae.confounder_mod.z_mu.0.weight")
            if _w2 is not None:
                confounder_dim = _w2.shape[0]
            n_confounders = sum(
                1 for k in _st
                if k.startswith("vgae.confounder_mod.z_mu.") and k.endswith(".weight")
            ) or n_confounders

        # context_raw_dim всегда из весов (надёжнее)
        _w_ctx = _st.get("state_builder.context_builder.proj.0.weight")
        if _w_ctx is not None:
            context_raw_dim = _w_ctx.shape[1]

    model = CAIRNModel(
        state_builder=StateBuilder(
            n_metrics=F, log_vocab_size=300, state_dim=D, context_dim=CTX,
            d_met=16, d_log=8, d_tr=8, d_ssm=8, d_brk=8, ssm_state_dim=16,
            window=15, context_raw_dim=context_raw_dim,
        ),
        gmm=ConditionalGMM(state_dim=D, context_dim=CTX, n_components=n_components),
        vgae=ConfoundedVGAE(state_dim=D, n_confounders=n_confounders, confounder_dim=confounder_dim),
        cf_module=CounterfactualModule(state_dim=D, n_conv_layers=1),
    )

    model_loaded = False
    if model_path.exists():
        state = torch.load(str(model_path), map_location="cpu", weights_only=True)
        key   = "model_state" if "model_state" in state else "model_state_dict"
        if key in state:
            model.load_state_dict(state[key], strict=False)
            ok(f"Модель загружена из {model_path}")
            model_loaded = True
        else:
            warn(f"Неверный формат чекпоинта: {list(state.keys())}")

    if not model_loaded:
        n_ep = args.epochs or 5
        warn(f"Обучаем {n_ep} эп. на лету...")
        ds  = create_demo_dataset(sc_dir, window_size=30, stride=15)
        cfg = TrainerConfig(
            pretrain_epochs=n_ep, main_epochs=n_ep, finetune_epochs=n_ep,
            lr=1e-4, freeze_epochs=0, patience=999, log_every=999,
            device="cpu", checkpoint_dir="/tmp/cairn_diag", save_every=999,
        )
        CAIRNTrainer(model, CAIRNLoss(), hg, cfg).train(ds)

    model.eval()

    # ── Подготовка тензоров ───────────────────────────────────────────────────
    def load_tensors(ts_start, ts_end):
        md  = metric_conn.fetch(ts_start, ts_end)
        ld  = FileLogConnector(sc_dir / "logs.txt").fetch(ts_start, ts_end)
        tok = DrainTokenizer(); tok.fit_transform(ld.messages)
        L = 10
        log_ids, log_lens = [], []
        for name in md.instance_names:
            msgs = ld.filter_instance(name).messages
            ids  = [tok.transform_one(m) for m in msgs[:L]] or [0]
            log_lens.append(len(ids))
            log_ids.append((ids + [0]*L)[:L])
        m_t  = torch.nan_to_num(torch.tensor(md.values.transpose(1,0,2), dtype=torch.float32))
        li_t = torch.tensor(log_ids, dtype=torch.long)
        ll_t = torch.tensor(log_lens, dtype=torch.long)
        d_t  = torch.zeros(md.n_instances, dtype=torch.long)
        return m_t, li_t, ll_t, d_t, md.instance_names

    # Читаем временные периоды из labels.json – не хардкодим константы
    _labels_path = sc_dir / "labels.json"
    if _labels_path.exists():
        _lab = json.loads(_labels_path.read_text(encoding="utf-8"))
        BASE       = float(_lab.get("normal_period",  {}).get("start", 1_700_000_000.0))
        normal_end = float(_lab.get("normal_period",  {}).get("end",   BASE + 199))
        anom_start = float(_lab.get("anomaly_period", {}).get("start", BASE + 200))
        anom_end   = float(_lab.get("anomaly_period", {}).get("end",   BASE + 299))
    else:
        BASE = 1_700_000_000.0
        normal_end = BASE + 199
        anom_start = BASE + 200
        anom_end   = BASE + 299

    m_n, li_n, ll_n, d_n, names = load_tensors(BASE, normal_end)
    m_a, li_a, ll_a, d_a, _    = load_tensors(anom_start, anom_end)

    with torch.no_grad():
        H_n, C_n = model.state_builder(m_n, li_n, d_n, log_lengths=ll_n)
        H_a, C_a = model.state_builder(m_a, li_a, d_a, log_lengths=ll_a)

    root_idx = names.index(true_root)
    N = len(names)

    # --- ИСПРАВЛЕНИЕ ПОРЯДКА ИНДЕКСОВ ---
    # H построен в алфавитном порядке (names), но inc_t – в порядке hg.instance_names
    # Создаём перестановки для выравнивания
    topo_names = hg.instance_names   # порядок topology.yaml
    # alpha→topo: H_alpha[i] должен быть на позиции alpha_to_topo[i] в H_topo
    try:
        topo_to_alpha = [names.index(n) for n in topo_names]  # topo_idx → alpha_idx
        alpha_to_topo = [topo_names.index(n) for n in names]  # alpha_idx → topo_idx
        import torch as _torch
        t2a = _torch.tensor(topo_to_alpha, dtype=_torch.long)
        a2t = _torch.tensor(alpha_to_topo, dtype=_torch.long)
        def to_topo(H_a_): return H_a_[t2a]   # alpha→topo: at topo[i], place alpha[t2a[i]]
        def to_alpha(H_t_): return H_t_[a2t]   # topo→alpha: at alpha[i], place topo[a2t[i]]
        def hg_fwd(H_a_, inc_, ew_):
            """Правильная свёртка: alpha→topo→conv→alpha."""
            return to_alpha(
                model.cf_module._hg_forward(to_topo(H_a_), inc_, ew_)
            )
        root_idx_topo = alpha_to_topo[root_idx]
    except ValueError:
        # Fallback если имена не совпадают
        def hg_fwd(H_a_, inc_, ew_): return model.cf_module._hg_forward(H_a_, inc_, ew_)
        root_idx_topo = root_idx
    C_ref = torch.zeros_like(C_a)  # референсный контекст

    # ══════════════════════════════════════════════════════════════════════════
    hdr("Ш А Г  1 – Данные: насколько выражена аномалия?")
    # ══════════════════════════════════════════════════════════════════════════

    md_n = metric_conn.fetch(BASE, normal_end)
    md_a = metric_conn.fetch(anom_start, anom_end)
    print()
    print(f"  {'Экземпляр':22s} {'Метрика':12s} {'Норма':>8s} {'Аномалия':>10s} {'Δ%':>8s}")
    print(f"  {'-'*70}")
    for mi, metric in enumerate(["cpu", "memory", "latency_ms", "rps"]):
        for ni, name in enumerate(names):
            v_n = float(np.nanmean(md_n.values[:, ni, mi]))
            v_a = float(np.nanmean(md_a.values[:, ni, mi]))
            pct = (v_a - v_n) / (abs(v_n) + 1e-6) * 100
            marker = " <- ROOT" if ni == root_idx else ""
            flag   = GRN if (ni == root_idx and abs(pct) > 30) else (YLW if abs(pct) < 5 else "")
            print(f"  {name:22s} {metric:12s} {v_n:8.3f} {v_a:10.3f} {pct:+8.1f}%{flag}{marker}{RST}")

    # ══════════════════════════════════════════════════════════════════════════
    hdr("Г И П О Т Е З А  1 – Прототип близок к аномальному состоянию?")
    # ══════════════════════════════════════════════════════════════════════════

    with torch.no_grad():
        # Правильный прототип = реальное нормальное состояние H_n
        protos = H_n   # (N, d) – реальные нормальные состояния
        dists  = torch.norm(H_a - protos, dim=1)

    sub("Расстояния ||h_аном - μ*(0)||:")
    for i, name in enumerate(names):
        marker = " <- PERVOPRICHINA" if i == root_idx else ""
        flag   = GRN if (i == root_idx and dists[i] > dists.mean() * 1.2) else ""
        print(f"  {name:22s}  ||h - μ*|| = {flag}{dists[i]:.4f}{RST}{marker}")

    root_dist_rank = int((dists > dists[root_idx]).sum()) + 1
    if root_dist_rank == 1:
        ok("H1 НЕ подтвердилась: первопричина имеет наибольшее расстояние до прототипа")
    else:
        fail(f"H1 ПОДТВЕРДИЛАСЬ: первопричина на ранге {root_dist_rank}/{N} по расстоянию до прототипа")
        warn("→ Аномалия слишком слабая или прототип некорректный")

    # ══════════════════════════════════════════════════════════════════════════
    hdr("Г И П О Т Е З А  2 – Гиперграф слишком связный?")
    # ══════════════════════════════════════════════════════════════════════════

    inc  = hg.incidence_matrix()  # (N, M)
    adj  = hg.adjacency_matrix()  # (N, N)
    degs = inc.sum(dim=1)         # степень каждого узла

    sub("Матрица смежности (1=связаны):")
    header = "  " + " " * 22
    for name in names:
        header += f"{name[:8]:>10s}"
    print(header)
    for i, name in enumerate(names):
        row = f"  {name:22s}"
        for j in range(N):
            row += f"{'■' if adj[i,j] > 0 else '·':>10s}"
        print(row)

    sub(f"Степени вершин (число гиперрёбер):")
    for i, name in enumerate(names):
        print(f"  {name:22s}  степень = {degs[i]:.0f}")
    print(f"  Всего гиперрёбер: {inc.shape[1]}, плотность adj: {adj.sum().item()/(N*N):.2f}")

    if adj.sum().item() > N * (N-1) * 0.7:
        fail("H2 ПОДТВЕРДИЛАСЬ: граф слишком плотный (>70% рёбер)")
    else:
        ok(f"H2 НЕ подтвердилась: граф имеет разумную плотность")

    # ══════════════════════════════════════════════════════════════════════════
    hdr("Г И П О Т Е З А  3 – NLL до/после вмешательства на первопричину")
    # ══════════════════════════════════════════════════════════════════════════

    with torch.no_grad():
        nll_before = model.gmm.nll(H_a, C_a)   # реальный контекст

        # Вмешательство на первопричину
        # H_n[root_idx] = реальное нормальное состояние первопричины
        # Это правильный прототип: «каким был узел до аномалии»
        proto_root = H_n[root_idx]  # реальная норма, не GMM-прототип
        mask = torch.zeros(N, 1)
        mask[root_idx] = 1.0
        H_cf = (1 - mask) * H_a + mask * proto_root.unsqueeze(0)
        inc_t = inc.to(H_a.device)
        ew_t  = hg.edge_weights().to(H_a.device)
        # NLL вычисляем БЕЗ свёртки – GMM обучена на чистых выходах энкодера
        nll_after  = model.gmm.nll(H_cf, C_a)   # до свёртки!
        # Distance-based CE в post-conv пространстве
        H_anom_conv = hg_fwd(H_a, inc_t, ew_t)
        H_cf_conv2  = hg_fwd(H_cf, inc_t, ew_t)
        H_norm_conv = hg_fwd(H_n, inc_t, ew_t)

    sub("NLL до и после вмешательства на первопричину:")
    print(f"  {'Экземпляр':22s} {'До':>10s} {'После':>10s} {'Δ':>10s}")
    print(f"  {'-'*55}")
    for i, name in enumerate(names):
        delta  = nll_before[i].item() - nll_after[i].item()
        marker = " <- ROOT" if i == root_idx else ""
        flag   = GRN if (i == root_idx and delta > 0.1) else (RED if (i == root_idx and delta < 0) else "")
        print(f"  {name:22s} {nll_before[i]:.4f} {nll_after[i]:>10.4f} {flag}{delta:>+10.4f}{RST}{marker}")

    total_ce = (nll_before.sum() - nll_after.sum()).item()
    root_delta = nll_before[root_idx].item() - nll_after[root_idx].item()
    dist_before_root = ((H_anom_conv[root_idx] - H_norm_conv[root_idx])**2).sum().item()
    dist_after_root  = ((H_cf_conv2[root_idx]  - H_norm_conv[root_idx])**2).sum().item()
    dist_ce_root = dist_before_root - dist_after_root
    print(f"\n  CE_NLL(первопричина) = {total_ce:.4f}")
    print(f"  ΔNLL(первопричина) = {root_delta:+.4f}")
    print(f"  CE_dist(первопричина) = {dist_ce_root:.4f}  (distance in post-conv space)")

    if root_delta > 0.05 or dist_ce_root > 0:
        ok("H3: вмешательство снижает NLL первопричины")
    else:
        fail(f"H3 ПОДТВЕРДИЛАСЬ: вмешательство не снижает NLL первопричины (Δ={root_delta:+.4f})")

    # ══════════════════════════════════════════════════════════════════════════
    hdr("Г И П О Т Е З А  4 – Свёртка перезатирает вмешательство?")
    # ══════════════════════════════════════════════════════════════════════════

    with torch.no_grad():
        H_cf_before = H_a.clone()
        H_cf_before[root_idx] = H_n[root_idx]  # реальная норма
        diff_before = torch.norm(H_cf_before[root_idx] - H_a[root_idx]).item()

        # H4 теперь менее критична – свёртка не используется для scoring
        H_cf_after     = hg_fwd(H_cf_before, inc_t, ew_t)
        H_orig_conv    = hg_fwd(H_a, inc_t, ew_t)
        diff_after     = torch.norm(H_cf_after[root_idx] - H_orig_conv[root_idx]).item()

    retention = diff_after / (diff_before + 1e-8)
    print(f"\n  Разница ДО свёртки:    {diff_before:.4f}")
    print(f"  Разница ПОСЛЕ свёртки: {diff_after:.4f}")
    print(f"  Коэффициент сохранения: {retention:.4f}")

    if retention < 0.1:
        fail(f"H4 ПОДТВЕРДИЛАСЬ: свёртка стирает вмешательство (retention={retention:.4f})")
        warn("→ Добавьте skip-connection или вычисляйте CE до свёртки")
    elif retention < 0.4:
        warn(f"H4 частично: свёртка ослабляет вмешательство (retention={retention:.4f})")
    else:
        ok(f"H4 НЕ подтвердилась: вмешательство сохраняется (retention={retention:.4f})")

    # ══════════════════════════════════════════════════════════════════════════
    hdr("Г И П О Т Е З А  5 – L_PE обучала модель?")
    # ══════════════════════════════════════════════════════════════════════════

    with torch.no_grad():
        # Вычисляем CE для всех кандидатов
        ces = {}
        for idx in range(N):
            p_i = H_n[idx]   # реальное нормальное состояние
            mask_i = torch.zeros(N, 1)
            mask_i[idx] = 1.0
            H_cf_i    = (1 - mask_i) * H_a + mask_i * p_i.unsqueeze(0)
            # NLL без свёртки – GMM не обучена на post-conv векторах
            nll_b = model.gmm.nll(H_a,   C_a).sum()
            nll_a = model.gmm.nll(H_cf_i, C_a).sum()
            ces[idx] = (nll_b - nll_a).item()

    ranked_by_ce = sorted(ces.items(), key=lambda x: -x[1])
    sub("Полное ранжирование по CE:")
    for rank, (idx, ce) in enumerate(ranked_by_ce, 1):
        marker = " <- PERVOPRICHINA" if idx == root_idx else ""
        flag   = GRN if idx == root_idx else ""
        print(f"  #{rank}  {names[idx]:22s}  CE={flag}{ce:+.4f}{RST}{marker}")

    ce_vals = list(ces.values())
    ce_spread = max(ce_vals) - min(ce_vals)
    root_rank  = [idx for idx, _ in ranked_by_ce].index(root_idx) + 1

    print(f"\n  Разброс CE: {ce_spread:.4f}")
    print(f"  CE(первопричина): {ces[root_idx]:+.4f}")
    print(f"  CE(максимальный): {max(ce_vals):+.4f}")

    if ce_spread < 0.1:
        fail(f"H5 ПОДТВЕРДИЛАСЬ: все CE одинаковые (spread={ce_spread:.4f}) – модель не обучена различать причины")
    elif root_rank > 1:
        warn(f"H5 частично: разброс есть, но первопричина на ранге {root_rank}")
    else:
        ok(f"H5 НЕ подтвердилась: первопричина на ранге 1, CE spread={ce_spread:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    hdr("Г И П О Т Е З А  6 – Данные: проверка на достаточность аномалии")
    # ══════════════════════════════════════════════════════════════════════════

    issues_h6 = []
    for mi, metric in enumerate(["cpu", "memory", "latency_ms", "rps"]):
        v_n_root = float(np.nanmean(md_n.values[:, root_idx, mi]))
        v_a_root = float(np.nanmean(md_a.values[:, root_idx, mi]))
        pct_root = abs(v_a_root - v_n_root) / (abs(v_n_root) + 1e-6) * 100

        # Проверяем, что другие узлы изменились меньше
        max_pct_other = max(
            abs(float(np.nanmean(md_a.values[:,i,mi])) - float(np.nanmean(md_n.values[:,i,mi])))
            / (abs(float(np.nanmean(md_n.values[:,i,mi]))) + 1e-6) * 100
            for i in range(N) if i != root_idx
        )
        ratio = pct_root / (max_pct_other + 1e-6)
        sym   = ok if ratio > 2.0 else (warn if ratio > 1.2 else fail)
        sym(f"{metric:12s}: ROOT Δ={pct_root:+.0f}%  others_max Δ={max_pct_other:+.0f}%  ratio={ratio:.1f}x")
        if ratio < 1.5:
            issues_h6.append(f"{metric}: первопричина выделяется недостаточно (ratio={ratio:.1f}x)")

    if issues_h6:
        fail("H6 ПОДТВЕРДИЛАСЬ:")
        for issue in issues_h6:
            warn(f"  → {issue}")
    else:
        ok("H6 НЕ подтвердилась: первопричина чётко выделяется в данных")

    # ══════════════════════════════════════════════════════════════════════════
    hdr("В Е Р Д И К Т")
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n  Сценарий {args.scenario}: {labels['scenario']}")
    print(f"  Первопричина: {true_root} [{fault}]")
    print(f"  Ранг в итоге: {root_rank}/{N}\n")

    confirmed = []
    if root_dist_rank > 1:                  confirmed.append("H1 – прототип не различает аномалию")
    if adj.sum().item() > N*(N-1)*0.7:      confirmed.append("H2 – граф слишком плотный")
    if root_delta < 0.05:                   confirmed.append("H3 – вмешательство не снижает NLL")
    if retention < 0.1:                     confirmed.append("H4 – свёртка стирает вмешательство")
    if ce_spread < 0.1:                     confirmed.append("H5 – модель не различает причины по CE")
    if issues_h6:                           confirmed.append("H6 – данные недостаточно выражены")

    if not confirmed:
        if root_rank == 1:
            ok("Все гипотезы опровергнуты. Конвейер работает корректно!")
        else:
            warn("Явных причин не обнаружено. Проверьте off-by-one в root_cause_idx")
    else:
        fail(f"Подтверждено {len(confirmed)} гипотез:")
        for h in confirmed:
            print(f"    {RED}✗{RST} {h}")
        print(f"\n  {BLD}Приоритет исправлений:{RST}")
        if "H5" in " ".join(confirmed):
            print(f"  1. Обучите модель дольше: python scripts/pretrain_demo.py --epochs 30")
        if "H1" in " ".join(confirmed):
            print(f"  2. Усильте аномалию в generate_demo_data.py (увеличьте ANOMALY_DELTA)")
        if "H4" in " ".join(confirmed):
            print(f"  3. Добавьте skip-connection в HypergraphConv")
        if "H6" in " ".join(confirmed):
            print(f"  4. Увеличьте Δ первопричины до >100%, остальных до <20%")


if __name__ == "__main__":
    main()