"""Предобучение модели CAIRN — версия Приоритета 2.

Изменения по сравнению с предыдущей версией:
  - Увеличено число эпох по умолчанию (10 вместо 3)
  - n_components GMM увеличен до 5 (лучше покрывает распределение нормы)
  - Разделение TRAIN/TEST сохранено: scenario_1+2 → train, scenario_3 → test

Использование:
    python scripts/pretrain_demo.py
    python scripts/pretrain_demo.py --epochs 15 --out data/sample
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Предобучение демо-модели CAIRN")
    parser.add_argument("--out",    default="data/sample", help="Директория вывода")
    parser.add_argument("--epochs", type=int, default=10,  help="Эпох на этап")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    out_dir  = Path(args.out)
    data_dir = out_dir

    # ── Генерируем данные если нужно ─────────────────────────────────────
    for sc in ["scenario_1", "scenario_2", "scenario_3"]:
        if not (data_dir / sc / "metrics.csv").exists():
            print(f"Данные {sc} не найдены — генерируем...")
            import subprocess
            subprocess.run(
                [sys.executable, "scripts/generate_demo_data.py",
                 "--out", str(data_dir), "--seed", str(args.seed)],
                check=True,
            )
            break

    # ── Импорты ───────────────────────────────────────────────────────────
    import torch
    from cairn.training import (
        create_demo_dataset, CAIRNDataset, CAIRNModel,
        CAIRNLoss, CAIRNTrainer, TrainerConfig,
    )
    from cairn.perception import StateBuilder, HypergraphBuilder
    from cairn.reasoning import ConditionalGMM, ConfoundedVGAE, CounterfactualModule
    from cairn.connectors.csv_file import YAMLTopologyConnector

    torch.manual_seed(args.seed)

    # ── Параметры архитектуры ─────────────────────────────────────────────
    # ИЗМЕНЕНИЕ П2: n_components 3 → 5
    # GMM с 5 компонентами лучше моделирует норму и острее реагирует
    # на отклонения — ключевое улучшение для AC@1.
    ARCH = {
        "state_dim":      32,
        "context_dim":    8,
        "n_metrics":      4,
        "n_components":   5,    # ← было 3, стало 5
        "n_confounders":  2,
        "confounder_dim": 8,
        "d_met":          16,
        "d_log":          8,
        "d_tr":           8,
        "d_ssm":          8,
        "d_brk":          8,
        "ssm_state_dim":  16,
        "window":         15,
        "log_vocab_size": 300,
        "n_conv_layers":  1,
    }
    D   = ARCH["state_dim"]
    CTX = ARCH["context_dim"]
    F   = ARCH["n_metrics"]

    print("Строим гиперграф из топологии...")
    topo = YAMLTopologyConnector(data_dir / "scenario_1" / "topology.yaml").fetch()
    hg   = HypergraphBuilder.from_topology_data(topo)
    print(f"  Узлов: {hg.n_nodes}, рёбер: {len(hg.edges)}")

    # ── Загружаем данные с честным разделением ────────────────────────────
    TRAIN_SCENARIOS = ["scenario_1", "scenario_2"]
    TEST_SCENARIO   = "scenario_3"

    print("\nЗагружаем TRAIN-сценарии...")
    train_incidents = []
    for sc in TRAIN_SCENARIOS:
        sc_dir = data_dir / sc
        if (sc_dir / "metrics.csv").exists():
            ds = create_demo_dataset(sc_dir, window_size=30, stride=15)
            train_incidents.extend([ds[i] for i in range(len(ds))])
            print(f"  [TRAIN] {sc}: {len(ds)} окон ({ds.n_normal} норм., {ds.n_anomaly} аном.)")

    print("\nЗагружаем TEST-сценарий (модель его НЕ увидит)...")
    test_sc_dir = data_dir / TEST_SCENARIO
    test_ds = None
    if (test_sc_dir / "metrics.csv").exists():
        test_ds = create_demo_dataset(test_sc_dir, window_size=30, stride=15)
        print(f"  [TEST]  {TEST_SCENARIO}: {len(test_ds)} окон "
              f"({test_ds.n_normal} норм., {test_ds.n_anomaly} аном.)")

    if not train_incidents:
        print("Ошибка: нет обучающих данных.")
        sys.exit(1)

    train_dataset = CAIRNDataset(train_incidents)
    print(f"\nИтого TRAIN: {len(train_dataset)} окон "
          f"({train_dataset.n_normal} норм., {train_dataset.n_anomaly} аном.)")

    # ── Модель ────────────────────────────────────────────────────────────
    print("\nИнициализируем модель...")
    model = CAIRNModel(
        state_builder=StateBuilder(
            n_metrics=F,
            log_vocab_size=ARCH["log_vocab_size"],
            state_dim=D,
            context_dim=CTX,
            d_met=ARCH["d_met"],
            d_log=ARCH["d_log"],
            d_tr=ARCH["d_tr"],
            d_ssm=ARCH["d_ssm"],
            d_brk=ARCH["d_brk"],
            ssm_state_dim=ARCH["ssm_state_dim"],
            window=ARCH["window"],
        ),
        gmm=ConditionalGMM(
            state_dim=D,
            context_dim=CTX,
            n_components=ARCH["n_components"],  # 5 компонент
        ),
        vgae=ConfoundedVGAE(
            state_dim=D,
            n_confounders=ARCH["n_confounders"],
            confounder_dim=ARCH["confounder_dim"],
        ),
        cf_module=CounterfactualModule(
            state_dim=D,
            n_conv_layers=ARCH["n_conv_layers"],
        ),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Параметров: {n_params:,}")
    print(f"  GMM компонент: {ARCH['n_components']} (увеличено для лучшего покрытия нормы)")

    # ── Обучение ──────────────────────────────────────────────────────────
    cfg = TrainerConfig(
        pretrain_epochs=args.epochs,
        main_epochs=args.epochs,
        finetune_epochs=args.epochs,
        freeze_epochs=0,
        patience=999,
        log_every=max(1, args.epochs // 5),  # логировать 5 раз за этап
        device="cpu",
        checkpoint_dir=str(out_dir / "checkpoints"),
        save_every=999,
    )
    loss_fn = CAIRNLoss(adaptive=True)
    trainer  = CAIRNTrainer(model, loss_fn, hg, cfg)

    print(f"\nОбучение ({args.epochs} эп/этап × 3 этапа) на TRAIN...")
    t0      = time.time()
    history = trainer.train(train_dataset)
    elapsed = time.time() - t0
    print(f"  Время обучения: {elapsed:.1f} с")

    for stage in ("pretrain_loss", "main_loss", "finetune_loss"):
        vals = history.get(stage, [])
        if vals:
            print(f"  {stage}: {vals[0]:.4f} → {vals[-1]:.4f}")

    # ── Оценка TRAIN / TEST ───────────────────────────────────────────────
    print("\nОценка на TRAIN:")
    train_metrics = trainer.evaluate(train_dataset)
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.3f}")

    test_metrics = {}
    if test_ds is not None:
        print("\nОценка на TEST (честные метрики):")
        test_metrics = trainer.evaluate(test_ds)
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.3f}")

        gap = train_metrics.get("AC@1", 0) - test_metrics.get("AC@1", 0)
        if gap > 0.2:
            print(f"\n  ⚠ Разрыв TRAIN/TEST по AC@1: {gap:.2f} — возможно переобучение.")
        else:
            print(f"\n  ✓ Разрыв TRAIN/TEST по AC@1: {gap:.2f} — обобщение нормальное.")

    # ── Сохранение ────────────────────────────────────────────────────────
    model_path = out_dir / "demo_model.pt"
    trainer.save(model_path)

    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    ckpt["arch_config"] = ARCH
    torch.save(ckpt, model_path)
    print(f"\nМодель сохранена: {model_path}")
    print(f"  arch_config встроен в чекпоинт ✓")

    # ── demo_config.yaml ──────────────────────────────────────────────────
    import yaml
    demo_cfg = {
        "model": ARCH,
        "split": {
            "train": TRAIN_SCENARIOS,
            "test":  TEST_SCENARIO,
        },
        "scenarios": {
            "1": {"name": "CPU Exhaustion",  "root": "order-service-1", "type": "cpu_exhaustion"},
            "2": {"name": "Memory Leak",     "root": "cache-service-1", "type": "memory_pressure"},
            "3": {"name": "Network Delay",   "root": "frontend-1",      "type": "latency_spike"},
        },
        "training": {
            "epochs_per_stage": args.epochs,
            "n_train":          len(train_dataset),
            "n_test":           len(test_ds) if test_ds else 0,
            "elapsed_sec":      round(elapsed, 1),
        },
        "train_metrics": train_metrics,
        "test_metrics":  test_metrics,
    }
    cfg_path = out_dir / "demo_config.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.dump(demo_cfg, f, allow_unicode=True, default_flow_style=False)
    print(f"Конфигурация: {cfg_path}")

    print("\n✅ Готово. Следующие шаги:")
    print(f"  1. Диагностика: python scripts/diagnose.py --scenario 3 "
          f"--model {model_path}")
    print(f"  2. Оценка:      python scripts/evaluate.py "
          f"--checkpoint {model_path} --data-dir {data_dir / TEST_SCENARIO}")


if __name__ == "__main__":
    main()