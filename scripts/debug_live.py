"""Диагностика live подключения."""
import sys, time, traceback
sys.path.insert(0, 'src')

print("=== 1. Контроллер ===")
try:
    from cairn.gui.controller import CAIRNController
    from cairn.config import CAIRNConfig
    ctrl = CAIRNController(CAIRNConfig())
    attrs = ['_model', '_hypergraph', '_metric_data', '_is_live_mode']
    for a in attrs:
        print(f"  {a}: {getattr(ctrl, a, 'MISSING')}")
    methods = ['load_live_data', '_load_checkpoint_silent']
    for m in methods:
        print(f"  {'OK' if hasattr(ctrl, m) else 'MISS'} ctrl.{m}")
except Exception as e:
    print(f"  ОШИБКА: {e}")

print("\n=== 2. Модель ===")
from pathlib import Path
p = Path("data/sample/demo_model.pt")
print(f"  demo_model.pt: {'exists' if p.exists() else 'NOT FOUND'}")
if p.exists():
    try:
        import torch
        ckpt = torch.load(str(p), map_location="cpu", weights_only=True)
        print(f"  arch_config: {ckpt.get('arch_config', 'NOT IN CKPT')}")
        ctrl2 = CAIRNController(CAIRNConfig())
        ctrl2._load_checkpoint_silent(str(p))
        print(f"  Авто-загрузка: OK, model={ctrl2._model}")
    except Exception as e:
        print(f"  Авто-загрузка ОШИБКА: {e}")
        traceback.print_exc()

print("\n=== 3. Коннектор + топология ===")
try:
    from cairn.connectors.live_connector import LiveSystemConnector
    conn = LiveSystemConnector("configs/connectors/online_boutique.yaml")
    ok, msg = conn.is_available()
    print(f"  Статус: {'OK' if ok else 'FAIL'} — {msg}")
    topo = conn.fetch_topology()
    print(f"  Топология: {len(topo.instances)} экземпляров")
    for inst in topo.instances[:3]:
        print(f"    - {inst.name} ({inst.service})")
except Exception as e:
    print(f"  ОШИБКА: {e}")
    traceback.print_exc()

print("\n=== 4. main_window методы ===")
try:
    from cairn.gui.main_window import CAIRNMainWindow
    mw_methods = ['_on_live_connected', '_auto_load_model',
                  '_start_progress_timer', '_on_anomaly_detected',
                  '_start_anomaly_monitor']
    for m in mw_methods:
        print(f"  {'OK' if hasattr(CAIRNMainWindow, m) else 'MISS'} mw.{m}")
    # load_live_data правильно в controller, не в main_window
    print(f"  OK ctrl.load_live_data (проверено выше)")
except Exception as e:
    print(f"  ОШИБКА: {e}")