import sys
sys.path.insert(0, 'src')

checks = []
try:
    from cairn.gui.widgets.sidebar import Sidebar
    checks.append(('connect_live сигнал в Sidebar', hasattr(Sidebar, 'connect_live')))
except Exception as e:
    checks.append(('Sidebar импорт', f'ОШИБКА: {e}'))

try:
    from cairn.connectors.live_connector import LiveSystemConnector
    checks.append(('docker_stats в LiveConnector', hasattr(LiveSystemConnector, '_fetch_docker_stats')))
except Exception as e:
    checks.append(('LiveConnector импорт', f'ОШИБКА: {e}'))

try:
    from cairn.gui.widgets.connect_dialog import ConnectDialog
    checks.append(('ConnectDialog', True))
except Exception as e:
    checks.append(('ConnectDialog импорт', f'ОШИБКА: {e}'))

try:
    from cairn.gui.main_window import CAIRNMainWindow
    checks.append(('_on_connect_live в MainWindow', hasattr(CAIRNMainWindow, '_on_connect_live')))
    checks.append(('_on_live_connected в MainWindow', hasattr(CAIRNMainWindow, '_on_live_connected')))
except Exception as e:
    checks.append(('MainWindow импорт', f'ОШИБКА: {e}'))

print("\n=== Проверка установленных файлов ===")
all_ok = True
for name, result in checks:
    ok = result is True
    if not ok:
        all_ok = False
    print(f"  {'OK' if ok else 'FAIL'}  {name}: {result}")

print(f"\n{'Все файлы установлены корректно.' if all_ok else 'Некоторые файлы не обновлены – замените их.'}")
