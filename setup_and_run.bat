@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul

:: ============================================================
:: CAIRN — автоматическая установка и запуск
:: Достаточно двойного клика. Интернет нужен только при первом запуске.
:: ============================================================

set "ROOT=%~dp0"
set "VENV=%ROOT%.venv"
set "PYTHON_MIN_VER=311"
set "PYTHON_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
set "PYTHON_INSTALLER=%TEMP%\python_cairn_installer.exe"
set "LOG=%ROOT%cairn_setup.log"

echo.
echo  ╔══════════════════════════════════════════╗
echo  ║        CAIRN — Запуск системы            ║
echo  ╚══════════════════════════════════════════╝
echo.

:: ── Шаг 1: Найти Python ─────────────────────────────────────
echo [1/4] Проверка Python...

set "PYTHON_EXE="

:: Сначала ищем py launcher (рекомендуемый способ для Windows)
where py >nul 2>&1
if !errorlevel! == 0 (
    for /f "tokens=*" %%i in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do (
        set "PYTHON_EXE=%%i"
    )
)

:: Если py не нашёл 3.11 — ищем python напрямую
if not defined PYTHON_EXE (
    for /f "tokens=*" %%i in ('where python 2^>nul') do (
        if not defined PYTHON_EXE (
            :: Проверяем версию: нужна 3.11+
            for /f "tokens=1,2 delims=." %%a in (
                '"%%i" -c "import sys; print(sys.version_info.major, sys.version_info.minor)" 2^>nul'
            ) do (
                set /a "VER=%%a*100+%%b"
                if !VER! GEQ !PYTHON_MIN_VER! set "PYTHON_EXE=%%i"
            )
        )
    )
)

if defined PYTHON_EXE (
    echo     OK: %PYTHON_EXE%
    goto :check_venv
)

:: ── Шаг 1b: Python не найден — установить автоматически ─────
echo     Python 3.11 не обнаружен. Начинаем установку...
echo     (Требуется подключение к интернету, ~25 МБ)
echo.

:: Проверяем интернет
ping -n 1 -w 3000 python.org >nul 2>&1
if !errorlevel! NEQ 0 (
    echo.
    echo  [ОШИБКА] Нет подключения к интернету.
    echo  Установите Python 3.11 вручную: https://www.python.org/downloads/
    echo  При установке отметьте галочку "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

echo     Загрузка Python 3.11.9...
powershell -Command "& {
    $ProgressPreference = 'SilentlyContinue'
    try {
        Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_INSTALLER%' -UseBasicParsing
        Write-Host '    Загрузка завершена.'
    } catch {
        Write-Host '    Ошибка загрузки: ' + $_.Exception.Message
        exit 1
    }
}"
if !errorlevel! NEQ 0 (
    echo  [ОШИБКА] Не удалось скачать Python.
    pause
    exit /b 1
)

echo     Установка Python (тихий режим, без вашего участия)...
:: Параметры тихой установки:
::   InstallAllUsers=0  — только для текущего пользователя (не требует прав администратора)
::   PrependPath=1      — добавить в PATH
::   Include_pip=1      — включить pip
::   Include_launcher=1 — включить py launcher
"%PYTHON_INSTALLER%" /quiet ^
    InstallAllUsers=0 ^
    PrependPath=1 ^
    Include_pip=1 ^
    Include_launcher=1 ^
    Include_tcltk=0 ^
    Include_test=0 ^
    2>>"%LOG%"

if !errorlevel! NEQ 0 (
    echo  [ОШИБКА] Установка Python завершилась с ошибкой.
    echo  Подробности: %LOG%
    pause
    exit /b 1
)

:: Обновляем PATH в текущем сеансе (установщик добавил, но сеанс не знает)
for /f "tokens=*" %%i in (
    'powershell -Command "[System.Environment]::GetEnvironmentVariable(\"Path\",\"User\")"'
) do set "PATH=%%i;%PATH%"

:: Ищем снова после установки
for /f "tokens=*" %%i in ('where python 2^>nul') do (
    if not defined PYTHON_EXE set "PYTHON_EXE=%%i"
)

if not defined PYTHON_EXE (
    echo  [ОШИБКА] Python установлен, но не найден в PATH.
    echo  Пожалуйста, перезапустите скрипт.
    pause
    exit /b 1
)

echo     Python успешно установлен: %PYTHON_EXE%

:: ── Шаг 2: Создать виртуальную среду ────────────────────────
:check_venv
echo.
echo [2/4] Проверка виртуальной среды...

if exist "%VENV%\Scripts\python.exe" (
    echo     OK: виртуальная среда уже существует.
    goto :check_deps
)

echo     Создание виртуальной среды .venv...
"%PYTHON_EXE%" -m venv "%VENV%" >>"%LOG%" 2>&1
if !errorlevel! NEQ 0 (
    echo  [ОШИБКА] Не удалось создать виртуальную среду.
    echo  Подробности: %LOG%
    pause
    exit /b 1
)
echo     Виртуальная среда создана.

:: ── Шаг 3: Установить зависимости ───────────────────────────
:check_deps
echo.
echo [3/4] Проверка зависимостей...

:: Проверяем, установлен ли уже cairn (маркер успешной установки)
"%VENV%\Scripts\python.exe" -c "import cairn" >nul 2>&1
if !errorlevel! == 0 (
    echo     OK: все зависимости уже установлены.
    goto :launch
)

echo     Установка зависимостей (первый запуск ~3-5 минут)...
echo     Пожалуйста, подождите...
echo.

:: Обновляем pip
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip --quiet >>"%LOG%" 2>&1

:: Устанавливаем проект в режиме редактирования
"%VENV%\Scripts\python.exe" -m pip install -e "%ROOT%." --quiet >>"%LOG%" 2>&1
if !errorlevel! NEQ 0 (
    echo  [ОШИБКА] Установка зависимостей завершилась с ошибкой.
    echo  Подробности: %LOG%
    pause
    exit /b 1
)
echo     Зависимости установлены успешно.

:: ── Шаг 4: Запуск CAIRN ─────────────────────────────────────
:launch
echo.
echo [4/4] Запуск CAIRN...
echo.

:: Запускаем через pythonw.exe (без консольного окна)
if exist "%VENV%\Scripts\pythonw.exe" (
    start "" "%VENV%\Scripts\pythonw.exe" -m cairn
) else (
    :: Fallback: запуск с окном консоли
    start "" "%VENV%\Scripts\python.exe" -m cairn
)

:: Небольшая пауза чтобы убедиться, что процесс стартовал
timeout /t 2 /nobreak >nul

echo  CAIRN запущен. Это окно можно закрыть.
echo.
timeout /t 3 /nobreak >nul
exit /b 0