@echo off
:: CAIRN – запуск GUI с консолью (для разработчиков)
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)
python -m cairn %*
if errorlevel 1 pause
