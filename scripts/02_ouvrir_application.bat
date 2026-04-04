@echo off
setlocal

if not exist .venv\Scripts\python.exe (
  echo Environnement manquant. Lancez d'abord scripts\01_install_windows.bat
  exit /b 1
)

call .venv\Scripts\activate.bat
python run.py
endlocal
