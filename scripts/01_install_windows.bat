@echo off
setlocal

where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher not found.
  echo Install Python 3.11 first from https://www.python.org/downloads/windows/
  exit /b 1
)

py -3.11 --version >nul 2>nul
if errorlevel 1 (
  echo Python 3.11 is not installed.
  echo Install Python 3.11 64-bit, then run this script again.
  echo https://www.python.org/downloads/windows/
  exit /b 1
)

if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>nul
  if not errorlevel 1 goto install
  echo Existing .venv is not using Python 3.11. Recreating it.
  rmdir /s /q .venv
)

py -3.11 -m venv .venv

:install
call .venv\Scripts\activate.bat
python -m ensurepip --upgrade >nul 2>nul
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if not exist data\videos\demo_wire_harness.mp4 (
  python scripts\generate_demo_video.py
)

echo.
echo Installation terminee.
echo Lancement : scripts\02_ouvrir_application.bat
echo.
python --version
endlocal
