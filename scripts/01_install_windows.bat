@echo off
setlocal

if exist .venv\Scripts\python.exe goto install

where py >nul 2>nul
if %errorlevel%==0 (
  py -3.11 -m venv .venv 2>nul
  if exist .venv\Scripts\python.exe goto install
)

python -m venv .venv

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
