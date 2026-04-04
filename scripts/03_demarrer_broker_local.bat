@echo off
setlocal

if not exist .venv\Scripts\python.exe (
  echo Environnement manquant. Lancez d'abord scripts\01_install_windows.bat
  exit /b 1
)

where mosquitto >nul 2>nul
if not errorlevel 1 (
  echo Demarrage du broker MQTT local avec Mosquitto sur 127.0.0.1:1883
  mosquitto -v
  exit /b %errorlevel%
)

call .venv\Scripts\activate.bat

if exist .venv\Scripts\amqtt.exe (
  echo Mosquitto introuvable. Utilisation du broker Python amqtt.
  .venv\Scripts\amqtt.exe -c config\broker.yaml
  exit /b %errorlevel%
)

echo Aucun broker MQTT disponible.
exit /b 1
