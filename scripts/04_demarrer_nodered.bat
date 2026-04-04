@echo off
setlocal

where node-red >nul 2>nul
if %errorlevel%==0 (
  node-red
  exit /b %errorlevel%
)

if exist "%ProgramFiles%\nodejs\node.exe" if exist "%APPDATA%\npm\node_modules\node-red\red.js" (
  "%ProgramFiles%\nodejs\node.exe" "%APPDATA%\npm\node_modules\node-red\red.js"
  exit /b %errorlevel%
)

echo Node-RED introuvable.
echo Installation possible :
echo   npm install -g node-red
exit /b 1
