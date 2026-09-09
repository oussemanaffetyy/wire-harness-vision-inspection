@echo off
if not exist "%~dp0nodered\node_modules\node-red\red.js" (
    echo Installation necessaire : npm.cmd --prefix nodered ci
    exit /b 1
)
call npm.cmd --prefix "%~dp0nodered" start -- %*
exit /b %errorlevel%
