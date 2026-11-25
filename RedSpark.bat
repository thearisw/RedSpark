@echo off
rem RedSpark: double-click to open the app. The first run sets everything up.
setlocal
cd /d "%~dp0"
title RedSpark
set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo First run: setting up RedSpark. This takes a minute or two...
    python -m venv .venv
    if errorlevel 1 goto nopython
)

"%PY%" -c "import redspark.app" >nul 2>&1
if errorlevel 1 (
    echo Installing packages...
    "%PY%" -m pip install --quiet --disable-pip-version-check -e ".[server]"
    if errorlevel 1 goto failed
)

"%PY%" -m redspark.app %*
if errorlevel 1 pause
exit /b

:nopython
echo.
echo Python was not found. Install Python 3.10 or newer from python.org,
echo tick "Add python.exe to PATH" in the installer, then run this again.
pause
exit /b 1

:failed
echo.
echo Setup failed. The messages above say why.
pause
exit /b 1
