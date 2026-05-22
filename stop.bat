@echo off
setlocal enabledelayedexpansion

set "PORT=8501"

echo Stopping Streamlit on port %PORT% ...

set "FOUND="
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
    set "FOUND=1"
    echo   killing PID %%a
    taskkill /F /T /PID %%a >nul 2>&1
)

if not defined FOUND (
    echo   no listener on port %PORT%.
)

endlocal
