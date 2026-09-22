@echo off
title BinSense - Smart Waste Assistant
color 0A
echo.
echo  =============================================
echo   BinSense - Starting...
echo  =============================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo  Setup not found. Please run install.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo  Starting server...
echo  Browser will open automatically.
echo.
echo  To stop, close this window.
echo  =============================================
echo.

python app.py
pause
