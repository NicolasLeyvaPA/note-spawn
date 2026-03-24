@echo off
title Lecture Notes AI
cd /d "%~dp0"

echo.
echo ========================================
echo   LECTURE NOTES AI - Setup
echo ========================================
echo.

REM Check for Python
py --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Install from python.org
    echo Make sure to check "Add Python to PATH" during install
    pause
    exit /b 1
)

REM Check/install requirements
echo Checking dependencies...
py -m pip install -r requirements.txt -q

REM Check for FFmpeg
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo WARNING: FFmpeg not found. Whisper needs it.
    echo Install with: winget install ffmpeg
    echo Or download from: https://ffmpeg.org/download.html
    echo.
)

REM Check API key
if "%ANTHROPIC_API_KEY%"=="" (
    echo.
    echo NOTE: ANTHROPIC_API_KEY not set - AI enhancement disabled
    echo To enable, run: set ANTHROPIC_API_KEY=sk-ant-your-key-here
    echo.
)

echo.
echo ========================================
echo   Starting server...
echo   Open http://localhost:5000
echo ========================================
echo.

py app.py

pause
