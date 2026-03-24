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
    echo ERROR: Python not found.
    echo.
    echo   Download: https://www.python.org/downloads/
    echo   IMPORTANT: Check "Add Python to PATH" during installation.
    echo   After installing, restart this terminal and try again.
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
    echo WARNING: FFmpeg not found ^(required by Whisper for audio processing^).
    echo.
    echo   Option 1: winget install ffmpeg
    echo   Option 2: https://ffmpeg.org/download.html
    echo   After installing, restart this terminal.
    echo.
)

REM Check API key
if "%ANTHROPIC_API_KEY%"=="" (
    echo.
    echo NOTE: No AI enhancement configured. Transcription still works.
    echo.
    echo   For free local AI:  Install Ollama from https://ollama.com
    echo                       Then run: ollama pull llama3.2
    echo   For Claude AI:      set ANTHROPIC_API_KEY=your-key-here
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
