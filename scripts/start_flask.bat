@echo off
REM Social Debate AI - Windows Startup Script
REM Run this script to start the web interface

echo Starting Social Debate AI...
echo ================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found
    pause
    exit /b 1
)

REM Install dependencies
if not exist "requirements.txt" (
    echo Error: requirements.txt not found
    pause
    exit /b 1
)

echo Installing dependencies...
pip install -r requirements.txt

REM Check environment variables
if "%OPENAI_API_KEY%"=="" (
    echo Warning: OPENAI_API_KEY not set
    echo Please set your OpenAI API key:
    echo set OPENAI_API_KEY=your-api-key-here
)

echo Starting Flask server...
echo Access at: http://localhost:5000
echo Press Ctrl+C to stop
echo.

python run_flask.py
pause 