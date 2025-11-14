@echo off
REM FYP Handbook RAG Assistant - Windows Setup Script

echo ====================================================================
echo FYP Handbook RAG Assistant - Setup
echo ====================================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

echo [1/4] Python found
echo.

REM Install dependencies
echo [2/4] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo.

REM Check for PDF
echo [3/4] Checking for handbook...
if not exist "FYP-Handbook-2023.pdf" (
    echo WARNING: FYP-Handbook-2023.pdf not found
    echo Please place the handbook PDF in this directory
    echo.
    pause
    exit /b 0
)
echo Handbook found!
echo.

REM Create index
echo [4/4] Creating FAISS index...
python ingest.py
if errorlevel 1 (
    echo ERROR: Failed to create index
    pause
    exit /b 1
)
echo.

echo ====================================================================
echo Setup complete!
echo ====================================================================
echo.
echo To launch the app, run:
echo   run.bat
pause
