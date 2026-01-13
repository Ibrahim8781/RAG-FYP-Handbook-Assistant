@echo off
REM Quick Setup for Groq LLM Integration

echo ====================================================================
echo RAG Implementation - Groq LLM Setup
echo ====================================================================
echo.

REM Check if .env exists
if exist ".env" (
    echo [OK] .env file found
) else (
    echo [!] Creating .env file from template...
    copy .env.example .env
    echo.
    echo IMPORTANT: Edit .env and add your Groq API key!
    echo Get your key from: https://console.groq.com/keys
    echo.
    pause
)

echo.
echo [1/3] Installing dependencies...
pip install -r requirements.txt

echo.
echo [2/3] Checking for handbook PDF...
if exist "FYP-Handbook-2023.pdf" (
    echo [OK] Handbook PDF found
) else (
    echo [!] Please add FYP-Handbook-2023.pdf to this directory
    pause
    exit /b 1
)

echo.
echo [3/3] Creating FAISS index...
python ingest.py

echo.
echo ====================================================================
echo Setup Complete!
echo ====================================================================
echo.
echo To run the app:
echo   streamlit run app.py
echo.
echo Or use CLI:
echo   python ask.py
echo.
echo See SETUP_GROQ.md for detailed instructions.
echo.
pause
