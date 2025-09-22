@echo off
cd /d "%~dp0"
echo Starting AI Travel Assistant...
echo Working directory: %CD%
echo.

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements_frontend.txt
)

echo.
echo Starting the Travel Assistant frontend...
echo Open your browser and go to: http://localhost:8503
echo.
echo Press Ctrl+C to stop the application
echo.

python -m streamlit run "%~dp0app.py" --server.port 8503
