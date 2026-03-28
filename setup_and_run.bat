@echo off
echo =======================================================
echo Physical-Bounded Multimodal Mode Discovery
echo Automated Setup and Execution Script (Windows)
echo =======================================================

echo.
echo [1/4] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.9 or higher and try again.
    exit /b 1
)

echo [2/4] Setting up virtual environment...
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment "venv"...
    python -m venv venv
)
call venv\Scripts\activate.bat

echo [3/4] Installing dependencies...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt

echo [4/4] Ensuring CWRU dataset is downloaded...
python src\download_cwru.py

echo.
echo =======================================================
echo Setup complete. Starting the evaluation pipeline...
echo =======================================================
echo.

python -m src.pipeline --demo cwru

echo.
echo Pipeline has finished executing. Results are saved in the results/ folder.
echo Press any key to exit.
pause >nul
