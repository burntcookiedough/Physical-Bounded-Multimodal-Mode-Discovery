#!/bin/bash
set -e

echo "======================================================="
echo "Physical-Bounded Multimodal Mode Discovery"
echo "Automated Setup and Execution Script (Linux/macOS)"
echo "======================================================="

echo ""
echo "[1/4] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH."
    echo "Please install Python 3.9 or higher and try again."
    exit 1
fi

echo "[2/4] Setting up virtual environment..."
if [ ! -f "venv/bin/activate" ]; then
    echo "Creating virtual environment 'venv'..."
    python3 -m venv venv
fi
source venv/bin/activate

echo "[3/4] Installing dependencies..."
python3 -m pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt

echo "[4/4] Ensuring CWRU dataset is downloaded..."
python src/download_cwru.py

echo ""
echo "======================================================="
echo "Setup complete. Starting the evaluation pipeline..."
echo "======================================================="
echo ""

python -m src.pipeline --demo cwru

echo ""
echo "Pipeline has finished executing. Results are saved in the results/ folder."
