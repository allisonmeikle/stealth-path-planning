#!/usr/bin/env bash
set -e  # stop on first error

# Deactivate venv if active
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Deactivating current venv..."
    deactivate || true
fi

# Remove old venv if it exists
if [[ -d "venv" ]]; then
    echo "Deleting existing venv..."
    rm -rf venv
fi

# Create new venv
echo "Creating new venv..."
python3 -m venv venv

# Activate new venv
echo "Activating venv..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
if [[ -f "requirements.txt" ]]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
else
    echo "ERROR: requirements.txt not found!"
    exit 1
fi

# Run your main script
echo "Running main.py..."
python main.py