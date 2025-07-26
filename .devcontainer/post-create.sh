#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "🚀 Starting post-creation setup..."

# Create a Python virtual environment.
echo "Creating Python virtual environment..."
python3 -m venv .venv

# Activate the venv and install dependencies from requirements.txt.
echo "Installing Python dependencies..."
source .venv/bin/activate && pip install -r .devcontainer/requirements.txt

echo "✅ Environment setup complete. Ready to run experiments!"