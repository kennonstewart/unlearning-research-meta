#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "🚀 Starting post-creation setup..."

# The dev container feature automatically clones submodules.
# This command ensures they are fully updated.
echo "Updating Git submodules..."
git submodule update --init --recursive

# Create a Python virtual environment.
echo "Creating Python virtual environment..."
python3 -m venv .venv

# Activate the venv and install dependencies from requirements.txt.
echo "Installing Python dependencies..."
source .venv/bin/activate && pip install -r requirements.txt

# Install the local package in editable mode.
# Note: The path 'code/memory_pair' is used from .gitmodules,
# which is the source of truth.
echo "Installing local 'memory_pair' package in editable mode..."
pip install -e code/memory_pair

echo "✅ Environment setup complete. Ready to run experiments!"