#!/bin/bash
set -e

echo "Starting build process..."

# Install Python dependencies
echo "Installing dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Run ingestion to create FAISS index
echo "Creating FAISS index..."
python ingest.py

echo "Build complete!"