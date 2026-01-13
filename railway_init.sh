#!/bin/bash

# Railway deployment initialization script
echo "Starting RAG deployment setup..."

# Check if required files exist
if [ ! -f "faiss_index.bin" ] || [ ! -f "chunks_metadata.pkl" ] || [ ! -f "config.json" ]; then
    echo "Generated index files not found!"
    
    if [ ! -f "FYP-Handbook-2023.pdf" ]; then
        echo "ERROR: FYP-Handbook-2023.pdf not found. Please upload the PDF."
        exit 1
    fi
    
    echo "Running ingestion pipeline..."
    python ingest.py
    
    if [ $? -eq 0 ]; then
        echo "Ingestion complete!"
    else
        echo "ERROR: Ingestion failed!"
        exit 1
    fi
else
    echo "Index files found, skipping ingestion."
fi

echo "Setup complete! Starting application..."
