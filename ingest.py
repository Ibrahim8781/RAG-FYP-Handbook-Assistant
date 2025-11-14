"""
FYP Handbook RAG - Ingestion Script
Loads PDF, chunks text, creates embeddings, and builds FAISS index
"""

import os
import re
import pickle
import json
from typing import List, Dict
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Configuration
PDF_PATH = "FYP-Handbook-2023.pdf"
CHUNK_SIZE = 300  # words (250-400 range)
OVERLAP = 90  # words (30% of 300)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "chunks_metadata.pkl"
CONFIG_PATH = "config.json"


def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract text from PDF page by page, preserving page numbers.
    
    Returns:
        List of dicts with 'page_number' and 'text' keys
    """
    print(f"Loading PDF: {pdf_path}")
    pages_data = []
    
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)
        print(f"Total pages: {total_pages}")
        
        for page_num in range(total_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            
            if text.strip():
                pages_data.append({
                    'page_number': page_num + 1,  # 1-indexed
                    'text': text
                })
    
    print(f"Extracted text from {len(pages_data)} pages")
    return pages_data


def extract_section_hint(text: str) -> str:
    """
    Extract the first heading-like text from a chunk.
    Looks for lines in ALL CAPS or with specific formatting.
    """
    lines = text.strip().split('\n')
    for line in lines[:5]:  # Check first 5 lines
        line = line.strip()
        # Check for ALL CAPS headings (common in handbooks)
        if line.isupper() and len(line) > 5 and len(line) < 100:
            return line
        # Check for numbered sections like "1. Introduction" or "1.1 Overview"
        if re.match(r'^\d+\.?\d*\s+[A-Z]', line):
            return line
    return "General Section"


def chunk_text(pages_data: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
    """
    Chunk text with word-based sliding window and overlap.
    Preserves page numbers and extracts section hints.
    
    Args:
        pages_data: List of page dictionaries
        chunk_size: Target chunk size in words
        overlap: Overlap size in words
        
    Returns:
        List of chunk dictionaries with metadata
    """
    print(f"Chunking text (size={chunk_size} words, overlap={overlap} words)")
    chunks = []
    chunk_id = 0
    
    for page_data in pages_data:
        page_num = page_data['page_number']
        text = page_data['text']
        
        # Split into words
        words = text.split()
        
        # Create chunks with overlap
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            # Only add chunks with substantial content
            if len(chunk_text.strip()) > 50:  # At least 50 characters
                section_hint = extract_section_hint(chunk_text)
                
                chunks.append({
                    'chunk_id': chunk_id,
                    'page_number': page_num,
                    'text': chunk_text,
                    'section_hint': section_hint,
                    'word_count': len(chunk_words)
                })
                chunk_id += 1
            
            # Move to next chunk with overlap
            start += chunk_size - overlap
            
            # If we're at the end, break
            if end >= len(words):
                break
    
    print(f"Created {len(chunks)} chunks")
    return chunks


def create_embeddings(chunks: List[Dict], model_name: str) -> np.ndarray:
    """
    Create embeddings for all chunks using Sentence-BERT.
    
    Args:
        chunks: List of chunk dictionaries
        model_name: Name of the sentence-transformers model
        
    Returns:
        Numpy array of embeddings
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Creating embeddings for {len(chunks)} chunks...")
    texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build FAISS index for cosine similarity search.
    Uses IndexFlatIP (Inner Product) which works with normalized vectors.
    
    Args:
        embeddings: Numpy array of embeddings
        
    Returns:
        FAISS index
    """
    print("Building FAISS index...")
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Add embeddings to index
    index.add(embeddings)
    
    print(f"FAISS index built with {index.ntotal} vectors")
    return index


def save_index_and_metadata(index: faiss.IndexFlatIP, chunks: List[Dict], 
                            index_path: str, metadata_path: str, config_path: str):
    """
    Save FAISS index, chunk metadata, and configuration to disk.
    """
    print(f"Saving FAISS index to {index_path}")
    faiss.write_index(index, index_path)
    
    print(f"Saving metadata to {metadata_path}")
    with open(metadata_path, 'wb') as f:
        pickle.dump(chunks, f)
    
    # Save configuration
    config = {
        'chunk_size': CHUNK_SIZE,
        'overlap': OVERLAP,
        'embedding_model': EMBEDDING_MODEL,
        'num_chunks': len(chunks),
        'embedding_dim': index.d
    }
    
    print(f"Saving configuration to {config_path}")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("All data saved successfully!")


def main():
    """
    Main ingestion pipeline:
    1. Load PDF and extract text per page
    2. Chunk text with overlap
    3. Create embeddings
    4. Build FAISS index
    5. Save to disk
    """
    print("=" * 60)
    print("FYP Handbook RAG - Ingestion Pipeline")
    print("=" * 60)
    
    # Check if PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"ERROR: PDF file not found: {PDF_PATH}")
        print("Please place the FYP Handbook PDF in the project directory.")
        return
    
    # Step 1: Extract text from PDF
    pages_data = extract_text_from_pdf(PDF_PATH)
    
    # Step 2: Chunk text
    chunks = chunk_text(pages_data, CHUNK_SIZE, OVERLAP)
    
    # Step 3: Create embeddings
    embeddings = create_embeddings(chunks, EMBEDDING_MODEL)
    
    # Step 4: Build FAISS index
    index = build_faiss_index(embeddings)
    
    # Step 5: Save everything
    save_index_and_metadata(index, chunks, FAISS_INDEX_PATH, METADATA_PATH, CONFIG_PATH)
    
    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print(f"Total chunks: {len(chunks)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
