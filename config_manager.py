"""
Configuration management for dev/prod environments
"""

import os
from typing import Dict, Any

class Config:
    """Base configuration"""
    # Common settings
    CHUNK_SIZE = 300
    OVERLAP = 90
    TOP_K = 5
    SIMILARITY_THRESHOLD = 0.25
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Paths
    FAISS_INDEX_PATH = "faiss_index.bin"
    METADATA_PATH = "chunks_metadata.pkl"
    CONFIG_PATH = "config.json"
    PDF_PATH = "FYP-Handbook-2023.pdf"
    
    # LLM settings
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1500"))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    
    # Environment
    ENV = os.getenv("ENVIRONMENT", "development")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"


class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    ENV = "development"


class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    ENV = "production"
    # Production-specific settings
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # Can upgrade for prod


# Configuration selector
def get_config() -> Config:
    """Get configuration based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    else:
        return DevelopmentConfig()


# Global config instance
config = get_config()
