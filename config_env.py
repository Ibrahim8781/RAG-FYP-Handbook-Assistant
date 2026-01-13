"""
Environment-based Configuration
Separate settings for development and production
"""

import os
from typing import Dict, Any

class Config:
    """Base configuration"""
    # Common settings
    CHUNK_SIZE = 300
    OVERLAP = 90
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    TOP_K = 5
    SIMILARITY_THRESHOLD = 0.25
    
    # LLM settings
    GROQ_MODEL = "llama-3.1-8b-instant"
    LLM_TEMPERATURE = 0.3
    LLM_MAX_TOKENS = 1024
    
    # Paths
    FAISS_INDEX_PATH = "faiss_index.bin"
    METADATA_PATH = "chunks_metadata.pkl"
    CONFIG_PATH = "config.json"
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and key.isupper()
        }


class DevelopmentConfig(Config):
    """Development environment settings"""
    # Enable debug features
    DEBUG = True
    VERBOSE_LOGGING = True
    SHOW_TIMING = True
    SHOW_CHUNK_SCORES = True
    
    # More detailed LLM responses in dev
    LLM_MAX_TOKENS = 1500
    
    # Streamlit settings
    STREAMLIT_THEME = "light"
    ENABLE_WIDE_MODE = True
    
    # Performance (relaxed in dev)
    CACHE_ENABLED = False  # Disable cache to see fresh results
    


class ProductionConfig(Config):
    """Production environment settings"""
    # Optimize for performance
    DEBUG = False
    VERBOSE_LOGGING = False
    SHOW_TIMING = False
    SHOW_CHUNK_SCORES = False
    
    # Optimized LLM settings
    LLM_MAX_TOKENS = 1024
    
    # Streamlit settings
    STREAMLIT_THEME = "light"
    ENABLE_WIDE_MODE = False
    
    # Performance (optimized for prod)
    CACHE_ENABLED = True
    
    # Security
    ENABLE_XSRF_PROTECTION = True
    ENABLE_CORS = False


# Determine environment
def get_config() -> Config:
    """
    Get configuration based on environment
    
    Checks RAILWAY_ENVIRONMENT or ENV variable:
    - 'production' → ProductionConfig
    - 'development' or missing → DevelopmentConfig
    """
    env = os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("ENV", "development")
    
    if env == "production":
        return ProductionConfig()
    else:
        return DevelopmentConfig()


# Global config instance
config = get_config()


def print_config_info():
    """Print current configuration (for debugging)"""
    env = "PRODUCTION" if isinstance(config, ProductionConfig) else "DEVELOPMENT"
    print(f"\n{'='*50}")
    print(f"🔧 Configuration: {env} MODE")
    print(f"{'='*50}")
    print(f"Debug: {config.DEBUG}")
    print(f"Verbose Logging: {config.VERBOSE_LOGGING}")
    print(f"LLM Model: {config.GROQ_MODEL}")
    print(f"LLM Max Tokens: {config.LLM_MAX_TOKENS}")
    print(f"Cache Enabled: {config.CACHE_ENABLED}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Test configuration
    print_config_info()
    print("Available settings:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
