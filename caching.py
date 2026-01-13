"""
Caching Module for Embeddings
Reduces API costs by caching query embeddings
"""

import hashlib
import pickle
import os
import time
from typing import Optional, Any, Dict
from datetime import datetime, timedelta
from logger import logger


class EmbeddingCache:
    """Cache for query embeddings with TTL support"""
    
    def __init__(
        self,
        cache_dir: str = ".cache",
        ttl_seconds: int = 86400,  # 24 hours
        max_size: int = 1000
    ):
        """
        Initialize embedding cache
        
        Args:
            cache_dir: Directory to store cache files
            ttl_seconds: Time-to-live for cache entries
            max_size: Maximum number of cached items
        """
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.memory_cache: Dict[str, tuple] = {}  # {key: (value, timestamp)}
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing cache from disk
        self._load_cache()
        
        logger.info(
            f"Embedding cache initialized: dir={cache_dir}, ttl={ttl_seconds}s, "
            f"max_size={max_size}, loaded={len(self.memory_cache)} items"
        )
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from query"""
        # Use hash for consistent, short keys
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> str:
        """Get file path for cache key"""
        return os.path.join(self.cache_dir, f"{key}.pkl")
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry is expired"""
        return time.time() - timestamp > self.ttl_seconds
    
    def _load_cache(self):
        """Load cache from disk"""
        if not os.path.exists(self.cache_dir):
            return
        
        loaded = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                key = filename[:-4]
                try:
                    with open(os.path.join(self.cache_dir, filename), 'rb') as f:
                        value, timestamp = pickle.load(f)
                        
                        # Only load non-expired entries
                        if not self._is_expired(timestamp):
                            self.memory_cache[key] = (value, timestamp)
                            loaded += 1
                        else:
                            # Delete expired file
                            os.remove(os.path.join(self.cache_dir, filename))
                except Exception as e:
                    logger.warning(f"Failed to load cache file {filename}: {e}")
        
        if loaded > 0:
            logger.info(f"Loaded {loaded} cached embeddings from disk")
    
    def get(self, query: str) -> Optional[Any]:
        """
        Get cached embedding for query
        
        Args:
            query: User query
            
        Returns:
            Cached embedding or None if not found/expired
        """
        key = self._get_cache_key(query)
        
        # Check memory cache first
        if key in self.memory_cache:
            value, timestamp = self.memory_cache[key]
            
            if self._is_expired(timestamp):
                # Expired, remove it
                logger.debug(f"Cache expired for query: {query[:50]}")
                self._delete(key)
                return None
            
            logger.info(f"Cache HIT for query: {query[:50]}")
            return value
        
        logger.debug(f"Cache MISS for query: {query[:50]}")
        return None
    
    def set(self, query: str, embedding: Any):
        """
        Cache embedding for query
        
        Args:
            query: User query
            embedding: Query embedding (numpy array)
        """
        key = self._get_cache_key(query)
        timestamp = time.time()
        
        # Check if we need to evict old entries
        if len(self.memory_cache) >= self.max_size:
            self._evict_oldest()
        
        # Store in memory
        self.memory_cache[key] = (embedding, timestamp)
        
        # Store on disk for persistence
        try:
            cache_path = self._get_cache_path(key)
            with open(cache_path, 'wb') as f:
                pickle.dump((embedding, timestamp), f)
            logger.debug(f"Cached embedding for query: {query[:50]}")
        except Exception as e:
            logger.warning(f"Failed to write cache to disk: {e}")
    
    def _delete(self, key: str):
        """Delete cache entry"""
        # Remove from memory
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        # Remove from disk
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except Exception as e:
                logger.warning(f"Failed to delete cache file: {e}")
    
    def _evict_oldest(self):
        """Evict oldest cache entry to make room"""
        if not self.memory_cache:
            return
        
        # Find oldest entry
        oldest_key = min(self.memory_cache.keys(), key=lambda k: self.memory_cache[k][1])
        logger.debug(f"Evicting oldest cache entry: {oldest_key}")
        self._delete(oldest_key)
    
    def clear(self):
        """Clear all cache"""
        self.memory_cache.clear()
        
        # Delete all cache files
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    try:
                        os.remove(os.path.join(self.cache_dir, filename))
                    except Exception as e:
                        logger.warning(f"Failed to delete cache file {filename}: {e}")
        
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        now = time.time()
        valid_entries = sum(
            1 for _, (_, ts) in self.memory_cache.items()
            if not self._is_expired(ts)
        )
        
        return {
            "total_entries": len(self.memory_cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self.memory_cache) - valid_entries,
            "max_size": self.max_size,
            "utilization_percent": (len(self.memory_cache) / self.max_size) * 100,
            "ttl_seconds": self.ttl_seconds,
            "cache_dir": self.cache_dir
        }
    
    def cleanup_expired(self):
        """Remove all expired entries"""
        expired_keys = [
            key for key, (_, ts) in self.memory_cache.items()
            if self._is_expired(ts)
        ]
        
        for key in expired_keys:
            self._delete(key)
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")


# Global embedding cache instance
embedding_cache = EmbeddingCache(
    cache_dir=".cache/embeddings",
    ttl_seconds=86400,  # 24 hours
    max_size=1000
)


if __name__ == "__main__":
    # Test caching
    print("Testing Embedding Cache...")
    
    # Create test cache
    test_cache = EmbeddingCache(cache_dir=".cache/test", ttl_seconds=5, max_size=3)
    
    # Test set and get
    import numpy as np
    
    query1 = "What are the FYP requirements?"
    embedding1 = np.random.rand(384)
    
    print(f"\n1. Caching embedding for: '{query1}'")
    test_cache.set(query1, embedding1)
    
    print(f"2. Retrieving cached embedding...")
    cached = test_cache.get(query1)
    print(f"   Cache hit: {cached is not None}")
    print(f"   Arrays equal: {np.array_equal(cached, embedding1) if cached is not None else False}")
    
    # Test cache miss
    print(f"\n3. Testing cache miss...")
    cached = test_cache.get("Different query")
    print(f"   Cache hit: {cached is not None}")
    
    # Test max size eviction
    print(f"\n4. Testing max size eviction (max=3)...")
    for i in range(4):
        test_cache.set(f"Query {i}", np.random.rand(384))
        print(f"   Cached query {i}, total entries: {len(test_cache.memory_cache)}")
    
    # Test expiration
    print(f"\n5. Testing TTL expiration (ttl=5s)...")
    print(f"   Waiting 6 seconds...")
    time.sleep(6)
    cached = test_cache.get(query1)
    print(f"   Cache hit after expiration: {cached is not None}")
    
    # Show stats
    print(f"\n6. Cache statistics:")
    stats = test_cache.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Cleanup
    print(f"\n7. Cleaning up test cache...")
    test_cache.clear()
    print(f"   Total entries after clear: {len(test_cache.memory_cache)}")
