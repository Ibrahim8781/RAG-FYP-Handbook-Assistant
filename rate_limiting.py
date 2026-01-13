"""
Rate Limiting Module
Prevents abuse and manages request quotas
"""

import time
from collections import deque
from datetime import datetime, timedelta
from typing import Optional, Dict
from logger import logger


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(
        self,
        max_requests: int,
        time_window: int,
        name: str = "default"
    ):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum requests allowed in time window
            time_window: Time window in seconds
            name: Name of this rate limiter (for logging)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.name = name
        self.requests: deque = deque()
        logger.info(f"Rate limiter '{name}' initialized: {max_requests} requests per {time_window}s")
    
    def is_allowed(self, identifier: str = "default") -> tuple[bool, Optional[float]]:
        """
        Check if request is allowed
        
        Args:
            identifier: Unique identifier for the requester (IP, user_id, etc.)
            
        Returns:
            (is_allowed, retry_after_seconds)
        """
        now = time.time()
        
        # Remove old requests outside the time window
        while self.requests and self.requests[0][0] < now - self.time_window:
            self.requests.popleft()
        
        # Check if under limit
        if len(self.requests) < self.max_requests:
            self.requests.append((now, identifier))
            logger.debug(f"Rate limiter '{self.name}': Request allowed ({len(self.requests)}/{self.max_requests})")
            return True, None
        
        # Rate limit exceeded
        oldest_request_time = self.requests[0][0]
        retry_after = (oldest_request_time + self.time_window) - now
        
        logger.warning(
            f"Rate limiter '{self.name}': Request denied for '{identifier}'. "
            f"Retry after {retry_after:.1f}s"
        )
        
        return False, retry_after
    
    def get_status(self) -> Dict:
        """Get current rate limiter status"""
        now = time.time()
        
        # Clean old requests
        while self.requests and self.requests[0][0] < now - self.time_window:
            self.requests.popleft()
        
        return {
            "name": self.name,
            "current_requests": len(self.requests),
            "max_requests": self.max_requests,
            "time_window": self.time_window,
            "remaining": self.max_requests - len(self.requests),
            "utilization_percent": (len(self.requests) / self.max_requests) * 100
        }


class PerUserRateLimiter:
    """Rate limiter with per-user tracking"""
    
    def __init__(
        self,
        max_requests: int,
        time_window: int,
        name: str = "per_user"
    ):
        """
        Initialize per-user rate limiter
        
        Args:
            max_requests: Max requests per user in time window
            time_window: Time window in seconds
            name: Name of this rate limiter
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.name = name
        self.user_requests: Dict[str, deque] = {}
        logger.info(f"Per-user rate limiter '{name}' initialized: {max_requests} requests per {time_window}s")
    
    def is_allowed(self, user_id: str) -> tuple[bool, Optional[float]]:
        """
        Check if request is allowed for specific user
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            (is_allowed, retry_after_seconds)
        """
        now = time.time()
        
        # Initialize user's request queue if doesn't exist
        if user_id not in self.user_requests:
            self.user_requests[user_id] = deque()
        
        user_queue = self.user_requests[user_id]
        
        # Remove old requests
        while user_queue and user_queue[0] < now - self.time_window:
            user_queue.popleft()
        
        # Check limit
        if len(user_queue) < self.max_requests:
            user_queue.append(now)
            logger.debug(
                f"Rate limiter '{self.name}': Request allowed for user '{user_id}' "
                f"({len(user_queue)}/{self.max_requests})"
            )
            return True, None
        
        # Rate limit exceeded
        oldest_request = user_queue[0]
        retry_after = (oldest_request + self.time_window) - now
        
        logger.warning(
            f"Rate limiter '{self.name}': Request denied for user '{user_id}'. "
            f"Retry after {retry_after:.1f}s"
        )
        
        return False, retry_after
    
    def get_user_status(self, user_id: str) -> Dict:
        """Get rate limit status for specific user"""
        now = time.time()
        
        if user_id not in self.user_requests:
            return {
                "user_id": user_id,
                "current_requests": 0,
                "max_requests": self.max_requests,
                "remaining": self.max_requests
            }
        
        user_queue = self.user_requests[user_id]
        
        # Clean old requests
        while user_queue and user_queue[0] < now - self.time_window:
            user_queue.popleft()
        
        return {
            "user_id": user_id,
            "current_requests": len(user_queue),
            "max_requests": self.max_requests,
            "remaining": self.max_requests - len(user_queue),
            "time_window": self.time_window
        }
    
    def cleanup_inactive_users(self, inactive_threshold: int = 3600):
        """Remove users who haven't made requests recently"""
        now = time.time()
        inactive_users = []
        
        for user_id, queue in self.user_requests.items():
            if not queue or queue[-1] < now - inactive_threshold:
                inactive_users.append(user_id)
        
        for user_id in inactive_users:
            del self.user_requests[user_id]
        
        if inactive_users:
            logger.info(f"Cleaned up {len(inactive_users)} inactive users from rate limiter")


# Global rate limiters
# General query rate limiter: 10 requests per minute
query_rate_limiter = RateLimiter(
    max_requests=10,
    time_window=60,
    name="query_limiter"
)

# Per-user rate limiter: 20 requests per hour
user_rate_limiter = PerUserRateLimiter(
    max_requests=20,
    time_window=3600,
    name="user_query_limiter"
)

# API rate limiter: 30 Groq API calls per minute
api_rate_limiter = RateLimiter(
    max_requests=30,
    time_window=60,
    name="groq_api_limiter"
)


def check_rate_limit(user_id: str = "anonymous") -> tuple[bool, Optional[str]]:
    """
    Check if request is allowed under all rate limits
    
    Args:
        user_id: User identifier
        
    Returns:
        (is_allowed, error_message)
    """
    # Check global query limit
    allowed, retry_after = query_rate_limiter.is_allowed(user_id)
    if not allowed:
        return False, f"Rate limit exceeded. Please try again in {int(retry_after) + 1} seconds."
    
    # Check per-user limit
    allowed, retry_after = user_rate_limiter.is_allowed(user_id)
    if not allowed:
        return False, f"You've exceeded your hourly query limit. Please try again in {int(retry_after/60) + 1} minutes."
    
    return True, None


if __name__ == "__main__":
    # Test rate limiter
    print("Testing Rate Limiter...")
    
    limiter = RateLimiter(max_requests=5, time_window=10, name="test")
    
    # Make 7 requests rapidly
    for i in range(7):
        allowed, retry_after = limiter.is_allowed(f"user_{i%2}")
        status = "✅ Allowed" if allowed else f"❌ Denied (retry in {retry_after:.1f}s)"
        print(f"Request {i+1}: {status}")
        
        # Show status
        if i % 2 == 0:
            print(f"  Status: {limiter.get_status()}")
    
    print("\nTesting Per-User Rate Limiter...")
    user_limiter = PerUserRateLimiter(max_requests=3, time_window=5, name="test_user")
    
    # Test with two users
    for i in range(5):
        user = "user_A" if i < 3 else "user_B"
        allowed, retry_after = user_limiter.is_allowed(user)
        status = "✅ Allowed" if allowed else f"❌ Denied (retry in {retry_after:.1f}s)"
        print(f"{user} Request {i+1}: {status}")
