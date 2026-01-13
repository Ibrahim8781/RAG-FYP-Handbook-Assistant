"""
Error Handling Module
Provides retry logic, custom exceptions, and graceful error handling
"""

import time
import functools
from typing import Callable, Any, Optional, Type
from logger import logger, log_error


class RAGError(Exception):
    """Base exception for RAG system"""
    pass


class APIError(RAGError):
    """Error calling external API (Groq, etc.)"""
    pass


class RetrievalError(RAGError):
    """Error during vector retrieval"""
    pass


class EmbeddingError(RAGError):
    """Error generating embeddings"""
    pass


class ValidationError(RAGError):
    """Input validation error"""
    pass


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
        
    Usage:
        @retry_with_backoff(max_retries=3, initial_delay=1.0)
        def call_api():
            # API call that might fail
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        # Final attempt failed
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries",
                            exc_info=True
                        )
                        break
                    
                    # Log retry attempt
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    
                    time.sleep(delay)
                    delay *= backoff_factor
            
            # If we get here, all retries failed
            raise APIError(
                f"Failed after {max_retries} retries: {str(last_exception)}"
            ) from last_exception
        
        return wrapper
    return decorator


def handle_errors(
    error_message: str = "An error occurred",
    return_value: Any = None,
    log: bool = True
):
    """
    Decorator for graceful error handling
    
    Args:
        error_message: User-friendly error message
        return_value: Value to return on error
        log: Whether to log the error
        
    Usage:
        @handle_errors(error_message="Failed to retrieve chunks", return_value=[])
        def retrieve_chunks(query):
            # Retrieval logic
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log:
                    log_error(e, {"function": func.__name__, "args": str(args)[:100]})
                
                # Return error info
                return {
                    "success": False,
                    "error": error_message,
                    "error_detail": str(e),
                    "data": return_value
                }
        
        return wrapper
    return decorator


def validate_input(
    query: str,
    min_length: int = 3,
    max_length: int = 500
) -> tuple[bool, Optional[str]]:
    """
    Validate user query input
    
    Args:
        query: User's question
        min_length: Minimum query length
        max_length: Maximum query length
        
    Returns:
        (is_valid, error_message)
    """
    if not query or not query.strip():
        return False, "Query cannot be empty"
    
    query = query.strip()
    
    if len(query) < min_length:
        return False, f"Query too short (minimum {min_length} characters)"
    
    if len(query) > max_length:
        return False, f"Query too long (maximum {max_length} characters)"
    
    return True, None


def safe_api_call(func: Callable, *args, **kwargs) -> dict:
    """
    Safely execute an API call with error handling
    
    Args:
        func: Function to call
        *args, **kwargs: Arguments for the function
        
    Returns:
        Dictionary with success status and result/error
    """
    try:
        result = func(*args, **kwargs)
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        log_error(e, {"function": func.__name__})
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


class ErrorContext:
    """Context manager for error handling"""
    
    def __init__(self, operation: str, raise_on_error: bool = False):
        self.operation = operation
        self.raise_on_error = raise_on_error
    
    def __enter__(self):
        logger.debug(f"Starting operation: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            log_error(exc_val, {"operation": self.operation})
            
            if self.raise_on_error:
                return False  # Re-raise exception
            else:
                logger.info(f"Operation {self.operation} failed, continuing...")
                return True  # Suppress exception


if __name__ == "__main__":
    # Test retry logic
    print("Testing retry with backoff...")
    
    attempt_count = 0
    
    @retry_with_backoff(max_retries=3, initial_delay=0.5)
    def flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError(f"Attempt {attempt_count} failed")
        return "Success!"
    
    try:
        result = flaky_function()
        print(f"Result: {result}")
    except APIError as e:
        print(f"Failed: {e}")
    
    # Test input validation
    print("\nTesting input validation...")
    tests = ["", "ab", "Valid query", "x" * 600]
    for test in tests:
        valid, error = validate_input(test)
        print(f"'{test[:20]}...': {valid} - {error}")
