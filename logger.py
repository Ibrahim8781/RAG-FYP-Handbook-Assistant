"""
Structured Logging Module with JSON Formatter
Provides consistent logging across the RAG application
"""

import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict
from config_env import config


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        return json.dumps(log_data)


class StandardFormatter(logging.Formatter):
    """Human-readable formatter for development"""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


def setup_logger(name: str = "rag_app") -> logging.Logger:
    """
    Setup logger with appropriate formatter based on environment
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Set level based on environment
    if config.DEBUG:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    
    # Use JSON formatter in production, standard in development
    if config.DEBUG:
        handler.setFormatter(StandardFormatter())
    else:
        handler.setFormatter(JSONFormatter())
    
    logger.addHandler(handler)
    
    return logger


class LogContext:
    """Context manager for adding extra fields to logs"""
    
    def __init__(self, logger: logging.Logger, **kwargs):
        self.logger = logger
        self.extra_data = kwargs
        self.old_factory = None
    
    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            record.extra_data = self.extra_data
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


# Global logger instance
logger = setup_logger("rag_app")


def log_query(query: str, user_id: str = "anonymous"):
    """Log a user query"""
    with LogContext(logger, query=query, user_id=user_id):
        logger.info(f"Query received: {query[:100]}")


def log_retrieval(num_chunks: int, top_score: float, query_time: float):
    """Log retrieval metrics"""
    with LogContext(logger, num_chunks=num_chunks, top_score=top_score, query_time=query_time):
        logger.info(f"Retrieved {num_chunks} chunks in {query_time:.3f}s")


def log_llm_call(model: str, tokens: int, latency: float, success: bool):
    """Log LLM API call"""
    with LogContext(logger, model=model, tokens=tokens, latency=latency, success=success):
        if success:
            logger.info(f"LLM call successful: {model} ({tokens} tokens, {latency:.2f}s)")
        else:
            logger.error(f"LLM call failed: {model}")


def log_error(error: Exception, context: Dict[str, Any] = None):
    """Log an error with context"""
    with LogContext(logger, error_type=type(error).__name__, context=context or {}):
        logger.error(f"Error occurred: {str(error)}", exc_info=True)


if __name__ == "__main__":
    # Test logging
    print("Testing JSON Logging...")
    
    logger.info("Application started")
    logger.debug("Debug message (should show in dev mode)")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test with context
    log_query("What are the FYP requirements?", user_id="test_user_123")
    log_retrieval(5, 0.87, 0.234)
    log_llm_call("llama-3.1-8b-instant", 512, 1.45, True)
    
    try:
        raise ValueError("Test error")
    except Exception as e:
        log_error(e, {"query": "test query", "step": "retrieval"})
