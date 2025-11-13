"""
Utility functions for the fake news verification system.
Includes caching, retry logic, validation, and helper functions.
"""

import os
import json
import hashlib
import time
from typing import Any, Dict, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple in-memory cache
_cache = {}
_cache_timestamps = {}


def get_cache_key(*args, **kwargs) -> str:
    """Generate a cache key from arguments, handling non-serializable objects."""
    # Filter out 'self' and other non-serializable objects
    serializable_args = []
    for arg in args:
        if isinstance(arg, (str, int, float, bool, type(None))):
            serializable_args.append(arg)
        elif isinstance(arg, (list, tuple, dict)):
            try:
                json.dumps(arg)
                serializable_args.append(arg)
            except (TypeError, ValueError):
                # Skip non-serializable complex objects
                pass
        else:
            # For objects, use their string representation
            serializable_args.append(str(type(arg).__name__))
    
    serializable_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            serializable_kwargs[key] = value
        elif isinstance(value, (list, tuple, dict)):
            try:
                json.dumps(value)
                serializable_kwargs[key] = value
            except (TypeError, ValueError):
                pass
    
    key_data = json.dumps({"args": serializable_args, "kwargs": serializable_kwargs}, sort_keys=True)
    return hashlib.md5(key_data.encode()).hexdigest()


def cache_result(ttl_seconds: int = 3600):
    """
    Decorator to cache function results.
    
    Args:
        ttl_seconds: Time to live for cached results in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from arguments
            try:
                cache_key = f"{func.__name__}_{get_cache_key(*args, **kwargs)}"
            except Exception as e:
                logger.warning(f"Failed to generate cache key: {e}, calling function without cache")
                return func(*args, **kwargs)
            
            # Check if cached result exists and is still valid
            if cache_key in _cache and cache_key in _cache_timestamps:
                cached_time = _cache_timestamps[cache_key]
                if time.time() - cached_time < ttl_seconds:
                    logger.info(f"Cache hit for {func.__name__}")
                    return _cache[cache_key]
            
            # Call function and cache result
            result = func(*args, **kwargs)
            _cache[cache_key] = result
            _cache_timestamps[cache_key] = time.time()
            logger.info(f"Cache miss for {func.__name__}, result cached")
            return result
        
        return wrapper
    return decorator


def retry_on_failure(max_attempts: int = 3):
    """
    Decorator to retry function on failure with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )


def validate_json_schema(data: Any, required_fields: list) -> bool:
    """
    Validate that data is a dict and contains all required fields.
    
    Args:
        data: Data to validate
        required_fields: List of required field names
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(data, dict):
        return False
    
    for field in required_fields:
        if field not in data:
            return False
    
    return True


def parse_json_response(text: str) -> Optional[Dict]:
    """
    Parse JSON from LLM response, handling common formatting issues.
    
    Args:
        text: Text response that may contain JSON
        
    Returns:
        Parsed JSON dict or None if parsing fails
    """
    # Try direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks
    if "```json" in text:
        try:
            start = text.index("```json") + 7
            end = text.index("```", start)
            return json.loads(text[start:end].strip())
        except (ValueError, json.JSONDecodeError):
            pass
    
    # Try to extract JSON from curly braces
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        pass
    
    return None


def calculate_trust_score(domain: str, published_date: Optional[str] = None) -> float:
    """
    Calculate trust score for a source based on domain and recency.
    
    Args:
        domain: Domain name of the source
        published_date: Publication date string (optional)
        
    Returns:
        Trust score between 0 and 1
    """
    score = 0.0
    domain_lower = domain.lower()
    
    # Known fact-checking sites
    fact_check_domains = [
        "snopes.com", "factcheck.org", "politifact.com", "fullfact.org",
        "apnews.com/APFactCheck", "reuters.com/fact-check", "bbc.com/news/reality_check",
        "factchecker.in", "indiatoday.in/fact-check", "altnews.in"
    ]
    
    # Reputable news outlets
    reputable_news = [
        "bbc.com", "reuters.com", "apnews.com", "nytimes.com", "theguardian.com",
        "washingtonpost.com", "wsj.com", "npr.org", "cnn.com", "nbcnews.com",
        "aljazeera.com", "thehindu.com", "indianexpress.com", "timesofindia.com"
    ]
    
    # Low-trust indicators
    low_trust_indicators = [
        "blog", "wordpress", "blogspot", "tumblr", "medium.com"
    ]
    
    # Score based on domain
    if any(fc_domain in domain_lower for fc_domain in fact_check_domains):
        score += 0.6
    elif any(news_domain in domain_lower for news_domain in reputable_news):
        score += 0.4
    else:
        score += 0.2  # Neutral baseline
    
    # Penalize low-trust indicators
    if any(indicator in domain_lower for indicator in low_trust_indicators):
        score -= 0.2
    
    # Bonus for recency (within 30 days)
    if published_date:
        try:
            from dateutil import parser
            pub_date = parser.parse(published_date)
            days_old = (datetime.now() - pub_date).days
            if days_old <= 30:
                score += 0.1
        except:
            pass
    
    # Normalize to 0-1 range
    return max(0.0, min(1.0, score))


def extract_domain(url: str) -> str:
    """
    Extract domain from URL.
    
    Args:
        url: Full URL
        
    Returns:
        Domain name
    """
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
        return parsed.netloc or parsed.path
    except:
        return url


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove control characters
    text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t")
    
    return text.strip()


def truncate_text(text: str, max_length: int = 500) -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def format_confidence(confidence: float) -> str:
    """
    Format confidence score as percentage string.
    
    Args:
        confidence: Confidence score (0-1)
        
    Returns:
        Formatted string like "85%"
    """
    return f"{int(confidence * 100)}%"


def get_verdict_color(verdict: str) -> str:
    """
    Get color code for verdict display.
    
    Args:
        verdict: Verdict string
        
    Returns:
        Color name or hex code
    """
    verdict_lower = verdict.lower()
    
    if verdict_lower == "true":
        return "#2ecc71"  # Green
    elif verdict_lower == "false":
        return "#e74c3c"  # Red
    elif verdict_lower == "mixed":
        return "#f39c12"  # Orange
    elif verdict_lower == "insufficient":
        return "#95a5a6"  # Gray
    else:
        return "#3498db"  # Blue (default)


def get_verdict_emoji(verdict: str) -> str:
    """
    Get emoji for verdict.
    
    Args:
        verdict: Verdict string
        
    Returns:
        Emoji character
    """
    verdict_lower = verdict.lower()
    
    if verdict_lower == "true":
        return "✅"
    elif verdict_lower == "false":
        return "❌"
    elif verdict_lower == "mixed":
        return "⚠️"
    elif verdict_lower == "insufficient":
        return "❓"
    else:
        return "ℹ️"


class RateLimiter:
    """Simple rate limiter to avoid API throttling."""
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls if now - call_time < 60]
        
        if len(self.calls) >= self.calls_per_minute:
            # Wait until oldest call is more than 1 minute old
            wait_time = 60 - (now - self.calls[0]) + 0.1
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
        
        self.calls.append(now)


# Global rate limiters for different APIs
serper_limiter = RateLimiter(calls_per_minute=30)
openrouter_limiter = RateLimiter(calls_per_minute=10)  # More conservative for free tier
gemini_limiter = RateLimiter(calls_per_minute=15)
