class HarnessError(Exception):
    """Base exception for the LLM harness"""

class InvalidInputError(HarnessError):
    """Raised when the input does not meet the expected schema"""

class InvalidOutputError(HarnessError):
    """Raised when the output does not meet the expected schema"""

class LLMProviderError(HarnessError):
    """Raised when the LLM provider fails"""

class RateLimitError(LLMProviderError):
    """Raised when the LLM provider rate limits the request"""

class ProviderServerError(LLMProviderError):
    """Raised when the LLM provider returns a server-side error"""