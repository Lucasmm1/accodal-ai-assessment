from .base import LLMProvider
from src.errors import ProviderServerError, RateLimitError, ProviderTimeoutError

class MockProvider(LLMProvider):
    def __init__(self, server_error_times=0, rate_limit_times=0, timeout_times=0):
        self.server_error_times = server_error_times
        self.rate_limit_times = rate_limit_times
        self.timeout_times = timeout_times
        self.attempts = 0

    def generate(self, prompt: str, timeout=None) -> str:
        self.attempts += 1

        if self.attempts <= self.rate_limit_times:
            raise RateLimitError("Simulated rate limit")

        if self.attempts <= self.server_error_times:
            raise ProviderServerError("Simulated provider server error")

        if self.attempts <= self.timeout_times:
            raise ProviderTimeoutError("Simulated provider timeout")

        return f"Mock response to: {prompt}"