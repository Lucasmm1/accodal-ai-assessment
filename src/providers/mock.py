from .base import LLMProvider
from src.errors import (
    ProviderServerError,
    RateLimitError,
    ProviderTimeoutError,
)


class MockProvider(LLMProvider):

    def __init__(
        self,
        server_error_times=0,
        rate_limit_times=0,
        timeout_times=0,
        response=None,
        responses=None,
        fail_on_attempts=None,
    ):
        self.server_error_times = server_error_times
        self.rate_limit_times = rate_limit_times
        self.timeout_times = timeout_times
        self.responses = responses
        self.response = response
        self.fail_on_attempts = fail_on_attempts or []
        self.attempts = 0

    def generate(
        self,
        prompt: str,
        timeout=None,
        response_schema=None,
    ):
        self.attempts += 1

        if self.attempts in self.fail_on_attempts:
            raise ProviderServerError(
                "Simulated provider server error"
            )

        if self.attempts <= self.rate_limit_times:
            raise RateLimitError(
                "Simulated rate limit"
            )

        if self.attempts <= self.server_error_times:
            raise ProviderServerError(
                "Simulated provider server error"
            )

        if self.attempts <= self.timeout_times:
            raise ProviderTimeoutError(
                "Simulated provider timeout"
            )

        if self.responses is not None:
            response_index = self.attempts - 1

            if response_index < len(self.responses):
                return self.responses[response_index]

        if self.response is not None:
            return self.response

        return f"Mock response to: {prompt}"