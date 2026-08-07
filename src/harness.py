import time
from src.providers.base import LLMProvider
from src.errors import HarnessError, InvalidInputError, ProviderServerError, RateLimitError

class LLMHarness:
    def __init__(self, provider: LLMProvider, sleep_function = time.sleep):
        self.provider = provider
        self.sleep_function = sleep_function

    def generate(self, prompt):
        if not isinstance(prompt, str):
            raise InvalidInputError("Prompt should be a string")
        
        if not prompt.strip():
            raise InvalidInputError("Prompt cannot be empty")

        MAX_RETRIES = 3

        for attempt in range(MAX_RETRIES):
            try:
                return self.provider.generate(prompt)
            except (RateLimitError, ProviderServerError):
                if attempt == MAX_RETRIES - 1:
                    raise

                delay = 2 ** attempt
                self.sleep_function(delay)