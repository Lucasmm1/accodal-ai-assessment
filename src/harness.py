import time
from src.logger import logger
from pydantic import ValidationError
from src.providers.base import LLMProvider
from src.errors import HarnessError, InvalidInputError, InvalidOutputError, ProviderServerError, RateLimitError, ProviderTimeoutError

class LLMHarness:
    def __init__(self, provider: LLMProvider, sleep_function=time.sleep, fallback_response=None):
        self.provider = provider
        self.sleep_function = sleep_function
        self.fallback_response = fallback_response

    def generate(self, prompt, response_schema=None):
        if not isinstance(prompt, str):
            raise InvalidInputError("Prompt should be a string")
        
        if not prompt.strip():
            raise InvalidInputError("Prompt cannot be empty")

        MAX_RETRIES = 3

        logger.info("LLM request started")

        for attempt in range(MAX_RETRIES):
            try:
                response = self.provider.generate(prompt)

                if response_schema:
                    try:
                        return response_schema.model_validate(response)
                    except ValidationError as exc:
                        raise InvalidOutputError("Provider returned invalid output") from exc

                logger.info("LLM request succeeded")
                return response

            except (RateLimitError, ProviderServerError):
                if attempt == MAX_RETRIES - 1:
                    raise

                logger.warning(f"LLM request failed on attempt {attempt + 1}, retrying")

                delay = 2 ** attempt
                self.sleep_function(delay)

            except ProviderTimeoutError:
                logger.warning("LLM request timed out, using fallback")

                if response_schema:
                    try:
                        return response_schema.model_validate(self.fallback_response)
                    except ValidationError as exc:
                        raise InvalidOutputError("Fallback response does not match the expected schema") from exc

                return self.fallback_response