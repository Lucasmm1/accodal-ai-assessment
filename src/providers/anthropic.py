from anthropic import (
    Anthropic,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError as AnthropicRateLimitError,
)

from src.errors import (
    ProviderServerError,
    RateLimitError,
    ProviderTimeoutError,
)
from src.providers.base import LLMProvider


class AnthropicProvider(LLMProvider):

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, timeout=None, response_schema=None):
        try:
            if response_schema is not None:
                response = self.client.messages.parse(
                    model=self.model,
                    max_tokens=1024,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    output_format=response_schema,
                    timeout=timeout,
                )

                return response.parsed_output

            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                timeout=timeout,
            )

            return response.content[0].text

        except AnthropicRateLimitError as exc:
            raise RateLimitError(
                "Anthropic rate limit exceeded"
            ) from exc

        except APITimeoutError as exc:
            raise ProviderTimeoutError(
                "Anthropic request timed out"
            ) from exc

        except APIStatusError as exc:
            if exc.status_code >= 500:
                raise ProviderServerError(
                    "Anthropic server error"
                ) from exc

            raise ProviderServerError(
                "Anthropic API request failed"
            ) from exc

        except APIConnectionError as exc:
            raise ProviderServerError(
                "Could not connect to Anthropic"
            ) from exc