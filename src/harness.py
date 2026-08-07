from src.providers.base import LLMProvider
from src.errors import HarnessError, InvalidInputError

class LLMHarness:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def generate(self, prompt):
        if not isinstance(prompt, str):
            raise InvalidInputError("Prompt should be a string")
        if not prompt.strip():
            raise InvalidInputError("Prompt cannot be empty")

        return self.provider.generate(prompt)
