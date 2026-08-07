from .base import LLMProvider

class MockProvider(LLMProvider):
    def generate(self, prompt: str) -> str:
        return f"Mock response to: {prompt}"