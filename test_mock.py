from src.harness import LLMHarness
from src.providers.mock import MockProvider

provider = MockProvider()
harness = LLMHarness(provider)

response = harness.generate("Hello, AI")

print(response)