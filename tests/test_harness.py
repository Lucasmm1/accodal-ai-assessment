import pytest

from src.harness import LLMHarness
from src.providers.mock import MockProvider
from src.errors import InvalidInputError

def test_harness_generates_response():
    provider = MockProvider()
    harness = LLMHarness(provider)

    result = harness.generate("Hello")

    assert result == "Mock response to: Hello"

def test_harness_rejects_empty_prompt():
    provider = MockProvider()
    harness = LLMHarness(provider)

    with pytest.raises(InvalidInputError):
        harness.generate("")