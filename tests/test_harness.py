import pytest

from src.harness import LLMHarness
from src.providers.mock import MockProvider
from src.errors import InvalidInputError, RateLimitError

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

def test_harness_retries_rate_limit():
    delays = []

    def fake_sleep(seconds):
        delays.append(seconds)
        
    provider = MockProvider(rate_limit_times=2)
    harness = LLMHarness(provider, sleep_function=fake_sleep)

    result = harness.generate("Hello")
    assert result == "Mock response to: Hello"
    assert provider.attempts == 3
    assert delays == [1,2]

def test_harness_stops_after_max_retries():

    def fake_sleep(seconds):
        pass

    provider = MockProvider(rate_limit_times=10)
    harness = LLMHarness(provider, sleep_function=fake_sleep)

    with pytest.raises(RateLimitError):
        harness.generate("Hello")

    assert provider.attempts == 3

def test_harness_retries_server_error():
    delays = []

    def fake_sleep(seconds):
        delays.append(seconds)

    provider = MockProvider(server_error_times=2)
    harness = LLMHarness(provider, sleep_function=fake_sleep)

    result = harness.generate("Hello")

    assert result == "Mock response to: Hello"
    assert provider.attempts == 3
    assert delays == [1, 2]