import pytest

from src.logger import logger

from src.harness import LLMHarness
from src.schemas import SimpleResponse
from src.providers.mock import MockProvider


from src.utils import redact_sensitive_fields
from src.errors import InvalidInputError, InvalidOutputError, RateLimitError, ProviderTimeoutError

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

def test_harness_validates_output_schema():
    class SchemaProvider:
        def generate(self, prompt):
            return {"answer": "Hello"}

    provider = SchemaProvider()
    harness = LLMHarness(provider)

    result = harness.generate("Hello", response_schema=SimpleResponse)

    assert isinstance(result, SimpleResponse)
    assert result.answer == "Hello"

def test_harness_rejects_invalid_output_schema():
    class SchemaProvider:
        def generate(self, prompt):
            return {"answer": 123}

    provider = SchemaProvider()
    harness = LLMHarness(provider)

    with pytest.raises(InvalidOutputError):
        harness.generate("Hello", response_schema=SimpleResponse)

def test_redacts_sensitive_fields():
    data = {
        "resident_name": "Maria",
        "medical_history": "Diabetes",
        "status": "success",
    }

    result = redact_sensitive_fields(data)

    assert result["resident_name"] == "[REDACTED]"
    assert result["medical_history"] == "[REDACTED]"
    assert result["status"] == "success"

def test_logger_records_event(caplog):
    
    with caplog.at_level("INFO"):
        logger.info("LLM request started")

    assert "LLM request started" in caplog.text

def test_harness_logs_request_start(caplog):
    provider = MockProvider()
    harness = LLMHarness(provider)

    with caplog.at_level("INFO"):
        harness.generate("Hello")

    assert "LLM request started" in caplog.text

def test_harness_logs_retry(caplog):
    provider = MockProvider(rate_limit_times=1)
    harness = LLMHarness(provider)

    with caplog.at_level("WARNING"):
        harness.generate("Hello")

    assert "LLM request failed on attempt 1, retrying" in caplog.text

def test_harness_logs_success(caplog):
    provider = MockProvider()
    harness = LLMHarness(provider)

    with caplog.at_level("INFO"):
        harness.generate("Hello")

    assert "LLM request succeeded" in caplog.text

def test_harness_uses_fallback_on_timeout():
    provider = MockProvider(timeout_times=1)

    harness = LLMHarness(provider, fallback_response="Service temporarily unavailable")

    result = harness.generate("Hello")

    assert result == "Service temporarily unavailable"

def test_harness_validates_fallback_against_schema():
    provider = MockProvider(timeout_times=1)

    harness = LLMHarness(provider, fallback_response={"answer": "Service temporarily unavailable"})

    result = harness.generate("Hello", response_schema=SimpleResponse)

    assert isinstance(result, SimpleResponse)
    assert result.answer == "Service temporarily unavailable"

def test_harness_rejects_invalid_fallback_schema():
    provider = MockProvider(timeout_times=1)

    harness = LLMHarness(provider, fallback_response={"answer": 123})

    with pytest.raises(InvalidOutputError):
        harness.generate("Hello", response_schema=SimpleResponse)