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
        def generate(self, prompt, timeout=None):
            return {"answer": "Hello"}

    provider = SchemaProvider()
    harness = LLMHarness(provider)

    result = harness.generate("Hello", response_schema=SimpleResponse)

    assert isinstance(result, SimpleResponse)
    assert result.answer == "Hello"

def test_harness_rejects_invalid_output_schema():
    class SchemaProvider:
        def generate(self, prompt, timeout=None):
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

def test_harness_passes_timeout_to_provider():

    class TimeoutProvider:
        def __init__(self):
            self.received_timeout = None

        def generate(self, prompt, timeout=None):
            self.received_timeout = timeout
            return "success"

    provider = TimeoutProvider()

    harness = LLMHarness(
        provider,
        timeout=15,
    )

    result = harness.generate("Hello")

    assert result == "success"
    assert provider.received_timeout == 15

def test_harness_redacts_sensitive_fields_in_response_log(caplog):

    class SensitiveProvider:
        def generate(self, prompt, timeout=None):
            return {
                "resident_name": "Maria",
                "medical_history": "Diabetes",
                "status": "success",
            }

    provider = SensitiveProvider()
    harness = LLMHarness(provider)

    with caplog.at_level("INFO"):
        harness.generate("Hello")

    assert "[REDACTED]" in caplog.text
    assert "Maria" not in caplog.text
    assert "Diabetes" not in caplog.text
    assert "success" in caplog.text

def test_harness_retries_anthropic_rate_limit():
    from src.providers.anthropic import AnthropicProvider

    delays = []

    provider = AnthropicProvider(api_key="test-key")

    class FakeClient:
        class Messages:
            attempts = 0

            def create(self, **kwargs):
                self.attempts += 1

                if self.attempts == 1:
                    import httpx
                    from anthropic import RateLimitError as AnthropicRateLimitError

                    request = httpx.Request(
                        "POST",
                        "https://api.anthropic.com/v1/messages",
                    )

                    response = httpx.Response(
                        429,
                        request=request,
                    )

                    raise AnthropicRateLimitError(
                        message="rate limited",
                        response=response,
                        body=None,
                    )

                return type(
                    "Response",
                    (),
                    {
                        "content": [
                            type("Content", (), {"text": "Success"})()
                        ]
                    },
                )()

        messages = Messages()

    provider.client = FakeClient()

    harness = LLMHarness(
        provider,
        sleep_function=lambda seconds: delays.append(seconds),
    )

    result = harness.generate("Hello")

    assert result == "Success"
    assert provider.client.messages.attempts == 2
    assert delays == [1]