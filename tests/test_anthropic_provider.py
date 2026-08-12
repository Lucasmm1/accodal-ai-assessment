import pytest

from src.errors import ProviderServerError, RateLimitError, ProviderTimeoutError

from src.providers.anthropic import AnthropicProvider


class FakeMessages:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def create(self, **kwargs):
        if self.error:
            raise self.error

        return self.response


class FakeClient:
    def __init__(self, response=None, error=None):
        self.messages = FakeMessages(response=response, error=error)


class FakeResponse:
    def __init__(self, text):
        self.content = [
            type("Content", (), {"text": text})()
        ]


def test_anthropic_provider_returns_text(monkeypatch):
    provider = AnthropicProvider(api_key="test-key")

    provider.client = FakeClient(
        response=FakeResponse("Hello from Claude")
    )

    result = provider.generate("Hello")

    assert result == "Hello from Claude"


def test_anthropic_provider_passes_timeout(monkeypatch):
    provider = AnthropicProvider(api_key="test-key")

    captured = {}

    class FakeMessagesWithCapture:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResponse("Hello")

    provider.client.messages = FakeMessagesWithCapture()

    provider.generate("Hello", timeout=15)

    assert captured["timeout"] == 15


def test_anthropic_provider_maps_rate_limit_error(monkeypatch):
    import httpx
    from anthropic import RateLimitError as AnthropicRateLimitError

    provider = AnthropicProvider(api_key="test-key")

    request = httpx.Request(
        "POST",
        "https://api.anthropic.com/v1/messages",
    )

    response = httpx.Response(
        429,
        request=request,
    )

    error = AnthropicRateLimitError(
        message="rate limited",
        response=response,
        body=None,
    )

    provider.client = FakeClient(error=error)

    with pytest.raises(RateLimitError):
        provider.generate("Hello")


def test_anthropic_provider_maps_timeout_error(monkeypatch):
    from anthropic import APITimeoutError

    provider = AnthropicProvider(api_key="test-key")

    error = APITimeoutError(request=None)

    provider.client = FakeClient(error=error)

    with pytest.raises(ProviderTimeoutError):
        provider.generate("Hello")

def test_anthropic_provider_maps_server_error():
    import httpx
    from anthropic import APIStatusError

    provider = AnthropicProvider(api_key="test-key")

    request = httpx.Request(
        "POST",
        "https://api.anthropic.com/v1/messages",
    )

    response = httpx.Response(
        500,
        request=request,
    )

    error = APIStatusError(
        message="internal server error",
        response=response,
        body=None,
    )

    provider.client = FakeClient(error=error)

    with pytest.raises(ProviderServerError):
        provider.generate("Hello")