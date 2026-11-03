from types import SimpleNamespace

import groq
import httpx
import pytest

from docqa.config import Settings
from docqa.errors import DocQAError
from docqa.llm import GroqLLM, build_llm

_REQUEST = httpx.Request("POST", "https://example.invalid")
SECRET = "sk-secret"


def _rate_limit():
    return groq.RateLimitError("rate", response=httpx.Response(429, request=_REQUEST), body=None)


def _unauthorized():
    return groq.AuthenticationError(
        "bad key", response=httpx.Response(401, request=_REQUEST), body=None
    )


class FakeClient:
    """Mimics client.chat.completions.create(...): raises or returns the next item."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        message = SimpleNamespace(content=outcome)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _llm(outcomes, sleeps, **kwargs):
    client = FakeClient(outcomes)
    llm = GroqLLM(SECRET, "some-model", client=client, sleep=sleeps.append, **kwargs)
    return llm, client


def test_retries_rate_limit_then_succeeds():
    sleeps = []
    llm, client = _llm([_rate_limit(), _rate_limit(), "hello"], sleeps)
    assert llm.complete("sys", "user") == "hello"
    assert sleeps == [5, 10]
    assert len(client.requests) == 3


def test_gives_up_after_max_attempts():
    sleeps = []
    llm, client = _llm([_rate_limit(), _rate_limit(), _rate_limit()], sleeps)
    with pytest.raises(DocQAError) as exc:
        llm.complete("sys", "user")
    assert "busy" in str(exc.value)
    assert len(client.requests) == 3
    assert sleeps == [5, 10]


def test_connection_error_is_retried_too():
    sleeps = []
    llm, _ = _llm([groq.APIConnectionError(request=_REQUEST), "ok"], sleeps)
    assert llm.complete("sys", "user") == "ok"
    assert sleeps == [5]


def test_bad_key_fails_fast_and_never_leaks_key():
    sleeps = []
    llm, client = _llm([_unauthorized()], sleeps)
    with pytest.raises(DocQAError) as exc:
        llm.complete("sys", "user")
    assert "GROQ_API_KEY" in str(exc.value)
    assert SECRET not in str(exc.value)
    assert len(client.requests) == 1
    assert sleeps == []


def test_request_uses_model_low_temperature_and_both_messages():
    llm, client = _llm(["x"], [])
    llm.complete("be brief", "hi")
    request = client.requests[0]
    assert request["model"] == "some-model"
    assert request["temperature"] == 0.1
    assert request["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]


def test_json_mode_asks_for_json_only():
    llm, client = _llm(["{}"], [])
    llm.complete("be brief", "hi", json_mode=True)
    assert "Reply with JSON only." in client.requests[0]["messages"][0]["content"]


def test_build_llm_uses_settings():
    llm = build_llm(Settings(SECRET, "m"))
    assert isinstance(llm, GroqLLM)
    assert llm.model == "m"


def _server_error():
    return groq.InternalServerError(
        "over capacity", response=httpx.Response(503, request=_REQUEST), body=None
    )


def _too_large():
    return groq.APIStatusError(
        "too big", response=httpx.Response(413, request=_REQUEST), body=None
    )


def test_server_error_is_retried():
    sleeps = []
    llm, _ = _llm([_server_error(), "ok"], sleeps)
    assert llm.complete("sys", "user") == "ok"
    assert sleeps == [5]


def test_server_error_every_time_says_busy_not_bad_key():
    llm, _ = _llm([_server_error()] * 3, [])
    with pytest.raises(DocQAError) as exc:
        llm.complete("sys", "user")
    assert "busy" in str(exc.value)
    assert "GROQ_API_KEY" not in str(exc.value)


def test_too_large_request_has_its_own_message():
    llm, client = _llm([_too_large()], [])
    with pytest.raises(DocQAError) as exc:
        llm.complete("sys", "user")
    assert "too large" in str(exc.value)
    assert "GROQ_API_KEY" not in str(exc.value)
    assert len(client.requests) == 1


def test_reasoning_block_is_removed_from_the_reply():
    llm, _ = _llm(['<think>maybe {"x": 1}</think>\n{"supported": true}'], [])
    assert llm.complete("sys", "user") == '{"supported": true}'


def test_sdk_own_retries_are_switched_off(monkeypatch):
    captured = {}
    monkeypatch.setattr(groq, "Groq", lambda **kwargs: captured.update(kwargs) or object())
    GroqLLM(SECRET, "m")
    assert captured["max_retries"] == 0
