# tests/providers/test_typesafe_provider.py
"""Offline tests for the TypeSafe.ai System One provider.

Covers:
* initialisation / credential + endpoint resolution (config vs env precedence),
* the ``Noul`` / ``Choice`` / ``Score`` builders and ``normalize_questions``,
* ``system_one()`` request building and typed answer parsing,
* HTTP error mapping (401 / 422 / 429 / 529 / 5xx / timeouts) and the
  in-provider retry loop (``Retry-After`` / ``retry-after-ms`` / backoff),
* ``list_models()``,
* model-card backed ``get_models_details()`` / ``get_max_context_length()``,
* the chat bridge (``chat_completion`` + ``extract_*`` helpers),
* token heuristics, ``close()``, and ``ProviderManager`` registration.

No network calls: the lazily-built ``httpx.AsyncClient`` is swapped for a
``MagicMock`` whose ``request`` is an ``AsyncMock`` returning real
``httpx.Response`` objects (the same seam as the Z.ai httpx backend tests).
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from llmcore.exceptions import ConfigError, ProviderError
from llmcore.models import Message, Role
from llmcore.providers import typesafe_provider as tp
from llmcore.providers.typesafe_provider import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResult,
    TypeSafeModelInfo,
    TypeSafeProvider,
    normalize_questions,
)

MINIMAL_CONFIG: dict[str, Any] = {
    "api_key": "ts-test-key-000",
    "default_model": "jev-latest",
    "timeout": 5,
    "max_retries": 2,
    # Zero backoff keeps retry tests instant when no Retry-After header is set.
    "retry_backoff_initial": 0.0,
    "retry_backoff_max": 0.0,
}

# Exact shape returned by the live API for the docs' support-ticket example.
SAMPLE_RESPONSE: dict[str, Any] = {
    "model": "jev-1.13.0",
    "answers": {
        "department": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.49,
            "probabilities": {"technical": 0.34, "billing": 0.66, "sales": 0.0},
        },
        "frustration": {
            "type": "score",
            "score": 1.0,
            "confidence": 1.0,
            "legend": {"0": "Calm", "1": "Frustrated but civil", "2": "Very angry"},
            "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0},
        },
        "is_urgent": {"type": "noul", "noul": 0.99},
    },
    "usage": {"input_tokens": 424, "output_tokens": 73},
}

SAMPLE_QUESTIONS: dict[str, Any] = {
    "department": Choice(
        instructions="Which team should handle this",
        criteria={"billing": "Payments", "technical": "Bugs", "sales": "Pricing"},
    ),
    "frustration": Score(
        instructions="How frustrated the customer appears",
        criteria=["Calm", "Frustrated but civil", "Very angry"],
    ),
    "is_urgent": Noul(instructions="The message conveys urgency"),
}

STATE = "Hi, my Stripe connection has been failing for 3 days. Please help ASAP."
REQ_ID = "req_01a0b7a1904f7ffbbf82c3d1696719c6"


def _resp(
    status: int = 200,
    json_body: Any = None,
    headers: dict[str, str] | None = None,
    text: str | None = None,
    method: str = "POST",
    path: str = "/v1/systemone",
) -> httpx.Response:
    """Build a real httpx.Response (no transport needed) for the mocked client."""
    request = httpx.Request(method, f"https://api.typesafe.ai{path}")
    if json_body is not None:
        return httpx.Response(status, json=json_body, headers=headers, request=request)
    return httpx.Response(status, text=text or "", headers=headers, request=request)


def _ok(headers: dict[str, str] | None = None) -> httpx.Response:
    hdrs = {"x-typesafe-request-id": REQ_ID}
    if headers:
        hdrs.update(headers)
    return _resp(200, SAMPLE_RESPONSE, headers=hdrs)


@pytest.fixture
def provider() -> TypeSafeProvider:
    p = TypeSafeProvider(MINIMAL_CONFIG.copy(), log_raw_payloads=False)
    http = MagicMock()
    http.request = AsyncMock(return_value=_ok())
    http.aclose = AsyncMock()
    p._http = http
    p._get_http = lambda: http  # type: ignore[method-assign]
    return p


@pytest.fixture
def sleep_mock():
    with patch("llmcore.providers.typesafe_provider.asyncio.sleep", new=AsyncMock()) as m:
        yield m


# =============================================================================
# Initialisation
# =============================================================================


class TestInitialization:
    def test_api_key_from_config(self):
        p = TypeSafeProvider(MINIMAL_CONFIG.copy())
        assert p._api_key == "ts-test-key-000"
        assert p.get_name() == "typesafe"
        assert p.default_model == "jev-latest"
        assert p.base_url == "https://api.typesafe.ai"
        assert p.supports_streaming is False
        assert p.supports_tools is False

    def test_api_key_from_default_env(self):
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "env-key"}, clear=False):
            p = TypeSafeProvider({"default_model": "jev-latest"})
        assert p._api_key == "env-key"

    def test_api_key_from_custom_env_var(self):
        with patch.dict(os.environ, {"MY_TS_KEY": "custom-key", "TYPESAFE_API_KEY": "default-key"}):
            p = TypeSafeProvider({"api_key_env_var": "MY_TS_KEY"})
        assert p._api_key == "custom-key"

    def test_custom_env_var_unset_falls_back_to_default_env(self):
        env = {k: v for k, v in os.environ.items() if k != "MY_TS_KEY"}
        env["TYPESAFE_API_KEY"] = "default-key"
        with patch.dict(os.environ, env, clear=True):
            p = TypeSafeProvider({"api_key_env_var": "MY_TS_KEY"})
        assert p._api_key == "default-key"

    def test_whitespace_env_key_is_ignored(self):
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "   "}, clear=True):
            with pytest.raises(ConfigError, match="TYPESAFE_API_KEY"):
                TypeSafeProvider({})

    def test_missing_api_key_raises_config_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigError, match="TYPESAFE_API_KEY"):
                TypeSafeProvider({"default_model": "jev-latest"})

    def test_config_beats_env_for_base_url_and_model(self):
        with patch.dict(
            os.environ,
            {"TYPESAFE_BASE_URL": "https://env.example/", "TYPESAFE_DEFAULT_MODEL": "jev-preview"},
        ):
            p = TypeSafeProvider(
                {
                    "api_key": "k",
                    "base_url": "https://cfg.example/v/",
                    "default_model": "jev-1.13.0",
                }
            )
        assert p.base_url == "https://cfg.example/v"  # trailing slash stripped
        assert p.default_model == "jev-1.13.0"

    def test_env_beats_defaults_for_base_url_and_model(self):
        with patch.dict(
            os.environ,
            {"TYPESAFE_BASE_URL": "https://env.example/", "TYPESAFE_DEFAULT_MODEL": "jev-preview"},
        ):
            p = TypeSafeProvider({"api_key": "k"})
        assert p.base_url == "https://env.example"
        assert p.default_model == "jev-preview"

    @pytest.mark.parametrize(
        "key,value",
        [
            ("timeout", 0),
            ("timeout", -1),
            ("timeout", "nope"),
            ("max_retries", -1),
            ("max_retries", "x"),
            ("retry_backoff_initial", -0.5),
            ("retry_backoff_max", float("inf")),
        ],
    )
    def test_invalid_numeric_config_raises(self, key, value):
        cfg = MINIMAL_CONFIG.copy()
        cfg[key] = value
        with pytest.raises(ConfigError, match=key):
            TypeSafeProvider(cfg)

    def test_instance_name_from_manager(self):
        cfg = MINIMAL_CONFIG.copy()
        cfg["_instance_name"] = "jev"
        assert TypeSafeProvider(cfg).get_name() == "jev"

    def test_httpx_missing_raises_import_error(self):
        with patch.object(tp, "httpx_available", False):
            with pytest.raises(ImportError, match=r"llmcore\[typesafe\]"):
                TypeSafeProvider(MINIMAL_CONFIG.copy())

    def test_http_client_is_lazy_and_carries_auth(self):
        p = TypeSafeProvider({**MINIMAL_CONFIG, "headers": {"X-Team": "ops"}})
        assert p._http is None
        client = p._get_http()
        assert isinstance(client, httpx.AsyncClient)
        assert client.headers["authorization"] == "Bearer ts-test-key-000"
        assert client.headers["accept"] == "application/json"
        assert client.headers["user-agent"].startswith("llmcore/")
        assert client.headers["x-team"] == "ops"
        assert str(client.base_url).rstrip("/") == "https://api.typesafe.ai"
        assert p._get_http() is client


# =============================================================================
# Question builders
# =============================================================================


class TestQuestionBuilders:
    def test_noul_wire_omits_unset_criteria(self):
        assert Noul(instructions="Urgent?").to_wire() == {"type": "noul", "instructions": "Urgent?"}

    def test_noul_wire_with_criteria(self):
        q = Noul(instructions="Urgent?", criteria={"true": "Time-sensitive", "false": "Can wait"})
        assert q.to_wire() == {
            "type": "noul",
            "instructions": "Urgent?",
            "criteria": {"true": "Time-sensitive", "false": "Can wait"},
        }

    def test_noul_rejects_unknown_criteria_keys(self):
        with pytest.raises(ValueError, match="'true' and 'false'"):
            Noul(instructions="?", criteria={"yes": "x"})

    def test_choice_preserves_none_descriptions(self):
        q = Choice(instructions="Tone?", criteria={"calm": None, "angry": "Hostile"})
        assert q.to_wire() == {
            "type": "choice",
            "instructions": "Tone?",
            "criteria": {"calm": None, "angry": "Hostile"},
        }

    def test_choice_requires_options(self):
        with pytest.raises(ValueError, match="at least one option"):
            Choice(instructions="?", criteria={})

    def test_score_wire_and_structured_levels(self):
        q = Score(instructions="Severity?", criteria=["low", {"level": "high", "examples": ["x"]}])
        assert q.to_wire()["criteria"] == ["low", {"level": "high", "examples": ["x"]}]
        assert q.to_wire()["type"] == "score"

    def test_score_requires_levels(self):
        with pytest.raises(ValueError, match="at least one level"):
            Score(instructions="?", criteria=[])

    def test_unknown_fields_rejected(self):
        with pytest.raises(ValueError):
            Noul(instructions="?", extra_field=1)  # type: ignore[call-arg]

    def test_instructions_optional_and_structured(self):
        assert Choice(criteria={"a": None}).to_wire() == {"type": "choice", "criteria": {"a": None}}
        q = Noul(instructions={"task": "Detect spam", "examples": ["buy now"]})
        assert q.to_wire()["instructions"] == {"task": "Detect spam", "examples": ["buy now"]}


class TestNormalizeQuestions:
    def test_builders_and_dicts_mixed(self):
        wire = normalize_questions(
            {
                "a": Noul(instructions="?"),
                "b": {"type": "choice", "instructions": "?", "criteria": {"x": None}, "future": 1},
                "c": {"type": "score", "criteria": ["lo", "hi"]},
            }
        )
        assert wire["a"] == {"type": "noul", "instructions": "?"}
        assert wire["b"]["future"] == 1  # unknown keys pass through for forward-compat
        assert wire["c"] == {"type": "score", "criteria": ["lo", "hi"]}

    def test_empty_map_rejected(self):
        with pytest.raises(ValueError, match="at least one question"):
            normalize_questions({})

    def test_unknown_type_rejected(self):
        with pytest.raises(ValueError, match="unsupported type 'rank'"):
            normalize_questions({"q": {"type": "rank", "criteria": ["a"]}})

    def test_choice_dict_requires_criteria(self):
        with pytest.raises(ValueError, match="requires non-empty 'criteria'"):
            normalize_questions({"q": {"type": "choice", "instructions": "?"}})

    def test_choice_dict_criteria_must_be_mapping(self):
        with pytest.raises(ValueError, match="mapping of option"):
            normalize_questions({"q": {"type": "choice", "criteria": ["a", "b"]}})

    def test_score_dict_criteria_must_be_list(self):
        with pytest.raises(ValueError, match="ordered list"):
            normalize_questions({"q": {"type": "score", "criteria": {"0": "lo"}}})

    def test_non_mapping_question_rejected(self):
        with pytest.raises(ValueError, match="must be a Noul/Choice/Score"):
            normalize_questions({"q": "is it urgent?"})  # type: ignore[dict-item]

    def test_bad_question_id_rejected(self):
        with pytest.raises(ValueError, match="non-empty strings"):
            normalize_questions({"": Noul(instructions="?")})


# =============================================================================
# system_one
# =============================================================================


class TestSystemOne:
    async def test_happy_path_types_and_accessors(self, provider):
        result = await provider.system_one(STATE, SAMPLE_QUESTIONS)
        assert isinstance(result, SystemOneResult)
        assert result.model == "jev-1.13.0"
        assert result.request_id == REQ_ID
        assert result.usage.input_tokens == 424
        assert result.usage.output_tokens == 73
        assert result.usage.total_tokens == 497
        assert set(result.answers) == {"department", "frustration", "is_urgent"}

        dept = result.choices["department"]
        assert isinstance(dept, ChoiceAnswer)
        assert dept.choice == "billing"
        assert dept.probabilities["billing"] == pytest.approx(0.66)
        assert dept.confidence == pytest.approx(0.49)

        frus = result.scores["frustration"]
        assert isinstance(frus, ScoreAnswer)
        assert frus.score == 1.0
        assert frus.legend == {0: "Calm", 1: "Frustrated but civil", 2: "Very angry"}
        assert frus.probabilities == {0: 0.0, 1: 1.0, 2: 0.0}

        urg = result.nouls["is_urgent"]
        assert isinstance(urg, NoulAnswer)
        assert urg.noul == pytest.approx(0.99)
        assert result.raw == SAMPLE_RESPONSE

    async def test_request_body_and_defaults(self, provider):
        await provider.system_one(STATE, SAMPLE_QUESTIONS)
        method, path = provider._http.request.call_args.args
        kwargs = provider._http.request.call_args.kwargs
        assert (method, path) == ("POST", "/v1/systemone")
        body = kwargs["json"]
        assert body["state"] == STATE
        assert body["model"] == "jev-latest"
        assert body["questions"]["is_urgent"] == {
            "type": "noul",
            "instructions": "The message conveys urgency",
        }
        assert body["questions"]["frustration"]["criteria"] == [
            "Calm",
            "Frustrated but civil",
            "Very angry",
        ]
        assert "timeout" not in kwargs and "headers" not in kwargs

    async def test_model_timeout_headers_and_extra_body(self, provider):
        await provider.system_one(
            {"ticket": STATE},
            {"q": Noul(instructions="?")},
            model="jev-1.13.0",
            timeout=2.5,
            extra_headers={"X-Trace": "abc"},
            extra_body={"metadata": {"tenant": "t1"}},
        )
        kwargs = provider._http.request.call_args.kwargs
        assert kwargs["json"]["model"] == "jev-1.13.0"
        assert kwargs["json"]["state"] == {"ticket": STATE}
        assert kwargs["json"]["metadata"] == {"tenant": "t1"}
        assert kwargs["timeout"] == 2.5
        assert kwargs["headers"] == {"X-Trace": "abc"}

    async def test_answers_dict_uses_wire_shape(self, provider):
        result = await provider.system_one(STATE, SAMPLE_QUESTIONS)
        as_dict = result.answers_dict()
        assert as_dict["frustration"]["legend"] == {
            "0": "Calm",
            "1": "Frustrated but civil",
            "2": "Very angry",
        }
        assert json.loads(result.answers_json())["is_urgent"] == {"type": "noul", "noul": 0.99}

    async def test_unknown_answer_type_is_dropped_but_kept_in_raw(self, provider, caplog):
        body = json.loads(json.dumps(SAMPLE_RESPONSE))
        body["answers"]["ranking"] = {"type": "rank", "order": ["a", "b"]}
        provider._http.request.return_value = _resp(200, body)
        with caplog.at_level("WARNING"):
            result = await provider.system_one(STATE, SAMPLE_QUESTIONS)
        assert "ranking" not in result.answers
        assert result.raw["answers"]["ranking"]["type"] == "rank"
        assert "unrecognised type 'rank'" in caplog.text

    async def test_missing_usage_and_model_fallbacks(self, provider):
        provider._http.request.return_value = _resp(
            200, {"answers": {"q": {"type": "noul", "noul": 0.5}}}
        )
        result = await provider.system_one(STATE, {"q": Noul(instructions="?")})
        assert result.model == "jev-latest"  # falls back to the requested model
        assert result.usage.input_tokens is None
        assert result.usage.total_tokens == 0
        assert result.request_id is None

    async def test_malformed_answer_raises_provider_error(self, provider):
        provider._http.request.return_value = _resp(
            200, {"model": "x", "answers": {"q": {"type": "choice", "choice": "a"}}}
        )
        with pytest.raises(ProviderError) as exc:
            await provider.system_one(STATE, {"q": Choice(criteria={"a": None})})
        assert exc.value.retryable is False
        assert "answers.q" in str(exc.value)

    async def test_non_json_body_raises_provider_error(self, provider):
        provider._http.request.return_value = _resp(200, text="<html>oops</html>")
        with pytest.raises(ProviderError, match="non-JSON"):
            await provider.system_one(STATE, {"q": Noul(instructions="?")})

    async def test_none_state_rejected_before_request(self, provider):
        with pytest.raises(ValueError, match="state is required"):
            await provider.system_one(None, {"q": Noul(instructions="?")})  # type: ignore[arg-type]
        provider._http.request.assert_not_called()

    async def test_invalid_questions_rejected_before_request(self, provider):
        with pytest.raises(ValueError):
            await provider.system_one(STATE, {})
        provider._http.request.assert_not_called()

    async def test_raw_payload_logging_does_not_break(self, provider, caplog):
        provider.log_raw_payloads_enabled = True
        with caplog.at_level("DEBUG", logger="llmcore.providers.typesafe_provider"):
            await provider.system_one(STATE, SAMPLE_QUESTIONS)
        assert "TypeSafe request body" in caplog.text


# =============================================================================
# Error mapping & retries
# =============================================================================


class TestErrorMapping:
    async def test_401_is_non_retryable_with_key_hint(self, provider):
        provider._http.request.return_value = _resp(
            401,
            {"detail": {"error_type": "authentication_error", "message": "Cannot authenticate."}},
            headers={"x-typesafe-request-id": "req_401"},
        )
        with pytest.raises(ProviderError) as exc:
            await provider.system_one(STATE, SAMPLE_QUESTIONS)
        err = exc.value
        assert err.status_code == 401
        assert err.retryable is False
        assert "TYPESAFE_API_KEY" in str(err)
        assert "Cannot authenticate." in str(err)
        assert "request_id=req_401" in str(err)
        assert provider._http.request.call_count == 1

    async def test_422_includes_validation_detail(self, provider):
        provider._http.request.return_value = _resp(
            422,
            {
                "detail": [
                    {
                        "type": "missing",
                        "loc": ["body", "questions", "q", "choice", "criteria"],
                        "msg": "Field required",
                    }
                ]
            },
        )
        with pytest.raises(ProviderError) as exc:
            await provider.system_one(STATE, {"q": {"type": "choice", "criteria": {"a": None}}})
        assert exc.value.status_code == 422
        assert exc.value.retryable is False
        assert "questions.q.choice.criteria: Field required" in str(exc.value)

    async def test_429_retried_with_retry_after_then_succeeds(self, provider, sleep_mock):
        provider._http.request.side_effect = [
            _resp(429, {"error": "slow down"}, headers={"Retry-After": "1"}),
            _resp(429, {"error": "slow down"}, headers={"Retry-After": "2"}),
            _ok(),
        ]
        result = await provider.system_one(STATE, SAMPLE_QUESTIONS)
        assert result.model == "jev-1.13.0"
        assert provider._http.request.call_count == 3
        assert [c.args[0] for c in sleep_mock.call_args_list] == [1.0, 2.0]
        # Retry attempts carry the retry-count header.
        second = provider._http.request.call_args_list[1].kwargs
        assert second["headers"]["X-TypeSafe-Retry-Count"] == "1"
        third = provider._http.request.call_args_list[2].kwargs
        assert third["headers"]["X-TypeSafe-Retry-Count"] == "2"

    async def test_429_exhausts_retries(self, provider, sleep_mock):
        provider._http.request.return_value = _resp(
            429, {"error": "slow down"}, headers={"Retry-After": "3", "x-typesafe-request-id": "r"}
        )
        with pytest.raises(ProviderError) as exc:
            await provider.system_one(STATE, SAMPLE_QUESTIONS)
        assert exc.value.status_code == 429
        assert exc.value.retryable is True
        assert exc.value.retry_after_seconds == 3.0
        assert provider._http.request.call_count == 3  # 1 + max_retries
        assert sleep_mock.call_count == 2

    async def test_retry_after_ms_wins(self, provider, sleep_mock):
        provider._http.request.side_effect = [
            _resp(429, {"error": "x"}, headers={"retry-after-ms": "250", "Retry-After": "9"}),
            _ok(),
        ]
        await provider.system_one(STATE, SAMPLE_QUESTIONS)
        assert sleep_mock.call_args_list[0].args[0] == pytest.approx(0.25)

    async def test_529_and_5xx_are_retried(self, provider, sleep_mock):
        provider._http.request.side_effect = [
            _resp(529, text="overloaded"),
            _resp(503, text=""),
            _ok(),
        ]
        result = await provider.system_one(STATE, SAMPLE_QUESTIONS)
        assert result.model == "jev-1.13.0"
        assert provider._http.request.call_count == 3

    async def test_529_error_message_and_retryable_flag(self, provider, sleep_mock):
        provider._http.request.return_value = _resp(529, text="overloaded")
        with pytest.raises(ProviderError) as exc:
            await provider.system_one(STATE, SAMPLE_QUESTIONS)
        assert exc.value.status_code == 529
        assert exc.value.retryable is True
        assert "overloaded" in str(exc.value).lower()

    async def test_400_is_not_retried(self, provider, sleep_mock):
        provider._http.request.return_value = _resp(400, {"message": "bad"})
        with pytest.raises(ProviderError) as exc:
            await provider.system_one(STATE, SAMPLE_QUESTIONS)
        assert exc.value.status_code == 400
        assert exc.value.retryable is False
        assert provider._http.request.call_count == 1
        sleep_mock.assert_not_called()

    async def test_timeout_then_success(self, provider, sleep_mock):
        provider._http.request.side_effect = [httpx.ReadTimeout("slow"), _ok()]
        result = await provider.system_one(STATE, SAMPLE_QUESTIONS)
        assert result.model == "jev-1.13.0"
        assert provider._http.request.call_count == 2

    async def test_timeout_exhausted_is_retryable_provider_error(self, provider, sleep_mock):
        provider._http.request.side_effect = httpx.ReadTimeout("slow")
        with pytest.raises(ProviderError) as exc:
            await provider.system_one(STATE, SAMPLE_QUESTIONS)
        assert exc.value.retryable is True
        assert "timed out" in str(exc.value)
        assert provider._http.request.call_count == 3

    async def test_connection_error_is_retryable(self, provider, sleep_mock):
        provider._http.request.side_effect = httpx.ConnectError("refused")
        with pytest.raises(ProviderError) as exc:
            await provider.system_one(STATE, SAMPLE_QUESTIONS)
        assert exc.value.retryable is True
        assert "connection error" in str(exc.value)

    async def test_max_retries_zero_disables_retries(self, sleep_mock):
        p = TypeSafeProvider({**MINIMAL_CONFIG, "max_retries": 0})
        http = MagicMock()
        http.request = AsyncMock(return_value=_resp(503, text="down"))
        p._get_http = lambda: http  # type: ignore[method-assign]
        with pytest.raises(ProviderError) as exc:
            await p.system_one(STATE, SAMPLE_QUESTIONS)
        assert exc.value.status_code == 503
        assert http.request.call_count == 1
        sleep_mock.assert_not_called()

    def test_error_body_fallbacks(self, provider):
        err = provider._map_error(
            _resp(500, text="Internal boom"), model="m", method="POST", path="/p"
        )
        assert err.status_code == 500 and err.retryable is True
        assert "Internal boom" in str(err)
        err = provider._map_error(_resp(418, text=""), model="m", method="POST", path="/p")
        assert err.status_code == 418
        err = provider._map_error(
            _resp(404, {"error": {"message": "no such route"}}), model="m", method="GET", path="/x"
        )
        assert "no such route" in str(err) and err.retryable is False


class TestRetryDelayHelpers:
    def test_parse_retry_after_ms_wins(self):
        assert (
            tp._parse_retry_after(httpx.Headers({"retry-after-ms": "1500", "Retry-After": "9"}))
            == 1.5
        )

    def test_parse_retry_after_seconds(self):
        assert tp._parse_retry_after(httpx.Headers({"Retry-After": "7"})) == 7.0

    def test_parse_retry_after_http_date(self):
        from email.utils import formatdate

        future = formatdate(timeval=__import__("time").time() + 30, usegmt=True)
        delay = tp._parse_retry_after(httpx.Headers({"Retry-After": future}))
        assert delay is not None and 25 <= delay <= 31

    def test_parse_retry_after_invalid_or_negative(self):
        assert tp._parse_retry_after(httpx.Headers({"Retry-After": "soon"})) is None
        assert tp._parse_retry_after(httpx.Headers({"Retry-After": "-5"})) is None
        assert tp._parse_retry_after(httpx.Headers({"retry-after-ms": "nan"})) is None
        assert tp._parse_retry_after(httpx.Headers({})) is None
        assert tp._parse_retry_after(None) is None

    def test_backoff_bounds(self):
        p = TypeSafeProvider(
            {**MINIMAL_CONFIG, "retry_backoff_initial": 0.5, "retry_backoff_max": 5.0}
        )
        for _ in range(20):
            assert 0.375 <= p._backoff_delay(0) <= 0.5
            assert 3.75 <= p._backoff_delay(10) <= 5.0
        zero = TypeSafeProvider(MINIMAL_CONFIG.copy())
        assert zero._backoff_delay(3) == 0.0

    def test_retry_delay_prefers_server_and_caps(self):
        p = TypeSafeProvider(
            {**MINIMAL_CONFIG, "retry_backoff_initial": 0.5, "retry_backoff_max": 5.0}
        )
        assert p._retry_delay(httpx.Headers({"Retry-After": "2"}), 0) == 2.0
        assert p._retry_delay(httpx.Headers({"Retry-After": "600"}), 0) == 60.0
        assert 0 < p._retry_delay(httpx.Headers({}), 0) <= 0.5


# =============================================================================
# list_models
# =============================================================================


class TestListModels:
    async def test_list_models(self, provider):
        provider._http.request.return_value = _resp(
            200,
            {
                "models": [
                    {
                        "name": "jev-latest",
                        "description": "Latest",
                        "release_date": "2026-09-10T18:38:01+00:00",
                    },
                    {
                        "name": "jev-preview",
                        "description": "Preview",
                        "release_date": "2026-09-10T18:39:06+00:00",
                    },
                    {"description": "malformed, no name"},
                ]
            },
            method="GET",
            path="/v1/models",
        )
        models = await provider.list_models()
        assert [m.name for m in models] == ["jev-latest", "jev-preview"]
        assert all(isinstance(m, TypeSafeModelInfo) for m in models)
        assert models[0].description == "Latest"
        method, path = provider._http.request.call_args.args
        assert (method, path) == ("GET", "/v1/models")

    async def test_list_models_non_json(self, provider):
        provider._http.request.return_value = _resp(
            200, text="nope", method="GET", path="/v1/models"
        )
        with pytest.raises(ProviderError, match="non-JSON"):
            await provider.list_models()


# =============================================================================
# Model cards
# =============================================================================


class TestModelCards:
    async def test_get_models_details_from_registry_includes_aliases(self, provider):
        details = await provider.get_models_details()
        ids = {d.id for d in details}
        assert {"jev-1.13.0", "jev-latest", "jev-preview"} <= ids
        by_id = {d.id: d for d in details}
        card = by_id["jev-1.13.0"]
        assert card.model_type == "decision"
        assert card.context_length == 65536
        assert card.supports_streaming is False and card.supports_tools is False
        assert card.provider_name == "typesafe"
        assert card.metadata["question_types"] == ["choice", "noul", "score"]
        assert by_id["jev-latest"].metadata["alias_of"] == "jev-1.13.0"

    async def test_get_models_details_static_fallback(self, provider):
        with patch(
            "llmcore.model_cards.registry.get_model_card_registry", side_effect=RuntimeError("boom")
        ):
            details = await provider.get_models_details()
        assert [d.id for d in details] == ["jev-1.13.0", "jev-latest", "jev-preview"]
        assert all(d.model_type == "decision" for d in details)

    def test_get_max_context_length_resolves_aliases(self, provider):
        assert provider.get_max_context_length("jev-latest") == 65536
        assert provider.get_max_context_length("jev-1.13.0") == 65536
        assert provider.get_max_context_length() == 65536

    def test_get_max_context_length_fallback(self):
        p = TypeSafeProvider({**MINIMAL_CONFIG, "fallback_context_length": 1234})
        assert p.get_max_context_length("jev-99") == 1234

    def test_supported_parameters(self, provider):
        params = provider.get_supported_parameters()
        assert set(params) == {"questions", "state", "timeout", "extra_body", "extra_headers"}
        assert params["questions"]["required"] is True


# =============================================================================
# Chat bridge
# =============================================================================


def _ctx() -> list[Message]:
    return [
        Message(role=Role.SYSTEM, content="You are a support triage assistant."),
        Message(role=Role.USER, content=STATE),
        Message(role=Role.ASSISTANT, content="   "),
        Message(role=Role.USER, content="Any update?"),
    ]


class TestChatBridge:
    async def test_requires_questions(self, provider):
        with pytest.raises(ProviderError) as exc:
            await provider.chat_completion(_ctx())
        assert exc.value.status_code == 400 and exc.value.retryable is False
        assert "system_one" in str(exc.value)
        provider._http.request.assert_not_called()

    async def test_rejects_streaming(self, provider):
        with pytest.raises(ProviderError, match="does not stream"):
            await provider.chat_completion(_ctx(), stream=True, questions=SAMPLE_QUESTIONS)

    async def test_rejects_tools(self, provider):
        from llmcore.models import Tool

        tool = Tool(name="t", description="d", parameters={"type": "object", "properties": {}})
        with pytest.raises(ProviderError, match="tool calling"):
            await provider.chat_completion(_ctx(), tools=[tool], questions=SAMPLE_QUESTIONS)
        with pytest.raises(ProviderError, match="tool calling"):
            await provider.chat_completion(_ctx(), tool_choice="auto", questions=SAMPLE_QUESTIONS)

    async def test_rejects_unknown_kwargs(self, provider):
        with pytest.raises(ValueError, match="temperature"):
            await provider.chat_completion(_ctx(), questions=SAMPLE_QUESTIONS, temperature=0.2)

    async def test_conversation_becomes_state(self, provider):
        response = await provider.chat_completion(_ctx(), questions=SAMPLE_QUESTIONS)
        body = provider._http.request.call_args.kwargs["json"]
        assert body["state"] == [
            {"role": "system", "content": "You are a support triage assistant."},
            {"role": "user", "content": STATE},
            {"role": "user", "content": "Any update?"},
        ]
        assert body["model"] == "jev-latest"
        assert response["object"] == "systemone.result"
        assert response["model"] == "jev-1.13.0"
        assert response["id"] == f"typesafe-{REQ_ID}"
        assert response["choices"][0]["finish_reason"] == "stop"
        assert response["usage"] == {
            "prompt_tokens": 424,
            "completion_tokens": 73,
            "total_tokens": 497,
        }
        assert response["typesafe"] == SAMPLE_RESPONSE

    async def test_explicit_state_and_model_override(self, provider):
        await provider.chat_completion(
            _ctx(), model="jev-preview", questions={"q": Noul(instructions="?")}, state={"doc": "x"}
        )
        body = provider._http.request.call_args.kwargs["json"]
        assert body["state"] == {"doc": "x"}
        assert body["model"] == "jev-preview"

    async def test_extract_helpers(self, provider):
        response = await provider.chat_completion(_ctx(), questions=SAMPLE_QUESTIONS)
        content = provider.extract_response_content(response)
        parsed = json.loads(content)
        assert parsed["department"]["choice"] == "billing"
        assert parsed["frustration"]["legend"]["2"] == "Very angry"
        usage = provider.extract_usage_details(response)
        assert usage["prompt_tokens"] == 424 and usage["input_tokens"] == 424
        assert usage["completion_tokens"] == 73 and usage["total_tokens"] == 497
        assert provider.extract_finish_reason(response) == "stop"
        assert provider.extract_tool_calls(response) == []
        assert provider.extract_delta_content({"anything": 1}) == ""

    def test_extract_helpers_defensive(self, provider):
        assert provider.extract_response_content({}) == ""
        assert provider.extract_response_content({"choices": []}) == ""
        assert provider.extract_usage_details({}) == {}
        assert provider.extract_usage_details({"usage": {"input_tokens": 5}}) == {
            "prompt_tokens": 5,
            "completion_tokens": None,
            "total_tokens": 5,
            "input_tokens": 5,
            "output_tokens": None,
        }
        assert provider.extract_finish_reason({}) == "stop"

    async def test_tool_messages_are_flattened(self, provider):
        ctx = [
            Message(role=Role.USER, content="run it"),
            Message(role=Role.TOOL, content='{"result": 42}', tool_call_id="call_1"),
        ]
        await provider.chat_completion(ctx, questions={"q": Noul(instructions="?")})
        state = provider._http.request.call_args.kwargs["json"]["state"]
        assert state[0] == {"role": "user", "content": "run it"}
        assert len(state) == 2
        assert state[1]["role"] != "tool"
        assert "42" in state[1]["content"]


# =============================================================================
# Tokens, lifecycle, registration
# =============================================================================


class TestTokens:
    async def test_count_tokens(self, provider):
        assert await provider.count_tokens("") == 0
        n = await provider.count_tokens("Hello, TypeSafe world!")
        assert 3 <= n <= 8

    async def test_count_tokens_heuristic_without_tiktoken(self, provider):
        provider._encoding = None
        assert await provider.count_tokens("x" * 400) == 100
        assert await provider.count_tokens("ab") == 1

    async def test_count_message_tokens_adds_overhead(self, provider):
        provider._encoding = None
        msgs = [Message(role=Role.USER, content="x" * 40), Message(role=Role.ASSISTANT, content="")]
        assert await provider.count_message_tokens(msgs) == 10 + 4 + 0 + 4


class TestLifecycle:
    async def test_warm_up_is_side_effect_free(self, provider):
        await provider.warm_up()
        provider._http.request.assert_not_called()

    async def test_close_releases_client(self, provider):
        http = provider._http
        await provider.close()
        http.aclose.assert_awaited_once()
        assert provider._http is None
        await provider.close()  # idempotent

    async def test_close_swallows_errors(self, provider):
        provider._http.aclose = AsyncMock(side_effect=RuntimeError("boom"))
        await provider.close()
        assert provider._http is None


class TestRegistration:
    def test_in_provider_map(self):
        from llmcore.providers.manager import PROVIDER_MAP

        assert PROVIDER_MAP["typesafe"] is TypeSafeProvider
        assert PROVIDER_MAP["jev"] is TypeSafeProvider

    def test_instance_alias(self):
        from llmcore.providers.manager import _PROVIDER_INSTANCE_ALIASES

        assert _PROVIDER_INSTANCE_ALIASES["jev"] == "typesafe"

    def test_manager_loads_from_config(self):
        from llmcore.providers.manager import ProviderManager

        cfg = {
            "llmcore.default_provider": "typesafe",
            "providers": {"typesafe": {"api_key": "k", "default_model": "jev-latest"}},
        }
        mgr = ProviderManager(cfg)
        assert isinstance(mgr.get_provider("typesafe"), TypeSafeProvider)
        assert mgr.get_provider("jev") is mgr.get_provider("typesafe")
        assert mgr.get_provider("typesafe").get_name() == "typesafe"

    def test_manager_skips_provider_without_key(self, caplog):
        from llmcore.providers.manager import ProviderManager

        cfg = {
            "llmcore.default_provider": "typesafe",
            "providers": {"typesafe": {"default_model": "jev-latest"}},
        }
        with patch.dict(os.environ, {}, clear=True), caplog.at_level("WARNING"):
            with pytest.raises(ConfigError, match="Default provider 'typesafe'"):
                ProviderManager(cfg)
        assert "TYPESAFE_API_KEY" in caplog.text
