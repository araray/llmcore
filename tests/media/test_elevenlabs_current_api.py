# tests/media/test_elevenlabs_current_api.py
"""ElevenLabs against the live API and elevenlabs 2.78 (2026-11-16).

- scribe_v3 list fields must be JSON strings: repeated form fields are a 500
  and Python's str(list) is not JSON (both measured);
- the SDK backend dropped voice_settings / seed / loop / ... on TTS, SFX and
  music: only text, model and format were sent;
- SFX takes wav_* now, which was labelled audio/mpeg.
"""

from __future__ import annotations

import functools
import json

import pytest

pytest.importorskip("elevenlabs")

from llmcore.providers.elevenlabs_provider import (  # noqa: E402
    ElevenLabsProvider,
    _form_value,
    _sdk_call,
)


def test_form_values():
    assert _form_value(["pt", "es"]) == json.dumps(["pt", "es"])
    assert _form_value(True) == "true" and _form_value({"a": 1}) == '{"a": 1}' and _form_value(3) == "3"


def test_sdk_calls_forward_every_known_field(caplog):
    from elevenlabs.client import AsyncElevenLabs

    real = AsyncElevenLabs(api_key="x").text_to_speech.convert
    seen = {}

    @functools.wraps(real)
    def convert(**kwargs):
        seen.update(kwargs)
        return iter(())

    _sdk_call(convert, voice_id="v", output_format="mp3_44100_128", text="hi", model_id="eleven_v4",
              voice_settings={"stability": 0.4}, seed=7, not_a_field=1)
    assert seen["voice_settings"] == {"stability": 0.4} and seen["seed"] == 7
    assert "not_a_field" not in seen and "not_a_field" in caplog.text


def test_current_defaults_and_wav():
    p = ElevenLabsProvider({"api_key": "x"})
    assert p._model_for("asr", None) == "scribe_v3"
    assert p._model_for("sfx", None) == "eleven_text_to_sound_v3"
    assert p._model_for("tts_stream", None) == "eleven_v4_turbo"
    assert ElevenLabsProvider._mime_for("wav_44100") == "audio/wav"
