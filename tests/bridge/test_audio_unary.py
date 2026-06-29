"""End-to-end tests for the one-shot (unary) AudioService RPCs.

Each RPC is exercised over **both** transports — gRPC (the ``grpc_channel``
fixture: real ``grpc.aio`` server on a Unix socket) and HTTP/JSON (the
``http_client`` fixture: Starlette app via ``httpx.ASGITransport``) — against
the deterministic ``FakeAudioProvider``. Audio is enabled per-test via
``LLMCORE_BRIDGE_FAKE_AUDIO=1`` (read dynamically by ``BridgeCore.audio_enabled``);
the disabled path must yield gRPC ``UNIMPLEMENTED`` / HTTP ``501``.

Struct/Value-bearing results (OCR, text analysis) are checked through
``MessageToDict`` so the proto-Struct mapping is verified end to end.
"""

from __future__ import annotations

import base64

import grpc
import pytest
from google.protobuf.json_format import MessageToDict

from llmcore.bridge._generated.llmcore.v1 import audio_pb2, audio_pb2_grpc

_P = "/llmcore.v1/AudioService"


def _d(msg) -> dict:
    return MessageToDict(msg, preserving_proto_field_name=True)


# --------------------------------------------------------------------------- #
# gRPC
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_grpc_synthesize(grpc_channel, monkeypatch):
    monkeypatch.setenv("LLMCORE_BRIDGE_FAKE_AUDIO", "1")
    stub = audio_pb2_grpc.AudioServiceStub(grpc_channel)
    resp = await stub.Synthesize(
        audio_pb2.SynthesizeRequest(text="hello", voice="nova", response_format="wav")
    )
    assert resp.audio_data == b"tts:hello"
    assert resp.format == "wav"
    assert resp.voice == "nova"
    assert resp.model == "fake-tts"


@pytest.mark.asyncio
async def test_grpc_transcribe(grpc_channel, monkeypatch):
    monkeypatch.setenv("LLMCORE_BRIDGE_FAKE_AUDIO", "1")
    stub = audio_pb2_grpc.AudioServiceStub(grpc_channel)
    resp = await stub.Transcribe(
        audio_pb2.TranscribeRequest(audio_data=b"hello world", language="en")
    )
    assert resp.text == "hello world"
    assert resp.language == "en"
    assert resp.model == "fake-stt"
    assert len(resp.segments) == 1
    assert resp.segments[0].text == "hello world"
    assert resp.segments[0].speaker == "spk_0"


@pytest.mark.asyncio
async def test_grpc_generate_image(grpc_channel, monkeypatch):
    monkeypatch.setenv("LLMCORE_BRIDGE_FAKE_AUDIO", "1")
    stub = audio_pb2_grpc.AudioServiceStub(grpc_channel)
    resp = await stub.GenerateImage(
        audio_pb2.GenerateImageRequest(prompt="a cat", n=2, size="512x512")
    )
    assert resp.model == "fake-img"
    assert len(resp.images) == 2
    assert base64.b64decode(resp.images[0].data) == b"img:a cat"
    assert resp.images[0].revised_prompt == "a cat"
    assert resp.images[0].format == "png"


@pytest.mark.asyncio
async def test_grpc_ocr(grpc_channel, monkeypatch):
    monkeypatch.setenv("LLMCORE_BRIDGE_FAKE_AUDIO", "1")
    stub = audio_pb2_grpc.AudioServiceStub(grpc_channel)
    resp = await stub.Ocr(audio_pb2.OcrRequest(data=b"PDFBYTES"))
    assert resp.model == "fake-ocr"
    assert resp.pages_processed == 1
    assert resp.doc_size_bytes == len(b"PDFBYTES")
    d = _d(resp)
    assert d["pages"][0]["text"] == "ocr-bytes"
    assert d["document_annotation"]["title"] == "fake-document"


@pytest.mark.asyncio
async def test_grpc_analyze_text(grpc_channel, monkeypatch):
    monkeypatch.setenv("LLMCORE_BRIDGE_FAKE_AUDIO", "1")
    stub = audio_pb2_grpc.AudioServiceStub(grpc_channel)
    req = audio_pb2.AnalyzeTextRequest(text="some text")
    req.features.update(
        {"summarize": True, "topics": True, "sentiment": True, "intents": True}
    )
    resp = await stub.AnalyzeText(req)
    d = _d(resp)
    assert d["summary"] == "summary:some text"
    assert d["topics"][0]["topic"] == "fake-topic"
    assert d["intents"][0]["intent"] == "fake-intent"
    assert d["sentiments"]["overall"] == "positive"
    assert d["model"] == "fake-analyze"
    assert d["request_id"] == "fake-req-1"


@pytest.mark.asyncio
async def test_grpc_synthesize_unimplemented_when_disabled(grpc_channel, monkeypatch):
    monkeypatch.delenv("LLMCORE_BRIDGE_FAKE_AUDIO", raising=False)
    stub = audio_pb2_grpc.AudioServiceStub(grpc_channel)
    with pytest.raises(grpc.aio.AioRpcError) as excinfo:
        await stub.Synthesize(audio_pb2.SynthesizeRequest(text="x"))
    assert excinfo.value.code() == grpc.StatusCode.UNIMPLEMENTED


# --------------------------------------------------------------------------- #
# HTTP / JSON
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_http_synthesize(http_client, monkeypatch):
    monkeypatch.setenv("LLMCORE_BRIDGE_FAKE_AUDIO", "1")
    r = await http_client.post(
        f"{_P}/Synthesize",
        json={"text": "hello", "voice": "nova", "response_format": "wav"},
    )
    assert r.status_code == 200
    body = r.json()
    assert base64.b64decode(body["audio_data"]) == b"tts:hello"
    assert body["format"] == "wav"
    assert body["voice"] == "nova"


@pytest.mark.asyncio
async def test_http_transcribe(http_client, monkeypatch):
    monkeypatch.setenv("LLMCORE_BRIDGE_FAKE_AUDIO", "1")
    audio_b64 = base64.b64encode(b"hello world").decode()
    r = await http_client.post(
        f"{_P}/Transcribe", json={"audio_data": audio_b64, "language": "en"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "hello world"
    assert body["language"] == "en"
    assert body["segments"][0]["speaker"] == "spk_0"


@pytest.mark.asyncio
async def test_http_generate_image(http_client, monkeypatch):
    monkeypatch.setenv("LLMCORE_BRIDGE_FAKE_AUDIO", "1")
    r = await http_client.post(f"{_P}/GenerateImage", json={"prompt": "a cat", "n": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "fake-img"
    assert len(body["images"]) == 3
    assert base64.b64decode(body["images"][0]["data"]) == b"img:a cat"


@pytest.mark.asyncio
async def test_http_ocr_url(http_client, monkeypatch):
    monkeypatch.setenv("LLMCORE_BRIDGE_FAKE_AUDIO", "1")
    url = "http://example.com/doc.pdf"
    r = await http_client.post(f"{_P}/Ocr", json={"url": url})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "fake-ocr"
    assert body["pages"][0]["text"] == f"ocr:{url}"
    assert body["document_annotation"]["title"] == "fake-document"


@pytest.mark.asyncio
async def test_http_analyze_text(http_client, monkeypatch):
    monkeypatch.setenv("LLMCORE_BRIDGE_FAKE_AUDIO", "1")
    r = await http_client.post(
        f"{_P}/AnalyzeText",
        json={"text": "some text", "features": {"summarize": True, "topics": True}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["summary"] == "summary:some text"
    assert body["topics"][0]["topic"] == "fake-topic"
    assert body["language"] == "en"


@pytest.mark.asyncio
async def test_http_ocr_missing_source_is_400(http_client, monkeypatch):
    monkeypatch.setenv("LLMCORE_BRIDGE_FAKE_AUDIO", "1")
    r = await http_client.post(f"{_P}/Ocr", json={})
    assert r.status_code == 400
    assert "error" in r.json()


@pytest.mark.asyncio
async def test_http_synthesize_501_when_disabled(http_client, monkeypatch):
    monkeypatch.delenv("LLMCORE_BRIDGE_FAKE_AUDIO", raising=False)
    r = await http_client.post(f"{_P}/Synthesize", json={"text": "x"})
    assert r.status_code == 501
    assert "error" in r.json()
