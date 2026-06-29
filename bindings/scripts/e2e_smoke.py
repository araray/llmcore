#!/usr/bin/env python3
"""Golden end-to-end smoke test for the llmcore bridge.

Spawns ``llmcore-bridge serve`` as a real subprocess over localhost TCP (with
the deterministic ``FakeFacade`` and ``--insecure``), waits for readiness, then
drives **both** transports from a separate client process and asserts the
results. Also verifies the managed-subprocess lifecycle: a SIGTERM produces a
clean, prompt shutdown (spec §12.1, §13.3).

This exercises the full wire path (real HTTP/2 + protobuf for gRPC, real TCP +
SSE for HTTP) end to end, unlike the in-process conformance suite.

Run::

    python bindings/scripts/e2e_smoke.py            # auto-selects free ports

Exit code 0 == PASS.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time

import grpc
import httpx

from llmcore.bridge._generated.llmcore.v1 import (
    catalog_pb2_grpc,
    common_pb2,
    control_pb2_grpc,
    inference_pb2,
    inference_pb2_grpc,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_http_ready(base_url: str, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base_url}/healthz", timeout=1.0)
            if r.status_code == 200 and r.json().get("ok") is True:
                return
        except Exception as exc:
            last = exc
        time.sleep(0.2)
    raise RuntimeError(f"HTTP bridge not ready at {base_url} ({last})")


def _wait_grpc_ready(address: str, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            with grpc.insecure_channel(address) as channel:
                stub = control_pb2_grpc.ControlServiceStub(channel)
                status = stub.Health(common_pb2.Empty(), timeout=1.0)
                if status.ok:
                    return
        except Exception as exc:
            last = exc
        time.sleep(0.2)
    raise RuntimeError(f"gRPC bridge not ready at {address} ({last})")


def _check_grpc(address: str) -> None:
    with grpc.insecure_channel(address) as channel:
        control = control_pb2_grpc.ControlServiceStub(channel)
        info = control.GetInfo(common_pb2.Empty())
        assert info.contract_version == "llmcore.v1", info.contract_version
        assert "tier0" in info.capabilities
        assert not any(c.startswith("tier2.") for c in info.capabilities)

        inference = inference_pb2_grpc.InferenceServiceStub(channel)
        resp = inference.Chat(inference_pb2.ChatRequest(message="hello world"))
        assert resp.text == "echo: hello world", resp.text
        assert resp.usage.total_tokens > 0

        chunks = [c.text for c in inference.ChatStream(
            inference_pb2.ChatRequest(message="stream this")) if not c.done]
        assert "".join(chunks) == "echo: stream this", chunks

        tok = inference.CountTokens(inference_pb2.CountTokensRequest(text="a b c"))
        assert tok.tokens == 3

        # Embed must be UNIMPLEMENTED in v1.
        try:
            inference.Embed(inference_pb2.EmbedRequest(input=["x"]))
            raise AssertionError("Embed should be UNIMPLEMENTED")
        except grpc.RpcError as exc:
            assert exc.code() == grpc.StatusCode.UNIMPLEMENTED, exc.code()

        catalog = catalog_pb2_grpc.CatalogServiceStub(channel)
        assert list(catalog.ListProviders(common_pb2.Empty()).providers) == ["fake"]
    print("  [grpc] OK")


def _check_http(base_url: str) -> None:
    base = base_url.rstrip("/")
    with httpx.Client(base_url=base, timeout=30.0) as client:
        info = client.post("/llmcore.v1/ControlService/GetInfo", json={}).json()
        assert info["contract_version"] == "llmcore.v1"

        chat = client.post("/llmcore.v1/InferenceService/Chat",
                           json={"message": "hello world"}).json()
        assert chat["text"] == "echo: hello world", chat

        # SSE stream concatenation.
        collected = []
        with client.stream("POST", "/llmcore.v1/InferenceService/ChatStream",
                           json={"message": "stream this"}) as resp:
            assert resp.status_code == 200
            event = "message"
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    body = json.loads(line.split(":", 1)[1].strip())
                    if event == "message" and "text" in body:
                        collected.append(body["text"])
                    event = "message"
        assert "".join(collected) == "echo: stream this", collected

        r = client.post("/llmcore.v1/InferenceService/Embed", json={"input": ["x"]})
        assert r.status_code == 501, r.status_code
    print("  [http] OK")


def main() -> int:
    grpc_port = _free_port()
    http_port = _free_port()
    grpc_address = f"127.0.0.1:{grpc_port}"
    http_base = f"http://127.0.0.1:{http_port}"

    env = dict(os.environ, LLMCORE_BRIDGE_FAKE="1")
    cmd = [
        sys.executable, "-m", "llmcore.bridge.cli", "serve",
        "--transport", "grpc,http",
        "--grpc-address", grpc_address,
        "--http-address", f"127.0.0.1:{http_port}",
        "--insecure", "--log-level", "WARNING",
    ]
    print(f"spawning: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    try:
        _wait_grpc_ready(grpc_address)
        _wait_http_ready(http_base)
        _check_grpc(grpc_address)
        _check_http(http_base)
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            print("FAIL: bridge did not shut down within 15s of SIGTERM")
            return 1

    if proc.returncode not in (0, -signal.SIGTERM):
        print(f"FAIL: unexpected exit code {proc.returncode}")
        return 1
    print("E2E SMOKE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
