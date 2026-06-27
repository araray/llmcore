#!/usr/bin/env python3
"""Reference client for the llmcore bridge (Python, synchronous).

Demonstrates calling the bridge over **both** transports against a running
``llmcore-bridge serve`` process. This is the B1 reference client; the foreign
language clients (C/C++/Rust/Go/TS) are delivered in phase B2 and generate their
stubs from the same ``bindings/proto`` contract.

Run a fake bridge in one terminal::

    LLMCORE_BRIDGE_FAKE=1 python -m llmcore.bridge.cli serve \
        --transport grpc,http \
        --grpc-address 127.0.0.1:50151 --http-address 127.0.0.1:50152 --insecure

then::

    python bindings/examples/python/client.py --transport both \
        --grpc-address 127.0.0.1:50151 --http-address http://127.0.0.1:50152
"""

from __future__ import annotations

import argparse
import json

import grpc
import httpx

from llmcore.bridge._generated.llmcore.v1 import (
    catalog_pb2,
    catalog_pb2_grpc,
    common_pb2,
    control_pb2_grpc,
    inference_pb2,
    inference_pb2_grpc,
)


def grpc_demo(address: str) -> None:
    """Exercise the gRPC transport with a synchronous channel."""
    print(f"\n=== gRPC @ {address} ===")
    with grpc.insecure_channel(address) as channel:
        control = control_pb2_grpc.ControlServiceStub(channel)
        info = control.GetInfo(common_pb2.Empty())
        print(f"server: llmcore={info.llmcore_version} bridge={info.bridge_version} "
              f"contract={info.contract_version}")
        print(f"capabilities: {list(info.capabilities)}")

        inference = inference_pb2_grpc.InferenceServiceStub(channel)
        resp = inference.Chat(inference_pb2.ChatRequest(message="hello from gRPC"))
        print(f"Chat -> {resp.text!r} (total_tokens={resp.usage.total_tokens})")

        print("ChatStream -> ", end="", flush=True)
        for chunk in inference.ChatStream(inference_pb2.ChatRequest(message="stream me")):
            if not chunk.done:
                print(chunk.text, end="", flush=True)
        print()

        tok = inference.CountTokens(inference_pb2.CountTokensRequest(text="one two three four"))
        print(f"CountTokens -> {tok.tokens}")

        cost = inference.EstimateCost(inference_pb2.EstimateCostRequest(
            provider_name="fake", model_name="fake-1",
            prompt_tokens=1000, completion_tokens=500))
        print(f"EstimateCost -> total={cost.total_cost:.6f} {cost.currency}")

        catalog = catalog_pb2_grpc.CatalogServiceStub(channel)
        providers = catalog.ListProviders(common_pb2.Empty())
        print(f"ListProviders -> {list(providers.providers)}")
        details = catalog.GetProviderDetails(catalog_pb2.GetProviderRequest(provider_name="fake"))
        print(f"GetProviderDetails -> id={details.id} ctx={details.context_length}")


def http_demo(base_url: str) -> None:
    """Exercise the HTTP/SSE transport."""
    print(f"\n=== HTTP @ {base_url} ===")
    base = base_url.rstrip("/")
    with httpx.Client(base_url=base, timeout=30.0) as client:
        info = client.post("/llmcore.v1/ControlService/GetInfo", json={}).json()
        print(f"server: contract={info['contract_version']} caps={info['capabilities']}")

        chat = client.post("/llmcore.v1/InferenceService/Chat",
                           json={"message": "hello from HTTP"}).json()
        print(f"Chat -> {chat['text']!r} (total_tokens={chat['usage'].get('total_tokens')})")

        print("ChatStream -> ", end="", flush=True)
        with client.stream("POST", "/llmcore.v1/InferenceService/ChatStream",
                           json={"message": "stream me"}) as resp:
            event = "message"
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    body = json.loads(line.split(":", 1)[1].strip())
                    if event == "message" and body.get("text"):
                        print(body["text"], end="", flush=True)
                    event = "message"
        print()

        tok = client.post("/llmcore.v1/InferenceService/CountTokens",
                          json={"text": "one two three four"}).json()
        print(f"CountTokens -> {tok['tokens']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="llmcore bridge reference client")
    parser.add_argument("--transport", choices=["grpc", "http", "both"], default="both")
    parser.add_argument("--grpc-address", default="127.0.0.1:50151")
    parser.add_argument("--http-address", default="http://127.0.0.1:50152")
    args = parser.parse_args(argv)

    if args.transport in ("grpc", "both"):
        grpc_demo(args.grpc_address)
    if args.transport in ("http", "both"):
        http_demo(args.http_address)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
