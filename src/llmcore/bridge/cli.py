"""``llmcore-bridge`` console entry point (spec §12, §13).

``serve`` starts the out-of-process sidecar exposing the LLMCore facade over
gRPC and/or HTTP. Lifecycle features:

* **Transports** — ``--transport grpc,http`` (either/both).
* **Endpoints** — per-transport ``--grpc-address`` / ``--http-address``; a value
  prefixed ``unix:`` is a Unix-domain socket, otherwise ``host:port`` TCP.
* **Security policy (§13.3)** — TCP endpoints require TLS *and* an auth mode, or
  the explicit ``--insecure`` opt-out (localhost/dev only). UDS relies on
  filesystem permissions (local trust).
* **Managed-subprocess hygiene** — on Linux the process requests
  ``PR_SET_PDEATHSIG`` so it dies with its parent; SIGINT/SIGTERM trigger a
  graceful drain of both servers.
* **Facade** — real ``LLMCore`` by default; set ``LLMCORE_BRIDGE_FAKE=1`` to use
  the deterministic in-process fake (used by the e2e smoke and local demos).

AuthFlow token introspection is a declared integration point: with
``--auth authflow`` a presence-checking interceptor is installed; full mTLS +
AuthFlow assurance enforcement lands in the security phase.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from typing import Any

logger = logging.getLogger("llmcore.bridge")


# --------------------------------------------------------------------------- #
# process hygiene
# --------------------------------------------------------------------------- #
def _set_parent_death_signal() -> None:
    """On Linux, ask the kernel to SIGTERM us if our parent dies."""
    if not sys.platform.startswith("linux"):
        return
    try:  # pragma: no cover - platform/runtime dependent
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception as exc:  # pragma: no cover
        logger.debug("PR_SET_PDEATHSIG unavailable: %s", exc)


def _is_unix(address: str) -> bool:
    return address.startswith("unix:")


def _unix_path(address: str) -> str:
    return address[len("unix:"):]


# --------------------------------------------------------------------------- #
# security policy
# --------------------------------------------------------------------------- #
def _validate_security(args: argparse.Namespace, transports: list[str]) -> None:
    """Enforce §13.3: no plaintext TCP without an explicit opt-out."""
    tcp_endpoints = []
    if "grpc" in transports and not _is_unix(args.grpc_address):
        tcp_endpoints.append(("grpc", args.grpc_address))
    if "http" in transports and not _is_unix(args.http_address):
        tcp_endpoints.append(("http", args.http_address))
    if not tcp_endpoints:
        return
    if args.insecure:
        logger.warning(
            "Serving %s over PLAINTEXT TCP (--insecure). Use only on localhost/dev.",
            ", ".join(f"{n}@{a}" for n, a in tcp_endpoints),
        )
        return
    has_tls = bool(args.tls_cert and args.tls_key)
    if not has_tls:
        raise SystemExit(
            "Refusing to serve over TCP without TLS. Provide --tls-cert/--tls-key "
            "(and --auth), bind a unix: socket, or pass --insecure for localhost/dev."
        )
    if args.auth == "none":
        raise SystemExit(
            "Refusing to serve TLS TCP with --auth none. Use --auth authflow "
            "(with --authflow-url) or pass --insecure for localhost/dev."
        )


# --------------------------------------------------------------------------- #
# gRPC
# --------------------------------------------------------------------------- #
def _grpc_credentials(args: argparse.Namespace):
    import grpc

    with open(args.tls_key, "rb") as fh:
        key = fh.read()
    with open(args.tls_cert, "rb") as fh:
        cert = fh.read()
    root_certs = None
    require_client_auth = False
    if args.tls_client_ca:
        with open(args.tls_client_ca, "rb") as fh:
            root_certs = fh.read()
        require_client_auth = True  # mTLS
    return grpc.ssl_server_credentials(
        [(key, cert)],
        root_certificates=root_certs,
        require_client_auth=require_client_auth,
    )


def _build_grpc_server(core, args: argparse.Namespace):
    from .grpc_server import create_grpc_server

    interceptors = []
    if args.auth == "authflow":
        interceptors.append(_AuthFlowInterceptor(args.authflow_url))
    server = create_grpc_server(core, interceptors=interceptors)

    address = args.grpc_address
    if _is_unix(address):
        server.add_insecure_port(f"unix:{_unix_path(address)}")
    elif args.insecure or not (args.tls_cert and args.tls_key):
        server.add_insecure_port(address)
    else:
        server.add_secure_port(address, _grpc_credentials(args))
    return server


try:  # interceptor base import guarded so module imports without grpc at doc time
    import grpc as _grpc

    class _AuthFlowInterceptor(_grpc.aio.ServerInterceptor):  # type: ignore[misc]
        """Presence-checking auth gate (full AuthFlow introspection: security phase)."""

        def __init__(self, authflow_url: str | None) -> None:
            self._url = authflow_url

        async def intercept_service(self, continuation, handler_call_details):
            metadata = dict(handler_call_details.invocation_metadata or ())
            if "authorization" not in metadata and "x-authflow-token" not in metadata:
                async def deny(request, context):
                    await context.abort(
                        _grpc.StatusCode.UNAUTHENTICATED,
                        "missing AuthFlow credentials",
                    )

                return _grpc.aio.unary_unary_rpc_method_handler(deny)
            return await continuation(handler_call_details)
except Exception:  # pragma: no cover
    class _AuthFlowInterceptor:  # type: ignore[no-redef]
        def __init__(self, *_a, **_k): ...


# --------------------------------------------------------------------------- #
# HTTP (uvicorn)
# --------------------------------------------------------------------------- #
def _build_uvicorn(core, args: argparse.Namespace):
    import uvicorn

    from .http_app import create_http_app

    app = create_http_app(core)
    address = args.http_address
    kwargs: dict[str, Any] = {"log_level": args.log_level.lower(), "lifespan": "off"}
    if _is_unix(address):
        kwargs["uds"] = _unix_path(address)
    else:
        host, _, port = address.rpartition(":")
        kwargs["host"] = host or "127.0.0.1"
        kwargs["port"] = int(port)
    if not args.insecure and args.tls_cert and args.tls_key and not _is_unix(address):
        kwargs["ssl_certfile"] = args.tls_cert
        kwargs["ssl_keyfile"] = args.tls_key
        if args.tls_client_ca:
            import ssl

            kwargs["ssl_ca_certs"] = args.tls_client_ca
            kwargs["ssl_cert_reqs"] = ssl.CERT_REQUIRED
    config = uvicorn.Config(app, **kwargs)
    return uvicorn.Server(config)


# --------------------------------------------------------------------------- #
# serve
# --------------------------------------------------------------------------- #
async def _serve_async(args: argparse.Namespace) -> int:
    _set_parent_death_signal()
    transports = [t.strip() for t in args.transport.split(",") if t.strip()]
    if not transports:
        raise SystemExit("no transports selected (use --transport grpc,http)")
    _validate_security(args, transports)

    from .core import BridgeCore

    if os.getenv("LLMCORE_BRIDGE_FAKE") == "1":
        from ._testing import FakeFacade, fake_count_tokens

        facade: Any = FakeFacade()
        core = BridgeCore(facade, count_tokens=fake_count_tokens, transports=transports)
        logger.warning("Using in-process FakeFacade (LLMCORE_BRIDGE_FAKE=1).")
    else:
        from .facade import build_facade

        facade = await build_facade(
            config_file_path=args.config, env_prefix=args.env_prefix
        )
        core = BridgeCore(facade, transports=transports)

    grpc_server = None
    http_server = None
    http_task = None

    if "grpc" in transports:
        grpc_server = _build_grpc_server(core, args)
        await grpc_server.start()
        logger.info("gRPC serving on %s", args.grpc_address)
    if "http" in transports:
        http_server = _build_uvicorn(core, args)
        http_task = asyncio.create_task(http_server.serve())
        logger.info("HTTP serving on %s", args.http_address)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):  # pragma: no cover
            pass

    await stop.wait()
    logger.info("shutting down bridge...")

    if http_server is not None:
        http_server.should_exit = True
    if grpc_server is not None:
        await grpc_server.stop(grace=5)
    if http_task is not None:
        try:
            await asyncio.wait_for(http_task, timeout=10)
        except asyncio.TimeoutError:  # pragma: no cover
            http_task.cancel()
    await core.close()
    return 0


def _add_serve_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--transport", default="grpc,http", help="comma list: grpc,http")
    p.add_argument("--grpc-address", default="127.0.0.1:50151",
                   help="host:port or unix:/path for gRPC")
    p.add_argument("--http-address", default="127.0.0.1:50152",
                   help="host:port or unix:/path for HTTP")
    p.add_argument("--config", default=None, help="path to a TOML config (confy)")
    p.add_argument("--env-prefix", default="LLMCORE", help="env var prefix for config")
    p.add_argument("--tls-cert", default=None, help="server TLS certificate (PEM)")
    p.add_argument("--tls-key", default=None, help="server TLS private key (PEM)")
    p.add_argument("--tls-client-ca", default=None, help="client CA bundle to enable mTLS")
    p.add_argument("--auth", choices=["none", "authflow"], default="none",
                   help="auth mode for TCP endpoints")
    p.add_argument("--authflow-url", default=None, help="AuthFlow introspection base URL")
    p.add_argument("--insecure", action="store_true",
                   help="permit plaintext TCP (localhost/dev only)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llmcore-bridge", description="LLMCore sidecar bridge")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="run the bridge server")
    _add_serve_args(serve)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "serve":
        try:
            return asyncio.run(_serve_async(args))
        except KeyboardInterrupt:  # pragma: no cover
            return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
