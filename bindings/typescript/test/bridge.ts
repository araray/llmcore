/**
 * Test harness: spawn the Python `llmcore-bridge` (with the deterministic
 * FakeFacade) over localhost TCP, wait until both transports are ready, and
 * return a handle with `stop()`.
 *
 * The Python interpreter is resolved from $LLMCORE_BRIDGE_PYTHON (default
 * "python3"); it must have `llmcore[bridge]` importable. CI for this package
 * sets it to the project venv.
 */
import { type ChildProcess, spawn } from "node:child_process";
import { createServer } from "node:net";

import { LlmcoreGrpcClient } from "../src/grpcClient";

export interface BridgeHandle {
  grpcAddress: string;
  httpBase: string;
  stop: () => Promise<void>;
}

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = createServer();
    srv.once("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const addr = srv.address();
      const port = typeof addr === "object" && addr ? addr.port : 0;
      srv.close(() => resolve(port));
    });
  });
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function waitHttpReady(httpBase: string, timeoutMs = 25_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  let lastErr: unknown;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${httpBase}/healthz`);
      if (res.ok) {
        const body = (await res.json()) as { ok?: boolean };
        if (body.ok === true) return;
      }
    } catch (e) {
      lastErr = e;
    }
    await sleep(200);
  }
  throw new Error(`HTTP bridge not ready at ${httpBase}: ${String(lastErr)}`);
}

async function waitGrpcReady(grpcAddress: string, timeoutMs = 25_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  let lastErr: unknown;
  while (Date.now() < deadline) {
    const client = new LlmcoreGrpcClient(grpcAddress);
    try {
      const status = await client.health();
      if (status.ok) {
        client.close();
        return;
      }
    } catch (e) {
      lastErr = e;
    } finally {
      client.close();
    }
    await sleep(200);
  }
  throw new Error(`gRPC bridge not ready at ${grpcAddress}: ${String(lastErr)}`);
}

export interface StartBridgeOptions {
  /** Enable the Tier-2 fake audio surface (advertises tier2.audio + audio.*). */
  audio?: boolean;
  /** Enable the Tier-1 fake sessions + vector stores (advertises tier1.*). */
  sessions?: boolean;
}

export async function startBridge(opts: StartBridgeOptions = {}): Promise<BridgeHandle> {
  const grpcPort = await freePort();
  const httpPort = await freePort();
  const grpcAddress = `127.0.0.1:${grpcPort}`;
  const httpBase = `http://127.0.0.1:${httpPort}`;
  const python = process.env.LLMCORE_BRIDGE_PYTHON ?? "python3";

  const env: NodeJS.ProcessEnv = { ...process.env, LLMCORE_BRIDGE_FAKE: "1" };
  if (opts.audio) env.LLMCORE_BRIDGE_FAKE_AUDIO = "1";
  if (opts.sessions) {
    env.LLMCORE_BRIDGE_FAKE_SESSIONS = "1";
    env.LLMCORE_BRIDGE_FAKE_VECTOR = "1";
  }

  const proc: ChildProcess = spawn(
    python,
    [
      "-m", "llmcore.bridge.cli", "serve",
      "--transport", "grpc,http",
      "--grpc-address", grpcAddress,
      "--http-address", `127.0.0.1:${httpPort}`,
      "--insecure", "--log-level", "WARNING",
    ],
    { env, stdio: "ignore" },
  );

  const exited = new Promise<void>((resolve) => proc.once("exit", () => resolve()));

  try {
    await waitGrpcReady(grpcAddress);
    await waitHttpReady(httpBase);
  } catch (err) {
    proc.kill("SIGKILL");
    throw err;
  }

  const stop = async (): Promise<void> => {
    if (proc.exitCode === null && proc.signalCode === null) {
      proc.kill("SIGTERM");
      await Promise.race([exited, sleep(8_000)]);
      if (proc.exitCode === null && proc.signalCode === null) proc.kill("SIGKILL");
    }
  };

  return { grpcAddress, httpBase, stop };
}
