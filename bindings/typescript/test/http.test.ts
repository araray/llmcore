import assert from "node:assert/strict";
import { after, before, test } from "node:test";

import { BridgeError, LlmcoreHttpClient } from "../src/index";
import { startBridge, type BridgeHandle } from "./bridge";

let bridge: BridgeHandle;
let client: LlmcoreHttpClient;

before(async () => {
  bridge = await startBridge();
  client = new LlmcoreHttpClient(bridge.httpBase);
});

after(async () => {
  await bridge?.stop();
});

test("ensureCompatible accepts llmcore.v1 + tier0", async () => {
  const info = await client.ensureCompatible(["tier0"]);
  assert.equal(info.contract_version, "llmcore.v1");
  assert.ok(info.capabilities.includes("transport.http"));
});

test("chat returns echo + usage", async () => {
  const res = await client.chat({ message: "hello world" });
  assert.equal(res.text, "echo: hello world");
  assert.equal(res.usage?.prompt_tokens, 2);
});

test("chatStream (SSE) concatenates to the unary text", async () => {
  const chunks: string[] = [];
  let done = false;
  for await (const chunk of client.chatStream({ message: "stream this please" })) {
    if (chunk.done) done = true;
    else if (chunk.text) chunks.push(chunk.text);
  }
  assert.equal(done, true);
  assert.equal(chunks.join(""), "echo: stream this please");
});

test("countTokens + estimateCost", async () => {
  assert.equal((await client.countTokens("a b c")).tokens, 3);
  const cost = await client.estimateCost({
    provider_name: "fake",
    model_name: "fake-1",
    prompt_tokens: 1_000_000,
    completion_tokens: 1_000_000,
  });
  assert.equal(cost.currency, "USD");
  assert.equal(cost.total_cost, 3);
});

test("catalog + health", async () => {
  assert.deepEqual((await client.listProviders()).providers, ["fake"]);
  assert.deepEqual((await client.listModels("fake")).models, ["fake-1", "fake-2"]);
  assert.equal((await client.getProviderDetails("fake")).id, "fake-1");
  assert.equal((await client.health()).ok, true);
});

test("Embed returns 501 (UNSUPPORTED)", async () => {
  await assert.rejects(
    () => client.embed({ input: ["x"] }),
    (err: unknown) =>
      err instanceof BridgeError &&
      err.httpStatus === 501 &&
      err.category === "ERROR_CATEGORY_UNSUPPORTED",
  );
});

test("provider rate-limit -> 429 structured error", async () => {
  await assert.rejects(
    () => client.chat({ message: "__error__:provider_rate_limited" }),
    (err: unknown) => {
      assert.ok(err instanceof BridgeError);
      assert.equal(err.httpStatus, 429);
      assert.equal(err.code, "provider.rate_limited");
      assert.equal(err.retryable, true);
      assert.equal(err.retryAfterMs, 2000);
      return true;
    },
  );
});

test("mid-stream error surfaces as an error event", async () => {
  const seen: string[] = [];
  await assert.rejects(
    async () => {
      for await (const chunk of client.chatStream({ message: "__error_mid__:internal" })) {
        if (chunk.text) seen.push(chunk.text);
      }
    },
    (err: unknown) =>
      err instanceof BridgeError && err.category === "ERROR_CATEGORY_INTERNAL",
  );
  assert.ok(seen.length > 0);
});
