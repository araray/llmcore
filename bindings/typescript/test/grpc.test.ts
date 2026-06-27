import assert from "node:assert/strict";
import { after, before, test } from "node:test";

import { BridgeError, LlmcoreGrpcClient } from "../src/index";
import { startBridge, type BridgeHandle } from "./bridge";

let bridge: BridgeHandle;
let client: LlmcoreGrpcClient;

before(async () => {
  bridge = await startBridge();
  client = new LlmcoreGrpcClient(bridge.grpcAddress);
});

after(async () => {
  client?.close();
  await bridge?.stop();
});

test("ensureCompatible accepts llmcore.v1 + tier0", async () => {
  const info = await client.ensureCompatible(["tier0"]);
  assert.equal(info.contractVersion, "llmcore.v1");
  assert.ok(info.capabilities.includes("transport.grpc"));
  assert.ok(!info.capabilities.some((c) => c.startsWith("tier2.")));
});

test("ensureCompatible rejects a missing capability", async () => {
  await assert.rejects(
    () => client.ensureCompatible(["tier2.audio"]),
    (err: unknown) => err instanceof BridgeError && err.code === "capability.missing",
  );
});

test("chat returns echo + usage", async () => {
  const res = await client.chat({ message: "hello world" });
  assert.equal(res.text, "echo: hello world");
  assert.equal(res.usage?.promptTokens, 2);
  assert.ok((res.usage?.totalTokens ?? 0) > 0);
});

test("chatStream concatenates to the unary text", async () => {
  const chunks: string[] = [];
  let done = false;
  for await (const chunk of client.chatStream({ message: "stream this please" })) {
    if (chunk.done) done = true;
    else chunks.push(chunk.text);
  }
  assert.equal(done, true);
  assert.equal(chunks.join(""), "echo: stream this please");
});

test("countTokens", async () => {
  const res = await client.countTokens({ text: "one two three four" });
  assert.equal(res.tokens, 4);
});

test("estimateCost", async () => {
  const res = await client.estimateCost({
    providerName: "fake",
    modelName: "fake-1",
    promptTokens: 1_000_000,
    completionTokens: 1_000_000,
  });
  assert.equal(res.currency, "USD");
  assert.equal(res.totalCost, 3);
});

test("catalog: providers / models / details", async () => {
  assert.deepEqual((await client.listProviders()).providers, ["fake"]);
  assert.deepEqual((await client.listModels("fake")).models, ["fake-1", "fake-2"]);
  const details = await client.getProviderDetails("fake");
  assert.equal(details.id, "fake-1");
  assert.equal(details.contextLength, 8192);
});

test("Embed is UNIMPLEMENTED (UNSUPPORTED)", async () => {
  await assert.rejects(
    () => client.embed({ input: ["x"] }),
    (err: unknown) =>
      err instanceof BridgeError && err.category === "ERROR_CATEGORY_UNSUPPORTED",
  );
});

test("provider rate-limit decodes structured error", async () => {
  await assert.rejects(
    () => client.chat({ message: "__error__:provider_rate_limited" }),
    (err: unknown) => {
      assert.ok(err instanceof BridgeError);
      assert.equal(err.category, "ERROR_CATEGORY_PROVIDER");
      assert.equal(err.code, "provider.rate_limited");
      assert.equal(err.httpStatus, 429);
      assert.equal(err.retryable, true);
      assert.equal(err.retryAfterMs, 2000);
      assert.equal(err.provider, "fake");
      return true;
    },
  );
});

test("chatStream cancellation stops the stream", async () => {
  const stream = client.chatStream({ message: "__cancel__" });
  const it = stream[Symbol.asyncIterator]();
  const first = await it.next();
  assert.equal(first.done, false);
  assert.ok(first.value.text.length > 0);
  stream.cancel();
  await assert.rejects(async () => {
    // Subsequent reads reject with a CANCELLED-mapped BridgeError.
    for (;;) {
      const n = await it.next();
      if (n.done) break;
    }
  });
});
