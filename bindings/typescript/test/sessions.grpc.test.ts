/**
 * gRPC Tier-1 e2e tests: sessions, context items, vector store, and presets,
 * against a bridge with the fake T1 stores enabled.
 */
import assert from "node:assert/strict";
import { after, before, test } from "node:test";

import { BridgeError, LlmcoreGrpcClient } from "../src/index";
import { startBridge, type BridgeHandle } from "./bridge";

let bridge: BridgeHandle;
let client: LlmcoreGrpcClient;

before(async () => {
  bridge = await startBridge({ sessions: true });
  client = new LlmcoreGrpcClient(bridge.grpcAddress);
});

after(async () => {
  client?.close();
  await bridge?.stop();
});

test("tier1 capabilities are advertised", async () => {
  const info = await client.getInfo();
  assert.ok(info.capabilities.includes("tier1.sessions"));
  assert.ok(info.capabilities.includes("tier1.vector"));
  assert.ok(info.tiers.includes("T1"));
});

test("session create/get round-trip", async () => {
  const created = await client.createSession({
    name: "ts-chat",
    systemMessage: "be brief",
  });
  assert.equal(created.name, "ts-chat");
  assert.equal(created.messages.length, 1);
  assert.equal(created.messages[0].content, "be brief");

  const got = await client.getSession({ sessionId: created.id });
  assert.equal(got.id, created.id);
});

test("context item lifecycle", async () => {
  const s = await client.createSession({});
  const added = await client.addContextItem({
    sessionId: s.id,
    content: "a fact",
    type: "rag_snippet",
  });
  assert.ok(added.itemId);
  const item = await client.getContextItem({ sessionId: s.id, itemId: added.itemId });
  assert.equal(item.type, "rag_snippet");
  assert.equal(item.content, "a fact");

  const removed = await client.removeContextItem({ sessionId: s.id, itemId: added.itemId });
  assert.equal(removed.removed, true);
});

test("get missing session is NOT_FOUND", async () => {
  await assert.rejects(
    () => client.getSession({ sessionId: "ghost" }),
    (err: unknown) => {
      assert.ok(err instanceof BridgeError);
      assert.equal(err.category, "ERROR_CATEGORY_NOT_FOUND");
      return true;
    },
  );
});

test("vector add and search", async () => {
  const added = await client.addDocuments({
    documents: [{ content: "the cat sat", metadata: { topic: "a" } }],
  });
  assert.equal(added.ids.length, 1);
  const res = await client.searchVectorStore({ query: "cat" });
  assert.equal(res.documents.length, 1);
  assert.equal(res.documents[0].content, "the cat sat");
});

test("preset save/get round-trip", async () => {
  await client.saveContextPreset({
    preset: {
      name: "ts-preset",
      description: "d",
      items: [{ type: "preset_text_content", content: "boilerplate" }],
    },
  });
  const got = await client.getContextPreset({ presetName: "ts-preset" });
  assert.equal(got.name, "ts-preset");
  assert.equal(got.items.length, 1);
  assert.equal(got.items[0].content, "boilerplate");

  await assert.rejects(() => client.getContextPreset({ presetName: "ghost" }));
});
