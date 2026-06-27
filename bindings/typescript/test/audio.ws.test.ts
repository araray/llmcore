/**
 * WebSocket audio (Tier 2) e2e tests: the three live duplex RPCs over the
 * WebSocket transport, against an audio-enabled fake bridge, plus the
 * disabled-audio close path.
 *
 * Frames are sent/received as proto-as-JSON; `write` is async (it awaits the
 * socket open) and `end()` sends the `{}` end-of-input sentinel.
 */
import assert from "node:assert/strict";
import { after, before, test } from "node:test";

import {
  BridgeError,
  LlmcoreWsAudioClient,
  StreamEventType,
  VoiceAgentEventType,
} from "../src/index";
import { startBridge, type BridgeHandle } from "./bridge";

let bridge: BridgeHandle;
let ws: LlmcoreWsAudioClient;

before(async () => {
  bridge = await startBridge({ audio: true });
  ws = new LlmcoreWsAudioClient(bridge.httpBase);
});

after(async () => {
  await bridge?.stop();
});

test("ws transcribeStream: interim x2 + final + utterance-end", async () => {
  const stream = ws.transcribeStream();
  await stream.write({ open: { model: "fake-stt" } });
  await stream.write({ audio: new TextEncoder().encode("hello") });
  await stream.write({ audio: new TextEncoder().encode("world") });
  await stream.end();

  const events = [];
  for await (const ev of stream) events.push(ev);

  assert.deepEqual(
    events.map((e) => e.type),
    [
      StreamEventType.STREAM_EVENT_TYPE_INTERIM,
      StreamEventType.STREAM_EVENT_TYPE_INTERIM,
      StreamEventType.STREAM_EVENT_TYPE_FINAL,
      StreamEventType.STREAM_EVENT_TYPE_UTTERANCE_END,
    ],
  );
  assert.equal(
    events.find((e) => e.type === StreamEventType.STREAM_EVENT_TYPE_FINAL)?.text,
    "hello world",
  );
});

test("ws synthesizeStream: audio chunks with monotonic seq", async () => {
  const pieces = ["foo", "bar", "baz"];
  const stream = ws.synthesizeStream();
  await stream.write({ open: { model: "fake-tts" } });
  for (const p of pieces) await stream.write({ text: p });
  await stream.end();

  const outs = [];
  for await (const o of stream) outs.push(o);

  assert.deepEqual(
    outs.map((o) => new TextDecoder().decode(o.audio)),
    pieces,
  );
  assert.deepEqual(
    outs.map((o) => o.seq),
    [0, 1, 2],
  );
});

test("ws voiceAgent duplex: welcome ... close, with conversation + audio", async () => {
  const stream = ws.voiceAgent();
  await stream.write({ settings: { provider_name: "fake" } });
  await stream.write({ injectUserMessage: "hi there" });
  await stream.write({ audio: new Uint8Array([1, 2]) });
  await stream.end();

  const events = [];
  for await (const ev of stream) events.push(ev);

  assert.equal(events[0]?.type, VoiceAgentEventType.VOICE_AGENT_EVENT_TYPE_WELCOME);
  assert.equal(
    events[events.length - 1]?.type,
    VoiceAgentEventType.VOICE_AGENT_EVENT_TYPE_CLOSE,
  );
  const conv = events.find(
    (e) => e.type === VoiceAgentEventType.VOICE_AGENT_EVENT_TYPE_CONVERSATION_TEXT,
  );
  assert.equal(conv?.role, "user");
  assert.equal(conv?.content, "hi there");
  const audioEv = events.find(
    (e) => e.type === VoiceAgentEventType.VOICE_AGENT_EVENT_TYPE_AUDIO,
  );
  assert.deepEqual(Array.from(audioEv?.audio ?? []), [...Buffer.from("agent:"), 1, 2]);
});

test("ws voiceAgent on a disabled bridge closes with a BridgeError", async () => {
  const off = await startBridge(); // audio off
  try {
    const offWs = new LlmcoreWsAudioClient(off.httpBase);
    const stream = offWs.voiceAgent();
    await assert.rejects(
      async () => {
        for await (const _ev of stream) {
          // drain — should reject before yielding anything
        }
      },
      (err: unknown) => err instanceof BridgeError,
    );
  } finally {
    await off.stop();
  }
});
