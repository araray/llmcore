/**
 * gRPC audio (Tier 2) e2e tests: the three live duplex RPCs and the five
 * one-shot RPCs, against an audio-enabled fake bridge.
 */
import assert from "node:assert/strict";
import { after, before, test } from "node:test";

import {
  LlmcoreGrpcClient,
  StreamEventType,
  SttControl,
  TtsControl,
  VoiceAgentEventType,
} from "../src/index";
import { startBridge, type BridgeHandle } from "./bridge";

let bridge: BridgeHandle;
let client: LlmcoreGrpcClient;

before(async () => {
  bridge = await startBridge({ audio: true });
  client = new LlmcoreGrpcClient(bridge.grpcAddress);
});

after(async () => {
  client?.close();
  await bridge?.stop();
});

test("audio capabilities are advertised when audio is on", async () => {
  const info = await client.getInfo();
  for (const cap of [
    "tier2.audio",
    "audio.transcribe_stream",
    "audio.synthesize_stream",
    "audio.voice_agent",
    "audio.synthesize",
    "audio.transcribe",
    "audio.generate_image",
    "audio.ocr",
    "audio.analyze_text",
  ]) {
    assert.ok(info.capabilities.includes(cap), `missing capability ${cap}`);
  }
});

// -- live duplex ---------------------------------------------------------- //
test("transcribeStream: interim x2 + final + utterance-end", async () => {
  const stream = client.transcribeStream();
  stream.write({ open: { model: "fake-stt" } });
  stream.write({ audio: new TextEncoder().encode("hello") });
  stream.write({ audio: new TextEncoder().encode("world") });
  stream.write({ control: SttControl.STT_CONTROL_CLOSE });
  stream.end();

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
  const finalEv = events.find((e) => e.type === StreamEventType.STREAM_EVENT_TYPE_FINAL);
  assert.equal(finalEv?.text, "hello world");
});

test("synthesizeStream: audio chunks with monotonic seq", async () => {
  const pieces = ["foo", "bar", "baz"];
  const stream = client.synthesizeStream();
  stream.write({ open: { model: "fake-tts" } });
  for (const p of pieces) stream.write({ text: p });
  stream.write({ control: TtsControl.TTS_CONTROL_CLOSE });
  stream.end();

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

test("voiceAgent duplex: welcome ... close, with conversation + audio", async () => {
  const stream = client.voiceAgent();
  stream.write({ settings: { provider_name: "fake" } });
  stream.write({ injectUserMessage: "hi there" });
  stream.write({ audio: new Uint8Array([1, 2]) });
  stream.end();

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

// -- one-shot ------------------------------------------------------------- //
test("synthesize", async () => {
  const r = await client.synthesize({ text: "hello", voice: "nova", responseFormat: "wav" });
  assert.equal(new TextDecoder().decode(r.audioData), "tts:hello");
  assert.equal(r.format, "wav");
  assert.equal(r.voice, "nova");
});

test("transcribe", async () => {
  const r = await client.transcribe({
    audioData: new TextEncoder().encode("hello world"),
    language: "en",
  });
  assert.equal(r.text, "hello world");
  assert.equal(r.language, "en");
  assert.equal(r.segments[0]?.speaker, "spk_0");
});

test("generateImage (n=2)", async () => {
  const r = await client.generateImage({ prompt: "a cat", n: 2 });
  assert.equal(r.images.length, 2);
  assert.equal(Buffer.from(r.images[0]?.data ?? "", "base64").toString("utf8"), "img:a cat");
});

test("ocr (bytes)", async () => {
  const r = await client.ocr({ data: new TextEncoder().encode("PDFBYTES") });
  assert.equal(r.model, "fake-ocr");
  assert.equal(r.pagesProcessed, 1);
  assert.equal(r.docSizeBytes, 8);
  assert.equal((r.pages[0] as { text: string }).text, "ocr-bytes");
  assert.equal((r.documentAnnotation as { title: string }).title, "fake-document");
});

test("analyzeText", async () => {
  const r = await client.analyzeText({
    text: "some text",
    features: { summarize: true, topics: true, sentiment: true, intents: true },
  });
  assert.equal(r.summary, "summary:some text");
  assert.equal((r.topics[0] as { topic: string }).topic, "fake-topic");
  assert.equal((r.intents[0] as { intent: string }).intent, "fake-intent");
  assert.equal((r.sentiments as { overall: string }).overall, "positive");
});
