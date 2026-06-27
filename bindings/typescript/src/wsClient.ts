/**
 * WebSocket client for the llmcore bridge's live duplex audio RPCs (Tier 2).
 *
 * The HTTP/SSE client ({@link LlmcoreHttpClient}) cannot stream client -> server,
 * so bidirectional audio (`TranscribeStream`, `SynthesizeStream`, `VoiceAgent`)
 * uses WebSocket. Uses the global `WebSocket` (Node >= 22 / browsers).
 *
 * Wire format mirrors the bridge: one proto-as-JSON per **text** frame. Requests
 * are encoded with each message's `toJSON` (proto3-canonical JSON, which the
 * server's `ParseDict` accepts); responses are decoded with `fromJSON` (which
 * reads the bridge's snake_case output). End-of-input is the empty `{}` sentinel
 * ({@link WsAudioStream.end}) — WebSocket has no half-close, so this lets the
 * server keep streaming responses (e.g. the voice-agent `CLOSE`). The server
 * closes 1000 when its response stream ends, sends `{"error": {...}}` then closes
 * 1011 on failure, and (audio disabled) closes 1011 with no frame.
 */
import { BridgeError, bridgeErrorFromHttp, ErrorCategory } from "./errors";
import { errorCategoryToJSON } from "./gen/llmcore/v1/errors";
import {
  AudioIn,
  AudioOut,
  type DeepPartial,
  SynthControl,
  TranscriptionStreamEvent,
  VoiceAgentClientEvent,
  VoiceAgentEvent,
} from "./gen/llmcore/v1/audio";

type Encoder<Req> = (req: DeepPartial<Req>) => unknown;
type Decoder<Res> = (obj: unknown) => Res;

/** Minimal shape of a WebSocket close event (avoids depending on DOM lib types). */
interface WsCloseEventLike {
  code?: number;
  reason?: string;
}

function connectError(message: string): BridgeError {
  return new BridgeError({
    category: errorCategoryToJSON(ErrorCategory.ERROR_CATEGORY_INTERNAL),
    code: "websocket.connect_failed",
    message,
    retryable: true,
  });
}

/**
 * A bidirectional WebSocket stream: {@link write} request frames, {@link end} to
 * signal end-of-input, and async-iterate the server's response frames. Errors
 * (error frames or abnormal closes) surface as {@link BridgeError} when iterating.
 */
export class WsAudioStream<Req, Res> implements AsyncIterable<Res> {
  private readonly ws: WebSocket;
  private readonly buffer: Res[] = [];
  private resolveNext: ((r: IteratorResult<Res>) => void) | null = null;
  private rejectNext: ((e: unknown) => void) | null = null;
  private finished = false;
  private error: unknown = null;
  private readonly ready: Promise<void>;

  constructor(
    url: string,
    private readonly enc: Encoder<Req>,
    private readonly dec: Decoder<Res>,
  ) {
    this.ws = new WebSocket(url);
    this.ws.binaryType = "arraybuffer";

    this.ready = new Promise<void>((resolve, reject) => {
      const onOpen = () => {
        cleanup();
        resolve();
      };
      const onErr = () => {
        cleanup();
        reject(connectError(`WebSocket connection to ${url} failed`));
      };
      const cleanup = () => {
        this.ws.removeEventListener("open", onOpen);
        this.ws.removeEventListener("error", onErr);
      };
      this.ws.addEventListener("open", onOpen);
      this.ws.addEventListener("error", onErr);
    });
    // Avoid an unhandled rejection if the caller never iterates.
    this.ready.catch(() => undefined);

    this.ws.addEventListener("message", (ev) => this.onMessage(ev as MessageEvent));
    this.ws.addEventListener("close", (ev) =>
      this.onClose(ev as unknown as WsCloseEventLike),
    );
  }

  private onMessage(ev: MessageEvent): void {
    let text: string;
    if (typeof ev.data === "string") text = ev.data;
    else if (ev.data instanceof ArrayBuffer) text = new TextDecoder().decode(ev.data);
    else text = String(ev.data);

    let obj: unknown;
    try {
      obj = JSON.parse(text);
    } catch {
      return; // ignore malformed frame
    }
    if (obj && typeof obj === "object" && "error" in obj) {
      const errObj = (obj as { error: Record<string, unknown> }).error;
      const status = (errObj.http_status as number | undefined) ?? 500;
      this.finish(bridgeErrorFromHttp(status, { error: errObj }));
      try {
        this.ws.close();
      } catch {
        // already closing
      }
      return;
    }
    this.push(this.dec(obj));
  }

  private onClose(ev: WsCloseEventLike): void {
    if (this.finished) return;
    if (ev.code && ev.code !== 1000 && ev.code !== 1005) {
      this.finish(
        new BridgeError({
          category: errorCategoryToJSON(ErrorCategory.ERROR_CATEGORY_UNSUPPORTED),
          code: "websocket.closed",
          message: `WebSocket closed (code ${ev.code})${ev.reason ? ": " + ev.reason : ""}`,
          retryable: false,
        }),
      );
    } else {
      this.finish(null);
    }
  }

  private push(msg: Res): void {
    if (this.resolveNext) {
      const resolve = this.resolveNext;
      this.resolveNext = null;
      this.rejectNext = null;
      resolve({ value: msg, done: false });
    } else {
      this.buffer.push(msg);
    }
  }

  private finish(err: unknown): void {
    if (this.finished) return;
    this.finished = true;
    this.error = err;
    const resolve = this.resolveNext;
    const reject = this.rejectNext;
    this.resolveNext = null;
    this.rejectNext = null;
    if (err) {
      if (reject) reject(err);
    } else if (resolve) {
      resolve({ value: undefined as never, done: true });
    }
  }

  private pull(): Promise<IteratorResult<Res>> {
    if (this.buffer.length > 0) {
      return Promise.resolve({ value: this.buffer.shift() as Res, done: false });
    }
    if (this.finished) {
      return this.error
        ? Promise.reject(this.error)
        : Promise.resolve({ value: undefined as never, done: true });
    }
    return new Promise<IteratorResult<Res>>((resolve, reject) => {
      this.resolveNext = resolve;
      this.rejectNext = reject;
    });
  }

  /** Send one request frame (encoded via the message's `toJSON`). */
  async write(req: DeepPartial<Req>): Promise<void> {
    await this.ready;
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(this.enc(req)));
    }
  }

  /** Signal end-of-input (sends the `{}` sentinel). */
  async end(): Promise<void> {
    await this.ready;
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send("{}");
    }
  }

  /** Close the socket immediately. */
  close(): void {
    try {
      this.ws.close();
    } catch {
      // already closed
    }
  }

  async *[Symbol.asyncIterator](): AsyncIterator<Res> {
    await this.ready;
    for (;;) {
      const r = await this.pull();
      if (r.done) return;
      yield r.value;
    }
  }
}

/** Live-audio WebSocket client. Construct with the bridge HTTP base URL. */
export class LlmcoreWsAudioClient {
  private readonly base: string;

  /**
   * @param baseUrl The bridge base URL. `http(s)://` is rewritten to `ws(s)://`;
   *   an explicit `ws(s)://` URL is used as-is.
   */
  constructor(baseUrl: string) {
    this.base = baseUrl.replace(/^http/, "ws").replace(/\/+$/, "");
  }

  private path(name: string): string {
    return `${this.base}/llmcore.v1/AudioService/${name}`;
  }

  /** Bidi speech-to-text. Write `AudioIn` frames; iterate `TranscriptionStreamEvent`s. */
  transcribeStream(): WsAudioStream<AudioIn, TranscriptionStreamEvent> {
    return new WsAudioStream(
      this.path("TranscribeStream"),
      (r) => AudioIn.toJSON(AudioIn.fromPartial(r)),
      (o) => TranscriptionStreamEvent.fromJSON(o),
    );
  }

  /** Bidi text-to-speech. Write `SynthControl` frames; iterate `AudioOut` chunks. */
  synthesizeStream(): WsAudioStream<SynthControl, AudioOut> {
    return new WsAudioStream(
      this.path("SynthesizeStream"),
      (r) => SynthControl.toJSON(SynthControl.fromPartial(r)),
      (o) => AudioOut.fromJSON(o),
    );
  }

  /** Bidi voice agent. Write `VoiceAgentClientEvent`s; iterate `VoiceAgentEvent`s. */
  voiceAgent(): WsAudioStream<VoiceAgentClientEvent, VoiceAgentEvent> {
    return new WsAudioStream(
      this.path("VoiceAgent"),
      (r) => VoiceAgentClientEvent.toJSON(VoiceAgentClientEvent.fromPartial(r)),
      (o) => VoiceAgentEvent.fromJSON(o),
    );
  }
}
