# Using llmcore-go

## Prerequisites

A running bridge. For local dev with the deterministic fake backend:

```bash
LLMCORE_BRIDGE_FAKE=1 python -m llmcore.bridge.cli serve \
  --transport grpc --grpc-address 127.0.0.1:50151 --insecure
```

## Connect

```go
c, err := llmcore.Dial("127.0.0.1:50151")          // insecure (dev only)
if err != nil { /* ... */ }
defer c.Close()
```

The connection is lazy (`grpc.NewClient`); the first RPC establishes it.

### TLS / mTLS

```go
import (
	"google.golang.org/grpc/credentials"
	llmcore "github.com/araray/llmcore-go"
)

creds, _ := credentials.NewClientTLSFromFile("ca.crt", "" /*serverNameOverride*/)
c, _ := llmcore.Dial("host:50151", llmcore.WithTransportCredentials(creds))
```

For mTLS build `credentials.NewTLS(&tls.Config{...})` with a client cert and pass
it the same way. Arbitrary tuning is available via `llmcore.WithDialOption(...)`.

## Capability negotiation

```go
info, err := c.EnsureCompatible(ctx, "tier0")      // variadic required caps
// err is a *BridgeError with Code "contract.mismatch" or "capability.missing"
```

## Inference

```go
// Unary
res, err := c.Chat(ctx, &llmcorev1.ChatRequest{
	Message:      "Summarize the contract.",
	ProviderName: proto.String("openai"),          // optional fields are pointers
	ModelName:    proto.String("gpt-4o-mini"),
	ProviderKwargs: mustStruct(map[string]any{      // google.protobuf.Struct
		"temperature": 0.2,
	}),
})
_ = res.GetText(); _ = res.GetUsage().GetTotalTokens()

// Streaming (+ cancellation)
stream, err := c.ChatStream(ctx, &llmcorev1.ChatRequest{Message: "Write a long answer."})
go func() { time.Sleep(time.Second); stream.Close() }() // optional: stop early
for {
	chunk, err := stream.Recv()
	if err == io.EOF { break }
	if err != nil { /* *BridgeError (e.g. CANCELLED) */ break }
	if !chunk.GetDone() { fmt.Print(chunk.GetText()) }
}
```

Helpers for the optional/struct fields:

```go
import (
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/structpb"
)

func mustStruct(m map[string]any) *structpb.Struct {
	s, err := structpb.NewStruct(m)
	if err != nil { panic(err) }
	return s
}
// proto.String / proto.Int32 / proto.Float64 create *T for optional fields.
```

## Tokens, cost, catalog

```go
c.CountTokens(ctx, &llmcorev1.CountTokensRequest{Text: "a b c"})            // GetTokens()
c.EstimateCost(ctx, &llmcorev1.EstimateCostRequest{
	ProviderName: "openai", ModelName: "gpt-4o-mini",
	PromptTokens: 1000, CompletionTokens: 500,
})
c.ListProviders(ctx)                 // GetProviders() []string
c.ListModels(ctx, "openai")          // GetModels()    []string
c.GetProviderDetails(ctx, "openai")  // *llmcorev1.ModelDetails
```

## Sessions, vector store & presets (Tier 1)

Available when the bridge advertises `tier1.sessions` (sessions, context items,
presets) and/or `tier1.vector` (vector store / RAG); negotiate with
`c.EnsureCompatible(ctx, "tier1.sessions", "tier1.vector")`. With the fake
backend, enable the stores via `LLMCORE_BRIDGE_FAKE_SESSIONS=1` and
`LLMCORE_BRIDGE_FAKE_VECTOR=1` on the bridge.

```go
// Sessions + context items
s, _ := c.CreateSession(ctx, &llmcorev1.CreateSessionRequest{Name: proto.String("demo")})
add, _ := c.AddContextItem(ctx, &llmcorev1.AddContextItemRequest{
	SessionId: s.GetId(), Content: "launch date is June 30.", Type: proto.String("user_text"),
})
item, _ := c.GetContextItem(ctx, &llmcorev1.GetContextItemRequest{
	SessionId: s.GetId(), ItemId: add.GetItemId(),
})
_ = item.GetContent()
// also: GetSession, ListSessions, UpdateSessionName, ForkSession, CloneSession,
//       DeleteMessages, GetMessagesByRange, RemoveContextItem, DeleteSession.

// Vector store / RAG
doc, _ := structpb.NewStruct(map[string]any{"content": "Paris is the capital of France."})
ids, _ := c.AddDocuments(ctx, &llmcorev1.AddDocumentsRequest{Documents: []*structpb.Struct{doc}})
hits, _ := c.SearchVectorStore(ctx, &llmcorev1.SearchVectorStoreRequest{Query: "capital of France", K: 3})
for _, d := range hits.GetDocuments() { fmt.Printf("%.3f %q\n", d.GetScore(), d.GetContent()) }
_ = ids.GetIds()
// also: ListVectorCollections, ListRagCollections, GetRagCollectionInfo, DeleteRagCollection.

// Context presets
_, _ = c.SaveContextPreset(ctx, &llmcorev1.SaveContextPresetRequest{Preset: &llmcorev1.ContextPreset{
	Name:  "preamble",
	Items: []*llmcorev1.ContextPresetItem{{Type: "preset_text_content", Content: proto.String("Always cite sources.")}},
}})
p, _ := c.GetContextPreset(ctx, &llmcorev1.GetContextPresetRequest{PresetName: "preamble"})
_ = p.GetItems()
// also: ListContextPresets, DeleteContextPreset.
```

A full runnable demo lives in `examples/sessions`:

```bash
LLMCORE_BRIDGE_FAKE=1 LLMCORE_BRIDGE_FAKE_SESSIONS=1 LLMCORE_BRIDGE_FAKE_VECTOR=1 \
  python -m llmcore.bridge.cli serve --transport grpc \
  --grpc-address 127.0.0.1:50151 --insecure
LLMCORE_GRPC=127.0.0.1:50151 GOFLAGS=-mod=mod go run ./examples/sessions
```

## Audio (Tier 2)

Available when the bridge advertises `tier2.audio`; negotiate with
`c.EnsureCompatible(ctx, "tier2.audio")`.

### One-shot

```go
sp, _ := c.Synthesize(ctx, &llmcorev1.SynthesizeRequest{Text: "hello"}) // sp.GetAudioData() []byte
tr, _ := c.Transcribe(ctx, &llmcorev1.TranscribeRequest{AudioData: pcm}) // tr.GetText(), tr.GetSegments()
img, _ := c.GenerateImage(ctx, &llmcorev1.GenerateImageRequest{Prompt: "a cat", N: 2})
//   img.GetImages()[0].GetData() is base64 (b64_json): base64.StdEncoding.DecodeString(...)
doc, _ := c.OCR(ctx, &llmcorev1.OcrRequest{Source: &llmcorev1.OcrRequest_Data{Data: pdf}}) // or _Url{Url: ...}
//   doc.GetPages() []*structpb.Struct, doc.GetDocumentAnnotation() *structpb.Value
feat, _ := structpb.NewStruct(map[string]any{"summarize": true, "topics": true})
an, _ := c.AnalyzeText(ctx, &llmcorev1.AnalyzeTextRequest{Text: "…", Features: feat})
//   an.GetSummary(), an.GetTopics(), an.GetIntents(), an.GetSentiments()
```

### Live duplex (bidi)

Each stream is `Send` / `Recv` / `CloseSend`, with `io.EOF` at the clean end and
`Close()` to cancel (gRPC CANCELLED). Send all frames then `CloseSend()`, then
drain with `Recv()`:

```go
// STT
stt, _ := c.TranscribeStream(ctx)
_ = stt.Send(&llmcorev1.AudioIn{Frame: &llmcorev1.AudioIn_Audio{Audio: pcmChunk}})
_ = stt.Send(&llmcorev1.AudioIn{Frame: &llmcorev1.AudioIn_Control{Control: llmcorev1.SttControl_STT_CONTROL_CLOSE}})
_ = stt.CloseSend()
for {
	ev, err := stt.Recv()
	if err == io.EOF { break }
	if err != nil { /* *BridgeError */ }
	if ev.GetType() == llmcorev1.StreamEventType_STREAM_EVENT_TYPE_FINAL { fmt.Println(ev.GetText()) }
}

// TTS
tts, _ := c.SynthesizeStream(ctx)
_ = tts.Send(&llmcorev1.SynthControl{Frame: &llmcorev1.SynthControl_Text{Text: "hello world"}})
_ = tts.Send(&llmcorev1.SynthControl{Frame: &llmcorev1.SynthControl_Control{Control: llmcorev1.TtsControl_TTS_CONTROL_CLOSE}})
_ = tts.CloseSend()
for { out, err := tts.Recv(); if err == io.EOF { break }; sink.Write(out.GetAudio()) } // out.GetSeq() int64

// Voice agent (leading settings Struct selects the provider)
settings, _ := structpb.NewStruct(map[string]any{"provider_name": "deepgram"})
va, _ := c.VoiceAgent(ctx)
_ = va.Send(&llmcorev1.VoiceAgentClientEvent{Event: &llmcorev1.VoiceAgentClientEvent_Settings{Settings: settings}})
_ = va.Send(&llmcorev1.VoiceAgentClientEvent{Event: &llmcorev1.VoiceAgentClientEvent_InjectUserMessage{InjectUserMessage: "hi"}})
_ = va.CloseSend()
for {
	ev, err := va.Recv()
	if err == io.EOF { break }
	if ev.GetType() == llmcorev1.VoiceAgentEventType_VOICE_AGENT_EVENT_TYPE_AUDIO { play(ev.GetAudio()) }
}
```

`structpb` is `google.golang.org/protobuf/types/known/structpb`.

## Error handling

```go
res, err := c.Chat(ctx, req)
if err != nil {
	var be *llmcore.BridgeError
	if errors.As(err, &be) {
		log.Printf("category=%s code=%s http=%d retryable=%t after=%.0fms provider=%s",
			be.Category, be.Code, be.HTTPStatus, be.Retryable, be.RetryAfterMs, be.Provider)
		if be.Retryable && be.RetryAfterMs > 0 {
			time.Sleep(time.Duration(be.RetryAfterMs) * time.Millisecond)
			// ...retry
		}
	}
}
```

Common `Code`s: `provider.rate_limited` (HTTP 429, `Retryable`, `RetryAfterMs`
set), `provider.unauthenticated` (401), `context.too_long` (413),
`not_found.session` (404), `unsupported.capability` (UNSUPPORTED),
`invalid_argument` (400), `internal` (500).

## Make targets

```bash
make gen     # regenerate stubs (buf + protoc-gen-go[-grpc])
make tidy    # go mod tidy
make build   # go build ./...
make vet     # go vet ./...
make test    # go test ./...  (set LLMCORE_BRIDGE_PYTHON first)
make all     # gen tidy build vet test
```
