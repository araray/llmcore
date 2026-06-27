# llmcore-go

Go client for the **llmcore bridge** over gRPC, generated from the frozen
`llmcore.v1` contract. Depends only on the contract, never on Python.

> **Build status / sandbox note.** The generated stubs (`gen/`) and a compiled
> build were **not produced in the authoring sandbox**: no Go toolchain was
> available there and the Go module proxy was not reachable. The hand-written
> client is complete and written against the stable `google.golang.org/grpc` +
> `protoc-gen-go` APIs; the `buf.gen.go.yaml` template was validated by `buf`
> 1.71 (it parsed and only failed on the absent `protoc-gen-go` binary). Run the
> two steps below in any environment with Go to generate + build + test.

## One-time: generate stubs and resolve deps

```bash
# protoc plugins (Go):
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
# buf (https://buf.build) must be on PATH.

cd bindings/go
make gen     # buf generate (managed mode) -> gen/llmcore/v1/*.pb.go, *_grpc.pb.go
make tidy    # go mod tidy  -> fills go.sum + indirect deps
make build   # go build ./...
```

`make gen` uses **managed mode**, so the `.proto` files are not edited; the Go
import path is injected as `<module>/gen/...`.

## Quick start

```go
package main

import (
	"context"
	"fmt"
	"log"

	llmcore "github.com/araray/llmcore-go"
	llmcorev1 "github.com/araray/llmcore-go/gen/llmcore/v1"
)

func main() {
	c, err := llmcore.Dial("127.0.0.1:50151") // insecure (dev); see TLS below
	if err != nil { log.Fatal(err) }
	defer c.Close()

	ctx := context.Background()
	if _, err := c.EnsureCompatible(ctx, "tier0"); err != nil { log.Fatal(err) }

	res, err := c.Chat(ctx, &llmcorev1.ChatRequest{Message: "hello"})
	if err != nil { log.Fatal(err) }
	fmt.Println(res.GetText(), res.GetUsage().GetTotalTokens())

	stream, _ := c.ChatStream(ctx, &llmcorev1.ChatRequest{Message: "stream me"})
	for {
		chunk, err := stream.Recv()
		if err != nil { break } // io.EOF at clean end
		if !chunk.GetDone() { fmt.Print(chunk.GetText()) }
	}
}
```

See **USAGE.md** for the full API, TLS/mTLS, error handling, and cancellation,
and `examples/quickstart` for a runnable program.

## Surface (Tier 0)

`Chat`, `ChatStream` (cancellable), `CountTokens`, `EstimateCost`,
`ListProviders`, `ListModels`, `GetProviderDetails`, `GetInfo` /
`EnsureCompatible`, `Health`, `ReloadConfig`, `Close`. `Embed(...)` returns a
`*BridgeError` (UNSUPPORTED) — Embed is UNIMPLEMENTED in `llmcore.v1`.

## Surface (Tier 2 — audio)

Available when the bridge advertises `tier2.audio`. One-shot: `Synthesize`,
`Transcribe`, `GenerateImage`, `OCR`, `AnalyzeText`. Live duplex (each a
cancellable `Send`/`Recv`/`CloseSend` stream, `io.EOF` at the clean end):
`TranscribeStream`, `SynthesizeStream`, `VoiceAgent`.

```go
stt, _ := c.TranscribeStream(ctx)
_ = stt.Send(&llmcorev1.AudioIn{Frame: &llmcorev1.AudioIn_Audio{Audio: pcm}})
_ = stt.Send(&llmcorev1.AudioIn{Frame: &llmcorev1.AudioIn_Control{Control: llmcorev1.SttControl_STT_CONTROL_CLOSE}})
_ = stt.CloseSend()
for {
	ev, err := stt.Recv()
	if err == io.EOF { break }
	if ev.GetType() == llmcorev1.StreamEventType_STREAM_EVENT_TYPE_FINAL { fmt.Println(ev.GetText()) }
}

sp, _ := c.Synthesize(ctx, &llmcorev1.SynthesizeRequest{Text: "hello"})
_ = sp.GetAudioData() // []byte
```

## Errors

Every method returns `*BridgeError` on failure, decoded from the gRPC binary
trailing metadata `llmcore-error-bin`:

```go
var be *llmcore.BridgeError
if errors.As(err, &be) {
	// be.Category, be.Code, be.HTTPStatus, be.Retryable, be.RetryAfterMs, be.Provider, be.Model, be.GRPCCode
}
```

## Test

```bash
export LLMCORE_BRIDGE_PYTHON=/path/to/venv/bin/python   # python with llmcore[bridge]
make test
```

`harness_test.go` spawns a real `llmcore-bridge` (FakeFacade) over localhost and
the suite drives the client end to end (chat / streaming / count / cost /
catalog / control, capability negotiation, Embed UNSUPPORTED, structured error
decode incl. `RetryAfterMs`, and cancellation; plus the **Tier-2 audio** surface
— 5 unary RPCs + 3 live duplex RPCs — on a separate audio-enabled bridge via
`startBridgeAudio`). The assertions match the same fixtures proven by the
TypeScript client. *(Unix-only harness: it uses SIGTERM.)*

## Distribution

Importable Go module; tag releases (e.g. `bindings/go/v0.1.0` or a dedicated
repo) per the per-language distribution plan. Rename the module by editing
`go.mod`, `go_package_prefix` in `buf.gen.go.yaml`, and the imports, then
`make gen tidy`.
