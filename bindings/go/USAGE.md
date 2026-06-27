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
