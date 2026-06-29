package llmcore_test

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"
	"time"

	llmcore "github.com/araray/llmcore-go"
	llmcorev1 "github.com/araray/llmcore-go/gen/llmcore/v1"
)

func testCtx() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), 10*time.Second)
}

// dialClient spins up a bridge + connected client; the returned cleanup stops both.
func dialClient(t *testing.T) (*llmcore.Client, func()) {
	t.Helper()
	b := startBridge(t)
	c, err := llmcore.Dial(b.grpcAddr)
	if err != nil {
		b.stop()
		t.Fatalf("dial: %v", err)
	}
	return c, func() { _ = c.Close(); b.stop() }
}

func TestEnsureCompatible(t *testing.T) {
	c, cleanup := dialClient(t)
	defer cleanup()
	ctx, cancel := testCtx()
	defer cancel()

	info, err := c.EnsureCompatible(ctx, "tier0")
	if err != nil {
		t.Fatalf("EnsureCompatible: %v", err)
	}
	if info.GetContractVersion() != "llmcore.v1" {
		t.Fatalf("contract=%q", info.GetContractVersion())
	}

	if _, err := c.EnsureCompatible(ctx, "tier2.audio"); err == nil {
		t.Fatal("expected missing-capability error")
	} else {
		var be *llmcore.BridgeError
		if !errors.As(err, &be) || be.Code != "capability.missing" {
			t.Fatalf("want capability.missing, got %v", err)
		}
	}
}

func TestChatAndUsage(t *testing.T) {
	c, cleanup := dialClient(t)
	defer cleanup()
	ctx, cancel := testCtx()
	defer cancel()

	res, err := c.Chat(ctx, &llmcorev1.ChatRequest{Message: "hello world"})
	if err != nil {
		t.Fatal(err)
	}
	if res.GetText() != "echo: hello world" {
		t.Fatalf("text=%q", res.GetText())
	}
	if res.GetUsage().GetPromptTokens() != 2 {
		t.Fatalf("promptTokens=%d", res.GetUsage().GetPromptTokens())
	}
}

func TestChatStreamConcatenation(t *testing.T) {
	c, cleanup := dialClient(t)
	defer cleanup()
	ctx, cancel := testCtx()
	defer cancel()

	stream, err := c.ChatStream(ctx, &llmcorev1.ChatRequest{Message: "stream this please"})
	if err != nil {
		t.Fatal(err)
	}
	var parts []string
	done := false
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if chunk.GetDone() {
			done = true
		} else {
			parts = append(parts, chunk.GetText())
		}
	}
	if !done {
		t.Fatal("no terminal done chunk")
	}
	if got := strings.Join(parts, ""); got != "echo: stream this please" {
		t.Fatalf("concatenated=%q", got)
	}
}

func TestCountAndCost(t *testing.T) {
	c, cleanup := dialClient(t)
	defer cleanup()
	ctx, cancel := testCtx()
	defer cancel()

	ct, err := c.CountTokens(ctx, &llmcorev1.CountTokensRequest{Text: "one two three four"})
	if err != nil {
		t.Fatal(err)
	}
	if ct.GetTokens() != 4 {
		t.Fatalf("tokens=%d", ct.GetTokens())
	}

	ce, err := c.EstimateCost(ctx, &llmcorev1.EstimateCostRequest{
		ProviderName:     "fake",
		ModelName:        "fake-1",
		PromptTokens:     1_000_000,
		CompletionTokens: 1_000_000,
	})
	if err != nil {
		t.Fatal(err)
	}
	if ce.GetCurrency() != "USD" || ce.GetTotalCost() != 3 {
		t.Fatalf("cost currency=%q total=%v", ce.GetCurrency(), ce.GetTotalCost())
	}
}

func TestCatalog(t *testing.T) {
	c, cleanup := dialClient(t)
	defer cleanup()
	ctx, cancel := testCtx()
	defer cancel()

	lp, err := c.ListProviders(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(lp.GetProviders(), ","); got != "fake" {
		t.Fatalf("providers=%q", got)
	}
	lm, err := c.ListModels(ctx, "fake")
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(lm.GetModels(), ","); got != "fake-1,fake-2" {
		t.Fatalf("models=%q", got)
	}
	d, err := c.GetProviderDetails(ctx, "fake")
	if err != nil {
		t.Fatal(err)
	}
	if d.GetId() != "fake-1" || d.GetContextLength() != 8192 {
		t.Fatalf("details id=%q ctx=%d", d.GetId(), d.GetContextLength())
	}
}

func TestEmbedUnsupported(t *testing.T) {
	c, cleanup := dialClient(t)
	defer cleanup()
	ctx, cancel := testCtx()
	defer cancel()

	_, err := c.Embed(ctx, &llmcorev1.EmbedRequest{Input: []string{"x"}})
	var be *llmcore.BridgeError
	if !errors.As(err, &be) || be.Category != "ERROR_CATEGORY_UNSUPPORTED" {
		t.Fatalf("want UNSUPPORTED, got %v", err)
	}
}

func TestProviderRateLimitDecode(t *testing.T) {
	c, cleanup := dialClient(t)
	defer cleanup()
	ctx, cancel := testCtx()
	defer cancel()

	_, err := c.Chat(ctx, &llmcorev1.ChatRequest{Message: "__error__:provider_rate_limited"})
	var be *llmcore.BridgeError
	if !errors.As(err, &be) {
		t.Fatalf("want *BridgeError, got %v", err)
	}
	if be.Category != "ERROR_CATEGORY_PROVIDER" || be.Code != "provider.rate_limited" {
		t.Fatalf("category=%q code=%q", be.Category, be.Code)
	}
	if be.HTTPStatus != 429 || !be.Retryable || be.RetryAfterMs != 2000 || be.Provider != "fake" {
		t.Fatalf("fields=%+v", be)
	}
}

func TestStreamCancel(t *testing.T) {
	c, cleanup := dialClient(t)
	defer cleanup()
	ctx, cancel := testCtx()
	defer cancel()

	stream, err := c.ChatStream(ctx, &llmcorev1.ChatRequest{Message: "__cancel__"})
	if err != nil {
		t.Fatal(err)
	}
	first, err := stream.Recv()
	if err != nil {
		t.Fatalf("first recv: %v", err)
	}
	if first.GetText() == "" {
		t.Fatal("expected non-empty first chunk")
	}
	stream.Close()
	// After cancel, the stream terminates (CANCELLED) within a bounded number of reads.
	for i := 0; i < 1000; i++ {
		if _, err := stream.Recv(); err != nil {
			return
		}
	}
	t.Fatal("stream did not terminate after cancel")
}
