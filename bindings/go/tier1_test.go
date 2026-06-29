package llmcore_test

import (
	"context"
	"testing"

	llmcore "github.com/araray/llmcore-go"
	llmcorev1 "github.com/araray/llmcore-go/gen/llmcore/v1"
	"google.golang.org/protobuf/types/known/structpb"
)

// dialSessionsClient spawns a bridge with the Tier-1 fake stores enabled and
// returns a connected client plus a cleanup func.
func dialSessionsClient(t *testing.T) (*llmcore.Client, func()) {
	t.Helper()
	b := startBridgeEnv(t, "LLMCORE_BRIDGE_FAKE_SESSIONS=1", "LLMCORE_BRIDGE_FAKE_VECTOR=1")
	c, err := llmcore.Dial(b.grpcAddr)
	if err != nil {
		b.stop()
		t.Fatalf("dial: %v", err)
	}
	return c, func() { _ = c.Close(); b.stop() }
}

func TestTier1CapabilitiesAdvertised(t *testing.T) {
	c, cleanup := dialSessionsClient(t)
	defer cleanup()
	info, err := c.GetInfo(context.Background())
	if err != nil {
		t.Fatalf("GetInfo: %v", err)
	}
	caps := map[string]bool{}
	for _, s := range info.Capabilities {
		caps[s] = true
	}
	if !caps["tier1.sessions"] || !caps["tier1.vector"] {
		t.Fatalf("missing tier1 capabilities: %v", info.Capabilities)
	}
}

func TestSessionCreateGetRoundtrip(t *testing.T) {
	c, cleanup := dialSessionsClient(t)
	defer cleanup()
	ctx := context.Background()
	name := "go-chat"
	sys := "be brief"
	created, err := c.CreateSession(ctx, &llmcorev1.CreateSessionRequest{
		Name:          &name,
		SystemMessage: &sys,
	})
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}
	if created.GetName() != "go-chat" || len(created.Messages) != 1 {
		t.Fatalf("unexpected session: %+v", created)
	}
	got, err := c.GetSession(ctx, &llmcorev1.GetSessionRequest{SessionId: created.Id})
	if err != nil {
		t.Fatalf("GetSession: %v", err)
	}
	if got.Id != created.Id {
		t.Fatalf("id mismatch: %s != %s", got.Id, created.Id)
	}
}

func TestSessionContextItemLifecycle(t *testing.T) {
	c, cleanup := dialSessionsClient(t)
	defer cleanup()
	ctx := context.Background()
	s, err := c.CreateSession(ctx, &llmcorev1.CreateSessionRequest{})
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}
	typ := "rag_snippet"
	added, err := c.AddContextItem(ctx, &llmcorev1.AddContextItemRequest{
		SessionId: s.Id,
		Content:   "a fact",
		Type:      &typ,
	})
	if err != nil {
		t.Fatalf("AddContextItem: %v", err)
	}
	item, err := c.GetContextItem(ctx, &llmcorev1.GetContextItemRequest{
		SessionId: s.Id,
		ItemId:    added.ItemId,
	})
	if err != nil {
		t.Fatalf("GetContextItem: %v", err)
	}
	if item.Type != "rag_snippet" || item.Content != "a fact" {
		t.Fatalf("unexpected item: %+v", item)
	}
	removed, err := c.RemoveContextItem(ctx, &llmcorev1.RemoveContextItemRequest{
		SessionId: s.Id,
		ItemId:    added.ItemId,
	})
	if err != nil {
		t.Fatalf("RemoveContextItem: %v", err)
	}
	if !removed.Removed {
		t.Fatalf("expected removed=true")
	}
}

func TestSessionGetMissingIsNotFound(t *testing.T) {
	c, cleanup := dialSessionsClient(t)
	defer cleanup()
	_, err := c.GetSession(context.Background(), &llmcorev1.GetSessionRequest{SessionId: "ghost"})
	if err == nil {
		t.Fatal("expected error")
	}
	be, ok := err.(*llmcore.BridgeError)
	if !ok {
		t.Fatalf("expected *llmcore.BridgeError, got %T", err)
	}
	if be.Category != "ERROR_CATEGORY_NOT_FOUND" {
		t.Fatalf("expected NOT_FOUND, got %s", be.Category)
	}
}

func TestVectorAddAndSearch(t *testing.T) {
	c, cleanup := dialSessionsClient(t)
	defer cleanup()
	ctx := context.Background()
	doc, err := structpb.NewStruct(map[string]any{"content": "the cat sat"})
	if err != nil {
		t.Fatalf("NewStruct: %v", err)
	}
	added, err := c.AddDocuments(ctx, &llmcorev1.AddDocumentsRequest{
		Documents: []*structpb.Struct{doc},
	})
	if err != nil {
		t.Fatalf("AddDocuments: %v", err)
	}
	if len(added.Ids) != 1 {
		t.Fatalf("expected 1 id, got %d", len(added.Ids))
	}
	res, err := c.SearchVectorStore(ctx, &llmcorev1.SearchVectorStoreRequest{Query: "cat"})
	if err != nil {
		t.Fatalf("SearchVectorStore: %v", err)
	}
	if len(res.Documents) != 1 || res.Documents[0].Content != "the cat sat" {
		t.Fatalf("unexpected search result: %+v", res.Documents)
	}
}

func TestPresetSaveGetRoundtrip(t *testing.T) {
	c, cleanup := dialSessionsClient(t)
	defer cleanup()
	ctx := context.Background()
	desc := "d"
	content := "boilerplate"
	preset := &llmcorev1.ContextPreset{
		Name:        "go-preset",
		Description: &desc,
		Items: []*llmcorev1.ContextPresetItem{
			{Type: "preset_text_content", Content: &content},
		},
	}
	if _, err := c.SaveContextPreset(ctx, &llmcorev1.SaveContextPresetRequest{Preset: preset}); err != nil {
		t.Fatalf("SaveContextPreset: %v", err)
	}
	got, err := c.GetContextPreset(ctx, &llmcorev1.GetContextPresetRequest{PresetName: "go-preset"})
	if err != nil {
		t.Fatalf("GetContextPreset: %v", err)
	}
	if got.Name != "go-preset" || len(got.Items) != 1 || got.Items[0].GetContent() != "boilerplate" {
		t.Fatalf("unexpected preset: %+v", got)
	}
	if _, err := c.GetContextPreset(ctx, &llmcorev1.GetContextPresetRequest{PresetName: "ghost"}); err == nil {
		t.Fatal("expected NOT_FOUND for missing preset")
	}
}
