// Command sessions demonstrates the Tier-1 surface (sessions, context items,
// vector store, and context presets) of the llmcore bridge over gRPC.
//
// Start a bridge with the Tier-1 fake stores enabled, then run this example:
//
//	LLMCORE_BRIDGE_FAKE=1 LLMCORE_BRIDGE_FAKE_SESSIONS=1 LLMCORE_BRIDGE_FAKE_VECTOR=1 \
//	  python -m llmcore.bridge.cli serve --transport grpc \
//	  --grpc-address 127.0.0.1:50151 --insecure
//	go run ./examples/sessions   # or set LLMCORE_GRPC=host:port
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	llmcore "github.com/araray/llmcore-go"
	llmcorev1 "github.com/araray/llmcore-go/gen/llmcore/v1"
	"google.golang.org/protobuf/types/known/structpb"
)

func main() {
	target := os.Getenv("LLMCORE_GRPC")
	if target == "" {
		target = "127.0.0.1:50151"
	}
	c, err := llmcore.Dial(target)
	if err != nil {
		log.Fatal(err)
	}
	defer c.Close()
	ctx := context.Background()

	// Negotiate: require the Tier-1 sessions capability.
	info, err := c.GetInfo(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("contract=%s tiers=%v\n", info.GetContractVersion(), info.GetTiers())

	// ---- sessions + context items ----
	name := "demo-session"
	sys := "You are a terse assistant."
	session, err := c.CreateSession(ctx, &llmcorev1.CreateSessionRequest{
		Name:          &name,
		SystemMessage: &sys,
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("created session %s (%d message[s])\n", session.GetId(), len(session.GetMessages()))

	itemType := "user_text"
	added, err := c.AddContextItem(ctx, &llmcorev1.AddContextItemRequest{
		SessionId: session.GetId(),
		Content:   "Remember: the launch date is June 30.",
		Type:      &itemType,
	})
	if err != nil {
		log.Fatal(err)
	}
	item, err := c.GetContextItem(ctx, &llmcorev1.GetContextItemRequest{
		SessionId: session.GetId(),
		ItemId:    added.GetItemId(),
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("context item %s [%s]: %q (%d tokens)\n",
		item.GetId(), item.GetType(), item.GetContent(), item.GetTokens())

	// ---- vector store ----
	doc, _ := structpb.NewStruct(map[string]any{
		"content":  "Paris is the capital of France.",
		"metadata": map[string]any{"topic": "geography"},
	})
	ids, err := c.AddDocuments(ctx, &llmcorev1.AddDocumentsRequest{
		Documents: []*structpb.Struct{doc},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("indexed %d document[s]\n", len(ids.GetIds()))

	hits, err := c.SearchVectorStore(ctx, &llmcorev1.SearchVectorStoreRequest{
		Query: "capital of France",
		K:     3,
	})
	if err != nil {
		log.Fatal(err)
	}
	for _, d := range hits.GetDocuments() {
		fmt.Printf("  hit (score=%.3f): %q\n", d.GetScore(), d.GetContent())
	}

	// ---- context presets ----
	desc := "Standard preamble"
	if _, err := c.SaveContextPreset(ctx, &llmcorev1.SaveContextPresetRequest{
		Preset: &llmcorev1.ContextPreset{
			Name:        "preamble",
			Description: &desc,
			Items: []*llmcorev1.ContextPresetItem{
				{Type: "preset_text_content", Content: strptr("Always cite sources.")},
			},
		},
	}); err != nil {
		log.Fatal(err)
	}
	preset, err := c.GetContextPreset(ctx, &llmcorev1.GetContextPresetRequest{PresetName: "preamble"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("preset %q has %d item[s]\n", preset.GetName(), len(preset.GetItems()))

	// ---- cleanup ----
	if _, err := c.DeleteSession(ctx, &llmcorev1.DeleteSessionRequest{SessionId: session.GetId()}); err != nil {
		log.Fatal(err)
	}
	fmt.Println("done.")
}

func strptr(s string) *string { return &s }
