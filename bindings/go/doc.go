// Package llmcore is a Go client for the llmcore bridge, generated from the
// frozen llmcore.v1 contract. It speaks gRPC (the bridge's primary transport)
// and depends only on the contract, never on Python.
//
// Surface (Tier 0): Chat, ChatStream (cancellable), CountTokens, EstimateCost,
// ListProviders, ListModels, GetProviderDetails, GetInfo/EnsureCompatible,
// Health, ReloadConfig. Embed returns a *BridgeError (UNSUPPORTED) — Embed is
// UNIMPLEMENTED in llmcore.v1. AudioService (Tier 2) is not yet surfaced.
//
// Every failure is a *BridgeError carrying the structured llmcore.v1.LlmcoreError
// decoded from the gRPC binary trailing metadata "llmcore-error-bin".
//
//	c, _ := llmcore.Dial("127.0.0.1:50151")
//	defer c.Close()
//	ctx := context.Background()
//	if _, err := c.EnsureCompatible(ctx, "tier0"); err != nil { log.Fatal(err) }
//	res, _ := c.Chat(ctx, &llmcorev1.ChatRequest{Message: "hello"})
//	fmt.Println(res.GetText())
package llmcore
