package llmcore

// Tier-1 surface: sessions, context items, vector store / RAG, and context
// presets. Each method is a thin wrapper over the generated service client,
// mirroring the unary pattern in client.go (trailer-aware error decoding).

import (
	"context"

	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"

	llmcorev1 "github.com/araray/llmcore-go/gen/llmcore/v1"
)

// ---- sessions (Tier 1) ------------------------------------------------- //

// CreateSession creates a new conversation session.
func (c *Client) CreateSession(ctx context.Context, req *llmcorev1.CreateSessionRequest) (*llmcorev1.ChatSession, error) {
	var tr metadata.MD
	res, err := c.session.CreateSession(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// GetSession fetches a session by id.
func (c *Client) GetSession(ctx context.Context, req *llmcorev1.GetSessionRequest) (*llmcorev1.ChatSession, error) {
	var tr metadata.MD
	res, err := c.session.GetSession(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// ListSessions lists sessions (optionally limited).
func (c *Client) ListSessions(ctx context.Context, req *llmcorev1.ListSessionsRequest) (*llmcorev1.ListSessionsResponse, error) {
	var tr metadata.MD
	res, err := c.session.ListSessions(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// DeleteSession deletes a session by id.
func (c *Client) DeleteSession(ctx context.Context, req *llmcorev1.DeleteSessionRequest) (*llmcorev1.Empty, error) {
	var tr metadata.MD
	res, err := c.session.DeleteSession(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// UpdateSessionName renames a session.
func (c *Client) UpdateSessionName(ctx context.Context, req *llmcorev1.UpdateSessionNameRequest) (*llmcorev1.Empty, error) {
	var tr metadata.MD
	res, err := c.session.UpdateSessionName(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// ForkSession forks a session, returning the new session id.
func (c *Client) ForkSession(ctx context.Context, req *llmcorev1.ForkSessionRequest) (*llmcorev1.ForkSessionResponse, error) {
	var tr metadata.MD
	res, err := c.session.ForkSession(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// CloneSession clones a session, returning the new session id.
func (c *Client) CloneSession(ctx context.Context, req *llmcorev1.CloneSessionRequest) (*llmcorev1.CloneSessionResponse, error) {
	var tr metadata.MD
	res, err := c.session.CloneSession(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// DeleteMessages removes messages from a session, returning the deleted count.
func (c *Client) DeleteMessages(ctx context.Context, req *llmcorev1.DeleteMessagesRequest) (*llmcorev1.DeleteMessagesResponse, error) {
	var tr metadata.MD
	res, err := c.session.DeleteMessages(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// GetMessagesByRange returns messages in the inclusive [start,end] index window.
func (c *Client) GetMessagesByRange(ctx context.Context, req *llmcorev1.GetMessagesByRangeRequest) (*llmcorev1.GetMessagesByRangeResponse, error) {
	var tr metadata.MD
	res, err := c.session.GetMessagesByRange(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// AddContextItem appends a context item, returning its id.
func (c *Client) AddContextItem(ctx context.Context, req *llmcorev1.AddContextItemRequest) (*llmcorev1.AddContextItemResponse, error) {
	var tr metadata.MD
	res, err := c.session.AddContextItem(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// GetContextItem fetches a context item by id (NOT_FOUND if absent).
func (c *Client) GetContextItem(ctx context.Context, req *llmcorev1.GetContextItemRequest) (*llmcorev1.ContextItem, error) {
	var tr metadata.MD
	res, err := c.session.GetContextItem(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// RemoveContextItem removes a context item by id.
func (c *Client) RemoveContextItem(ctx context.Context, req *llmcorev1.RemoveContextItemRequest) (*llmcorev1.RemoveContextItemResponse, error) {
	var tr metadata.MD
	res, err := c.session.RemoveContextItem(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// ---- vector store & RAG (Tier 1) --------------------------------------- //

// AddDocuments adds documents to a vector-store collection, returning their ids.
func (c *Client) AddDocuments(ctx context.Context, req *llmcorev1.AddDocumentsRequest) (*llmcorev1.AddDocumentsResponse, error) {
	var tr metadata.MD
	res, err := c.vector.AddDocuments(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// SearchVectorStore runs a similarity search.
func (c *Client) SearchVectorStore(ctx context.Context, req *llmcorev1.SearchVectorStoreRequest) (*llmcorev1.SearchVectorStoreResponse, error) {
	var tr metadata.MD
	res, err := c.vector.SearchVectorStore(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// ListVectorCollections lists vector-store collections.
func (c *Client) ListVectorCollections(ctx context.Context) (*llmcorev1.ListCollectionsResponse, error) {
	var tr metadata.MD
	res, err := c.vector.ListVectorCollections(ctx, &llmcorev1.Empty{}, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// ListRagCollections lists RAG collections.
func (c *Client) ListRagCollections(ctx context.Context) (*llmcorev1.ListCollectionsResponse, error) {
	var tr metadata.MD
	res, err := c.vector.ListRagCollections(ctx, &llmcorev1.Empty{}, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// GetRagCollectionInfo returns store-specific metadata (NOT_FOUND if absent).
func (c *Client) GetRagCollectionInfo(ctx context.Context, req *llmcorev1.GetRagCollectionInfoRequest) (*llmcorev1.RagCollectionInfo, error) {
	var tr metadata.MD
	res, err := c.vector.GetRagCollectionInfo(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// DeleteRagCollection deletes a RAG collection.
func (c *Client) DeleteRagCollection(ctx context.Context, req *llmcorev1.DeleteRagCollectionRequest) (*llmcorev1.DeleteRagCollectionResponse, error) {
	var tr metadata.MD
	res, err := c.vector.DeleteRagCollection(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// ---- context presets (Tier 1) ------------------------------------------ //

// SaveContextPreset persists a context preset.
func (c *Client) SaveContextPreset(ctx context.Context, req *llmcorev1.SaveContextPresetRequest) (*llmcorev1.Empty, error) {
	var tr metadata.MD
	res, err := c.preset.SaveContextPreset(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// GetContextPreset fetches a preset by name (NOT_FOUND if absent).
func (c *Client) GetContextPreset(ctx context.Context, req *llmcorev1.GetContextPresetRequest) (*llmcorev1.ContextPreset, error) {
	var tr metadata.MD
	res, err := c.preset.GetContextPreset(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// ListContextPresets lists preset summaries.
func (c *Client) ListContextPresets(ctx context.Context) (*llmcorev1.ListContextPresetsResponse, error) {
	var tr metadata.MD
	res, err := c.preset.ListContextPresets(ctx, &llmcorev1.Empty{}, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// DeleteContextPreset deletes a preset by name.
func (c *Client) DeleteContextPreset(ctx context.Context, req *llmcorev1.DeleteContextPresetRequest) (*llmcorev1.DeleteContextPresetResponse, error) {
	var tr metadata.MD
	res, err := c.preset.DeleteContextPreset(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}
