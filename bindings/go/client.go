package llmcore

import (
	"context"
	"fmt"
	"io"

	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"

	llmcorev1 "github.com/araray/llmcore-go/gen/llmcore/v1"
)

// Client wraps the llmcore.v1 service clients over one gRPC connection.
type Client struct {
	conn      *grpc.ClientConn
	inference llmcorev1.InferenceServiceClient
	catalog   llmcorev1.CatalogServiceClient
	control   llmcorev1.ControlServiceClient
}

// Dial connects to a bridge at target ("host:port"). The connection is lazy
// (grpc.NewClient); the first RPC establishes it. Insecure by default — pass
// WithTransportCredentials for TLS/mTLS.
func Dial(target string, opts ...Option) (*Client, error) {
	cfg := defaultConfig()
	for _, o := range opts {
		o(cfg)
	}
	dialOpts := append([]grpc.DialOption{grpc.WithTransportCredentials(cfg.creds)}, cfg.dialOpts...)
	conn, err := grpc.NewClient(target, dialOpts...)
	if err != nil {
		return nil, fmt.Errorf("llmcore: dial %s: %w", target, err)
	}
	return &Client{
		conn:      conn,
		inference: llmcorev1.NewInferenceServiceClient(conn),
		catalog:   llmcorev1.NewCatalogServiceClient(conn),
		control:   llmcorev1.NewControlServiceClient(conn),
	}, nil
}

// Close closes the underlying gRPC connection.
func (c *Client) Close() error { return c.conn.Close() }

// ---- control plane ---------------------------------------------------- //

// GetInfo returns the server handshake payload.
func (c *Client) GetInfo(ctx context.Context) (*llmcorev1.ServerInfo, error) {
	var tr metadata.MD
	res, err := c.control.GetInfo(ctx, &llmcorev1.Empty{}, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// Health returns liveness/readiness.
func (c *Client) Health(ctx context.Context) (*llmcorev1.HealthStatus, error) {
	var tr metadata.MD
	res, err := c.control.Health(ctx, &llmcorev1.Empty{}, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// ReloadConfig asks the bridge to reload its configuration. A nil req is allowed.
func (c *Client) ReloadConfig(ctx context.Context, req *llmcorev1.ReloadConfigRequest) (*llmcorev1.ReloadConfigResponse, error) {
	if req == nil {
		req = &llmcorev1.ReloadConfigRequest{}
	}
	var tr metadata.MD
	res, err := c.control.ReloadConfig(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// EnsureCompatible verifies the contract version and that every required
// capability is advertised. Returns a *BridgeError on mismatch.
func (c *Client) EnsureCompatible(ctx context.Context, requiredCapabilities ...string) (*llmcorev1.ServerInfo, error) {
	info, err := c.GetInfo(ctx)
	if err != nil {
		return nil, err
	}
	if info.GetContractVersion() != "llmcore.v1" {
		return nil, &BridgeError{
			Category: llmcorev1.ErrorCategory_ERROR_CATEGORY_INVALID_ARGUMENT.String(),
			Code:     "contract.mismatch",
			Message:  fmt.Sprintf("server contract %s != llmcore.v1", info.GetContractVersion()),
		}
	}
	have := make(map[string]struct{}, len(info.GetCapabilities()))
	for _, cap := range info.GetCapabilities() {
		have[cap] = struct{}{}
	}
	var missing []string
	for _, want := range requiredCapabilities {
		if _, ok := have[want]; !ok {
			missing = append(missing, want)
		}
	}
	if len(missing) > 0 {
		return nil, &BridgeError{
			Category: llmcorev1.ErrorCategory_ERROR_CATEGORY_UNSUPPORTED.String(),
			Code:     "capability.missing",
			Message:  fmt.Sprintf("server lacks required capabilities: %v", missing),
		}
	}
	return info, nil
}

// ---- inference -------------------------------------------------------- //

// Chat performs a unary chat completion.
func (c *Client) Chat(ctx context.Context, req *llmcorev1.ChatRequest) (*llmcorev1.ChatResponse, error) {
	var tr metadata.MD
	res, err := c.inference.Chat(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// CountTokens returns the token count for text under the given provider/model.
func (c *Client) CountTokens(ctx context.Context, req *llmcorev1.CountTokensRequest) (*llmcorev1.CountTokensResponse, error) {
	var tr metadata.MD
	res, err := c.inference.CountTokens(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// EstimateCost returns a cost estimate for a token budget.
func (c *Client) EstimateCost(ctx context.Context, req *llmcorev1.EstimateCostRequest) (*llmcorev1.CostEstimate, error) {
	var tr metadata.MD
	res, err := c.inference.EstimateCost(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// Embed is UNIMPLEMENTED in llmcore.v1 and always returns a *BridgeError
// (UNSUPPORTED). Provided for surface completeness.
func (c *Client) Embed(ctx context.Context, req *llmcorev1.EmbedRequest) (*llmcorev1.EmbedResponse, error) {
	var tr metadata.MD
	res, err := c.inference.Embed(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// chatStreamRecv decouples ChatStream from the protoc-gen-go-grpc stream type,
// whose concrete name differs across generator versions (a named interface in
// <v1.5, grpc.ServerStreamingClient[T] in >=v1.5). Both satisfy this.
type chatStreamRecv interface {
	Recv() (*llmcorev1.ChatChunk, error)
	Trailer() metadata.MD
}

// ChatStream is a cancellable server stream of ChatChunk frames.
type ChatStream struct {
	stream chatStreamRecv
	cancel context.CancelFunc
}

// Recv returns the next chunk, io.EOF at the clean end of the stream, or a
// *BridgeError on failure.
func (s *ChatStream) Recv() (*llmcorev1.ChatChunk, error) {
	chunk, err := s.stream.Recv()
	if err == io.EOF {
		return nil, io.EOF
	}
	if err != nil {
		return nil, decode(err, s.stream.Trailer())
	}
	return chunk, nil
}

// Close cancels the stream (maps to gRPC CANCELLED). Safe to call once.
func (s *ChatStream) Close() { s.cancel() }

// ChatStream opens a streaming chat completion. Cancel via the returned
// stream's Close(), or by cancelling ctx.
func (c *Client) ChatStream(ctx context.Context, req *llmcorev1.ChatRequest) (*ChatStream, error) {
	ctx, cancel := context.WithCancel(ctx)
	stream, err := c.inference.ChatStream(ctx, req)
	if err != nil {
		cancel()
		return nil, decode(err, nil)
	}
	return &ChatStream{stream: stream, cancel: cancel}, nil
}

// ---- catalog ---------------------------------------------------------- //

// ListProviders lists configured provider names.
func (c *Client) ListProviders(ctx context.Context) (*llmcorev1.ListProvidersResponse, error) {
	var tr metadata.MD
	res, err := c.catalog.ListProviders(ctx, &llmcorev1.Empty{}, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// ListModels lists model ids for a provider.
func (c *Client) ListModels(ctx context.Context, providerName string) (*llmcorev1.ListModelsResponse, error) {
	var tr metadata.MD
	res, err := c.catalog.ListModels(ctx, &llmcorev1.ListModelsRequest{ProviderName: providerName}, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// GetProviderDetails returns ModelDetails for a provider (empty string =
// default/first provider, per the bridge).
func (c *Client) GetProviderDetails(ctx context.Context, providerName string) (*llmcorev1.ModelDetails, error) {
	req := &llmcorev1.GetProviderRequest{}
	if providerName != "" {
		req.ProviderName = &providerName
	}
	var tr metadata.MD
	res, err := c.catalog.GetProviderDetails(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}
