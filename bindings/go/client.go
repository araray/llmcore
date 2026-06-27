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
	audio     llmcorev1.AudioServiceClient
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
		audio:     llmcorev1.NewAudioServiceClient(conn),
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

// ---- audio: one-shot (Tier 2) ---------------------------------------- //

// Synthesize performs one-shot text-to-speech (generate_speech).
func (c *Client) Synthesize(ctx context.Context, req *llmcorev1.SynthesizeRequest) (*llmcorev1.SpeechResult, error) {
	var tr metadata.MD
	res, err := c.audio.Synthesize(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// Transcribe performs one-shot speech-to-text (transcribe_audio).
func (c *Client) Transcribe(ctx context.Context, req *llmcorev1.TranscribeRequest) (*llmcorev1.TranscriptionResult, error) {
	var tr metadata.MD
	res, err := c.audio.Transcribe(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// GenerateImage performs one-shot image generation (generate_image).
func (c *Client) GenerateImage(ctx context.Context, req *llmcorev1.GenerateImageRequest) (*llmcorev1.ImageGenerationResult, error) {
	var tr metadata.MD
	res, err := c.audio.GenerateImage(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// OCR performs one-shot document OCR (ocr). The request carries either a URL or
// raw bytes in its source oneof.
func (c *Client) OCR(ctx context.Context, req *llmcorev1.OcrRequest) (*llmcorev1.OCRResult, error) {
	var tr metadata.MD
	res, err := c.audio.Ocr(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// AnalyzeText performs one-shot text analysis (analyze_text). The request's
// features Struct selects the analysis flags (summarize / topics / sentiment /
// intents / language).
func (c *Client) AnalyzeText(ctx context.Context, req *llmcorev1.AnalyzeTextRequest) (*llmcorev1.TextAnalysisResult, error) {
	var tr metadata.MD
	res, err := c.audio.AnalyzeText(ctx, req, grpc.Trailer(&tr))
	if err != nil {
		return nil, decode(err, tr)
	}
	return res, nil
}

// ---- audio: live duplex (Tier 2) ------------------------------------- //
//
// The structural stream interfaces below decouple the wrappers from the
// protoc-gen-go-grpc bidi stream type, whose concrete name differs across
// generator versions (a named interface in <v1.5, grpc.BidiStreamingClient[Req,
// Res] in >=v1.5). Both satisfy these.

type transcribeStreamIO interface {
	Send(*llmcorev1.AudioIn) error
	Recv() (*llmcorev1.TranscriptionStreamEvent, error)
	CloseSend() error
	Trailer() metadata.MD
}

// TranscribeStream is a cancellable bidi STT stream. Send AudioIn frames and
// Recv TranscriptionStreamEvents (io.EOF at the clean end).
type TranscribeStream struct {
	stream transcribeStreamIO
	cancel context.CancelFunc
}

// Send transmits one AudioIn frame (open / audio / control).
func (s *TranscribeStream) Send(m *llmcorev1.AudioIn) error { return s.stream.Send(m) }

// CloseSend half-closes the request stream (no more frames will be sent).
func (s *TranscribeStream) CloseSend() error { return s.stream.CloseSend() }

// Recv returns the next event, io.EOF at the clean end, or a *BridgeError.
func (s *TranscribeStream) Recv() (*llmcorev1.TranscriptionStreamEvent, error) {
	ev, err := s.stream.Recv()
	if err == io.EOF {
		return nil, io.EOF
	}
	if err != nil {
		return nil, decode(err, s.stream.Trailer())
	}
	return ev, nil
}

// Close cancels the stream (maps to gRPC CANCELLED). Safe to call once.
func (s *TranscribeStream) Close() { s.cancel() }

// TranscribeStream opens a bidi speech-to-text stream. Cancel via the returned
// stream's Close(), or by cancelling ctx.
func (c *Client) TranscribeStream(ctx context.Context) (*TranscribeStream, error) {
	ctx, cancel := context.WithCancel(ctx)
	stream, err := c.audio.TranscribeStream(ctx)
	if err != nil {
		cancel()
		return nil, decode(err, nil)
	}
	return &TranscribeStream{stream: stream, cancel: cancel}, nil
}

type synthesizeStreamIO interface {
	Send(*llmcorev1.SynthControl) error
	Recv() (*llmcorev1.AudioOut, error)
	CloseSend() error
	Trailer() metadata.MD
}

// SynthesizeStream is a cancellable bidi TTS stream. Send SynthControl frames
// and Recv AudioOut chunks (io.EOF at the clean end).
type SynthesizeStream struct {
	stream synthesizeStreamIO
	cancel context.CancelFunc
}

// Send transmits one SynthControl frame (open / text / control).
func (s *SynthesizeStream) Send(m *llmcorev1.SynthControl) error { return s.stream.Send(m) }

// CloseSend half-closes the request stream.
func (s *SynthesizeStream) CloseSend() error { return s.stream.CloseSend() }

// Recv returns the next AudioOut chunk, io.EOF at the clean end, or a
// *BridgeError.
func (s *SynthesizeStream) Recv() (*llmcorev1.AudioOut, error) {
	out, err := s.stream.Recv()
	if err == io.EOF {
		return nil, io.EOF
	}
	if err != nil {
		return nil, decode(err, s.stream.Trailer())
	}
	return out, nil
}

// Close cancels the stream.
func (s *SynthesizeStream) Close() { s.cancel() }

// SynthesizeStream opens a bidi text-to-speech stream.
func (c *Client) SynthesizeStream(ctx context.Context) (*SynthesizeStream, error) {
	ctx, cancel := context.WithCancel(ctx)
	stream, err := c.audio.SynthesizeStream(ctx)
	if err != nil {
		cancel()
		return nil, decode(err, nil)
	}
	return &SynthesizeStream{stream: stream, cancel: cancel}, nil
}

type voiceAgentStreamIO interface {
	Send(*llmcorev1.VoiceAgentClientEvent) error
	Recv() (*llmcorev1.VoiceAgentEvent, error)
	CloseSend() error
	Trailer() metadata.MD
}

// VoiceAgentStream is a cancellable bidi voice-agent stream. Send
// VoiceAgentClientEvents and Recv VoiceAgentEvents (io.EOF at the clean end).
type VoiceAgentStream struct {
	stream voiceAgentStreamIO
	cancel context.CancelFunc
}

// Send transmits one VoiceAgentClientEvent (settings / audio / inject / ...).
func (s *VoiceAgentStream) Send(m *llmcorev1.VoiceAgentClientEvent) error { return s.stream.Send(m) }

// CloseSend half-closes the request stream.
func (s *VoiceAgentStream) CloseSend() error { return s.stream.CloseSend() }

// Recv returns the next VoiceAgentEvent, io.EOF at the clean end, or a
// *BridgeError.
func (s *VoiceAgentStream) Recv() (*llmcorev1.VoiceAgentEvent, error) {
	ev, err := s.stream.Recv()
	if err == io.EOF {
		return nil, io.EOF
	}
	if err != nil {
		return nil, decode(err, s.stream.Trailer())
	}
	return ev, nil
}

// Close cancels the stream.
func (s *VoiceAgentStream) Close() { s.cancel() }

// VoiceAgent opens a bidi voice-agent stream. The leading frame may carry a
// settings Struct (e.g. provider_name); otherwise the default provider is used.
func (c *Client) VoiceAgent(ctx context.Context) (*VoiceAgentStream, error) {
	ctx, cancel := context.WithCancel(ctx)
	stream, err := c.audio.VoiceAgent(ctx)
	if err != nil {
		cancel()
		return nil, decode(err, nil)
	}
	return &VoiceAgentStream{stream: stream, cancel: cancel}, nil
}
