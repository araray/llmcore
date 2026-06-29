package llmcore

import (
	"fmt"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	llmcorev1 "github.com/araray/llmcore-go/gen/llmcore/v1"
)

// ErrorMetadataKey is the binary trailing-metadata key under which the bridge
// returns the structured llmcore.v1.LlmcoreError.
const ErrorMetadataKey = "llmcore-error-bin"

// BridgeError is the normalized, transport-neutral error surfaced by all client
// methods. It mirrors the fields of llmcore.v1.LlmcoreError.
type BridgeError struct {
	// Category is the ErrorCategory enum name, e.g. "ERROR_CATEGORY_PROVIDER".
	Category string
	// Code is a stable dotted token, e.g. "provider.rate_limited".
	Code string
	// Message is secret-redacted human-readable detail.
	Message string
	// HTTPStatus is the upstream HTTP status (0 if unset).
	HTTPStatus int32
	// Retryable indicates the caller may retry.
	Retryable bool
	// RetryAfterMs is the suggested backoff in milliseconds (0 if unset).
	RetryAfterMs float64
	// Provider / Model identify the upstream when applicable.
	Provider string
	Model    string
	// GRPCCode is the gRPC status code (codes.OK if not derived from gRPC).
	GRPCCode codes.Code
}

// Error implements the error interface.
func (e *BridgeError) Error() string {
	if e.Code != "" {
		return fmt.Sprintf("%s: %s", e.Code, e.Message)
	}
	return e.Message
}

func bridgeErrorFromProto(pb *llmcorev1.LlmcoreError, code codes.Code) *BridgeError {
	return &BridgeError{
		Category:     pb.GetCategory().String(),
		Code:         pb.GetCode(),
		Message:      pb.GetMessage(),
		HTTPStatus:   pb.GetHttpStatus(),   // getter returns 0 when unset
		Retryable:    pb.GetRetryable(),
		RetryAfterMs: pb.GetRetryAfterMs(), // getter returns 0 when unset
		Provider:     pb.GetProvider(),
		Model:        pb.GetModel(),
		GRPCCode:     code,
	}
}

// decode converts a gRPC error plus its trailing metadata into a *BridgeError.
// It first tries to decode the structured payload from ErrorMetadataKey, then
// falls back to synthesizing one from the gRPC status. A nil tr is safe.
func decode(err error, tr metadata.MD) error {
	if err == nil {
		return nil
	}
	if vals := tr.Get(ErrorMetadataKey); len(vals) > 0 {
		var pb llmcorev1.LlmcoreError
		// gRPC base64-decodes "-bin" metadata on receipt, so vals[0] is raw bytes.
		if e := proto.Unmarshal([]byte(vals[0]), &pb); e == nil {
			return bridgeErrorFromProto(&pb, status.Code(err))
		}
	}
	st, _ := status.FromError(err)
	return &BridgeError{
		Category: llmcorev1.ErrorCategory_ERROR_CATEGORY_INTERNAL.String(),
		Code:     "grpc." + st.Code().String(),
		Message:  st.Message(),
		GRPCCode: st.Code(),
	}
}
