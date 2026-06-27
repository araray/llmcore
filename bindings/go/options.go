package llmcore

import (
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
)

// config holds dial-time settings assembled from Options.
type config struct {
	creds    credentials.TransportCredentials
	dialOpts []grpc.DialOption
}

// Option configures the client at Dial time.
type Option func(*config)

// WithTransportCredentials sets TLS/mTLS credentials. Defaults to insecure
// (suitable for localhost/dev only).
func WithTransportCredentials(c credentials.TransportCredentials) Option {
	return func(cfg *config) { cfg.creds = c }
}

// WithDialOption appends a raw grpc.DialOption for advanced tuning
// (keepalives, interceptors, message-size limits, ...).
func WithDialOption(o grpc.DialOption) Option {
	return func(cfg *config) { cfg.dialOpts = append(cfg.dialOpts, o) }
}

func defaultConfig() *config {
	return &config{creds: insecure.NewCredentials()}
}
