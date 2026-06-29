#!/usr/bin/env sh
# Regenerate Go stubs from bindings/proto via buf + the Go protoc plugins.
set -eu
GO_DIR="$(cd "$(dirname "$0")/.." && pwd)"          # bindings/go
BINDINGS_DIR="$(cd "$GO_DIR/.." && pwd)"            # bindings
command -v protoc-gen-go >/dev/null 2>&1 || {
  echo "missing protoc-gen-go. Install: go install google.golang.org/protobuf/cmd/protoc-gen-go@latest" >&2; exit 1; }
command -v protoc-gen-go-grpc >/dev/null 2>&1 || {
  echo "missing protoc-gen-go-grpc. Install: go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest" >&2; exit 1; }
rm -rf "$GO_DIR/gen"; mkdir -p "$GO_DIR/gen"
( cd "$BINDINGS_DIR" && buf generate --template "$GO_DIR/buf.gen.go.yaml" --output "$GO_DIR" )
echo "generated Go stubs into $GO_DIR/gen"
