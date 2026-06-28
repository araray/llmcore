#!/usr/bin/env sh
# Generate Python gRPC stubs for the bridge from the llmcore.v1 contract.
#
# Output: src/llmcore/bridge/_generated/llmcore/v1/*_pb2.py(+_grpc/+pyi)
# Cross-file imports are rewritten from `llmcore.v1` to the fully-qualified
# `llmcore.bridge._generated.llmcore.v1` to avoid colliding with the real
# top-level `llmcore` package. (B2 uses `buf generate` for the other languages.)
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROTO_DIR="$ROOT/bindings/proto"
OUT="$ROOT/src/llmcore/bridge/_generated"
mkdir -p "$OUT"
python -m grpc_tools.protoc \
  -I"$PROTO_DIR" \
  --python_out="$OUT" \
  --grpc_python_out="$OUT" \
  --pyi_out="$OUT" \
  "$PROTO_DIR"/llmcore/v1/common.proto \
  "$PROTO_DIR"/llmcore/v1/errors.proto \
  "$PROTO_DIR"/llmcore/v1/inference.proto \
  "$PROTO_DIR"/llmcore/v1/catalog.proto \
  "$PROTO_DIR"/llmcore/v1/control.proto \
  "$PROTO_DIR"/llmcore/v1/audio.proto \
  "$PROTO_DIR"/llmcore/v1/sessions.proto \
  "$PROTO_DIR"/llmcore/v1/vector.proto

# Package markers
touch "$OUT/__init__.py" "$OUT/llmcore/__init__.py" "$OUT/llmcore/v1/__init__.py"

# Rewrite absolute proto-package imports to the nested bridge package path.
# Matches both `from llmcore.v1 import x_pb2` and `from llmcore.v1.x_pb2 import`.
for f in "$OUT"/llmcore/v1/*.py; do
  sed -i \
    -e 's/^from llmcore\.v1 import /from llmcore.bridge._generated.llmcore.v1 import /' \
    -e 's/^from llmcore\.v1\./from llmcore.bridge._generated.llmcore.v1./' \
    "$f"
done
echo "generated stubs into $OUT"
