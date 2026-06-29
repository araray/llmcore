#!/usr/bin/env sh
# Regenerate the TypeScript stubs from bindings/proto via buf + ts-proto.
set -eu
TS_DIR="$(cd "$(dirname "$0")/.." && pwd)"        # bindings/typescript
BINDINGS_DIR="$(cd "$TS_DIR/.." && pwd)"           # bindings
rm -rf "$TS_DIR/src/gen"
mkdir -p "$TS_DIR/src/gen"
# buf uses bindings/buf.yaml (module path: proto). Local plugin resolved from node_modules/.bin.
( cd "$BINDINGS_DIR" \
  && PATH="$TS_DIR/node_modules/.bin:$PATH" \
     buf generate --template "$TS_DIR/buf.gen.ts.yaml" --output "$TS_DIR" )
echo "generated TS stubs into $TS_DIR/src/gen"
