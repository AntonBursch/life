#!/usr/bin/env bash
# Build the flow-wasm crate and run wasm-bindgen to produce ES module
# bindings the viewer can import. No wasm-pack needed.

set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
core="$repo/core"
out="$repo/viewer/pkg"

echo "Building flow-wasm (release, wasm32-unknown-unknown)..."
( cd "$core" && cargo build --release --target wasm32-unknown-unknown -p flow-wasm )

wasm="$core/target/wasm32-unknown-unknown/release/flow_wasm.wasm"
if [ ! -f "$wasm" ]; then
    echo "expected wasm artefact not found at $wasm" >&2
    exit 1
fi

mkdir -p "$out"

echo "Generating JS bindings into $out ..."
wasm-bindgen "$wasm" \
    --out-dir "$out" \
    --out-name flow_wasm \
    --target web \
    --no-typescript

echo "Done."
ls -la "$out"
