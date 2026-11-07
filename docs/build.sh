#!/usr/bin/env bash

set -euo pipefail

DOCS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JULIA_BIN="${JULIA_BIN:-julia}"

if ! command -v "$JULIA_BIN" >/dev/null 2>&1; then
  echo "Julia executable not found: $JULIA_BIN" >&2
  echo "Set JULIA_BIN=/path/to/julia and try again." >&2
  exit 1
fi

cd "$DOCS_DIR"

echo "[docs] Instantiating the documentation environment"
"$JULIA_BIN" --color=yes --project=. -e \
  'using Pkg; Pkg.develop(path=".."); Pkg.resolve(); Pkg.instantiate()'

echo "[docs] Building the site"
DOCUMENTER_DEPLOY=false "$JULIA_BIN" --color=yes --project=. make.jl

echo "[docs] Built $DOCS_DIR/build"
