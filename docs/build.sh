#!/usr/bin/env bash

set -euo pipefail

JULIA_BIN="${JULIA_BIN:-julia-x86_64}"
if ! command -v "$JULIA_BIN" >/dev/null 2>&1; then
  JULIA_BIN="julia"
fi

PROJECT_ARG="${1:-@.}"

echo "[build] Using Julia binary: $JULIA_BIN"
echo "[build] Using project: $PROJECT_ARG"

# Ensure all manifest-recorded dependencies are present (fixes missing *_jll errors).
"$JULIA_BIN" --color=yes --project="$PROJECT_ARG" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Avoid stale files from previous builds.
rm -rf build
"$JULIA_BIN" --color=yes --project="$PROJECT_ARG" make.jl

# Backward-compatibility redirect for legacy URL:
# .../polfed/documentation/generated/  -> .../polfed/
mkdir -p build/documentation/generated
cat > build/documentation/generated/index.html <<'EOF'
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="refresh" content="0; url=../../" />
    <title>Redirecting...</title>
  </head>
  <body>
    Redirecting to <a href="../../">home</a>.
  </body>
</html>
EOF

rm -rf /project/rokpintar/public_html/polfed
cp -R build /project/rokpintar/public_html/polfed

echo "[build] Docs deployed to /project/rokpintar/public_html/polfed"
