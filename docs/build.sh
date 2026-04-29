#!/usr/bin/env bash

set -euo pipefail

JULIA_BIN="${JULIA_BIN:-julia-x86_64}"
if ! command -v "$JULIA_BIN" >/dev/null 2>&1; then
  JULIA_BIN="julia"
fi

PROJECT_ARG="${1:-@.}"
DEPLOY_DIR="${DEPLOY_DIR:-/project/rokpintar/public_html/polfed}"
VOLUME_ROOT="$(df -P . | awk 'NR==2 {print $6}')"
CLEANUP_ROOT="${DOCS_CLEANUP_DIR:-$VOLUME_ROOT/.polfed_docs_cleanup}"

move_appledouble() {
  find . -name '._*' | while IFS= read -r apple_file; do
    rel="${apple_file#./}"
    dest="$CLEANUP_ROOT/appledouble/$rel"
    mkdir -p "$(dirname "$dest")"
    mv "$apple_file" "$dest"
  done
}

echo "[build] Using Julia binary: $JULIA_BIN"
echo "[build] Using project: $PROJECT_ARG"
echo "[build] Deploy dir: $DEPLOY_DIR"
echo "[build] Cleanup dir: $CLEANUP_ROOT"

# Ensure required docs dependencies are present. Avoid eager precompilation of
# unrelated packages, because optional docs tooling can fail even when the real
# build dependencies are fine.
"$JULIA_BIN" --color=yes --project="$PROJECT_ARG" -e 'using Pkg; Pkg.instantiate()'

# Avoid stale files and macOS metadata from previous builds.
mkdir -p "$CLEANUP_ROOT"

move_appledouble

if [[ -d src/documentation/generated ]]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  mv src/documentation/generated "$CLEANUP_ROOT/generated_$stamp"
fi
mkdir -p src/documentation/generated

for stale in build build_locked_*; do
  if [[ -e "$stale" ]]; then
    stamp="$(date +%Y%m%d_%H%M%S)"
    mv "$stale" "$CLEANUP_ROOT/$(basename "$stale")_$stamp"
  fi
done

"$JULIA_BIN" --color=yes --project="$PROJECT_ARG" make.jl
move_appledouble

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
move_appledouble

deploy_parent="$(dirname "$DEPLOY_DIR")"
if [[ -d "$deploy_parent" ]]; then
  rm -rf "$DEPLOY_DIR"
  cp -R build "$DEPLOY_DIR"
  echo "[build] Docs deployed to $DEPLOY_DIR"
else
  echo "[build] Build completed locally at $(pwd)/build"
  echo "[build] Skipping deploy because $deploy_parent does not exist on this machine"
fi
