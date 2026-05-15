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
  find . -name '._*' 2>/dev/null | while IFS= read -r apple_file; do
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

mkdir -p "$CLEANUP_ROOT"

if [[ -f Manifest.toml ]]; then
  JULIA_VERSION="$("$JULIA_BIN" --version | awk '{print $3}')"
  MANIFEST_JULIA_VERSION="$(awk -F'"' '/^julia_version =/ {print $2; exit}' Manifest.toml || true)"
  if [[ -n "$MANIFEST_JULIA_VERSION" && "${MANIFEST_JULIA_VERSION%.*}" != "${JULIA_VERSION%.*}" ]]; then
    stamp="$(date +%Y%m%d_%H%M%S)"
    mv Manifest.toml "$CLEANUP_ROOT/Manifest_${MANIFEST_JULIA_VERSION}-to-${JULIA_VERSION}_${stamp}.toml"
    echo "[build] Moved stale docs Manifest.toml resolved with Julia $MANIFEST_JULIA_VERSION"
  fi
fi

# Ensure required docs dependencies are present. `Polfed` is the package in the
# parent directory, so keep it developed through the relative path `..`; this
# makes the docs environment portable between machines with different checkout
# locations.
"$JULIA_BIN" --color=yes --project="$PROJECT_ARG" -e 'using Pkg; Pkg.develop(path=".."); Pkg.resolve(); Pkg.instantiate()'

# Avoid stale build outputs and macOS metadata from previous builds.
move_appledouble

for stale in build build_locked_*; do
  if [[ -e "$stale" ]]; then
    stamp="$(date +%Y%m%d_%H%M%S)"
    mv "$stale" "$CLEANUP_ROOT/$(basename "$stale")_$stamp"
  fi
done

"$JULIA_BIN" --color=yes --project="$PROJECT_ARG" make.jl
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
