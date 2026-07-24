#!/usr/bin/env bash

set -euo pipefail

JULIA_BIN="${JULIA_BIN:-julia-x86_64}"
if ! command -v "$JULIA_BIN" >/dev/null 2>&1; then
  JULIA_BIN="julia"
fi

PROJECT_ARG="${1:-@.}"
if [[ -n "${DEPLOY_DIRS:-}" ]]; then
  read -r -a DEPLOY_TARGETS <<< "$DEPLOY_DIRS"
elif [[ -n "${DEPLOY_DIR:-}" ]]; then
  DEPLOY_TARGETS=("$DEPLOY_DIR")
else
  DEPLOY_TARGETS=(
    "/project/rokpintar/public_html/UniPolfed.jl"
  )
fi
VOLUME_ROOT="$(df -P . | awk 'NR==2 {print $6}')"
CLEANUP_ROOT="${DOCS_CLEANUP_DIR:-$VOLUME_ROOT/.unipolfed_docs_cleanup}"

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
echo "[build] Deploy targets:"
printf '  - %s\n' "${DEPLOY_TARGETS[@]}"
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

# Ensure required docs dependencies are present. `UniPolfed` is the package in the
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

deployed=false
for deploy_dir in "${DEPLOY_TARGETS[@]}"; do
  if [[ -z "$deploy_dir" ]]; then
    continue
  fi

  deploy_parent="$(dirname "$deploy_dir")"
  if [[ -d "$deploy_parent" ]]; then
    rm -rf "$deploy_dir"
    mkdir -p "$deploy_dir"
    cp -R build/. "$deploy_dir/"
    echo "[build] Docs deployed to $deploy_dir"
    deployed=true
  else
    echo "[build] Skipping deploy to $deploy_dir because $deploy_parent does not exist on this machine"
  fi
done

if [[ "$deployed" == false ]]; then
  echo "[build] Build completed locally at $(pwd)/build"
fi
