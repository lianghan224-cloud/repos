#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNAPSHOT_DIR="$REPO_DIR/artifacts/newlog2_csv_snapshot"
OUT_ROOT="${OUT_ROOT:-/data/project/lianghan/work/logs/newlog2}"

if [[ ! -d "$SNAPSHOT_DIR" ]]; then
  echo "ERROR: missing snapshot directory: $SNAPSHOT_DIR" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"
rsync -avh "$SNAPSHOT_DIR/" "$OUT_ROOT/"
echo "[done] restored CSV snapshot to $OUT_ROOT"
