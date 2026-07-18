#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNAPSHOT_DIR="$REPO_DIR/artifacts/newlog2_csv_snapshot"
OUT_ROOT="${OUT_ROOT:-/data/project/lianghan/work/logs/newlog2}"
ALLOW_OLD_TIMING_RESTORE="${ALLOW_OLD_TIMING_RESTORE:-0}"

if [[ ! -d "$SNAPSHOT_DIR" ]]; then
  echo "ERROR: missing snapshot directory: $SNAPSHOT_DIR" >&2
  exit 2
fi

if [[ "$ALLOW_OLD_TIMING_RESTORE" != "1" ]]; then
  cat >&2 <<'EOF'
ERROR: artifacts/newlog2_csv_snapshot uses the old train_gpu timing definition.

Current newlog2 runs use strict GPU training time:
  train_gpu excludes H2D, D2H, memset/init, objective/fit, and metric evaluation.

Do not restore this snapshot into a strict-timing OUT_ROOT, or the runner will
skip experiments that must be rerun. To restore only for archival comparison,
set ALLOW_OLD_TIMING_RESTORE=1 explicitly.
EOF
  exit 2
fi

mkdir -p "$OUT_ROOT"
rsync -avh "$SNAPSHOT_DIR/" "$OUT_ROOT/"
echo "[done] restored CSV snapshot to $OUT_ROOT"
