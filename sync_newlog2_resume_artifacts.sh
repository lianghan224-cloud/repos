#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./sync_newlog2_resume_artifacts.sh USER@A40_HOST

Environment:
  SRC_BASE=/data/project/lianghan/work   Source machine work root.
  DST_BASE=/data/project/lianghan/work   Target A40 work root.
  MODE=full                              full: sync all datasets; resume: legacy tpdata-only sync.
  SYNC_TNS=1                             Also sync prepared_common_splits_tns.
  SYNC_LOGS=0                            Also sync old newlog2 CSV progress. Keep 0 for strict timing runs.

Examples:
  ./sync_newlog2_resume_artifacts.sh lianghan@a40-host
  SYNC_LOGS=1 ./sync_newlog2_resume_artifacts.sh lianghan@a40-host
  DST_BASE=/data/project/lianghan/work ./sync_newlog2_resume_artifacts.sh lianghan@a40-host
EOF
}

REMOTE="${1:-}"
if [[ -z "$REMOTE" || "$REMOTE" == "-h" || "$REMOTE" == "--help" ]]; then
  usage
  exit 1
fi

SRC_BASE="${SRC_BASE:-/data/project/lianghan/work}"
DST_BASE="${DST_BASE:-/data/project/lianghan/work}"
MODE="${MODE:-full}"
SYNC_TNS="${SYNC_TNS:-1}"
SYNC_LOGS="${SYNC_LOGS:-0}"

SRC_SPLITS="$SRC_BASE/data/prepared_common_splits"
SRC_TNS="$SRC_BASE/data/prepared_common_splits_tns"
SRC_LOGS="$SRC_BASE/logs/newlog2"
DST_SPLITS="$DST_BASE/data/prepared_common_splits"
DST_TNS="$DST_BASE/data/prepared_common_splits_tns"
DST_LOGS="$DST_BASE/logs/newlog2"

if [[ "$MODE" != "resume" && "$MODE" != "full" ]]; then
  echo "ERROR: MODE must be resume or full, got: $MODE" >&2
  exit 2
fi

for path in "$SRC_SPLITS"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing source path: $path" >&2
    exit 2
  fi
done

if [[ "$SYNC_LOGS" == "1" && ! -e "$SRC_LOGS" ]]; then
  echo "ERROR: SYNC_LOGS=1 but missing source path: $SRC_LOGS" >&2
  exit 2
fi

if [[ "$SYNC_TNS" == "1" && ! -e "$SRC_TNS" ]]; then
  echo "ERROR: SYNC_TNS=1 but missing source path: $SRC_TNS" >&2
  exit 2
fi

echo "[sync] remote=$REMOTE mode=$MODE sync_tns=$SYNC_TNS sync_logs=$SYNC_LOGS"
ssh "$REMOTE" "mkdir -p '$DST_SPLITS' '$DST_TNS' '$DST_LOGS'"

if [[ "$MODE" == "full" ]]; then
  echo "[sync] full prepared_common_splits"
  rsync -avh --info=progress2 "$SRC_SPLITS/" "$REMOTE:$DST_SPLITS/"
  if [[ "$SYNC_TNS" == "1" ]]; then
    echo "[sync] full prepared_common_splits_tns"
    rsync -avh --info=progress2 "$SRC_TNS/" "$REMOTE:$DST_TNS/"
  fi
else
  echo "[sync] legacy resume metadata: completed datasets"
  rsync -avh --info=progress2 \
    "$SRC_SPLITS/run_manifest.json" \
    "$SRC_SPLITS/prepared_dataset_summary.csv" \
    "$REMOTE:$DST_SPLITS/"
  for ds in DARPA LANL2 BJTaxi; do
    ssh "$REMOTE" "mkdir -p '$DST_SPLITS/$ds'"
    rsync -avh --info=progress2 \
      "$SRC_SPLITS/$ds/${ds}_metadata.json" \
      "$SRC_SPLITS/$ds/README.txt" \
      "$REMOTE:$DST_SPLITS/$ds/"
  done

  echo "[sync] legacy resume dataset: tpdata"
  rsync -avh --info=progress2 "$SRC_SPLITS/tpdata/" "$REMOTE:$DST_SPLITS/tpdata/"

  if [[ "$SYNC_TNS" == "1" ]]; then
    echo "[sync] legacy resume TNS: manifest + tpdata"
    rsync -avh --info=progress2 "$SRC_TNS/manifest.json" "$REMOTE:$DST_TNS/"
    rsync -avh --info=progress2 "$SRC_TNS/tpdata/" "$REMOTE:$DST_TNS/tpdata/"
  fi
fi

if [[ "$SYNC_LOGS" == "1" ]]; then
  cat <<'EOF'
[sync] WARNING: syncing newlog2 CSV progress.
[sync] Current strict timing runs should normally start with an empty OUT_ROOT,
[sync] because old CSVs use a different train_gpu timing definition.
EOF
  rsync -avh --info=progress2 --prune-empty-dirs \
    --include='*/' \
    --include='*.csv' \
    --include='driver.log' \
    --exclude='*' \
    "$SRC_LOGS/" "$REMOTE:$DST_LOGS/"
else
  echo "[skip] newlog2 CSV progress not synced; strict timing runs should not mix old CSVs"
fi

echo "[done] artifacts synced to $REMOTE:$DST_BASE"
