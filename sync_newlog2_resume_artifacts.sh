#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./sync_newlog2_resume_artifacts.sh USER@A40_HOST

Environment:
  SRC_BASE=/data/project/lianghan/work   Source machine work root.
  DST_BASE=/data/project/lianghan/work   Target A40 work root.
  MODE=resume                            resume: sync tpdata only; full: sync all datasets.
  SYNC_TNS=1                             Also sync prepared_common_splits_tns.

Examples:
  ./sync_newlog2_resume_artifacts.sh lianghan@a40-host
  MODE=full ./sync_newlog2_resume_artifacts.sh lianghan@a40-host
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
MODE="${MODE:-resume}"
SYNC_TNS="${SYNC_TNS:-1}"

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

for path in "$SRC_SPLITS" "$SRC_LOGS"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing source path: $path" >&2
    exit 2
  fi
done

if [[ "$SYNC_TNS" == "1" && ! -e "$SRC_TNS" ]]; then
  echo "ERROR: SYNC_TNS=1 but missing source path: $SRC_TNS" >&2
  exit 2
fi

echo "[sync] remote=$REMOTE mode=$MODE sync_tns=$SYNC_TNS"
ssh "$REMOTE" "mkdir -p '$DST_SPLITS' '$DST_TNS' '$DST_LOGS'"

if [[ "$MODE" == "full" ]]; then
  echo "[sync] full prepared_common_splits"
  rsync -avh --info=progress2 "$SRC_SPLITS/" "$REMOTE:$DST_SPLITS/"
  if [[ "$SYNC_TNS" == "1" ]]; then
    echo "[sync] full prepared_common_splits_tns"
    rsync -avh --info=progress2 "$SRC_TNS/" "$REMOTE:$DST_TNS/"
  fi
else
  echo "[sync] resume metadata: completed datasets"
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

  echo "[sync] resume dataset: tpdata"
  rsync -avh --info=progress2 "$SRC_SPLITS/tpdata/" "$REMOTE:$DST_SPLITS/tpdata/"

  if [[ "$SYNC_TNS" == "1" ]]; then
    echo "[sync] resume TNS: manifest + tpdata"
    rsync -avh --info=progress2 "$SRC_TNS/manifest.json" "$REMOTE:$DST_TNS/"
    rsync -avh --info=progress2 "$SRC_TNS/tpdata/" "$REMOTE:$DST_TNS/tpdata/"
  fi
fi

echo "[sync] newlog2 CSV progress"
rsync -avh --info=progress2 --prune-empty-dirs \
  --include='*/' \
  --include='*.csv' \
  --include='driver.log' \
  --exclude='*' \
  "$SRC_LOGS/" "$REMOTE:$DST_LOGS/"

echo "[done] artifacts synced to $REMOTE:$DST_BASE"
