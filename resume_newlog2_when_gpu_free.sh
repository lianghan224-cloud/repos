#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="/data/project/lianghan/work/logs/newlog2"
RUNNER="/data/project/lianghan/work/repos/run_newlog2_common_splits.sh"
LOG="$OUT_ROOT/driver.log"
PIDFILE="$OUT_ROOT/driver.pid"
LOCKDIR="$OUT_ROOT/resume_wait.lock"
MIN_FREE_MIB="${MIN_FREE_MIB:-43000}"
INTERVAL_SEC="${INTERVAL_SEC:-120}"

mkdir -p "$OUT_ROOT"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "[resume-wait] $(date -Is) another resume waiter is active: $LOCKDIR"
  exit 0
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

echo "[resume-wait] $(date -Is) waiting for GPU free memory >= ${MIN_FREE_MIB} MiB"
while true; do
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1 {print int($1)}')"
  used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR==1 {print int($1)}')"
  echo "[resume-wait] $(date -Is) gpu_free=${free_mib}MiB gpu_used=${used_mib}MiB"
  if [[ "$free_mib" -ge "$MIN_FREE_MIB" ]]; then
    break
  fi
  sleep "$INTERVAL_SEC"
done

echo "[resume-start] $(date -Is) skip_ccd_als=1 tol=1e-4 min_free=${MIN_FREE_MIB}MiB"
printf '%s\n' "$$" > "$PIDFILE"
RUN_CUTC_CCD=0 RUN_CUTC_ALS=0 CUTC_TOLERANCE=1e-4 "$RUNNER"
rc=$?
echo "[resume-exit] $(date -Is) rc=$rc"
exit "$rc"
