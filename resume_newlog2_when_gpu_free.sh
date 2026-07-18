#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="${OUT_ROOT:-/data/project/lianghan/work/logs/newlog2}"
OUT_ROOT="${OUT_ROOT%/}"
RUNNER="${RUNNER:-$SCRIPT_DIR/run_newlog2_common_splits.sh}"
LOG="$OUT_ROOT/driver.log"
PIDFILE="$OUT_ROOT/driver.pid"
WAIT_PIDFILE="${WAIT_PIDFILE:-$OUT_ROOT/resume_wait.pid}"
LOCKDIR="$OUT_ROOT/resume_wait.lock"
MIN_FREE_MIB="${MIN_FREE_MIB:-43000}"
INTERVAL_SEC="${INTERVAL_SEC:-120}"

mkdir -p "$OUT_ROOT"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  existing_pid="$(cat "$WAIT_PIDFILE" 2>/dev/null || true)"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "[resume-wait] $(date -Is) another resume waiter is active: pid=$existing_pid lock=$LOCKDIR"
    exit 0
  fi
  echo "[resume-wait] $(date -Is) removing stale resume lock: $LOCKDIR"
  rm -rf "$LOCKDIR"
  mkdir "$LOCKDIR"
fi
printf '%s\n' "$$" > "$WAIT_PIDFILE"
trap 'rm -f "$WAIT_PIDFILE"; rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

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
RUN_CUTC_CCD="${RUN_CUTC_CCD:-0}" \
RUN_CUTC_ALS="${RUN_CUTC_ALS:-0}" \
CUTC_TOLERANCE="${CUTC_TOLERANCE:-1e-4}" \
"$RUNNER" 2>&1 | tee -a "$LOG"
rc=$?
echo "[resume-exit] $(date -Is) rc=$rc"
exit "$rc"
