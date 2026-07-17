#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" != "0" ]]; then
  echo "ERROR: run as root" >&2
  exit 1
fi

OUT_ROOT="/data/project/lianghan/work/logs/newlog2"
PIDFILE="$OUT_ROOT/resume_wait.pid"
OWNER_USER="${OWNER_USER:-lianghan}"
OWNER_GROUP="${OWNER_GROUP:-$(id -gn "$OWNER_USER")}"
BACKUP_DIR="/root/nvidia-perm-backup"
mkdir -p "$BACKUP_DIR" "$OUT_ROOT"

watch_pid="${1:-}"
if [[ -z "$watch_pid" ]]; then
  watch_pid="$(cat "$PIDFILE" 2>/dev/null || true)"
fi
if [[ -z "$watch_pid" ]] || ! kill -0 "$watch_pid" 2>/dev/null; then
  echo "ERROR: watched PID is not running: ${watch_pid:-<empty>}" >&2
  exit 1
fi

devices=()
for dev in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset; do
  [[ -e "$dev" ]] && devices+=("$dev")
done
if [[ "${#devices[@]}" -eq 0 ]]; then
  echo "ERROR: no NVIDIA devices found" >&2
  exit 1
fi

backup="$BACKUP_DIR/perms.$(date +%Y%m%dT%H%M%S).txt"
stat -c '%a %u %g %n' "${devices[@]}" > "$backup"

chgrp "$OWNER_GROUP" "${devices[@]}"
chmod 660 "${devices[@]}"

monitor_log="$BACKUP_DIR/restore-monitor.$watch_pid.log"
nohup bash -lc '
set -euo pipefail
watch_pid="$1"
backup="$2"
out_root="$3"
echo "[gpu-lock] $(date -Is) waiting for PID ${watch_pid} to exit; backup=${backup}"
while kill -0 "$watch_pid" 2>/dev/null; do
  sleep 30
done
while read -r mode uid gid path; do
  [[ -e "$path" ]] || continue
  chown "${uid}:${gid}" "$path" 2>/dev/null || true
  chmod "$mode" "$path" 2>/dev/null || true
done < "$backup"
echo "[gpu-lock] $(date -Is) restored NVIDIA device permissions after PID ${watch_pid} exited"
echo "[gpu-lock] $(date -Is) restored NVIDIA device permissions after PID ${watch_pid} exited" >> "$out_root/gpu_perm_restore.log"
' _ "$watch_pid" "$backup" "$OUT_ROOT" > "$monitor_log" 2>&1 &

echo "Locked NVIDIA devices for ${OWNER_USER}:${OWNER_GROUP}"
echo "Watched PID: $watch_pid"
echo "Restore monitor PID: $!"
echo "Permission backup: $backup"
echo "Monitor log: $monitor_log"
