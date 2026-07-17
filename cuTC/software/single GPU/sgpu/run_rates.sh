#!/bin/bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <input_tns> <output_dir> <name_prefix> <comma_rates>"
  echo "Example: $0 /data/x.tns /data/logs/CUTC/X-r8-cutc X_r8 0.05,0.1,0.5"
  exit 1
fi

INPUT_TNS="$1"
OUTPUT_DIR="$2"
NAME_PREFIX="$3"
RATES_CSV="$4"
LOG_TO_CSV="/data/project/lianghan/work/repos/cuTC/software/scripts/log_to_csv.py"

if [ ! -f "$INPUT_TNS" ]; then
  echo "Input .tns not found: $INPUT_TNS" >&2
  exit 1
fi
if [ ! -f "$LOG_TO_CSV" ]; then
  echo "log_to_csv.py not found: $LOG_TO_CSV" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p "$OUTPUT_DIR"

IFS=',' read -ra RATES <<< "$RATES_CSV"

echo "[start] $(date '+%F %T') rates=${RATES_CSV}"
for rate in "${RATES[@]}"; do
  rate="${rate//[[:space:]]/}"
  if [ -z "$rate" ]; then
    continue
  fi
  rate_tag="${rate/./p}"
  name="${NAME_PREFIX}_s${rate_tag}"
  out_csv="$OUTPUT_DIR/${name}.csv"

  echo "[run] $(date '+%F %T') ${name} rate=${rate}"
  CUTC_SAMPLING_RATE="$rate" ./tc "$INPUT_TNS" \
    | python3 "$LOG_TO_CSV" --input - --output "$out_csv" >/dev/null
  echo "[done] $(date '+%F %T') ${name}"
done
echo "[finish] $(date '+%F %T') complete"
