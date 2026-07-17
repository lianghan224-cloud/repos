#!/bin/bash
set -euo pipefail

# CSV-only batch runner for cuTC.
# Usage:
#   ./run.sh <input_tns> <output_dir> <name_prefix> [num_rates]
# Example:
#   ./run.sh /data/.../ECW_08_norm.tns /data/.../logs/CUTC/ECW_08-cutc ECW_08 10

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: $0 <input_tns> <output_dir> <name_prefix> [num_rates]"
  exit 1
fi

INPUT_TNS="$1"
OUTPUT_DIR="$2"
NAME_PREFIX="$3"
NUM_RATES="${4:-10}"
LOG_TO_CSV="/data/project/lianghan/work/repos/cuTC/software/scripts/log_to_csv.py"

if [ ! -f "$INPUT_TNS" ]; then
  echo "Input .tns not found: $INPUT_TNS"
  exit 1
fi
if [ ! -f "$LOG_TO_CSV" ]; then
  echo "log_to_csv.py not found: $LOG_TO_CSV"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "[start] $(date '+%F %T') csv-only run"
for i in $(seq 1 "$NUM_RATES"); do
  rate="$(awk -v i="$i" 'BEGIN{printf("%.1f", i/10.0)}')"
  name="$(printf '%s%02d' "$NAME_PREFIX" "$i")"
  out_csv="$OUTPUT_DIR/${name}.csv"

  echo "[run] $(date '+%F %T') ${name} rate=${rate}"
  CUTC_SAMPLING_RATE="$rate" ./tc "$INPUT_TNS" \
    | python3 "$LOG_TO_CSV" --input - --output "$out_csv" >/dev/null
  echo "[done] $(date '+%F %T') ${name}"
done
echo "[finish] $(date '+%F %T') csv-only run complete"
