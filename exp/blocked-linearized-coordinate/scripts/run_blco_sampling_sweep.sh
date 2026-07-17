#!/usr/bin/env bash
set -euo pipefail

# Run BLCO CPD for sampling rates 0.1..1.0 and export CUTC-compatible CSV files.
#
# Output directory:
#   /data/project/lianghan/work/logs/BLCO/<ALIAS>-blco
#
# File names:
#   <ALIAS>01.csv ... <ALIAS>10.csv
#
# CSV-only behavior:
#   Any stale .log files under the output directory are removed before running.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="/data/project/lianghan/work/logs/BLCO"

INPUT_TNS=""
ALIAS=""
BINARY="${REPO_DIR}/cpd64"
RANK=16
MAX_ITERS=150
KERNEL=10
DEVICE=0
THREAD_CF=2
VAL_RATE=0.05
SAMPLING_MODE=0
SAMPLING_ALPHA=1.0
SAMPLING_EPS=1e-6
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --input <tensor.tns> --alias <XXX> [options] [-- <extra args>]

Required:
  --input PATH           Input .tns tensor path.
  --alias NAME           Dataset short name for output naming.

Optional:
  --binary PATH          BLCO binary (default: ${REPO_DIR}/cpd64)
  --rank INT             CP rank (default: ${RANK})
  --max-iter INT         Max epochs (default: ${MAX_ITERS})
  --kernel INT           GPU kernel id (default: ${KERNEL})
  --device INT           CUDA device id (default: ${DEVICE})
  --thread-cf INT        Thread coarsening factor (default: ${THREAD_CF})
  --val-rate FLOAT       Validation split rate (default: ${VAL_RATE})
  --sampling-mode INT    0=uniform, 1=value-biased (default: ${SAMPLING_MODE})
  --sampling-alpha FLOAT Value-biased alpha (default: ${SAMPLING_ALPHA})
  --sampling-eps FLOAT   Value-biased eps (default: ${SAMPLING_EPS})

Examples:
  $(basename "$0") --input /path/CBW.tns --alias CBW --max-iter 50
  $(basename "$0") --input /path/GS.tns --alias GS -- --stream-data --max-block-size 8388608
EOF
}

while (($#)); do
  case "$1" in
    --input) INPUT_TNS="$2"; shift 2 ;;
    --alias) ALIAS="$2"; shift 2 ;;
    --binary) BINARY="$2"; shift 2 ;;
    --rank) RANK="$2"; shift 2 ;;
    --max-iter) MAX_ITERS="$2"; shift 2 ;;
    --kernel) KERNEL="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --thread-cf) THREAD_CF="$2"; shift 2 ;;
    --val-rate) VAL_RATE="$2"; shift 2 ;;
    --sampling-mode) SAMPLING_MODE="$2"; shift 2 ;;
    --sampling-alpha) SAMPLING_ALPHA="$2"; shift 2 ;;
    --sampling-eps) SAMPLING_EPS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${INPUT_TNS}" || -z "${ALIAS}" ]]; then
  usage
  exit 1
fi

if [[ ! -f "${INPUT_TNS}" ]]; then
  echo "Input tensor not found: ${INPUT_TNS}" >&2
  exit 1
fi

if [[ ! -x "${BINARY}" ]]; then
  echo "Binary not executable: ${BINARY}" >&2
  exit 1
fi

OUT_DIR="${LOG_ROOT}/${ALIAS}-blco"
mkdir -p "${OUT_DIR}"
# Keep this workflow CSV-only to avoid log/csv mismatch from stale log files.
find "${OUT_DIR}" -maxdepth 1 -type f -name "*.log" -delete

echo "Output dir: ${OUT_DIR}"
for i in $(seq 1 10); do
  if [[ "${i}" -lt 10 ]]; then
    rate="0.${i}"
    tag="0${i}"
  else
    rate="1.0"
    tag="10"
  fi

  csv_file="${OUT_DIR}/${ALIAS}${tag}.csv"

  echo "==> sampling_rate=${rate} -> ${csv_file}"
  "${BINARY}" \
    -i "${INPUT_TNS}" \
    --rank "${RANK}" \
    -m "${MAX_ITERS}" \
    -k "${KERNEL}" \
    --device "${DEVICE}" \
    --thread-cf "${THREAD_CF}" \
    --val-rate "${VAL_RATE}" \
    --sampling-rate "${rate}" \
    --sampling-mode "${SAMPLING_MODE}" \
    --sampling-alpha "${SAMPLING_ALPHA}" \
    --sampling-eps "${SAMPLING_EPS}" \
    "${EXTRA_ARGS[@]}" 2>&1 \
    | python3 "${REPO_DIR}/scripts/log_to_csv_blco.py" --input - --output "${csv_file}"
done

echo "Done. Generated CSVs in: ${OUT_DIR}"
