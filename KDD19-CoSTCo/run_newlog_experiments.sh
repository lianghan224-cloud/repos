#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out_dir="/data/project/lianghan/work/logs/newlog/CoSTCo"
mkdir -p "$out_dir"

run_one() {
  local name="$1"
  local input="$2"
  local rate="$3"
  local batch_size="$4"
  local eval_batch_size="$5"
  local out="$out_dir/${name}_r8_costco_s${rate/./p}.csv"
  local tmp="${out}.tmp"
  rm -f "$tmp"
  "$repo_dir/run_costco.py" \
    --input "$input" \
    --output "$tmp" \
    --rank 8 \
    --sampling-rate "$rate" \
    --epochs "${COSTCO_EPOCHS:-300}" \
    --patience "${COSTCO_PATIENCE:-20}" \
    --tolerance "${COSTCO_TOLERANCE:-1e-4}" \
    --lr 1e-4 \
    --batch-size "$batch_size" \
    --eval-batch-size "$eval_batch_size"
  mv "$tmp" "$out"
}

run_one "DARPA" "/data/project/lianghan/work/data/frostt/darpa/1998darpa_norm.tns" "0.1" 262144 262144
run_one "LANL2" "/data/project/lianghan/work/data/testdata/newdata/lanl2.tns_clean_norm.tns" "0.5" 262144 262144
run_one "TAXI" "/data/project/lianghan/work/data/tensordata/TNS/taxi_norm.tns" "0.5" 262144 262144
run_one "tpdata" "/data/project/lianghan/work/data/testdata/tpdata/tpdata_norm.tns" "0.5" 4194304 4194304
