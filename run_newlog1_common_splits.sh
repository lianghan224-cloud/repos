#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/project/lianghan/work/data/prepared_common_splits_powerlaw_abc"
TNS_ROOT="/data/project/lianghan/work/data/prepared_common_splits_powerlaw_abc_tns"
OUT_ROOT="/data/project/lianghan/work/logs/newlog1"
REPOS="/data/project/lianghan/work/repos"

CUTC_DIR="${REPOS}/cuTC/software/single GPU/sgpu"
CUTC_LOG_TO_CSV="${REPOS}/cuTC/software/scripts/log_to_csv.py"
BLCO_DIR="${REPOS}/exp/blocked-linearized-coordinate"
BLCO_BIN="${BLCO_DIR}/cpd64"
BLCO_LOG_TO_CSV="${BLCO_DIR}/scripts/log_to_csv_blco.py"
GENTEN_TOOLS="${REPOS}/GenTen/tools"
GENTEN_BIN="${REPOS}/GenTen/build/cuda/bin/genten"
COSTCO_DIR="${REPOS}/KDD19-CoSTCo"
WORK_TMP="/data/project/lianghan/work/tmp/newlog1_common_splits"

DATASETS=("DARPA" "LANL2" "BJTaxi" "tpdata")

rate_for() {
  python3 - "$ROOT/$1/${1}_metadata.json" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    meta = json.load(f)
rate = meta.get("sampling", {}).get("sampling_rate_requested")
if rate is None:
    name = meta.get("dataset", "")
    rate = 0.1 if name == "DARPA" else 0.5
print(rate)
PY
}

shape_for() {
  python3 - "$ROOT/$1/${1}_metadata.json" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    meta = json.load(f)
print(",".join(str(int(x)) for x in meta["shape"]))
PY
}

csv_done() {
  local path="$1"
  [[ -s "$path" ]] && [[ "$(wc -l < "$path")" -gt 1 ]]
}

ensure_dirs() {
  mkdir -p "$OUT_ROOT"/{CUTC-sgd,CUTC-ccd,CUTC-als,BLCO,GenTen,CoSTCo}
  mkdir -p "$WORK_TMP"
}

build_bins() {
  PATH="/usr/local/cuda/bin:$PATH" make -C "$CUTC_DIR" tc
  rm -rf "$BLCO_DIR/build-64"
  mkdir -p "$BLCO_DIR/build-64"
  rm -f "$BLCO_BIN"
  env -u CXXFLAGS -u CPPFLAGS -u LDFLAGS PATH="/usr/local/cuda/bin:$PATH" \
    make -C "$BLCO_DIR" \
    COMPILER=GCC \
    BLAS_LIBRARY=OPENBLAS \
    BLASINC= \
    BLASLIBS="/opt/anaconda3/envs/zhangsac/lib/liblapacke.so /opt/anaconda3/envs/zhangsac/lib/liblapack.so /opt/anaconda3/envs/zhangsac/lib/libopenblas.so -lgfortran" \
    cpd64
}

export_tns() {
  if [[ -f "$TNS_ROOT/manifest.json" ]]; then
    echo "[skip] existing TNS export: $TNS_ROOT"
    return
  fi
  python3 "$REPOS/export_common_splits_to_tns.py" \
    --root "$ROOT" \
    --output-root "$TNS_ROOT"
}

run_cutc_one() {
  local alg="$1"
  local group="$2"
  local out="$3"
  local train="$4"
  local val="$5"
  local test="$6"
  if csv_done "$out"; then
    echo "[skip] $out"
    return
  fi
  local tmp="${out}.tmp"
  rm -f "$tmp"
  echo "[run] CUTC-${alg} -> $out"
  local alg_env=()
  case "$alg" in
    als)
      alg_env+=(CUTC_ALS_REGULARIZATION="${CUTC_ALS_REGULARIZATION:-1e-6}")
      ;;
    ccd)
      alg_env+=(CUTC_CCD_CISS="${CUTC_CCD_CISS:-0}")
      alg_env+=(CUTC_CCD_REGULARIZATION="${CUTC_CCD_REGULARIZATION:-0.2}")
      ;;
  esac
  (
    cd "$CUTC_DIR"
    env "${alg_env[@]}" \
    CUTC_PRESPLIT=1 \
    CUTC_ALGORITHM="$alg" \
    CUTC_MAX_ITERATE=300 \
    CUTC_MAX_BADEPOCHS="${CUTC_MAX_BADEPOCHS:-20}" \
    CUTC_TOLERANCE="${CUTC_TOLERANCE:-1e-6}" \
    ./tc "$train" "$train" "$train" "$val" "$test"
  ) | python3 "$CUTC_LOG_TO_CSV" --input - --output "$tmp" >/dev/null
  mv "$tmp" "$out"
}

run_cutc() {
  local ds="$1"
  local rate="$2"
  local rate_tag="${rate/./p}"
  local train="$TNS_ROOT/$ds/${ds}_train.tns"
  local val="$TNS_ROOT/$ds/${ds}_val.tns"
  local test="$TNS_ROOT/$ds/${ds}_test.tns"
  run_cutc_one "sgd" "CUTC-sgd" "$OUT_ROOT/CUTC-sgd/${ds}_norm_r8_lr001_s${rate_tag}.csv" "$train" "$val" "$test"
  run_cutc_one "ccd" "CUTC-ccd" "$OUT_ROOT/CUTC-ccd/${ds}_norm_r8_lr001_ccd_s${rate_tag}.csv" "$train" "$val" "$test"
  run_cutc_one "als" "CUTC-als" "$OUT_ROOT/CUTC-als/${ds}_norm_r8_als_s${rate_tag}.csv" "$train" "$val" "$test"
}

run_blco() {
  local ds="$1"
  local rate="$2"
  local rate_tag="${rate/./p}"
  local out="$OUT_ROOT/BLCO/${ds}_r8_blco_s${rate_tag}.csv"
  if csv_done "$out"; then
    echo "[skip] $out"
    return
  fi
  local tmp="${out}.tmp"
  rm -f "$tmp"
  echo "[run] BLCO -> $out"
  LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/opt/anaconda3/envs/zhangsac/lib:${LD_LIBRARY_PATH:-}" \
  "$BLCO_BIN" \
    --train-file "$TNS_ROOT/$ds/${ds}_train.tns" \
    --val-file "$TNS_ROOT/$ds/${ds}_val.tns" \
    --test-file "$TNS_ROOT/$ds/${ds}_test.tns" \
    --dims "$(shape_for "$ds")" \
    --rank 8 \
    --max-iter 150 \
    --kernel-id 10 \
    --device 0 \
    --thread-cf 2 \
    2>&1 | python3 "$BLCO_LOG_TO_CSV" --input - --output "$tmp" >/dev/null
  mv "$tmp" "$out"
}

run_genten() {
  local ds="$1"
  local rate="$2"
  local rate_tag="${rate/./p}"
  local out="$OUT_ROOT/GenTen/${ds}_r8_genten_s${rate_tag}.csv"
  if csv_done "$out"; then
    echo "[skip] $out"
    return
  fi
  local tmp="${out}.tmp"
  rm -f "$tmp"
  echo "[run] GenTen -> $out"
  (
    cd "$GENTEN_TOOLS"
    python3 run_genten_sparse_tns_epoch_metrics.py \
      --train-npz "$ROOT/$ds/${ds}_train.npz" \
      --val-npz "$ROOT/$ds/${ds}_val.npz" \
      --test-npz "$ROOT/$ds/${ds}_test.npz" \
      --genten-bin "$GENTEN_BIN" \
      --dataset-tag "$ds" \
      --output-file "$tmp" \
      --sample-rate "$rate" \
      --rank 8 \
      --maxiters 20 \
      --work-dir "$WORK_TMP/genten"
  )
  mv "$tmp" "$out"
}

run_costco() {
  local ds="$1"
  local rate="$2"
  local rate_tag="${rate/./p}"
  local out="$OUT_ROOT/CoSTCo/${ds}_r8_costco_s${rate_tag}.csv"
  if csv_done "$out"; then
    echo "[skip] $out"
    return
  fi
  local tmp="${out}.tmp"
  rm -f "$tmp"
  echo "[run] CoSTCo -> $out"
  "$COSTCO_DIR/run_costco.py" \
    --train-npz "$ROOT/$ds/${ds}_train.npz" \
    --val-npz "$ROOT/$ds/${ds}_val.npz" \
    --test-npz "$ROOT/$ds/${ds}_test.npz" \
    --output "$tmp" \
    --rank 8 \
    --sampling-rate "$rate" \
    --epochs 300 \
    --patience 20 \
    --tolerance 1e-6 \
    --lr 1e-4 \
    --batch-size 262144 \
    --eval-batch-size 262144
  mv "$tmp" "$out"
}

main() {
  ensure_dirs
  build_bins
  export_tns
  for ds in "${DATASETS[@]}"; do
    local rate
    rate="$(rate_for "$ds")"
    echo "[dataset] $ds rate=$rate"
    run_cutc "$ds" "$rate"
    run_blco "$ds" "$rate"
    run_genten "$ds" "$rate"
    run_costco "$ds" "$rate"
  done
  echo "[done] outputs: $OUT_ROOT"
}

main "$@"
