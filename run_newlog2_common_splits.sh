#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WORK_ROOT="/data/project/lianghan/work"

ROOT="${ROOT:-${DATA_ROOT:-$DEFAULT_WORK_ROOT/data/prepared_common_splits}}"
ROOT="${ROOT%/}"
TNS_ROOT="${TNS_ROOT:-${ROOT}_tns}"
TNS_ROOT="${TNS_ROOT%/}"
OUT_ROOT="${OUT_ROOT:-$DEFAULT_WORK_ROOT/logs/newlog2}"
OUT_ROOT="${OUT_ROOT%/}"
OUT_PARENT="$(dirname "$OUT_ROOT")"
WORK_BASE="$(dirname "$OUT_PARENT")"
if [[ "$WORK_BASE" == "/" ]]; then
  DEFAULT_WORK_TMP="/tmp/newlog2_common_splits"
else
  DEFAULT_WORK_TMP="$WORK_BASE/tmp/newlog2_common_splits"
fi
REPOS="${REPOS:-$SCRIPT_DIR}"
REPOS="${REPOS%/}"

CUTC_DIR="${REPOS}/cuTC/software/single GPU/sgpu"
CUTC_LOG_TO_CSV="${REPOS}/cuTC/software/scripts/log_to_csv.py"
BLCO_DIR="${REPOS}/exp/blocked-linearized-coordinate"
BLCO_BIN="${BLCO_BIN:-${BLCO_DIR}/cpd64}"
BLCO_LOG_TO_CSV="${BLCO_DIR}/scripts/log_to_csv_blco.py"
GENTEN_TOOLS="${REPOS}/GenTen/tools"
GENTEN_BIN="${GENTEN_BIN:-${REPOS}/GenTen/build/cuda/bin/genten}"
COSTCO_DIR="${REPOS}/KDD19-CoSTCo"
WORK_TMP="${WORK_TMP:-$DEFAULT_WORK_TMP}"
WORK_TMP="${WORK_TMP%/}"

DATASETS_ENV="${DATASETS:-DARPA LANL2 BJTaxi tpdata}"
DATASETS_ENV="${DATASETS_ENV//,/ }"
read -r -a DATASETS <<< "$DATASETS_ENV"
RUN_CUTC_SGD="${RUN_CUTC_SGD:-1}"
RUN_CUTC_CCD="${RUN_CUTC_CCD:-0}"
RUN_CUTC_ALS="${RUN_CUTC_ALS:-0}"
RUN_BLCO="${RUN_BLCO:-1}"
RUN_GENTEN="${RUN_GENTEN:-1}"
RUN_COSTCO="${RUN_COSTCO:-1}"
FORCE_BUILD="${FORCE_BUILD:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
DRY_RUN="${DRY_RUN:-0}"
GPU_DEVICE="${GPU_DEVICE:-0}"
BLCO_BLASLIBS="${BLCO_BLASLIBS:-/opt/anaconda3/envs/zhangsac/lib/liblapacke.so /opt/anaconda3/envs/zhangsac/lib/liblapack.so /opt/anaconda3/envs/zhangsac/lib/libopenblas.so -lgfortran}"
CUTC_MAX_ITERATE="${CUTC_MAX_ITERATE:-300}"
CUTC_MAX_BADEPOCHS="${CUTC_MAX_BADEPOCHS:-20}"
CUTC_TOLERANCE="${CUTC_TOLERANCE:-1e-4}"
BLCO_MAX_ITER="${BLCO_MAX_ITER:-300}"
BLCO_MAX_BADEPOCHS="${BLCO_MAX_BADEPOCHS:-20}"
BLCO_TOLERANCE="${BLCO_TOLERANCE:-1e-4}"
GENTEN_MAXITERS="${GENTEN_MAXITERS:-300}"
GENTEN_FROZENITERS="${GENTEN_FROZENITERS:-0}"
GENTEN_PATIENCE="${GENTEN_PATIENCE:-20}"
GENTEN_TOLERANCE="${GENTEN_TOLERANCE:-1e-4}"
COSTCO_EPOCHS="${COSTCO_EPOCHS:-300}"
COSTCO_PATIENCE="${COSTCO_PATIENCE:-20}"
COSTCO_TOLERANCE="${COSTCO_TOLERANCE:-1e-4}"

rate_for() {
  python3 - "$ROOT/$1/${1}_metadata.json" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8-sig") as f:
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
with open(sys.argv[1], "r", encoding="utf-8-sig") as f:
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
  if [[ "$SKIP_BUILD" == "1" ]]; then
    echo "[skip] SKIP_BUILD=1"
    return
  fi
  if [[ "$RUN_CUTC_SGD" == "1" || "$RUN_CUTC_CCD" == "1" || "$RUN_CUTC_ALS" == "1" ]]; then
    PATH="/usr/local/cuda/bin:$PATH" make -C "$CUTC_DIR" tc
  else
    echo "[skip] cuTC build disabled"
  fi
  if [[ "$RUN_BLCO" == "1" && ( "$FORCE_BUILD" == "1" || ! -x "$BLCO_BIN" ) ]]; then
    rm -rf "$BLCO_DIR/build-64"
    mkdir -p "$BLCO_DIR/build-64"
    rm -f "$BLCO_BIN"
    env -u CXXFLAGS -u CPPFLAGS -u LDFLAGS PATH="/usr/local/cuda/bin:$PATH" \
      make -C "$BLCO_DIR" \
      COMPILER=GCC \
      BLAS_LIBRARY=OPENBLAS \
      BLASINC= \
      BLASLIBS="$BLCO_BLASLIBS" \
      cpd64
  elif [[ "$RUN_BLCO" == "1" ]]; then
    echo "[skip] existing BLCO binary: $BLCO_BIN"
  else
    echo "[skip] BLCO build disabled"
  fi
}

needs_tns() {
  [[ "$RUN_CUTC_SGD" == "1" || "$RUN_CUTC_CCD" == "1" || "$RUN_CUTC_ALS" == "1" || "$RUN_BLCO" == "1" ]]
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
  local raw="${out%.csv}.raw.log"
  rm -f "$tmp"
  rm -f "$raw"
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
    CUTC_MAX_ITERATE="$CUTC_MAX_ITERATE" \
    CUTC_MAX_BADEPOCHS="$CUTC_MAX_BADEPOCHS" \
    CUTC_TOLERANCE="$CUTC_TOLERANCE" \
    stdbuf -oL -eL ./tc "$train" "$train" "$train" "$val" "$test"
  ) | tee "$raw" | python3 "$CUTC_LOG_TO_CSV" --input - --output "$tmp" >/dev/null
  mv "$tmp" "$out"
}

run_cutc() {
  local ds="$1"
  local rate="$2"
  local rate_tag="${rate/./p}"
  local train="$TNS_ROOT/$ds/${ds}_train.tns"
  local val="$TNS_ROOT/$ds/${ds}_val.tns"
  local test="$TNS_ROOT/$ds/${ds}_test.tns"
  if [[ "$RUN_CUTC_SGD" == "1" ]]; then
    run_cutc_one "sgd" "CUTC-sgd" "$OUT_ROOT/CUTC-sgd/${ds}_norm_r8_lr001_s${rate_tag}.csv" "$train" "$val" "$test"
  fi
  if [[ "$RUN_CUTC_CCD" == "1" ]]; then
    run_cutc_one "ccd" "CUTC-ccd" "$OUT_ROOT/CUTC-ccd/${ds}_norm_r8_lr001_ccd_s${rate_tag}.csv" "$train" "$val" "$test"
  else
    echo "[skip] CUTC-ccd disabled for $ds"
  fi
  if [[ "$RUN_CUTC_ALS" == "1" ]]; then
    run_cutc_one "als" "CUTC-als" "$OUT_ROOT/CUTC-als/${ds}_norm_r8_als_s${rate_tag}.csv" "$train" "$val" "$test"
  else
    echo "[skip] CUTC-als disabled for $ds"
  fi
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
  BLCO_MAX_BADEPOCHS="$BLCO_MAX_BADEPOCHS" \
  LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/opt/anaconda3/envs/zhangsac/lib:${LD_LIBRARY_PATH:-}" \
  "$BLCO_BIN" \
    --train-file "$TNS_ROOT/$ds/${ds}_train.tns" \
    --val-file "$TNS_ROOT/$ds/${ds}_val.tns" \
    --test-file "$TNS_ROOT/$ds/${ds}_test.tns" \
    --dims "$(shape_for "$ds")" \
    --rank 8 \
    --max-iter "$BLCO_MAX_ITER" \
    --epsilon "$BLCO_TOLERANCE" \
    --kernel-id 10 \
    --device "$GPU_DEVICE" \
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
      --maxiters "$GENTEN_MAXITERS" \
      --frozeniters "$GENTEN_FROZENITERS" \
      --rmse-stop-set val \
      --rmse-patience "$GENTEN_PATIENCE" \
      --rmse-min-improve "$GENTEN_TOLERANCE" \
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
    --epochs "$COSTCO_EPOCHS" \
    --patience "$COSTCO_PATIENCE" \
    --tolerance "$COSTCO_TOLERANCE" \
    --lr 1e-4 \
    --batch-size 262144 \
    --eval-batch-size 262144
  mv "$tmp" "$out"
}

main() {
  echo "[config] REPOS=$REPOS"
  echo "[config] ROOT=$ROOT"
  echo "[config] TNS_ROOT=$TNS_ROOT"
  echo "[config] OUT_ROOT=$OUT_ROOT"
  echo "[config] WORK_TMP=$WORK_TMP"
  echo "[config] DATASETS=${DATASETS[*]}"
  echo "[config] RUN_CUTC_SGD=$RUN_CUTC_SGD RUN_CUTC_CCD=$RUN_CUTC_CCD RUN_CUTC_ALS=$RUN_CUTC_ALS RUN_BLCO=$RUN_BLCO RUN_GENTEN=$RUN_GENTEN RUN_COSTCO=$RUN_COSTCO DRY_RUN=$DRY_RUN"
  echo "[config] stop CUTC=max:$CUTC_MAX_ITERATE patience:$CUTC_MAX_BADEPOCHS tol:$CUTC_TOLERANCE BLCO=max:$BLCO_MAX_ITER patience:$BLCO_MAX_BADEPOCHS tol:$BLCO_TOLERANCE GenTen=max:$GENTEN_MAXITERS patience:$GENTEN_PATIENCE tol:$GENTEN_TOLERANCE frozen:$GENTEN_FROZENITERS CoSTCo=max:$COSTCO_EPOCHS patience:$COSTCO_PATIENCE tol:$COSTCO_TOLERANCE"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] configuration printed; no build, export, or training started"
    return 0
  fi
  ensure_dirs
  build_bins
  if needs_tns; then
    export_tns
  else
    echo "[skip] TNS export not needed"
  fi
  for ds in "${DATASETS[@]}"; do
    local rate
    rate="$(rate_for "$ds")"
    echo "[dataset] $ds rate=$rate"
    run_cutc "$ds" "$rate"
    if [[ "$RUN_BLCO" == "1" ]]; then
      run_blco "$ds" "$rate"
    else
      echo "[skip] BLCO disabled for $ds"
    fi
    if [[ "$RUN_GENTEN" == "1" ]]; then
      run_genten "$ds" "$rate"
    else
      echo "[skip] GenTen disabled for $ds"
    fi
    if [[ "$RUN_COSTCO" == "1" ]]; then
      run_costco "$ds" "$rate"
    else
      echo "[skip] CoSTCo disabled for $ds"
    fi
  done
  echo "[done] outputs: $OUT_ROOT"
}

main "$@"
