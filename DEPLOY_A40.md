# A40 Deployment Notes

This repository contains the comparison experiment code for:

- `cuTC/`
- `exp/blocked-linearized-coordinate/` for BLCO
- `KDD19-CoSTCo/`
- `GenTen/`
- top-level runners for `newlog1` and `newlog2`

Use the latest strict GPU timing commit:

```bash
git pull
git rev-parse --short HEAD
# expected: 7002c10 or newer
```

## Timing Definition

For current `newlog2` runs, `train_gpu` strictly counts GPU training work only.
It excludes:

- H2D and D2H data transfer
- GPU memset/init
- objective/fit computation
- train/val/test metric evaluation
- CPU-side data loading, logging, and CSV parsing

Old `newlog2` CSV snapshots use a different timing definition. Do not restore
`artifacts/newlog2_csv_snapshot` into a strict-timing output directory.

## Configurable Paths

Runner scripts use their own checkout directory as `REPOS` by default. Dataset,
TNS, output, and temp paths can be configured:

```bash
ROOT=/path/to/prepared_common_splits
TNS_ROOT=/path/to/prepared_common_splits_tns
OUT_ROOT=/path/to/newlog2
WORK_TMP=/path/to/tmp
DATASETS="DARPA LANL2 BJTaxi tpdata"
```

Use these variables with:

```bash
./run_newlog1_common_splits.sh
./run_newlog2_common_splits.sh
./resume_newlog2_when_gpu_free.sh
```

## Build Notes

`run_newlog2_common_splits.sh` builds cuTC and BLCO when their methods are
enabled. GenTen must have this binary before running:

```bash
GenTen/build/cuda/bin/genten
```

Typical GenTen build:

```bash
cd GenTen
mkdir -p build/cuda
cd build/cuda
cmake ../.. -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_CUDA=ON
make -j
```

## Current newlog2 Scope

Current strict-timing `newlog2` should be regenerated from an empty `OUT_ROOT`.
Run:

- `CUTC-sgd`
- `BLCO`
- `GenTen`
- `CoSTCo`

for:

- `DARPA`
- `LANL2`
- `BJTaxi`
- `tpdata`

`CUTC-ccd` and `CUTC-als` remain disabled by default:

```bash
RUN_CUTC_CCD=0
RUN_CUTC_ALS=0
```

Full workflow:

```bash
A40_PULL_AND_RESUME.md
```
