# A40 Deployment Notes

This repository contains the experiment code used for the common-split runs:

- `cuTC/`
- `exp/blocked-linearized-coordinate/` for BLCO
- `KDD19-CoSTCo/`
- `GenTen/`
- top-level runner scripts for `newlog1` and `newlog2`

The runner scripts now use their own checkout directory as `REPOS` by default.
Dataset and output locations are configurable through environment variables:

```bash
ROOT=/path/to/prepared_common_splits
TNS_ROOT=/path/to/prepared_common_splits_tns
OUT_ROOT=/path/to/newlog2
WORK_TMP=/path/to/tmp
DATASETS=tpdata
```

Use these variables with:

```bash
run_newlog1_common_splits.sh
run_newlog2_common_splits.sh
resume_newlog2_when_gpu_free.sh
```

Before running GenTen experiments on a fresh machine, build the CUDA binary so
this path exists:

```bash
GenTen/build/cuda/bin/genten
```

`run_newlog2_common_splits.sh` will build cuTC and BLCO if needed, and will
skip completed CSV files in the output directory.

The current `newlog2` configuration intentionally skips cuTC `ccd` and `als`:

```bash
RUN_CUTC_CCD=0
RUN_CUTC_ALS=0
```

For the current `newlog2` resume workflow, see:

```bash
A40_PULL_AND_RESUME.md
```
