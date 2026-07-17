# A40 Deployment Notes

This repository contains the experiment code used for the common-split runs:

- `cuTC/`
- `exp/blocked-linearized-coordinate/` for BLCO
- `KDD19-CoSTCo/`
- `GenTen/`
- top-level runner scripts for `newlog1` and `newlog2`

The runner scripts currently assume this checkout path:

```bash
/data/project/lianghan/work/repos
```

For a different machine, either clone to that path or edit `ROOT`, `TNS_ROOT`,
`OUT_ROOT`, `REPOS`, and `WORK_TMP` in:

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
