#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
python normalize_tensor_npy.py /data/project/lianghan/work/data/testdata/ECW_08/ECW_08.npy /data/project/lianghan/work/data/testdata/data/testdata/ECW_08/ECW_08_norm.npy --mode 2 --force-c-order

python normalize_tensor_npy.py input.npy output_norm.npy --mode 1 --force-c-order


'''

import argparse
import numpy as np


def make_valid_mask_from_nan(x: np.ndarray) -> np.ndarray:
    # C++: ok = (v == v)  <=> not NaN
    return np.isfinite(x)  # 更强：把 inf 也当无效；如果你要“只排除NaN”，用 ~np.isnan(x)


def print_valid_minmax_maxabs(x: np.ndarray, valid: np.ndarray, tag: str):
    xv = x[valid]
    if xv.size == 0:
        print(f"[stats] {tag}: valid count=0 (all NaN/Inf?)")
        return
    mn = float(np.min(xv))
    mx = float(np.max(xv))
    maxabs = float(np.max(np.abs(xv)))
    print(f"[stats] {tag}: n={xv.size} min={mn:.6f} max={mx:.6f} maxabs={maxabs:.6f}")


def normalize_maxabs_like_cpp(x: np.ndarray, valid: np.ndarray) -> float:
    # C++:
    # maxabs = max(|x[i]|) over valid
    # if (maxabs <= 0) maxabs = 1
    # x[i] /= maxabs for valid only
    xv = x[valid]
    if xv.size == 0:
        return 1.0
    maxabs = float(np.max(np.abs(xv)))
    if not (maxabs > 0.0):
        maxabs = 1.0
    x[valid] = x[valid] / maxabs
    print(f"[normalize] MaxAbs maxabs={maxabs:.6f} (range roughly [-1,1])")
    return maxabs


def normalize_minmax01_like_cpp(x: np.ndarray, valid: np.ndarray):
    # C++:
    # mn = min(x) over valid, mx = max(x) over valid
    # den = mx - mn; if (!(den > 0)) den = 1
    # x = (x - mn) / den for valid only
    xv = x[valid]
    if xv.size == 0:
        print("[normalize] MinMax01 skipped (no valid)")
        return 0.0, 1.0
    mn = float(np.min(xv))
    mx = float(np.max(xv))
    den = mx - mn
    if not (den > 0.0):
        den = 1.0
    x[valid] = (x[valid] - mn) / den
    print(f"[normalize] MinMax01 min={mn:.6f} max={mx:.6f} (range [0,1])")
    return mn, mx


def main():
    ap = argparse.ArgumentParser(description="Normalize 3D tensor .npy exactly like the provided main.cpp")
    ap.add_argument("input", type=str, help="input .npy (3D)")
    ap.add_argument("output", type=str, help="output .npy")
    ap.add_argument("--mode", type=int, default=2, choices=[0, 1, 2],
                    help="0=none, 1=maxabs([-1,1]), 2=minmax01([0,1]) (default: 2)")
    ap.add_argument("--keep-nan-only", action="store_true",
                    help="Treat +inf/-inf as valid (match v==v strictly). Default treats inf as invalid.")
    ap.add_argument("--force-c-order", action="store_true",
                    help="Save output as C-contiguous (recommended for your decode_idx_host).")
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"],
                    help="output dtype (default: float32)")
    args = ap.parse_args()

    x = np.load(args.input, allow_pickle=False)
    if x.ndim != 3:
        raise SystemExit(f"expected 3D array, got shape={x.shape}")

    # Ensure float array
    x = x.astype(np.float32 if args.dtype == "float32" else np.float64, copy=True)

    # valid mask
    if args.keep_nan_only:
        valid = ~np.isnan(x)   # 精确对应 C++ 的 v==v
    else:
        valid = np.isfinite(x) # 更安全（把 inf 当无效）

    print(f"[npy] shape={x.shape} dtype={x.dtype} fortran_order={x.flags['F_CONTIGUOUS'] and not x.flags['C_CONTIGUOUS']}")
    print_valid_minmax_maxabs(x, valid, "raw(valid)")

    if args.mode == 0:
        print("[normalize] none")
    elif args.mode == 1:
        normalize_maxabs_like_cpp(x, valid)
    elif args.mode == 2:
        normalize_minmax01_like_cpp(x, valid)

    print_valid_minmax_maxabs(x, valid, "after(valid)")

    # Save: strongly recommend C-order to match decode_idx_host (i,j,k from linear idx)
    if args.force_c_order:
        x = np.ascontiguousarray(x)
    np.save(args.output, x)
    print(f"[out] saved to {args.output} (C_contig={x.flags['C_CONTIGUOUS']}, F_contig={x.flags['F_CONTIGUOUS']})")


if __name__ == "__main__":
    main()
