#!/usr/bin/env python3
"""
Convert normalized 3D .npy tensors into BLCO .tns text format.

Input format assumption:
- Dense 3D numpy array in C-order.
- NaN indicates invalid entries and will be skipped.

Output .tns format:
- One-based coordinates per mode followed by value.
- One line per valid entry.

Directory layout:
- <output_root>/<alias>/full/<alias>.tns
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


ALIAS_MAP = {
    "CBW_norm": "CBW",
    "ECW_08_norm": "ECW_08",
    "GeantStore_GeantDataNorm1_f32_norm": "GS",
    "pmse_norm": "pmse",
    "rtdata_webservice_matrix_rtt_f32_norm": "rt",
    "taxi_norm": "taxi",
}


def infer_alias(stem: str) -> str:
    if stem in ALIAS_MAP:
        return ALIAS_MAP[stem]
    if stem.endswith("_norm"):
        stem = stem[: -len("_norm")]
    return stem


def iter_npy_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".npy":
            raise ValueError(f"Expected a .npy file, got: {input_path}")
        yield input_path
        return
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    for p in sorted(input_path.glob("*.npy")):
        if p.is_file():
            yield p


def write_tns_from_npy(npy_path: Path, out_tns: Path) -> dict:
    arr = np.load(npy_path, mmap_mode="r")
    if arr.ndim != 3:
        raise ValueError(f"Only 3D tensors are supported. {npy_path} has ndim={arr.ndim}")

    shape = tuple(int(x) for x in arr.shape)
    i_dim, j_dim, k_dim = shape

    valid_count = 0
    nan_count = 0
    out_tns.parent.mkdir(parents=True, exist_ok=True)

    with out_tns.open("w", encoding="utf-8") as f:
        # C-order flatten: i major then j then k
        for i in range(i_dim):
            plane = arr[i]
            for j in range(j_dim):
                row = plane[j]
                # row is length k_dim
                for k in range(k_dim):
                    v = float(row[k])
                    if np.isnan(v):
                        nan_count += 1
                        continue
                    f.write(f"{i + 1} {j + 1} {k + 1} {v:.9g}\n")
                    valid_count += 1

    return {
        "source_npy": str(npy_path),
        "output_tns": str(out_tns),
        "shape": list(shape),
        "dtype": str(arr.dtype),
        "total_entries": int(i_dim * j_dim * k_dim),
        "valid_entries": int(valid_count),
        "nan_entries": int(nan_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert normalized 3D .npy tensors to .tns.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a .npy file or a directory that contains .npy files.",
    )
    parser.add_argument(
        "--output-root",
        default="/data/project/lianghan/work/data/tensordata/BLCO",
        help="Root directory for converted datasets.",
    )
    parser.add_argument(
        "--alias",
        default="",
        help="Optional alias for single-file conversion. If empty, infer from file name.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    npy_files = list(iter_npy_files(input_path))
    if not npy_files:
        raise SystemExit(f"No .npy files found under: {input_path}")

    for npy_path in npy_files:
        alias = args.alias.strip() if args.alias and len(npy_files) == 1 else infer_alias(npy_path.stem)
        out_dir = output_root / alias / "full"
        out_tns = out_dir / f"{alias}.tns"
        meta_path = out_dir / f"{alias}.meta.json"

        meta = write_tns_from_npy(npy_path, out_tns)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=True)

        print(
            f"[done] {npy_path.name} -> {out_tns} "
            f"(shape={meta['shape']}, valid={meta['valid_entries']}, nan={meta['nan_entries']})"
        )


if __name__ == "__main__":
    main()
