#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PARTS = ("train", "val", "test")


def write_tns(npz_path: Path, out_path: Path) -> int:
    data = np.load(npz_path)
    shape = tuple(int(x) for x in data["shape"])
    arrays = [data["i"], data["j"], data["k"], data["v"]]
    n = int(arrays[3].shape[0])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("# shape " + " ".join(str(x) for x in shape) + "\n")
        for start in range(0, n, 1_000_000):
            end = min(start + 1_000_000, n)
            i = arrays[0][start:end].astype(np.int64, copy=False) + 1
            j = arrays[1][start:end].astype(np.int64, copy=False) + 1
            k = arrays[2][start:end].astype(np.int64, copy=False) + 1
            v = arrays[3][start:end].astype(np.float64, copy=False)
            lines = (
                f"{int(a)} {int(b)} {int(c)} {float(d):.17e}\n"
                for a, b, c, d in zip(i, j, k, v)
            )
            fh.writelines(lines)
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export prepared common split npz files to 1-based TNS files."
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    out_root = args.output_root.resolve()
    manifest = []
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        name = ds_dir.name
        meta_path = ds_dir / f"{name}_metadata.json"
        rate = None
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
            rate = meta.get("sampling", {}).get("sampling_rate_requested")
        for part in PARTS:
            npz_path = ds_dir / f"{name}_{part}.npz"
            if not npz_path.exists():
                continue
            out_path = out_root / name / f"{name}_{part}.tns"
            n = write_tns(npz_path, out_path)
            manifest.append(
                {
                    "dataset": name,
                    "part": part,
                    "sampling_rate": rate,
                    "nnz": n,
                    "path": str(out_path),
                }
            )
            print(f"[write] {out_path} nnz={n}")

    manifest_path = out_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[done] manifest={manifest_path}")


if __name__ == "__main__":
    main()
