#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

import run_genten_epoch_metrics as base


CSV_FIELDS = [
    "epoch",
    "t_total_s",
    "train_gpu",
    "train_rmse",
    "train_mae",
    "train_er",
    "val_rmse",
    "val_mae",
    "val_er",
    "u_rmse",
    "u_mae",
    "u_er",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GenTen on sparse TNS and export cuTC-format CSV.")
    p.add_argument("--tns-file", type=pathlib.Path, default=None)
    p.add_argument("--train-npz", type=pathlib.Path, default=None)
    p.add_argument("--val-npz", type=pathlib.Path, default=None)
    p.add_argument("--test-npz", type=pathlib.Path, default=None)
    p.add_argument("--genten-bin", type=pathlib.Path, required=True)
    p.add_argument("--logs-root", type=pathlib.Path, default=pathlib.Path("/data/project/lianghan/work/logs/GenTen"))
    p.add_argument("--dataset-tag", required=True)
    p.add_argument("--output-file", type=pathlib.Path, default=None)
    p.add_argument("--sample-rate", type=base.parse_rate, required=True)
    p.add_argument("--validation-rate", type=base.parse_rate, default=0.05)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--maxiters", type=int, default=150)
    p.add_argument("--start-epoch", type=int, default=1)
    p.add_argument("--initial-file", type=pathlib.Path, default=None)
    p.add_argument("--append", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--omega-seed", type=int, default=2025)
    p.add_argument("--val-seed", type=int, default=20250101)
    p.add_argument("--exec-space", default="cuda")
    p.add_argument("--sampling", default="stratified")
    p.add_argument("--rate", type=float, default=5.0e-3)
    p.add_argument("--decay", type=float, default=5.0e-1)
    p.add_argument("--gcp-step", default="adam", choices=("sgd", "adam", "adagrad", "amsgrad", "sgd-momentum", "demon"))
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--adam-eps", type=float, default=1.0e-8)
    p.add_argument("--gcp-tol", type=float, default=1.0e-4)
    p.add_argument("--fails", type=int, default=10)
    p.add_argument("--printitn", type=int, default=0)
    p.add_argument("--fuse-sa", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--keep-artifacts", action="store_true")
    p.add_argument("--work-dir", type=pathlib.Path, default=None)
    return p.parse_args()


def load_npz(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    data = np.load(path)
    coords = np.stack([data["i"], data["j"], data["k"]], axis=1).astype(np.int64, copy=False)
    vals = data["v"].astype(np.float64, copy=False)
    dims = tuple(int(x) for x in data["shape"])
    return coords, vals, dims


def write_tns_arrays(
    coords: np.ndarray,
    vals: np.ndarray,
    path: pathlib.Path,
    dims: tuple[int, ...] | None = None,
) -> int:
    with path.open("w", encoding="utf-8") as f:
        for start in range(0, vals.shape[0], 1_000_000):
            end = min(start + 1_000_000, vals.shape[0])
            for c, v in zip(coords[start:end], vals[start:end]):
                f.write(f"{int(c[0])} {int(c[1])} {int(c[2])} {float(v):.17e}\n")
        if dims is not None:
            f.write(f"{dims[0] - 1} {dims[1] - 1} {dims[2] - 1} 0.0\n")
    return int(vals.shape[0])


def load_tns(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    coords = []
    vals = []
    min_coord = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            c = [int(parts[0]), int(parts[1]), int(parts[2])]
            min_coord = min(c) if min_coord is None else min(min_coord, min(c))
            coords.append(c)
            vals.append(float(parts[3]))
    if not coords:
        raise RuntimeError(f"no TNS rows found in {path}")
    arr = np.asarray(coords, dtype=np.int64)
    if min_coord == 1:
        arr -= 1
    values = np.asarray(vals, dtype=np.float64)
    dims = tuple(int(arr[:, m].max()) + 1 for m in range(arr.shape[1]))
    return arr, values, dims


def split_rows(n: int, validation_rate: float, sample_rate: float, val_seed: int, omega_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng_val = np.random.Generator(np.random.MT19937(val_seed))
    rng_omega = np.random.Generator(np.random.MT19937(omega_seed))
    val = rng_val.random(n) < validation_rate
    pool = ~val
    omega = np.zeros(n, dtype=bool)
    omega[pool] = rng_omega.random(int(pool.sum())) < sample_rate
    u = pool & (~omega)
    return omega, val, u


def write_tns(
    coords: np.ndarray,
    vals: np.ndarray,
    mask: np.ndarray,
    path: pathlib.Path,
    dims: tuple[int, ...] | None = None,
) -> int:
    idx = np.flatnonzero(mask)
    with path.open("w", encoding="utf-8") as f:
        for n in idx:
            c = coords[n]
            f.write(f"{int(c[0])} {int(c[1])} {int(c[2])} {float(vals[n]):.17e}\n")
        if dims is not None:
            f.write(f"{dims[0] - 1} {dims[1] - 1} {dims[2] - 1} 0.0\n")
    return int(idx.size)


def predict_at(coords: np.ndarray, weights: np.ndarray, factors: list[np.ndarray], chunk: int = 1_000_000) -> np.ndarray:
    pred = np.empty(coords.shape[0], dtype=np.float64)
    for start in range(0, coords.shape[0], chunk):
        end = min(start + chunk, coords.shape[0])
        c = coords[start:end]
        out = np.zeros(end - start, dtype=np.float64)
        for r, w in enumerate(weights):
            term = np.full(end - start, w, dtype=np.float64)
            for m, fac in enumerate(factors):
                term *= fac[c[:, m], r]
            out += term
        pred[start:end] = out
    return pred


def read_ktensor_stream(path: pathlib.Path) -> tuple[np.ndarray, list[np.ndarray]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        def next_nonempty() -> str:
            for line in f:
                line = line.strip()
                if line:
                    return line
            raise RuntimeError(f"unexpected EOF in {path}")

        if next_nonempty() != "ktensor":
            raise RuntimeError(f"{path} is not a ktensor file")
        ndims = int(next_nonempty())
        _dims = [int(x) for x in next_nonempty().split()]
        rank = int(next_nonempty())
        weights = np.fromstring(next_nonempty(), sep=" ", dtype=np.float64)
        if weights.size != rank:
            raise RuntimeError(f"weights size mismatch in {path}: {weights.size} != {rank}")

        factors: list[np.ndarray] = []
        for _ in range(ndims):
            if next_nonempty() != "matrix":
                raise RuntimeError(f"expected matrix header in {path}")
            mat_ndims = int(next_nonempty())
            if mat_ndims != 2:
                raise RuntimeError(f"expected matrix ndim=2 in {path}, got {mat_ndims}")
            nrows, ncols = map(int, next_nonempty().split())
            mat = np.empty((nrows, ncols), dtype=np.float64)
            for row_idx in range(nrows):
                row = np.fromstring(next_nonempty(), sep=" ", dtype=np.float64)
                if row.size != ncols:
                    raise RuntimeError(f"bad matrix row length in {path} at row {row_idx}")
                mat[row_idx, :] = row
            factors.append(mat)
    return weights, factors


def metric(vals: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return float("nan"), float("nan"), float("nan")
    diff = vals[idx] - pred[idx]
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mae = float(np.mean(np.abs(diff)))
    denom = float(np.sum(vals[idx] * vals[idx]))
    er = float(np.sqrt(np.sum(diff * diff) / denom)) if denom != 0.0 else float("nan")
    return er, mae, rmse


def metric_all(vals: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    diff = vals - pred
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mae = float(np.mean(np.abs(diff)))
    denom = float(np.sum(vals * vals))
    er = float(np.sqrt(np.sum(diff * diff) / denom)) if denom != 0.0 else float("nan")
    return er, mae, rmse


def main() -> int:
    args = parse_args()
    genten_bin = args.genten_bin.resolve()
    dataset_tag = args.dataset_tag
    output_dir = args.logs_root.resolve() / f"{dataset_tag}-genten"
    output_dir.mkdir(parents=True, exist_ok=True)
    rate_code = base.sample_rate_code(args.sample_rate)
    csv_path = args.output_file or output_dir / f"{dataset_tag}_r8_genten_s{str(args.sample_rate).replace('.', 'p')}.csv"
    if not csv_path.is_absolute():
        csv_path = output_dir / csv_path

    use_npz_splits = args.train_npz is not None or args.val_npz is not None or args.test_npz is not None
    if use_npz_splits:
        if args.train_npz is None or args.val_npz is None or args.test_npz is None:
            raise RuntimeError("--train-npz, --val-npz, and --test-npz must be provided together")
        train_coords, train_vals, dims = load_npz(args.train_npz.resolve())
        val_coords, val_vals, val_dims = load_npz(args.val_npz.resolve())
        u_coords, u_vals, u_dims = load_npz(args.test_npz.resolve())
        if val_dims != dims or u_dims != dims:
            raise RuntimeError("split shapes do not match")
    else:
        if args.tns_file is None:
            raise RuntimeError("--tns-file is required unless split npz files are provided")
        tns_file = args.tns_file.resolve()
        coords, vals, dims = load_tns(tns_file)
        omega, val, u = split_rows(coords.shape[0], args.validation_rate, args.sample_rate, args.val_seed, args.omega_seed)
        if int(omega.sum()) == 0:
            raise RuntimeError("Omega split is empty")

    tmp_parent = args.work_dir.resolve() if args.work_dir else None
    if tmp_parent is not None:
        tmp_parent.mkdir(parents=True, exist_ok=True)
    tmp_ctx = tempfile.TemporaryDirectory(prefix=f"{dataset_tag}_{rate_code}_", dir=str(tmp_parent) if tmp_parent else None)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_ctx as td:
        work = pathlib.Path(td)
        train_path = work / f"{dataset_tag}_{rate_code}.omega.tns"
        if use_npz_splits:
            if train_vals.shape[0] == 0:
                raise RuntimeError("train split is empty")
            write_tns_arrays(train_coords, train_vals, train_path, dims=dims)
        else:
            write_tns(coords, vals, omega, train_path, dims=dims)
        prev_ktns = args.initial_file.resolve() if args.initial_file is not None else None
        total_time = 0.0
        total_gpu = 0.0
        if args.append and csv_path.exists():
            with csv_path.open("r", encoding="utf-8", errors="ignore") as old_csv:
                rows = [line.strip().split(",") for line in old_csv if line.strip()]
            if len(rows) > 1:
                last = rows[-1]
                total_time = float(last[1])
                total_gpu = float(last[2])
        mode = "a" if args.append and csv_path.exists() else "w"
        with csv_path.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if mode == "w":
                writer.writeheader()
            for epoch in range(args.start_epoch, args.maxiters + 1):
                hist_path = work / f"{dataset_tag}_{rate_code}_ep{epoch}_history.tsv"
                ktns_path = work / f"{dataset_tag}_{rate_code}_ep{epoch}.ktns"
                runlog_path = work / f"{dataset_tag}_{rate_code}_ep{epoch}.log"
                cmd = base.build_genten_cmd(
                    genten_bin=genten_bin,
                    exec_space=args.exec_space,
                    train_path=train_path,
                    rank=args.rank,
                    maxiters=1,
                    epochiters=1,
                    frozeniters=1,
                    sampling=args.sampling,
                    rate=args.rate,
                    decay=args.decay,
                    gcp_step=args.gcp_step,
                    adam_beta1=args.adam_beta1,
                    adam_beta2=args.adam_beta2,
                    adam_eps=args.adam_eps,
                    fuse_sa=args.fuse_sa,
                    seed=args.seed,
                    gcp_tol=args.gcp_tol,
                    fails=args.fails,
                    printitn=args.printitn,
                    hist_path=hist_path,
                    ktns_path=ktns_path,
                    initial_file=prev_ktns,
                )
                cmd.extend(["--dims", "[" + ",".join(str(d) for d in dims) + "]"])
                with runlog_path.open("w", encoding="utf-8", errors="ignore") as runlog:
                    proc = subprocess.run(cmd, stdout=runlog, stderr=subprocess.STDOUT, text=True)
                if proc.returncode != 0:
                    print(runlog_path.read_text(errors="ignore")[-4000:], file=sys.stderr)
                    raise subprocess.CalledProcessError(proc.returncode, cmd)
                _local_epoch, _fest, step_time, step_gpu = base.parse_last_history_row(hist_path)
                total_time += step_time
                total_gpu += step_gpu
                weights, factors = read_ktensor_stream(ktns_path)
                if use_npz_splits:
                    train_pred = predict_at(train_coords, weights, factors)
                    val_pred = predict_at(val_coords, weights, factors)
                    u_pred = predict_at(u_coords, weights, factors)
                    train_er, train_mae, train_rmse = metric_all(train_vals, train_pred)
                    val_er, val_mae, val_rmse = metric_all(val_vals, val_pred)
                    u_er, u_mae, u_rmse = metric_all(u_vals, u_pred)
                    del train_pred, val_pred, u_pred
                else:
                    pred = predict_at(coords, weights, factors)
                    train_er, train_mae, train_rmse = metric(vals, pred, omega)
                    val_er, val_mae, val_rmse = metric(vals, pred, val)
                    u_er, u_mae, u_rmse = metric(vals, pred, u)
                writer.writerow(
                    {
                        "epoch": epoch,
                        "t_total_s": base.fmt(total_time),
                        "train_gpu": base.fmt(total_gpu),
                        "train_rmse": base.fmt(train_rmse),
                        "train_mae": base.fmt(train_mae),
                        "train_er": base.fmt(train_er),
                        "val_rmse": base.fmt(val_rmse),
                        "val_mae": base.fmt(val_mae),
                        "val_er": base.fmt(val_er),
                        "u_rmse": base.fmt(u_rmse),
                        "u_mae": base.fmt(u_mae),
                        "u_er": base.fmt(u_er),
                    }
                )
                f.flush()
                if prev_ktns is not None and not args.keep_artifacts:
                    prev_ktns.unlink(missing_ok=True)
                prev_ktns = ktns_path
                if not args.keep_artifacts:
                    hist_path.unlink(missing_ok=True)
                    runlog_path.unlink(missing_ok=True)
            if not args.keep_artifacts and prev_ktns is not None:
                prev_ktns.unlink(missing_ok=True)

    print(f"Done. CSV: {csv_path}")
    print(f"Total rows: {args.maxiters}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
