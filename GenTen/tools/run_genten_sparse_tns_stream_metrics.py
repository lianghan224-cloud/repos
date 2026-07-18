#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import pathlib
import subprocess
import sys
import tempfile
from typing import Iterator

import numpy as np

import run_genten_epoch_metrics as base
from run_genten_sparse_tns_epoch_metrics import CSV_FIELDS, read_ktensor_stream


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GenTen on large sparse TNS with streaming metrics.")
    p.add_argument("--tns-file", type=pathlib.Path, required=True)
    p.add_argument("--genten-bin", type=pathlib.Path, required=True)
    p.add_argument("--logs-root", type=pathlib.Path, default=pathlib.Path("/data/project/lianghan/work/logs/GenTen"))
    p.add_argument("--dataset-tag", required=True)
    p.add_argument("--sample-rate", type=base.parse_rate, required=True)
    p.add_argument("--validation-rate", type=base.parse_rate, default=0.05)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--maxiters", type=int, default=300)
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
    p.add_argument(
        "--frozeniters",
        type=int,
        default=0,
        help="Frozen iterations in each one-step GenTen call. Default 0 so each exported epoch updates parameters.",
    )
    p.add_argument(
        "--rmse-stop-set",
        choices=("off", "train", "val", "u"),
        default="val",
        help="RMSE set used for cuTC-style early stop.",
    )
    p.add_argument(
        "--rmse-patience",
        type=int,
        default=20,
        help="Stop after this many epochs without the requested RMSE improvement.",
    )
    p.add_argument(
        "--rmse-min-improve",
        type=float,
        default=1.0e-4,
        help="Minimum RMSE decrease required to reset early-stop patience.",
    )
    p.add_argument("--printitn", type=int, default=0)
    p.add_argument("--fuse-sa", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--work-dir", type=pathlib.Path, default=None)
    p.add_argument("--keep-artifacts", action="store_true")
    return p.parse_args()


def iter_tns(path: pathlib.Path) -> Iterator[tuple[int, int, int, float]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            yield int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])


def scan_dims(path: pathlib.Path) -> tuple[tuple[int, int, int], int, int]:
    mins = [10**30, 10**30, 10**30]
    maxs = [-1, -1, -1]
    n = 0
    for i, j, k, _v in iter_tns(path):
        vals = (i, j, k)
        for m, x in enumerate(vals):
            mins[m] = min(mins[m], x)
            maxs[m] = max(maxs[m], x)
        n += 1
    if n == 0:
        raise RuntimeError(f"no TNS rows found in {path}")
    base_idx = 1 if min(mins) == 1 else 0
    dims = tuple(x - base_idx + 1 for x in maxs)
    return dims, base_idx, n


def mask_stream(n: int, val_rate: float, sample_rate: float, val_seed: int, omega_seed: int) -> Iterator[str]:
    rng_val = np.random.Generator(np.random.MT19937(val_seed))
    rng_omega = np.random.Generator(np.random.MT19937(omega_seed))
    for _ in range(n):
        if rng_val.random() < val_rate:
            yield "val"
        elif rng_omega.random() < sample_rate:
            yield "omega"
        else:
            yield "u"


def write_omega(path: pathlib.Path, out: pathlib.Path, dims: tuple[int, int, int], base_idx: int, n: int, args: argparse.Namespace) -> int:
    count = 0
    masks = mask_stream(n, args.validation_rate, args.sample_rate, args.val_seed, args.omega_seed)
    with out.open("w", encoding="utf-8") as f:
        for (i, j, k, v), split in zip(iter_tns(path), masks):
            if split == "omega":
                f.write(f"{i - base_idx} {j - base_idx} {k - base_idx} {v:.17e}\n")
                count += 1
        f.write(f"{dims[0] - 1} {dims[1] - 1} {dims[2] - 1} 0.0\n")
    return count


def compute_metrics_stream(
    path: pathlib.Path,
    dims: tuple[int, int, int],
    base_idx: int,
    n: int,
    weights: np.ndarray,
    factors: list[np.ndarray],
    args: argparse.Namespace,
    chunk_size: int = 1_000_000,
) -> dict[str, tuple[float, float, float]]:
    sums = {name: [0, 0.0, 0.0, 0.0] for name in ("omega", "val", "u")}
    masks = mask_stream(n, args.validation_rate, args.sample_rate, args.val_seed, args.omega_seed)
    coords: list[tuple[int, int, int]] = []
    vals: list[float] = []
    splits: list[str] = []

    def flush() -> None:
        if not coords:
            return
        c = np.asarray(coords, dtype=np.int64)
        v = np.asarray(vals, dtype=np.float64)
        pred = np.zeros(v.shape[0], dtype=np.float64)
        for r, w in enumerate(weights):
            pred += w * factors[0][c[:, 0], r] * factors[1][c[:, 1], r] * factors[2][c[:, 2], r]
        diff = v - pred
        for split_name in ("omega", "val", "u"):
            idx = np.asarray([s == split_name for s in splits], dtype=bool)
            if not idx.any():
                continue
            local_diff = diff[idx]
            local_v = v[idx]
            sums[split_name][0] += int(idx.sum())
            sums[split_name][1] += float(np.sum(local_diff * local_diff))
            sums[split_name][2] += float(np.sum(np.abs(local_diff)))
            sums[split_name][3] += float(np.sum(local_v * local_v))
        coords.clear()
        vals.clear()
        splits.clear()

    for (i, j, k, v), split in zip(iter_tns(path), masks):
        coords.append((i - base_idx, j - base_idx, k - base_idx))
        vals.append(v)
        splits.append(split)
        if len(coords) >= chunk_size:
            flush()
    flush()

    out: dict[str, tuple[float, float, float]] = {}
    for name, (cnt, sse, sae, denom) in sums.items():
        if cnt == 0:
            out[name] = (float("nan"), float("nan"), float("nan"))
        else:
            er = math.sqrt(sse / denom) if denom > 0.0 else float("nan")
            mae = sae / cnt
            rmse = math.sqrt(sse / cnt)
            out[name] = (er, mae, rmse)
    return out


def main() -> int:
    args = parse_args()
    tns = args.tns_file.resolve()
    dims, base_idx, n = scan_dims(tns)
    tag = args.dataset_tag
    rate_code = base.sample_rate_code(args.sample_rate)
    output_dir = args.logs_root.resolve() / f"{tag}-genten"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{tag}_r8_genten_s{str(args.sample_rate).replace('.', 'p')}.csv"
    tmp_parent = args.work_dir.resolve() if args.work_dir else None
    if tmp_parent is not None:
        tmp_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{tag}_{rate_code}_", dir=str(tmp_parent) if tmp_parent else None) as td:
        work = pathlib.Path(td)
        train_path = work / f"{tag}_{rate_code}.omega.tns"
        omega_count = write_omega(tns, train_path, dims, base_idx, n, args)
        if omega_count == 0:
            raise RuntimeError("Omega split is empty")

        prev_ktns = None
        total_time = 0.0
        total_gpu = 0.0
        best_stop_rmse = float("inf")
        rmse_no_improve = 0
        rows_written = 0
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for epoch in range(1, args.maxiters + 1):
                hist_path = work / f"{tag}_{rate_code}_ep{epoch}_history.tsv"
                ktns_path = work / f"{tag}_{rate_code}_ep{epoch}.ktns"
                runlog_path = work / f"{tag}_{rate_code}_ep{epoch}.log"
                cmd = base.build_genten_cmd(
                    genten_bin=args.genten_bin.resolve(),
                    exec_space=args.exec_space,
                    train_path=train_path,
                    rank=args.rank,
                    maxiters=1,
                    epochiters=1,
                    frozeniters=args.frozeniters,
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
                metrics = compute_metrics_stream(tns, dims, base_idx, n, weights, factors, args)
                train_er, train_mae, train_rmse = metrics["omega"]
                val_er, val_mae, val_rmse = metrics["val"]
                u_er, u_mae, u_rmse = metrics["u"]
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
                rows_written += 1
                f.flush()
                if prev_ktns is not None and not args.keep_artifacts:
                    prev_ktns.unlink(missing_ok=True)
                prev_ktns = ktns_path
                if not args.keep_artifacts:
                    hist_path.unlink(missing_ok=True)
                    runlog_path.unlink(missing_ok=True)

                if args.rmse_stop_set == "train":
                    tracked_rmse = train_rmse
                elif args.rmse_stop_set == "val":
                    tracked_rmse = val_rmse
                elif args.rmse_stop_set == "u":
                    tracked_rmse = u_rmse
                else:
                    tracked_rmse = float("nan")

                if math.isfinite(tracked_rmse):
                    if tracked_rmse < best_stop_rmse - args.rmse_min_improve:
                        best_stop_rmse = tracked_rmse
                        rmse_no_improve = 0
                    else:
                        rmse_no_improve += 1
                    if args.rmse_stop_set != "off" and rmse_no_improve >= args.rmse_patience:
                        print(
                            f"Early stop at epoch {epoch}: {args.rmse_stop_set}_rmse "
                            f"did not improve by {args.rmse_min_improve:g} for "
                            f"{rmse_no_improve} epochs."
                        )
                        break
            if prev_ktns is not None and not args.keep_artifacts:
                prev_ktns.unlink(missing_ok=True)
    print(f"Done. CSV: {csv_path}")
    print(f"Total rows: {rows_written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
