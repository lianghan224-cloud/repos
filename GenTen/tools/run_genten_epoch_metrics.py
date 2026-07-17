#!/usr/bin/env python3
"""
Run GenTen GCP-SGD and export per-epoch CSV metrics in the cuTC-compatible
fixed format:

epoch,t_total_s,train_gpu,
train_rmse,train_mae,train_er,
val_rmse,val_mae,val_er,
u_rmse,u_mae,u_er

Split rule:
1) valid points = non-NaN entries
2) sample validation set V from valid by val_rate and val_seed
3) train_pool = valid \\ V
4) sample Omega from train_pool by sampling_rate and omega_seed
5) U = train_pool \\ Omega

Training rule:
- Only Omega participates in training objective.
- V and U are for evaluation only.
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import subprocess
import sys
import tempfile
from typing import Sequence

import numpy as np

DEFAULT_LOGS_ROOT = pathlib.Path("/data/project/lianghan/work/logs")


def parse_rate(value: str) -> float:
    rate = float(value)
    if rate < 0.0 or rate > 1.0:
        raise argparse.ArgumentTypeError("rate must be in [0, 1].")
    return rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GenTen and export fixed-format per-epoch CSV metrics."
    )
    parser.add_argument("--npy-file", type=pathlib.Path, required=True, help="Input .npy tensor.")
    parser.add_argument("--genten-bin", type=pathlib.Path, required=True, help="Path to genten executable.")
    parser.add_argument(
        "--logs-root",
        type=pathlib.Path,
        default=DEFAULT_LOGS_ROOT,
        help="Log root directory (default: /data/project/lianghan/work/logs).",
    )
    parser.add_argument(
        "--dataset-tag",
        type=str,
        default=None,
        help="Dataset short tag used in <logs-root>/XXX-genten/XXXxy.csv (default: auto from dataset name).",
    )
    parser.add_argument(
        "--output-file",
        type=pathlib.Path,
        default=None,
        help=(
            "Optional output CSV file name/path. Relative paths are created under "
            "<logs-root>/XXX-genten/. Default is XXXxy.csv."
        ),
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Prefix for internal intermediate files (default: derived from naming rule).",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32", help="Reserved.")
    parser.add_argument("--sample-rate", type=parse_rate, required=True, help="Sampling ratio for Omega.")
    parser.add_argument(
        "--validation-rate",
        type=parse_rate,
        default=0.05,
        help="Validation ratio from valid points (default: 0.05).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for GenTen training (passed to --seed/--gcp-seed).",
    )
    parser.add_argument(
        "--omega-seed",
        type=int,
        default=2025,
        help="Random seed for Omega sampling split.",
    )
    parser.add_argument(
        "--val-seed",
        type=int,
        default=20250101,
        help="Random seed for validation split V.",
    )
    parser.add_argument(
        "--split-sampling-mode",
        type=int,
        choices=(0, 1),
        default=0,
        help="Split sampling mode for Omega: 0=uniform Bernoulli, 1=value-biased.",
    )
    parser.add_argument(
        "--sampling-alpha",
        type=float,
        default=1.0,
        help="Alpha for value-biased split sampling probability.",
    )
    parser.add_argument(
        "--sampling-eps",
        type=float,
        default=1.0e-6,
        help="Epsilon for value-biased split sampling probability.",
    )
    parser.add_argument("--exec-space", type=str, default="cuda", help="GenTen execution space.")
    parser.add_argument(
        "--epoch-mode",
        choices=("cumulative", "stepwise"),
        default="stepwise",
        help=(
            "cumulative = rerun from scratch to epoch k each outer epoch; "
            "stepwise = one update per outer epoch with warm-start from previous .ktns."
        ),
    )
    parser.add_argument("--rank", type=int, default=16, help="CP rank.")
    parser.add_argument(
        "--maxiters",
        type=int,
        default=160,
        help="Maximum outer epochs to export.",
    )
    parser.add_argument("--epochiters", type=int, default=50, help="Iterations per epoch (cumulative mode).")
    parser.add_argument("--frozeniters", type=int, default=1, help="Frozen iters (cumulative mode).")
    parser.add_argument(
        "--gcp-tol",
        type=float,
        default=1.0e-4,
        help="GCP-SGD convergence tolerance for auto-stop.",
    )
    parser.add_argument(
        "--fails",
        type=int,
        default=10,
        help="Maximum failed epochs before auto-stop.",
    )
    parser.add_argument(
        "--min-fest-improve",
        type=float,
        default=1.0e-12,
        help=(
            "In stepwise mode, minimum required decrease of fest versus the previous step. "
            "If not improved by at least this amount, count as a failed step."
        ),
    )
    parser.add_argument(
        "--rmse-stop-set",
        choices=("off", "omega", "val", "u"),
        default="val",
        help=(
            "Stepwise early-stop metric set. "
            "'val' tracks val_rmse (default), 'omega' tracks omega_rmse, "
            "'u' tracks u_rmse, 'off' disables RMSE plateau early-stop."
        ),
    )
    parser.add_argument(
        "--rmse-patience",
        type=int,
        default=10,
        help="Stepwise early-stop patience for RMSE plateau.",
    )
    parser.add_argument(
        "--rmse-min-improve",
        type=float,
        default=1.0e-8,
        help=(
            "Minimum RMSE decrease to reset RMSE patience in stepwise mode."
        ),
    )
    parser.add_argument("--sampling", type=str, default="stratified", help="Sampling type for GCP-SGD.")
    parser.add_argument(
        "--rate",
        type=float,
        default=5.0e-3,
        help=(
            "Initial learning rate for GCP-SGD. "
            "Default is tuned for stable stepwise one-step updates."
        ),
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=5.0e-1,
        help=(
            "Learning-rate decay factor on failed epochs. "
            "Default is tuned for stable stepwise one-step updates."
        ),
    )
    parser.add_argument(
        "--gcp-step",
        type=str,
        choices=("sgd", "adam", "adagrad", "amsgrad", "sgd-momentum", "demon"),
        default="adam",
        help="GCP-SGD step type. Default uses non-baseline optimizer path (adam).",
    )
    parser.add_argument(
        "--adam-beta1",
        type=float,
        default=0.9,
        help="Adam beta1 parameter.",
    )
    parser.add_argument(
        "--adam-beta2",
        type=float,
        default=0.999,
        help="Adam beta2 parameter.",
    )
    parser.add_argument(
        "--adam-eps",
        type=float,
        default=1.0e-8,
        help="Adam epsilon parameter.",
    )
    parser.add_argument(
        "--fuse-sa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fused sparse-array gradient path for GCP-SGD (enabled by default).",
    )
    parser.add_argument("--printitn", type=int, default=0, help="Print interval for GenTen.")
    parser.add_argument(
        "--monitor-rmse",
        choices=("auto", "omega", "val", "u"),
        default="auto",
        help=(
            "Stepwise monotonic monitor set: auto prefers val (if non-empty), otherwise omega. "
            "Only updates that improve this RMSE are accepted as a new epoch."
        ),
    )
    parser.add_argument(
        "--min-rmse-improve",
        type=float,
        default=0.0,
        help="Required RMSE improvement to accept a stepwise update.",
    )
    parser.add_argument(
        "--max-retries-per-epoch",
        type=int,
        default=8,
        help="Maximum backtracking retries per stepwise epoch when RMSE does not improve.",
    )
    parser.add_argument(
        "--min-rate",
        type=float,
        default=1.0e-8,
        help="Minimum learning rate allowed during stepwise retry backtracking.",
    )
    parser.add_argument(
        "--significant-drop-pct",
        type=float,
        default=1.0,
        help=(
            "If RMSE drops by at least this percentage from epoch 1 to epoch 3, "
            "print a significant-drop notice."
        ),
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help=(
            "Keep intermediate files (*.omega.tns/*.split_masks.npz/_ep*.ktns/_ep*.log/_ep*_history.tsv). "
            "Default keeps only final CSV."
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=pathlib.Path,
        default=None,
        help="Optional base directory for intermediate files.",
    )
    return parser.parse_args()


def split_valid_points(
    array: np.ndarray,
    validation_rate: float,
    sample_rate: float,
    rng_val: np.random.Generator,
    rng_omega: np.random.Generator,
    sampling_mode: int,
    sampling_alpha: float,
    sampling_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat = array.ravel(order="C")
    n_total = flat.size

    valid_flat = ~np.isnan(flat)
    if not np.any(valid_flat):
        empty = np.zeros(array.shape, dtype=bool)
        return empty, empty, empty, empty

    val_flat = np.zeros(n_total, dtype=bool)
    val_draw = rng_val.random(n_total)
    val_flat[valid_flat] = val_draw[valid_flat] < validation_rate

    train_pool_flat = valid_flat & (~val_flat)

    omega_flat = np.zeros(n_total, dtype=bool)
    if sampling_mode == 0:
        omega_draw = rng_omega.random(n_total)
        omega_flat[train_pool_flat] = omega_draw[train_pool_flat] < sample_rate
    else:
        idx = np.flatnonzero(train_pool_flat)
        if idx.size > 0:
            vals = flat[idx]
            weights = sampling_eps + np.power(np.abs(vals), sampling_alpha)
            avg_w = float(np.mean(weights))
            if avg_w <= 0.0:
                avg_w = 1.0
            probs = sample_rate * (weights / avg_w)
            probs = np.clip(probs, 0.0, 1.0)
            omega_flat[idx] = rng_omega.random(idx.size) < probs

    u_flat = valid_flat & (~val_flat) & (~omega_flat)

    return (
        omega_flat.reshape(array.shape),
        val_flat.reshape(array.shape),
        u_flat.reshape(array.shape),
        valid_flat.reshape(array.shape),
    )


def write_sparse_tns(
    values: np.ndarray, mask: np.ndarray, output_path: pathlib.Path, index_base: int = 0
) -> int:
    if values.shape != mask.shape:
        raise ValueError("values/mask shape mismatch for sparse training tensor export.")

    flat_values = values.ravel(order="C")
    flat_mask = mask.ravel(order="C")
    nnz_ids = np.flatnonzero(flat_mask)
    shape = values.shape

    with output_path.open("w", encoding="utf-8") as f:
        for linear_idx in nnz_ids:
            coord = np.unravel_index(int(linear_idx), shape, order="C")
            idx_txt = " ".join(str(int(c + index_base)) for c in coord)
            f.write(f"{idx_txt} {float(flat_values[linear_idx]):.17e}\n")

    return int(nnz_ids.size)


def metric_triplet(truth: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    count = int(mask.sum())
    if count == 0:
        return float("nan"), float("nan"), float("nan")
    t = truth[mask]
    p = pred[mask]
    diff = t - p
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mae = float(np.mean(np.abs(diff)))
    denom = float(np.sum(t * t))
    er = float(np.sqrt(np.sum(diff * diff) / denom)) if denom != 0.0 else float("nan")
    return er, mae, rmse


def read_ktensor(path: pathlib.Path) -> tuple[list[int], np.ndarray, list[np.ndarray]]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    idx = 0
    if lines[idx] != "ktensor":
        raise RuntimeError(f"{path} is not a ktensor file")
    idx += 1
    ndims = int(lines[idx])
    idx += 1
    dims = list(map(int, lines[idx].split()))
    idx += 1
    rank = int(lines[idx])
    idx += 1
    weights = np.fromstring(lines[idx], sep=" ", dtype=np.float64)
    idx += 1
    if weights.size != rank:
        raise RuntimeError(f"weights size mismatch in {path}: {weights.size} != {rank}")

    factors: list[np.ndarray] = []
    for _ in range(ndims):
        if lines[idx] != "matrix":
            raise RuntimeError(f"expected matrix header in {path}, got {lines[idx]}")
        idx += 1
        mat_ndims = int(lines[idx])
        idx += 1
        if mat_ndims != 2:
            raise RuntimeError(f"expected matrix ndim=2 in {path}, got {mat_ndims}")
        nrows, ncols = map(int, lines[idx].split())
        idx += 1
        mat = np.empty((nrows, ncols), dtype=np.float64)
        for i in range(nrows):
            row = np.fromstring(lines[idx], sep=" ", dtype=np.float64)
            idx += 1
            if row.size != ncols:
                raise RuntimeError(f"bad matrix row length in {path}")
            mat[i, :] = row
        factors.append(mat)

    return dims, weights, factors


def reconstruct_from_ktensor(weights: np.ndarray, factors: Sequence[np.ndarray]) -> np.ndarray:
    symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ndims = len(factors)
    if ndims > len(symbols):
        raise RuntimeError(f"Unsupported ndims={ndims} for einsum symbols.")
    lhs = ",".join(f"{symbols[i]}r" for i in range(ndims))
    rhs = "".join(symbols[:ndims])
    expr = f"{lhs},r->{rhs}"
    return np.einsum(expr, *factors, weights, optimize=True)


def parse_last_history_row(path: pathlib.Path) -> tuple[int, float, float, float]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    toks = lines[-1].split()
    # iter residual fit ||grad|| comm-time mmtkrp-time time gpu-time mttkrp
    return int(toks[0]), float(toks[1]), float(toks[6]), float(toks[7])


def fmt(v: float) -> str:
    return "nan" if math.isnan(v) else f"{v:.10e}"


def parse_metric(v: str | int) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def stepwise_stop_triggered(step_fest: float, nfails: int, gcp_tol: float, fails: int) -> bool:
    # Legacy stepwise stop rule used in prior runs:
    # stop when fest < gcp_tol OR nfails > fails.
    return (math.isfinite(step_fest) and step_fest < gcp_tol) or (nfails > fails)


def resolve_monitor_key(mode: str, has_val: bool, has_u: bool) -> tuple[str, str | None]:
    if mode == "auto":
        if has_val:
            return "val_rmse", None
        return "omega_rmse", "validation set is empty; fallback monitor set to omega_rmse."
    if mode == "omega":
        return "omega_rmse", None
    if mode == "val":
        if has_val:
            return "val_rmse", None
        return "omega_rmse", "monitor set val requested but validation set is empty; fallback to omega_rmse."
    if mode == "u":
        if has_u:
            return "u_rmse", None
        return "omega_rmse", "monitor set u requested but U set is empty; fallback to omega_rmse."
    raise RuntimeError(f"Unsupported monitor mode: {mode}")


def get_monitor_rmse(
    monitor_key: str, omega_rmse: float, val_rmse: float, u_rmse: float
) -> float:
    if monitor_key == "omega_rmse":
        return omega_rmse
    if monitor_key == "val_rmse":
        return val_rmse
    if monitor_key == "u_rmse":
        return u_rmse
    raise RuntimeError(f"Unsupported monitor key: {monitor_key}")


def significant_drop_report(
    rows: list[dict[str, str | int]], threshold_pct: float
) -> list[tuple[str, float]]:
    if len(rows) < 3:
        return []

    out: list[tuple[str, float]] = []
    metric_keys = ("omega_rmse", "val_rmse", "u_rmse")
    r1, r2, r3 = rows[0], rows[1], rows[2]
    for k in metric_keys:
        e1 = parse_metric(r1[k])
        e2 = parse_metric(r2[k])
        e3 = parse_metric(r3[k])
        if not (math.isfinite(e1) and math.isfinite(e2) and math.isfinite(e3)):
            continue
        if e1 == 0.0:
            continue
        drop_pct = (e1 - e3) / abs(e1) * 100.0
        monotone_nonincreasing = (e1 >= e2) and (e2 >= e3)
        if monotone_nonincreasing and drop_pct >= threshold_pct:
            out.append((k, drop_pct))
    return out


def auto_dataset_tag(npy_stem: str) -> str:
    first = npy_stem.split("_")[0]
    uppers = "".join(ch for ch in first if ch.isupper())
    if len(uppers) >= 2:
        return uppers[:3].upper()
    letters = "".join(ch for ch in first if ch.isalpha())
    if len(letters) >= 3:
        return letters[:3].upper()
    if len(letters) > 0:
        return letters.upper()
    return "DS"


def sample_rate_code(rate: float) -> str:
    scaled = int(round(rate * 10.0))
    return f"{scaled:02d}"


def build_genten_cmd(
    *,
    genten_bin: pathlib.Path,
    exec_space: str,
    train_path: pathlib.Path,
    rank: int,
    maxiters: int,
    epochiters: int,
    frozeniters: int,
    sampling: str,
    rate: float,
    decay: float,
    gcp_step: str,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
    fuse_sa: bool,
    seed: int,
    gcp_tol: float,
    fails: int,
    printitn: int,
    hist_path: pathlib.Path,
    ktns_path: pathlib.Path,
    initial_file: pathlib.Path | None = None,
) -> list[str]:
    cmd = [
        str(genten_bin),
        "--exec-space",
        exec_space,
        "--sparse",
        "--index-base",
        "0",
        "--method",
        "gcp-sgd",
        "--input",
        str(train_path),
        "--rank",
        str(rank),
        "--maxiters",
        str(maxiters),
        "--epochiters",
        str(epochiters),
        "--frozeniters",
        str(frozeniters),
        "--sampling",
        sampling,
        "--rate",
        str(rate),
        "--decay",
        str(decay),
        "--step",
        gcp_step,
        "--adam-beta1",
        str(adam_beta1),
        "--adam-beta2",
        str(adam_beta2),
        "--adam-eps",
        str(adam_eps),
        # Keep unsampled region U out of optimization objective.
        "--fzs",
        "1",
        "--gzs",
        "1",
        "--fzw",
        "0.0",
        "--gzw",
        "0.0",
        "--seed",
        str(seed),
        "--gcp-seed",
        str(seed),
        "--gcp-tol",
        str(gcp_tol),
        "--fails",
        str(fails),
        "--printitn",
        str(printitn),
        "--history-file",
        str(hist_path),
        "--output-file",
        str(ktns_path),
    ]
    if fuse_sa:
        cmd.append("--fuse-sa")
    if initial_file is not None:
        cmd.extend(["--initial-file", str(initial_file)])
    return cmd


def main() -> int:
    args = parse_args()
    npy_file = args.npy_file.resolve()
    genten_bin = args.genten_bin.resolve()
    logs_root = args.logs_root.resolve()
    dataset_tag = args.dataset_tag.upper() if args.dataset_tag else auto_dataset_tag(npy_file.stem)
    output_dir = logs_root / f"{dataset_tag}-genten"
    rate_code = sample_rate_code(args.sample_rate)
    default_name = pathlib.Path(f"{dataset_tag}{rate_code}.csv")
    if args.output_file is None:
        csv_path = output_dir / default_name
    else:
        csv_path = args.output_file if args.output_file.is_absolute() else (output_dir / args.output_file)
    if csv_path.suffix.lower() != ".csv":
        csv_path = csv_path.with_suffix(".csv")
    artifact_prefix = args.prefix if args.prefix else f"{dataset_tag}{rate_code}"

    rng_val = np.random.Generator(np.random.MT19937(args.val_seed))
    rng_omega = np.random.Generator(np.random.MT19937(args.omega_seed))

    if not npy_file.exists():
        print(f"Input file not found: {npy_file}", file=sys.stderr)
        return 1
    if not genten_bin.exists():
        print(f"genten binary not found: {genten_bin}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    truth = np.asarray(np.load(npy_file), dtype=np.float64)
    omega_mask, val_mask, u_mask, valid_mask = split_valid_points(
        truth,
        args.validation_rate,
        args.sample_rate,
        rng_val,
        rng_omega,
        args.split_sampling_mode,
        args.sampling_alpha,
        args.sampling_eps,
    )
    if int(val_mask.sum()) == 0:
        print(
            "WARN: validation set is empty; val_er/val_mae/val_rmse will be nan. "
            "Increase --validation-rate.",
            file=sys.stderr,
        )

    def run_with_workspace(work_root: pathlib.Path) -> tuple[list[dict[str, str | int]], int]:
        train_path = work_root / f"{artifact_prefix}.omega.tns"
        omega_train_count = write_sparse_tns(truth, omega_mask, train_path, index_base=0)
        if omega_train_count == 0:
            raise RuntimeError("Omega split is empty; cannot train GenTen with no sampled points.")

        if args.keep_artifacts:
            masks_path = work_root / f"{artifact_prefix}.split_masks.npz"
            np.savez_compressed(
                masks_path, omega=omega_mask, val=val_mask, u=u_mask, valid=valid_mask
            )

        stop_epoch = int(args.maxiters)
        rows: list[dict[str, str | int]] = []
        cleanup_paths: list[pathlib.Path] = []

        if args.epoch_mode == "cumulative":
            for epoch_cap in range(1, stop_epoch + 1):
                hist_path = work_root / f"{artifact_prefix}_ep{epoch_cap}_history.tsv"
                ktns_path = work_root / f"{artifact_prefix}_ep{epoch_cap}.ktns"
                runlog_path = work_root / f"{artifact_prefix}_ep{epoch_cap}.log"

                cmd = build_genten_cmd(
                    genten_bin=genten_bin,
                    exec_space=args.exec_space,
                    train_path=train_path,
                    rank=args.rank,
                    maxiters=epoch_cap,
                    epochiters=args.epochiters,
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
                    initial_file=None,
                )

                proc = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True
                )
                if args.keep_artifacts:
                    runlog_path.write_text(proc.stdout)
                else:
                    cleanup_paths.extend([hist_path, ktns_path])

                epoch, _fest, total_time, gpu_time = parse_last_history_row(hist_path)
                _, weights, factors = read_ktensor(ktns_path)
                pred = reconstruct_from_ktensor(weights, factors)

                omega_er, omega_mae, omega_rmse = metric_triplet(truth, pred, omega_mask)
                val_er, val_mae, val_rmse = metric_triplet(truth, pred, val_mask)
                u_er, u_mae, u_rmse = metric_triplet(truth, pred, u_mask)

                rows.append(
                    {
                        "epoch": epoch,
                        "total_time": fmt(total_time),
                        "gpu_time": fmt(gpu_time),
                        "omega_er": fmt(omega_er),
                        "omega_mae": fmt(omega_mae),
                        "omega_rmse": fmt(omega_rmse),
                        "val_er": fmt(val_er),
                        "val_mae": fmt(val_mae),
                        "val_rmse": fmt(val_rmse),
                        "u_er": fmt(u_er),
                        "u_mae": fmt(u_mae),
                        "u_rmse": fmt(u_rmse),
                    }
                )
                if epoch < epoch_cap:
                    break
        else:
            prev_ktns_path: pathlib.Path | None = None
            prev_fest: float | None = None
            nfails = 0
            total_time_acc = 0.0
            gpu_time_acc = 0.0
            best_stop_rmse = float("inf")
            rmse_no_improve = 0

            for step in range(1, stop_epoch + 1):
                hist_path = work_root / f"{artifact_prefix}_ep{step}_history.tsv"
                ktns_path = work_root / f"{artifact_prefix}_ep{step}.ktns"
                runlog_path = work_root / f"{artifact_prefix}_ep{step}.log"

                # Legacy stepwise semantics:
                # one external epoch == one GenTen call with 1/1/1 iterations.
                cmd = build_genten_cmd(
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
                    initial_file=prev_ktns_path,
                )

                proc = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True
                )
                if args.keep_artifacts:
                    runlog_path.write_text(proc.stdout)
                else:
                    cleanup_paths.extend([hist_path, ktns_path])

                _local_epoch, step_fest, step_total_time, step_gpu_time = parse_last_history_row(
                    hist_path
                )
                total_time_acc += step_total_time
                gpu_time_acc += step_gpu_time

                deteriorated = False
                if math.isnan(step_fest):
                    deteriorated = True
                elif (
                    prev_fest is not None
                    and math.isfinite(prev_fest)
                    and math.isfinite(step_fest)
                    and step_fest >= (prev_fest - args.min_fest_improve)
                ):
                    # Treat plateau as a failed step so stepwise mode can auto-stop on convergence.
                    deteriorated = True

                if deteriorated:
                    nfails += 1
                else:
                    nfails = 0

                # Stepwise path always accepts the latest update and warm-starts from it.
                prev_fest = step_fest
                prev_ktns_path = ktns_path

                _, weights, factors = read_ktensor(ktns_path)
                pred = reconstruct_from_ktensor(weights, factors)

                omega_er, omega_mae, omega_rmse = metric_triplet(truth, pred, omega_mask)
                val_er, val_mae, val_rmse = metric_triplet(truth, pred, val_mask)
                u_er, u_mae, u_rmse = metric_triplet(truth, pred, u_mask)

                rows.append(
                    {
                        "epoch": step,
                        "total_time": fmt(total_time_acc),
                        "gpu_time": fmt(gpu_time_acc),
                        "omega_er": fmt(omega_er),
                        "omega_mae": fmt(omega_mae),
                        "omega_rmse": fmt(omega_rmse),
                        "val_er": fmt(val_er),
                        "val_mae": fmt(val_mae),
                        "val_rmse": fmt(val_rmse),
                        "u_er": fmt(u_er),
                        "u_mae": fmt(u_mae),
                        "u_rmse": fmt(u_rmse),
                    }
                )

                if args.rmse_stop_set == "omega":
                    tracked_rmse = omega_rmse
                elif args.rmse_stop_set == "u":
                    tracked_rmse = u_rmse
                elif args.rmse_stop_set == "val":
                    tracked_rmse = val_rmse
                else:
                    tracked_rmse = float("nan")

                if math.isfinite(tracked_rmse):
                    if tracked_rmse < (best_stop_rmse - args.rmse_min_improve):
                        best_stop_rmse = tracked_rmse
                        rmse_no_improve = 0
                    else:
                        rmse_no_improve += 1

                rmse_plateau_stop = (
                    args.rmse_stop_set != "off" and rmse_no_improve > args.rmse_patience
                )

                if rmse_plateau_stop or stepwise_stop_triggered(step_fest, nfails, args.gcp_tol, args.fails):
                    break

        if not args.keep_artifacts:
            for p in cleanup_paths:
                if p.exists():
                    p.unlink()

        return rows, len(rows)

    if args.keep_artifacts:
        work_root = args.work_dir.resolve() if args.work_dir is not None else output_dir
        work_root.mkdir(parents=True, exist_ok=True)
        rows, stop_epoch = run_with_workspace(work_root)
    else:
        tmp_base = args.work_dir.resolve() if args.work_dir is not None else None
        if tmp_base is not None:
            tmp_base.mkdir(parents=True, exist_ok=True)
            tmp_ctx = tempfile.TemporaryDirectory(prefix=f"{artifact_prefix}_", dir=str(tmp_base))
        else:
            tmp_ctx = tempfile.TemporaryDirectory(prefix=f"{artifact_prefix}_")
        with tmp_ctx as td:
            rows, stop_epoch = run_with_workspace(pathlib.Path(td))

    fieldnames = [
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
    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                "epoch": row["epoch"],
                "t_total_s": row["total_time"],
                "train_gpu": row["gpu_time"],
                "train_rmse": row["omega_rmse"],
                "train_mae": row["omega_mae"],
                "train_er": row["omega_er"],
                "val_rmse": row["val_rmse"],
                "val_mae": row["val_mae"],
                "val_er": row["val_er"],
                "u_rmse": row["u_rmse"],
                "u_mae": row["u_mae"],
                "u_er": row["u_er"],
            }
        )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    sig_drops = significant_drop_report(rows, args.significant_drop_pct)

    print(f"Done. CSV: {csv_path}")
    print(f"Completed epochs: {stop_epoch}")
    if len(rows) >= 3:
        if sig_drops:
            details = ", ".join(f"{k} ({v:.2f}%)" for k, v in sig_drops)
            print(f"Significant drop detected in first 3 epochs: {details}")
        else:
            print(
                f"No significant RMSE drop (threshold {args.significant_drop_pct:.2f}%) "
                "in first 3 epochs."
            )
    if args.keep_artifacts:
        work_root = args.work_dir.resolve() if args.work_dir is not None else output_dir
        print(f"Artifacts kept in: {work_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
