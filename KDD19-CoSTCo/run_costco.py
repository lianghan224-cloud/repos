#!/usr/bin/env python3
"""Run a CoSTCo-style tensor completion experiment on normalized .tns data.

The original KDD19-CoSTCo repo ships a Python2/TensorFlow-1 notebook.  This
script keeps the model structure from that notebook but uses PyTorch so the
baseline can run in the current environment and emit the same CSV schema used
by the other experiments.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn


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


def read_tns(path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    idx_chunks = []
    val_chunks = []
    max_idx = np.zeros(3, dtype=np.int64)

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        idx_buf = []
        val_buf = []
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            i = int(parts[0]) - 1
            j = int(parts[1]) - 1
            k = int(parts[2]) - 1
            v = float(parts[3])
            idx_buf.append((i, j, k))
            val_buf.append(v)
            if len(idx_buf) >= 1_000_000:
                idx_arr = np.asarray(idx_buf, dtype=np.int64)
                val_arr = np.asarray(val_buf, dtype=np.float32)
                idx_chunks.append(idx_arr)
                val_chunks.append(val_arr)
                max_idx = np.maximum(max_idx, idx_arr.max(axis=0))
                idx_buf.clear()
                val_buf.clear()
        if idx_buf:
            idx_arr = np.asarray(idx_buf, dtype=np.int64)
            val_arr = np.asarray(val_buf, dtype=np.float32)
            idx_chunks.append(idx_arr)
            val_chunks.append(val_arr)
            max_idx = np.maximum(max_idx, idx_arr.max(axis=0))

    if not idx_chunks:
        raise ValueError(f"no tensor entries found in {path}")

    indices = np.concatenate(idx_chunks, axis=0)
    values = np.concatenate(val_chunks, axis=0)
    shape = tuple(int(x) + 1 for x in max_idx)
    return indices, values, shape  # type: ignore[return-value]


def read_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    data = np.load(path)
    indices = np.stack([data["i"], data["j"], data["k"]], axis=1).astype(np.int64, copy=False)
    values = data["v"].astype(np.float32, copy=False)
    shape = tuple(int(x) for x in data["shape"])
    return indices, values, shape  # type: ignore[return-value]


def split_three_sets(
    indices: np.ndarray,
    values: np.ndarray,
    val_rate: float,
    sampling_rate: float,
    val_seed: int,
    omega_seed: int,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    n = values.shape[0]
    val_rng = np.random.Generator(np.random.MT19937(val_seed))
    omega_rng = np.random.Generator(np.random.MT19937(omega_seed))

    val_mask = val_rng.random(n, dtype=np.float32) < np.float32(val_rate)
    omega_mask = np.zeros(n, dtype=bool)
    pool = ~val_mask
    omega_mask[pool] = (
        omega_rng.random(int(pool.sum()), dtype=np.float32) < np.float32(sampling_rate)
    )
    u_mask = pool & ~omega_mask

    return {
        "train": (indices[omega_mask], values[omega_mask]),
        "val": (indices[val_mask], values[val_mask]),
        "u": (indices[u_mask], values[u_mask]),
    }


class CoSTCo(nn.Module):
    def __init__(self, shape: Tuple[int, int, int], rank: int, nc: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(dim, rank) for dim in shape])
        self.conv1 = nn.Conv2d(1, nc, kernel_size=(1, len(shape)))
        self.conv2 = nn.Conv2d(nc, nc, kernel_size=(rank, 1))
        self.fc = nn.Linear(nc, nc)
        self.out = nn.Linear(nc, 1)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        embeds = [emb(idx[:, m]) for m, emb in enumerate(self.embeddings)]
        x = torch.stack(embeds, dim=2).unsqueeze(1)  # B x 1 x rank x nmodes
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc(x))
        return torch.relu(self.out(x)).squeeze(1)


def batches(
    indices: np.ndarray,
    values: np.ndarray,
    batch_size: int,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    for start in range(0, values.shape[0], batch_size):
        end = min(start + batch_size, values.shape[0])
        yield indices[start:end], values[start:end]


@torch.no_grad()
def metrics(
    model: nn.Module,
    indices: np.ndarray,
    values: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    sse = 0.0
    sae = 0.0
    st2 = 0.0
    n = 0
    for idx_np, val_np in batches(indices, values, batch_size):
        idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
        y = torch.as_tensor(val_np, dtype=torch.float32, device=device)
        pred = model(idx)
        err = pred - y
        sse += float(torch.sum(err * err).detach().cpu())
        sae += float(torch.sum(torch.abs(err)).detach().cpu())
        st2 += float(torch.sum(y * y).detach().cpu())
        n += int(y.numel())
    if n == 0:
        return 0.0, 0.0, 0.0
    rmse = math.sqrt(sse / n)
    mae = sae / n
    er = math.sqrt(sse / st2) if st2 > 0.0 else 0.0
    return rmse, mae, er


def metric_row(
    epoch: int,
    t_total_s: str,
    train_gpu: str,
    train: Tuple[float, float, float],
    val: Tuple[float, float, float],
    u: Tuple[float, float, float],
) -> Dict[str, str]:
    return {
        "epoch": str(epoch),
        "t_total_s": t_total_s,
        "train_gpu": train_gpu,
        "train_rmse": f"{train[0]:.5e}",
        "train_mae": f"{train[1]:.5e}",
        "train_er": f"{train[2]:.5e}",
        "val_rmse": f"{val[0]:.5e}",
        "val_mae": f"{val[1]:.5e}",
        "val_er": f"{val[2]:.5e}",
        "u_rmse": f"{u[0]:.5e}",
        "u_mae": f"{u[1]:.5e}",
        "u_er": f"{u[2]:.5e}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--train-npz", type=Path)
    parser.add_argument("--val-npz", type=Path)
    parser.add_argument("--test-npz", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--sampling-rate", type=float, required=True)
    parser.add_argument("--val-rate", type=float, default=0.05)
    parser.add_argument("--val-seed", type=int, default=20250101)
    parser.add_argument("--omega-seed", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    start_time = time.perf_counter()
    if args.train_npz or args.val_npz or args.test_npz:
        if not (args.train_npz and args.val_npz and args.test_npz):
            raise SystemExit("--train-npz, --val-npz, and --test-npz must be provided together")
        train_idx, train_vals, shape = read_npz(args.train_npz)
        val_idx, val_vals, val_shape = read_npz(args.val_npz)
        test_idx, test_vals, test_shape = read_npz(args.test_npz)
        if val_shape != shape or test_shape != shape:
            raise SystemExit("split shapes do not match")
        sets = {
            "train": (train_idx, train_vals),
            "val": (val_idx, val_vals),
            "u": (test_idx, test_vals),
        }
    else:
        if args.input is None:
            raise SystemExit("--input is required unless split npz files are provided")
        indices, values, shape = read_tns(args.input)
        sets = split_three_sets(
            indices,
            values,
            val_rate=args.val_rate,
            sampling_rate=args.sampling_rate,
            val_seed=args.val_seed,
            omega_seed=args.omega_seed,
        )
        del indices, values

    model = CoSTCo(shape, rank=args.rank, nc=args.rank).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_gpu_s = 0.0
    best_val = float("inf")
    bad_epochs = 0

    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()

        train_metrics = metrics(model, *sets["train"], args.eval_batch_size, device)
        val_metrics = metrics(model, *sets["val"], args.eval_batch_size, device)
        u_metrics = metrics(model, *sets["u"], args.eval_batch_size, device)
        writer.writerow(metric_row(0, "", "", train_metrics, val_metrics, u_metrics))
        fh.flush()

        train_idx, train_vals = sets["train"]
        for epoch in range(1, args.epochs + 1):
            model.train()
            if device.type == "cuda":
                ev_start = torch.cuda.Event(enable_timing=True)
                ev_end = torch.cuda.Event(enable_timing=True)
                ev_start.record()
            epoch_loss = 0.0
            epoch_n = 0
            for idx_np, val_np in batches(train_idx, train_vals, args.batch_size):
                idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
                y = torch.as_tensor(val_np, dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(idx)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu()) * int(y.numel())
                epoch_n += int(y.numel())
            if device.type == "cuda":
                ev_end.record()
                torch.cuda.synchronize()
                total_gpu_s += ev_start.elapsed_time(ev_end) / 1000.0

            train_metrics = metrics(model, *sets["train"], args.eval_batch_size, device)
            val_metrics = metrics(model, *sets["val"], args.eval_batch_size, device)
            u_metrics = metrics(model, *sets["u"], args.eval_batch_size, device)
            writer.writerow(
                metric_row(
                    epoch,
                    f"{time.perf_counter() - start_time:.6f}",
                    f"{total_gpu_s:.6f}",
                    train_metrics,
                    val_metrics,
                    u_metrics,
                )
            )
            fh.flush()

            if val_metrics[0] < best_val - args.tolerance:
                best_val = val_metrics[0]
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    break

            _ = epoch_loss / max(epoch_n, 1)


if __name__ == "__main__":
    main()
