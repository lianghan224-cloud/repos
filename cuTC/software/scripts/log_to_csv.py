#!/usr/bin/env python3
"""
Convert cuTC log file(s) to CSV with selected columns only.

Output columns:
  epoch, t_total_s, train_gpu,
  train_rmse, train_mae, train_er,
  val_rmse, val_mae, val_er,
  u_rmse, u_mae, u_er

Note:
  train_gpu is parsed from `total-gpu-us` (prefix cumulative GPU time),
  not per-epoch `gpu-us`.

Examples:
  python3 log_to_csv.py --input /path/to/ECW_0801.log
  python3 log_to_csv.py --input /path/to/log_dir --output /path/to/all_logs.csv
  python3 log_to_csv.py --input /path/to/log_dir --pattern "ECW_08*.log"
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List


EPOCH_TIME_RE = re.compile(
    r"epoch:(\d+)\s+time-us:(\d+)\s+gpu-us:(\d+)\s+total-us:(\d+)\s+total-gpu-us:(\d+)"
)

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

METRIC_FIELD_MAP = {
    "RMSE-tr": "train_rmse",
    "MAE-tr": "train_mae",
    "ER-tr": "train_er",
    "RMSE-vl": "val_rmse",
    "MAE-vl": "val_mae",
    "ER-vl": "val_er",
    "RMSE-u": "u_rmse",
    "MAE-u": "u_mae",
    "ER-u": "u_er",
}


def parse_log_lines(lines: Iterable[str]) -> List[Dict[str, str]]:
    epochs: Dict[int, Dict[str, str]] = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("epoch:") and "loss:" in line:
            epoch_m = re.search(r"epoch:(\d+)", line)
            if not epoch_m:
                continue
            epoch = int(epoch_m.group(1))
            row = epochs.setdefault(epoch, {"epoch": str(epoch)})
            for metric_key, metric_value in re.findall(
                r"([A-Za-z]+-[A-Za-z]+):\s*([0-9.eE+-]+)", line
            ):
                target = METRIC_FIELD_MAP.get(metric_key)
                if target:
                    row[target] = metric_value
            continue

        time_match = EPOCH_TIME_RE.search(line)
        if time_match:
            epoch = int(time_match.group(1))
            row = epochs.setdefault(epoch, {"epoch": str(epoch)})
            row.update(
                {
                    # Use prefix cumulative GPU time to match reporting requirement.
                    "train_gpu": f"{int(time_match.group(5)) / 1_000_000.0:.6f}",
                    "t_total_s": f"{int(time_match.group(4)) / 1_000_000.0:.6f}",
                }
            )

    rows: List[Dict[str, str]] = []
    for epoch in sorted(epochs):
        rows.append(epochs[epoch])
    return rows


def parse_log_file(log_path: Path) -> List[Dict[str, str]]:
    with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
        return parse_log_lines(fh)


def collect_logs(input_path: Path, pattern: str) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.glob(pattern) if p.is_file())
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def default_output_path(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.with_suffix(".csv")
    return input_path / "logs.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert cuTC log file(s) to CSV.")
    parser.add_argument(
        "--input",
        required=True,
        help="A single .log path, a directory, or '-' to read from stdin.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output CSV file path. Default: <input>.csv for file, or <input>/logs.csv for directory.",
    )
    parser.add_argument(
        "--pattern",
        default="*.log",
        help="Glob pattern used when --input is a directory (default: *.log).",
    )
    args = parser.parse_args()

    all_rows: List[Dict[str, str]] = []
    source_desc = ""

    if args.input == "-":
        if not args.output:
            raise SystemExit("When --input is '-', --output must be provided.")
        output_path = Path(args.output).expanduser().resolve()
        all_rows = parse_log_lines(sys.stdin)
        source_desc = "stdin"
        source_count = 1
    else:
        input_path = Path(args.input).expanduser().resolve()
        output_path = (
            Path(args.output).expanduser().resolve()
            if args.output
            else default_output_path(input_path)
        )

        log_files = collect_logs(input_path, args.pattern)
        if not log_files:
            raise SystemExit(f"No log files found in: {input_path}")

        for log_file in log_files:
            all_rows.extend(parse_log_file(log_file))
        source_desc = str(output_path)
        source_count = len(log_files)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    if args.input == "-":
        print(f"Converted {source_desc} -> {output_path}")
    else:
        print(f"Converted {source_count} log file(s) -> {output_path}")
    print(f"Total rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
