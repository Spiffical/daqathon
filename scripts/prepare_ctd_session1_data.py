#!/usr/bin/env python3
"""Build the typed parquet cache used by the DAQathon notebooks.

This script converts the raw ONC CTD CSV exports into two reusable artifacts:

- row-level parquet parts for supervised workflows
- fixed-window summary parquet for clustering and sequence demos

It is intentionally simple and deterministic so the notebooks can explain the
preparation step without hiding any logic in a black box.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import pandas as pd

MEASUREMENT_COLUMNS = [
    "Conductivity (S/m)",
    "Density (kg/m3)",
    "Depth (m)",
    "Practical Salinity (psu)",
    "Pressure (decibar)",
    "Sigma-t (kg/m3)",
    "Sigma-theta (0 dbar) (kg/m3)",
    "Sound Speed (m/s)",
    "Temperature (C)",
]

QC_COLUMNS = [
    "Conductivity QC Flag",
    "Density QC Flag",
    "Depth QC Flag",
    "Practical Salinity QC Flag",
    "Pressure QC Flag",
    "Sigma-t QC Flag",
    "Sigma-theta (0 dbar) QC Flag",
    "Sound Speed QC Flag",
    "Temperature QC Flag",
]

KEEP_COLUMNS = ["Time UTC", *MEASUREMENT_COLUMNS, *QC_COLUMNS]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the cache-preparation workflow."""
    parser = argparse.ArgumentParser(
        description="Prepare ONC CTD CSV files for the DAQathon Session 1 notebook."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing the ONC CTD CSV files.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="Directory where parquet caches and metadata should be written.",
    )
    parser.add_argument(
        "--target-flag",
        default="Conductivity QC Flag",
        choices=QC_COLUMNS,
        help="QC flag column to use as the supervised target.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on how many CSV files to process.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional maximum number of data rows to read from each CSV file.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=256,
        help="Number of sequential rows per summary window for clustering/CNN prep.",
    )
    return parser.parse_args()


def clean_header_value(value: str) -> str:
    """Normalize one raw ONC header field into a stable column name."""
    cleaned = value.strip()
    if cleaned.startswith("#"):
        cleaned = cleaned[1:]
    cleaned = cleaned.strip().strip('"')
    if cleaned.startswith("Time UTC"):
        return "Time UTC"
    return cleaned


def locate_header(path: Path) -> tuple[int, list[str]]:
    """Find the first tabular header row and return its line number and cleaned columns."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for line_number, row in enumerate(reader, start=1):
            if row and "Time UTC" in row[0]:
                columns = [clean_header_value(value) for value in row]
                return line_number, columns
    raise ValueError(f"Could not locate the tabular header in {path}")


def read_ctd_csv(path: Path, sample_rows: int | None) -> pd.DataFrame:
    """Load one ONC CTD CSV into a typed, time-sorted dataframe.

    The loader:

    - skips the metadata block above the real table header
    - keeps only the columns needed for DAQathon
    - converts timestamps and numeric fields into typed columns
    - records the source filename for later provenance and plotting
    """
    header_line_number, columns = locate_header(path)
    missing_columns = [column for column in KEEP_COLUMNS if column not in columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns in {path}: {missing_columns}")

    frame = pd.read_csv(
        path,
        header=None,
        names=columns,
        skiprows=header_line_number,
        usecols=KEEP_COLUMNS,
        nrows=sample_rows,
        low_memory=False,
    )
    frame["Time UTC"] = pd.to_datetime(frame["Time UTC"], utc=True, errors="coerce", format="ISO8601")
    numeric_columns = [column for column in KEEP_COLUMNS if column != "Time UTC"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["Time UTC"]).sort_values("Time UTC").reset_index(drop=True)
    frame["source_file"] = path.name
    return frame


def build_window_features(
    frame: pd.DataFrame,
    target_flag: str,
    window_size: int,
) -> pd.DataFrame:
    """Summarize contiguous row windows into one feature row per window.

    These summaries are used by the k-means section and by sequence-model
    preparation where short operating regimes matter more than isolated rows.
    """
    # Windows are row-count based so later demos can rely on fixed-length
    # chunks even if the raw timestamps are not perfectly evenly spaced.
    work = frame.reset_index(drop=True).copy()
    work["window_id"] = work.index // window_size

    named_aggs: dict[str, tuple[str, str | callable]] = {
        "window_start": ("Time UTC", "min"),
        "window_end": ("Time UTC", "max"),
        "source_file": ("source_file", "first"),
        "row_count": ("Time UTC", "size"),
        "issue_count": (target_flag, lambda values: int(values.isin([3, 4, 9]).sum())),
        "issue_rate": (target_flag, lambda values: float(values.isin([3, 4, 9]).mean())),
        "target_mode": (
            target_flag,
            lambda values: int(values.mode(dropna=True).iloc[0])
            if not values.mode(dropna=True).empty
            else -1,
        ),
    }

    for column in MEASUREMENT_COLUMNS:
        named_aggs[f"{column}_mean"] = (column, "mean")
        named_aggs[f"{column}_std"] = (column, "std")
        named_aggs[f"{column}_min"] = (column, "min")
        named_aggs[f"{column}_max"] = (column, "max")

    window_frame = work.groupby("window_id", sort=True).agg(**named_aggs).reset_index(drop=True)
    return window_frame


def write_metadata(
    cache_root: Path,
    target_flag: str,
    processed_files: list[dict[str, object]],
    target_counts: Counter,
    row_count: int,
    window_count: int,
    sample_rows: int | None,
    window_size: int,
) -> Path:
    """Write a JSON metadata manifest describing the prepared cache."""
    issue_count = sum(target_counts.get(flag, 0) for flag in (3, 4, 9))
    metadata = {
        "target_flag": target_flag,
        "measurement_columns": MEASUREMENT_COLUMNS,
        "qc_columns": QC_COLUMNS,
        "processed_file_count": len(processed_files),
        "row_count": row_count,
        "window_count": window_count,
        "sample_rows_per_file": sample_rows,
        "window_size": window_size,
        "target_distribution": {str(key): int(value) for key, value in sorted(target_counts.items())},
        "issue_fraction": issue_count / row_count if row_count else 0.0,
        "processed_files": processed_files,
        "row_level_cache": str(cache_root / "ctd_session1_row_level"),
        "window_cache": str(cache_root / "ctd_session1_windowed_features.parquet"),
    }
    metadata_path = cache_root / "ctd_session1_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def clear_old_outputs(cache_root: Path) -> None:
    """Remove stale cache files before writing a fresh preparation run."""
    row_level_dir = cache_root / "ctd_session1_row_level"
    if row_level_dir.exists():
        for existing in row_level_dir.glob("*.parquet"):
            existing.unlink()

    for existing_file in (
        cache_root / "ctd_session1_windowed_features.parquet",
        cache_root / "ctd_session1_metadata.json",
    ):
        if existing_file.exists():
            existing_file.unlink()


def main() -> None:
    """Prepare the row-level and window-level parquet cache from raw CSV files."""
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    row_level_dir = cache_root / "ctd_session1_row_level"
    row_level_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(data_root.glob("*.csv"))
    if args.max_files is not None:
        csv_files = csv_files[: args.max_files]
    if not csv_files:
        raise SystemExit(f"No CSV files found in {data_root}")

    clear_old_outputs(cache_root)
    row_level_dir.mkdir(parents=True, exist_ok=True)

    processed_files: list[dict[str, object]] = []
    target_counts: Counter = Counter()
    window_frames: list[pd.DataFrame] = []
    total_rows = 0

    for index, csv_path in enumerate(csv_files, start=1):
        # Read, type, and sort one source file before writing the parquet part.
        frame = read_ctd_csv(csv_path, sample_rows=args.sample_rows)
        total_rows += len(frame)
        target_counts.update(frame[args.target_flag].dropna().astype(int).tolist())

        part_path = row_level_dir / f"part-{index:03d}.parquet"
        frame.to_parquet(part_path, index=False)

        # Build the fixed-window summaries from the same typed dataframe so both
        # caches stay aligned file by file.
        window_frame = build_window_features(frame, target_flag=args.target_flag, window_size=args.window_size)
        window_frames.append(window_frame)

        processed_files.append(
            {
                "source_file": csv_path.name,
                "row_count": int(len(frame)),
                "time_start": frame["Time UTC"].min().isoformat() if not frame.empty else None,
                "time_end": frame["Time UTC"].max().isoformat() if not frame.empty else None,
                "row_level_part": part_path.name,
            }
        )

        print(f"[{index}/{len(csv_files)}] wrote {part_path} with {len(frame):,} rows")

    window_cache_path = cache_root / "ctd_session1_windowed_features.parquet"
    combined_windows = pd.concat(window_frames, ignore_index=True)
    combined_windows.to_parquet(window_cache_path, index=False)

    metadata_path = write_metadata(
        cache_root=cache_root,
        target_flag=args.target_flag,
        processed_files=processed_files,
        target_counts=target_counts,
        row_count=total_rows,
        window_count=len(combined_windows),
        sample_rows=args.sample_rows,
        window_size=args.window_size,
    )

    print(
        json.dumps(
            {
                "row_level_cache": str(row_level_dir),
                "window_cache": str(window_cache_path),
                "metadata": str(metadata_path),
                "rows": total_rows,
                "windows": int(len(combined_windows)),
                "target_distribution": {str(key): int(value) for key, value in sorted(target_counts.items())},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
