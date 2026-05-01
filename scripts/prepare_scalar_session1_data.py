#!/usr/bin/env python3
"""Build a typed parquet cache from ONC scalar CSV exports.

This script is the "raw data to ML-ready cache" bridge used by the Session 1
notebooks. Its job is to take one folder of ONC CSV exports, clean and type the
raw values, optionally merge secondary device streams onto one primary time
base, and then write two reusable parquet artifacts:

- a row-level cache with one row per timestamp
- a window-summary cache with one row per fixed window

The implementation is intentionally explicit rather than clever. The notebook
walkthrough points readers to this file, so the functions are written to be read
as well as executed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

DEVICE_PATTERNS = {
    "ctd": "ConductivityTemperatureDepth",
    "fluorometer": "Fluorometer",
    "oxygen": "OxygenSensor",
}

FILENAME_TIME_RANGE_RE = re.compile(r"_(\d{8}T\d{6}(?:\.\d+)?Z)_(\d{8}T\d{6}(?:\.\d+)?Z)(?:-NaN)?\.csv$")

DEFAULT_CACHE_STEM = "scalar_session1"
DEFAULT_ISSUE_LABELS = [3, 4, 9]

DEFAULT_MEASUREMENT_COLUMNS = [
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

PREFERRED_MEASUREMENT_COLUMNS = [
    *DEFAULT_MEASUREMENT_COLUMNS,
    "Chlorophyll (ug/l)",
    "Fluorescence (mg/m3)",
    "Turbidity (NTU)",
    "Oxygen Concentration Corrected (ml/l)",
    "Oxygen Concentration Uncorrected (ml/l)",
    "Dissolved Oxygen (mL/L)",
    "Dissolved Oxygen (umol/L)",
    "Oxygen Saturation (%)",
]


@dataclass(frozen=True)
class CacheBundlePaths:
    """Resolved filesystem paths for one named cache bundle.

    A "cache bundle" is the trio of outputs created by one preparation run:

    - a directory of row-level parquet parts
    - one window-summary parquet file
    - one metadata JSON file

    Grouping those related paths into one dataclass makes the rest of the
    script easier to read and avoids passing several loosely related paths
    around as separate arguments.
    """

    root: Path
    stem: str
    row_level_dir: Path
    window_cache_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class CsvFileInfo:
    """Parsed filename metadata for one ONC export file.

    This small record is mainly used when we need to reduce the number of files
    for a quick run while still keeping multiple device families aligned in
    time. The filename interval and series key let us match "companion" CTD,
    fluorometer, or oxygen files to the same chunk of time.
    """

    path: Path
    device: str
    series_key: str
    start_time: pd.Timestamp | None
    end_time: pd.Timestamp | None


@dataclass
class RowLevelCacheResult:
    """Summary returned after writing row-level parquet parts."""

    all_available_columns: list[str]
    grouped_infos: dict[str, list[CsvFileInfo]]
    limited_paths: dict[str, list[Path]]
    selected_infos: dict[str, list[CsvFileInfo]]
    requested_measurement_columns: list[str] | None
    measurement_columns: list[str]
    missing_measurement_columns: list[str]
    qc_columns: list[str]
    processed_files: list[dict[str, object]]
    total_rows: int
    target_counts: Counter[int]
    row_columns: list[str]


@dataclass
class WindowLevelCacheResult:
    """Summary returned after writing the window-summary parquet file."""

    window_count: int
    window_columns: list[str]


def normalize_cache_stem(value: str) -> str:
    """Normalize and validate the user-facing cache bundle name.

    The cache stem becomes part of directory and file names on disk, so we keep
    it simple: non-empty and path-free. This lets multiple prepared dataset
    presets coexist safely under one cache root.
    """

    stem = value.strip()
    if not stem:
        raise ValueError("cache stem must not be empty")
    if any(separator in stem for separator in ("/", "\\")):
        raise ValueError("cache stem must be a simple name, not a path")
    return stem


def build_cache_bundle_paths(cache_root: Path, cache_stem: str = DEFAULT_CACHE_STEM) -> CacheBundlePaths:
    """Build the canonical output paths for one cache bundle.

    Parameters
    ----------
    cache_root:
        Root directory under which the cache bundle should be created.
    cache_stem:
        Short bundle name used to namespace the row parquet directory, the
        window parquet file, and the metadata file.

    Returns
    -------
    CacheBundlePaths
        A dataclass containing all output paths used by the rest of the script.
    """

    stem = normalize_cache_stem(cache_stem)
    root = cache_root.expanduser().resolve()
    return CacheBundlePaths(
        root=root,
        stem=stem,
        row_level_dir=root / f"{stem}_row_level",
        window_cache_path=root / f"{stem}_windowed_features.parquet",
        metadata_path=root / f"{stem}_metadata.json",
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the scalar cache-preparation workflow.

    The notebooks call this script as a subprocess, but it is also designed to
    be runnable directly from the command line. Keeping all user-facing knobs in
    one place makes the preparation step easier to inspect and reproduce.
    """

    parser = argparse.ArgumentParser(
        description="Prepare ONC scalar CSV files for the DAQathon Session 1 notebooks."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--cache-stem",
        default=DEFAULT_CACHE_STEM,
        help="Bundle name used to namespace row, window, and metadata cache artifacts.",
    )
    parser.add_argument("--target-flag", required=True)
    parser.add_argument(
        "--primary-device",
        default="ctd",
        choices=sorted([*DEVICE_PATTERNS, "other"]),
        help="Device family that provides the main time base and target flag.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional maximum number of files to read per detected device family.",
    )
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument(
        "--issue-label",
        dest="issue_labels",
        action="append",
        type=int,
        default=None,
        help="Repeatable issue-label override used for issue summaries. Defaults to 3, 4, and 9.",
    )
    parser.add_argument(
        "--measurement-column",
        dest="measurement_columns",
        action="append",
        default=None,
        help="Repeatable measurement column override. Missing columns are reported in metadata.",
    )
    parser.add_argument(
        "--merge-tolerance-seconds",
        type=int,
        default=5,
        help="Nearest-merge tolerance in seconds when joining secondary devices onto the primary stream.",
    )
    return parser.parse_args()


def clean_header_value(value: str) -> str:
    """Normalize one raw ONC header field into a stable column name.

    ONC CSV exports often contain small formatting quirks such as leading
    comment markers or extra whitespace. This helper performs the light cleanup
    needed so later column matching is reliable.
    """

    cleaned = value.strip()
    if cleaned.startswith("#"):
        cleaned = cleaned[1:]
    cleaned = cleaned.strip().strip('"')
    if cleaned.startswith("Time UTC") or cleaned in {"sampleTime", "sample_time"}:
        return "Time UTC"
    return cleaned


def locate_header(path: Path) -> tuple[int, list[str]]:
    """Locate the real tabular header inside one ONC CSV export.

    ONC exports usually start with a metadata block rather than the table
    itself. This function scans line by line until it finds the row whose first
    field starts with ``Time UTC``, then returns:

    - the line number where the table begins
    - the cleaned list of column names

    That information is later reused by :func:`read_scalar_csv`.
    """

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for line_number, row in enumerate(reader, start=1):
            cleaned_row = [clean_header_value(value) for value in row]
            if cleaned_row and "Time UTC" in cleaned_row:
                return line_number, cleaned_row
    raise ValueError(f"Could not locate the tabular header in {path}")


def parse_measurement_columns(raw_values: list[str] | None) -> list[str] | None:
    """Parse repeated measurement-column overrides from CLI args.

    The notebooks sometimes pass measurement-column overrides either as
    repeated ``--measurement-column`` flags, comma-separated strings, or JSON
    list strings. This helper normalizes those formats into one deduplicated
    Python list.
    """

    if not raw_values:
        return None

    measurement_columns: list[str] = []
    for raw_value in raw_values:
        value = raw_value.strip()
        if not value:
            continue
        if value.startswith("["):
            parsed = json.loads(value)
            if not isinstance(parsed, list):
                raise ValueError("measurement column JSON must decode to a list")
            candidates = [str(item) for item in parsed]
        else:
            candidates = value.split(",")

        for candidate in candidates:
            column = candidate.strip()
            if column and column not in measurement_columns:
                measurement_columns.append(column)

    return measurement_columns or None


def detect_device(path: Path) -> str:
    """Infer a device family from the ONC export filename.

    The generalized scalar workflow supports one primary device plus optional
    secondary companions. The filename patterns are a lightweight way to
    separate CTD, fluorometer, and oxygen exports before we attempt any aligned
    selection or merging.
    """

    name = path.name
    for device, marker in DEVICE_PATTERNS.items():
        if marker in name:
            return device
    return "other"


def parse_file_info(path: Path) -> CsvFileInfo:
    """Extract device and time-range metadata from one CSV filename.

    Many ONC filenames encode a start and end timestamp. We parse those values
    so reduced runs can stay time-aligned across device families. The returned
    ``series_key`` captures the filename prefix before the time range, which is
    often enough to identify related files from the same deployment or export
    series.
    """

    device = detect_device(path)
    name = path.name

    match = FILENAME_TIME_RANGE_RE.search(name)
    if match:
        prefix_without_time = name[:match.start()]
        series_key = prefix_without_time.rsplit("_", 1)[0] if "_" in prefix_without_time else prefix_without_time
    else:
        series_key = path.stem

    if match:
        start_time = parse_filename_timestamp(match.group(1))
        end_time = parse_filename_timestamp(match.group(2))
    else:
        start_time = None
        end_time = None

    return CsvFileInfo(
        path=path,
        device=device,
        series_key=series_key,
        start_time=start_time,
        end_time=end_time,
    )


def discover_available_columns(csv_paths: list[Path]) -> list[str]:
    """Inspect headers across all candidate files and return the union of columns.

    This discovery pass happens *before* any file-count truncation. That design
    avoids an easy failure mode in mixed-device datasets where reading only the
    first few files would hide a secondary device family and make the downstream
    schema look incomplete.
    """

    column_order: list[str] = []
    seen: set[str] = set()
    for path in csv_paths:
        _, columns = locate_header(path)
        for column in columns:
            if column not in seen:
                seen.add(column)
                column_order.append(column)
    return column_order


def is_qc_like_column(column: str) -> bool:
    """Return whether a column looks like a QC or label bookkeeping field."""

    lowered = column.lower()
    return column.endswith("QC Flag") or "_qaqc_" in lowered or lowered.endswith("_qaqc")


def parse_filename_timestamp(value: str) -> pd.Timestamp:
    """Parse one compact timestamp token extracted from a CSV filename."""

    for time_format in ("%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S.%fZ"):
        try:
            return pd.Timestamp(datetime.strptime(value, time_format), tz="UTC")
        except ValueError:
            continue
    raise ValueError(f"Unsupported filename timestamp format: {value}")


def read_scalar_csv(
    path: Path,
    sample_rows: int | None,
    required_columns: list[str] | None = None,
    *,
    allow_missing_columns: bool = True,
) -> pd.DataFrame:
    """Load one ONC scalar CSV into a typed, time-sorted dataframe.

    This is the core row-cleaning function in the script. It:

    - finds the real table header
    - optionally prunes the file to a requested subset of columns
    - parses ``Time UTC`` as timezone-aware datetimes
    - converts every other loaded column to numeric values
    - drops rows whose timestamps could not be parsed
    - sorts by time and records the source filename

    The result is still "raw-row-shaped" data, but it is now typed and safe for
    further merging or parquet export.
    """

    header_line_number, columns = locate_header(path)
    if "Time UTC" not in columns:
        raise ValueError(f"Missing Time UTC in {path}")

    use_columns = None
    if required_columns is not None:
        requested_columns = ["Time UTC", *[column for column in required_columns if column != "Time UTC"]]
        available_columns = [column for column in requested_columns if column in columns]
        missing_columns = [column for column in requested_columns if column not in columns]
        if missing_columns and not allow_missing_columns:
            raise ValueError(f"Missing expected columns in {path}: {missing_columns}")
        use_columns = available_columns

    frame = pd.read_csv(
        path,
        header=None,
        names=columns,
        skiprows=header_line_number,
        usecols=use_columns,
        nrows=sample_rows,
        low_memory=False,
    )
    frame["Time UTC"] = pd.to_datetime(
        frame["Time UTC"], utc=True, errors="coerce", format="ISO8601"
    )
    for column in [column for column in frame.columns if column != "Time UTC"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["Time UTC"]).sort_values("Time UTC").reset_index(drop=True)
    frame["source_file"] = path.name
    return frame


def choose_measurement_columns(
    columns: list[str],
    requested_measurement_columns: list[str] | None = None,
) -> list[str]:
    """Choose which measurement columns should be kept in the prepared cache.

    The script distinguishes between:

    - timestamp / bookkeeping columns
    - QC flag columns
    - numeric measurement columns that are useful as model inputs

    If the caller requests a specific subset, we preserve that choice while
    quietly dropping unavailable or non-measurement fields. Otherwise we start
    from a preferred ordering and then append any remaining numeric columns.
    """

    qc_columns = {column for column in columns if is_qc_like_column(column)}
    excluded = {"Time UTC", "source_file", *qc_columns}

    if requested_measurement_columns:
        return [
            column
            for column in requested_measurement_columns
            if column in columns and column not in excluded
        ]

    ordered = [column for column in PREFERRED_MEASUREMENT_COLUMNS if column in columns and column not in excluded]
    extras = [column for column in columns if column not in excluded and column not in ordered]
    return ordered + extras


def build_window_features(
    frame: pd.DataFrame,
    target_flag: str,
    window_size: int,
    measurement_columns: list[str],
    issue_labels: list[int],
) -> pd.DataFrame:
    """Summarize contiguous row windows into one feature row per window.

    The row-level parquet cache is ideal for supervised row-wise modeling, but
    clustering and short-window analyses often benefit from one row per fixed
    time window instead. This function builds that second representation by
    computing:

    - window start/end timestamps
    - row count
    - issue count and issue rate
    - the modal target flag in the window
    - mean / std / min / max for each selected measurement column
    """

    work = frame.reset_index(drop=True).copy()
    work["window_id"] = work.index // window_size

    named_aggs: dict[str, tuple[str, str | callable]] = {
        "window_start": ("Time UTC", "min"),
        "window_end": ("Time UTC", "max"),
        "source_file": ("source_file", "first"),
        "row_count": ("Time UTC", "size"),
        "issue_count": (target_flag, lambda values: int(values.isin(issue_labels).sum())),
        "issue_rate": (target_flag, lambda values: float(values.isin(issue_labels).mean())),
        "target_mode": (
            target_flag,
            lambda values: int(values.mode(dropna=True).iloc[0])
            if not values.mode(dropna=True).empty
            else -1,
        ),
    }

    for column in measurement_columns:
        named_aggs[f"{column}_mean"] = (column, "mean")
        named_aggs[f"{column}_std"] = (column, "std")
        named_aggs[f"{column}_min"] = (column, "min")
        named_aggs[f"{column}_max"] = (column, "max")

    return work.groupby("window_id", sort=True).agg(**named_aggs).reset_index(drop=True)


def clear_old_outputs(paths: CacheBundlePaths) -> None:
    """Remove stale bundle outputs before writing a fresh preparation run.

    Preparation is designed to be rerunnable. Clearing the previous bundle
    contents up front prevents old parquet parts or metadata from silently
    surviving across runs and confusing later notebook reads.
    """

    if paths.row_level_dir.exists():
        for existing in paths.row_level_dir.glob("*.parquet"):
            existing.unlink()

    for existing_file in (paths.window_cache_path, paths.metadata_path):
        if existing_file.exists():
            existing_file.unlink()


def _interval_overlap_seconds(left: CsvFileInfo, right: CsvFileInfo) -> float:
    """Return the overlapping duration between two filename intervals in seconds.

    A negative value is used as a sentinel when one or both filenames do not
    carry a usable interval. This helper feeds the ranking logic for selecting
    the best companion file across device families.
    """

    if left.start_time is None or left.end_time is None or right.start_time is None or right.end_time is None:
        return -1.0
    overlap_start = max(left.start_time, right.start_time)
    overlap_end = min(left.end_time, right.end_time)
    return max(0.0, float((overlap_end - overlap_start).total_seconds()))


def _choose_best_companion(primary: CsvFileInfo, candidates: list[CsvFileInfo]) -> CsvFileInfo | None:
    """Choose the best companion file for one primary interval.

    Preference order:
    1. same series key and exact filename interval match
    2. same series key and greatest interval overlap
    3. same series key and smallest start/end mismatch

    This logic is the clean replacement for the older "temporary hand-picked
    folder" workaround. Instead of assuming the first ``N`` files line up
    across devices, we explicitly choose the secondary file that best matches
    the selected primary interval.
    """

    same_series = [candidate for candidate in candidates if candidate.series_key == primary.series_key]
    if not same_series:
        return None

    exact_matches = [
        candidate
        for candidate in same_series
        if candidate.start_time == primary.start_time and candidate.end_time == primary.end_time
    ]
    if exact_matches:
        return exact_matches[0]

    ranked = sorted(
        same_series,
        key=lambda candidate: (
            -_interval_overlap_seconds(primary, candidate),
            abs(float(((candidate.start_time or pd.Timestamp(0, tz="UTC")) - (primary.start_time or pd.Timestamp(0, tz="UTC"))).total_seconds())),
            abs(float(((candidate.end_time or pd.Timestamp(0, tz="UTC")) - (primary.end_time or pd.Timestamp(0, tz="UTC"))).total_seconds())),
            candidate.path.name,
        ),
    )
    return ranked[0] if ranked else None


def select_time_aligned_paths(
    grouped_infos: dict[str, list[CsvFileInfo]],
    *,
    primary_device: str,
    max_files: int | None,
) -> dict[str, list[Path]]:
    """Select files, matching secondary devices to the chosen primary intervals.

    When ``max_files`` is provided, we first choose primary-device files, then
    choose the best companion CTD/oxygen/fluorometer files for those same
    filename time ranges. This keeps reduced runs schema-stable without needing
    a temporary hand-curated data directory.

    When ``max_files`` is ``None``, every discovered file is kept.
    """

    if max_files is None:
        return {device: [info.path for info in infos] for device, infos in grouped_infos.items()}

    limited: dict[str, list[Path]] = {device: [] for device in grouped_infos}
    primary_infos = grouped_infos.get(primary_device, [])[:max_files]
    limited[primary_device] = [info.path for info in primary_infos]

    for device, infos in grouped_infos.items():
        if device == primary_device:
            continue
        chosen: list[Path] = []
        seen: set[Path] = set()
        for primary_info in primary_infos:
            companion = _choose_best_companion(primary_info, infos)
            if companion and companion.path not in seen:
                chosen.append(companion.path)
                seen.add(companion.path)
        limited[device] = chosen

    return limited


def simulate_merged_columns(
    selected_infos: dict[str, list[CsvFileInfo]],
    *,
    primary_device: str,
) -> list[str]:
    """Predict the merged row-level column order for the selected file bundle.

    The full preparation pass writes one merged parquet part per primary file.
    To keep memory use low we no longer concatenate the full dataset before
    writing, but the notebooks still benefit from stable row-column metadata.
    This helper builds that expected merged schema from CSV headers alone by
    replaying the same secondary-device merge order and rename rules used in
    ``main()``.
    """

    merged_columns = ["Time UTC"]

    primary_paths = [info.path for info in selected_infos.get(primary_device, [])]
    for column in discover_available_columns(primary_paths):
        if column not in merged_columns:
            merged_columns.append(column)
    if "source_file" not in merged_columns:
        merged_columns.append("source_file")

    secondary_devices = [device for device in ("ctd", "fluorometer", "oxygen", "other") if device != primary_device]
    for device_name in secondary_devices:
        device_paths = [info.path for info in selected_infos.get(device_name, [])]
        if not device_paths:
            continue
        for column in discover_available_columns(device_paths):
            if column in {"Time UTC", "source_file"}:
                continue
            merged_name = f"{column} [{device_name}]" if column in merged_columns else column
            if merged_name not in merged_columns:
                merged_columns.append(merged_name)

    return merged_columns


def write_row_level_parquet_cache(
    *,
    data_root: Path,
    bundle_paths: CacheBundlePaths,
    target_flag: str,
    primary_device: str,
    max_files: int | None,
    sample_rows: int | None,
    requested_measurement_columns: list[str] | None,
    merge_tolerance_seconds: int,
    clear_existing: bool = True,
) -> RowLevelCacheResult:
    """Create the row-level parquet cache from raw scalar CSV files.

    This step owns the raw-data work:

    - discover available CSV columns,
    - choose primary and companion device files,
    - read and type CSV rows,
    - align secondary devices onto the primary timestamps,
    - write one row-level parquet part per primary CSV file.

    The output is still row-shaped: one row per timestamp.
    """

    data_root = data_root.expanduser().resolve()
    bundle_paths.root.mkdir(parents=True, exist_ok=True)
    bundle_paths.row_level_dir.mkdir(parents=True, exist_ok=True)
    if clear_existing:
        for existing in bundle_paths.row_level_dir.glob("*.parquet"):
            existing.unlink()

    all_csv_files = sorted(data_root.glob("*.csv"))
    if not all_csv_files:
        raise ValueError(f"No CSV files found in {data_root}")

    all_available_columns = discover_available_columns(all_csv_files)
    grouped_infos: dict[str, list[CsvFileInfo]] = {"ctd": [], "fluorometer": [], "oxygen": [], "other": []}
    for csv_path in all_csv_files:
        grouped_infos[detect_device(csv_path)].append(parse_file_info(csv_path))

    sort_fallback_time = pd.Timestamp("2262-04-11T23:47:16Z")
    for infos in grouped_infos.values():
        infos.sort(key=lambda info: (info.series_key, info.start_time or sort_fallback_time, info.path.name))

    limited_paths = select_time_aligned_paths(
        grouped_infos,
        primary_device=primary_device,
        max_files=max_files,
    )
    selected_infos = {
        device: [info for info in infos if info.path in set(limited_paths.get(device, []))]
        for device, infos in grouped_infos.items()
    }

    primary_infos = selected_infos.get(primary_device, [])
    if not primary_infos:
        raise ValueError(f"Expected at least one {primary_device!r} file based on primary_device")

    merge_tolerance = pd.Timedelta(seconds=merge_tolerance_seconds)
    all_columns = simulate_merged_columns(selected_infos, primary_device=primary_device)
    qc_columns = [column for column in all_columns if is_qc_like_column(column)]

    if target_flag not in all_columns:
        raise ValueError(f"Target flag '{target_flag}' not found after merge")

    measurement_columns = choose_measurement_columns(
        all_columns,
        requested_measurement_columns=requested_measurement_columns,
    )
    measurement_columns = [column for column in measurement_columns if column != target_flag]
    missing_measurement_columns = [
        column
        for column in (requested_measurement_columns or [])
        if column not in measurement_columns
    ]

    secondary_devices = [device for device in ("ctd", "fluorometer", "oxygen", "other") if device != primary_device]
    total_rows = 0
    target_counts: Counter[int] = Counter()
    processed_files: list[dict[str, object]] = []

    for index, primary_info in enumerate(primary_infos, start=1):
        merged = read_scalar_csv(primary_info.path, sample_rows).sort_values("Time UTC").reset_index(drop=True)

        for device_name in secondary_devices:
            device_infos = selected_infos.get(device_name, [])
            if not device_infos:
                continue

            companion_info = _choose_best_companion(primary_info, device_infos)
            if companion_info is None:
                continue

            secondary = read_scalar_csv(companion_info.path, sample_rows).sort_values("Time UTC").reset_index(drop=True)
            rename_map = {}
            for column in secondary.columns:
                if column in {"Time UTC", "source_file"}:
                    continue
                if column in merged.columns:
                    rename_map[column] = f"{column} [{device_name}]"
            if rename_map:
                secondary = secondary.rename(columns=rename_map)

            secondary = secondary.drop(columns=["source_file"], errors="ignore")
            merged = pd.merge_asof(
                merged.sort_values("Time UTC"),
                secondary.sort_values("Time UTC"),
                on="Time UTC",
                direction="nearest",
                tolerance=merge_tolerance,
            )

        merged = merged.sort_values("Time UTC").reset_index(drop=True)
        for column in all_columns:
            if column not in merged.columns:
                merged[column] = pd.NA
        merged = merged[[column for column in all_columns if column in merged.columns]]

        part_path = bundle_paths.row_level_dir / f"part-{index:03d}.parquet"
        merged.to_parquet(part_path, index=False)

        total_rows += len(merged)
        target_counts.update(merged[target_flag].dropna().astype(int).tolist())

        processed_files.append(
            {
                "source_file": primary_info.path.name,
                "row_count": int(len(merged)),
                "time_start": merged["Time UTC"].min().isoformat() if not merged.empty else None,
                "time_end": merged["Time UTC"].max().isoformat() if not merged.empty else None,
                "row_level_part": part_path.name,
            }
        )
        print(f"[row {index}] wrote {part_path} with {len(merged):,} rows")

    return RowLevelCacheResult(
        all_available_columns=all_available_columns,
        grouped_infos=grouped_infos,
        limited_paths=limited_paths,
        selected_infos=selected_infos,
        requested_measurement_columns=requested_measurement_columns,
        measurement_columns=measurement_columns,
        missing_measurement_columns=missing_measurement_columns,
        qc_columns=qc_columns,
        processed_files=processed_files,
        total_rows=int(total_rows),
        target_counts=target_counts,
        row_columns=all_columns,
    )


def write_window_level_parquet_cache(
    *,
    bundle_paths: CacheBundlePaths,
    processed_files: list[dict[str, object]],
    target_flag: str,
    window_size: int,
    measurement_columns: list[str],
    issue_labels: list[int],
) -> WindowLevelCacheResult:
    """Create the window-summary parquet cache from row-level parquet parts.

    This step does not read raw CSV files. It starts from the row-level parquet
    files and creates one summary row per fixed-size contiguous row window.
    """

    window_frames: list[pd.DataFrame] = []
    for index, file_info in enumerate(processed_files, start=1):
        row_part_path = bundle_paths.row_level_dir / Path(str(file_info["row_level_part"])).name
        row_frame = pd.read_parquet(row_part_path)
        window_frame = build_window_features(
            row_frame,
            target_flag=target_flag,
            window_size=window_size,
            measurement_columns=measurement_columns,
            issue_labels=issue_labels,
        )
        window_frames.append(window_frame)
        print(f"[window {index}] summarized {row_part_path.name} into {len(window_frame):,} windows")

    if window_frames:
        window_cache_frame = pd.concat(window_frames, ignore_index=True)
    else:
        window_cache_frame = pd.DataFrame()
    window_cache_frame.to_parquet(bundle_paths.window_cache_path, index=False)
    return WindowLevelCacheResult(
        window_count=int(len(window_cache_frame)),
        window_columns=window_cache_frame.columns.tolist(),
    )


def write_cache_metadata(
    *,
    bundle_paths: CacheBundlePaths,
    row_result: RowLevelCacheResult,
    window_result: WindowLevelCacheResult,
    target_flag: str,
    sample_rows: int | None,
    window_size: int,
    issue_labels: list[int],
    merge_tolerance_seconds: int,
    primary_device: str,
    file_selection_strategy: str,
) -> dict[str, object]:
    """Write metadata describing the row-level and window-level parquet cache."""

    metadata = {
        "target_flag": target_flag,
        "cache_root": str(bundle_paths.root),
        "cache_stem": bundle_paths.stem,
        "measurement_columns": row_result.measurement_columns,
        "requested_measurement_columns": row_result.requested_measurement_columns or [],
        "missing_measurement_columns": row_result.missing_measurement_columns,
        "qc_columns": row_result.qc_columns,
        "processed_file_count": len(row_result.processed_files),
        "row_count": int(row_result.total_rows),
        "window_count": window_result.window_count,
        "sample_rows_per_file": sample_rows,
        "window_size": window_size,
        "issue_labels": issue_labels,
        "target_distribution": {str(key): int(value) for key, value in sorted(row_result.target_counts.items())},
        "issue_fraction": (
            float(sum(row_result.target_counts.get(flag, 0) for flag in issue_labels) / row_result.total_rows)
            if row_result.total_rows
            else 0.0
        ),
        "processed_files": row_result.processed_files,
        "row_level_cache": str(bundle_paths.row_level_dir),
        "window_cache": str(bundle_paths.window_cache_path),
        "row_columns": row_result.row_columns,
        "window_columns": window_result.window_columns,
        "device_file_counts": {device: len(infos) for device, infos in row_result.grouped_infos.items()},
        "limited_device_file_counts": {device: len(paths) for device, paths in row_result.limited_paths.items()},
        "file_selection_strategy": file_selection_strategy,
        "merge_tolerance_seconds": merge_tolerance_seconds,
        "primary_device": primary_device,
        "all_available_columns": row_result.all_available_columns,
    }
    bundle_paths.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    """Run the full scalar cache-preparation pipeline.

    High-level flow
    ----------------
    1. parse command-line settings
    2. discover all CSV files and their available columns
    3. optionally reduce the file set while keeping secondary devices aligned
    4. read and merge the selected device streams
    5. choose the measurement columns to preserve
    6. write row-level parquet parts
    7. build and write the window-summary parquet
    8. write metadata describing exactly what was produced

    The notebooks rely heavily on the metadata file, so this function is also
    responsible for recording enough context to make later reads transparent.
    """

    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    bundle_paths = build_cache_bundle_paths(args.cache_root, args.cache_stem)
    bundle_paths.root.mkdir(parents=True, exist_ok=True)
    requested_measurement_columns = parse_measurement_columns(args.measurement_columns)
    clear_old_outputs(bundle_paths)
    issue_labels = sorted(dict.fromkeys(args.issue_labels or DEFAULT_ISSUE_LABELS))
    try:
        row_result = write_row_level_parquet_cache(
            data_root=data_root,
            bundle_paths=bundle_paths,
            target_flag=args.target_flag,
            primary_device=args.primary_device,
            max_files=args.max_files,
            sample_rows=args.sample_rows,
            requested_measurement_columns=requested_measurement_columns,
            merge_tolerance_seconds=args.merge_tolerance_seconds,
            clear_existing=False,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    window_result = write_window_level_parquet_cache(
        bundle_paths=bundle_paths,
        processed_files=row_result.processed_files,
        target_flag=args.target_flag,
        window_size=args.window_size,
        measurement_columns=row_result.measurement_columns,
        issue_labels=issue_labels,
    )

    # The metadata file is what makes the prepared cache self-describing. The
    # notebooks use it to discover measurement columns, cache stems, source-file
    # coverage, target distributions, and other context without hard-coding
    # assumptions about a specific dataset.
    write_cache_metadata(
        bundle_paths=bundle_paths,
        row_result=row_result,
        window_result=window_result,
        target_flag=args.target_flag,
        sample_rows=args.sample_rows,
        window_size=args.window_size,
        issue_labels=issue_labels,
        merge_tolerance_seconds=args.merge_tolerance_seconds,
        primary_device=args.primary_device,
        file_selection_strategy="primary_time_aligned_selection" if args.max_files is not None else "all_files",
    )

    print(
        json.dumps(
            {
                "row_level_cache": str(bundle_paths.row_level_dir),
                "window_cache": str(bundle_paths.window_cache_path),
                "metadata": str(bundle_paths.metadata_path),
                "cache_stem": bundle_paths.stem,
                "rows": int(row_result.total_rows),
                "windows": window_result.window_count,
                "target_distribution": {str(key): int(value) for key, value in sorted(row_result.target_counts.items())},
                "device_file_counts": {device: len(infos) for device, infos in row_result.grouped_infos.items()},
                "limited_device_file_counts": {device: len(paths) for device, paths in row_result.limited_paths.items()},
                "missing_measurement_columns": row_result.missing_measurement_columns,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
