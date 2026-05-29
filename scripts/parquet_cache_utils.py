from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .onc_scalar_cache_utils import (
    create_onc_row_level_parquet_cache,
    create_onc_window_summary_parquet_cache,
)
from .prepare_scalar_session1_data import build_cache_bundle_paths, read_scalar_csv

__all__ = [
    "csv_files_to_row_parquet_cache",
    "resolve_csv_paths",
    "resolve_or_create_parquet_cache",
]


def resolve_csv_paths(
    *,
    raw_data_dir: str | Path | None = None,
    raw_csv_paths: Iterable[str | Path] | None = None,
) -> list[Path]:
    """Resolve CSV inputs from either an explicit file list or a directory.

    Parameters
    ----------
    raw_data_dir:
        Directory containing ``*.csv`` files. Ignored when ``raw_csv_paths`` is
        provided.
    raw_csv_paths:
        Optional explicit list of CSV files. Use this when files are spread
        across directories or when you want a specific order/subset.

    Returns
    -------
    list[Path]
        Existing-looking CSV paths, sorted for reproducible cache builds.
    """

    if raw_csv_paths:
        return sorted(Path(path).expanduser() for path in raw_csv_paths)
    if raw_data_dir is None:
        return []
    raw_path = Path(raw_data_dir).expanduser()
    return sorted(raw_path.glob("*.csv")) if raw_path.exists() else []


def csv_files_to_row_parquet_cache(
    csv_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    cache_name: str = "scalar_cache",
    columns: list[str] | None = None,
    required_columns: list[str] | tuple[str, ...] | None = None,
    target_column: str | None = None,
    issue_labels: list[int] | tuple[int, ...] | None = None,
    time_column: str = "Time UTC",
    source_file_column: str = "source_file",
    sample_rows_per_file: int | None = None,
    header: str | int = "auto",
    force: bool = False,
    compression: str = "zstd",
) -> dict[str, object]:
    """Convert CSV files into a reusable row-level parquet cache.

    This is the simplest cache builder for participant-owned datasets. Each CSV
    file becomes one parquet part, all rows get a canonical ``"Time UTC"``
    timestamp column, and the metadata JSON records source files, row counts,
    label distribution, and output columns.

    Parameters
    ----------
    csv_paths:
        CSV files to convert. Each file becomes one parquet part.
    output_dir:
        Directory where the cache folder and metadata file will be written.
    cache_name:
        Cache stem used for the row-level folder and metadata filename.
    columns:
        Optional ordered subset of columns to read and keep. The time column is
        always included, and the source-file column is added after reading.
    required_columns:
        Columns that must be present in every CSV. If omitted, ``columns`` are
        treated as the required input columns.
    target_column:
        Optional label column to summarize in the cache metadata.
    issue_labels:
        Optional label ids counted as issues when ``target_column`` is present.
    time_column:
        Timestamp column to parse, sort by, and preserve in the parquet output.
    source_file_column:
        Name of the output column that records which CSV each row came from.
    sample_rows_per_file:
        Optional row cap per CSV, useful for quick test runs.
    header:
        ``"auto"`` uses ONC-style metadata/header detection, ``"first_row"``
        uses the first CSV row as the header, and an integer uses that
        zero-based row number as the header while skipping rows above it.
    force:
        If ``True``, replace an existing cache with the same name.
    compression:
        Parquet compression codec passed to pandas.

    Returns
    -------
    dict
        Metadata describing the generated cache.
    """

    csv_path_list = sorted(Path(path).expanduser() for path in csv_paths)
    if not csv_path_list:
        raise ValueError("csv_paths must contain at least one CSV file.")

    canonical_time_column = "Time UTC"
    bundle_paths = build_cache_bundle_paths(Path(output_dir), cache_name)
    if (bundle_paths.row_level_dir.exists() or bundle_paths.metadata_path.exists()) and not force:
        raise FileExistsError(
            f"Cache '{cache_name}' already exists under {bundle_paths.root}. "
            "Pass force=True to rebuild it."
        )

    if force:
        shutil.rmtree(bundle_paths.row_level_dir, ignore_errors=True)
        bundle_paths.metadata_path.unlink(missing_ok=True)
        bundle_paths.window_cache_path.unlink(missing_ok=True)
    bundle_paths.root.mkdir(parents=True, exist_ok=True)
    bundle_paths.row_level_dir.mkdir(parents=True, exist_ok=True)

    requested_columns = [
        column
        for column in dict.fromkeys(columns or [])
        if column != source_file_column
    ]
    required_input_columns = [
        column
        for column in dict.fromkeys(required_columns or requested_columns)
        if column != source_file_column
    ]
    if time_column not in required_input_columns:
        required_input_columns.insert(0, time_column)
    if time_column not in requested_columns:
        requested_columns.insert(0, time_column)
    read_columns = (
        list(dict.fromkeys([*requested_columns, *required_input_columns]))
        if columns is not None
        else None
    )

    processed_files: list[dict[str, object]] = []
    total_rows = 0
    all_output_columns: list[str] = []
    target_distribution: dict[int, int] = {}

    for index, csv_path in enumerate(csv_path_list, start=1):
        frame = _read_csv_for_row_parquet_cache(
            csv_path,
            header=header,
            columns=read_columns,
            required_columns=required_input_columns,
            sample_rows_per_file=sample_rows_per_file,
            time_column=time_column,
        )

        # Use one canonical timestamp column in the parquet cache so the ML code
        # can be reused across datasets with different original timestamp names.
        frame = frame.copy()
        if source_file_column != "source_file" and "source_file" in frame.columns:
            frame = frame.drop(columns=["source_file"])
        frame[canonical_time_column] = pd.to_datetime(frame[time_column], utc=True, errors="coerce", format="ISO8601")
        if time_column != canonical_time_column and time_column in frame.columns:
            frame = frame.drop(columns=[time_column])

        # Convert non-time, non-source columns to numeric values because the
        # downstream modelling examples expect numeric feature/label columns.
        for column in [column for column in frame.columns if column != canonical_time_column]:
            if column != source_file_column:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=[canonical_time_column]).sort_values(canonical_time_column).reset_index(drop=True)
        frame[source_file_column] = csv_path.name

        if columns is not None:
            output_columns = [canonical_time_column]
            output_columns.extend(
                column
                for column in requested_columns
                if column in frame.columns and column != canonical_time_column
            )
            if source_file_column not in output_columns:
                output_columns.append(source_file_column)
            frame = frame[output_columns]

        if target_column and target_column in frame.columns:
            target_counts = frame[target_column].dropna().astype(int).value_counts()
            for label, count in target_counts.items():
                target_distribution[int(label)] = target_distribution.get(int(label), 0) + int(count)

        for column in frame.columns:
            if column not in all_output_columns:
                all_output_columns.append(column)

        part_path = bundle_paths.row_level_dir / f"part-{index:03d}.parquet"
        frame.to_parquet(part_path, index=False, compression=compression)

        row_count = int(len(frame))
        total_rows += row_count
        processed_files.append(
            {
                "source_file": csv_path.name,
                "source_path": str(csv_path),
                "row_count": row_count,
                "time_start": frame[canonical_time_column].min().isoformat() if row_count else None,
                "time_end": frame[canonical_time_column].max().isoformat() if row_count else None,
                "row_level_part": part_path.name,
            }
        )

    normalized_issue_labels = [int(label) for label in (issue_labels or [])]
    issue_count = sum(target_distribution.get(label, 0) for label in normalized_issue_labels)

    # Keep metadata portable between laptops, clusters, and scratch directories.
    # The caller resolves the active cache directory at runtime.
    metadata: dict[str, object] = {
        "cache_root": ".",
        "cache_path_base": "metadata_directory",
        "cache_stem": bundle_paths.stem,
        "target_flag": target_column,
        "row_level_cache": bundle_paths.row_level_dir.name,
        "metadata_path": bundle_paths.metadata_path.name,
        "processed_file_count": len(processed_files),
        "row_count": int(total_rows),
        "window_count": 0,
        "target_distribution": {str(label): int(count) for label, count in sorted(target_distribution.items())},
        "issue_fraction": float(issue_count / total_rows) if total_rows else 0.0,
        "sample_rows_per_file": sample_rows_per_file,
        "header": header,
        "time_column": time_column,
        "canonical_time_column": canonical_time_column,
        "source_file_column": source_file_column,
        "row_columns": all_output_columns,
        "processed_files": processed_files,
        "part_to_source_file": {
            str(file_info["row_level_part"]): str(file_info["source_file"])
            for file_info in processed_files
        },
    }
    bundle_paths.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def resolve_or_create_parquet_cache(
    *,
    cache_root: str | Path,
    cache_bundle_name: str,
    target_flag: str,
    measurement_columns: list[str] | tuple[str, ...],
    issue_labels: list[int] | tuple[int, ...],
    raw_data_dir: str | Path | None = None,
    raw_csv_paths: Iterable[str | Path] | None = None,
    optional_qc_columns: list[str] | tuple[str, ...] | None = None,
    time_column: str = "Time UTC",
    csv_header: str | int = "auto",
    build_cache_if_missing: bool = False,
    force_rebuild_cache: bool = False,
    generic_csv_cache: bool = False,
    primary_device: str = "ctd",
    max_files: int | None = None,
    sample_rows: int | None = None,
    window_size: int = 256,
    merge_tolerance_seconds: int = 5,
) -> dict[str, object]:
    """Return an existing parquet cache or build one from CSV files.

    Use ``generic_csv_cache=True`` for ordinary CSV files. That path writes a
    row-level parquet cache directly from one or more CSVs. Use
    ``generic_csv_cache=False`` for the ONC scalar prep path, which can align
    companion device streams and build both row-level and window-summary caches.

    The function returns a dictionary with ``cache_paths`` and ``summary`` in
    every branch, so notebooks and batch scripts can print the same compact
    status information whether they reused a cache or built a new one.
    """

    cache_root_path = Path(cache_root).expanduser()
    cache_paths = build_cache_bundle_paths(cache_root_path, cache_bundle_name)
    metadata_exists = cache_paths.metadata_path.exists()
    if metadata_exists and not force_rebuild_cache:
        return {
            "cache_paths": cache_paths,
            "cache_exists": True,
            "cache_built": False,
            "cache_mode": "existing",
            "summary": {
                "cache_status": "using_existing_cache",
                "metadata": str(cache_paths.metadata_path),
                "row_level_cache": str(cache_paths.row_level_dir),
                "window_cache": str(cache_paths.window_cache_path),
            },
        }

    if not build_cache_if_missing and not force_rebuild_cache:
        return {
            "cache_paths": cache_paths,
            "cache_exists": False,
            "cache_built": False,
            "cache_mode": "missing",
            "summary": {
                "cache_status": "missing_cache_not_built",
                "metadata": str(cache_paths.metadata_path),
                "message": "Set build_cache_if_missing=True or force_rebuild_cache=True to build from CSV files.",
            },
        }

    csv_paths = resolve_csv_paths(raw_data_dir=raw_data_dir, raw_csv_paths=raw_csv_paths)
    if max_files is not None:
        csv_paths = csv_paths[: int(max_files)]
    if not csv_paths:
        raise FileNotFoundError(
            "No CSV files were found for cache building. Provide raw_data_dir or raw_csv_paths."
        )

    if generic_csv_cache:
        keep_columns = list(
            dict.fromkeys(
                [
                    time_column,
                    target_flag,
                    *(optional_qc_columns or []),
                    *measurement_columns,
                ]
            )
        )
        metadata = csv_files_to_row_parquet_cache(
            csv_paths,
            cache_root_path,
            cache_name=cache_bundle_name,
            columns=keep_columns,
            required_columns=[time_column, target_flag, *measurement_columns],
            target_column=target_flag,
            issue_labels=issue_labels,
            time_column=time_column,
            sample_rows_per_file=sample_rows,
            header=csv_header,
            force=True,
        )
        return {
            "cache_paths": cache_paths,
            "cache_exists": True,
            "cache_built": True,
            "cache_mode": "generic_csv_row_cache",
            "metadata": metadata,
            "summary": {
                "cache_status": "built_generic_row_level_cache",
                "csv_files": len(csv_paths),
                "rows": metadata["row_count"],
                "metadata": str(cache_paths.metadata_path),
                "row_level_cache": str(cache_paths.row_level_dir),
            },
        }

    if raw_data_dir is None:
        raw_data_dir = Path(csv_paths[0]).parent
    row_cache_result = create_onc_row_level_parquet_cache(
        raw_data_dir=raw_data_dir,
        cache_root=cache_root_path,
        cache_bundle_name=cache_bundle_name,
        target_flag=target_flag,
        primary_device=primary_device,
        measurement_columns=measurement_columns,
        max_files=max_files,
        sample_rows=sample_rows,
        merge_tolerance_seconds=merge_tolerance_seconds,
    )
    window_cache_result = create_onc_window_summary_parquet_cache(
        row_cache_result,
        target_flag=target_flag,
        issue_labels=issue_labels,
        window_size=window_size,
        sample_rows=sample_rows,
        merge_tolerance_seconds=merge_tolerance_seconds,
        primary_device=primary_device,
        max_files=max_files,
    )
    return {
        "cache_paths": cache_paths,
        "cache_exists": True,
        "cache_built": True,
        "cache_mode": "onc_row_and_window_cache",
        "row_cache_result": row_cache_result,
        "window_cache_result": window_cache_result,
        "summary": {
            "cache_status": "built_onc_row_and_window_cache",
            "row_summary": row_cache_result["summary"],
            "window_summary": window_cache_result["summary"],
        },
    }


def _read_csv_for_row_parquet_cache(
    csv_path: Path,
    *,
    header: str | int,
    columns: list[str] | None,
    required_columns: list[str],
    sample_rows_per_file: int | None,
    time_column: str,
) -> pd.DataFrame:
    """Read one CSV using the requested header style before parquet export."""

    if header == "auto":
        frame = read_scalar_csv(
            csv_path,
            sample_rows=sample_rows_per_file,
            required_columns=columns,
            allow_missing_columns=False,
        )
        missing_columns = [column for column in required_columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"Missing expected columns in {csv_path}: {missing_columns}")
        return frame

    if header == "first_row":
        header_row = 0
    elif isinstance(header, int):
        header_row = int(header)
    else:
        raise ValueError('header must be "auto", "first_row", or an integer row number.')
    if header_row < 0:
        raise ValueError("header row number must be zero or greater.")

    frame = pd.read_csv(
        csv_path,
        header=header_row,
        usecols=columns,
        nrows=sample_rows_per_file,
        low_memory=False,
    )
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns in {csv_path}: {missing_columns}")
    if time_column not in frame.columns:
        raise ValueError(f"Missing {time_column} in {csv_path}")
    return frame
