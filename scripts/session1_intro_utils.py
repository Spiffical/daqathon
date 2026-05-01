from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from IPython.display import display
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import ConfusionMatrixDisplay

from . import prepare_scalar_session1_data
from .prepare_scalar_session1_data import locate_header, read_scalar_csv
from .session1_modeling import (
    DEFAULT_FLAG_PALETTE,
    build_labeled_intervals,
    build_model_frame,
    build_reviewed_target_frame,
    build_cache_bundle_paths,
    build_sequence_label_interval_data,
    build_window_classification_interval_data,
    compute_interval_classification_metrics,
    compute_split_share_gap,
    infer_interval_origin,
    load_full_row_level_frame,
    load_rows_for_time_range,
    merge_adjacent_intervals,
    plot_time_series_with_bands,
    predict_cnn_window_model,
    predict_sequence_label_cnn,
    report_average,
    reviewed_label_mask,
    resolve_cache_bundle_paths,
    select_time_range,
    split_frame_by_strategy,
    summarize_split_distributions,
    summarize_target_by_time_bin,
)


def choose_cache_bundle_paths(
    cache_roots: list[str | Path | None],
    *,
    cache_stem: str,
):
    """Return the first cache bundle that exists, or a fallback under the first root."""

    fallback_root: Path | None = None
    for cache_root in cache_roots:
        if cache_root is None:
            continue
        root_path = Path(cache_root).expanduser()
        if fallback_root is None:
            fallback_root = root_path
        candidate = resolve_cache_bundle_paths(root_path, cache_stem=cache_stem)
        if candidate.metadata_path.exists():
            return candidate
    if fallback_root is None:
        raise ValueError("At least one cache root is required")
    return build_cache_bundle_paths(fallback_root, cache_stem=cache_stem)


def read_parquet_head(
    parquet_path: str | Path,
    *,
    columns: list[str] | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Read the first rows of a parquet file without loading it all into memory."""

    parquet_path = Path(parquet_path)
    if max_rows is None:
        return pd.read_parquet(parquet_path, columns=columns)

    parquet_file = pq.ParquetFile(parquet_path)
    batch_size = max(1, min(int(max_rows), 65536))
    frames: list[pd.DataFrame] = []
    rows_remaining = int(max_rows)

    for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
        if rows_remaining <= 0:
            break
        frame = batch.to_pandas()
        if len(frame) > rows_remaining:
            frame = frame.iloc[:rows_remaining].copy()
        frames.append(frame)
        rows_remaining -= len(frame)

    if not frames:
        return pd.DataFrame(columns=columns or parquet_file.schema.names)
    return pd.concat(frames, ignore_index=True)


def create_session1_row_level_parquet_cache(
    *,
    raw_data_dir: str | Path,
    cache_root: str | Path,
    cache_bundle_name: str,
    target_flag: str,
    primary_device: str,
    measurement_columns: list[str] | tuple[str, ...],
    max_files: int | None = None,
    sample_rows: int | None = None,
    merge_tolerance_seconds: int = 5,
) -> dict[str, object]:
    """Create the row-level Session 1 parquet cache from raw CSV files.

    This function writes one row-level parquet part per selected primary CSV.
    If the dataset has companion CTD/oxygen/fluorometer files, they are aligned
    onto the primary timestamps before the parquet parts are written.
    """

    cache_paths = prepare_scalar_session1_data.build_cache_bundle_paths(
        Path(cache_root),
        cache_bundle_name,
    )
    prepare_scalar_session1_data.clear_old_outputs(cache_paths)
    row_result = prepare_scalar_session1_data.write_row_level_parquet_cache(
        data_root=Path(raw_data_dir),
        bundle_paths=cache_paths,
        target_flag=target_flag,
        primary_device=primary_device,
        max_files=max_files,
        sample_rows=sample_rows,
        requested_measurement_columns=list(measurement_columns),
        merge_tolerance_seconds=merge_tolerance_seconds,
        clear_existing=False,
    )
    return {
        "row_result": row_result,
        "cache_paths": cache_paths,
        "summary": {
            "step": "row_level_parquet",
            "row_level_cache": str(cache_paths.row_level_dir),
            "row_parts": len(row_result.processed_files),
            "rows": int(row_result.total_rows),
            "measurement_columns": row_result.measurement_columns,
            "missing_measurement_columns": row_result.missing_measurement_columns,
        },
    }


def create_session1_window_level_parquet_cache(
    row_cache_result: dict[str, object],
    *,
    target_flag: str,
    issue_labels: list[int] | tuple[int, ...],
    window_size: int = 256,
    sample_rows: int | None = None,
    merge_tolerance_seconds: int = 5,
    primary_device: str,
    max_files: int | None = None,
) -> dict[str, object]:
    """Create the Session 1 window-summary parquet from row-level parquet parts."""

    row_result = row_cache_result["row_result"]
    cache_paths = row_cache_result["cache_paths"]
    issue_label_list = sorted(dict.fromkeys(int(label) for label in issue_labels))
    window_result = prepare_scalar_session1_data.write_window_level_parquet_cache(
        bundle_paths=cache_paths,
        processed_files=row_result.processed_files,
        target_flag=target_flag,
        window_size=window_size,
        measurement_columns=row_result.measurement_columns,
        issue_labels=issue_label_list,
    )
    metadata = prepare_scalar_session1_data.write_cache_metadata(
        bundle_paths=cache_paths,
        row_result=row_result,
        window_result=window_result,
        target_flag=target_flag,
        sample_rows=sample_rows,
        window_size=window_size,
        issue_labels=issue_label_list,
        merge_tolerance_seconds=merge_tolerance_seconds,
        primary_device=primary_device,
        file_selection_strategy="primary_time_aligned_selection" if max_files is not None else "all_files",
    )
    return {
        "window_result": window_result,
        "metadata": metadata,
        "cache_paths": cache_paths,
        "summary": {
            "step": "window_level_parquet",
            "window_cache": str(cache_paths.window_cache_path),
            "metadata": str(cache_paths.metadata_path),
            "windows": int(window_result.window_count),
            "window_columns": window_result.window_columns,
        },
    }


def csv_files_to_parquet_cache(
    csv_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    cache_name: str = "scalar_session1",
    columns: list[str] | None = None,
    required_columns: list[str] | tuple[str, ...] | None = None,
    time_column: str = "Time UTC",
    source_file_column: str = "source_file",
    sample_rows_per_file: int | None = None,
    header: str | int = "auto",
    force: bool = False,
    compression: str = "zstd",
) -> dict[str, object]:
    """Convert one or more scalar CSV files into a row-level parquet cache.

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
    time_column:
        Timestamp column to parse, sort by, and preserve in the parquet output.
    source_file_column:
        Name of the output column that records which CSV each row came from.
    sample_rows_per_file:
        Optional row cap per CSV, useful for quick test runs.
    header:
        ``"auto"`` uses the ONC metadata/header detector, ``"first_row"`` uses
        the first CSV row as the header, and an integer uses that zero-based row
        number as the header while skipping rows above it.
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

    bundle_paths = build_cache_bundle_paths(output_dir, cache_name)
    if (bundle_paths.row_level_dir.exists() or bundle_paths.metadata_path.exists()) and not force:
        raise FileExistsError(
            f"Cache '{cache_name}' already exists under {bundle_paths.root}. "
            "Pass force=True to rebuild it."
        )

    if force:
        shutil.rmtree(bundle_paths.row_level_dir, ignore_errors=True)
        bundle_paths.metadata_path.unlink(missing_ok=True)
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

    for index, csv_path in enumerate(csv_path_list, start=1):
        frame = _read_csv_for_parquet_cache(
            csv_path,
            header=header,
            columns=read_columns,
            required_columns=required_input_columns,
            sample_rows_per_file=sample_rows_per_file,
            time_column=time_column,
        )
        frame = frame.copy()
        if source_file_column != "source_file" and "source_file" in frame.columns:
            frame = frame.drop(columns=["source_file"])
        frame[time_column] = pd.to_datetime(frame[time_column], utc=True, errors="coerce", format="ISO8601")
        for column in [column for column in frame.columns if column != time_column]:
            if column != source_file_column:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=[time_column]).sort_values(time_column).reset_index(drop=True)
        frame[source_file_column] = csv_path.name

        if columns is not None:
            output_columns = [column for column in requested_columns if column in frame.columns]
            if source_file_column not in output_columns:
                output_columns.append(source_file_column)
            frame = frame[output_columns]

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
                "time_start": frame[time_column].min().isoformat() if row_count else None,
                "time_end": frame[time_column].max().isoformat() if row_count else None,
                "row_level_part": part_path.name,
            }
        )

    metadata: dict[str, object] = {
        "cache_root": str(bundle_paths.root),
        "cache_stem": bundle_paths.stem,
        "row_level_cache": str(bundle_paths.row_level_dir),
        "metadata_path": str(bundle_paths.metadata_path),
        "processed_file_count": len(processed_files),
        "row_count": int(total_rows),
        "sample_rows_per_file": sample_rows_per_file,
        "header": header,
        "time_column": time_column,
        "source_file_column": source_file_column,
        "row_columns": all_output_columns,
        "processed_files": processed_files,
        "part_to_source_file": {
            str(file_info["row_level_part"]): str(file_info["source_file"])
            for file_info in processed_files
        },
    }
    bundle_paths.metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def _read_csv_for_parquet_cache(
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


def evenly_spaced_take(
    frame: pd.DataFrame,
    limit: int | None,
    *,
    time_column: str | None = None,
) -> pd.DataFrame:
    """Take rows spread across time instead of only the earliest rows."""

    if time_column is not None:
        frame = frame.sort_values(time_column)
    if limit is None or len(frame) <= limit:
        return frame.reset_index(drop=True)
    indices = np.linspace(0, len(frame) - 1, num=limit, dtype=int)
    return frame.iloc[indices].reset_index(drop=True)


def derive_fractional_row_limit(
    row_count: int,
    data_fraction: float,
) -> int | None:
    """Return a row limit that means "this fraction of the available rows"."""

    if not 0 < data_fraction <= 1:
        raise ValueError("data_fraction must be in the interval (0, 1].")
    if data_fraction >= 0.999:
        return None
    return min(int(row_count), max(1, int(round(int(row_count) * data_fraction))))


def select_part_paths(part_paths: list[Path], limit: int | None, mode: str) -> list[Path]:
    """Choose parquet or CSV files either sequentially or spread through time."""

    if limit is None or limit >= len(part_paths):
        return part_paths
    if mode == "first":
        return part_paths[:limit]
    if mode == "spread":
        indices = np.linspace(0, len(part_paths) - 1, num=limit, dtype=int)
        selected: list[Path] = []
        seen: set[Path] = set()
        for index in indices:
            candidate = part_paths[int(index)]
            if candidate not in seen:
                selected.append(candidate)
                seen.add(candidate)
        return selected
    raise ValueError(f"Unsupported file selection mode: {mode}")


def load_raw_scalar_sample(
    csv_paths: list[Path],
    sample_rows_per_file: int | None,
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load a small raw CSV sample without reading every full file."""

    row_frames = []
    for path in csv_paths:
        frame = read_scalar_csv(
            path,
            sample_rows=sample_rows_per_file,
            required_columns=columns,
            allow_missing_columns=True,
        ).sort_values("Time UTC").reset_index(drop=True)
        row_frames.append(frame)

    return pd.concat(row_frames, ignore_index=True).sort_values("Time UTC").reset_index(drop=True)


def filter_csv_paths_with_required_columns(
    csv_paths: list[Path],
    required_columns: list[str] | tuple[str, ...],
) -> list[Path]:
    """Keep only raw CSVs that contain every required column."""

    required = [column for column in required_columns if column]
    matching_paths = []
    for path in csv_paths:
        _, csv_columns = locate_header(path)
        if all(column in csv_columns for column in required):
            matching_paths.append(path)
    return matching_paths


def load_raw_flag_context_sample(
    csv_paths: list[Path],
    *,
    target_flag: str,
    classes: list[int] | tuple[int, ...],
    context_rows_per_class: int,
    columns: list[str] | None = None,
    chunk_rows: int | None = None,
    max_context_windows_per_class: int = 4,
) -> pd.DataFrame:
    """Pull local raw-data context windows around requested QC or target labels."""

    requested_classes = [int(flag) for flag in classes]
    requested_classes = list(dict.fromkeys(requested_classes))
    context_rows_per_class = max(int(context_rows_per_class), 1)
    max_context_windows_per_class = max(int(max_context_windows_per_class), 1)
    chunk_rows = max(int(chunk_rows or max(context_rows_per_class * 2, 50000)), context_rows_per_class)
    use_columns = list(dict.fromkeys(["Time UTC", target_flag, *(columns or [])]))
    context_frames = []
    class_window_counts = {flag: 0 for flag in requested_classes}
    seen_spans: set[tuple[str, int, str, str]] = set()

    def contiguous_spans(row_indices: list[int]) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        if not row_indices:
            return spans
        span_start = row_indices[0]
        previous_index = row_indices[0]
        for row_index in row_indices[1:]:
            if row_index == previous_index + 1:
                previous_index = row_index
                continue
            spans.append((span_start, previous_index))
            span_start = row_index
            previous_index = row_index
        spans.append((span_start, previous_index))
        return spans

    for path in csv_paths:
        if all(count >= max_context_windows_per_class for count in class_window_counts.values()):
            break

        header_line_number, csv_columns = locate_header(path)
        available_columns = [column for column in use_columns if column in csv_columns]
        if "Time UTC" not in available_columns or target_flag not in available_columns:
            continue
        trailing_context = pd.DataFrame(columns=available_columns)

        for chunk in pd.read_csv(
            path,
            header=None,
            names=csv_columns,
            skiprows=header_line_number,
            usecols=available_columns,
            chunksize=chunk_rows,
            low_memory=False,
        ):
            chunk["Time UTC"] = pd.to_datetime(chunk["Time UTC"], utc=True, errors="coerce", format="ISO8601")
            for column in [column for column in chunk.columns if column != "Time UTC"]:
                chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
            chunk = chunk.dropna(subset=["Time UTC"]).sort_values("Time UTC").reset_index(drop=True)
            chunk["source_file"] = path.name
            if chunk.empty:
                continue

            combined = pd.concat([trailing_context, chunk], ignore_index=True) if not trailing_context.empty else chunk.copy()
            combined_flags = combined[target_flag].fillna(-1).astype(int)

            for flag in requested_classes:
                if class_window_counts[flag] >= max_context_windows_per_class:
                    continue
                flag_positions = combined.index[combined_flags == flag].tolist()
                if not flag_positions:
                    continue

                for span_start_index, span_end_index in contiguous_spans(flag_positions):
                    span_start_time = combined.iloc[span_start_index]["Time UTC"]
                    span_end_time = combined.iloc[span_end_index]["Time UTC"]
                    span_key = (path.name, flag, span_start_time.isoformat(), span_end_time.isoformat())
                    if span_key in seen_spans:
                        continue

                    span_midpoint_index = (span_start_index + span_end_index) // 2
                    start = max(span_midpoint_index - context_rows_per_class // 2, 0)
                    stop = min(start + context_rows_per_class, len(combined))
                    if stop - start < context_rows_per_class:
                        start = max(stop - context_rows_per_class, 0)

                    context_frames.append(combined.iloc[start:stop].copy())
                    seen_spans.add(span_key)
                    class_window_counts[flag] += 1
                    if class_window_counts[flag] >= max_context_windows_per_class:
                        break

            trailing_context = combined.iloc[-context_rows_per_class:].copy()

    if not context_frames:
        return pd.DataFrame(columns=use_columns + ["source_file"])
    sample = pd.concat(context_frames, ignore_index=True).sort_values("Time UTC").reset_index(drop=True)
    duplicate_columns = [column for column in ["source_file", "Time UTC"] if column in sample.columns]
    if duplicate_columns:
        sample = sample.drop_duplicates(subset=duplicate_columns).reset_index(drop=True)
    return sample


def directory_size_bytes(path: Path, pattern: str) -> int:
    """Compute the size of matching files under one directory."""

    return sum(file_path.stat().st_size for file_path in path.glob(pattern) if file_path.is_file())


def inspect_session1_parquet_cache(
    *,
    raw_data_dir: str | Path,
    runtime_cache_dir: str | Path,
    read_cache_dir: str | Path,
    cache_bundle_name: str,
    target_flag: str,
    measurement_columns: list[str] | tuple[str, ...],
    optional_qc_columns: list[str] | tuple[str, ...],
    plot_measurement_column: str,
    plot_secondary_column: str,
) -> dict[str, object]:
    """Collect the cache paths, column choices, and summary tables for the notebook.

    The returned ``notebook_values`` dictionary contains the variables used by
    later notebook cells, while the table entries are only for display in this
    cache-inspection section.
    """

    active_cache_paths = choose_cache_bundle_paths(
        [Path(runtime_cache_dir), Path(read_cache_dir)],
        cache_stem=cache_bundle_name,
    )
    cache_metadata_path = active_cache_paths.metadata_path
    row_cache_dir = active_cache_paths.row_level_dir
    window_cache_path = active_cache_paths.window_cache_path

    raw_files = sorted(Path(raw_data_dir).glob("*.csv"))
    row_cache_parts = sorted(row_cache_dir.glob("*.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"No raw CSV files found in {raw_data_dir}")
    if not row_cache_parts:
        raise FileNotFoundError(f"No parquet parts found in {row_cache_dir}")
    if not cache_metadata_path.exists():
        raise FileNotFoundError(f"No cache metadata found at {cache_metadata_path}")

    cache_metadata = json.loads(cache_metadata_path.read_text())
    available_row_columns = cache_metadata.get("row_columns", [])
    available_window_columns = cache_metadata.get("window_columns", [])

    # Older cache bundles may not record row/window columns in metadata.
    if not available_row_columns:
        available_row_columns = pq.ParquetFile(row_cache_parts[0]).schema.names
    if not available_window_columns and window_cache_path.exists():
        available_window_columns = pq.ParquetFile(window_cache_path).schema.names

    if target_flag not in available_row_columns:
        raise KeyError(
            f"Target flag {target_flag!r} is not present in the prepared row-level cache. "
            "Choose a different dataset profile or override TARGET_FLAG."
        )

    requested_measurement_columns = list(measurement_columns)
    requested_optional_qc_columns = list(optional_qc_columns)
    active_measurement_columns = [
        column for column in requested_measurement_columns if column in available_row_columns
    ]
    active_optional_qc_columns = [
        column for column in requested_optional_qc_columns if column in available_row_columns
    ]
    missing_measurement_columns = [
        column for column in requested_measurement_columns if column not in active_measurement_columns
    ]
    missing_optional_qc_columns = [
        column for column in requested_optional_qc_columns if column not in active_optional_qc_columns
    ]

    if not active_measurement_columns:
        raise ValueError(
            "None of the requested measurement columns are present in the prepared cache. "
            "Check the dataset profile or use manual overrides."
        )

    if plot_measurement_column in active_measurement_columns:
        active_plot_measurement_column = plot_measurement_column
    else:
        active_plot_measurement_column = active_measurement_columns[0]

    secondary_candidates = [
        column for column in active_measurement_columns if column != active_plot_measurement_column
    ]
    if plot_secondary_column in secondary_candidates:
        active_plot_secondary_column = plot_secondary_column
    elif secondary_candidates:
        active_plot_secondary_column = secondary_candidates[0]
    else:
        active_plot_secondary_column = active_plot_measurement_column

    window_feature_columns = [
        f"{column}_{stat}"
        for column in active_measurement_columns
        for stat in ("mean", "std")
        if f"{column}_{stat}" in available_window_columns
    ]
    row_use_columns = [
        column
        for column in dict.fromkeys(
            ["Time UTC", "source_file", target_flag, *active_optional_qc_columns, *active_measurement_columns]
        )
        if column in available_row_columns
    ]
    window_use_columns = [
        column
        for column in ["window_start", "window_end", "source_file", "issue_rate", *window_feature_columns]
        if column in available_window_columns
    ]

    representative_info = cache_metadata["processed_files"][0]
    representative_raw_path = Path(raw_data_dir) / representative_info["source_file"]
    representative_parquet_path = row_cache_dir / Path(representative_info["row_level_part"]).name
    header_line_number, parsed_columns = locate_header(representative_raw_path)

    raw_total_bytes = directory_size_bytes(Path(raw_data_dir), "*.csv")
    row_cache_bytes = directory_size_bytes(row_cache_dir, "*.parquet")
    window_cache_bytes = window_cache_path.stat().st_size if window_cache_path.exists() else 0
    cache_size_table = pd.DataFrame(
        [
            {"artifact": "Raw CSV files", "bytes": raw_total_bytes},
            {"artifact": "Row-level parquet cache", "bytes": row_cache_bytes},
            {"artifact": "Window-summary parquet", "bytes": window_cache_bytes},
        ]
    )
    cache_size_table["megabytes"] = cache_size_table["bytes"] / (1024 ** 2)
    cache_size_table["gigabytes"] = cache_size_table["bytes"] / (1024 ** 3)

    column_summary_table = pd.DataFrame(
        [
            {
                "column_set": "Parsed columns in representative CSV",
                "count": len(parsed_columns),
                "examples": _join_examples(parsed_columns),
            },
            {
                "column_set": "Row-level parquet columns available",
                "count": len(available_row_columns),
                "examples": _join_examples(available_row_columns),
            },
            {
                "column_set": "Notebook row columns selected",
                "count": len(row_use_columns),
                "examples": _join_examples(row_use_columns),
            },
            {
                "column_set": "Window-summary columns selected",
                "count": len(window_use_columns),
                "examples": _join_examples(window_use_columns),
            },
        ]
    )

    column_selection_summary = {
        "requested_measurement_columns": requested_measurement_columns,
        "active_measurement_columns": active_measurement_columns,
        "missing_measurement_columns": missing_measurement_columns,
        "requested_optional_qc_columns": requested_optional_qc_columns,
        "active_optional_qc_columns": active_optional_qc_columns,
        "missing_optional_qc_columns": missing_optional_qc_columns,
        "plot_measurement_column": active_plot_measurement_column,
        "plot_secondary_column": active_plot_secondary_column,
    }
    cache_summary = {
        "raw_file_count": len(raw_files),
        "parquet_part_count": len(row_cache_parts),
        "metadata_lines_before_table": header_line_number - 1,
        "parsed_column_count": len(parsed_columns),
        "row_columns_available": len(available_row_columns),
        "window_columns_available": len(available_window_columns),
        "full_row_count": cache_metadata["row_count"],
        "full_window_count": cache_metadata.get("window_count", 0),
        "cache_bundle_name": cache_metadata.get("cache_stem", cache_bundle_name),
    }
    notebook_values = {
        "active_cache_paths": active_cache_paths,
        "cache_metadata_path": cache_metadata_path,
        "row_cache_dir": row_cache_dir,
        "window_cache_path": window_cache_path,
        "raw_files": raw_files,
        "row_cache_parts": row_cache_parts,
        "cache_metadata": cache_metadata,
        "available_row_columns": available_row_columns,
        "available_window_columns": available_window_columns,
        "MEASUREMENT_COLUMNS": active_measurement_columns,
        "OPTIONAL_QC_COLUMNS": active_optional_qc_columns,
        "PLOT_MEASUREMENT_COLUMN": active_plot_measurement_column,
        "PLOT_SECONDARY_COLUMN": active_plot_secondary_column,
        "WINDOW_FEATURE_COLUMNS": window_feature_columns,
        "ROW_USE_COLUMNS": row_use_columns,
        "WINDOW_USE_COLUMNS": window_use_columns,
        "representative_info": representative_info,
        "representative_raw_path": representative_raw_path,
        "representative_parquet_path": representative_parquet_path,
        "header_line_number": header_line_number,
        "parsed_columns": parsed_columns,
    }
    return {
        "notebook_values": notebook_values,
        "column_selection_summary": column_selection_summary,
        "cache_summary": cache_summary,
        "cache_size_table": cache_size_table,
        "column_summary_table": column_summary_table,
    }


def show_session1_cache_inspection(**kwargs) -> dict[str, object]:
    """Display a concise cache overview and return notebook variables."""

    cache_context = inspect_session1_parquet_cache(**kwargs)
    _print_json(cache_context["column_selection_summary"])
    display(cache_context["cache_size_table"])
    display(cache_context["column_summary_table"])
    _print_json(cache_context["cache_summary"])
    plot_session1_cache_inspection(
        cache_context["cache_size_table"],
        cache_context["column_summary_table"],
    )
    return cache_context


def plot_session1_cache_inspection(
    cache_size_table: pd.DataFrame,
    column_summary_table: pd.DataFrame,
) -> None:
    """Plot cache size and available/selected column counts."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].bar(
        cache_size_table["artifact"],
        cache_size_table["gigabytes"],
        color=["#5B7C99", "#2A9D8F", "#E9C46A"],
    )
    axes[0].set_ylabel("Size on disk (GB)")
    axes[0].set_title("Raw files and reusable parquet artifacts")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(
        column_summary_table["column_set"],
        column_summary_table["count"],
        color=["#7A6FF0", "#3A86FF", "#2A9D8F", "#00A896"],
    )
    axes[1].set_ylabel("Column count")
    axes[1].set_title("Parquet lets us select only the columns we need")
    axes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.show()


def benchmark_session1_cache_reads(
    *,
    representative_raw_path: str | Path,
    representative_parquet_path: str | Path,
    row_use_columns: list[str],
    sample_rows: int | None,
) -> dict[str, object]:
    """Time CSV, full-parquet, and selected-column parquet reads for one part."""

    benchmark_scope_label = (
        f"the first {sample_rows:,} rows"
        if sample_rows is not None
        else "one full representative file"
    )

    csv_start = perf_counter()
    csv_frame = read_scalar_csv(Path(representative_raw_path), sample_rows=sample_rows)
    csv_elapsed = perf_counter() - csv_start

    parquet_full_start = perf_counter()
    parquet_full_frame = read_parquet_head(
        representative_parquet_path,
        max_rows=sample_rows,
    )
    parquet_full_elapsed = perf_counter() - parquet_full_start

    parquet_selected_start = perf_counter()
    parquet_selected_frame = read_parquet_head(
        representative_parquet_path,
        columns=row_use_columns,
        max_rows=sample_rows,
    )
    parquet_selected_elapsed = perf_counter() - parquet_selected_start

    benchmark_summary = pd.DataFrame(
        [
            {
                "read_path": "Raw CSV parse (all columns)",
                "seconds": csv_elapsed,
                "rows_loaded": len(csv_frame),
                "columns_loaded": len(csv_frame.columns),
                "memory_mb": csv_frame.memory_usage(deep=True).sum() / (1024 ** 2),
            },
            {
                "read_path": "Parquet full row part",
                "seconds": parquet_full_elapsed,
                "rows_loaded": len(parquet_full_frame),
                "columns_loaded": len(parquet_full_frame.columns),
                "memory_mb": parquet_full_frame.memory_usage(deep=True).sum() / (1024 ** 2),
            },
            {
                "read_path": "Parquet selected row columns",
                "seconds": parquet_selected_elapsed,
                "rows_loaded": len(parquet_selected_frame),
                "columns_loaded": len(parquet_selected_frame.columns),
                "memory_mb": parquet_selected_frame.memory_usage(deep=True).sum() / (1024 ** 2),
            },
        ]
    )
    benchmark_summary["rows_per_second"] = benchmark_summary["rows_loaded"] / benchmark_summary["seconds"].clip(lower=1e-9)

    result_summary = {
        "benchmark_scope": benchmark_scope_label,
        "csv_seconds": round(csv_elapsed, 4),
        "parquet_full_seconds": round(parquet_full_elapsed, 4),
        "parquet_selected_seconds": round(parquet_selected_elapsed, 4),
        "full_parquet_vs_csv_ratio": round(parquet_full_elapsed / max(csv_elapsed, 1e-9), 2),
        "selected_parquet_vs_csv_ratio": round(parquet_selected_elapsed / max(csv_elapsed, 1e-9), 2),
        "selected_columns_loaded": list(row_use_columns),
    }
    if parquet_full_elapsed > csv_elapsed:
        result_summary["interpretation"] = (
            "A full parquet read can be slower than CSV for a small, compact sample because parquet has "
            "fixed schema and decompression overhead. The repeated-analysis benefit is typed storage and "
            "selected-column reads."
        )
    else:
        result_summary["interpretation"] = (
            "The full parquet read is faster in this run. The selected-column read is the closest match "
            "to how later notebook sections usually load the cache."
        )
    return {
        "benchmark_summary": benchmark_summary,
        "summary": result_summary,
    }


def show_session1_cache_read_benchmark(**kwargs) -> dict[str, object]:
    """Display the optional CSV-vs-parquet read benchmark."""

    benchmark_result = benchmark_session1_cache_reads(**kwargs)
    benchmark_summary = benchmark_result["benchmark_summary"]
    display(benchmark_summary)
    plot_session1_cache_read_benchmark(
        benchmark_summary,
        benchmark_result["summary"]["benchmark_scope"],
    )
    _print_json(benchmark_result["summary"])
    return benchmark_result


def show_session1_cache_read_comparison(**kwargs) -> dict[str, object]:
    """Display the CSV-vs-parquet read comparison used in the notebook."""

    return show_session1_cache_read_benchmark(**kwargs)


def plot_session1_cache_read_benchmark(
    benchmark_summary: pd.DataFrame,
    benchmark_scope_label: str,
) -> None:
    """Plot elapsed time and pandas memory for the read benchmark."""

    colors = ["#E76F51", "#2A9D8F", "#457B9D"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].bar(benchmark_summary["read_path"], benchmark_summary["seconds"], color=colors)
    axes[0].set_ylabel("Elapsed time (s)")
    axes[0].set_title(f"Read time for {benchmark_scope_label}")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(benchmark_summary["read_path"], benchmark_summary["memory_mb"], color=colors)
    axes[1].set_ylabel("Pandas memory (MB)")
    axes[1].set_title("How much data the read materializes")
    axes[1].tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.show()


def _join_examples(values: list[str], limit: int = 6) -> str:
    """Format a short comma-separated preview of column names."""

    examples = [str(value) for value in values[:limit]]
    return ", ".join(examples) + (" ..." if len(values) > limit else "")


def _print_json(value: object) -> None:
    """Print simple dictionaries/lists in a notebook-friendly JSON block."""

    print(json.dumps(value, indent=2, ensure_ascii=False))


def show_reviewed_model_row_accounting(
    frame: pd.DataFrame,
    *,
    target_flag: str,
    task_mode: str,
    good_labels: list[int] | tuple[int, ...],
    issue_labels: list[int] | tuple[int, ...],
    model_row_limit: int | None,
    data_fraction: float,
    flag_palette: dict[int, str] | None = None,
    label_meanings: dict[int, str] | None = None,
) -> dict[str, object]:
    """Show how selected cache rows become the reviewed modelling dataframe.

    The plot answers two focused questions:

    1. how many rows survive each filtering/budget step, and
    2. whether the issue share changes after we keep only reviewed rows and
       apply the optional ``DATA_FRACTION`` row cap.

    The returned ``reviewed_model_df`` is the same row-level dataframe that the
    split comparison should use, because it already reflects the modelling row
    budget.
    """

    selected_frame = frame.copy()
    selected_frame[target_flag] = pd.to_numeric(selected_frame[target_flag], errors="coerce")
    reviewed_mask = reviewed_label_mask(
        selected_frame[target_flag],
        good_labels=good_labels,
        issue_labels=issue_labels,
    )
    reviewed_label_df = selected_frame.loc[reviewed_mask].copy().reset_index(drop=True)
    effective_model_row_limit = model_row_limit
    if effective_model_row_limit is None:
        effective_model_row_limit = derive_fractional_row_limit(len(reviewed_label_df), data_fraction)
    reviewed_model_df, active_labels = build_reviewed_target_frame(
        selected_frame,
        target_flag=target_flag,
        task_mode=task_mode,
        good_labels=good_labels,
        issue_labels=issue_labels,
        model_row_limit=effective_model_row_limit,
    )

    normalized_task_mode = str(task_mode).lower().strip()
    model_good_labels = (
        [0]
        if normalized_task_mode == "binary"
        else [int(label) for label in active_labels if int(label) in set(int(item) for item in good_labels)]
    )
    model_issue_labels = (
        [1]
        if normalized_task_mode == "binary"
        else [int(label) for label in active_labels if int(label) in set(int(item) for item in issue_labels)]
    )

    selected_counts = selected_frame[target_flag].dropna().astype(int).value_counts().sort_index()
    reviewed_counts = reviewed_label_df[target_flag].dropna().astype(int).value_counts().sort_index()
    model_counts = reviewed_model_df[target_flag].dropna().astype(int).value_counts().sort_index()

    labels = sorted(set(selected_counts.index).union(reviewed_counts.index).union(model_counts.index))
    distribution_counts = (
        pd.DataFrame(
            {
                "selected_cache_rows": selected_counts.reindex(labels, fill_value=0),
                "usable_reviewed_rows": reviewed_counts.reindex(labels, fill_value=0),
                "reviewed_model_rows": model_counts.reindex(labels, fill_value=0),
            }
        )
        .fillna(0)
        .astype(int)
    )
    distribution_shares = distribution_counts.div(distribution_counts.sum(axis=0), axis=1).fillna(0.0)

    issue_label_set = sorted(set(int(label) for label in issue_labels))

    def _issue_rows(counts: pd.Series) -> int:
        return int(counts.reindex(issue_label_set, fill_value=0).sum())

    selected_issue_rows = _issue_rows(selected_counts)
    reviewed_issue_rows = _issue_rows(reviewed_counts)
    model_issue_rows = _issue_rows(model_counts)

    row_summary = pd.DataFrame(
        [
            {
                "step": "selected cache rows",
                "rows": len(selected_frame),
                "share_of_selected_pct": 100.0,
                "issue_rows": selected_issue_rows,
                "issue_share_pct": 100 * selected_issue_rows / max(len(selected_frame), 1),
                "what_changed": "All selected parquet rows with the target column loaded.",
            },
            {
                "step": "usable reviewed rows",
                "rows": len(reviewed_label_df),
                "share_of_selected_pct": 100 * len(reviewed_label_df) / max(len(selected_frame), 1),
                "issue_rows": reviewed_issue_rows,
                "issue_share_pct": 100 * reviewed_issue_rows / max(len(reviewed_label_df), 1),
                "what_changed": "Keep only labels listed in GOOD_LABELS or ISSUE_LABELS.",
            },
            {
                "step": "reviewed modelling rows",
                "rows": len(reviewed_model_df),
                "share_of_selected_pct": 100 * len(reviewed_model_df) / max(len(selected_frame), 1),
                "issue_rows": model_issue_rows,
                "issue_share_pct": 100 * model_issue_rows / max(len(reviewed_model_df), 1),
                "what_changed": "Apply the optional DATA_FRACTION row budget.",
            },
        ]
    )
    row_summary["share_of_selected_pct"] = row_summary["share_of_selected_pct"].round(2)
    row_summary["issue_share_pct"] = row_summary["issue_share_pct"].round(3)

    default_label_meanings = {
        0: "no QC / custom good",
        1: "good",
        2: "probably good",
        3: "probably bad",
        4: "bad",
        6: "bad down-sampling",
        7: "averaged",
        8: "interpolated",
        9: "missing / NaN",
    }
    if label_meanings:
        default_label_meanings.update({int(label): str(meaning) for label, meaning in label_meanings.items()})
    flag_meanings = pd.DataFrame(
        {
            "label": labels,
            "meaning": [default_label_meanings.get(int(label), "dataset-specific label") for label in labels],
        }
    )
    display(flag_meanings[flag_meanings["label"].isin(labels)].reset_index(drop=True))
    display(row_summary)

    reviewed_label_summary = pd.DataFrame(index=sorted(set(int(label) for label in good_labels).union(issue_labels)))
    reviewed_label_summary["usable_reviewed_count"] = reviewed_counts.reindex(reviewed_label_summary.index, fill_value=0)
    reviewed_label_summary["reviewed_model_count"] = model_counts.reindex(reviewed_label_summary.index, fill_value=0)
    reviewed_label_summary["reviewed_model_share_pct"] = (
        100 * reviewed_label_summary["reviewed_model_count"] / max(int(reviewed_label_summary["reviewed_model_count"].sum()), 1)
    ).round(2)
    reviewed_label_summary.index.name = target_flag
    display(reviewed_label_summary)

    fig, (ax_rows, ax_issue) = plt.subplots(
        1,
        2,
        figsize=(13.8, 4.8),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )

    row_colors = ["#94a3b8", "#2563eb", "#0f766e"]
    bars = ax_rows.barh(row_summary["step"], row_summary["share_of_selected_pct"], color=row_colors, height=0.58)
    ax_rows.set_xlim(0, 100)
    ax_rows.xaxis.set_major_formatter(PercentFormatter(100))
    ax_rows.set_xlabel("Share of selected cache rows")
    ax_rows.set_title("How many rows move forward?")
    ax_rows.grid(axis="x", alpha=0.18)
    ax_rows.invert_yaxis()
    for bar, row in zip(bars, row_summary.itertuples(index=False)):
        pct = float(row.share_of_selected_pct)
        if pct >= 68:
            x_position = pct - 1.0
            horizontal_alignment = "right"
            text_color = "white"
        else:
            x_position = pct + 1.0
            horizontal_alignment = "left"
            text_color = "#111827"
        ax_rows.text(
            x_position,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row.rows):,} rows ({pct:.1f}%)",
            va="center",
            ha=horizontal_alignment,
            fontsize=9,
            color=text_color,
            fontweight="bold" if row.step == "reviewed modelling rows" else "normal",
        )

    y_positions = np.arange(len(row_summary))
    ax_issue.plot(
        row_summary["issue_share_pct"],
        y_positions,
        color="#0f766e",
        linewidth=2.0,
        marker="o",
        markersize=7,
    )
    ax_issue.set_yticks(y_positions)
    ax_issue.set_yticklabels(row_summary["step"])
    ax_issue.invert_yaxis()
    max_issue_share = float(row_summary["issue_share_pct"].max()) if len(row_summary) else 0.0
    ax_issue.set_xlim(0, max(1.0, max_issue_share * 1.45))
    ax_issue.xaxis.set_major_formatter(PercentFormatter(100))
    ax_issue.set_xlabel("Issue share")
    ax_issue.set_title("Did the issue share change?")
    ax_issue.grid(axis="x", alpha=0.18)
    for y_position, row in zip(y_positions, row_summary.itertuples(index=False)):
        ax_issue.text(
            float(row.issue_share_pct) + max(0.05, max_issue_share * 0.03),
            y_position,
            f"{row.issue_share_pct:.2f}% ({int(row.issue_rows):,} issue rows)",
            va="center",
            ha="left",
            fontsize=9,
            clip_on=False,
        )

    fig.suptitle(f"Rows used for modelling when DATA_FRACTION = {data_fraction:g}", y=1.04)
    plt.tight_layout()
    plt.show()

    summary = {
        "DATA_FRACTION": data_fraction,
        "reviewed_model_row_limit": effective_model_row_limit,
        "selected_rows": int(len(selected_frame)),
        "usable_reviewed_rows": int(len(reviewed_label_df)),
        "reviewed_model_rows": int(len(reviewed_model_df)),
        "active_model_labels": active_labels,
        "model_good_labels": model_good_labels,
        "model_issue_labels": model_issue_labels,
    }
    _print_json(summary)

    return {
        "reviewed_label_df": reviewed_label_df,
        "reviewed_model_df": reviewed_model_df,
        "active_labels": active_labels,
        "model_good_labels": model_good_labels,
        "model_issue_labels": model_issue_labels,
        "selected_target_counts": selected_counts,
        "selected_reviewed_counts": reviewed_counts,
        "reviewed_model_counts": model_counts,
        "row_summary": row_summary,
        "distribution_counts": distribution_counts,
        "distribution_shares": distribution_shares,
        "summary": summary,
    }


def show_temporal_flag_summary(
    frame: pd.DataFrame,
    *,
    target_flag: str,
    selected_path_count: int,
    temporal_summary_bin_count: int,
    good_labels: list[int] | tuple[int, ...],
    issue_labels: list[int] | tuple[int, ...],
    target_display_name: str = "target label",
    flag_palette: dict[int, str] | None = None,
    time_column: str = "Time UTC",
) -> dict[str, object]:
    """Display a time-bin summary table plus reviewed and issue-only label plots."""

    available_labels = set(
        pd.to_numeric(frame[target_flag], errors="coerce").dropna().astype(int).tolist()
    )
    temporal_labels = [
        int(label)
        for label in sorted(set(int(label) for label in good_labels).union(int(label) for label in issue_labels))
        if int(label) in available_labels
    ]
    temporal_bin_count = min(int(temporal_summary_bin_count), max(8, int(selected_path_count) * 2))
    temporal_counts, temporal_shares, temporal_summary = summarize_target_by_time_bin(
        frame,
        time_column=time_column,
        label_column=target_flag,
        bin_count=temporal_bin_count,
        labels=temporal_labels,
        good_labels=good_labels,
        issue_labels=issue_labels,
    )

    temporal_summary = temporal_summary.copy()
    if not temporal_summary.empty:
        dominant_labels = []
        for time_bin in temporal_counts.index:
            bin_counts = temporal_counts.loc[time_bin]
            dominant_labels.append(int(bin_counts.idxmax()) if int(bin_counts.sum()) > 0 else None)
        temporal_summary["dominant_label"] = dominant_labels
        temporal_summary["time_window"] = (
            temporal_summary["time_start"].dt.strftime("%Y-%m-%d %H:%M")
            + " -> "
            + temporal_summary["time_end"].dt.strftime("%Y-%m-%d %H:%M")
        )
    else:
        temporal_summary["dominant_label"] = pd.Series(dtype="Int64")
        temporal_summary["time_window"] = pd.Series(dtype="string")

    summary_columns = [
        "time_window",
        "rows",
        "reviewed_rows",
        "unreviewed_rows",
        "reviewed_share_pct",
        "issue_rows",
        "issue_share_pct",
        "dominant_label",
    ]
    summary_table = temporal_summary.reindex(columns=summary_columns).reset_index(drop=True)

    issue_label_set = {int(label) for label in issue_labels}
    issue_columns = [column for column in temporal_counts.columns if int(column) in issue_label_set]
    issue_only_shares = temporal_counts.reindex(columns=issue_columns, fill_value=0)
    issue_only_shares = issue_only_shares.div(
        issue_only_shares.sum(axis=1).replace(0, np.nan),
        axis=0,
    ).fillna(0.0)

    if not summary_table.empty:
        display(summary_table)

    if not temporal_summary.empty:
        palette = dict(DEFAULT_FLAG_PALETTE)
        if flag_palette:
            palette.update(flag_palette)

        temporal_plot_labels = temporal_summary["time_start"].dt.strftime("%m-%d\n%H:%M").tolist()
        x_positions = np.arange(len(temporal_summary))

        fig, (ax_top, ax_middle, ax_bottom) = plt.subplots(
            3,
            1,
            figsize=(13.5, 10.6),
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.35, 1.35]},
        )

        ax_top.bar(
            x_positions,
            temporal_summary["reviewed_rows"],
            color="#cbd5e1",
            edgecolor="#94a3b8",
            width=0.82,
            label="reviewed rows in time bin",
        )
        ax_top.set_ylabel("Reviewed rows")
        ax_top.grid(axis="y", alpha=0.2)
        ax_top.set_title(
            f"{target_flag}: where reviewed {target_display_name} values live through time"
        )

        ax_top_issue = ax_top.twinx()
        ax_top_issue.plot(
            x_positions,
            temporal_summary["issue_share_pct"],
            color="#dc2626",
            marker="o",
            linewidth=2.0,
            label="issue share among reviewed rows (%)",
        )
        ax_top_issue.set_ylabel("Issue share (%)", color="#dc2626")
        ax_top_issue.tick_params(axis="y", colors="#dc2626")

        top_handles, top_labels = ax_top.get_legend_handles_labels()
        top_issue_handles, top_issue_labels = ax_top_issue.get_legend_handles_labels()
        ax_top.legend(
            top_handles + top_issue_handles,
            top_labels + top_issue_labels,
            bbox_to_anchor=(1.01, 1.0),
            loc="upper left",
        )

        reviewed_share_plot = temporal_shares.copy()
        reviewed_share_plot.index = temporal_plot_labels
        reviewed_palette = [palette.get(int(label), "#64748b") for label in reviewed_share_plot.columns]
        reviewed_share_plot.plot(
            kind="bar",
            stacked=True,
            ax=ax_middle,
            color=reviewed_palette,
            width=0.85,
        )
        ax_middle.set_ylim(0, 1)
        ax_middle.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax_middle.set_ylabel("Share of reviewed rows")
        ax_middle.set_title(f"Reviewed {target_display_name} composition across the time axis")
        ax_middle.grid(axis="y", alpha=0.2)
        ax_middle.legend(title=target_display_name, bbox_to_anchor=(1.01, 1.0), loc="upper left")

        issue_share_plot = issue_only_shares.copy()
        issue_share_plot.index = temporal_plot_labels
        if issue_share_plot.empty or float(issue_share_plot.to_numpy().sum()) == 0.0:
            ax_bottom.set_ylim(0, 1)
            ax_bottom.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax_bottom.set_ylabel("Share of issue rows")
            ax_bottom.set_xlabel("Time bin start (UTC)")
            ax_bottom.set_title(f"Issue {target_display_name} composition across the time axis")
            ax_bottom.grid(axis="y", alpha=0.2)
            ax_bottom.set_xticks(x_positions, temporal_plot_labels)
            ax_bottom.text(
                0.5,
                0.5,
                "No issue rows were found in these time bins.",
                transform=ax_bottom.transAxes,
                ha="center",
                va="center",
            )
        else:
            issue_palette = [palette.get(int(label), "#64748b") for label in issue_share_plot.columns]
            issue_share_plot.plot(
                kind="bar",
                stacked=True,
                ax=ax_bottom,
                color=issue_palette,
                width=0.85,
            )
            ax_bottom.set_ylim(0, 1)
            ax_bottom.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax_bottom.set_ylabel("Share of issue rows")
            ax_bottom.set_xlabel("Time bin start (UTC)")
            ax_bottom.set_title(f"Issue {target_display_name} composition across the time axis")
            ax_bottom.grid(axis="y", alpha=0.2)
            ax_bottom.legend(title=f"Issue {target_display_name}", bbox_to_anchor=(1.01, 1.0), loc="upper left")

        plt.tight_layout()
        plt.show()

    return {
        "bin_count": temporal_bin_count,
        "labels": temporal_labels,
        "counts": temporal_counts,
        "shares": temporal_shares,
        "issue_only_shares": issue_only_shares,
        "summary": temporal_summary,
        "summary_table": summary_table,
    }


def show_reviewed_split_summary(
    split_frames: dict[str, pd.DataFrame],
    *,
    issue_labels: list[int] | tuple[int, ...],
    label_column: str = "model_target",
    time_column: str = "Time UTC",
    split_display_names: dict[str, str] | None = None,
    flag_palette: dict[int, str] | None = None,
    plot_title: str | None = None,
    legend_title: str = "target label",
) -> dict[str, object]:
    """Display a split summary table plus reviewed and issue-only composition plots."""

    split_counts, split_shares = summarize_split_distributions(split_frames, label_column=label_column)
    split_display_names = split_display_names or {}
    palette = dict(DEFAULT_FLAG_PALETTE)
    if flag_palette:
        palette.update(flag_palette)
    issue_label_set = {int(label) for label in issue_labels}
    issue_only_counts = split_counts.reindex(
        [label for label in split_counts.index if int(label) in issue_label_set],
        fill_value=0,
    )
    issue_only_shares = issue_only_counts.div(
        issue_only_counts.sum(axis=0).replace(0, np.nan),
        axis=1,
    ).fillna(0.0)

    overview_rows = []
    for split_name, frame in split_frames.items():
        label_counts = (
            split_counts[split_name]
            if split_name in split_counts.columns
            else pd.Series(dtype=int)
        )
        issue_rows = int(label_counts.reindex([int(label) for label in issue_labels], fill_value=0).sum())
        time_start = frame[time_column].min() if len(frame) else None
        time_end = frame[time_column].max() if len(frame) else None
        overview_rows.append(
            {
                "split": split_display_names.get(split_name, split_name),
                "rows": len(frame),
                "issue_rows": issue_rows,
                "issue_share_pct": round(100 * issue_rows / max(len(frame), 1), 2),
                "time_start": time_start.isoformat() if pd.notna(time_start) else None,
                "time_end": time_end.isoformat() if pd.notna(time_end) else None,
            }
        )
    overview_df = pd.DataFrame(overview_rows).set_index("split")

    if not overview_df.empty:
        display(
            overview_df.style.format(
                {
                    "rows": "{:,.0f}",
                    "issue_rows": "{:,.0f}",
                    "issue_share_pct": "{:.2f}",
                }
            )
        )

    if not split_shares.empty:
        split_order = [split_name for split_name in ["train", "validation", "test"] if split_name in split_frames]
        fig, (ax_left, ax_right) = plt.subplots(
            1,
            2,
            figsize=(15.2, 4.9),
            sharey=True,
        )

        _plot_split_share_panel(
            ax_left,
            split_shares,
            split_counts,
            split_order=split_order,
            split_display_names=split_display_names,
            palette=palette,
            xlabel="Share of reviewed rows in the split",
            title="All reviewed labels",
            empty_message="No reviewed rows were found in these splits.",
        )
        _plot_split_share_panel(
            ax_right,
            issue_only_shares,
            issue_only_counts,
            split_order=split_order,
            split_display_names=split_display_names,
            palette=palette,
            xlabel="Share of issue rows in the split",
            title="Issue labels only",
            empty_message="No issue rows were found in these splits.",
            count_prefix="issue n",
        )

        all_handles = [Patch(facecolor=palette.get(int(label), "#64748b"), label=str(label)) for label in split_shares.index]
        issue_handles = [
            Patch(facecolor=palette.get(int(label), "#64748b"), label=str(label))
            for label in issue_only_shares.index
        ]
        if all_handles:
            ax_left.legend(handles=all_handles, title=legend_title, bbox_to_anchor=(1.01, 1.0), loc="upper left")
        if issue_handles:
            ax_right.legend(
                handles=issue_handles,
                title=f"issue {legend_title}",
                bbox_to_anchor=(1.01, 1.0),
                loc="upper left",
            )

        fig.suptitle(plot_title or "Target composition across train / validation / test", y=1.02)
        plt.tight_layout()
        plt.show()

    return {
        "overview": overview_df,
        "counts": split_counts,
        "shares": split_shares,
        "issue_only_counts": issue_only_counts,
        "issue_only_shares": issue_only_shares,
    }


def build_reviewed_modelling_split(
    *,
    selected_paths: list[Path],
    target_flag: str,
    task_mode: str,
    good_labels: list[int] | tuple[int, ...],
    issue_labels: list[int] | tuple[int, ...],
    model_row_limit: int | None,
    train_fraction: float,
    validation_fraction: float,
    split_strategy: str,
    split_block_rows: int | None = None,
    measurement_columns: list[str] | tuple[str, ...] | None = None,
    episode_context_rows: int = 0,
    episode_merge_gap_rows: int = 0,
    purge_gap_rows: int = 0,
) -> dict[str, object]:
    """Build the reviewed modelling table and fixed train/validation/test split.

    Parameters are deliberately close to the controls shown in the notebooks:
    ``selected_paths`` chooses which row-level parquet parts are loaded,
    ``task_mode`` decides whether labels stay multi-class or collapse to binary
    issue/not-issue, and ``model_row_limit`` is the optional row budget derived
    from ``DATA_FRACTION``. The split arguments define the fixed train,
    validation, and test frames that all later model sections reuse.

    The function centralises the repetitive operational work: load reviewed
    rows, create ``issue`` and ``model_target``, apply the requested split, and
    return the standard notebook dataframe names.
    """

    normalized_task_mode = task_mode.lower().strip()
    if normalized_task_mode not in {"multiclass", "binary"}:
        raise ValueError(f"Unsupported task_mode: {task_mode}")

    label_columns = ["Time UTC", "source_file", target_flag]
    label_source_df = load_full_row_level_frame(selected_paths, columns=label_columns)
    source_rows = len(label_source_df)
    reviewed_source_rows = int(
        reviewed_label_mask(
            label_source_df[target_flag],
            good_labels=good_labels,
            issue_labels=issue_labels,
        ).sum()
    )

    reviewed_model_df, active_labels = build_reviewed_target_frame(
        label_source_df,
        target_flag=target_flag,
        task_mode=normalized_task_mode,
        good_labels=good_labels,
        issue_labels=issue_labels,
        model_row_limit=model_row_limit,
    )
    if reviewed_model_df.empty:
        raise ValueError("The selected cache parts did not contain any reviewed modelling rows.")

    model_good_labels = (
        [0]
        if normalized_task_mode == "binary"
        else [int(label) for label in active_labels if int(label) in set(int(item) for item in good_labels)]
    )
    model_issue_labels = (
        [1]
        if normalized_task_mode == "binary"
        else [int(label) for label in active_labels if int(label) in set(int(item) for item in issue_labels)]
    )

    effective_block_rows = int(split_block_rows or 1024)
    fixed_split_frames = split_frame_by_strategy(
        reviewed_model_df,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        strategy=split_strategy,
        block_rows=effective_block_rows,
        issue_column="issue",
        target_column=target_flag,
        issue_labels=issue_labels,
        episode_context_rows=episode_context_rows,
        episode_merge_gap_rows=episode_merge_gap_rows,
        purge_gap_rows=purge_gap_rows,
    )

    train_full_df = fixed_split_frames["train"]
    valid_df = fixed_split_frames["validation"]
    test_df = fixed_split_frames["test"]
    target_name = target_flag if normalized_task_mode == "multiclass" else "issue"

    summary = {
        "source_rows": source_rows,
        "reviewed_rows_before_limit": reviewed_source_rows,
        "reviewed_modelling_rows": len(reviewed_model_df),
        "dropped_unreviewed_rows": int(source_rows - reviewed_source_rows),
        "dropped_for_row_limit": int(max(reviewed_source_rows - len(reviewed_model_df), 0)),
        "reviewed_row_limit": model_row_limit,
        "task_mode": normalized_task_mode,
        "target_name": target_name,
        "active_good_labels": model_good_labels,
        "active_issue_labels": model_issue_labels,
        "split_strategy": split_strategy,
        "split_block_rows": effective_block_rows if split_strategy in {"interleaved_blocks", "episode_aware"} else None,
        "episode_context_rows": episode_context_rows if split_strategy == "episode_aware" else 0,
        "episode_merge_gap_rows": episode_merge_gap_rows if split_strategy == "episode_aware" else 0,
        "episode_purge_gap_rows": purge_gap_rows if split_strategy == "episode_aware" else 0,
        "full_train_rows": len(train_full_df),
        "validation_rows": len(valid_df),
        "test_rows": len(test_df),
        "train_time_span": _time_span(train_full_df),
        "validation_time_span": _time_span(valid_df),
        "test_time_span": _time_span(test_df),
    }

    notebook_values = {
        "measurement_columns": list(measurement_columns or []),
        "task_mode": normalized_task_mode,
        "target_name": target_name,
        "reviewed_model_label_columns": label_columns,
        "reviewed_model_df": reviewed_model_df,
        "model_df": reviewed_model_df,
        "active_labels": active_labels,
        "model_good_labels": model_good_labels,
        "model_issue_labels": model_issue_labels,
        "fixed_split_frames": fixed_split_frames,
        "train_full_df": train_full_df,
        "valid_df": valid_df,
        "test_df": test_df,
    }
    preview_columns = ["Time UTC", "source_file", target_flag, "issue", "model_target"]
    return {
        "notebook_values": notebook_values,
        "summary": summary,
        "preview": reviewed_model_df[preview_columns].head(8),
    }


def show_reviewed_modelling_split_build(split_bundle: dict[str, object]) -> None:
    """Display the small preview and JSON summary from ``build_reviewed_modelling_split``."""

    display(split_bundle["preview"])
    _print_json(split_bundle["summary"])


def show_fixed_split_review(
    split_frames: dict[str, pd.DataFrame],
    *,
    issue_labels: list[int] | tuple[int, ...],
    split_strategy_label: str,
    flag_palette: dict[int, str] | None = None,
    label_column: str = "model_target",
    legend_title: str = "model target",
    min_issue_rows: int | None = None,
    min_issue_share_pct: float | None = None,
    min_rows_per_issue_label: int | None = None,
) -> dict[str, object]:
    """Display the fixed split summary and optional validation adequacy table.

    The returned dictionary keeps the displayed tables available to later
    notebook cells as ``split_overview``, ``split_counts``, ``split_shares``,
    and ``split_issue_only_shares``. The adequacy arguments are only used by the
    advanced notebook, where we still want a compact diagnostic for validation
    issue coverage.
    """

    review = show_reviewed_split_summary(
        split_frames,
        issue_labels=issue_labels,
        label_column=label_column,
        flag_palette=flag_palette,
        legend_title=legend_title,
        plot_title=f"Target composition for {split_strategy_label} train / validation / test splits",
    )

    if (
        min_issue_rows is not None
        and min_issue_share_pct is not None
        and min_rows_per_issue_label is not None
    ):
        split_adequacy = summarize_issue_adequacy(
            split_frames,
            label_column=label_column,
            issue_labels=issue_labels,
            min_issue_rows=min_issue_rows,
            min_issue_share_pct=min_issue_share_pct,
            min_rows_per_issue_label=min_rows_per_issue_label,
        )
        validation_adequacy = (
            split_adequacy.loc[["validation"]]
            .rename(
                columns={
                    "meets_issue_row_floor": "enough_issue_rows",
                    "meets_issue_share_floor": "enough_issue_share",
                    "meets_per_issue_label_floor": "enough_each_issue_label",
                }
            )
        )
        display(validation_adequacy)
        review["adequacy"] = split_adequacy
        review["validation_adequacy"] = validation_adequacy

    return review


def show_cnn_interval_demo(
    *,
    cnn_run: bool,
    cnn_config: dict[str, object] | None = None,
    model: object | None = None,
    device: object | None = None,
    metadata: dict[str, object] | None = None,
    row_cache_dir: str | Path | None = None,
    row_use_columns: list[str] | tuple[str, ...] | None = None,
    train_df: pd.DataFrame | None = None,
    valid_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    target_flag: str | None = None,
    measurement_columns: list[str] | tuple[str, ...] | None = None,
    task_mode: str | None = None,
    class_labels: list[int] | tuple[int, ...] | np.ndarray | None = None,
    channel_mean: object | None = None,
    channel_std: object | None = None,
    range_start: str | pd.Timestamp | None = None,
    range_end: str | pd.Timestamp | None = None,
    auto_select_test_range: bool = True,
    max_points_to_plot: int | None = 1024,
    plot_measurement_column: str | None = None,
    plot_secondary_column: str | None = None,
    flag_display_name: str = "target label",
    label_meanings: dict[int, str] | None = None,
    flag_palette: dict[int, str] | None = None,
) -> dict[str, object]:
    """Show CNN predictions on one selected time interval.

    The helper contains the plotting and evaluation plumbing that is useful in
    the workshop but distracting inside the notebook. It supports both CNN
    output modes:

    - ``output_mode="window"`` predicts one label for each complete window.
    - ``output_mode="per_timestep"`` predicts one label for every timestamp
      inside each complete window.

    The selected interval is usually taken from ``test_df`` so the visual check
    stays held out from training.
    """

    if not cnn_run:
        print("CNN date-range demo skipped because the CNN was not trained in this run.")
        return {"ran": False, "interval_metrics": None, "figure": None}

    required_values = {
        "cnn_config": cnn_config,
        "model": model,
        "device": device,
        "metadata": metadata,
        "row_cache_dir": row_cache_dir,
        "row_use_columns": row_use_columns,
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "target_flag": target_flag,
        "measurement_columns": measurement_columns,
        "task_mode": task_mode,
        "class_labels": class_labels,
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "plot_measurement_column": plot_measurement_column,
        "plot_secondary_column": plot_secondary_column,
    }
    missing_names = [name for name, value in required_values.items() if value is None]
    if missing_names:
        raise ValueError(
            "show_cnn_interval_demo needs these values when cnn_run=True: "
            + ", ".join(missing_names)
        )

    output_mode = str(cnn_config.get("output_mode", "window"))
    if output_mode not in {"window", "per_timestep"}:
        raise ValueError('CNN output_mode must be either "window" or "per_timestep".')

    # Pick the time span to visualise. With auto-selection enabled, this looks
    # inside the test split for a compact range that includes labelled examples.
    cnn_range_selection = select_time_range(
        test_df,
        time_column="Time UTC",
        label_column=target_flag,
        start=range_start,
        end=range_end,
        auto_select=auto_select_test_range,
        max_points=max_points_to_plot,
    )

    # Load only the raw rows needed for this one interval, then rebuild the
    # same model-ready columns that the training cells used.
    cnn_interval_rows = load_rows_for_time_range(
        metadata,
        row_cache_dir=Path(row_cache_dir),
        start=cnn_range_selection["start"],
        end=cnn_range_selection["end"],
        columns=list(row_use_columns),
    )
    if cnn_interval_rows.empty:
        print("No row-level data was found in the requested CNN range.")
        return {
            "ran": False,
            "range_selection": cnn_range_selection,
            "interval_metrics": None,
            "figure": None,
        }

    cnn_interval_model_df, _, _ = build_model_frame(
        cnn_interval_rows,
        target_flag=target_flag,
        measurement_columns=list(measurement_columns),
        task_mode=task_mode,
        model_row_limit=None,
    )
    cnn_interval_model_df = cnn_interval_model_df[
        (cnn_interval_model_df["Time UTC"] >= cnn_range_selection["start"])
        & (cnn_interval_model_df["Time UTC"] <= cnn_range_selection["end"])
    ].reset_index(drop=True)

    # Record whether the interval came from train/validation/test so readers can
    # confirm that the demo is usually using held-out test data.
    cnn_interval_origin = infer_interval_origin(
        cnn_range_selection["start"],
        cnn_range_selection["end"],
        {"train": train_df, "validation": valid_df, "test": test_df},
    )
    cnn_plot_palette = (
        flag_palette
        if str(task_mode).lower().strip() == "multiclass"
        else {0: "#1f77b4", 1: "#d62728"}
    )

    cnn_interval_metrics = None
    cnn_demo_figure = None
    class_label_list = [int(label) for label in class_labels]

    if output_mode == "window":
        # Window mode compresses each complete window down to one true label and
        # one predicted label. ``label_reduction`` controls how row labels become
        # that one window label during data preparation.
        cnn_interval_bundle = build_window_classification_interval_data(
            cnn_interval_model_df,
            feature_columns=list(measurement_columns),
            target_column="model_target",
            task_mode=task_mode,
            window_size=int(cnn_config["window_size"]),
            label_reduction=str(cnn_config["label_reduction"]),
        )

        if cnn_interval_bundle["window_frame"].empty:
            print(
                "The selected CNN range is shorter than one full window after preprocessing, "
                "so the window-level demo is skipped."
            )
        else:
            # Apply the same channel normalisation learned from the training
            # split before asking the CNN for predictions.
            cnn_predicted_labels = predict_cnn_window_model(
                model,
                cnn_interval_bundle["raw_sequences"],
                task_mode=task_mode,
                class_labels=class_label_list,
                device=str(device),
                channel_mean=channel_mean,
                channel_std=channel_std,
                batch_size=int(cnn_config["batch_size"]),
            )
            cnn_window_frame = cnn_interval_bundle["window_frame"].copy()
            cnn_window_frame["predicted_label"] = cnn_predicted_labels
            cnn_interval_metrics = compute_interval_classification_metrics(
                cnn_window_frame["true_label"],
                cnn_window_frame["predicted_label"],
                labels=class_label_list,
                average=report_average(task_mode),
                target_names=[str(label) for label in class_label_list],
            )
            cnn_true_intervals = merge_adjacent_intervals(
                cnn_window_frame.rename(
                    columns={"window_start": "start", "window_end": "end", "true_label": "label"}
                )[["start", "end", "label"]]
            )
            cnn_pred_intervals = merge_adjacent_intervals(
                cnn_window_frame.rename(
                    columns={"window_start": "start", "window_end": "end", "predicted_label": "label"}
                )[["start", "end", "label"]]
            )

            print(
                {
                    "output_mode": output_mode,
                    "selection_mode": cnn_range_selection["selection_mode"],
                    "selected_priority_flag": cnn_range_selection["selected_label"],
                    "interval_origin": cnn_interval_origin,
                    "range_start": cnn_range_selection["start"].isoformat(),
                    "range_end": cnn_range_selection["end"].isoformat(),
                    "window_count_in_interval": int(len(cnn_window_frame)),
                    "interval_f1": cnn_interval_metrics["f1"],
                }
            )
            print(cnn_interval_metrics["report_text"])
            display(
                pd.DataFrame(
                    {
                        "true_count": cnn_window_frame["true_label"].value_counts().sort_index(),
                        "predicted_count": cnn_window_frame["predicted_label"].value_counts().sort_index(),
                    }
                ).fillna(0).astype(int)
            )

            cnn_demo_figure = plot_time_series_with_bands(
                cnn_interval_model_df,
                band_specs=[
                    {
                        "title": f"True window {flag_display_name}",
                        "intervals": cnn_true_intervals,
                        "palette": cnn_plot_palette,
                    },
                    {
                        "title": f"CNN window {flag_display_name} prediction",
                        "intervals": cnn_pred_intervals,
                        "palette": cnn_plot_palette,
                    },
                ],
                measurement_column=plot_measurement_column,
                secondary_column=plot_secondary_column,
                max_points=max_points_to_plot,
                title="Baseline CNN window predictions on a selected time range",
                label_meanings=label_meanings,
                legend_title=flag_display_name,
            )
            plt.show()
    else:
        # Per-timestep mode keeps the time axis: every timestamp inside each
        # complete window receives its own true label and predicted label.
        cnn_interval_bundle = build_sequence_label_interval_data(
            cnn_interval_model_df,
            feature_columns=list(measurement_columns),
            target_column="model_target",
            window_size=int(cnn_config["window_size"]),
        )

        if len(cnn_interval_bundle["raw_sequences"]) == 0:
            print(
                "The selected CNN range is shorter than one full window after preprocessing, "
                "so the per-timestep demo is skipped."
            )
        else:
            cnn_predicted_labels = predict_sequence_label_cnn(
                model,
                cnn_interval_bundle["raw_sequences"],
                task_mode=task_mode,
                class_labels=class_label_list,
                device=str(device),
                channel_mean=channel_mean,
                channel_std=channel_std,
                batch_size=int(cnn_config["batch_size"]),
            )
            cnn_point_frame = pd.DataFrame(
                {
                    "Time UTC": pd.to_datetime(cnn_interval_bundle["raw_times"].reshape(-1), utc=True),
                    "true_label": cnn_interval_bundle["raw_targets"].reshape(-1).astype(int),
                    "predicted_label": cnn_predicted_labels.reshape(-1).astype(int),
                }
            )
            cnn_interval_metrics = compute_interval_classification_metrics(
                cnn_point_frame["true_label"],
                cnn_point_frame["predicted_label"],
                labels=class_label_list,
                average=report_average(task_mode),
                target_names=[str(label) for label in class_label_list],
            )
            cnn_true_intervals = merge_adjacent_intervals(
                build_labeled_intervals(cnn_point_frame, time_column="Time UTC", label_column="true_label")
            )
            cnn_pred_intervals = merge_adjacent_intervals(
                build_labeled_intervals(cnn_point_frame, time_column="Time UTC", label_column="predicted_label")
            )

            print(
                {
                    "output_mode": output_mode,
                    "selection_mode": cnn_range_selection["selection_mode"],
                    "selected_priority_flag": cnn_range_selection["selected_label"],
                    "interval_origin": cnn_interval_origin,
                    "range_start": cnn_range_selection["start"].isoformat(),
                    "range_end": cnn_range_selection["end"].isoformat(),
                    "point_count_in_interval": int(len(cnn_point_frame)),
                    "interval_f1": cnn_interval_metrics["f1"],
                }
            )
            print(cnn_interval_metrics["report_text"])
            display(
                pd.DataFrame(
                    {
                        "true_count": cnn_point_frame["true_label"].value_counts().sort_index(),
                        "predicted_count": cnn_point_frame["predicted_label"].value_counts().sort_index(),
                    }
                ).fillna(0).astype(int)
            )

            cnn_demo_figure = plot_time_series_with_bands(
                cnn_interval_model_df,
                band_specs=[
                    {
                        "title": f"True per-timestep {flag_display_name}",
                        "intervals": cnn_true_intervals,
                        "palette": cnn_plot_palette,
                    },
                    {
                        "title": f"CNN per-timestep {flag_display_name} prediction",
                        "intervals": cnn_pred_intervals,
                        "palette": cnn_plot_palette,
                    },
                ],
                measurement_column=plot_measurement_column,
                secondary_column=plot_secondary_column,
                max_points=max_points_to_plot,
                title="Baseline CNN per-timestep predictions on a selected time range",
                label_meanings=label_meanings,
                legend_title=flag_display_name,
            )
            plt.show()

    if cnn_interval_metrics is not None:
        cnn_cm_fig, cnn_cm_ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(
            confusion_matrix=cnn_interval_metrics["confusion_matrix"],
            display_labels=cnn_interval_metrics["display_labels"],
        ).plot(ax=cnn_cm_ax, cmap="Blues", colorbar=False)
        cnn_cm_ax.set_title("CNN confusion matrix on the selected range")
        plt.tight_layout()
        plt.show()

    return {
        "ran": cnn_interval_metrics is not None,
        "range_selection": cnn_range_selection,
        "interval_origin": cnn_interval_origin,
        "interval_metrics": cnn_interval_metrics,
        "figure": cnn_demo_figure,
    }


def summarize_issue_adequacy(
    split_frames: dict[str, pd.DataFrame],
    *,
    label_column: str,
    issue_labels: list[int] | tuple[int, ...],
    min_issue_rows: int,
    min_issue_share_pct: float,
    min_rows_per_issue_label: int,
) -> pd.DataFrame:
    """Summarize whether each split has enough issue examples for evaluation."""

    adequacy_rows = []
    normalized_issue_labels = [int(label) for label in issue_labels]

    for split_name, frame in split_frames.items():
        labels = pd.to_numeric(frame[label_column], errors="coerce").dropna().astype(int)
        total_rows = int(len(labels))
        label_counts = labels.value_counts().sort_index()
        issue_counts = label_counts.reindex(normalized_issue_labels, fill_value=0).astype(int)
        issue_rows = int(issue_counts.sum())
        issue_share_pct = round(100 * issue_rows / total_rows, 2) if total_rows else 0.0
        per_label_ok = all(int(issue_counts.get(label, 0)) >= min_rows_per_issue_label for label in normalized_issue_labels)

        adequacy_rows.append(
            {
                "split": split_name,
                "rows": total_rows,
                "issue_rows": issue_rows,
                "issue_share_pct": issue_share_pct,
                "meets_issue_row_floor": issue_rows >= min_issue_rows,
                "meets_issue_share_floor": issue_share_pct >= min_issue_share_pct,
                "meets_per_issue_label_floor": per_label_ok,
                "adequate_for_model_selection": (
                    issue_rows >= min_issue_rows
                    and issue_share_pct >= min_issue_share_pct
                    and per_label_ok
                ),
                **{f"flag_{label}_count": int(issue_counts.get(label, 0)) for label in normalized_issue_labels},
            }
        )

    return pd.DataFrame(adequacy_rows).set_index("split")


def show_episode_aware_split_comparison(
    *,
    reviewed_label_df: pd.DataFrame,
    base_strategy: str,
    base_strategy_label: str,
    train_fraction: float,
    validation_fraction: float,
    active_split_block_rows: int,
    episode_clean_block_rows: int,
    episode_context_rows: int,
    episode_merge_gap_rows: int,
    episode_purge_gap_rows: int,
    target_flag: str,
    issue_labels: list[int] | tuple[int, ...],
    comparison_labels: list[int] | tuple[int, ...],
    flag_palette: dict[int, str] | None = None,
    legend_title: str = "model target",
    validation_min_issue_rows: int = 100,
    validation_min_issue_share_pct: float = 1.0,
    validation_min_rows_per_issue_label: int = 20,
) -> dict[str, object]:
    """Compare a selected basic split with the advanced episode-aware split.

    The basic split shows what participants already saw in the intro notebook.
    The episode-aware split keeps issue episodes, their context rows, and purge
    buffers together so nearby windows are less likely to leak across train,
    validation, and test. The function displays the comparison tables and plot,
    then returns the underlying split frames and metrics for later inspection.
    """

    split_candidates = {
        base_strategy_label: split_frame_by_strategy(
            reviewed_label_df,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            strategy=base_strategy,
            block_rows=active_split_block_rows,
            target_column=target_flag,
            issue_labels=issue_labels,
        ),
        "Episode-aware blocks": split_frame_by_strategy(
            reviewed_label_df,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            strategy="episode_aware",
            block_rows=episode_clean_block_rows,
            target_column=target_flag,
            issue_labels=issue_labels,
            episode_context_rows=episode_context_rows,
            episode_merge_gap_rows=episode_merge_gap_rows,
            purge_gap_rows=episode_purge_gap_rows,
        ),
    }

    metric_rows = []
    adequacy_rows = []
    plot_rows = []
    for strategy_label, strategy_frames in split_candidates.items():
        strategy_counts, strategy_shares = summarize_split_distributions(
            strategy_frames,
            label_column=target_flag,
        )
        adequacy_frame = summarize_issue_adequacy(
            strategy_frames,
            label_column=target_flag,
            issue_labels=issue_labels,
            min_issue_rows=validation_min_issue_rows,
            min_issue_share_pct=validation_min_issue_share_pct,
            min_rows_per_issue_label=validation_min_rows_per_issue_label,
        ).reset_index()
        adequacy_frame.insert(0, "strategy", strategy_label)
        adequacy_rows.append(adequacy_frame)

        validation_row = adequacy_frame[adequacy_frame["split"] == "validation"].iloc[0]
        metric_rows.append(
            {
                "strategy": strategy_label,
                **compute_split_share_gap(strategy_shares),
                "train_rows": int(strategy_counts["train"].sum()),
                "validation_rows": int(strategy_counts["validation"].sum()),
                "test_rows": int(strategy_counts["test"].sum()),
                "validation_issue_rows": int(validation_row["issue_rows"]),
                "validation_issue_share_pct": float(validation_row["issue_share_pct"]),
                "validation_adequate": bool(validation_row["adequate_for_model_selection"]),
            }
        )

        for split_name in ["train", "validation", "test"]:
            row = strategy_shares[split_name].reindex(comparison_labels, fill_value=0.0).copy()
            row.name = f"{strategy_label} | {split_name} (n={int(strategy_counts[split_name].sum()):,})"
            plot_rows.append(row)

    metric_frame, split_detail = build_split_strategy_tables(metric_rows, adequacy_rows)
    display(
        metric_frame.style.format(
            {
                "train_rows": "{:,.0f}",
                "validation_rows": "{:,.0f}",
                "test_rows": "{:,.0f}",
                "validation_issue_rows": "{:,.0f}",
                "validation_issue_share_pct": "{:.2f}",
                "max_split_gap": "{:.4f}",
                "mean_split_gap": "{:.4f}",
            }
        )
    )
    display(
        split_detail.style.format(
            {
                "train_rows": "{:,.0f}",
                "train_issue_rows": "{:,.0f}",
                "train_issue_share_pct": "{:.2f}",
                "validation_rows": "{:,.0f}",
                "validation_issue_rows": "{:,.0f}",
                "validation_issue_share_pct": "{:.2f}",
                "test_rows": "{:,.0f}",
                "test_issue_rows": "{:,.0f}",
                "test_issue_share_pct": "{:.2f}",
            }
        )
    )

    plot_frame = pd.DataFrame(plot_rows)
    palette = dict(DEFAULT_FLAG_PALETTE)
    if flag_palette:
        palette.update(flag_palette)
    plot_palette = [palette.get(int(label), "#64748b") for label in plot_frame.columns]

    fig, ax = plt.subplots(figsize=(13.2, 4.8))
    plot_frame.plot(kind="barh", stacked=True, ax=ax, color=plot_palette, width=0.72)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Share of reviewed rows in the split")
    ax.set_ylabel("")
    ax.set_title("Selected basic split vs the advanced episode-aware fixed split")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(title=legend_title, bbox_to_anchor=(1.01, 1.0), loc="upper left")
    ax.axhline(2.5, color="#cbd5e1", linewidth=1.2)
    plt.tight_layout()
    plt.show()

    return {
        "split_candidates": split_candidates,
        "metric_frame": metric_frame,
        "split_detail": split_detail,
        "plot_frame": plot_frame,
    }


def build_split_strategy_tables(
    strategy_metric_rows: list[dict[str, object]],
    strategy_adequacy_frames: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build compact summary/detail tables for split-strategy comparisons."""

    summary_frame = pd.DataFrame(strategy_metric_rows).set_index("strategy").copy()
    summary_frame = summary_frame.rename(
        columns={
            "rows": "reviewed_rows_total",
            "train_rows": "train_rows",
            "validation_rows": "validation_rows",
            "test_rows": "test_rows",
            "validation_issue_rows": "validation_issue_rows",
            "validation_issue_share_pct": "validation_issue_share_pct",
            "validation_adequate": "validation_ok",
            "max_pairwise_total_variation": "max_split_gap",
            "mean_pairwise_total_variation": "mean_split_gap",
        }
    )
    if "validation_ok" in summary_frame.columns:
        summary_frame["validation_ok"] = summary_frame["validation_ok"].map({True: "yes", False: "no"})

    detail_frame = pd.concat(strategy_adequacy_frames, ignore_index=True).copy()
    if detail_frame.empty:
        return summary_frame, detail_frame

    detail_frame["adequate_for_model_selection"] = detail_frame["adequate_for_model_selection"].map(
        {True: "yes", False: "no"}
    )
    detail_frame = detail_frame.rename(columns={"adequate_for_model_selection": "adequate"})
    pivot_frame = (
        detail_frame.set_index(["strategy", "split"])[["rows", "issue_rows", "issue_share_pct", "adequate"]]
        .unstack("split")
        .swaplevel(0, 1, axis=1)
        .sort_index(axis=1, level=0)
    )
    pivot_frame.columns = [f"{split}_{metric}" for split, metric in pivot_frame.columns]
    pivot_frame = pivot_frame.rename(
        columns={
            "train_adequate": "train_ok",
            "validation_adequate": "validation_ok",
            "test_adequate": "test_ok",
        }
    )
    return summary_frame, pivot_frame


def _time_span(frame: pd.DataFrame, time_column: str = "Time UTC") -> list[str] | None:
    """Return ``[start, end]`` ISO strings for a dataframe time span."""

    if frame.empty or time_column not in frame.columns:
        return None
    return [
        frame[time_column].min().isoformat(),
        frame[time_column].max().isoformat(),
    ]


def show_split_strategy_comparison(
    strategy_frames_by_name: dict[str, dict[str, pd.DataFrame]],
    *,
    strategy_display_names: dict[str, str],
    issue_labels: list[int] | tuple[int, ...],
    label_column: str,
    flag_palette: dict[int, str] | None = None,
    legend_title: str = "target label",
    figure_title: str = "How the split strategy changes label balance on the reviewed modelling dataset",
) -> dict[str, object]:
    """Plot each split strategy in its own column with reviewed and issue-only views."""

    palette = dict(DEFAULT_FLAG_PALETTE)
    if flag_palette:
        palette.update(flag_palette)

    strategy_names = [name for name in strategy_display_names if name in strategy_frames_by_name]
    split_order = ["train", "validation", "test"]
    strategy_counts: dict[str, pd.DataFrame] = {}
    strategy_shares: dict[str, pd.DataFrame] = {}
    strategy_issue_counts: dict[str, pd.DataFrame] = {}
    strategy_issue_shares: dict[str, pd.DataFrame] = {}

    fig, axes = plt.subplots(
        2,
        max(len(strategy_names), 1),
        figsize=(5.2 * max(len(strategy_names), 1), 8.0),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    for column_index, strategy_name in enumerate(strategy_names):
        strategy_frames = strategy_frames_by_name[strategy_name]
        split_counts, split_shares = summarize_split_distributions(strategy_frames, label_column=label_column)
        issue_only_counts = split_counts.reindex(
            [label for label in split_counts.index if int(label) in {int(issue) for issue in issue_labels}],
            fill_value=0,
        )
        issue_only_shares = issue_only_counts.div(
            issue_only_counts.sum(axis=0).replace(0, np.nan),
            axis=1,
        ).fillna(0.0)

        strategy_counts[strategy_name] = split_counts
        strategy_shares[strategy_name] = split_shares
        strategy_issue_counts[strategy_name] = issue_only_counts
        strategy_issue_shares[strategy_name] = issue_only_shares

        display_names = {split_name: split_name for split_name in split_order}
        _plot_split_share_panel(
            axes[0, column_index],
            split_shares,
            split_counts,
            split_order=split_order,
            split_display_names=display_names,
            palette=palette,
            xlabel="Share of reviewed rows",
            title=strategy_display_names[strategy_name],
            empty_message="No reviewed rows were found.",
        )
        _plot_split_share_panel(
            axes[1, column_index],
            issue_only_shares,
            issue_only_counts,
            split_order=split_order,
            split_display_names=display_names,
            palette=palette,
            xlabel="Share of issue rows",
            title="",
            empty_message="No issue rows were found.",
            count_prefix="issue n",
        )

    axes[0, 0].set_ylabel("All reviewed labels")
    axes[1, 0].set_ylabel("Issue labels only")

    all_labels = []
    for counts in strategy_counts.values():
        all_labels.extend([int(label) for label in counts.index])
    all_labels = sorted(set(all_labels))
    issue_only_labels = []
    for counts in strategy_issue_counts.values():
        issue_only_labels.extend([int(label) for label in counts.index])
    issue_only_labels = sorted(set(issue_only_labels))

    all_handles = [Patch(facecolor=palette.get(int(label), "#64748b"), label=str(label)) for label in all_labels]
    issue_handles = [Patch(facecolor=palette.get(int(label), "#64748b"), label=str(label)) for label in issue_only_labels]
    if all_handles:
        fig.legend(all_handles, [handle.get_label() for handle in all_handles], title=legend_title, bbox_to_anchor=(1.01, 0.79), loc="center left")
    if issue_handles:
        fig.legend(
            issue_handles,
            [handle.get_label() for handle in issue_handles],
            title=f"Issue {legend_title}",
            bbox_to_anchor=(1.01, 0.27),
            loc="center left",
        )

    fig.suptitle(figure_title, y=0.98)
    plt.tight_layout(rect=[0.03, 0.02, 0.88, 0.95])
    plt.show()

    return {
        "counts": strategy_counts,
        "shares": strategy_shares,
        "issue_only_counts": strategy_issue_counts,
        "issue_only_shares": strategy_issue_shares,
    }


def show_split_strategy_timeline(
    strategy_frames_by_name: dict[str, dict[str, pd.DataFrame]],
    *,
    strategy_display_names: dict[str, str] | None = None,
    split_display_names: dict[str, str] | None = None,
    split_palette: dict[str, str] | None = None,
    time_column: str = "Time UTC",
    split_order: tuple[str, ...] = ("train", "validation", "test"),
    max_points_per_split: int = 1_500,
    interleaved_zoom_rows: int = 9_000,
    max_zoom_points_per_split: int = 3_000,
    figure_title: str = "Where each split lands across the selected dataset timeline",
) -> dict[str, object]:
    """Plot which timestamps each split receives for every split strategy.

    The plot uses a time-spread sample from each split instead of drawing every
    row. This keeps the figure readable for large parquet caches while still
    showing whether a split is one contiguous block, per-source slices, or
    repeated interleaved blocks through time. When an interleaved strategy is
    present, a second zoom panel shows a local row window so the alternating
    blocks are not hidden by the full time range.
    """

    strategy_display_names = strategy_display_names or {}
    split_display_names = split_display_names or {}
    split_palette = {
        "train": "#2563eb",
        "validation": "#f59e0b",
        "test": "#dc2626",
        **(split_palette or {}),
    }

    timeline_rows = []
    sampled_points = []
    y_labels = []
    y_positions = []
    strategy_groups = []
    y_position = 0.0
    row_step = 1.0
    group_gap = 0.85

    for strategy_name, split_frames in strategy_frames_by_name.items():
        strategy_label = strategy_display_names.get(strategy_name, strategy_name)
        group_start = y_position
        for split_name in split_order:
            if split_name not in split_frames:
                continue

            split_frame = split_frames[split_name]
            split_label = split_display_names.get(split_name, split_name)
            y_labels.append(f"{strategy_label} | {split_label}")
            y_positions.append(y_position)

            if split_frame.empty or time_column not in split_frame.columns:
                timeline_rows.append(
                    {
                        "strategy_key": strategy_name,
                        "strategy": strategy_label,
                        "split_key": split_name,
                        "split": split_label,
                        "rows": int(len(split_frame)),
                        "sampled_points": 0,
                        "time_start": None,
                        "time_end": None,
                    }
                )
                y_position += row_step
                continue

            time_frame = split_frame[[time_column]].copy()
            time_frame[time_column] = pd.to_datetime(time_frame[time_column], utc=True, errors="coerce")
            time_frame = time_frame.dropna(subset=[time_column]).sort_values(time_column).reset_index(drop=True)
            sampled_frame = evenly_spaced_take(
                time_frame,
                max_points_per_split,
                time_column=time_column,
            )
            sampled_points.append(
                pd.DataFrame(
                    {
                        "time": sampled_frame[time_column],
                        "y": y_position,
                        "strategy_key": strategy_name,
                        "strategy": strategy_label,
                        "split": split_label,
                        "split_key": split_name,
                    }
                )
            )
            timeline_rows.append(
                {
                    "strategy_key": strategy_name,
                    "strategy": strategy_label,
                    "split_key": split_name,
                    "split": split_label,
                    "rows": int(len(split_frame)),
                    "sampled_points": int(len(sampled_frame)),
                    "time_start": time_frame[time_column].min().isoformat() if len(time_frame) else None,
                    "time_end": time_frame[time_column].max().isoformat() if len(time_frame) else None,
                }
            )
            y_position += row_step
        group_end = y_position - row_step
        if group_end >= group_start:
            strategy_groups.append(
                {
                    "strategy_key": strategy_name,
                    "strategy": strategy_label,
                    "start": group_start,
                    "end": group_end,
                    "mid": (group_start + group_end) / 2,
                }
            )
            y_position += group_gap

    timeline_summary = pd.DataFrame(timeline_rows)
    points_frame = pd.concat(sampled_points, ignore_index=True) if sampled_points else pd.DataFrame()

    if timeline_summary.empty:
        print("No split frames were available for the timeline plot.")
        return {"summary": timeline_summary, "points": points_frame, "figure": None}

    zoom_points = _build_interleaved_zoom_points(
        strategy_frames_by_name,
        strategy_display_names=strategy_display_names,
        split_display_names=split_display_names,
        split_order=split_order,
        time_column=time_column,
        interleaved_zoom_rows=interleaved_zoom_rows,
        max_zoom_points_per_split=max_zoom_points_per_split,
    )
    has_zoom = not zoom_points.empty

    fig_height = max(5.6, 0.52 * len(y_labels) + (3.6 if has_zoom else 2.0))
    if has_zoom:
        fig, (ax, zoom_ax) = plt.subplots(
            2,
            1,
            figsize=(14.8, fig_height),
            gridspec_kw={"height_ratios": [max(3.2, 0.45 * len(y_labels)), 2.4]},
        )
    else:
        fig, ax = plt.subplots(figsize=(14.8, fig_height))
        zoom_ax = None

    for group_index, group in enumerate(strategy_groups):
        group_color = "#f8fafc" if group_index % 2 == 0 else "#eef2f7"
        ax.axhspan(group["start"] - 0.48, group["end"] + 0.48, color=group_color, alpha=0.92, zorder=0)
        ax.axhline(group["start"] - 0.48, color="#cbd5e1", linewidth=1.0, zorder=1)
        ax.axhline(group["end"] + 0.48, color="#cbd5e1", linewidth=1.0, zorder=1)

    for split_name in split_order:
        if points_frame.empty:
            continue
        split_points = points_frame[points_frame["split_key"] == split_name]
        if split_points.empty:
            continue
        ax.scatter(
            split_points["time"],
            split_points["y"],
            marker="|",
            s=190,
            linewidths=1.4,
            color=split_palette.get(split_name, "#64748b"),
            alpha=0.82,
            label=split_display_names.get(split_name, split_name),
        )

    ax.set_yticks(y_positions, y_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Time UTC")
    ax.set_ylabel("")
    ax.set_title(figure_title)
    ax.grid(axis="x", alpha=0.22)
    ax.legend(title="Split", bbox_to_anchor=(1.01, 1.0), loc="upper left")
    ax.text(
        0.0,
        -0.12,
        "Each row is a time-spread sample from that split. The zoom below shows the interleaved blocks over a local row window.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#475569",
    )

    if has_zoom and zoom_ax is not None:
        zoom_y_positions = {split_name: index for index, split_name in enumerate(split_order)}
        for split_name in split_order:
            split_zoom = zoom_points[zoom_points["split_key"] == split_name]
            if split_zoom.empty:
                continue
            zoom_ax.scatter(
                split_zoom["time"],
                split_zoom["split_key"].map(zoom_y_positions),
                marker="|",
                s=190,
                linewidths=1.6,
                color=split_palette.get(split_name, "#64748b"),
                alpha=0.86,
                label=split_display_names.get(split_name, split_name),
            )
        zoom_labels = [split_display_names.get(split_name, split_name) for split_name in split_order]
        zoom_ax.set_yticks(range(len(split_order)), zoom_labels)
        zoom_ax.invert_yaxis()
        zoom_ax.set_xlabel("Time UTC")
        zoom_ax.set_ylabel("Interleaved split")
        zoom_ax.set_title(
            "Zoom: local interleaved blocks assign nearby rows to different splits"
        )
        zoom_ax.grid(axis="x", alpha=0.22)
        zoom_locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
        zoom_ax.xaxis.set_major_locator(zoom_locator)
        zoom_ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(zoom_locator))
        zoom_ax.text(
            0.0,
            -0.24,
            f"Zoom uses {len(zoom_points):,} sampled rows from a local consecutive-row window, so gaps and block alternation are visible.",
            transform=zoom_ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#475569",
        )

    plt.tight_layout(rect=[0.17, 0.03, 0.88, 0.97])
    plt.show()

    return {"summary": timeline_summary, "points": points_frame, "zoom_points": zoom_points, "figure": fig}


def _build_interleaved_zoom_points(
    strategy_frames_by_name: dict[str, dict[str, pd.DataFrame]],
    *,
    strategy_display_names: dict[str, str],
    split_display_names: dict[str, str],
    split_order: tuple[str, ...],
    time_column: str,
    interleaved_zoom_rows: int,
    max_zoom_points_per_split: int,
) -> pd.DataFrame:
    """Build a local time-window sample for the interleaved split strategy."""

    interleaved_strategy_name = None
    for strategy_name in strategy_frames_by_name:
        strategy_label = strategy_display_names.get(strategy_name, strategy_name)
        if "interleaved" in strategy_name.lower() or "interleaved" in strategy_label.lower():
            interleaved_strategy_name = strategy_name
            break
    if interleaved_strategy_name is None:
        return pd.DataFrame()

    combined_frames = []
    strategy_label = strategy_display_names.get(interleaved_strategy_name, interleaved_strategy_name)
    split_frames = strategy_frames_by_name[interleaved_strategy_name]
    for split_name in split_order:
        split_frame = split_frames.get(split_name)
        if split_frame is None or split_frame.empty or time_column not in split_frame.columns:
            continue
        time_frame = split_frame[[time_column]].copy()
        time_frame[time_column] = pd.to_datetime(time_frame[time_column], utc=True, errors="coerce")
        time_frame = time_frame.dropna(subset=[time_column]).sort_values(time_column).reset_index(drop=True)
        if time_frame.empty:
            continue
        time_frame["split_key"] = split_name
        time_frame["split"] = split_display_names.get(split_name, split_name)
        time_frame["strategy"] = strategy_label
        combined_frames.append(time_frame.rename(columns={time_column: "time"}))

    if not combined_frames:
        return pd.DataFrame()

    combined = pd.concat(combined_frames, ignore_index=True).sort_values("time").reset_index(drop=True)
    if combined.empty:
        return pd.DataFrame()

    zoom_row_count = max(1, min(int(interleaved_zoom_rows), len(combined)))
    center = int(len(combined) * 0.45)
    start = max(0, center - zoom_row_count // 2)
    stop = min(len(combined), start + zoom_row_count)
    start = max(0, stop - zoom_row_count)
    zoom_window = combined.iloc[start:stop].copy()

    sampled_frames = []
    for split_name in split_order:
        split_window = zoom_window[zoom_window["split_key"] == split_name].reset_index(drop=True)
        if split_window.empty:
            continue
        sampled_frames.append(
            evenly_spaced_take(
                split_window,
                max_zoom_points_per_split,
                time_column="time",
            )
        )
    return pd.concat(sampled_frames, ignore_index=True) if sampled_frames else pd.DataFrame()


def _plot_split_share_panel(
    ax,
    share_frame: pd.DataFrame,
    count_frame: pd.DataFrame,
    *,
    split_order: list[str],
    split_display_names: dict[str, str],
    palette: dict[int, str],
    xlabel: str,
    title: str,
    empty_message: str,
    count_prefix: str = "n",
) -> None:
    """Draw one stacked horizontal split-composition panel."""

    available_splits = [split_name for split_name in split_order if split_name in count_frame.columns]
    if not available_splits:
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(axis="x", alpha=0.2)
        ax.text(0.5, 0.5, empty_message, transform=ax.transAxes, ha="center", va="center")
        return

    plot_frame = share_frame.T.reindex(available_splits).fillna(0.0)
    plot_frame.index = [
        f"{split_display_names.get(split_name, split_name)} ({count_prefix}={int(count_frame[split_name].sum()):,})"
        for split_name in plot_frame.index
    ]

    if plot_frame.empty or float(plot_frame.to_numpy().sum()) == 0.0:
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
        ax.tick_params(axis="y", labelleft=True)
        ax.set_yticks(range(len(plot_frame.index)), plot_frame.index.tolist())
        ax.text(0.5, 0.5, empty_message, transform=ax.transAxes, ha="center", va="center")
        return

    plot_colors = [palette.get(int(label), "#64748b") for label in plot_frame.columns]
    plot_frame.plot(kind="barh", stacked=True, ax=ax, color=plot_colors, width=0.64, legend=False)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelleft=True)
    ax.grid(axis="x", alpha=0.2)
