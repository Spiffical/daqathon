from __future__ import annotations

from pathlib import Path

from . import prepare_scalar_session1_data

__all__ = [
    "create_onc_row_level_parquet_cache",
    "create_onc_window_summary_parquet_cache",
]


def create_onc_row_level_parquet_cache(
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
    """Create a row-level parquet cache from ONC scalar CSV exports.

    This is the ONC-specific cache builder. It reads the raw scalar CSV files,
    detects the selected primary device stream, optionally aligns companion
    device streams onto the primary timestamps, and writes one parquet part per
    selected primary CSV file.

    Parameters
    ----------
    raw_data_dir:
        Directory containing ONC scalar CSV exports.
    cache_root:
        Directory where the cache bundle should be written.
    cache_bundle_name:
        Short cache stem used for the row-level folder and metadata filename.
    target_flag:
        Label column to preserve and summarize, such as ``"Conductivity QC Flag"``.
    primary_device:
        Device family that defines the row timestamps. Supported values are
        ``"ctd"``, ``"fluorometer"``, ``"oxygen"``, and ``"other"``.
    measurement_columns:
        Numeric columns to preserve as modelling features.
    max_files:
        Optional cap on the number of primary-device CSV files to process.
    sample_rows:
        Optional per-file row cap for quick smoke tests.
    merge_tolerance_seconds:
        Maximum timestamp distance allowed when aligning companion devices.

    Returns
    -------
    dict
        The low-level row-cache result, resolved cache paths, and a concise
        summary suitable for printing in a notebook or script.
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


def create_onc_window_summary_parquet_cache(
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
    """Create an ONC window-summary parquet cache from row-level parquet parts.

    The row-level cache keeps every timestamp. This second step groups those
    rows into fixed-size contiguous windows and writes one summary row per
    window. The k-means section uses these summaries so clustering can operate
    on compact windows instead of millions of raw rows.

    Parameters
    ----------
    row_cache_result:
        Result returned by ``create_onc_row_level_parquet_cache``.
    target_flag:
        Label column summarized inside each window.
    issue_labels:
        Label ids counted as issue/anomaly labels in window summaries.
    window_size:
        Number of row-level timestamps per summary window.
    sample_rows:
        Per-file row cap used in the row-level step; recorded in metadata.
    merge_tolerance_seconds:
        Timestamp alignment tolerance used in the row-level step; recorded in
        metadata for reproducibility.
    primary_device:
        Primary device family used in the row-level step.
    max_files:
        Primary-device file cap used in the row-level step; recorded in metadata.

    Returns
    -------
    dict
        The window-cache result, metadata dictionary, resolved cache paths, and
        a concise printable summary.
    """

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
            "step": "window_summary_parquet",
            "window_cache": str(cache_paths.window_cache_path),
            "metadata": str(cache_paths.metadata_path),
            "windows": int(window_result.window_count),
            "window_columns": window_result.window_columns,
        },
    }
