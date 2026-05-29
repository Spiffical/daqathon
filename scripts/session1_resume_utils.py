from __future__ import annotations

import builtins
import importlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import prepare_scalar_session1_data as _prepare_scalar_session1_data
from . import session1_intro_utils as _session1_intro_utils
from . import session1_modeling as _session1_modeling
from .session1_profiles import DATASET_PROFILES, label_display_context as build_label_display_context
from .session1_intro_notebook_setup import (
    INTRO_UTIL_EXPORTS,
    MODELING_EXPORTS,
    PREP_EXPORTS,
    build_intro_notebook_namespace,
)


def _first_existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _first_existing_csv_dir(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir() and any(candidate.glob("*.csv")):
            return candidate
    return None


def _find_notebook_root(notebook_root: str | Path | None) -> Path:
    start = Path.cwd() if notebook_root is None else Path(notebook_root)
    start = start.expanduser().resolve()
    for candidate_root in [start, *start.parents]:
        if (candidate_root / "notebooks").exists() and (candidate_root / "scripts").exists():
            return candidate_root
    return start


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def show_setup_json(value) -> None:
    """Print small notebook status dictionaries in a readable JSON format."""

    builtins.print(json.dumps(_jsonable(value), indent=2, ensure_ascii=False))


def _select_exports(module, names: list[str]) -> dict[str, object]:
    return {name: getattr(module, name) for name in names}


def evenly_spaced_take(frame: pd.DataFrame, limit: int | None, *, time_column: str = "Time UTC") -> pd.DataFrame:
    """Take rows from across the full time range instead of only the start.

    This mirrors the small teachable helper in the intro notebook. If the time
    column is present, rows are sorted first so the sampled result is still easy
    to read chronologically.
    """

    ordered = (
        frame.sort_values(time_column).reset_index(drop=True).copy()
        if time_column in frame.columns
        else frame.reset_index(drop=True).copy()
    )
    if limit is None or len(ordered) <= limit:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, num=limit, dtype=int)
    return ordered.iloc[indices].reset_index(drop=True)


def _available_columns(row_part_paths: list[Path], metadata: Mapping[str, object]) -> list[str]:
    metadata_columns = metadata.get("row_columns")
    if isinstance(metadata_columns, list) and metadata_columns:
        return [str(column) for column in metadata_columns]
    if not row_part_paths:
        return []
    return pq.ParquetFile(row_part_paths[0]).schema_arrow.names


def _resolve_profile_paths(
    *,
    notebook_root: Path,
    profile: Mapping[str, object],
    read_raw_data_dir: str | Path | None,
    read_cache_dir: str | Path | None,
) -> tuple[Path, Path, Path | None, Path | None]:
    shared_root_candidates = [
        Path("/project/def-kmoran/shared/daqathon"),
        Path("/project/6062898/shared/daqathon"),
        Path.home() / "projects" / "def-kmoran" / "shared" / "daqathon",
    ]
    shared_daqathon_root = _first_existing_path(shared_root_candidates)
    local_cache_dir = notebook_root / "data" / "cache" / "session1"
    shared_cache_dir = (
        _first_existing_path([shared_daqathon_root / "data" / "cache" / "session1"])
        if shared_daqathon_root
        else None
    )

    profile_raw_candidates = [Path(str(subpath)) for subpath in profile["raw_subpaths"]]
    local_raw_candidates = [notebook_root / "data" / "raw" / subpath for subpath in profile_raw_candidates]
    shared_raw_candidates = (
        [shared_daqathon_root / "data" / "raw" / subpath for subpath in profile_raw_candidates]
        if shared_daqathon_root
        else []
    )

    detected_raw_data_dir = (
        _first_existing_csv_dir([candidate for candidate in [*shared_raw_candidates, *local_raw_candidates] if candidate is not None])
        or _first_existing_path(
            [candidate for candidate in [*shared_raw_candidates, *local_raw_candidates, notebook_root / "data" / "raw"] if candidate is not None]
        )
    )
    detected_cache_dir = _first_existing_path([candidate for candidate in [local_cache_dir, shared_cache_dir] if candidate is not None])

    raw_dir = (
        Path(read_raw_data_dir).expanduser().resolve()
        if read_raw_data_dir is not None
        else detected_raw_data_dir or local_raw_candidates[0]
    )
    cache_dir = (
        Path(read_cache_dir).expanduser().resolve()
        if read_cache_dir is not None
        else detected_cache_dir or local_cache_dir
    )
    return raw_dir, cache_dir, shared_daqathon_root, shared_cache_dir


def load_ml_section_state(
    *,
    notebook_root: str | Path | None = None,
    dataset_profile_id: str = "ctd_conductivity",
    data_fraction: float = 0.1,
    use_data_fraction_budgets: bool = True,
    split_strategy: str = "global_contiguous",
    train_subset_strategy: str = "time_spread",
    reviewed_file_selection_mode: str = "spread",
    reviewed_file_limit: int | None = None,
    reviewed_model_row_limit: int | None = None,
    train_subset_max_rows: int | None = None,
    issue_rows_per_file: int | None = None,
    balanced_reviewed_target_issue_share: float = 0.25,
    interleaved_block_rows: int = 1024,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 21,
    read_raw_data_dir: str | Path | None = None,
    read_cache_dir: str | Path | None = None,
    raw_csv_dir: str | Path | None = None,
    raw_csv_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    parquet_cache_dir: str | Path | None = None,
    cache_bundle_name: str | None = None,
    target_flag: str | None = None,
    measurement_columns: list[str] | tuple[str, ...] | None = None,
    optional_qc_columns: list[str] | tuple[str, ...] | None = None,
    good_labels: list[int] | tuple[int, ...] | None = None,
    issue_labels: list[int] | tuple[int, ...] | None = None,
    plot_measurement_column: str | None = None,
    plot_secondary_column: str | None = None,
    primary_device: str | None = None,
    kmeans_feature_mode: str | None = None,
    csv_header: str | int = "auto",
    time_column: str = "Time UTC",
    build_cache_if_missing: bool = False,
    force_rebuild_cache: bool = False,
    generic_csv_cache: bool | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    """Rebuild the objects needed to start any ML section after a skipped run.

    The returned dictionary is intentionally explicit: notebooks can store it as
    ``ML_STATE`` and then assign the specific variables they need. It restores
    the common imports, dataset preset choices, cache paths, selected reviewed
    modelling rows, train/validation/test frames, train-only subset, and
    dataset-specific label-display variables used by the Random Forest,
    k-means, CNN, and Transformer sections.

    Most workshop runs only need to adjust the dataset preset,
    ``data_fraction``, split strategy, and train-subset strategy.
    ``data_fraction`` controls the reviewed modelling row cap as a fraction of
    the actual usable reviewed rows, plus the train-only subset cap. For
    example, ``0.1`` gives a quick 10% reviewed-row run and ``0.9`` gives a
    much larger 90% reviewed-row run. Use ``1.0`` to remove row caps.

    Set ``use_data_fraction_budgets=False`` to keep every reviewed row for the
    setup-time dataset and train-subset steps. Model-specific row/window caps
    are intentionally defined in the notebook section that uses each model.

    Custom-data runs can pass ``raw_csv_dir`` or ``raw_csv_paths`` plus
    ``target_flag`` and ``measurement_columns``. If ``build_cache_if_missing``
    is true, the helper will create a simple row-level parquet cache when the
    requested cache bundle is not already present.
    """

    if dataset_profile_id not in DATASET_PROFILES:
        supported = ", ".join(sorted(DATASET_PROFILES))
        raise ValueError(f"Unknown dataset_profile_id={dataset_profile_id!r}. Choose one of: {supported}.")
    if not 0 < data_fraction <= 1:
        raise ValueError("data_fraction must be in the interval (0, 1].")
    repo_root = _find_notebook_root(notebook_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    importlib.invalidate_caches()
    prepare_scalar_session1_data = importlib.reload(_prepare_scalar_session1_data)
    session1_intro_utils = importlib.reload(_session1_intro_utils)
    session1_modeling = importlib.reload(_session1_modeling)

    profile = DATASET_PROFILES[dataset_profile_id]
    # These defaults mirror the visible row-budget cells in the notebooks.
    # They are compute caps for a live workshop, not statistical rules.
    train_subset_base_rows = 1_000_000
    train_subset_min_rows = 10_000
    issue_rows_per_file_base = 12_000
    issue_rows_per_file_min = 1_000
    if use_data_fraction_budgets and data_fraction < 0.999:
        budget_mode = "DATA_FRACTION dataset/split caps"
        budget_data_fraction = data_fraction

        def scaled(base_rows: int, minimum_rows: int) -> int:
            return max(minimum_rows, int(base_rows * data_fraction))

        computed_train_subset_max_rows = scaled(train_subset_base_rows, train_subset_min_rows)
        computed_issue_rows_per_file = scaled(issue_rows_per_file_base, issue_rows_per_file_min)
    else:
        budget_mode = "full reviewed data / no dataset row caps"
        budget_data_fraction = 1.0
        computed_train_subset_max_rows = None
        computed_issue_rows_per_file = issue_rows_per_file_base

    if train_subset_max_rows is None:
        train_subset_max_rows = computed_train_subset_max_rows
    if issue_rows_per_file is None:
        issue_rows_per_file = computed_issue_rows_per_file

    raw_dir, cache_dir, shared_daqathon_root, shared_cache_dir = _resolve_profile_paths(
        notebook_root=repo_root,
        profile=profile,
        read_raw_data_dir=raw_csv_dir or read_raw_data_dir,
        read_cache_dir=parquet_cache_dir or read_cache_dir,
    )

    custom_label_groups = good_labels is not None or issue_labels is not None
    target_flag = str(target_flag or profile["target_flag"])
    task_mode = str(profile["task_mode"])
    good_labels = [
        int(label)
        for label in (
            profile.get("good_labels", [1])
            if good_labels is None
            else good_labels
        )
    ]
    issue_labels = [
        int(label)
        for label in (
            profile.get("issue_labels", [3, 4, 9])
            if issue_labels is None
            else issue_labels
        )
    ]
    flag_example_source = [*good_labels, *issue_labels] if custom_label_groups else profile.get("flag_example_classes", issue_labels)
    flag_example_classes = tuple(int(label) for label in flag_example_source)
    cache_bundle_name = str(cache_bundle_name or profile["cache_stem"])
    label_display_context = build_label_display_context(
        dataset_profile_id=dataset_profile_id,
        target_flag=target_flag,
    )
    measurement_columns = list(profile["measurement_columns"] if measurement_columns is None else measurement_columns)
    optional_qc_columns = list(
        dict.fromkeys(profile["optional_qc_columns"] if optional_qc_columns is None else optional_qc_columns)
    )
    plot_measurement_column = str(plot_measurement_column or profile["plot_measurement_column"])
    plot_secondary_column_value = (
        profile["plot_secondary_column"]
        if plot_secondary_column is None
        else plot_secondary_column
    )
    plot_secondary_column = None if plot_secondary_column_value is None else str(plot_secondary_column_value)
    primary_device = str(primary_device or profile["primary_device"])
    requested_kmeans_feature_mode = kmeans_feature_mode
    kmeans_feature_mode = str(requested_kmeans_feature_mode or profile["kmeans_feature_mode"])

    namespace = build_intro_notebook_namespace(
        notebook_root=repo_root,
        read_raw_data_dir=raw_dir,
        read_cache_dir=cache_dir,
        cache_bundle_name=cache_bundle_name,
        seed=seed,
    )
    namespace["show_setup_json"] = show_setup_json

    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    runtime_raw_data_dir = Path(namespace["RUNTIME_RAW_DATA_DIR"])
    runtime_cache_dir = Path(namespace["RUNTIME_CACHE_DIR"])
    read_raw_path = Path(raw_dir)
    read_cache_path = Path(cache_dir)
    if generic_csv_cache is None:
        generic_csv_cache = raw_csv_dir is not None or raw_csv_paths is not None
    if generic_csv_cache and requested_kmeans_feature_mode is None:
        kmeans_feature_mode = "row_level"

    prebuilt_cache_paths = session1_intro_utils.choose_cache_bundle_paths(
        [read_cache_path],
        cache_stem=cache_bundle_name,
    )
    if force_rebuild_cache or (build_cache_if_missing and not prebuilt_cache_paths.metadata_path.exists()):
        cache_build_result = session1_intro_utils.resolve_or_create_parquet_cache(
            cache_root=read_cache_path,
            cache_bundle_name=cache_bundle_name,
            raw_data_dir=raw_dir,
            raw_csv_paths=raw_csv_paths,
            target_flag=target_flag,
            measurement_columns=measurement_columns,
            optional_qc_columns=optional_qc_columns,
            issue_labels=issue_labels,
            time_column=time_column,
            csv_header=csv_header,
            build_cache_if_missing=build_cache_if_missing,
            force_rebuild_cache=force_rebuild_cache,
            generic_csv_cache=bool(generic_csv_cache),
            primary_device=primary_device,
            sample_rows=None,
        )
        if verbose:
            show_setup_json(cache_build_result["summary"])

    active_raw_data_dir = (
        runtime_raw_data_dir
        if slurm_tmpdir and runtime_raw_data_dir.exists() and any(runtime_raw_data_dir.glob("*.csv"))
        else read_raw_path
    )
    active_cache_paths = session1_intro_utils.choose_cache_bundle_paths(
        [runtime_cache_dir, read_cache_path],
        cache_stem=cache_bundle_name,
    )
    row_cache_dir = active_cache_paths.row_level_dir
    window_cache_path = active_cache_paths.window_cache_path
    metadata_path = active_cache_paths.metadata_path

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Could not find the parquet cache metadata for {dataset_profile_id!r}: {metadata_path}. "
            "Run the data-preparation section once, or choose a profile with an existing cache."
        )

    metadata = json.loads(metadata_path.read_text())
    part_paths = sorted(row_cache_dir.glob("*.parquet"))
    if not part_paths:
        raise FileNotFoundError(f"No row-level parquet parts were found in {row_cache_dir}.")

    selected_paths = session1_intro_utils.select_part_paths(
        part_paths,
        limit=reviewed_file_limit,
        mode=reviewed_file_selection_mode,
    )
    processed_files = metadata.get("processed_files", [])
    part_to_source = {
        Path(str(file_info["row_level_part"])).name: str(file_info["source_file"])
        for file_info in processed_files
        if "row_level_part" in file_info and "source_file" in file_info
    }
    part_to_source.update(
        {
            str(part_name): str(source_file)
            for part_name, source_file in metadata.get("part_to_source_file", {}).items()
        }
    )
    source_to_row_part = {
        str(file_info["source_file"]): row_cache_dir / Path(str(file_info["row_level_part"])).name
        for file_info in processed_files
        if "row_level_part" in file_info and "source_file" in file_info
    }
    for part_name, source_file in part_to_source.items():
        source_to_row_part.setdefault(str(source_file), row_cache_dir / Path(part_name).name)
    selected_source_files = [
        part_to_source.get(path.name, path.name.replace(".parquet", ".csv"))
        for path in selected_paths
    ]

    available_row_columns = _available_columns(part_paths, metadata)
    row_use_columns = [
        column
        for column in dict.fromkeys(["Time UTC", "source_file", target_flag, *optional_qc_columns, *measurement_columns])
        if column in available_row_columns
    ]
    missing_core_columns = [column for column in ["Time UTC", "source_file", target_flag] if column not in row_use_columns]
    if missing_core_columns:
        raise ValueError(f"The active cache is missing required columns: {missing_core_columns}")

    window_feature_columns = [
        f"{column}_{stat}"
        for column in measurement_columns
        for stat in ("mean", "std")
    ]
    window_use_columns = [
        "window_start",
        "window_end",
        "source_file",
        "issue_rate",
        *window_feature_columns,
    ]

    reviewed_label_source_df = session1_modeling.load_full_row_level_frame(
        selected_paths,
        columns=["Time UTC", "source_file", target_flag],
    )
    if "source_file" in reviewed_label_source_df.columns and not reviewed_label_source_df.empty:
        selected_source_files = sorted(reviewed_label_source_df["source_file"].dropna().astype(str).unique().tolist())

    available_window_columns = []
    window_df = pd.DataFrame()
    if window_cache_path.exists():
        available_window_columns = pq.ParquetFile(window_cache_path).schema_arrow.names
        resume_window_columns = [column for column in window_use_columns if column in available_window_columns]
        if {"window_start", "source_file"}.issubset(resume_window_columns):
            window_df = pd.read_parquet(window_cache_path, columns=resume_window_columns)
            window_df["window_start"] = pd.to_datetime(window_df["window_start"], utc=True)
            if "window_end" in window_df.columns:
                window_df["window_end"] = pd.to_datetime(window_df["window_end"], utc=True)
            window_df = (
                window_df[window_df["source_file"].isin(selected_source_files)]
                .sort_values("window_start")
                .reset_index(drop=True)
            )
    source_rows = len(reviewed_label_source_df)
    reviewed_label_df, _ = session1_modeling.build_reviewed_target_frame(
        reviewed_label_source_df,
        target_flag=target_flag,
        task_mode=task_mode,
        good_labels=good_labels,
        issue_labels=issue_labels,
        model_row_limit=None,
    )
    reviewed_source_rows = len(reviewed_label_df)
    if reviewed_model_row_limit is None:
        if use_data_fraction_budgets and data_fraction < 0.999:
            reviewed_model_row_limit = max(1, int(round(reviewed_source_rows * data_fraction)))
        else:
            reviewed_model_row_limit = reviewed_source_rows
    reviewed_model_df = (
        session1_modeling.evenly_spaced_take(reviewed_label_df, reviewed_model_row_limit)
        if reviewed_model_row_limit is not None and len(reviewed_label_df) > reviewed_model_row_limit
        else reviewed_label_df.copy()
    )
    active_labels = sorted(reviewed_model_df["model_target"].dropna().astype(int).unique().tolist())
    if reviewed_model_df.empty:
        raise ValueError("The selected cache parts did not contain any reviewed modelling rows.")
    del reviewed_label_source_df

    fixed_split_frames = session1_modeling.split_frame_by_strategy(
        reviewed_model_df,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        strategy=split_strategy,
        source_column="source_file",
        block_rows=interleaved_block_rows,
        issue_column="issue",
        target_column=target_flag,
        issue_labels=issue_labels,
    )
    train_full_df = fixed_split_frames["train"].reset_index(drop=True)
    valid_df = fixed_split_frames["validation"].reset_index(drop=True)
    test_df = fixed_split_frames["test"].reset_index(drop=True)

    model_good_labels = [0] if task_mode == "binary" else list(good_labels)
    model_issue_labels = [1] if task_mode == "binary" else list(issue_labels)
    train_df = session1_modeling.sample_frame_by_strategy(
        train_full_df,
        rows_limit=train_subset_max_rows,
        sample_strategy=train_subset_strategy,
        target_flag="model_target",
        good_labels=model_good_labels,
        issue_labels=model_issue_labels,
        issue_rows=issue_rows_per_file,
        balanced_issue_share=balanced_reviewed_target_issue_share,
    )

    is_custom_dataset = dataset_profile_id == "custom" or any(
        value is not None
        for value in [raw_csv_dir, raw_csv_paths, parquet_cache_dir]
    )
    dataset_config = {
        "DATASET_PROFILES": DATASET_PROFILES,
        "DATASET_PROFILE_ID": dataset_profile_id,
        "DATASET_PROFILE": profile,
        "DATASET_LABEL": profile["label"],
        "IS_CUSTOM_DATASET": is_custom_dataset,
        "READ_RAW_DATA_DIR": str(raw_dir),
        "READ_CACHE_DIR": str(cache_dir),
        "RAW_CSV_PATHS": [str(path) for path in (raw_csv_paths or [])],
        "TIME_COLUMN": time_column,
        "CSV_HEADER": csv_header,
        "TARGET_FLAG": target_flag,
        "TASK_MODE": task_mode,
        "GOOD_LABELS": good_labels,
        "ISSUE_LABELS": issue_labels,
        "FLAG_EXAMPLE_CLASSES": flag_example_classes,
        "CACHE_BUNDLE_NAME": cache_bundle_name,
        "PRIMARY_DEVICE": primary_device,
        "KMEANS_FEATURE_MODE": kmeans_feature_mode,
        "DEFAULT_SEQUENCE_OUTPUT_MODE": profile.get("default_sequence_output_mode", "window"),
        "DEFAULT_SEQUENCE_TARGET_STRATEGY": profile.get("default_sequence_target_strategy", "collapsed_1_34_9"),
        "RAW_CACHE_READ_SAMPLE_ROWS": profile.get("cache_read_sample_rows"),
        "AUTO_BUILD_MISSING_CACHE": bool(profile.get("auto_build_missing_cache", True)),
        "DEFAULT_PREP_SAMPLE_ROWS": profile.get("default_prep_sample_rows"),
        "MEASUREMENT_COLUMNS": measurement_columns,
        "OPTIONAL_QC_COLUMNS": optional_qc_columns,
        "PLOT_MEASUREMENT_COLUMN": plot_measurement_column,
        "PLOT_SECONDARY_COLUMN": plot_secondary_column,
        "ROW_USE_COLUMNS": row_use_columns,
        "WINDOW_FEATURE_COLUMNS": window_feature_columns,
        "WINDOW_USE_COLUMNS": window_use_columns,
        **label_display_context,
    }

    setup_summary = {
        "DATASET_PROFILE_ID": dataset_profile_id,
        "DATA_FRACTION": data_fraction,
        "USE_DATA_FRACTION_BUDGETS": use_data_fraction_budgets,
        "BUDGET_MODE": budget_mode,
        "TARGET_FLAG": target_flag,
        "TASK_MODE": task_mode,
        "CACHE_DIR": str(active_cache_paths.root),
        "ROW_CACHE_DIR": str(row_cache_dir),
        "WINDOW_CACHE_PATH": str(window_cache_path),
        "selected_cache_parts": len(selected_paths),
        "reviewed_file_selection_mode": reviewed_file_selection_mode,
        "reviewed_file_limit": reviewed_file_limit,
        "source_rows": source_rows,
        "reviewed_rows_before_limit": reviewed_source_rows,
        "reviewed_model_row_limit": reviewed_model_row_limit,
        "reviewed_model_rows": len(reviewed_model_df),
        "split_strategy": split_strategy,
        "interleaved_block_rows": interleaved_block_rows,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "train_full_rows": len(train_full_df),
        "validation_rows": len(valid_df),
        "test_rows": len(test_df),
        "train_subset_strategy": train_subset_strategy,
        "train_subset_max_rows": train_subset_max_rows,
        "issue_rows_per_file": issue_rows_per_file,
        "balanced_reviewed_target_issue_share": balanced_reviewed_target_issue_share,
        "train_subset_rows": len(train_df),
    }
    if verbose:
        show_setup_json(setup_summary)

    namespace.update(_select_exports(session1_modeling, MODELING_EXPORTS))
    namespace.update(_select_exports(session1_intro_utils, INTRO_UTIL_EXPORTS))
    namespace.update(_select_exports(prepare_scalar_session1_data, PREP_EXPORTS))
    namespace.update(
        {
            "NOTEBOOK_ROOT": repo_root,
            "SHARED_DAQATHON_ROOT": shared_daqathon_root,
            "LOCAL_CACHE_DIR": repo_root / "data" / "cache" / "session1",
            "SHARED_CACHE_DIR": shared_cache_dir,
            "DATASET_PROFILES": DATASET_PROFILES,
            "DATASET_CONFIG": dataset_config,
            "DATASET_PROFILE_ID": dataset_profile_id,
            "DATASET_PROFILE": profile,
            "DATASET_LABEL": profile["label"],
            "READ_RAW_DATA_DIR": str(raw_dir),
            "READ_CACHE_DIR": str(cache_dir),
            "RAW_CSV_PATHS": [str(path) for path in (raw_csv_paths or [])],
            "TIME_COLUMN": time_column,
            "CSV_HEADER": csv_header,
            "RAW_DATA_DIR": str(active_raw_data_dir),
            "CACHE_DIR": str(active_cache_paths.root),
            "ROW_CACHE_DIR": str(row_cache_dir),
            "WINDOW_CACHE_PATH": str(window_cache_path),
            "METADATA_PATH": str(metadata_path),
            "CACHE_BUNDLE_NAME": cache_bundle_name,
            "TARGET_FLAG": target_flag,
            "TASK_MODE": task_mode,
            "task_mode": task_mode,
            "target_name": target_flag if task_mode == "multiclass" else "issue",
            **label_display_context,
            "GOOD_LABELS": good_labels,
            "ISSUE_LABELS": issue_labels,
            "FLAG_EXAMPLE_CLASSES": flag_example_classes,
            "MEASUREMENT_COLUMNS": measurement_columns,
            "measurement_columns": measurement_columns,
            "OPTIONAL_QC_COLUMNS": optional_qc_columns,
            "PLOT_MEASUREMENT_COLUMN": plot_measurement_column,
            "PLOT_SECONDARY_COLUMN": plot_secondary_column,
            "PRIMARY_DEVICE": primary_device,
            "KMEANS_FEATURE_MODE": kmeans_feature_mode,
            "DEFAULT_SEQUENCE_OUTPUT_MODE": profile.get("default_sequence_output_mode", "window"),
            "DEFAULT_SEQUENCE_TARGET_STRATEGY": profile.get("default_sequence_target_strategy", "collapsed_1_34_9"),
            "DATA_FRACTION": data_fraction,
            "USE_DATA_FRACTION_BUDGETS": use_data_fraction_budgets,
            "BUDGET_MODE": budget_mode,
            "BUDGET_DATA_FRACTION": budget_data_fraction,
            "FULL_REVIEWED_ROW_COUNT": reviewed_source_rows,
            "TRAIN_SUBSET_BASE_ROWS": train_subset_base_rows,
            "TRAIN_SUBSET_MIN_ROWS": train_subset_min_rows,
            "ISSUE_ROWS_PER_FILE_BASE": issue_rows_per_file_base,
            "ISSUE_ROWS_PER_FILE_MIN": issue_rows_per_file_min,
            "BASE_ISSUE_ROWS_PER_FILE": 12_000,
            "BASE_MODEL_ROW_LIMIT": 1_000_000,
            "BASE_WINDOW_POINT_LIMIT": 1.0,
            "BASE_FLAG_EXAMPLE_POINTS_PER_PANEL": 30_000,
            "TRAIN_FRACTION": train_fraction,
            "VALIDATION_FRACTION": validation_fraction,
            "SEED": seed,
            "ROW_USE_COLUMNS": row_use_columns,
            "WINDOW_FEATURE_COLUMNS": window_feature_columns,
            "WINDOW_USE_COLUMNS": window_use_columns,
            "available_window_columns": available_window_columns,
            "metadata": metadata,
            "cache_paths": active_cache_paths,
            "active_cache_paths": active_cache_paths,
            "row_cache_dir": row_cache_dir,
            "window_cache_path": window_cache_path,
            "part_paths": part_paths,
            "selected_paths": selected_paths,
            "part_to_source": part_to_source,
            "source_to_row_part": source_to_row_part,
            "selected_source_files": selected_source_files,
            "available_row_columns": available_row_columns,
            "source_rows": source_rows,
            "reviewed_source_rows": reviewed_source_rows,
            "reviewed_label_df": reviewed_label_df,
            "reviewed_model_df": reviewed_model_df,
            "model_df": reviewed_model_df,
            "active_labels": active_labels,
            "model_good_labels": model_good_labels,
            "model_issue_labels": model_issue_labels,
            "REVIEWED_FILE_SELECTION_MODE": reviewed_file_selection_mode,
            "REVIEWED_FILE_LIMIT": reviewed_file_limit,
            "SPLIT_STRATEGY": split_strategy,
            "FIXED_SPLIT_STRATEGY": split_strategy,
            "REVIEWED_MODEL_ROW_LIMIT": reviewed_model_row_limit,
            "INTERLEAVED_BLOCK_ROWS": interleaved_block_rows,
            "SPLIT_BLOCK_ROWS": interleaved_block_rows,
            "ACTIVE_SPLIT_BLOCK_ROWS": interleaved_block_rows,
            "FIXED_SPLIT_BLOCK_ROWS": interleaved_block_rows if split_strategy == "interleaved_blocks" else None,
            "fixed_split_frames": fixed_split_frames,
            "train_full_df": train_full_df,
            "valid_df": valid_df,
            "test_df": test_df,
            "window_df": window_df,
            "TRAIN_SUBSET_STRATEGY": train_subset_strategy,
            "TRAIN_SUBSET_MAX_ROWS": train_subset_max_rows,
            "ISSUE_ROWS_PER_FILE": issue_rows_per_file,
            "BALANCED_REVIEWED_TARGET_ISSUE_SHARE": balanced_reviewed_target_issue_share,
            "train_df": train_df,
            "CNN_RUN": False,
            "SEQUENCE_CNN_RUN": False,
            "TRANSFORMER_RUN": False,
            "FORECAST_TRANSFORMER_RUN": False,
            "evenly_spaced_take": evenly_spaced_take,
            "RUN_HYPERPARAMETER_SEARCH": True,
            "AUTO_SELECT_SPLIT_BLOCK_ROWS": False,
            "EPISODE_CONTEXT_ROWS": 0,
            "EPISODE_MERGE_GAP_ROWS": 0,
            "EPISODE_PURGE_GAP_ROWS": 0,
            "ML_SECTION_STATE_SUMMARY": setup_summary,
        }
    )
    return namespace
