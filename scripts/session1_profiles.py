"""Dataset-profile resolver functions for Session 1.

Participant-editable preset values live in ``workshop_config/``. This module is
the code layer that validates those values, resolves paths, and turns a selected
profile into the explicit notebook variables used downstream.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from workshop_config.session1_dataset_profiles import (
    CUSTOM_DATASET_PROFILE_ID,
    CUSTOM_DATASET_TEMPLATE,
    DATASET_PROFILES,
    WORKSHOP_DATASET_PROFILE_IDS,
    WORKSHOP_DATASET_PROFILES,
)
from workshop_config.session1_defaults import (
    CONDUCTIVITY_PLUG_LABEL_MEANINGS,
    DEFAULT_GOOD_LABELS,
    DEFAULT_ISSUE_LABELS,
)

def get_dataset_profile(dataset_profile_id: str) -> dict[str, Any]:
    """Return a copy of one supported dataset profile."""

    if dataset_profile_id not in DATASET_PROFILES:
        supported = ", ".join(sorted(DATASET_PROFILES))
        raise ValueError(f"Unknown dataset_profile_id={dataset_profile_id!r}. Choose one of: {supported}.")
    return deepcopy(DATASET_PROFILES[dataset_profile_id])


def profile_option_summary(*, include_custom: bool = False) -> list[dict[str, object]]:
    """Return a compact table explaining the selectable dataset profile ids."""

    profile_ids = (
        list(DATASET_PROFILES)
        if include_custom
        else list(WORKSHOP_DATASET_PROFILE_IDS)
    )
    return [
        {
            "dataset_profile_id": profile_id,
            "label": profile["label"],
            "target_flag": profile["target_flag"],
            "primary_device": profile["primary_device"],
        }
        for profile_id in profile_ids
        for profile in [DATASET_PROFILES[profile_id]]
    ]


def label_display_context(*, dataset_profile_id: str, target_flag: str) -> dict[str, object]:
    """Return plotting labels that match the active target column."""

    if dataset_profile_id == "conductivity_plugs":
        return {
            "FLAG_EXAMPLE_TARGET": "ml_label",
            "FLAG_EXAMPLE_DISPLAY_NAME": "ml_label",
            "FLAG_EXAMPLE_LABEL_MEANINGS": CONDUCTIVITY_PLUG_LABEL_MEANINGS,
            "FLAG_EXAMPLE_AVOID_CONTEXT_LABELS": (4,),
        }

    if dataset_profile_id == "custom":
        return {
            "FLAG_EXAMPLE_TARGET": target_flag,
            "FLAG_EXAMPLE_DISPLAY_NAME": target_flag,
            "FLAG_EXAMPLE_LABEL_MEANINGS": None,
            "FLAG_EXAMPLE_AVOID_CONTEXT_LABELS": (),
        }

    return {
        "FLAG_EXAMPLE_TARGET": target_flag,
        "FLAG_EXAMPLE_DISPLAY_NAME": "QC flag",
        "FLAG_EXAMPLE_LABEL_MEANINGS": None,
        "FLAG_EXAMPLE_AVOID_CONTEXT_LABELS": (9,),
    }


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


def _optional_str(value: Any) -> str | None:
    """Return ``None`` unchanged, otherwise coerce display/config values to text."""

    return None if value is None else str(value)


def _resolve_project_base(base_dir: str | Path | None) -> Path:
    """Resolve a generic project/job directory without requiring a notebook."""

    start = Path.cwd() if base_dir is None else Path(base_dir).expanduser()
    for candidate in [start, *start.parents]:
        if (candidate / "scripts").exists() and (candidate / "notebooks").exists():
            return candidate
    return start


def build_window_feature_columns(measurement_columns: list[str]) -> list[str]:
    """Return the window-summary feature names used by k-means."""

    return [
        f"{column}_{stat}"
        for column in measurement_columns
        for stat in ("mean", "std")
    ]


def build_dataset_profile_namespace(
    *,
    dataset_profile_id: str,
    notebook_root: str | Path | None = None,
    shared_daqathon_root: str | Path | None,
    local_cache_dir: str | Path,
    shared_cache_dir: str | Path | None,
    manual_raw_data_dir: str | Path | None = None,
    manual_target_flag: str | None = None,
    manual_measurement_columns: list[str] | None = None,
    manual_optional_qc_columns: list[str] | None = None,
    manual_plot_measurement_column: str | None = None,
    manual_plot_secondary_column: str | None = None,
    manual_read_cache_dir: str | Path | None = None,
    manual_raw_csv_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    manual_cache_bundle_name: str | None = None,
    manual_kmeans_feature_mode: str | None = None,
    manual_dataset_label: str | None = None,
    manual_task_mode: str | None = None,
    manual_good_labels: list[int] | tuple[int, ...] | None = None,
    manual_issue_labels: list[int] | tuple[int, ...] | None = None,
    manual_flag_example_classes: list[int] | tuple[int, ...] | None = None,
    manual_primary_device: str | None = None,
    manual_sequence_output_mode: str | None = None,
    manual_sequence_target_strategy: str | None = None,
    manual_csv_header: str | int = "auto",
    manual_time_column: str = "Time UTC",
    manual_auto_build_missing_cache: bool | None = None,
    manual_default_prep_sample_rows: int | None = None,
) -> dict[str, object]:
    """Build the small set of notebook variables derived from a profile.

    The returned dictionary contains values that the notebooks historically used
    as top-level variables. Centralising the derivation here keeps the visible
    notebook cells focused on the controls a participant is likely to change.

    New code should usually call ``resolve_workshop_dataset_config`` for the
    prepared workshop datasets or ``resolve_custom_dataset_config`` for
    participant-owned data. This lower-level function remains available for
    backwards compatibility with older notebook cells.
    """

    profile = get_dataset_profile(dataset_profile_id)
    notebook_root = _resolve_project_base(notebook_root)
    local_cache_dir = Path(local_cache_dir).expanduser()
    shared_root = Path(shared_daqathon_root).expanduser() if shared_daqathon_root else None
    shared_cache = Path(shared_cache_dir).expanduser() if shared_cache_dir else None

    profile_raw_candidates = [Path(str(subpath)) for subpath in profile["raw_subpaths"]]
    local_raw_candidates = [notebook_root / "data" / "raw" / subpath for subpath in profile_raw_candidates]
    shared_raw_candidates = (
        [shared_root / "data" / "raw" / subpath for subpath in profile_raw_candidates]
        if shared_root
        else []
    )

    raw_candidates = [candidate for candidate in [*shared_raw_candidates, *local_raw_candidates] if candidate is not None]
    detected_raw_data_dir = (
        _first_existing_csv_dir(raw_candidates)
        or _first_existing_path([*raw_candidates, notebook_root / "data" / "raw"])
    )
    detected_cache_dir = _first_existing_path([candidate for candidate in [local_cache_dir, shared_cache] if candidate is not None])

    read_raw_data_dir = manual_raw_data_dir or (
        str(detected_raw_data_dir)
        if detected_raw_data_dir is not None
        else str(local_raw_candidates[0] if local_raw_candidates else notebook_root / "data" / "raw")
    )
    read_cache_dir = (
        str(Path(manual_read_cache_dir).expanduser())
        if manual_read_cache_dir is not None
        else str(detected_cache_dir) if detected_cache_dir is not None else str(local_cache_dir)
    )

    target_flag = manual_target_flag or str(profile["target_flag"])
    task_mode = manual_task_mode or str(profile["task_mode"])
    measurement_columns = list(
        profile["measurement_columns"]
        if manual_measurement_columns is None
        else manual_measurement_columns
    )
    optional_qc_columns = list(
        dict.fromkeys(
            profile["optional_qc_columns"]
            if manual_optional_qc_columns is None
            else manual_optional_qc_columns
        )
    )
    plot_measurement_column = str(manual_plot_measurement_column or profile["plot_measurement_column"])
    plot_secondary_column = _optional_str(
        profile["plot_secondary_column"]
        if manual_plot_secondary_column is None
        else manual_plot_secondary_column
    )
    window_feature_columns = build_window_feature_columns(measurement_columns)
    raw_csv_paths = [str(Path(path).expanduser()) for path in (manual_raw_csv_paths or [])]
    is_custom_dataset = dataset_profile_id == CUSTOM_DATASET_PROFILE_ID or any(
        value is not None
        for value in [
            manual_raw_data_dir,
            manual_read_cache_dir,
            manual_raw_csv_paths,
            manual_target_flag,
            manual_measurement_columns,
            manual_cache_bundle_name,
        ]
    )
    good_labels = [
        int(label)
        for label in (
            profile.get("good_labels", DEFAULT_GOOD_LABELS)
            if manual_good_labels is None
            else manual_good_labels
        )
    ]
    issue_labels = [
        int(label)
        for label in (
            profile.get("issue_labels", DEFAULT_ISSUE_LABELS)
            if manual_issue_labels is None
            else manual_issue_labels
        )
    ]
    flag_example_classes = tuple(
        int(label)
        for label in (manual_flag_example_classes or profile.get("flag_example_classes", [*good_labels, *issue_labels]))
    )

    return {
        "DATASET_PROFILES": DATASET_PROFILES,
        "DATASET_PROFILE_ID": dataset_profile_id,
        "DATASET_PROFILE": profile,
        "DATASET_LABEL": manual_dataset_label or profile["label"],
        "IS_CUSTOM_DATASET": is_custom_dataset,
        "READ_RAW_DATA_DIR": str(read_raw_data_dir),
        "READ_CACHE_DIR": read_cache_dir,
        "RAW_CSV_PATHS": raw_csv_paths,
        "TIME_COLUMN": str(manual_time_column),
        "CSV_HEADER": manual_csv_header,
        "TARGET_FLAG": target_flag,
        "TASK_MODE": str(task_mode),
        "GOOD_LABELS": good_labels,
        "ISSUE_LABELS": issue_labels,
        "FLAG_EXAMPLE_CLASSES": flag_example_classes,
        "CACHE_BUNDLE_NAME": manual_cache_bundle_name or str(profile["cache_stem"]),
        "PRIMARY_DEVICE": manual_primary_device or str(profile["primary_device"]),
        "KMEANS_FEATURE_MODE": manual_kmeans_feature_mode or str(profile["kmeans_feature_mode"]),
        "DEFAULT_SEQUENCE_OUTPUT_MODE": str(manual_sequence_output_mode or profile.get("default_sequence_output_mode", "window")),
        "DEFAULT_SEQUENCE_TARGET_STRATEGY": str(
            manual_sequence_target_strategy or profile.get("default_sequence_target_strategy", "collapsed_1_34_9")
        ),
        "RAW_CACHE_READ_SAMPLE_ROWS": profile.get("cache_read_sample_rows"),
        "AUTO_BUILD_MISSING_CACHE": (
            bool(manual_auto_build_missing_cache)
            if manual_auto_build_missing_cache is not None
            else bool(profile.get("auto_build_missing_cache", True))
        ),
        "DEFAULT_PREP_SAMPLE_ROWS": manual_default_prep_sample_rows if manual_default_prep_sample_rows is not None else profile.get("default_prep_sample_rows"),
        "MEASUREMENT_COLUMNS": measurement_columns,
        "OPTIONAL_QC_COLUMNS": optional_qc_columns,
        "PLOT_MEASUREMENT_COLUMN": plot_measurement_column,
        "PLOT_SECONDARY_COLUMN": plot_secondary_column,
        "ROW_USE_COLUMNS": list(dict.fromkeys(["Time UTC", "source_file", target_flag, *optional_qc_columns, *measurement_columns])),
        "WINDOW_FEATURE_COLUMNS": window_feature_columns,
        "WINDOW_USE_COLUMNS": ["window_start", "window_end", "source_file", "issue_rate", *window_feature_columns],
        **label_display_context(dataset_profile_id=dataset_profile_id, target_flag=target_flag),
    }


def resolve_workshop_dataset_config(
    *,
    dataset_profile_id: str,
    base_dir: str | Path | None = None,
    notebook_root: str | Path | None = None,
    shared_daqathon_root: str | Path | None = None,
    local_cache_dir: str | Path | None = None,
    shared_cache_dir: str | Path | None = None,
) -> dict[str, object]:
    """Resolve one pre-loaded workshop dataset into notebook/job settings.

    This is the preferred entry point for the prepared DAQathon datasets on a
    laptop, Nibi, Narval, or FIR. It intentionally does not accept custom label
    or column overrides; use ``resolve_custom_dataset_config`` when the data are
    not one of the prepared workshop presets. ``base_dir`` can be any project or
    job working directory. ``notebook_root`` is accepted only as a
    backwards-compatible alias for older calls.
    """

    if base_dir is not None and notebook_root is not None:
        raise ValueError("Use only one of base_dir or notebook_root.")

    if dataset_profile_id not in WORKSHOP_DATASET_PROFILE_IDS:
        supported = ", ".join(WORKSHOP_DATASET_PROFILE_IDS)
        raise ValueError(
            f"{dataset_profile_id!r} is not a prepared workshop dataset. "
            f"Choose one of: {supported}. For participant data, call "
            "resolve_custom_dataset_config(...)."
        )

    project_base = _resolve_project_base(base_dir if base_dir is not None else notebook_root)
    return build_dataset_profile_namespace(
        dataset_profile_id=dataset_profile_id,
        notebook_root=project_base,
        shared_daqathon_root=shared_daqathon_root,
        local_cache_dir=local_cache_dir or project_base / "data" / "cache" / "session1",
        shared_cache_dir=shared_cache_dir,
    )


def resolve_custom_dataset_config(
    *,
    target_flag: str,
    measurement_columns: list[str] | tuple[str, ...],
    base_dir: str | Path | None = None,
    notebook_root: str | Path | None = None,
    raw_data_dir: str | Path | None = None,
    raw_csv_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    parquet_cache_dir: str | Path | None = None,
    cache_bundle_name: str = "custom_session1",
    dataset_label: str = "Custom scalar dataset",
    task_mode: str = "multiclass",
    good_labels: list[int] | tuple[int, ...] = (0,),
    issue_labels: list[int] | tuple[int, ...] = (1,),
    optional_qc_columns: list[str] | tuple[str, ...] | None = None,
    plot_measurement_column: str | None = None,
    plot_secondary_column: str | None = None,
    flag_example_classes: list[int] | tuple[int, ...] | None = None,
    kmeans_feature_mode: str = "row_level",
    primary_device: str = "other",
    sequence_output_mode: str = "per_timestep",
    sequence_target_strategy: str = "raw_multiclass",
    csv_header: str | int = "first_row",
    time_column: str = "Time UTC",
    auto_build_missing_cache: bool | None = None,
    default_prep_sample_rows: int | None = None,
) -> dict[str, object]:
    """Resolve participant-owned scalar data into notebook/job settings.

    Use this for scripts, batch jobs, or notebook cells that should run outside
    the workshop preset system. The caller supplies the target label, feature
    columns, label groups, and either CSV inputs or an existing parquet cache
    location.

    ``base_dir`` is only used for default cache/raw-data folders when explicit
    paths are omitted. It can be any project or job working directory; it does
    not need to be a notebook repository. ``notebook_root`` is accepted only as a
    backwards-compatible alias for older calls.
    """

    if not measurement_columns:
        raise ValueError("measurement_columns must contain at least one numeric feature column.")

    if base_dir is not None and notebook_root is not None:
        raise ValueError("Use only one of base_dir or notebook_root.")

    resolved_base_dir = Path.cwd()
    if base_dir is not None:
        resolved_base_dir = Path(base_dir).expanduser()
    elif notebook_root is not None:
        resolved_base_dir = Path(notebook_root).expanduser()

    resolved_cache_dir = (
        Path(parquet_cache_dir).expanduser()
        if parquet_cache_dir is not None
        else resolved_base_dir / "data" / "cache" / "session1"
    )
    resolved_raw_dir = raw_data_dir
    if resolved_raw_dir is None and raw_csv_paths:
        resolved_raw_dir = Path(raw_csv_paths[0]).expanduser().parent
    if resolved_raw_dir is None:
        resolved_raw_dir = resolved_base_dir / "data" / "raw"

    if auto_build_missing_cache is None:
        auto_build_missing_cache = raw_data_dir is not None or bool(raw_csv_paths)

    return build_dataset_profile_namespace(
        dataset_profile_id=CUSTOM_DATASET_PROFILE_ID,
        notebook_root=resolved_base_dir,
        shared_daqathon_root=None,
        local_cache_dir=resolved_cache_dir,
        shared_cache_dir=None,
        manual_dataset_label=dataset_label,
        manual_raw_data_dir=resolved_raw_dir,
        manual_raw_csv_paths=raw_csv_paths,
        manual_read_cache_dir=resolved_cache_dir,
        manual_cache_bundle_name=cache_bundle_name,
        manual_target_flag=target_flag,
        manual_task_mode=task_mode,
        manual_measurement_columns=list(measurement_columns),
        manual_optional_qc_columns=list(optional_qc_columns or []),
        manual_good_labels=list(good_labels),
        manual_issue_labels=list(issue_labels),
        manual_flag_example_classes=list(flag_example_classes or issue_labels),
        manual_plot_measurement_column=plot_measurement_column or str(measurement_columns[0]),
        manual_plot_secondary_column=plot_secondary_column,
        manual_kmeans_feature_mode=kmeans_feature_mode,
        manual_primary_device=primary_device,
        manual_sequence_output_mode=sequence_output_mode,
        manual_sequence_target_strategy=sequence_target_strategy,
        manual_csv_header=csv_header,
        manual_time_column=time_column,
        manual_auto_build_missing_cache=auto_build_missing_cache,
        manual_default_prep_sample_rows=default_prep_sample_rows,
    )


def dataset_profile_summary(config: dict[str, object]) -> dict[str, object]:
    """Return the profile-derived values worth printing in a notebook setup cell."""

    return {
        "DATASET_PROFILE_ID": config["DATASET_PROFILE_ID"],
        "DATASET_LABEL": config["DATASET_LABEL"],
        "IS_CUSTOM_DATASET": config["IS_CUSTOM_DATASET"],
        "READ_RAW_DATA_DIR": config["READ_RAW_DATA_DIR"],
        "READ_CACHE_DIR": config["READ_CACHE_DIR"],
        "RAW_CSV_PATHS": config["RAW_CSV_PATHS"],
        "TIME_COLUMN": config["TIME_COLUMN"],
        "CSV_HEADER": config["CSV_HEADER"],
        "TARGET_FLAG": config["TARGET_FLAG"],
        "CACHE_BUNDLE_NAME": config["CACHE_BUNDLE_NAME"],
        "KMEANS_FEATURE_MODE": config["KMEANS_FEATURE_MODE"],
        "DEFAULT_SEQUENCE_OUTPUT_MODE": config["DEFAULT_SEQUENCE_OUTPUT_MODE"],
        "DEFAULT_SEQUENCE_TARGET_STRATEGY": config["DEFAULT_SEQUENCE_TARGET_STRATEGY"],
        "GOOD_LABELS": config["GOOD_LABELS"],
        "ISSUE_LABELS": config["ISSUE_LABELS"],
        "MEASUREMENT_COLUMNS": config["MEASUREMENT_COLUMNS"],
        "PLOT_MEASUREMENT_COLUMN": config["PLOT_MEASUREMENT_COLUMN"],
        "PLOT_SECONDARY_COLUMN": config["PLOT_SECONDARY_COLUMN"],
    }


def dataset_profile_summary_rows(config: dict[str, object]) -> list[dict[str, object]]:
    """Return profile-derived values as grouped rows for notebook display."""

    return [
        {"group": "dataset", "name": "DATASET_PROFILE_ID", "value": config["DATASET_PROFILE_ID"], "what_it_controls": "Which preset/template supplied the dataset settings."},
        {"group": "dataset", "name": "DATASET_LABEL", "value": config["DATASET_LABEL"], "what_it_controls": "Human-readable dataset name."},
        {"group": "data source", "name": "READ_RAW_DATA_DIR", "value": config["READ_RAW_DATA_DIR"], "what_it_controls": "Folder used when raw CSVs need to be read or cached."},
        {"group": "data source", "name": "RAW_CSV_PATHS", "value": config["RAW_CSV_PATHS"], "what_it_controls": "Optional explicit CSV files for custom datasets."},
        {"group": "cache", "name": "READ_CACHE_DIR", "value": config["READ_CACHE_DIR"], "what_it_controls": "Folder where prepared parquet cache bundles are read."},
        {"group": "cache", "name": "CACHE_BUNDLE_NAME", "value": config["CACHE_BUNDLE_NAME"], "what_it_controls": "Shared stem for row parquet, window parquet, and metadata files."},
        {"group": "labels", "name": "TARGET_FLAG", "value": config["TARGET_FLAG"], "what_it_controls": "Column predicted by supervised models."},
        {"group": "labels", "name": "GOOD_LABELS", "value": config["GOOD_LABELS"], "what_it_controls": "Labels treated as good/normal examples."},
        {"group": "labels", "name": "ISSUE_LABELS", "value": config["ISSUE_LABELS"], "what_it_controls": "Labels treated as issue/anomaly examples."},
        {"group": "features", "name": "MEASUREMENT_COLUMNS", "value": config["MEASUREMENT_COLUMNS"], "what_it_controls": "Numeric sensor columns used by ML models."},
        {"group": "features", "name": "OPTIONAL_QC_COLUMNS", "value": config["OPTIONAL_QC_COLUMNS"], "what_it_controls": "Extra QC/context columns kept for inspection."},
        {"group": "plots", "name": "PLOT_MEASUREMENT_COLUMN", "value": config["PLOT_MEASUREMENT_COLUMN"], "what_it_controls": "Primary y-axis column in time-series plots."},
        {"group": "plots", "name": "PLOT_SECONDARY_COLUMN", "value": config["PLOT_SECONDARY_COLUMN"], "what_it_controls": "Optional secondary y-axis column in time-series plots."},
        {"group": "models", "name": "KMEANS_FEATURE_MODE", "value": config["KMEANS_FEATURE_MODE"], "what_it_controls": "Whether k-means reads window summaries or row-level features."},
        {"group": "models", "name": "DEFAULT_SEQUENCE_OUTPUT_MODE", "value": config["DEFAULT_SEQUENCE_OUTPUT_MODE"], "what_it_controls": "Default CNN/Transformer prediction shape."},
    ]
