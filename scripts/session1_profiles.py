"""Dataset profiles and profile-derived notebook variables for Session 1.

The notebooks should let participants choose a dataset and adjust a small
number of meaningful controls. Dataset-specific plumbing such as cache names,
label columns, feature columns, and default plot columns belongs here so every
notebook and skip-ahead utility starts from the same source of truth.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .session1_defaults import (
    CONDUCTIVITY_PLUG_LABEL_MEANINGS,
    CTD_MEASUREMENT_COLUMNS,
    DEFAULT_GOOD_LABELS,
    DEFAULT_ISSUE_LABELS,
)


DATASET_PROFILES: dict[str, dict[str, Any]] = {
    "ctd_conductivity": {
        "label": "CTD conductivity QC",
        "description": "Strait of Georgia East CTD data with conductivity QC as the supervised target.",
        "raw_subpaths": ["SoGEast_CTD_202503_202603"],
        "cache_stem": "conductivity_scalar_session1",
        "target_flag": "Conductivity QC Flag",
        "task_mode": "multiclass",
        "good_labels": [1],
        "issue_labels": [3, 4, 9],
        "flag_example_classes": [1, 3, 4, 9],
        "measurement_columns": CTD_MEASUREMENT_COLUMNS,
        "optional_qc_columns": ["Temperature QC Flag"],
        "plot_measurement_column": "Conductivity (S/m)",
        "plot_secondary_column": "Temperature (C)",
        "primary_device": "ctd",
        "kmeans_feature_mode": "window_summary",
        "default_sequence_output_mode": "window",
        "default_sequence_target_strategy": "collapsed_1_34_9",
        "cache_read_sample_rows": None,
        "auto_build_missing_cache": True,
        "default_prep_sample_rows": None,
    },
    "fluorometer_turbidity": {
        "label": "Fluorometer turbidity QC",
        "description": "Merged scalar data around a fluorometer/turbidity target, including CTD and oxygen context columns.",
        "raw_subpaths": ["Fluorometer/SoGCentral", "Fluorometer/Folger", "Fluorometer/SoGCentral_test", "SoGCentral_test"],
        "cache_stem": "sogcentral_turbidity",
        "target_flag": "Turbidity QC Flag",
        "task_mode": "multiclass",
        "good_labels": [1],
        "issue_labels": [3, 4, 9],
        "flag_example_classes": [1, 3, 4, 9],
        "measurement_columns": [
            "Chlorophyll (ug/l)",
            "Turbidity (NTU)",
            "Conductivity (S/m)",
            "Density (kg/m3)",
            "Practical Salinity (psu)",
            "Pressure (decibar)",
            "Sigma-t (kg/m3)",
            "Sigma-theta (0 dbar) (kg/m3)",
            "Sound Speed (m/s)",
            "Temperature (C)",
            "Oxygen Concentration Corrected (ml/l)",
            "Oxygen Concentration Uncorrected (ml/l)",
        ],
        "optional_qc_columns": ["Chlorophyll QC Flag"],
        "plot_measurement_column": "Turbidity (NTU)",
        "plot_secondary_column": "Temperature (C)",
        "primary_device": "fluorometer",
        "kmeans_feature_mode": "row_level",
        "default_sequence_output_mode": "per_timestep",
        "default_sequence_target_strategy": "collapsed_1_34_9",
        "cache_read_sample_rows": None,
        "auto_build_missing_cache": True,
        "default_prep_sample_rows": None,
    },
    "oxygen": {
        "label": "Oxygen QC",
        "description": "Scalar oxygen data with oxygen concentration QC as the supervised target.",
        "raw_subpaths": ["Fluorometer/SoGCentral", "Fluorometer/Folger", "SoGEast_Oxygen_202503_202603", "Oxygen", "oxygen"],
        "cache_stem": "sogcentral_oxygen",
        "target_flag": "Oxygen Concentration Corrected QC Flag",
        "task_mode": "multiclass",
        "good_labels": [1],
        "issue_labels": [3, 4, 9],
        "flag_example_classes": [1, 2, 3, 4, 9],
        "measurement_columns": [
            "Oxygen Concentration Corrected (ml/l)",
            "Oxygen Concentration Uncorrected (ml/l)",
            "Temperature (C)",
            "Pressure (decibar)",
        ],
        "optional_qc_columns": ["Temperature QC Flag"],
        "plot_measurement_column": "Oxygen Concentration Corrected (ml/l)",
        "plot_secondary_column": "Temperature (C)",
        "primary_device": "oxygen",
        "kmeans_feature_mode": "window_summary",
        "default_sequence_output_mode": "window",
        "default_sequence_target_strategy": "collapsed_12_34_9",
        "cache_read_sample_rows": None,
        "auto_build_missing_cache": True,
        "default_prep_sample_rows": None,
    },
    "conductivity_plugs": {
        "label": "Conductivity plugs",
        "description": "Conductivity-plug data with a custom ml_label target and CTD/oxygen context columns.",
        "raw_subpaths": ["ConductivityPlugs"],
        "cache_stem": "conductivity_plugs_session1",
        "target_flag": "ml_label",
        "task_mode": "multiclass",
        "good_labels": [0],
        "issue_labels": [1, 2, 3, 4],
        "flag_example_classes": [1, 2, 3, 4],
        "measurement_columns": [
            "cond_value_ctd",
            "density_value_ctd",
            "Pressure_value_ctd",
            "salinity_value_ctd",
            "sigmaT_value_ctd",
            "SIGMA_THETA_value_ctd",
            "Sound_Speed_value_ctd",
            "Temperature_value_ctd",
            "oxygen_corrected_value_oxy",
            "oxygen_uncorrected_value_oxy",
            "temperature_value_oxy",
            "temperature_offset",
            "temperature_offset_anomaly",
            "temperature_offset_over_start_mean",
        ],
        "optional_qc_columns": ["cond_qaqc_ctd", "oxygen_corrected_qaqc_oxy", "temperature_qaqc_oxy"],
        "plot_measurement_column": "cond_value_ctd",
        "plot_secondary_column": "Temperature_value_ctd",
        "primary_device": "other",
        "kmeans_feature_mode": "window_summary",
        "default_sequence_output_mode": "per_timestep",
        "default_sequence_target_strategy": "raw_multiclass",
        "cache_read_sample_rows": 250_000,
        "auto_build_missing_cache": False,
        "default_prep_sample_rows": 500_000,
    },
}


def get_dataset_profile(dataset_profile_id: str) -> dict[str, Any]:
    """Return a copy of one supported dataset profile."""

    if dataset_profile_id not in DATASET_PROFILES:
        supported = ", ".join(sorted(DATASET_PROFILES))
        raise ValueError(f"Unknown dataset_profile_id={dataset_profile_id!r}. Choose one of: {supported}.")
    return deepcopy(DATASET_PROFILES[dataset_profile_id])


def profile_option_summary() -> list[dict[str, object]]:
    """Return a compact table explaining the supported profile ids."""

    return [
        {
            "dataset_profile_id": profile_id,
            "label": profile["label"],
            "target_flag": profile["target_flag"],
            "primary_device": profile["primary_device"],
        }
        for profile_id, profile in DATASET_PROFILES.items()
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
    notebook_root: str | Path,
    shared_daqathon_root: str | Path | None,
    local_cache_dir: str | Path,
    shared_cache_dir: str | Path | None,
    manual_raw_data_dir: str | Path | None = None,
    manual_target_flag: str | None = None,
    manual_measurement_columns: list[str] | None = None,
    manual_optional_qc_columns: list[str] | None = None,
    manual_plot_measurement_column: str | None = None,
    manual_plot_secondary_column: str | None = None,
    manual_cache_bundle_name: str | None = None,
    manual_kmeans_feature_mode: str | None = None,
) -> dict[str, object]:
    """Build the small set of notebook variables derived from a profile.

    The returned dictionary contains values that the notebooks historically used
    as top-level variables. Centralising the derivation here keeps the visible
    notebook cells focused on the controls a participant is likely to change.
    """

    profile = get_dataset_profile(dataset_profile_id)
    notebook_root = Path(notebook_root).expanduser()
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
    read_cache_dir = str(detected_cache_dir) if detected_cache_dir is not None else str(local_cache_dir)

    target_flag = manual_target_flag or str(profile["target_flag"])
    measurement_columns = list(manual_measurement_columns or profile["measurement_columns"])
    optional_qc_columns = list(dict.fromkeys(manual_optional_qc_columns or profile["optional_qc_columns"]))
    plot_measurement_column = manual_plot_measurement_column or str(profile["plot_measurement_column"])
    plot_secondary_column = manual_plot_secondary_column or str(profile["plot_secondary_column"])
    window_feature_columns = build_window_feature_columns(measurement_columns)

    return {
        "DATASET_PROFILES": DATASET_PROFILES,
        "DATASET_PROFILE_ID": dataset_profile_id,
        "DATASET_PROFILE": profile,
        "DATASET_LABEL": profile["label"],
        "READ_RAW_DATA_DIR": str(read_raw_data_dir),
        "READ_CACHE_DIR": read_cache_dir,
        "TARGET_FLAG": target_flag,
        "TASK_MODE": str(profile["task_mode"]),
        "GOOD_LABELS": [int(label) for label in profile.get("good_labels", DEFAULT_GOOD_LABELS)],
        "ISSUE_LABELS": [int(label) for label in profile.get("issue_labels", DEFAULT_ISSUE_LABELS)],
        "FLAG_EXAMPLE_CLASSES": tuple(int(label) for label in profile.get("flag_example_classes", [1, 3, 4, 9])),
        "CACHE_BUNDLE_NAME": manual_cache_bundle_name or str(profile["cache_stem"]),
        "PRIMARY_DEVICE": str(profile["primary_device"]),
        "KMEANS_FEATURE_MODE": manual_kmeans_feature_mode or str(profile["kmeans_feature_mode"]),
        "DEFAULT_SEQUENCE_OUTPUT_MODE": str(profile.get("default_sequence_output_mode", "window")),
        "DEFAULT_SEQUENCE_TARGET_STRATEGY": str(profile.get("default_sequence_target_strategy", "collapsed_1_34_9")),
        "RAW_CACHE_READ_SAMPLE_ROWS": profile.get("cache_read_sample_rows"),
        "AUTO_BUILD_MISSING_CACHE": bool(profile.get("auto_build_missing_cache", True)),
        "DEFAULT_PREP_SAMPLE_ROWS": profile.get("default_prep_sample_rows"),
        "MEASUREMENT_COLUMNS": measurement_columns,
        "OPTIONAL_QC_COLUMNS": optional_qc_columns,
        "PLOT_MEASUREMENT_COLUMN": plot_measurement_column,
        "PLOT_SECONDARY_COLUMN": plot_secondary_column,
        "ROW_USE_COLUMNS": list(dict.fromkeys(["Time UTC", "source_file", target_flag, *optional_qc_columns, *measurement_columns])),
        "WINDOW_FEATURE_COLUMNS": window_feature_columns,
        "WINDOW_USE_COLUMNS": ["window_start", "window_end", "source_file", "issue_rate", *window_feature_columns],
        **label_display_context(dataset_profile_id=dataset_profile_id, target_flag=target_flag),
    }


def dataset_profile_summary(config: dict[str, object]) -> dict[str, object]:
    """Return the profile-derived values worth printing in a notebook setup cell."""

    return {
        "DATASET_PROFILE_ID": config["DATASET_PROFILE_ID"],
        "DATASET_LABEL": config["DATASET_LABEL"],
        "READ_RAW_DATA_DIR": config["READ_RAW_DATA_DIR"],
        "READ_CACHE_DIR": config["READ_CACHE_DIR"],
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
