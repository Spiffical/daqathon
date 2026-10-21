"""Participant-editable dataset presets for the Session 1 notebooks.

This file contains values, not workflow code. Edit it when a prepared workshop
dataset needs a different target label, measurement-column list, cache name,
plot column, or label grouping. The resolver functions that turn these presets
into notebook variables live in ``scripts/session1_profiles.py``.
"""

from __future__ import annotations

from typing import Any

# Dataset profile property guide
#
# Each entry in WORKSHOP_DATASET_PROFILES describes one pre-loaded workshop
# dataset. Participant-owned data should use resolve_custom_dataset_config()
# from scripts/session1_profiles.py instead of pretending to be a workshop
# preset. The resolver functions turn these properties into explicit variables
# such as TARGET_FLAG and MEASUREMENT_COLUMNS.
#
# - label: short display name used in summaries.
# - description: one-sentence explanation of the dataset and modelling target.
# - raw_subpaths: candidate folders under data/raw that may contain source CSVs.
# - cache_stem: base name for the prepared parquet cache bundle.
# - target_flag: label column predicted by supervised models.
# - task_mode: target framing used by modelling helpers, usually "multiclass".
# - good_labels: label ids treated as good/normal examples for modelling.
# - issue_labels: label ids treated as issue/anomaly examples for modelling.
# - flag_example_classes: label ids to prefer when choosing example plots.
# - measurement_columns: numeric sensor columns used as ML features.
# - optional_qc_columns: extra QC columns loaded for context but not predicted.
# - plot_measurement_column: primary y-axis column for time-series examples.
# - plot_secondary_column: optional secondary y-axis column for context plots.
# - primary_device: device family used by raw-prep file grouping/merge logic.
# - kmeans_feature_mode: "window_summary" uses cached window stats; "row_level"
#   clusters sampled rows directly.
# - default_sequence_output_mode: "window" predicts one label per window;
#   "per_timestep" predicts one label at each timestamp inside a window.
# - default_sequence_target_strategy: default label-collapsing strategy for CNN
#   and Transformer examples.
# - sequence_indicator_channels: optional extra 0/1 input channels for the CNN and
#   Transformer, one per range check, e.g. {"column": "Turbidity (NTU)", "below": 0.0}
#   adds a channel that is 1 when turbidity is below zero. Use these when a QC flag
#   is a simple range check that standardised inputs would hide. Each entry needs a
#   "column" and either "below" or "above".
# - cache_read_sample_rows: optional row cap for cache-read demos on large data.
# - auto_build_missing_cache: whether the notebook may build a missing cache.
# - default_prep_sample_rows: optional per-file row cap when rebuilding caches.

WORKSHOP_DATASET_PROFILE_IDS = (
    "ctd_conductivity",
    "fluorometer_turbidity",
    "oxygen",
    "conductivity_plugs",
)

CUSTOM_DATASET_PROFILE_ID = "custom"

CUSTOM_DATASET_TEMPLATE: dict[str, Any] = {
    "label": "Custom scalar dataset",
    "description": "Template profile for participant-owned scalar CSV or parquet-cache data.",
    "raw_subpaths": ["custom"],
    "cache_stem": "custom_session1",
    "target_flag": "label",
    "task_mode": "multiclass",
    "good_labels": [0],
    "issue_labels": [1],
    "flag_example_classes": [1],
    "measurement_columns": ["sensor_value"],
    "optional_qc_columns": [],
    "plot_measurement_column": "sensor_value",
    "plot_secondary_column": None,
    "primary_device": "other",
    "kmeans_feature_mode": "row_level",
    "default_sequence_output_mode": "per_timestep",
    "default_sequence_target_strategy": "raw_multiclass",
    "sequence_indicator_channels": [],
    "cache_read_sample_rows": None,
    "auto_build_missing_cache": False,
    "default_prep_sample_rows": None,
    # Short "what to expect" notes printed by the ML notebook's score explanations.
    "teaching_notes": [],
}

WORKSHOP_DATASET_PROFILES: dict[str, dict[str, Any]] = {
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
        # Depth (m) is left out: it nearly duplicates Pressure (decibar), and the
        # March 2026 deployment file is missing it on 92% of rows. The CNN and
        # Transformer skip rows with missing values, so keeping it would cost
        # them most of the last ~10 days of the test period.
        "measurement_columns": [
            "Conductivity (S/m)",
            "Density (kg/m3)",
            "Practical Salinity (psu)",
            "Pressure (decibar)",
            "Sigma-t (kg/m3)",
            "Sigma-theta (0 dbar) (kg/m3)",
            "Sound Speed (m/s)",
            "Temperature (C)",
        ],
        "optional_qc_columns": ["Temperature QC Flag"],
        "plot_measurement_column": "Conductivity (S/m)",
        "plot_secondary_column": "Temperature (C)",
        "primary_device": "ctd",
        "kmeans_feature_mode": "window_summary",
        "default_sequence_output_mode": "window",
        "default_sequence_target_strategy": "collapsed_1_34_9",
        "sequence_indicator_channels": [],
        "cache_read_sample_rows": None,
        "auto_build_missing_cache": True,
        "default_prep_sample_rows": None,
        "teaching_notes": [
            "Flags 3 (probably bad) and 4 (bad) look alike in this dataset: mostly conductivity dropouts while temperature stays steady, plus single-point spikes. Flag 3 covers only six episodes in the year, so models rarely learn to tell it apart from flag 4.",
            "Flag 9 marks missing conductivity values, so it is easy to learn.",
        ],
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
        # Turbidity QC flag 4 marks readings below 0 NTU. Standardised inputs hide that
        # 0.02 NTU gap, so the CNN and Transformer get the range check as its own channel.
        "sequence_indicator_channels": [{"column": "Turbidity (NTU)", "below": 0.0}],
        "cache_read_sample_rows": None,
        "auto_build_missing_cache": True,
        "default_prep_sample_rows": None,
        "teaching_notes": [
            "Flag 4 is a range check: it marks turbidity readings below 0 NTU. The Random Forest can learn that threshold directly; the CNN and Transformer get an extra 0/1 input channel that marks readings below 0 NTU, because scaling the inputs hides such a small gap.",
        ],
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
        # Corrected oxygen is computed from salinity, and its QC flag 4 follows the CTD's
        # salinity flag 4 on almost every row. Practical Salinity and Conductivity give the
        # models the inputs behind that flag.
        "measurement_columns": [
            "Oxygen Concentration Corrected (ml/l)",
            "Oxygen Concentration Uncorrected (ml/l)",
            "Temperature (C)",
            "Pressure (decibar)",
            "Practical Salinity (psu)",
            "Conductivity (S/m)",
        ],
        "optional_qc_columns": ["Temperature QC Flag"],
        "plot_measurement_column": "Oxygen Concentration Corrected (ml/l)",
        "plot_secondary_column": "Temperature (C)",
        "primary_device": "oxygen",
        "kmeans_feature_mode": "window_summary",
        "default_sequence_output_mode": "window",
        "default_sequence_target_strategy": "collapsed_12_34_9",
        "sequence_indicator_channels": [],
        "cache_read_sample_rows": None,
        "auto_build_missing_cache": True,
        "default_prep_sample_rows": None,
        "teaching_notes": [
            "Corrected oxygen is computed from salinity, so whenever the CTD's salinity is flagged bad (4), corrected oxygen is flagged 4 too. That is why Practical Salinity and Conductivity are in this preset's measurement columns: without them, flag 4 is hard to learn.",
            "One flag-4 episode (July 2025) has normal-looking salinity, conductivity, and oxygen values, so nothing in these inputs separates it from good data. The episode_aware split puts it in the test set, where the Random Forest and the window-level CNN and Transformer score close to 0 on flag 4.",
        ],
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
        "sequence_indicator_channels": [],
        "cache_read_sample_rows": 250_000,
        "auto_build_missing_cache": False,
        "default_prep_sample_rows": 500_000,
        "teaching_notes": [
            "Label 1 (bad plug) shows up as long conductivity dropouts and spikes, and it is the most learnable issue.",
            "Label 3 (non-conductivity failure) marks failures in other instruments, often single points with no visible change in these inputs, so expect it to be missed. Label 2 is very rare, so its scores are unreliable.",
            "Label 4 marks missing data.",
        ],
    },
}

DATASET_PROFILES: dict[str, dict[str, Any]] = {
    **WORKSHOP_DATASET_PROFILES,
    CUSTOM_DATASET_PROFILE_ID: CUSTOM_DATASET_TEMPLATE,
}

