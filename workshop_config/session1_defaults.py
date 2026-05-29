"""Participant-editable Session 1 defaults.

These values are shared by notebooks and helper modules. Keep code in
``scripts/``; keep workshop defaults, labels, palettes, and cache-name choices
here so they are easy to find and change.
"""

from __future__ import annotations

# Default numeric CTD measurement columns used as model features when a dataset
# profile does not need a more specific feature list.
CTD_MEASUREMENT_COLUMNS = [
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

# Broader scalar feature preference list used by prep utilities when they scan
# raw ONC scalar CSV files and decide which measurement columns to keep.
PREFERRED_SCALAR_MEASUREMENT_COLUMNS = [
    *CTD_MEASUREMENT_COLUMNS,
    "Chlorophyll (ug/l)",
    "Fluorescence (mg/m3)",
    "Turbidity (NTU)",
    "Oxygen Concentration Corrected (ml/l)",
    "Oxygen Concentration Uncorrected (ml/l)",
    "Dissolved Oxygen (mL/L)",
    "Dissolved Oxygen (umol/L)",
    "Oxygen Saturation (%)",
]

# Default label ids treated as usable "good" examples for standard ONC QC flags.
DEFAULT_GOOD_LABELS = (1,)

# Default label ids treated as issues for standard ONC QC flags.
DEFAULT_ISSUE_LABELS = (3, 4, 9)

# Human-readable meanings for standard ONC QC flag values shown in plots,
# summaries, and reflection prompts.
QC_FLAG_MEANINGS = {
    0: "no QC",
    1: "good",
    2: "probably good",
    3: "probably bad",
    4: "bad",
    6: "bad down-sampling",
    7: "averaged",
    8: "interpolated",
    9: "missing / NaN",
}

# Human-readable meanings for the custom conductivity-plug ml_label values.
CONDUCTIVITY_PLUG_LABEL_MEANINGS = {
    0: "good",
    1: "conductivity bad plug",
    2: "conductivity bad other",
    3: "non-conductivity failure",
    4: "missing data",
}

# Shared colour palette for target-label plots. Keys 12 and 34 are collapsed
# labels used by some binary/merged sequence-modelling strategies.
DEFAULT_FLAG_PALETTE = {
    0: "#94a3b8",
    1: "#1f77b4",
    2: "#60a5fa",
    3: "#ff7f0e",
    4: "#d62728",
    6: "#8b5cf6",
    7: "#14b8a6",
    8: "#a855f7",
    9: "#7f7f7f",
    12: "#2563eb",
    34: "#e76f51",
}

# Generic cache stem used by older/default scalar Session 1 caches.
DEFAULT_CACHE_STEM = "scalar_session1"

# Cache stems from earlier notebook versions that should still be discoverable.
LEGACY_CACHE_STEMS = (DEFAULT_CACHE_STEM, "ctd_session1")

# Alternative cache stems checked when a requested dataset cache has been saved
# under an older or cluster-specific name.
CACHE_STEM_FALLBACKS = {
    "conductivity_scalar_session1": ("ctd_session1", DEFAULT_CACHE_STEM),
    "fluorometer_scalar_session1": ("sogcentral_turbidity", "folger_turbidity"),
    "sogcentral_turbidity": ("fluorometer_scalar_session1", "folger_turbidity"),
    "oxygen_scalar_session1": ("sogcentral_oxygen", "folger_oxygen"),
    "sogcentral_oxygen": ("oxygen_scalar_session1", "folger_oxygen"),
}

