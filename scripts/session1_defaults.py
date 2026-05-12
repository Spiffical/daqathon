"""Canonical Session 1 defaults shared by notebooks and helper modules.

Keeping these values in one module prevents the notebooks, resume helpers, and
prep/model utilities from drifting apart. Notebook cells can still override the
participant-facing values, but the starting defaults live here.
"""

from __future__ import annotations

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

DEFAULT_GOOD_LABELS = (1,)
DEFAULT_ISSUE_LABELS = (3, 4, 9)

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

CONDUCTIVITY_PLUG_LABEL_MEANINGS = {
    0: "good",
    1: "conductivity bad plug",
    2: "conductivity bad other",
    3: "non-conductivity failure",
    4: "missing data",
}

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

DEFAULT_CACHE_STEM = "scalar_session1"
LEGACY_CACHE_STEMS = (DEFAULT_CACHE_STEM, "ctd_session1")
CACHE_STEM_FALLBACKS = {
    "conductivity_scalar_session1": ("ctd_session1", DEFAULT_CACHE_STEM),
    "fluorometer_scalar_session1": ("sogcentral_turbidity", "folger_turbidity"),
    "sogcentral_turbidity": ("fluorometer_scalar_session1", "folger_turbidity"),
    "oxygen_scalar_session1": ("sogcentral_oxygen", "folger_oxygen"),
    "sogcentral_oxygen": ("oxygen_scalar_session1", "folger_oxygen"),
}

