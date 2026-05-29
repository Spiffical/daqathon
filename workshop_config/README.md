# Workshop Configuration

This folder contains the values participants are most likely to inspect or
change outside the notebooks.

- `session1_dataset_profiles.py` defines the prepared Session 1 dataset
  presets: raw-data folders, cache names, target-label columns, feature columns,
  plot columns, label groups, and model defaults.
- `session1_defaults.py` defines shared labels, palettes, cache fallback names,
  and default measurement-column lists.

The `scripts/` folder contains code that uses these values. If you only want to
change what a workshop dataset means, start here instead of editing helper
functions.

