# Scripts Folder Audit

This folder contains executable workflow code and notebook helper functions.
Participant-editable workshop values live in `../workshop_config/`.

## Active Notebook Support

- `session1_intro_notebook_setup.py`: imports and path setup shared by the notebooks.
- `session1_intro_utils.py`: plotting, cache inspection, split visualisation, and notebook display helpers.
- `session1_modeling.py`: train/test split helpers, feature construction, model training, evaluation, and plotting.
- `session1_profiles.py`: resolver functions that read `workshop_config/` values and produce notebook variables.
- `session1_resume_utils.py`: skip-ahead setup for ML sections.
- `session1_notebook_bootstrap.py`: small bootstrap helpers used near the top of notebooks.

## Cache Preparation

- `parquet_cache_utils.py`: generic CSV-to-row-parquet helpers for participant datasets.
- `onc_scalar_cache_utils.py`: reusable ONC scalar row/window cache helpers.
- `onc_scalar_cache_pipeline.py`: concise wrapper around the ONC row/window cache path.
- `prepare_scalar_session1_data.py`: original explicit ONC scalar cache builder and CLI.

## Standalone Experiments

- `forecast_transformer_experiments.py`: command-line experiment runner for the transformer next-value anomaly work. It is not imported by the notebooks, but it remains useful for batch/local experiments.

## Compatibility Shims

- `session1_defaults.py`: re-exports `workshop_config/session1_defaults.py` for older scripts. New code should import defaults from `workshop_config/`.

## Removal Notes

No script in this folder is currently safe to remove without either breaking a notebook path, removing a reusable cache-prep utility, or deleting a standalone experiment workflow. The main organisation improvement from the audit is that preset values and shared defaults have moved out of `scripts/` into `workshop_config/`.

