from __future__ import annotations

import builtins
import copy
import importlib
import json
import os
import pickle
import subprocess
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from IPython.display import display
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from . import prepare_scalar_session1_data as _prepare_scalar_session1_data
from . import session1_intro_utils as _session1_intro_utils
from . import session1_modeling as _session1_modeling

MODELING_EXPORTS = [
    "add_temporal_context_features",
    "add_tabular_baseline_features",
    "apply_target_strategy",
    "build_cache_bundle_paths",
    "build_labeled_intervals",
    "build_model_frame",
    "build_reviewed_target_frame",
    "build_sequence_split_bundle",
    "build_sequence_split_bundle_from_frames",
    "build_sequence_label_interval_data",
    "build_window_classification_interval_data",
    "clean_source_file_label",
    "compute_contiguous_split_target_distribution",
    "compute_interval_classification_metrics",
    "compute_split_share_gap",
    "DEFAULT_FLAG_PALETTE",
    "evaluate_classifier",
    "fit_extra_trees",
    "fit_kmeans",
    "fit_random_forest",
    "infer_interval_origin",
    "load_full_row_level_frame",
    "load_rows_for_time_range",
    "load_selected_row_level_frame",
    "materialize_reviewed_split_frames",
    "merge_adjacent_intervals",
    "plot_cluster_window_examples",
    "plot_flag_examples",
    "plot_time_series_with_bands",
    "predict_cnn_window_model",
    "predict_sequence_label_cnn",
    "predict_transformer_sequence_model",
    "predict_transformer_window_model",
    "resolve_cache_bundle_paths",
    "resolve_runtime_output_root",
    "report_average",
    "run_cnn_search",
    "run_cnn_search_from_frames",
    "run_rf_search",
    "sample_frame_by_strategy",
    "scan_interleaved_block_rows",
    "select_time_range",
    "split_frame_by_strategy",
    "stage_cache_into_runtime",
    "stage_directory_into_runtime",
    "summarize_split_distributions",
    "summarize_target_by_time_bin",
    "SUPPORTED_SPLIT_STRATEGIES",
]

INTRO_UTIL_EXPORTS = [
    "choose_cache_bundle_paths",
    "create_session1_row_level_parquet_cache",
    "create_session1_window_level_parquet_cache",
    "csv_files_to_parquet_cache",
    "directory_size_bytes",
    "derive_fractional_row_limit",
    "filter_csv_paths_with_required_columns",
    "load_raw_flag_context_sample",
    "load_raw_scalar_sample",
    "build_reviewed_modelling_split",
    "show_reviewed_modelling_split_build",
    "show_reviewed_model_row_accounting",
    "show_fixed_split_review",
    "show_cnn_interval_demo",
    "show_episode_aware_split_comparison",
    "summarize_issue_adequacy",
    "build_split_strategy_tables",
    "show_session1_cache_inspection",
    "read_parquet_head",
    "select_part_paths",
    "show_reviewed_split_summary",
    "show_session1_cache_read_comparison",
    "show_split_strategy_comparison",
    "show_split_strategy_timeline",
    "show_temporal_flag_summary",
]

PREP_EXPORTS = [
    "DEFAULT_MEASUREMENT_COLUMNS",
    "locate_header",
    "read_scalar_csv",
]


def _to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_to_jsonable(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    return value


def _build_notebook_print():
    builtin_print = builtins.print

    def pretty_json(value):
        builtin_print(json.dumps(_to_jsonable(value), indent=2, ensure_ascii=False))

    def notebook_print(*args, **kwargs):
        if len(args) == 1 and not kwargs and isinstance(args[0], Mapping):
            pretty_json(args[0])
            return
        builtin_print(*args, **kwargs)

    return notebook_print


def _select_exports(module, names: list[str]) -> dict[str, object]:
    return {name: getattr(module, name) for name in names}


def build_intro_notebook_namespace(
    *,
    notebook_root: str | Path,
    read_raw_data_dir: str | Path,
    read_cache_dir: str | Path,
    cache_bundle_name: str,
    seed: int,
) -> dict[str, object]:
    importlib.invalidate_caches()
    prepare_scalar_session1_data = importlib.reload(_prepare_scalar_session1_data)
    session1_intro_utils = importlib.reload(_session1_intro_utils)
    session1_modeling = importlib.reload(_session1_modeling)

    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    scratch = os.environ.get("SCRATCH")

    runtime_output_root = session1_modeling.resolve_runtime_output_root(
        notebook_root,
        slurm_tmpdir=slurm_tmpdir,
        scratch_dir=scratch,
    )
    runtime_raw_data_dir = runtime_output_root / "data" / "raw" / Path(read_raw_data_dir).name
    runtime_cache_dir = runtime_output_root / "cache" / "session1"
    artifact_dir = runtime_output_root / "artifacts"
    model_output_dir = runtime_output_root / "models"
    plot_output_dir = runtime_output_root / "plots"
    report_output_dir = runtime_output_root / "reports"

    for output_dir in [runtime_output_root, artifact_dir, model_output_dir, plot_output_dir, report_output_dir]:
        output_dir.mkdir(parents=True, exist_ok=True)

    use_runtime_raw_data_for_reads = bool(slurm_tmpdir)
    use_runtime_cache_for_reads = bool(slurm_tmpdir)
    raw_data_dir = str(runtime_raw_data_dir if use_runtime_raw_data_for_reads else Path(read_raw_data_dir))
    cache_dir = str(runtime_cache_dir if use_runtime_cache_for_reads else Path(read_cache_dir))
    cache_paths = session1_modeling.build_cache_bundle_paths(cache_dir, cache_bundle_name)

    rf_model_path = model_output_dir / "best_random_forest.pkl"
    cnn_model_path = model_output_dir / "best_cnn_checkpoint.pt"
    transformer_model_path = model_output_dir / "best_transformer_checkpoint.pt"

    notebook_print = _build_notebook_print()
    plt.style.use("seaborn-v0_8-whitegrid")
    pd.set_option("display.max_columns", 100)
    np.random.seed(seed)

    summary = {
        "READ_RAW_DATA_DIR": str(read_raw_data_dir),
        "READ_CACHE_DIR": str(read_cache_dir),
        "USE_RUNTIME_RAW_DATA_FOR_READS": use_runtime_raw_data_for_reads,
        "USE_RUNTIME_CACHE_FOR_READS": use_runtime_cache_for_reads,
        "RUNTIME_OUTPUT_ROOT": str(runtime_output_root),
        "RUNTIME_RAW_DATA_DIR": str(runtime_raw_data_dir),
        "CACHE_BUNDLE_NAME": cache_bundle_name,
        "ROW_CACHE_DIR": str(cache_paths.row_level_dir),
        "WINDOW_CACHE_PATH": str(cache_paths.window_cache_path),
    }

    namespace = {
        "copy": copy,
        "display": display,
        "json": json,
        "KMeans": KMeans,
        "Line2D": Line2D,
        "np": np,
        "os": os,
        "Path": Path,
        "pd": pd,
        "PercentFormatter": PercentFormatter,
        "perf_counter": perf_counter,
        "pickle": pickle,
        "Pipeline": Pipeline,
        "plt": plt,
        "pq": pq,
        "prepare_scalar_session1_data": prepare_scalar_session1_data,
        "print": notebook_print,
        "RandomForestClassifier": RandomForestClassifier,
        "ConfusionMatrixDisplay": ConfusionMatrixDisplay,
        "classification_report": classification_report,
        "f1_score": f1_score,
        "SimpleImputer": SimpleImputer,
        "StandardScaler": StandardScaler,
        "subprocess": subprocess,
        "session1_intro_utils": session1_intro_utils,
        "session1_modeling": session1_modeling,
        "SLURM_TMPDIR": slurm_tmpdir,
        "SCRATCH": scratch,
        "RUNTIME_OUTPUT_ROOT": runtime_output_root,
        "RUNTIME_RAW_DATA_DIR": runtime_raw_data_dir,
        "RUNTIME_CACHE_DIR": runtime_cache_dir,
        "ARTIFACT_DIR": artifact_dir,
        "MODEL_OUTPUT_DIR": model_output_dir,
        "PLOT_OUTPUT_DIR": plot_output_dir,
        "REPORT_OUTPUT_DIR": report_output_dir,
        "USE_RUNTIME_RAW_DATA_FOR_READS": use_runtime_raw_data_for_reads,
        "USE_RUNTIME_CACHE_FOR_READS": use_runtime_cache_for_reads,
        "RAW_DATA_DIR": raw_data_dir,
        "CACHE_DIR": cache_dir,
        "ROW_CACHE_DIR": str(cache_paths.row_level_dir),
        "WINDOW_CACHE_PATH": str(cache_paths.window_cache_path),
        "METADATA_PATH": str(cache_paths.metadata_path),
        "RF_MODEL_PATH": rf_model_path,
        "CNN_MODEL_PATH": cnn_model_path,
        "TRANSFORMER_MODEL_PATH": transformer_model_path,
        "compute_class_weight": compute_class_weight,
        "INTRO_NOTEBOOK_SETUP_SUMMARY": summary,
    }
    namespace.update(_select_exports(session1_modeling, MODELING_EXPORTS))
    namespace.update(_select_exports(session1_intro_utils, INTRO_UTIL_EXPORTS))
    namespace.update(_select_exports(prepare_scalar_session1_data, PREP_EXPORTS))
    return namespace
