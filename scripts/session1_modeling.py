"""Shared modeling helpers used by the DAQathon notebooks and study scripts."""

from __future__ import annotations

import copy
import itertools
import json
import math
import pickle
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - optional dependency in some environments
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

MEASUREMENT_COLUMNS = [
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

DEFAULT_CACHE_STEM = "scalar_session1"
LEGACY_CACHE_STEMS = (DEFAULT_CACHE_STEM, "ctd_session1")
CACHE_STEM_FALLBACKS = {
    "conductivity_scalar_session1": ("ctd_session1", DEFAULT_CACHE_STEM),
}

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

DEFAULT_GOOD_LABELS = (1,)
DEFAULT_ISSUE_LABELS = (3, 4, 9)
SUPPORTED_SPLIT_STRATEGIES = (
    "global_contiguous",
    "per_source_contiguous",
    "interleaved_blocks",
    "episode_aware",
)


def normalize_label_list(labels: list[int] | tuple[int, ...] | None, default: tuple[int, ...]) -> list[int]:
    """Normalize an optional label list into a stable integer list."""

    if labels is None:
        return list(default)
    return [int(label) for label in dict.fromkeys(labels)]


def issue_mask(values: pd.Series, issue_labels: list[int] | tuple[int, ...] | None = None) -> pd.Series:
    """Return a boolean mask for rows whose labels count as issues."""

    normalized_issue_labels = normalize_label_list(issue_labels, DEFAULT_ISSUE_LABELS)
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.isin(normalized_issue_labels)


def reviewed_label_mask(
    values: pd.Series,
    *,
    good_labels: list[int] | tuple[int, ...] | None = None,
    issue_labels: list[int] | tuple[int, ...] | None = None,
) -> pd.Series:
    """Return a boolean mask for rows whose labels have actually been reviewed.

    Some datasets include placeholder or "not yet reviewed" labels such as ``0``.
    Those rows are useful to count during dataset inspection, but they should not
    be treated as supervised training targets. This helper keeps the reviewed set
    explicit by combining the accepted ``good`` and ``issue`` labels.
    """

    normalized_good_labels = normalize_label_list(good_labels, DEFAULT_GOOD_LABELS)
    normalized_issue_labels = normalize_label_list(issue_labels, DEFAULT_ISSUE_LABELS)
    numeric = pd.to_numeric(values, errors="coerce")
    reviewed_labels = set(normalized_good_labels).union(normalized_issue_labels)
    return numeric.isin(reviewed_labels)


@dataclass(frozen=True)
class CacheBundlePaths:
    """Paths for one named cache bundle under a shared cache root."""

    root: Path
    stem: str
    row_level_dir: Path
    window_cache_path: Path
    metadata_path: Path


@dataclass
class SplitUnit:
    """One contiguous row slice that must stay together during split assignment."""

    source_key: object
    start: int
    end: int
    kind: str
    split: str | None = None

    @property
    def row_count(self) -> int:
        return max(int(self.end - self.start), 0)


def _normalize_cache_stem(cache_stem: str) -> str:
    """Normalize and validate a cache bundle stem."""

    normalized = cache_stem.strip()
    if not normalized:
        raise ValueError("cache stem must not be empty")
    if any(separator in normalized for separator in ("/", "\\")):
        raise ValueError("cache stem must be a simple name, not a path")
    return normalized


def build_cache_bundle_paths(
    cache_dir: str | Path,
    cache_stem: str = DEFAULT_CACHE_STEM,
) -> CacheBundlePaths:
    """Build row/window/metadata paths for one named cache bundle."""

    cache_root = Path(cache_dir).expanduser().resolve()
    stem = _normalize_cache_stem(cache_stem)
    return CacheBundlePaths(
        root=cache_root,
        stem=stem,
        row_level_dir=cache_root / f"{stem}_row_level",
        window_cache_path=cache_root / f"{stem}_windowed_features.parquet",
        metadata_path=cache_root / f"{stem}_metadata.json",
    )


def resolve_cache_bundle_paths(
    cache_dir: str | Path,
    cache_stem: str | None = None,
) -> CacheBundlePaths:
    """Resolve an existing bundle, with legacy CTD fallbacks when needed."""
    if cache_stem is None:
        stems_to_try = list(LEGACY_CACHE_STEMS)
    else:
        stems_to_try = [cache_stem, *CACHE_STEM_FALLBACKS.get(cache_stem, ())]
    for stem in stems_to_try:
        candidate = build_cache_bundle_paths(cache_dir, stem)
        if candidate.metadata_path.exists():
            return candidate
    return build_cache_bundle_paths(cache_dir, stems_to_try[0])


def resolve_runtime_output_root(
    notebook_root: str | Path,
    *,
    slurm_tmpdir: str | None = None,
    scratch_dir: str | None = None,
) -> Path:
    """Choose the writable runtime output directory for the current environment.

    Precedence:

    1. ``$SLURM_TMPDIR``
    2. ``$SCRATCH``
    3. repo-local ``tmp/session1_outputs``
    """
    if slurm_tmpdir:
        return Path(slurm_tmpdir).expanduser().resolve() / "daqathon" / "session1_outputs"
    if scratch_dir:
        return Path(scratch_dir).expanduser().resolve() / "daqathon" / "session1_outputs"
    return Path(notebook_root).expanduser().resolve() / "tmp" / "session1_outputs"


def _render_copy_progress(
    label: str,
    copied_files: int,
    total_files: int,
    copied_bytes: int,
    total_bytes: int,
) -> None:
    """Render a simple terminal-friendly progress bar for notebook staging steps."""
    total_files = max(total_files, 1)
    total_bytes = max(total_bytes, 1)
    fraction = copied_bytes / total_bytes
    bar_width = 24
    filled = min(bar_width, int(round(bar_width * fraction)))
    bar = "#" * filled + "-" * (bar_width - filled)
    copied_gb = copied_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    message = (
        f"\r{label}: [{bar}] {fraction:6.1%} | "
        f"{copied_files:>4}/{total_files:<4} files | "
        f"{copied_gb:6.2f}/{total_gb:6.2f} GB"
    )
    sys.stdout.write(message)
    sys.stdout.flush()


def stage_directory_into_runtime(
    source_dir: str | Path,
    runtime_dir: str | Path,
    *,
    force: bool = False,
    show_progress: bool = False,
    progress_label: str = "Staging files",
) -> dict[str, object]:
    """Copy a read-only source directory into a writable runtime directory.

    This is used for FIR notebook runs where shared project storage is the
    long-lived source of truth, but node-local job storage such as
    ``$SLURM_TMPDIR`` is much faster for interactive work.
    """
    source_dir = Path(source_dir).expanduser().resolve()
    runtime_dir = Path(runtime_dir).expanduser().resolve()

    if source_dir == runtime_dir:
        return {"staged": False, "reason": "runtime directory already points at the source directory"}
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    files_to_copy: list[tuple[Path, Path]] = []
    total_bytes = 0
    copied_files = 0
    copied_bytes = 0
    runtime_dir.mkdir(parents=True, exist_ok=True)

    for source_path in source_dir.rglob("*"):
        relative_path = source_path.relative_to(source_dir)
        destination_path = runtime_dir / relative_path
        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
            continue

        should_copy = force or not destination_path.exists()
        if not should_copy and source_path.stat().st_size != destination_path.stat().st_size:
            should_copy = True

        if should_copy:
            files_to_copy.append((source_path, destination_path))
            total_bytes += source_path.stat().st_size

    total_files = len(files_to_copy)
    if show_progress:
        _render_copy_progress(progress_label, 0, total_files, 0, total_bytes)

    for source_path, destination_path in files_to_copy:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        file_size = source_path.stat().st_size
        shutil.copy2(source_path, destination_path)
        copied_files += 1
        copied_bytes += file_size
        if show_progress:
            _render_copy_progress(progress_label, copied_files, total_files, copied_bytes, total_bytes)

    if show_progress:
        if total_files == 0:
            sys.stdout.write(f"{progress_label}: already staged; nothing to copy.\n")
        else:
            sys.stdout.write("\n")
        sys.stdout.flush()

    return {
        "staged": True,
        "source_dir": str(source_dir),
        "runtime_dir": str(runtime_dir),
        "copied_files": copied_files,
        "total_files": total_files,
        "copied_gb": round(copied_bytes / (1024 ** 3), 3),
        "total_gb": round(total_bytes / (1024 ** 3), 3),
    }


def stage_cache_into_runtime(
    persistent_cache_dir: str | Path,
    runtime_cache_dir: str | Path,
    *,
    force: bool = False,
    show_progress: bool = False,
    progress_label: str = "Staging cache",
) -> dict[str, object]:
    """Copy a read-only prepared cache into a writable runtime directory."""
    stage_result = stage_directory_into_runtime(
        source_dir=persistent_cache_dir,
        runtime_dir=runtime_cache_dir,
        force=force,
        show_progress=show_progress,
        progress_label=progress_label,
    )
    if stage_result.get("staged"):
        stage_result["runtime_cache_dir"] = str(Path(runtime_cache_dir).expanduser().resolve())
    return stage_result


def evenly_spaced_take(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """Take evenly spaced rows from a dataframe while preserving sort order."""
    if limit is None or len(frame) <= limit:
        return frame.reset_index(drop=True)
    indices = np.linspace(0, len(frame) - 1, num=limit, dtype=int)
    return frame.iloc[indices].reset_index(drop=True)


def select_part_paths(part_paths: list[Path], limit: int | None, mode: str = "spread") -> list[Path]:
    """Choose parquet parts either from the front or spread across the full time range."""
    if limit is None or limit >= len(part_paths):
        return part_paths
    if mode == "first":
        return part_paths[:limit]
    if mode == "spread":
        indices = np.linspace(0, len(part_paths) - 1, num=limit, dtype=int)
        selected = []
        seen = set()
        for index in indices:
            candidate = part_paths[int(index)]
            if candidate not in seen:
                selected.append(candidate)
                seen.add(candidate)
        return selected
    raise ValueError(f"Unsupported selection mode: {mode}")


def load_cache_bundle(
    cache_dir: str | Path,
    *,
    cache_stem: str | None = None,
    row_file_limit: int | None = None,
    part_selection_mode: str = "spread",
    rows_per_file: int = 45000,
    issue_rows_per_file: int = 12000,
    sample_strategy: str = "time_spread",
    window_limit: int | None = None,
    target_flag: str = "Conductivity QC Flag",
    good_labels: list[int] | tuple[int, ...] | None = None,
    issue_labels: list[int] | tuple[int, ...] | None = None,
    row_columns: list[str] | None = None,
    window_columns: list[str] | None = None,
) -> dict[str, object]:
    """Load metadata plus row- and window-level samples from the prepared cache."""
    bundle_paths = resolve_cache_bundle_paths(cache_dir, cache_stem=cache_stem)
    metadata_path = bundle_paths.metadata_path
    row_cache_dir = bundle_paths.row_level_dir
    window_cache_path = bundle_paths.window_cache_path

    metadata = json.loads(metadata_path.read_text())
    part_paths = sorted(row_cache_dir.glob("*.parquet"))
    if not part_paths:
        raise FileNotFoundError(f"No parquet parts found in {row_cache_dir}")

    selected_paths = select_part_paths(part_paths, limit=row_file_limit, mode=part_selection_mode)
    part_to_source = {
        Path(file_info["row_level_part"]).name: file_info["source_file"]
        for file_info in metadata["processed_files"]
    }
    selected_source_files = {part_to_source[path.name] for path in selected_paths}

    # Sample rows per parquet part so the notebooks stay responsive while still
    # covering the full time span of the deployment.
    row_df = load_row_level_sample(
        selected_paths,
        rows_per_file=rows_per_file,
        issue_rows_per_file=issue_rows_per_file,
        sample_strategy=sample_strategy,
        target_flag=target_flag,
        good_labels=good_labels,
        issue_labels=issue_labels,
        columns=row_columns,
    )

    window_df = pd.read_parquet(window_cache_path, columns=window_columns)
    window_df["window_start"] = pd.to_datetime(window_df["window_start"], utc=True)
    window_df["window_end"] = pd.to_datetime(window_df["window_end"], utc=True)
    window_df = window_df[window_df["source_file"].isin(selected_source_files)].sort_values("window_start")
    window_df = evenly_spaced_take(window_df, window_limit)

    return {
        "metadata": metadata,
        "selected_paths": selected_paths,
        "selected_source_files": selected_source_files,
        "row_df": row_df,
        "window_df": window_df.reset_index(drop=True),
    }


def load_row_level_sample(
    part_paths: list[Path],
    *,
    rows_per_file: int | None,
    issue_rows_per_file: int,
    sample_strategy: str = "time_spread",
    target_flag: str,
    good_labels: list[int] | tuple[int, ...] | None = None,
    issue_labels: list[int] | tuple[int, ...] | None = None,
    balanced_issue_share: float = 0.5,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load a row-level sample from each parquet part using one sampling strategy.

    Supported strategies:

    - ``time_spread``: keep an evenly spaced slice from the full parquet part.
    - ``issue_focused``: start from a time-spread slice, then add extra reviewed
      issue rows before trimming back to ``rows_per_file``.
    - ``balanced_reviewed``: move the reviewed sample toward a configurable
      issue share by drawing from the reviewed good and reviewed issue groups
      separately. ``balanced_issue_share`` controls the target reviewed issue
      fraction for that mode.
    """
    row_frames = []
    for path in part_paths:
        frame = pd.read_parquet(path, columns=columns).sort_values("Time UTC").reset_index(drop=True)
        frame["Time UTC"] = pd.to_datetime(frame["Time UTC"], utc=True)
        if rows_per_file is None:
            sampled_frame = frame
        else:
            sampled_frame = sample_frame_by_strategy(
                frame,
                rows_limit=rows_per_file,
                sample_strategy=sample_strategy,
                target_flag=target_flag,
                good_labels=good_labels,
                issue_labels=issue_labels,
                issue_rows=issue_rows_per_file,
                balanced_issue_share=balanced_issue_share,
            )
        row_frames.append(sampled_frame)
    return pd.concat(row_frames, ignore_index=True).sort_values("Time UTC").reset_index(drop=True)


def sample_frame_by_strategy(
    frame: pd.DataFrame,
    *,
    rows_limit: int | None,
    sample_strategy: str,
    target_flag: str,
    good_labels: list[int] | tuple[int, ...] | None = None,
    issue_labels: list[int] | tuple[int, ...] | None = None,
    issue_rows: int = 0,
    balanced_issue_share: float = 0.5,
) -> pd.DataFrame:
    """Sample one dataframe according to a notebook sampling strategy.

    This is used in two places:

    - when creating a lightweight inspection sample from the cache, and
    - when shrinking only the training split in the stricter benchmark flow.

    ``balanced_issue_share`` is only used by ``balanced_reviewed``. It sets the
    target fraction of reviewed rows that should come from issue labels. A value
    of ``0.5`` asks for a 50/50 reviewed good-vs-issue mix, while smaller
    values such as ``0.2`` or ``0.25`` give issue rows a gentler boost.
    """

    work = frame.sort_values("Time UTC").reset_index(drop=True).copy()
    if rows_limit is None or len(work) <= rows_limit:
        return work

    normalized_good_labels = normalize_label_list(good_labels, DEFAULT_GOOD_LABELS)
    normalized_issue_labels = normalize_label_list(issue_labels, DEFAULT_ISSUE_LABELS)

    if sample_strategy == "time_spread":
        return evenly_spaced_take(work, rows_limit)

    if sample_strategy == "issue_focused":
        base_limit = max(rows_limit - issue_rows, 0)
        sampled_frame = evenly_spaced_take(work, base_limit)
        if issue_rows > 0 and target_flag in work.columns:
            issue_frame = work[issue_mask(work[target_flag], normalized_issue_labels)].reset_index(drop=True)
            issue_sample = evenly_spaced_take(issue_frame, issue_rows)
            sampled_frame = pd.concat([sampled_frame, issue_sample], ignore_index=True)
            sampled_frame = sampled_frame.drop_duplicates(subset=["Time UTC"]).sort_values("Time UTC")
        return evenly_spaced_take(sampled_frame, rows_limit)

    if sample_strategy == "balanced_reviewed":
        if not 0 < balanced_issue_share < 1:
            raise ValueError("balanced_issue_share must be in the open interval (0, 1)")
        reviewed_frame = work[
            reviewed_label_mask(
                work[target_flag],
                good_labels=normalized_good_labels,
                issue_labels=normalized_issue_labels,
            )
        ].reset_index(drop=True)
        if reviewed_frame.empty:
            return evenly_spaced_take(work, rows_limit)

        good_frame = reviewed_frame[reviewed_frame[target_flag].astype(int).isin(normalized_good_labels)].reset_index(drop=True)
        issue_frame = reviewed_frame[reviewed_frame[target_flag].astype(int).isin(normalized_issue_labels)].reset_index(drop=True)

        def _sample_group(group_frame: pd.DataFrame, group_labels: list[int], target_rows: int) -> pd.DataFrame:
            if group_frame.empty or target_rows <= 0:
                return group_frame.iloc[0:0].copy()
            present_labels = [label for label in group_labels if label in set(group_frame[target_flag].astype(int).unique().tolist())]
            if not present_labels:
                return evenly_spaced_take(group_frame, target_rows)
            per_label_limit = max(int(np.ceil(target_rows / len(present_labels))), 1)
            label_samples = []
            for label in present_labels:
                label_frame = group_frame[group_frame[target_flag].astype(int) == label].reset_index(drop=True)
                label_samples.append(evenly_spaced_take(label_frame, per_label_limit))
            group_sample = pd.concat(label_samples, ignore_index=True)
            group_sample = group_sample.drop_duplicates(subset=["Time UTC"]).sort_values("Time UTC")
            if len(group_sample) < target_rows:
                group_fill = evenly_spaced_take(group_frame, target_rows)
                group_sample = pd.concat([group_sample, group_fill], ignore_index=True)
                group_sample = group_sample.drop_duplicates(subset=["Time UTC"]).sort_values("Time UTC")
            return evenly_spaced_take(group_sample, target_rows)

        issue_target = int(round(rows_limit * balanced_issue_share))
        issue_target = min(max(issue_target, 0), rows_limit)
        good_target = rows_limit - issue_target

        good_sample = _sample_group(good_frame, normalized_good_labels, good_target)
        issue_sample = _sample_group(issue_frame, normalized_issue_labels, issue_target)

        sampled_frame = pd.concat([good_sample, issue_sample], ignore_index=True)
        sampled_frame = sampled_frame.drop_duplicates(subset=["Time UTC"]).sort_values("Time UTC")
        if len(sampled_frame) < rows_limit:
            reviewed_fill = evenly_spaced_take(reviewed_frame, rows_limit)
            sampled_frame = pd.concat([sampled_frame, reviewed_fill], ignore_index=True)
            sampled_frame = sampled_frame.drop_duplicates(subset=["Time UTC"]).sort_values("Time UTC")
        return evenly_spaced_take(sampled_frame, rows_limit)

    raise ValueError(f"Unsupported sample_strategy: {sample_strategy}")


def load_full_row_level_frame(
    part_paths: list[Path],
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load and concatenate the full row-level parquet data for selected parts."""

    frames = []
    for path in part_paths:
        frame = pd.read_parquet(path, columns=columns).sort_values("Time UTC").reset_index(drop=True)
        frame["Time UTC"] = pd.to_datetime(frame["Time UTC"], utc=True)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=columns or [])
    return pd.concat(frames, ignore_index=True).sort_values("Time UTC").reset_index(drop=True)


def load_selected_row_level_frame(
    part_paths: list[Path],
    selection_frame: pd.DataFrame,
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load only the row-level parquet rows that match a selected split frame.

    The selection frame is expected to contain ``source_file`` and ``Time UTC``.
    This helper keeps memory use lower than concatenating the full parquet cache
    when the notebook only needs the rows that survived the chosen split and
    train-subset settings.
    """

    required_columns = ["source_file", "Time UTC"]
    requested_columns = list(dict.fromkeys(required_columns + list(columns or [])))
    if selection_frame.empty:
        return pd.DataFrame(columns=requested_columns)

    if "source_file" not in selection_frame.columns or "Time UTC" not in selection_frame.columns:
        raise ValueError("selection_frame must contain 'source_file' and 'Time UTC'.")

    selection_keys = selection_frame[required_columns].drop_duplicates().copy()
    selection_keys["source_file"] = selection_keys["source_file"].astype(str)
    selection_keys["Time UTC"] = pd.to_datetime(selection_keys["Time UTC"], utc=True)
    selection_by_source = {
        str(source_file): group["Time UTC"].astype("int64").to_numpy(copy=False)
        for source_file, group in selection_keys.groupby("source_file", sort=False)
    }

    frames: list[pd.DataFrame] = []
    for path in part_paths:
        frame = pd.read_parquet(path, columns=requested_columns)
        if frame.empty:
            continue
        frame["source_file"] = frame["source_file"].astype(str)
        frame["Time UTC"] = pd.to_datetime(frame["Time UTC"], utc=True)

        part_source_files = frame["source_file"].dropna().unique().tolist()
        relevant_sources = [source_file for source_file in part_source_files if source_file in selection_by_source]
        if not relevant_sources:
            continue

        part_time_values = frame["Time UTC"].astype("int64").to_numpy(copy=False)
        part_source_values = frame["source_file"].to_numpy(copy=False)
        match_mask = np.zeros(len(frame), dtype=bool)

        for source_file in relevant_sources:
            source_mask = part_source_values == source_file
            if not source_mask.any():
                continue
            match_mask[source_mask] = np.isin(
                part_time_values[source_mask],
                selection_by_source[source_file],
                assume_unique=False,
            )

        matched = frame.loc[match_mask].copy()
        if not matched.empty:
            frames.append(matched)

    if not frames:
        return pd.DataFrame(columns=requested_columns)
    return pd.concat(frames, ignore_index=True).sort_values("Time UTC").reset_index(drop=True)


def build_reviewed_target_frame(
    df: pd.DataFrame,
    *,
    target_flag: str,
    task_mode: str,
    good_labels: list[int] | tuple[int, ...] | None = None,
    issue_labels: list[int] | tuple[int, ...] | None = None,
    model_row_limit: int | None = None,
) -> tuple[pd.DataFrame, list[int]]:
    """Filter to reviewed rows and add ``issue`` / ``model_target`` columns.

    This lighter-weight helper is useful before split selection, when the
    notebook only needs label semantics and timestamps rather than the full
    tabular feature set.
    """

    work = df.copy()
    work = work[
        reviewed_label_mask(
            work[target_flag],
            good_labels=good_labels,
            issue_labels=issue_labels,
        )
    ].copy()
    work = work.dropna(subset=[target_flag]).reset_index(drop=True)
    work["issue"] = issue_mask(work[target_flag], issue_labels).astype(int)

    if model_row_limit is not None and len(work) > model_row_limit:
        work = evenly_spaced_take(work, model_row_limit)

    if task_mode == "multiclass":
        work["model_target"] = work[target_flag].astype(int)
    elif task_mode == "binary":
        work["model_target"] = work["issue"].astype(int)
    else:
        raise ValueError(f"Unsupported task mode: {task_mode}")

    active_labels = sorted(work["model_target"].dropna().astype(int).unique().tolist())
    return work, active_labels


def add_tabular_baseline_features(
    frame: pd.DataFrame,
    *,
    measurement_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Add the baseline RF-style tabular features to a reviewed frame."""

    work = frame.copy()
    work["hour_utc"] = work["Time UTC"].dt.hour
    work["minute_utc"] = work["Time UTC"].dt.minute
    work["day_of_year"] = work["Time UTC"].dt.dayofyear

    for column in measurement_columns:
        work[f"{column} abs_delta"] = work[column].diff().abs().fillna(0.0)

    feature_columns = measurement_columns + [f"{column} abs_delta" for column in measurement_columns] + [
        "hour_utc",
        "minute_utc",
        "day_of_year",
    ]
    return work, feature_columns


def materialize_reviewed_split_frames(
    part_paths: list[Path],
    split_frames: dict[str, pd.DataFrame],
    *,
    columns: list[str],
    target_flag: str,
    task_mode: str,
    good_labels: list[int] | tuple[int, ...] | None = None,
    issue_labels: list[int] | tuple[int, ...] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load raw rows for selected splits, then rebuild reviewed label frames."""

    materialized: dict[str, pd.DataFrame] = {}
    for split_name, selection_frame in split_frames.items():
        raw_frame = load_selected_row_level_frame(part_paths, selection_frame, columns=columns)
        reviewed_frame, _ = build_reviewed_target_frame(
            raw_frame,
            target_flag=target_flag,
            task_mode=task_mode,
            good_labels=good_labels,
            issue_labels=issue_labels,
            model_row_limit=None,
        )
        materialized[split_name] = reviewed_frame
    return materialized


def build_distribution_frame(metadata: dict[str, object], df: pd.DataFrame, target_flag: str) -> pd.DataFrame:
    """Compare the full target distribution with the currently loaded sample."""
    full_target_counts = pd.Series(
        {int(key): int(value) for key, value in metadata["target_distribution"].items()}
    ).sort_index()
    sample_target_counts = df[target_flag].dropna().astype(int).value_counts().sort_index()
    return pd.DataFrame({"full_cache": full_target_counts, "loaded_sample": sample_target_counts}).fillna(0).astype(int)


def summarize_split_distributions(
    split_frames: dict[str, pd.DataFrame],
    *,
    label_column: str = "model_target",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-split label counts and normalized shares for a split dictionary."""

    count_frame = pd.DataFrame(
        {
            split_name: frame[label_column].value_counts().sort_index()
            for split_name, frame in split_frames.items()
        }
    ).fillna(0).astype(int)
    share_frame = count_frame.div(count_frame.sum(axis=0), axis=1).fillna(0.0)
    return count_frame, share_frame


def summarize_target_by_time_bin(
    frame: pd.DataFrame,
    *,
    time_column: str,
    label_column: str,
    bin_count: int = 24,
    labels: list[int] | None = None,
    good_labels: list[int] | tuple[int, ...] | None = None,
    issue_labels: list[int] | tuple[int, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize how target labels are distributed across evenly spaced time bins.

    This is meant as an early dataset-verification helper for the notebooks.
    It answers a simple but important question before any model is trained:
    do the interesting labels appear throughout time, or are they concentrated
    in just a few regions?

    Returns three aligned dataframes:

    - ``count_frame``: rows are time bins, columns are labels, values are counts
    - ``share_frame``: same shape, normalized within each time bin
    - ``summary_frame``: one row per bin with start/end timestamps, total rows,
      issue rows, and issue share percentage
    """

    work = frame[[time_column, label_column]].dropna().copy()
    if work.empty:
        empty_counts = pd.DataFrame(columns=labels or [])
        empty_summary = pd.DataFrame(
            columns=[
                "bin_index",
                "time_start",
                "time_end",
                "rows",
                "reviewed_rows",
                "unreviewed_rows",
                "reviewed_share_pct",
                "issue_rows",
                "issue_share_pct",
            ]
        )
        return empty_counts, empty_counts.copy(), empty_summary

    work[time_column] = pd.to_datetime(work[time_column], utc=True)
    work[label_column] = work[label_column].astype(int)

    if labels is None:
        labels = sorted(work[label_column].unique().tolist())

    issue_label_set = set(normalize_label_list(issue_labels, DEFAULT_ISSUE_LABELS))
    if not issue_label_set:
        issue_label_set = set(DEFAULT_ISSUE_LABELS)
    reviewed_mask = reviewed_label_mask(
        work[label_column],
        good_labels=good_labels,
        issue_labels=issue_labels,
    )

    time_values = work[time_column].astype("int64").to_numpy()
    requested_bin_count = max(int(bin_count), 1)
    effective_bin_count = min(requested_bin_count, max(len(work), 1))
    raw_edges = np.linspace(time_values.min(), time_values.max(), effective_bin_count + 1, dtype=np.int64)
    edges = np.unique(raw_edges)
    if len(edges) < 2:
        edges = np.array([time_values.min(), time_values.min() + 1], dtype=np.int64)

    work["_time_bin"] = pd.cut(
        time_values,
        bins=edges,
        include_lowest=True,
        labels=False,
        duplicates="drop",
    )
    work = work.dropna(subset=["_time_bin"]).copy()
    work["_time_bin"] = work["_time_bin"].astype(int)
    work["_is_issue"] = work[label_column].isin(issue_label_set).astype(int)
    work["_is_reviewed"] = reviewed_mask.loc[work.index].astype(int)

    bin_index = pd.Index(sorted(work["_time_bin"].unique().tolist()), name="time_bin")
    count_frame = (
        work.groupby(["_time_bin", label_column]).size().unstack(fill_value=0).reindex(index=bin_index, columns=labels, fill_value=0)
    )
    share_frame = count_frame.div(count_frame.sum(axis=1), axis=0).fillna(0.0)

    issue_rows = work.groupby("_time_bin")["_is_issue"].sum().reindex(bin_index, fill_value=0)
    total_rows = work.groupby("_time_bin").size().reindex(bin_index, fill_value=0)
    reviewed_rows = work.groupby("_time_bin")["_is_reviewed"].sum().reindex(bin_index, fill_value=0)

    summary_rows = []
    for time_bin in bin_index:
        start_index = int(time_bin)
        end_index = min(start_index + 1, len(edges) - 1)
        summary_rows.append(
            {
                "bin_index": int(time_bin),
                "time_start": pd.to_datetime(int(edges[start_index]), utc=True),
                "time_end": pd.to_datetime(int(edges[end_index]), utc=True),
                "rows": int(total_rows.loc[time_bin]),
                "reviewed_rows": int(reviewed_rows.loc[time_bin]),
                "unreviewed_rows": int(total_rows.loc[time_bin] - reviewed_rows.loc[time_bin]),
                "issue_rows": int(issue_rows.loc[time_bin]),
                "issue_share_pct": round(
                    100.0 * float(issue_rows.loc[time_bin]) / float(reviewed_rows.loc[time_bin])
                    if int(reviewed_rows.loc[time_bin]) > 0
                    else 0.0,
                    2,
                ),
                "reviewed_share_pct": round(
                    100.0 * float(reviewed_rows.loc[time_bin]) / float(total_rows.loc[time_bin])
                    if int(total_rows.loc[time_bin]) > 0
                    else 0.0,
                    2,
                ),
            }
        )
    summary_frame = pd.DataFrame(summary_rows)
    return count_frame, share_frame, summary_frame


def compute_split_share_gap(share_frame: pd.DataFrame) -> dict[str, float]:
    """Summarize how different the split label compositions are from one another.

    The metric is the total variation distance between two split-share vectors,
    averaged and maximized over every split pair. A value of ``0`` means the
    splits have identical composition; larger values mean stronger drift.
    """

    split_names = list(share_frame.columns)
    pairwise_gaps: list[float] = []
    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1 :]:
            left = share_frame[left_name].reindex(share_frame.index, fill_value=0.0)
            right = share_frame[right_name].reindex(share_frame.index, fill_value=0.0)
            pairwise_gaps.append(float(np.abs(left - right).sum() / 2.0))

    if not pairwise_gaps:
        return {"max_pairwise_total_variation": 0.0, "mean_pairwise_total_variation": 0.0}
    return {
        "max_pairwise_total_variation": float(max(pairwise_gaps)),
        "mean_pairwise_total_variation": float(np.mean(pairwise_gaps)),
    }


def _split_slot_name(
    block_index: int,
    *,
    train_fraction: float,
    validation_fraction: float,
    cycle_length: int = 20,
) -> str:
    """Map one repeating block index onto an interleaved train/validation/test slot."""

    if cycle_length < 3:
        raise ValueError("cycle_length must be at least 3")

    train_slots = min(max(int(round(cycle_length * train_fraction)), 1), cycle_length - 2)
    validation_slots = min(max(int(round(cycle_length * validation_fraction)), 1), cycle_length - train_slots - 1)
    test_slots = max(cycle_length - train_slots - validation_slots, 1)

    target_counts = {
        "train": train_slots,
        "validation": validation_slots,
        "test": test_slots,
    }
    built_counts = {split_name: 0 for split_name in target_counts}
    schedule: list[str] = []

    for slot_index in range(cycle_length):
        horizon = slot_index + 1

        def _slot_score(split_name: str) -> tuple[float, int, int]:
            target = target_counts[split_name]
            deficit = (target * horizon / cycle_length) - built_counts[split_name]
            return (deficit, target, -built_counts[split_name])

        chosen = max(target_counts, key=_slot_score)
        schedule.append(chosen)
        built_counts[chosen] += 1

    return schedule[int(block_index % cycle_length)]


def _find_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous ``True`` runs as half-open ``(start, end)`` row slices."""

    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index, is_true in enumerate(mask.tolist()):
        if is_true and run_start is None:
            run_start = index
        elif not is_true and run_start is not None:
            runs.append((run_start, index))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(mask)))
    return runs


def _merge_intervals(intervals: list[tuple[int, int]], gap_rows: int = 0) -> list[tuple[int, int]]:
    """Merge half-open intervals that overlap or sit within ``gap_rows`` rows."""

    if not intervals:
        return []

    merged: list[tuple[int, int]] = []
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end + gap_rows:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def _build_episode_intervals(
    issue_mask_values: np.ndarray,
    *,
    context_rows: int,
    merge_gap_rows: int,
) -> list[tuple[int, int]]:
    """Build contiguous episode intervals from issue rows plus local clean context."""

    raw_runs = _find_true_runs(issue_mask_values.astype(bool))
    if not raw_runs:
        return []

    merged_runs = _merge_intervals(raw_runs, gap_rows=max(int(merge_gap_rows), 0))
    expanded_runs = [
        (
            max(0, start - max(int(context_rows), 0)),
            min(len(issue_mask_values), end + max(int(context_rows), 0)),
        )
        for start, end in merged_runs
    ]
    return _merge_intervals(expanded_runs, gap_rows=0)


def _build_clean_units(
    source_key: object,
    start: int,
    end: int,
    *,
    block_rows: int,
) -> list[SplitUnit]:
    """Chunk one clean interval into fixed-size split units."""

    units: list[SplitUnit] = []
    for block_start in range(start, end, block_rows):
        block_end = min(block_start + block_rows, end)
        if block_end > block_start:
            units.append(SplitUnit(source_key=source_key, start=block_start, end=block_end, kind="clean"))
    return units


def _build_episode_aware_units(
    source_frame: pd.DataFrame,
    *,
    source_key: object,
    issue_mask_values: np.ndarray,
    block_rows: int,
    episode_context_rows: int,
    episode_merge_gap_rows: int,
) -> list[SplitUnit]:
    """Partition one source into episode units plus clean background blocks."""

    episode_intervals = _build_episode_intervals(
        issue_mask_values,
        context_rows=episode_context_rows,
        merge_gap_rows=episode_merge_gap_rows,
    )
    if not episode_intervals:
        return _build_clean_units(source_key, 0, len(source_frame), block_rows=block_rows)

    units: list[SplitUnit] = []
    cursor = 0
    for start, end in episode_intervals:
        if cursor < start:
            units.extend(_build_clean_units(source_key, cursor, start, block_rows=block_rows))
        units.append(SplitUnit(source_key=source_key, start=start, end=end, kind="episode"))
        cursor = end
    if cursor < len(source_frame):
        units.extend(_build_clean_units(source_key, cursor, len(source_frame), block_rows=block_rows))
    return units


def _issue_mask_from_frame(
    frame: pd.DataFrame,
    *,
    issue_column: str,
    target_column: str | None,
    issue_labels: list[int] | tuple[int, ...] | None,
) -> np.ndarray:
    """Extract a boolean issue mask from either an explicit issue column or raw labels."""

    if issue_column in frame.columns:
        numeric = pd.to_numeric(frame[issue_column], errors="coerce").fillna(0).astype(int)
        return numeric.eq(1).to_numpy()
    if target_column is not None and target_column in frame.columns:
        return issue_mask(frame[target_column], issue_labels).to_numpy()
    raise ValueError(
        "episode_aware split requires either an issue column or a target column plus issue labels."
    )


def _assign_episode_aware_units(
    units: list[SplitUnit],
    *,
    train_fraction: float,
    validation_fraction: float,
) -> None:
    """Assign episode units first, then fill remaining clean units toward row targets."""

    if not units:
        return

    split_names = ("train", "validation", "test")
    total_rows = float(sum(unit.row_count for unit in units))
    target_rows = {
        "train": total_rows * float(train_fraction),
        "validation": total_rows * float(validation_fraction),
        "test": max(total_rows * float(1.0 - train_fraction - validation_fraction), 0.0),
    }
    current_rows = {split_name: 0.0 for split_name in split_names}

    episode_units = [unit for unit in units if unit.kind == "episode"]
    episode_cycle_length = max(3, min(len(episode_units), 20))
    episode_index = 0
    for unit in episode_units:
        unit.split = _split_slot_name(
            episode_index,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            cycle_length=episode_cycle_length,
        )
        current_rows[unit.split] += float(unit.row_count)
        episode_index += 1

    for unit in units:
        if unit.split is not None:
            continue

        def _deficit_score(split_name: str) -> tuple[float, float, float]:
            target = max(target_rows[split_name], 1.0)
            deficit = target_rows[split_name] - current_rows[split_name]
            return (deficit / target, deficit, -current_rows[split_name])

        unit.split = max(split_names, key=_deficit_score)
        current_rows[unit.split] += float(unit.row_count)


def _append_trimmed_unit_frames(
    split_frames: dict[str, list[pd.DataFrame]],
    source_frame: pd.DataFrame,
    source_units: list[SplitUnit],
    *,
    purge_gap_rows: int,
) -> None:
    """Append source slices to their assigned split, trimming purge gaps at boundaries."""

    if not source_units:
        return

    left_trim = max(int(purge_gap_rows), 0) // 2
    right_trim = max(int(purge_gap_rows), 0) - left_trim

    for index, unit in enumerate(source_units):
        if unit.split is None:
            continue

        start = int(unit.start)
        end = int(unit.end)

        if purge_gap_rows > 0:
            if index > 0 and source_units[index - 1].split != unit.split:
                start += left_trim
            if index < len(source_units) - 1 and source_units[index + 1].split != unit.split:
                end -= right_trim

        start = max(start, int(unit.start))
        end = min(end, int(unit.end))
        if end <= start:
            continue

        split_frames[unit.split].append(source_frame.iloc[start:end].copy())


def _finalize_split_frames(
    split_frames: dict[str, list[pd.DataFrame]],
    *,
    template_frame: pd.DataFrame,
    time_column: str,
) -> dict[str, pd.DataFrame]:
    """Concatenate per-split frame lists and keep a stable split dictionary."""

    empty_template = template_frame.iloc[0:0].copy()
    finalized: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "validation", "test"):
        parts = split_frames.get(split_name, [])
        if parts:
            split_frame = pd.concat(parts, ignore_index=True)
            if time_column in split_frame.columns:
                split_frame = split_frame.sort_values(time_column)
            finalized[split_name] = split_frame.reset_index(drop=True)
        else:
            finalized[split_name] = empty_template.copy()
    return finalized


def split_frame_by_strategy(
    frame: pd.DataFrame,
    *,
    train_fraction: float,
    validation_fraction: float,
    strategy: str = "global_contiguous",
    time_column: str = "Time UTC",
    source_column: str = "source_file",
    block_rows: int = 1024,
    block_cycle_length: int = 20,
    issue_column: str = "issue",
    target_column: str | None = None,
    issue_labels: list[int] | tuple[int, ...] | None = None,
    episode_context_rows: int = 0,
    episode_merge_gap_rows: int = 0,
    purge_gap_rows: int = 0,
) -> dict[str, pd.DataFrame]:
    """Split a time-ordered dataframe with one of the notebook split strategies.

    Supported strategies:

    - ``global_contiguous``: one early/middle/late cut across the entire frame.
    - ``per_source_contiguous``: make an early/middle/late cut inside each source
      file, then concatenate the per-file train/validation/test slices.
    - ``interleaved_blocks``: keep short local time blocks intact, but distribute
      them across train/validation/test in a repeating schedule so each split sees
      more of the deployment's operating regimes.
    - ``episode_aware``: keep detected issue episodes, plus local clean context,
      entirely inside one split and optionally leave purge gaps between splits.
    """

    if strategy not in SUPPORTED_SPLIT_STRATEGIES:
        raise ValueError(f"Unsupported split strategy: {strategy}")
    if block_rows <= 0:
        raise ValueError("block_rows must be positive")

    if time_column in frame.columns:
        ordered = frame.sort_values(time_column).reset_index(drop=True).copy()
    else:
        ordered = frame.reset_index(drop=True).copy()

    if strategy == "global_contiguous" or source_column not in ordered.columns:
        train_frame, valid_frame, test_frame = contiguous_split(
            ordered,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
        return {"train": train_frame, "validation": valid_frame, "test": test_frame}

    split_frames: dict[str, list[pd.DataFrame]] = {"train": [], "validation": [], "test": []}

    if strategy == "per_source_contiguous":
        for _, source_frame in ordered.groupby(source_column, sort=False, observed=False):
            local_train, local_valid, local_test = contiguous_split(
                source_frame.reset_index(drop=True),
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
            )
            split_frames["train"].append(local_train)
            split_frames["validation"].append(local_valid)
            split_frames["test"].append(local_test)
        return _finalize_split_frames(split_frames, template_frame=ordered, time_column=time_column)

    if strategy == "episode_aware":
        source_units_by_key: dict[object, list[SplitUnit]] = {}
        ordered_units: list[SplitUnit] = []

        for source_key, source_frame in ordered.groupby(source_column, sort=False, observed=False):
            source_frame = source_frame.reset_index(drop=True)
            local_issue_mask = _issue_mask_from_frame(
                source_frame,
                issue_column=issue_column,
                target_column=target_column,
                issue_labels=issue_labels,
            )
            source_units = _build_episode_aware_units(
                source_frame,
                source_key=source_key,
                issue_mask_values=local_issue_mask,
                block_rows=block_rows,
                episode_context_rows=episode_context_rows,
                episode_merge_gap_rows=episode_merge_gap_rows,
            )
            source_units_by_key[source_key] = source_units
            ordered_units.extend(source_units)

        _assign_episode_aware_units(
            ordered_units,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )

        for source_key, source_frame in ordered.groupby(source_column, sort=False, observed=False):
            source_frame = source_frame.reset_index(drop=True)
            _append_trimmed_unit_frames(
                split_frames,
                source_frame,
                source_units_by_key.get(source_key, []),
                purge_gap_rows=purge_gap_rows,
            )
        return _finalize_split_frames(split_frames, template_frame=ordered, time_column=time_column)

    global_block_index = 0
    for _, source_frame in ordered.groupby(source_column, sort=False, observed=False):
        source_frame = source_frame.reset_index(drop=True)
        for start in range(0, len(source_frame), block_rows):
            block = source_frame.iloc[start : start + block_rows].copy()
            if block.empty:
                continue
            split_name = _split_slot_name(
                global_block_index,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
                cycle_length=block_cycle_length,
            )
            split_frames[split_name].append(block)
            global_block_index += 1
    return _finalize_split_frames(split_frames, template_frame=ordered, time_column=time_column)


def scan_interleaved_block_rows(
    frame: pd.DataFrame,
    *,
    label_column: str,
    train_fraction: float,
    validation_fraction: float,
    candidate_block_rows: tuple[int, ...] = (512, 1024, 2048, 4096),
    time_column: str = "Time UTC",
    source_column: str = "source_file",
) -> pd.DataFrame:
    """Compare several interleaved-block sizes and summarize split balance."""

    rows = []
    for block_rows in candidate_block_rows:
        split_frames = split_frame_by_strategy(
            frame,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            strategy="interleaved_blocks",
            time_column=time_column,
            source_column=source_column,
            block_rows=block_rows,
        )
        split_counts, split_shares = summarize_split_distributions(split_frames, label_column=label_column)
        rows.append(
            {
                "block_rows": int(block_rows),
                **compute_split_share_gap(split_shares),
                "train_rows": int(split_counts["train"].sum()),
                "validation_rows": int(split_counts["validation"].sum()),
                "test_rows": int(split_counts["test"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["max_pairwise_total_variation", "mean_pairwise_total_variation", "block_rows"]
    ).reset_index(drop=True)


def compute_contiguous_split_target_distribution(
    part_paths: list[str | Path],
    *,
    target_flag: str,
    train_fraction: float,
    validation_fraction: float,
    labels: list[int] | None = None,
    part_row_counts: list[int] | None = None,
    batch_size: int = 250_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute full-cache target balance for a contiguous time split without loading all rows.

    This helper scans only the target column from each parquet part, preserving
    time order across parts. That lets the notebook compare the *true* full-cache
    contiguous split with whatever smaller sampled dataframe is loaded later.
    """

    ordered_paths = [Path(path) for path in part_paths]
    if part_row_counts is None:
        part_row_counts = [pq.ParquetFile(path).metadata.num_rows for path in ordered_paths]

    total_rows = int(sum(part_row_counts))
    train_cut = int(total_rows * train_fraction)
    validation_cut = int(total_rows * (train_fraction + validation_fraction))

    split_order = ("train", "validation", "test")
    split_counts: dict[str, dict[int, int]] = {split_name: {} for split_name in split_order}
    global_row_offset = 0

    def add_segment_counts(split_name: str, values: pd.Series) -> None:
        if values.empty:
            return
        for label, count in values.astype(int).value_counts().items():
            split_counts[split_name][int(label)] = split_counts[split_name].get(int(label), 0) + int(count)

    for path in ordered_paths:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(columns=[target_flag], batch_size=batch_size):
            numeric = pd.to_numeric(batch.column(0).to_pandas(), errors="coerce")
            batch_length = len(numeric)
            batch_start = global_row_offset
            batch_end = global_row_offset + batch_length

            segments = [
                ("train", batch_start, min(batch_end, train_cut)),
                ("validation", max(batch_start, train_cut), min(batch_end, validation_cut)),
                ("test", max(batch_start, validation_cut), batch_end),
            ]
            for split_name, segment_start, segment_end in segments:
                if segment_end <= segment_start:
                    continue
                local_start = segment_start - batch_start
                local_end = segment_end - batch_start
                segment = numeric.iloc[local_start:local_end].dropna()
                add_segment_counts(split_name, segment)

            global_row_offset = batch_end

    if labels is None:
        labels = sorted({label for counts in split_counts.values() for label in counts})

    count_frame = pd.DataFrame(
        {
            split_name: pd.Series(split_counts[split_name], dtype="int64")
            for split_name in split_order
        }
    ).reindex(labels, fill_value=0).fillna(0).astype(int)
    share_frame = count_frame.div(count_frame.sum(axis=0), axis=1).fillna(0.0)
    return count_frame, share_frame


def build_model_frame(
    df: pd.DataFrame,
    *,
    target_flag: str,
    measurement_columns: list[str] | None = None,
    columns: list[str] | None = None,
    task_mode: str,
    good_labels: list[int] | tuple[int, ...] | None = None,
    issue_labels: list[int] | tuple[int, ...] | None = None,
    model_row_limit: int | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Create the baseline tabular feature frame used by the supervised models.

    The older notebook flow passed the selected sensor columns through a
    `columns=` keyword, while the newer generalized notebooks use
    `measurement_columns=`. Accepting both keeps reruns and partially refreshed
    kernels from failing when those two versions briefly overlap.
    """
    if measurement_columns is None:
        measurement_columns = columns
    if measurement_columns is None:
        raise ValueError("build_model_frame requires measurement_columns or columns.")

    model_df, active_labels = build_reviewed_target_frame(
        df,
        target_flag=target_flag,
        task_mode=task_mode,
        good_labels=good_labels,
        issue_labels=issue_labels,
        model_row_limit=model_row_limit,
    )
    model_df, feature_columns = add_tabular_baseline_features(
        model_df,
        measurement_columns=measurement_columns,
    )
    return model_df, feature_columns, active_labels


def add_temporal_context_features(
    frame: pd.DataFrame,
    *,
    measurement_columns: list[str],
    lag_steps: tuple[int, ...] = (1, 3, 5),
    rolling_windows: tuple[int, ...] = (5, 15),
) -> tuple[pd.DataFrame, list[str]]:
    """Add lag and rolling statistics that summarize recent sensor history."""
    work = frame.sort_values("Time UTC").reset_index(drop=True).copy()
    context_columns: list[str] = []

    for column in measurement_columns:
        for lag in lag_steps:
            feature_name = f"{column} lag_{lag}"
            work[feature_name] = work[column].shift(lag)
            context_columns.append(feature_name)

        for window in rolling_windows:
            mean_name = f"{column} roll_mean_{window}"
            std_name = f"{column} roll_std_{window}"
            work[mean_name] = work[column].rolling(window=window, min_periods=1).mean()
            work[std_name] = work[column].rolling(window=window, min_periods=2).std()
            context_columns.extend([mean_name, std_name])

    work[context_columns] = work[context_columns].replace([np.inf, -np.inf], np.nan)
    return work, context_columns


def apply_target_strategy(
    frame: pd.DataFrame,
    target_flag: str,
    strategy: str,
    *,
    issue_labels: list[int] | tuple[int, ...] | None = None,
) -> tuple[pd.DataFrame, list[int], str]:
    """Map raw QC flags into one of the teaching target strategies."""
    work = frame.copy()
    if strategy == "raw_multiclass":
        work["strategy_target"] = work[target_flag].astype(int)
        labels = sorted(work["strategy_target"].dropna().astype(int).unique().tolist())
        average = "macro"
    elif strategy == "multiclass_1_3_4_9":
        work["strategy_target"] = work[target_flag].astype(int)
        labels = [1, 3, 4, 9]
        average = "macro"
    elif strategy == "collapsed_1_34_9":
        mapping = {1: 1, 3: 34, 4: 34, 9: 9}
        work["strategy_target"] = work[target_flag].astype(int).map(mapping)
        labels = [1, 34, 9]
        average = "macro"
    elif strategy in {"collapsed_12_34_9", "oxygen_collapse_12_34_9"}:
        mapping = {1: 12, 2: 12, 3: 34, 4: 34, 9: 9}
        work["strategy_target"] = work[target_flag].astype(int).map(mapping)
        labels = [12, 34, 9]
        average = "macro"
    elif strategy == "binary_issue":
        work["strategy_target"] = issue_mask(work[target_flag], issue_labels).astype(int)
        labels = [0, 1]
        average = "binary"
    else:
        raise ValueError(f"Unsupported target strategy: {strategy}")

    work = work.dropna(subset=["strategy_target"]).copy()
    work["strategy_target"] = work["strategy_target"].astype(int)
    return work, labels, average


def contiguous_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-sorted dataframe into contiguous train/valid/test segments."""
    train_end = int(len(frame) * train_fraction)
    valid_end = int(len(frame) * (train_fraction + validation_fraction))
    train_frame = frame.iloc[:train_end].copy()
    valid_frame = frame.iloc[train_end:valid_end].copy()
    test_frame = frame.iloc[valid_end:].copy()
    return train_frame, valid_frame, test_frame


def report_average(task_mode: str) -> str:
    """Return the F1 averaging mode that matches the task definition."""
    return "binary" if task_mode == "binary" else "macro"


def fit_random_forest(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    *,
    seed: int,
    config: dict[str, object],
) -> Pipeline:
    """Train the baseline Random Forest pipeline used in the notebooks."""
    rf_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=config["n_estimators"],
                    max_depth=config["max_depth"],
                    min_samples_leaf=config["min_samples_leaf"],
                    min_samples_split=config.get("min_samples_split", 2),
                    max_features=config.get("max_features", "sqrt"),
                    class_weight=config.get("class_weight", "balanced_subsample"),
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )
    rf_pipeline.fit(train_df[feature_columns], train_df["model_target"])
    return rf_pipeline


def fit_extra_trees(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    *,
    seed: int,
    config: dict[str, object],
    target_column: str = "model_target",
) -> Pipeline:
    """Train an ExtraTrees pipeline with the same preprocessing pattern as RF."""
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=config["n_estimators"],
                    max_depth=config["max_depth"],
                    min_samples_leaf=config["min_samples_leaf"],
                    min_samples_split=config.get("min_samples_split", 2),
                    max_features=config.get("max_features", "sqrt"),
                    class_weight=config.get("class_weight", "balanced_subsample"),
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit(train_df[feature_columns], train_df[target_column])
    return pipeline


def evaluate_classifier(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    labels: list[int],
    task_mode: str,
) -> dict[str, object]:
    """Compute prediction outputs and summary metrics for a fitted classifier."""
    y_true = frame["model_target"]
    y_pred = pipeline.predict(frame[feature_columns])
    return {
        "f1": float(f1_score(y_true, y_pred, average=report_average(task_mode), zero_division=0)),
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=False),
        "predictions": y_pred,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels, normalize="true"),
    }


def run_rf_search(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    *,
    labels: list[int],
    task_mode: str,
    seed: int,
    search_space: dict[str, list[object]],
) -> tuple[pd.DataFrame, dict[str, object], Pipeline]:
    """Grid-search a small Random Forest search space and keep the best model."""
    keys = list(search_space.keys())
    results = []
    best_score = -math.inf
    best_config = None
    best_model = None

    for trial_index, values in enumerate(itertools.product(*(search_space[key] for key in keys)), start=1):
        config = dict(zip(keys, values))
        model = fit_random_forest(train_df, feature_columns, seed=seed, config=config)
        valid_result = evaluate_classifier(model, valid_df, feature_columns, labels=labels, task_mode=task_mode)
        row = {"trial": trial_index, **config, "validation_f1": valid_result["f1"]}
        results.append(row)
        if valid_result["f1"] > best_score:
            best_score = valid_result["f1"]
            best_config = config
            best_model = model

    result_frame = pd.DataFrame(results).sort_values("validation_f1", ascending=False).reset_index(drop=True)
    return result_frame, best_config, best_model


def clean_source_file_label(value: str) -> str:
    """Trim verbose filename suffixes for cleaner notebook display tables."""
    return str(value).replace(".csv", "").split("_2025")[0].split("_2026")[0]


def _span_boundaries(times: pd.Series, start_index: int, stop_index: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Expand a contiguous run to midpoints with neighboring timestamps for clearer shading."""
    times = pd.to_datetime(times, utc=True).reset_index(drop=True)
    start_time = times.iloc[start_index]
    end_time = times.iloc[stop_index]

    if start_index > 0:
        previous_time = times.iloc[start_index - 1]
        start_time = previous_time + (start_time - previous_time) / 2
    if stop_index < len(times) - 1:
        next_time = times.iloc[stop_index + 1]
        end_time = end_time + (next_time - end_time) / 2

    return start_time, end_time


def _flag_span_boundaries(panel: pd.DataFrame, start_index: int, stop_index: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Delegate QC-flag shading boundaries to the generic span helper."""
    return _span_boundaries(panel["Time UTC"], start_index, stop_index)


def _iter_flag_spans(
    panel: pd.DataFrame,
    target_flag: str,
    *,
    good_labels: list[int] | tuple[int, ...] | None = None,
) -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    """Return contiguous non-good QC regions as (flag, span_start, span_end)."""
    normalized_good_labels = set(normalize_label_list(good_labels, DEFAULT_GOOD_LABELS))
    flag_values = panel[target_flag].copy()
    if pd.api.types.is_numeric_dtype(flag_values):
        flag_values = flag_values.fillna(9).astype(int)
    else:
        flag_values = pd.to_numeric(flag_values, errors="coerce").fillna(9).astype(int)

    spans: list[tuple[int, pd.Timestamp, pd.Timestamp]] = []
    run_start = 0

    for index in range(1, len(panel) + 1):
        reached_end = index == len(panel)
        if reached_end or flag_values.iloc[index] != flag_values.iloc[run_start]:
            run_flag = int(flag_values.iloc[run_start])
            if run_flag not in normalized_good_labels:
                span_start, span_end = _flag_span_boundaries(panel, run_start, index - 1)
                spans.append((run_flag, span_start, span_end))
            run_start = index

    return spans


def parse_optional_utc_datetime(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    """Parse an optional datetime-like value into a timezone-aware UTC timestamp."""
    if value is None or value == "":
        return None
    parsed = pd.to_datetime(value, utc=True)
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def build_row_part_index(metadata: dict[str, object], row_cache_dir: str | Path) -> pd.DataFrame:
    """Convert cache metadata into a searchable dataframe of parquet parts."""
    row_cache_path = Path(row_cache_dir)
    records = []
    for file_info in metadata.get("processed_files", []):
        time_start = parse_optional_utc_datetime(file_info.get("time_start"))
        time_end = parse_optional_utc_datetime(file_info.get("time_end"))
        records.append(
            {
                "source_file": file_info["source_file"],
                "time_start": time_start,
                "time_end": time_end,
                "row_part_path": row_cache_path / Path(str(file_info["row_level_part"])).name,
            }
        )
    return pd.DataFrame(records)


def select_overlapping_row_parts(
    metadata: dict[str, object],
    row_cache_dir: str | Path,
    *,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
) -> pd.DataFrame:
    """Return only the parquet parts that overlap the requested time interval."""
    part_index = build_row_part_index(metadata, row_cache_dir)
    if part_index.empty:
        return part_index

    start_ts = parse_optional_utc_datetime(start)
    end_ts = parse_optional_utc_datetime(end)

    if start_ts is not None:
        part_index = part_index[part_index["time_end"].isna() | (part_index["time_end"] >= start_ts)]
    if end_ts is not None:
        part_index = part_index[part_index["time_start"].isna() | (part_index["time_start"] <= end_ts)]
    return part_index.sort_values("time_start").reset_index(drop=True)


def load_rows_for_time_range(
    metadata: dict[str, object],
    row_cache_dir: str | Path,
    *,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load only the row-level parquet parts needed for a selected time interval."""
    overlapping_parts = select_overlapping_row_parts(metadata, row_cache_dir, start=start, end=end)
    if overlapping_parts.empty:
        if columns:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame()

    start_ts = parse_optional_utc_datetime(start)
    end_ts = parse_optional_utc_datetime(end)
    frames = []
    for row in overlapping_parts.itertuples(index=False):
        frame = pd.read_parquet(row.row_part_path, columns=columns).sort_values("Time UTC").reset_index(drop=True)
        frame["Time UTC"] = pd.to_datetime(frame["Time UTC"], utc=True)
        if start_ts is not None:
            frame = frame[frame["Time UTC"] >= start_ts]
        if end_ts is not None:
            frame = frame[frame["Time UTC"] <= end_ts]
        if not frame.empty:
            frames.append(frame)

    if not frames:
        if columns:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True).sort_values("Time UTC").reset_index(drop=True)


def select_time_range(
    reference_frame: pd.DataFrame,
    *,
    time_column: str = "Time UTC",
    label_column: str | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    auto_select: bool = True,
    max_points: int = 800,
    preferred_labels: tuple[int, ...] = (4, 3, 9),
) -> dict[str, object]:
    """Choose either an explicit or representative interval for a notebook demo."""
    work = reference_frame.dropna(subset=[time_column]).sort_values(time_column).reset_index(drop=True).copy()
    if work.empty:
        raise ValueError("Cannot select a time range from an empty frame.")
    work[time_column] = pd.to_datetime(work[time_column], utc=True)

    start_ts = parse_optional_utc_datetime(start)
    end_ts = parse_optional_utc_datetime(end)

    if start_ts is not None or end_ts is not None:
        if start_ts is None:
            start_ts = work[time_column].min()
        if end_ts is None:
            end_ts = work[time_column].max()
        if end_ts < start_ts:
            raise ValueError("Range end must be greater than or equal to range start.")
        explicit_slice = work[(work[time_column] >= start_ts) & (work[time_column] <= end_ts)].copy()
        return {
            "start": start_ts,
            "end": end_ts,
            "slice": explicit_slice.reset_index(drop=True),
            "selection_mode": "explicit",
            "selected_label": None,
        }

    if not auto_select:
        selected = evenly_spaced_take(work, min(max_points, len(work)))
        return {
            "start": selected[time_column].min(),
            "end": selected[time_column].max(),
            "slice": selected.reset_index(drop=True),
            "selection_mode": "manual-full-range",
            "selected_label": None,
        }

    chosen_index = len(work) // 2
    chosen_label = None
    # When auto-selecting, bias toward intervals that contain more informative
    # non-good labels so the demo is visually useful.
    if label_column is not None and label_column in work.columns:
        label_series = pd.to_numeric(work[label_column], errors="coerce")
        for label in preferred_labels:
            candidate_indices = work.index[label_series == label].tolist()
            if candidate_indices:
                chosen_index = candidate_indices[len(candidate_indices) // 2]
                chosen_label = label
                break

    point_count = min(max_points, len(work))
    start_index = max(chosen_index - point_count // 2, 0)
    stop_index = min(start_index + point_count, len(work))
    start_index = max(stop_index - point_count, 0)
    selected = work.iloc[start_index:stop_index].copy().reset_index(drop=True)

    return {
        "start": selected[time_column].min(),
        "end": selected[time_column].max(),
        "slice": selected,
        "selection_mode": "auto",
        "selected_label": chosen_label,
    }


def infer_interval_origin(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    split_frames: dict[str, pd.DataFrame],
    *,
    time_column: str = "Time UTC",
) -> str:
    """Label an interval as train, validation, test, mixed, or outside range."""
    start_ts = parse_optional_utc_datetime(start)
    end_ts = parse_optional_utc_datetime(end)
    if start_ts is None or end_ts is None:
        return "unknown"

    overlaps: list[str] = []
    for split_name, frame in split_frames.items():
        if frame.empty or time_column not in frame.columns:
            continue
        frame_times = pd.to_datetime(frame[time_column], utc=True)
        frame_start = frame_times.min()
        frame_end = frame_times.max()
        if pd.isna(frame_start) or pd.isna(frame_end):
            continue
        if start_ts <= frame_end and end_ts >= frame_start:
            overlaps.append(split_name)

    if not overlaps:
        return "outside modeled range"
    if len(overlaps) == 1:
        return overlaps[0]
    return f"mixed ({', '.join(overlaps)})"


def build_labeled_intervals(
    frame: pd.DataFrame,
    *,
    time_column: str,
    label_column: str,
    fill_value: object | None = None,
) -> pd.DataFrame:
    """Collapse row-by-row labels into contiguous labeled time spans."""
    work = frame[[time_column, label_column]].copy().dropna(subset=[time_column]).sort_values(time_column).reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(columns=["start", "end", "label"])

    work[time_column] = pd.to_datetime(work[time_column], utc=True)
    if fill_value is not None:
        work[label_column] = work[label_column].fillna(fill_value)
    else:
        work = work.dropna(subset=[label_column]).reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(columns=["start", "end", "label"])

    labels = work[label_column].tolist()
    intervals = []
    run_start = 0
    for index in range(1, len(work) + 1):
        reached_end = index == len(work)
        if reached_end or labels[index] != labels[run_start]:
            span_start, span_end = _span_boundaries(work[time_column], run_start, index - 1)
            intervals.append({"start": span_start, "end": span_end, "label": labels[run_start]})
            run_start = index
    return pd.DataFrame(intervals)


def merge_adjacent_intervals(
    interval_frame: pd.DataFrame,
    *,
    label_column: str = "label",
    start_column: str = "start",
    end_column: str = "end",
) -> pd.DataFrame:
    """Merge touching intervals that share the same label."""
    if interval_frame.empty:
        return interval_frame.copy()

    work = interval_frame.copy().sort_values(start_column).reset_index(drop=True)
    merged = [work.iloc[0].to_dict()]
    for row in work.iloc[1:].itertuples(index=False):
        current = row._asdict()
        previous = merged[-1]
        if current[label_column] == previous[label_column] and current[start_column] <= previous[end_column]:
            previous[end_column] = max(previous[end_column], current[end_column])
        else:
            merged.append(current)
    return pd.DataFrame(merged)


def build_label_palette(labels: list[object], *, palette: dict[object, str] | None = None) -> dict[object, object]:
    """Assign display colors to an ordered set of labels."""
    if palette is not None:
        return {label: palette.get(label, palette.get(int(label), "#64748b")) if isinstance(label, (int, np.integer)) else palette.get(label, "#64748b") for label in labels}

    unique_labels = list(dict.fromkeys(labels))
    color_values = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))
    return {label: color_values[index] for index, label in enumerate(unique_labels)}


def plot_time_series_with_bands(
    row_frame: pd.DataFrame,
    *,
    band_specs: list[dict[str, object]],
    measurement_column: str = "Conductivity (S/m)",
    secondary_column: str = "Temperature (C)",
    max_points: int | None = None,
    title: str = "Time-range model demo",
) -> plt.Figure:
    """Plot sensor traces with one or more aligned label-band panels underneath."""
    if row_frame.empty:
        raise ValueError("row_frame must not be empty.")

    plot_frame = row_frame.sort_values("Time UTC").reset_index(drop=True).copy()
    plot_frame["Time UTC"] = pd.to_datetime(plot_frame["Time UTC"], utc=True)
    if max_points is not None:
        plot_frame = evenly_spaced_take(plot_frame, max_points)

    row_count = 1 + len(band_specs)
    height_ratios = [3.6] + [0.9] * len(band_specs)
    fig, axes = plt.subplots(
        row_count,
        1,
        figsize=(16.5, 3.2 + 1.3 * len(band_specs)),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if row_count == 1:
        axes = [axes]

    main_axis = axes[0]
    band_axes = axes[1:]
    main_axis.plot(plot_frame["Time UTC"], plot_frame[measurement_column], color="#0f172a", linewidth=1.8, label=measurement_column)
    main_axis.set_ylabel(measurement_column)
    main_axis.grid(alpha=0.25)

    twin_axis = main_axis.twinx()
    twin_axis.plot(plot_frame["Time UTC"], plot_frame[secondary_column], color="#059669", linewidth=1.2, alpha=0.7, label=secondary_column)
    twin_axis.set_ylabel(secondary_column, color="#059669")
    twin_axis.tick_params(axis="y", colors="#059669")

    main_axis.legend(
        handles=[
            Line2D([0], [0], color="#0f172a", linewidth=2, label=measurement_column),
            Line2D([0], [0], color="#059669", linewidth=2, label=secondary_column),
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=True,
    )

    for axis, spec in zip(band_axes, band_specs):
        intervals = spec["intervals"].copy()
        label_order = list(dict.fromkeys(intervals["label"].tolist())) if not intervals.empty else []
        palette = build_label_palette(label_order, palette=spec.get("palette"))
        axis.set_ylim(0, 1)
        axis.set_yticks([])
        axis.grid(alpha=0.15)
        axis.set_title(str(spec["title"]), loc="left", fontsize=11, pad=4)

        if intervals.empty:
            axis.text(0.01, 0.5, "No intervals in this selected range.", transform=axis.transAxes, va="center")
            continue

        intervals = intervals.sort_values("start").reset_index(drop=True)
        for row in intervals.itertuples(index=False):
            color = palette.get(row.label, "#64748b")
            axis.axvspan(row.start, row.end, color=color, alpha=0.7, linewidth=0)

        if len(intervals) <= 14:
            for row in intervals.itertuples(index=False):
                midpoint = row.start + (row.end - row.start) / 2
                axis.text(midpoint, 0.5, str(row.label), ha="center", va="center", fontsize=9, color="#111827")

        legend_handles = [
            Patch(facecolor=palette.get(label, "#64748b"), edgecolor="none", alpha=0.7, label=str(label))
            for label in label_order[:8]
        ]
        if legend_handles:
            axis.legend(
                handles=legend_handles,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0.0,
                ncol=1,
                frameon=True,
            )

    axes[-1].set_xlabel("Time UTC")
    fig.suptitle(title, y=1.01)
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    return fig


def compute_interval_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    labels: list[int],
    average: str,
    target_names: list[str] | None = None,
) -> dict[str, object]:
    """Compute the text report, normalized confusion matrix, and F1 for a slice."""
    label_names = target_names or [str(label) for label in labels]
    if average == "binary":
        interval_f1 = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
    else:
        interval_f1 = float(f1_score(y_true, y_pred, labels=labels, average=average, zero_division=0))
    return {
        "f1": interval_f1,
        "report_text": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=label_names,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
            normalize="true",
        ),
        "display_labels": label_names,
    }


def build_window_classification_interval_data(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    task_mode: str,
    window_size: int,
    label_reduction: str,
    time_column: str = "Time UTC",
) -> dict[str, object]:
    """Prepare fixed windows for window-level CNN or transformer inference."""
    work = frame[[time_column, *feature_columns, target_column]].copy().dropna().sort_values(time_column).reset_index(drop=True)
    usable_rows = (len(work) // window_size) * window_size
    work = work.iloc[:usable_rows].copy()
    if work.empty:
        return {"raw_sequences": np.empty((0, window_size, len(feature_columns)), dtype=np.float32), "window_frame": pd.DataFrame(), "class_labels": []}

    raw_sequences = work[feature_columns].to_numpy(dtype=np.float32).reshape(-1, window_size, len(feature_columns))
    raw_targets = work[target_column].to_numpy().reshape(-1, window_size)
    raw_times = pd.to_datetime(work[time_column], utc=True).to_numpy().reshape(-1, window_size)

    if task_mode == "multiclass":
        true_labels = np.array([reduce_window_target(row, mode=label_reduction) for row in raw_targets], dtype=int)
        class_labels = sorted(np.unique(true_labels).tolist())
    else:
        true_labels = raw_targets.max(axis=1).astype(int)
        class_labels = [0, 1]

    window_frame = pd.DataFrame(
        {
            "window_start": pd.to_datetime(raw_times[:, 0], utc=True),
            "window_end": pd.to_datetime(raw_times[:, -1], utc=True),
            "true_label": true_labels,
        }
    )
    return {
        "raw_sequences": raw_sequences,
        "window_frame": window_frame,
        "class_labels": class_labels,
    }


def build_sequence_label_interval_data(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    window_size: int,
    time_column: str = "Time UTC",
) -> dict[str, object]:
    """Prepare fixed windows for per-timestamp sequence-label inference."""
    work = frame[[time_column, *feature_columns, target_column]].copy().dropna().sort_values(time_column).reset_index(drop=True)
    usable_rows = (len(work) // window_size) * window_size
    work = work.iloc[:usable_rows].copy()
    if work.empty:
        return {
            "raw_sequences": np.empty((0, window_size, len(feature_columns)), dtype=np.float32),
            "raw_targets": np.empty((0, window_size)),
            "raw_times": np.empty((0, window_size), dtype="datetime64[ns]"),
        }

    raw_sequences = work[feature_columns].to_numpy(dtype=np.float32).reshape(-1, window_size, len(feature_columns))
    raw_targets = work[target_column].to_numpy().reshape(-1, window_size)
    raw_times = pd.to_datetime(work[time_column], utc=True).to_numpy().reshape(-1, window_size)
    return {"raw_sequences": raw_sequences, "raw_targets": raw_targets, "raw_times": raw_times}


def predict_cnn_window_model(
    model: nn.Module,
    raw_sequences: np.ndarray,
    *,
    task_mode: str,
    class_labels: list[int],
    device: str,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Run a window-classification CNN on raw sequences and decode labels."""
    _require_torch()
    if len(raw_sequences) == 0:
        return np.array([], dtype=int)

    model.eval()
    tensor = torch.from_numpy(np.transpose(raw_sequences, (0, 2, 1))).float()
    normalized = (tensor - torch.as_tensor(channel_mean).float()) / torch.as_tensor(channel_std).float()
    predictions = []
    with torch.no_grad():
        for start_index in range(0, len(normalized), batch_size):
            batch = normalized[start_index : start_index + batch_size].to(device)
            logits = model(batch)
            if task_mode == "multiclass":
                batch_predictions = logits.argmax(dim=1).cpu().numpy()
                batch_predictions = np.array([class_labels[index] for index in batch_predictions], dtype=int)
            else:
                batch_predictions = (torch.sigmoid(logits.squeeze(-1)) >= 0.5).long().cpu().numpy().astype(int)
            predictions.append(batch_predictions)
    return np.concatenate(predictions) if predictions else np.array([], dtype=int)


def predict_transformer_window_model(
    model: nn.Module,
    raw_sequences: np.ndarray,
    *,
    task_mode: str,
    class_labels: list[int],
    device: str,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Run the notebook transformer on raw sequences and decode labels."""
    _require_torch()
    if len(raw_sequences) == 0:
        return np.array([], dtype=int)

    model.eval()
    tensor = torch.from_numpy(raw_sequences).float()
    normalized = (tensor - torch.as_tensor(feature_mean).float()) / torch.as_tensor(feature_std).float()
    predictions = []
    with torch.no_grad():
        for start_index in range(0, len(normalized), batch_size):
            batch = normalized[start_index : start_index + batch_size].to(device)
            logits = model(batch)
            if task_mode == "multiclass":
                batch_predictions = logits.argmax(dim=1).cpu().numpy()
                batch_predictions = np.array([class_labels[index] for index in batch_predictions], dtype=int)
            else:
                batch_predictions = (torch.sigmoid(logits.squeeze(-1)) >= 0.5).long().cpu().numpy().astype(int)
            predictions.append(batch_predictions)
    return np.concatenate(predictions) if predictions else np.array([], dtype=int)


def predict_transformer_sequence_model(
    model: nn.Module,
    raw_sequences: np.ndarray,
    *,
    task_mode: str,
    class_labels: list[int],
    device: str,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Run a sequence-labeling transformer and return one prediction per timestep."""
    _require_torch()
    if len(raw_sequences) == 0:
        return np.empty((0, 0), dtype=int)

    model.eval()
    tensor = torch.from_numpy(raw_sequences).float()
    normalized = (tensor - torch.as_tensor(feature_mean).float()) / torch.as_tensor(feature_std).float()
    predictions = []
    with torch.no_grad():
        for start_index in range(0, len(normalized), batch_size):
            batch = normalized[start_index : start_index + batch_size].to(device)
            logits = model(batch)
            if task_mode == "multiclass":
                batch_predictions = logits.argmax(dim=-1).cpu().numpy()
                batch_predictions = np.take(np.asarray(class_labels, dtype=int), batch_predictions)
            else:
                batch_predictions = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy().astype(int)
            predictions.append(batch_predictions)
    return np.concatenate(predictions, axis=0) if predictions else np.empty((0, 0), dtype=int)


def predict_sequence_label_cnn(
    model: nn.Module,
    raw_sequences: np.ndarray,
    *,
    task_mode: str,
    class_labels: list[int],
    device: str,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    batch_size: int = 128,
) -> np.ndarray:
    """Run a sequence-labeling CNN and return one prediction per timestamp."""
    _require_torch()
    if len(raw_sequences) == 0:
        return np.empty((0, 0), dtype=int)

    model.eval()
    tensor = torch.from_numpy(np.transpose(raw_sequences, (0, 2, 1))).float()
    normalized = (tensor - torch.as_tensor(channel_mean).float()) / torch.as_tensor(channel_std).float()
    predictions = []
    with torch.no_grad():
        for start_index in range(0, len(normalized), batch_size):
            batch = normalized[start_index : start_index + batch_size].to(device)
            logits = model(batch)
            if task_mode == "multiclass":
                if logits.ndim != 3:
                    raise ValueError(
                        "Expected multiclass sequence CNN logits to have three dimensions "
                        f"(batch, time, classes) or (batch, classes, time), got shape {tuple(logits.shape)}."
                    )
                if logits.shape[-1] == len(class_labels):
                    batch_predictions = logits.argmax(dim=-1).cpu().numpy()
                elif logits.shape[1] == len(class_labels):
                    batch_predictions = logits.argmax(dim=1).cpu().numpy()
                else:
                    raise ValueError(
                        "Could not align CNN logits with class labels: "
                        f"logits shape {tuple(logits.shape)}, class count {len(class_labels)}."
                    )
                batch_predictions = np.take(np.asarray(class_labels, dtype=int), batch_predictions)
            else:
                batch_predictions = (torch.sigmoid(logits.squeeze(1)) >= 0.5).long().cpu().numpy().astype(int)
            predictions.append(batch_predictions)
    return np.concatenate(predictions, axis=0) if predictions else np.empty((0, 0), dtype=int)


def plot_flag_examples(
    df: pd.DataFrame,
    *,
    target_flag: str,
    measurement_column: str = "Conductivity (S/m)",
    secondary_column: str | None = "Temperature (C)",
    points_per_panel: int = 300,
    classes: tuple[int, ...] = (1, 3, 4, 9),
    good_labels: list[int] | tuple[int, ...] | None = None,
    region_alpha: float = 0.18,
    show_flag_points: bool = True,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot representative QC-flag examples with shaded regions and sensor traces.

    The important detail is that each plotted panel should stay *locally
    contiguous in time*. When the incoming dataframe contains rows from several
    source files or discontinuous time chunks, we therefore choose the plotting
    window from the same ``source_file`` as the representative flagged row
    rather than slicing across the globally sorted dataframe.

    Some raw orientation datasets are also missing the requested secondary
    column, especially when a profile points at mixed-device raw exports. In
    that case we simply omit the secondary axis instead of failing.
    """
    available_classes = [flag for flag in classes if flag in set(df[target_flag].dropna().astype(int).unique())]
    if not available_classes:
        raise ValueError(f"No requested classes found in {target_flag}")

    work = df.sort_values("Time UTC").reset_index(drop=True).copy()
    fig, axes = plt.subplots(len(available_classes), 1, figsize=(15, 3.6 * len(available_classes)), sharex=False)
    if len(available_classes) == 1:
        axes = [axes]

    example_rows = []
    normalized_good_labels = set(normalize_label_list(good_labels, DEFAULT_GOOD_LABELS))
    colors = dict(DEFAULT_FLAG_PALETTE)
    line_color = "#0f172a"
    temp_color = "#059669"
    for axis, flag in zip(axes, available_classes):
        flag_rows = work.index[work[target_flag].fillna(-1).astype(int) == flag].tolist()
        center_index = flag_rows[len(flag_rows) // 2]
        center_row = work.loc[center_index]

        # Keep the context panel inside the same source file when possible so a
        # "local" example does not accidentally jump across large time gaps.
        if "source_file" in work.columns and pd.notna(center_row.get("source_file")):
            source_file = center_row["source_file"]
            source_panel = work[work["source_file"] == source_file].sort_values("Time UTC").reset_index(drop=True)
            source_flag_rows = source_panel.index[source_panel[target_flag].fillna(-1).astype(int) == flag].tolist()
            if source_flag_rows:
                source_center_index = source_flag_rows[len(source_flag_rows) // 2]
                span_start_index = source_center_index
                while span_start_index > 0 and int(source_panel.iloc[span_start_index - 1][target_flag]) == flag:
                    span_start_index -= 1
                span_end_index = source_center_index
                while (
                    span_end_index + 1 < len(source_panel)
                    and int(source_panel.iloc[span_end_index + 1][target_flag]) == flag
                ):
                    span_end_index += 1

                # Center the panel on the midpoint of the contiguous target span
                # rather than on an arbitrary example row. This keeps the
                # highlighted issue closer to the middle of the subplot.
                span_midpoint_index = (span_start_index + span_end_index) // 2
                start = max(span_midpoint_index - points_per_panel // 2, 0)
                stop = min(start + points_per_panel, len(source_panel))
                if stop - start < points_per_panel:
                    start = max(stop - points_per_panel, 0)
                panel = source_panel.iloc[start:stop].copy()
                example_time = source_panel.iloc[span_midpoint_index]["Time UTC"]
            else:
                start = max(center_index - points_per_panel // 2, 0)
                stop = min(start + points_per_panel, len(work))
                panel = work.iloc[start:stop].copy()
                example_time = center_row["Time UTC"]
        else:
            start = max(center_index - points_per_panel // 2, 0)
            stop = min(start + points_per_panel, len(work))
            panel = work.iloc[start:stop].copy()
            example_time = center_row["Time UTC"]
        panel_spans = _iter_flag_spans(panel, target_flag, good_labels=good_labels)
        flags_in_panel = sorted({span_flag for span_flag, _, _ in panel_spans})

        # Shade all non-good QC spans so participants can see the local context
        # around the example class rather than only isolated points.
        for span_flag, span_start, span_end in panel_spans:
            axis.axvspan(
                span_start,
                span_end,
                color=colors.get(span_flag, "#9467bd"),
                alpha=region_alpha,
                linewidth=0,
                zorder=0,
            )

        if measurement_column not in panel.columns:
            raise KeyError(
                f"Measurement column {measurement_column!r} is not available in the flag-example panel. "
                f"Available columns: {sorted(panel.columns.tolist())}"
            )

        axis.plot(panel["Time UTC"], panel[measurement_column], color=line_color, linewidth=1.8, label=measurement_column)
        if show_flag_points and flag not in normalized_good_labels:
            target_points = panel.loc[panel[target_flag].fillna(-1).astype(int) == flag, ["Time UTC", measurement_column]]
            target_points = target_points.dropna(subset=[measurement_column])
            if not target_points.empty:
                axis.scatter(
                    target_points["Time UTC"],
                    target_points[measurement_column],
                    color=colors.get(flag, "#9467bd"),
                    s=20,
                    zorder=3,
                    label=f"Rows with QC flag {flag}",
                )
        axis.set_title(f"Example around QC flag {flag}: {QC_FLAG_MEANINGS.get(flag, 'unknown')}")
        axis.set_ylabel(measurement_column)
        axis.grid(alpha=0.25)

        secondary_available = bool(
            secondary_column
            and secondary_column in panel.columns
            and panel[secondary_column].notna().any()
        )
        if secondary_available:
            twin_axis = axis.twinx()
            twin_axis.plot(panel["Time UTC"], panel[secondary_column], color=temp_color, linewidth=1.2, alpha=0.6, label=secondary_column)
            twin_axis.set_ylabel(secondary_column, color=temp_color)
            twin_axis.tick_params(axis="y", colors=temp_color)

        legend_handles = [
            Line2D([0], [0], color=line_color, linewidth=2, label=measurement_column),
        ]
        if secondary_available:
            legend_handles.append(Line2D([0], [0], color=temp_color, linewidth=2, label=secondary_column))
        if show_flag_points and flag not in normalized_good_labels:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=colors.get(flag, "#9467bd"),
                    markersize=7,
                    label=f"QC flag {flag} points",
                )
            )
        legend_handles.extend(
            [
                Patch(
                    facecolor=colors.get(span_flag, "#9467bd"),
                    edgecolor="none",
                    alpha=region_alpha,
                    label=f"QC region {span_flag}: {QC_FLAG_MEANINGS.get(span_flag, 'unknown')}",
                )
                for span_flag in flags_in_panel
            ]
        )
        axis.legend(handles=legend_handles, loc="upper left", frameon=True)

        example_rows.append(
            {
                "qc_flag": flag,
                "meaning": QC_FLAG_MEANINGS.get(flag, "unknown"),
                "panel_start": panel["Time UTC"].min(),
                "panel_end": panel["Time UTC"].max(),
                "example_time": example_time,
                "source_file": clean_source_file_label(center_row["source_file"]) if "source_file" in work.columns else None,
            }
        )

    axes[-1].set_xlabel("Time UTC")
    fig.suptitle("Representative time-series examples for different QC flags", y=1.02)
    fig.tight_layout()
    return fig, pd.DataFrame(example_rows)


def plot_cluster_window_examples(
    clustered_window_df: pd.DataFrame,
    *,
    source_to_row_part: dict[str, str | Path],
    measurement_column: str = "Conductivity (S/m)",
    secondary_column: str = "Temperature (C)",
    target_flag: str = "Conductivity QC Flag",
    good_labels: list[int] | tuple[int, ...] | None = None,
    examples_per_cluster: int = 1,
    context_points: int = 1500,
    highlight_alpha: float = 0.22,
    flag_region_alpha: float = 0.14,
    flag_palette: dict[int, str] | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Show representative k-means windows inside a wider sensor-time context.

    Each panel now includes both:

    - the highlighted window that supplied one k-means example, and
    - the true QC-flag regions from the underlying row-level data.

    That makes it easier to see whether a cluster prototype lines up with known
    flagged behavior or with apparently normal operating periods.
    """
    required_columns = {"cluster", "window_start", "window_end", "source_file", "distance_to_centroid", "issue_rate"}
    missing = required_columns.difference(clustered_window_df.columns)
    if missing:
        raise ValueError(f"clustered_window_df is missing required columns: {sorted(missing)}")

    work = clustered_window_df.sort_values(["cluster", "distance_to_centroid", "issue_rate"], ascending=[True, True, False]).copy()
    cluster_ids = sorted(work["cluster"].dropna().astype(int).unique().tolist())
    if not cluster_ids:
        raise ValueError("No clusters found in clustered_window_df")

    palette = flag_palette or DEFAULT_FLAG_PALETTE
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_ids)))
    cluster_palette = {cluster_id: cluster_colors[idx] for idx, cluster_id in enumerate(cluster_ids)}

    # Use the windows closest to each centroid as readable cluster prototypes.
    representative_rows = (
        work.groupby("cluster", group_keys=False)
        .head(examples_per_cluster)
        .reset_index(drop=True)
    )

    figure_row_count = len(representative_rows)
    fig, axes = plt.subplots(figure_row_count, 1, figsize=(15, 3.8 * figure_row_count), sharex=False)
    if figure_row_count == 1:
        axes = [axes]

    example_records = []
    for axis, (_, row) in zip(axes, representative_rows.iterrows()):
        source_file = row["source_file"]
        row_part_path = Path(source_to_row_part[source_file])
        panel = pd.read_parquet(
            row_part_path,
            columns=["Time UTC", measurement_column, secondary_column, target_flag],
        ).sort_values("Time UTC").reset_index(drop=True)
        panel["Time UTC"] = pd.to_datetime(panel["Time UTC"], utc=True)

        window_start = pd.to_datetime(row["window_start"], utc=True)
        window_end = pd.to_datetime(row["window_end"], utc=True)
        center_time = window_start + (window_end - window_start) / 2
        time_delta = (panel["Time UTC"] - center_time).abs()
        center_index = int(time_delta.idxmin())
        start_index = max(center_index - context_points // 2, 0)
        stop_index = min(start_index + context_points, len(panel))
        context_panel = panel.iloc[start_index:stop_index].copy()

        cluster_id = int(row["cluster"])
        cluster_color = cluster_palette[cluster_id]
        flag_spans = _iter_flag_spans(context_panel, target_flag, good_labels=good_labels)

        shown_flag_labels: set[int] = set()
        for flag_value, span_start, span_end in flag_spans:
            axis.axvspan(
                span_start,
                span_end,
                color=palette.get(flag_value, "#9ca3af"),
                alpha=flag_region_alpha,
                linewidth=0,
                zorder=-1,
            )
            shown_flag_labels.add(flag_value)

        axis.plot(
            context_panel["Time UTC"],
            context_panel[measurement_column],
            color="#0f172a",
            linewidth=1.8,
            label=measurement_column,
        )
        target_points = context_panel[
            (context_panel["Time UTC"] >= window_start) & (context_panel["Time UTC"] <= window_end)
        ].dropna(subset=[measurement_column])
        if not target_points.empty:
            axis.scatter(
                target_points["Time UTC"],
                target_points[measurement_column],
                color=cluster_color,
                s=14,
                alpha=0.9,
                zorder=3,
                label="Points in highlighted k-means window",
        )
        axis.axvspan(
            window_start,
            window_end,
            color=cluster_color,
            alpha=highlight_alpha,
            linewidth=0,
            zorder=0,
        )
        axis.set_ylabel(measurement_column)
        axis.grid(alpha=0.25)

        twin_axis = axis.twinx()
        twin_axis.plot(
            context_panel["Time UTC"],
            context_panel[secondary_column],
            color="#059669",
            linewidth=1.2,
            alpha=0.65,
            label=secondary_column,
        )
        twin_axis.set_ylabel(secondary_column, color="#059669")
        twin_axis.tick_params(axis="y", colors="#059669")

        axis.set_title(
            f"Cluster {cluster_id} example | issue rate={float(row['issue_rate']):.2f} | "
            f"distance={float(row['distance_to_centroid']):.3f}"
        )

        legend_handles = [
            Line2D([0], [0], color="#0f172a", linewidth=2, label=measurement_column),
            Line2D([0], [0], color="#059669", linewidth=2, label=secondary_column),
            Patch(
                facecolor=cluster_color,
                edgecolor="none",
                alpha=highlight_alpha,
                label=f"Highlighted k-means window (Cluster {cluster_id})",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=cluster_color,
                markersize=6,
                label="Datapoints used inside that window",
            ),
        ]
        for flag_value in sorted(shown_flag_labels):
            legend_handles.append(
                Patch(
                    facecolor=palette.get(flag_value, "#9ca3af"),
                    edgecolor="none",
                    alpha=flag_region_alpha,
                    label=f"QC region {flag_value}: {QC_FLAG_MEANINGS.get(flag_value, 'flagged')}",
                )
            )
        axis.legend(handles=legend_handles, loc="upper left", frameon=True)

        highlighted_flags = (
            target_points[target_flag]
            .pipe(pd.to_numeric, errors="coerce")
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
            .to_dict()
        )

        example_records.append(
            {
                "cluster": cluster_id,
                "source_file": clean_source_file_label(source_file),
                "window_start": window_start,
                "window_end": window_end,
                "issue_rate": float(row["issue_rate"]),
                "distance_to_centroid": float(row["distance_to_centroid"]),
                "context_start": context_panel["Time UTC"].min(),
                "context_end": context_panel["Time UTC"].max(),
                "rows_in_highlighted_window": int(len(target_points)),
                "highlighted_window_flag_counts": highlighted_flags,
            }
        )

    axes[-1].set_xlabel("Time UTC")
    fig.suptitle("Representative time-series context for k-means windows", y=1.02)
    fig.tight_layout()
    return fig, pd.DataFrame(example_records)


def fit_kmeans(
    frame: pd.DataFrame,
    *,
    n_clusters: int,
    seed: int,
    n_init: str | int = "auto",
    feature_mode: str = "window_summary",
    feature_columns: list[str] | None = None,
    time_column: str = "Time UTC",
    source_column: str = "source_file",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit k-means on either window summaries or row-level features.

    Parameters
    ----------
    frame:
        Input dataframe used for clustering.
    feature_mode:
        `"window_summary"` clusters on `_mean`/`_std` window features.
        `"row_level"` clusters on the explicit `feature_columns`.
    feature_columns:
        Required for row-level mode. These columns become the clustering input.
    """

    if feature_mode == "window_summary":
        cluster_feature_columns = [
            column
            for column in frame.columns
            if column.endswith("_mean") or column.endswith("_std")
        ]
        if not cluster_feature_columns:
            raise ValueError("No window-summary _mean/_std columns were found for k-means")
        result = frame.copy()
        if "issue_rate" not in result.columns:
            raise ValueError("window_summary mode requires an issue_rate column")
    elif feature_mode == "row_level":
        if not feature_columns:
            raise ValueError("row_level mode requires feature_columns")
        cluster_feature_columns = [column for column in feature_columns if column in frame.columns]
        if not cluster_feature_columns:
            raise ValueError("None of the requested row-level feature columns were found for k-means")

        result = frame.copy()
        if time_column not in result.columns:
            raise ValueError(f"row_level mode requires {time_column!r}")
        if source_column not in result.columns:
            raise ValueError(f"row_level mode requires {source_column!r}")

        result["window_start"] = pd.to_datetime(result[time_column], utc=True)
        result["window_end"] = pd.to_datetime(result[time_column], utc=True)
        if "issue_rate" not in result.columns:
            if "issue" in result.columns:
                result["issue_rate"] = result["issue"].astype(float)
            else:
                raise ValueError("row_level mode requires either issue_rate or issue")
    else:
        raise ValueError(f"Unsupported k-means feature_mode: {feature_mode}")

    cluster_input = result[cluster_feature_columns]
    cluster_input = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(cluster_input),
        columns=cluster_feature_columns,
    )
    cluster_scaled = StandardScaler().fit_transform(cluster_input)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init)
    result["cluster"] = kmeans.fit_predict(cluster_scaled)
    result["distance_to_centroid"] = kmeans.transform(cluster_scaled).min(axis=1)
    summary = (
        result.groupby("cluster")
        .agg(
            window_count=("cluster", "size"),
            mean_issue_rate=("issue_rate", "mean"),
            max_issue_rate=("issue_rate", "max"),
            avg_distance=("distance_to_centroid", "mean"),
            first_window=("window_start", "min"),
            last_window=("window_end", "max"),
        )
        .sort_index()
    )
    return result, summary


def reduce_window_target(values: np.ndarray, mode: str, severity_order: tuple[int, ...] = (1, 3, 4, 9)) -> int:
    """Reduce row labels inside one window to a single label for window models."""
    labels = [int(value) for value in values if pd.notna(value)]
    if not labels:
        return severity_order[0]
    effective_order = tuple(sorted(set(severity_order).union(labels)))
    severity_rank = {label: index for index, label in enumerate(effective_order)}
    if mode == "worst":
        return max(labels, key=lambda label: severity_rank.get(label, -1))
    counts = pd.Series(labels).value_counts()
    tied_labels = counts[counts == counts.max()].index.tolist()
    return max(tied_labels, key=lambda label: severity_rank.get(int(label), -1))


@dataclass
class CnnDataBundle:
    """Normalized numpy arrays and metadata used by the notebook CNN helpers."""
    X_train: np.ndarray
    X_valid: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray
    class_labels: list[int]
    window_size: int
    feature_columns: list[str]


@dataclass
class SequenceSplitBundle:
    """Raw sequence splits before model-specific layout or normalization."""

    X_train: np.ndarray
    X_valid: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray
    class_labels: list[int]
    window_size: int
    feature_columns: list[str]
    output_mode: str


def _frame_to_fixed_windows(
    frame: pd.DataFrame,
    *,
    measurement_columns: list[str],
    target_column: str,
    window_size: int,
    time_column: str = "Time UTC",
    source_column: str = "source_file",
) -> tuple[np.ndarray, np.ndarray]:
    """Turn one split frame into fixed windows without crossing source boundaries."""

    required_columns = [*measurement_columns, target_column]
    if time_column in frame.columns:
        required_columns.append(time_column)
    if source_column in frame.columns:
        required_columns.append(source_column)

    window_frame = frame[required_columns].copy().dropna().reset_index(drop=True)
    if window_frame.empty:
        return (
            np.empty((0, window_size, len(measurement_columns)), dtype=np.float32),
            np.empty((0, window_size), dtype=np.int64),
        )

    if source_column in window_frame.columns:
        grouped_frames = (group.reset_index(drop=True) for _, group in window_frame.groupby(source_column, sort=False))
    else:
        grouped_frames = (window_frame.reset_index(drop=True),)

    raw_sequences: list[np.ndarray] = []
    raw_targets: list[np.ndarray] = []
    for source_frame in grouped_frames:
        if time_column in source_frame.columns:
            source_frame = source_frame.sort_values(time_column).reset_index(drop=True)
        usable_rows = (len(source_frame) // window_size) * window_size
        if usable_rows < window_size:
            continue
        trimmed = source_frame.iloc[:usable_rows]
        raw_sequences.append(
            trimmed[measurement_columns].to_numpy(dtype=np.float32).reshape(-1, window_size, len(measurement_columns))
        )
        raw_targets.append(trimmed[target_column].to_numpy().reshape(-1, window_size))

    if not raw_sequences:
        return (
            np.empty((0, window_size, len(measurement_columns)), dtype=np.float32),
            np.empty((0, window_size), dtype=np.int64),
        )
    return np.concatenate(raw_sequences, axis=0), np.concatenate(raw_targets, axis=0)


def build_sequence_split_bundle(
    model_df: pd.DataFrame,
    *,
    measurement_columns: list[str],
    target_column: str,
    task_mode: str,
    output_mode: str,
    window_size: int,
    train_fraction: float,
    validation_fraction: float,
    label_reduction: str,
    split_strategy: str = "global_contiguous",
    split_block_rows: int = 1024,
    time_column: str = "Time UTC",
    source_column: str = "source_file",
) -> SequenceSplitBundle:
    """Build raw fixed-window splits for CNN/transformer experiments.

    The split strategy is applied to the row-level dataframe first, and only then
    is each split converted into fixed windows. That prevents windows from
    crossing split boundaries and lets every model section share the same split
    logic.
    """

    split_frames = split_frame_by_strategy(
        model_df,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        strategy=split_strategy,
        time_column=time_column,
        source_column=source_column,
        block_rows=split_block_rows,
    )

    return build_sequence_split_bundle_from_frames(
        split_frames,
        measurement_columns=measurement_columns,
        target_column=target_column,
        task_mode=task_mode,
        output_mode=output_mode,
        window_size=window_size,
        label_reduction=label_reduction,
        time_column=time_column,
        source_column=source_column,
    )


def build_sequence_split_bundle_from_frames(
    split_frames: dict[str, pd.DataFrame],
    *,
    measurement_columns: list[str],
    target_column: str,
    task_mode: str,
    output_mode: str,
    window_size: int,
    label_reduction: str,
    time_column: str = "Time UTC",
    source_column: str = "source_file",
) -> SequenceSplitBundle:
    """Build fixed-window sequence arrays from precomputed split frames.

    This is the stricter benchmark path: the caller can define train/validation/
    test on the full reviewed dataset first, optionally subsample only the train
    split, and then turn those finalized splits into windows.
    """

    raw_train, train_targets = _frame_to_fixed_windows(
        split_frames["train"],
        measurement_columns=measurement_columns,
        target_column=target_column,
        window_size=window_size,
        time_column=time_column,
        source_column=source_column,
    )
    raw_valid, valid_targets = _frame_to_fixed_windows(
        split_frames["validation"],
        measurement_columns=measurement_columns,
        target_column=target_column,
        window_size=window_size,
        time_column=time_column,
        source_column=source_column,
    )
    raw_test, test_targets = _frame_to_fixed_windows(
        split_frames["test"],
        measurement_columns=measurement_columns,
        target_column=target_column,
        window_size=window_size,
        time_column=time_column,
        source_column=source_column,
    )

    if task_mode == "multiclass":
        if output_mode == "window":
            train_labels = np.array([reduce_window_target(row, mode=label_reduction) for row in train_targets], dtype=np.int64)
            valid_labels = np.array([reduce_window_target(row, mode=label_reduction) for row in valid_targets], dtype=np.int64)
            test_labels = np.array([reduce_window_target(row, mode=label_reduction) for row in test_targets], dtype=np.int64)
        else:
            train_labels = train_targets.astype(np.int64)
            valid_labels = valid_targets.astype(np.int64)
            test_labels = test_targets.astype(np.int64)

        active_values: list[int] = []
        for values in (train_labels, valid_labels, test_labels):
            if values.size:
                active_values.extend(np.asarray(values).reshape(-1).tolist())
        class_labels = sorted({int(value) for value in active_values})
        label_to_index = {label: index for index, label in enumerate(class_labels)}

        if output_mode == "window":
            y_train = np.array([label_to_index[label] for label in train_labels], dtype=np.int64) if len(train_labels) else np.array([], dtype=np.int64)
            y_valid = np.array([label_to_index[label] for label in valid_labels], dtype=np.int64) if len(valid_labels) else np.array([], dtype=np.int64)
            y_test = np.array([label_to_index[label] for label in test_labels], dtype=np.int64) if len(test_labels) else np.array([], dtype=np.int64)
        else:
            vectorize = np.vectorize(label_to_index.get)
            y_train = vectorize(train_labels).astype(np.int64) if train_labels.size else np.empty((0, window_size), dtype=np.int64)
            y_valid = vectorize(valid_labels).astype(np.int64) if valid_labels.size else np.empty((0, window_size), dtype=np.int64)
            y_test = vectorize(test_labels).astype(np.int64) if test_labels.size else np.empty((0, window_size), dtype=np.int64)
    else:
        class_labels = [0, 1]
        if output_mode == "window":
            y_train = train_targets.max(axis=1).astype(np.float32) if len(train_targets) else np.array([], dtype=np.float32)
            y_valid = valid_targets.max(axis=1).astype(np.float32) if len(valid_targets) else np.array([], dtype=np.float32)
            y_test = test_targets.max(axis=1).astype(np.float32) if len(test_targets) else np.array([], dtype=np.float32)
        else:
            y_train = train_targets.astype(np.float32)
            y_valid = valid_targets.astype(np.float32)
            y_test = test_targets.astype(np.float32)

    return SequenceSplitBundle(
        X_train=raw_train,
        X_valid=raw_valid,
        X_test=raw_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        class_labels=class_labels,
        window_size=window_size,
        feature_columns=measurement_columns,
        output_mode=output_mode,
    )


def build_cnn_data(
    model_df: pd.DataFrame,
    *,
    measurement_columns: list[str],
    task_mode: str,
    window_size: int,
    train_fraction: float,
    validation_fraction: float,
    label_reduction: str,
    split_strategy: str = "global_contiguous",
    split_block_rows: int = 1024,
) -> CnnDataBundle:
    """Build fixed-length windows and normalized tensors for the notebook CNN."""
    sequence_bundle = build_sequence_split_bundle(
        model_df,
        measurement_columns=measurement_columns,
        target_column="model_target",
        task_mode=task_mode,
        output_mode="window",
        window_size=window_size,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        label_reduction=label_reduction,
        split_strategy=split_strategy,
        split_block_rows=split_block_rows,
    )

    return build_cnn_data_from_sequence_bundle(
        sequence_bundle,
        measurement_columns=measurement_columns,
        window_size=window_size,
    )


def build_cnn_data_from_frames(
    split_frames: dict[str, pd.DataFrame],
    *,
    measurement_columns: list[str],
    task_mode: str,
    window_size: int,
    label_reduction: str,
) -> CnnDataBundle:
    """Build fixed-length CNN tensors from precomputed split frames."""
    sequence_bundle = build_sequence_split_bundle_from_frames(
        split_frames,
        measurement_columns=measurement_columns,
        target_column="model_target",
        task_mode=task_mode,
        output_mode="window",
        window_size=window_size,
        label_reduction=label_reduction,
    )

    return build_cnn_data_from_sequence_bundle(
        sequence_bundle,
        measurement_columns=measurement_columns,
        window_size=window_size,
    )


def build_cnn_data_from_sequence_bundle(
    sequence_bundle: SequenceSplitBundle,
    *,
    measurement_columns: list[str],
    window_size: int,
) -> CnnDataBundle:
    """Convert a raw sequence bundle into normalized CNN tensors."""

    X_train = np.transpose(sequence_bundle.X_train, (0, 2, 1))
    X_valid = np.transpose(sequence_bundle.X_valid, (0, 2, 1))
    X_test = np.transpose(sequence_bundle.X_test, (0, 2, 1))
    y_train = sequence_bundle.y_train
    y_valid = sequence_bundle.y_valid
    y_test = sequence_bundle.y_test

    if len(X_train) == 0 or len(X_valid) == 0 or len(X_test) == 0:
        return CnnDataBundle(
            X_train=X_train,
            X_valid=X_valid,
            X_test=X_test,
            y_train=y_train,
            y_valid=y_valid,
            y_test=y_test,
            class_labels=sequence_bundle.class_labels,
            window_size=window_size,
            feature_columns=measurement_columns,
        )

    # Fit normalization on the training split only to avoid information leakage.
    channel_mean = X_train.mean(axis=(0, 2), keepdims=True)
    channel_std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
    X_train = (X_train - channel_mean) / channel_std
    X_valid = (X_valid - channel_mean) / channel_std
    X_test = (X_test - channel_mean) / channel_std

    return CnnDataBundle(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        class_labels=sequence_bundle.class_labels,
        window_size=window_size,
        feature_columns=measurement_columns,
    )


def _require_torch() -> None:
    """Raise a clear error when PyTorch is unavailable in the current environment."""
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise ImportError("PyTorch is required for the CNN sections.")


def train_cnn_model(
    data: CnnDataBundle,
    *,
    task_mode: str,
    config: dict[str, object],
    seed: int,
    checkpoint_path: Path | None = None,
    device_name: str | None = None,
) -> dict[str, object]:
    """Train the baseline notebook CNN with early stopping and checkpointing."""
    _require_torch()
    torch.manual_seed(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))

    class TinyQCNet(nn.Module):
        """Minimal 1D CNN used in the teaching notebook."""
        def __init__(self, channels: int, output_dim: int) -> None:
            """Build the convolutional encoder and simple pooled prediction head."""
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(channels, config["conv_channels"][0], kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv1d(config["conv_channels"][0], config["conv_channels"][1], kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Dropout(config["dropout"]),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(config["conv_channels"][1], output_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the CNN forward pass for one batch of normalized windows."""
            return self.net(x)

    if task_mode == "multiclass":
        train_targets_tensor = torch.from_numpy(data.y_train).long()
        valid_targets_tensor = torch.from_numpy(data.y_valid).long()
        test_targets_tensor = torch.from_numpy(data.y_test).long()
        class_counts = np.bincount(data.y_train, minlength=len(data.class_labels)).clip(min=1)
        class_weights = len(data.y_train) / (len(class_counts) * class_counts)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
        output_dim = len(data.class_labels)
    else:
        train_targets_tensor = torch.from_numpy(data.y_train.astype(np.float32))
        valid_targets_tensor = torch.from_numpy(data.y_valid.astype(np.float32))
        test_targets_tensor = torch.from_numpy(data.y_test.astype(np.float32))
        positive_count = max(float(data.y_train.sum()), 1.0)
        negative_count = max(float(len(data.y_train) - data.y_train.sum()), 1.0)
        pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        output_dim = 1

    loader_kwargs: dict[str, object] = {}
    num_workers = int(config.get("num_workers", 0))
    if num_workers > 0:
        loader_kwargs["num_workers"] = num_workers
        loader_kwargs["persistent_workers"] = bool(config.get("persistent_workers", True))
        if "prefetch_factor" in config:
            loader_kwargs["prefetch_factor"] = int(config["prefetch_factor"])
    if config.get("pin_memory", torch.cuda.is_available()):
        loader_kwargs["pin_memory"] = True

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(data.X_train).float(), train_targets_tensor),
        batch_size=config["batch_size"],
        shuffle=True,
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        TensorDataset(torch.from_numpy(data.X_valid).float(), valid_targets_tensor),
        batch_size=config["batch_size"],
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(data.X_test).float(), test_targets_tensor),
        batch_size=config["batch_size"],
        shuffle=False,
        **loader_kwargs,
    )

    model = TinyQCNet(channels=len(data.feature_columns), output_dim=output_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.get("lr_decay_factor", 0.5),
        patience=config.get("lr_patience", 1),
    )

    best_metric = -math.inf
    best_epoch = 0
    patience_counter = 0
    history = []
    best_state = None

    def run_epoch(loader: DataLoader, training: bool) -> tuple[float, np.ndarray, np.ndarray]:
        """Run one epoch and collect predictions for metrics."""
        model.train(training)
        total_loss = 0.0
        predictions = []
        targets = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            with torch.set_grad_enabled(training):
                logits = model(batch_x)
                if task_mode == "multiclass":
                    loss = loss_fn(logits, batch_y)
                    batch_predictions = logits.argmax(dim=1)
                else:
                    logits = logits.squeeze(-1)
                    loss = loss_fn(logits, batch_y)
                    batch_predictions = (torch.sigmoid(logits) >= 0.5).long()
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    if config.get("gradient_clip_norm"):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
                    optimizer.step()
            total_loss += float(loss.item()) * len(batch_x)
            predictions.append(batch_predictions.detach().cpu().numpy())
            targets.append(batch_y.detach().cpu().numpy())
        predictions_array = np.concatenate(predictions) if predictions else np.array([])
        targets_array = np.concatenate(targets) if targets else np.array([])
        average_loss = total_loss / max(len(loader.dataset), 1)
        return average_loss, predictions_array, targets_array

    for epoch in range(1, config["epochs"] + 1):
        train_loss, _, _ = run_epoch(train_loader, training=True)
        valid_loss, valid_preds, valid_targets = run_epoch(valid_loader, training=False)
        valid_f1 = float(f1_score(valid_targets, valid_preds, average=report_average(task_mode), zero_division=0))
        scheduler.step(valid_f1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "valid_f1": valid_f1,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        # Keep the best checkpoint according to validation F1, not the final epoch.
        if valid_f1 > best_metric + config["min_delta"]:
            best_metric = valid_f1
            best_epoch = epoch
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "task_mode": task_mode,
                        "class_labels": data.class_labels,
                        "feature_columns": data.feature_columns,
                        "window_size": data.window_size,
                        "config": config,
                    },
                    checkpoint_path,
                )
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                break

    if best_state is None:
        raise RuntimeError("CNN training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    test_loss, test_preds, test_targets = run_epoch(test_loader, training=False)

    if task_mode == "multiclass":
        report_labels = list(range(len(data.class_labels)))
        report_names = [str(label) for label in data.class_labels]
    else:
        report_labels = [0, 1]
        report_names = ["0", "1"]

    result = {
        "history": pd.DataFrame(history),
        "best_validation_f1": best_metric,
        "best_epoch": best_epoch,
        "test_loss": float(test_loss),
        "test_predictions": test_preds,
        "test_targets": test_targets,
        "test_report_text": classification_report(
            test_targets,
            test_preds,
            labels=report_labels,
            target_names=report_names,
            zero_division=0,
        ),
        "test_confusion_matrix": confusion_matrix(
            test_targets,
            test_preds,
            labels=report_labels,
            normalize="true",
        ),
        "report_labels": report_labels,
        "report_names": report_names,
        "device": str(device),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
    }
    return result


def run_cnn_search(
    model_df: pd.DataFrame,
    *,
    measurement_columns: list[str],
    task_mode: str,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    search_space: dict[str, list[object]],
    checkpoint_dir: Path | None = None,
    split_strategy: str = "global_contiguous",
    split_block_rows: int = 1024,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    """Grid-search a compact CNN configuration space for the advanced notebook."""
    keys = list(search_space.keys())
    results = []
    best_score = -math.inf
    best_config = None
    best_result = None

    for trial_index, values in enumerate(itertools.product(*(search_space[key] for key in keys)), start=1):
        config = dict(zip(keys, values))
        data = build_cnn_data(
            model_df,
            measurement_columns=measurement_columns,
            task_mode=task_mode,
            window_size=config["window_size"],
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            label_reduction=config["label_reduction"],
            split_strategy=split_strategy,
            split_block_rows=split_block_rows,
        )
        if len(data.X_train) == 0 or len(data.X_valid) == 0 or len(data.X_test) == 0:
            results.append(
                {
                    "trial": trial_index,
                    **config,
                    "validation_f1": float("nan"),
                    "best_epoch": None,
                    "device": "skipped",
                    "skip_reason": "At least one split produced no full windows.",
                }
            )
            continue
        checkpoint_path = None
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"cnn_trial_{trial_index:02d}.pt"
        result = train_cnn_model(
            data,
            task_mode=task_mode,
            config=config,
            seed=seed,
            checkpoint_path=checkpoint_path,
        )
        row = {
            "trial": trial_index,
            **config,
            "validation_f1": result["best_validation_f1"],
            "best_epoch": result["best_epoch"],
            "device": result["device"],
        }
        results.append(row)
        if result["best_validation_f1"] > best_score:
            best_score = result["best_validation_f1"]
            best_config = config
            best_result = result

    result_frame = pd.DataFrame(results).sort_values("validation_f1", ascending=False).reset_index(drop=True)
    return result_frame, best_config, best_result


def run_cnn_search_from_frames(
    split_frames: dict[str, pd.DataFrame],
    *,
    measurement_columns: list[str],
    task_mode: str,
    seed: int,
    search_space: dict[str, list[object]],
    checkpoint_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    """Grid-search the CNN while keeping validation/test fixed from precomputed splits."""
    keys = list(search_space.keys())
    results = []
    best_score = -math.inf
    best_config = None
    best_result = None

    for trial_index, values in enumerate(itertools.product(*(search_space[key] for key in keys)), start=1):
        config = dict(zip(keys, values))
        data = build_cnn_data_from_frames(
            split_frames,
            measurement_columns=measurement_columns,
            task_mode=task_mode,
            window_size=config["window_size"],
            label_reduction=config["label_reduction"],
        )
        if len(data.X_train) == 0 or len(data.X_valid) == 0 or len(data.X_test) == 0:
            results.append(
                {
                    "trial": trial_index,
                    **config,
                    "validation_f1": float("nan"),
                    "best_epoch": None,
                    "device": "skipped",
                    "skip_reason": "At least one split produced no full windows.",
                }
            )
            continue
        checkpoint_path = None
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"cnn_trial_{trial_index:02d}.pt"
        result = train_cnn_model(
            data,
            task_mode=task_mode,
            config=config,
            seed=seed,
            checkpoint_path=checkpoint_path,
        )
        row = {
            "trial": trial_index,
            **config,
            "validation_f1": result["best_validation_f1"],
            "best_epoch": result["best_epoch"],
            "device": result["device"],
        }
        results.append(row)
        if result["best_validation_f1"] > best_score:
            best_score = result["best_validation_f1"]
            best_config = config
            best_result = result

    result_frame = pd.DataFrame(results).sort_values("validation_f1", ascending=False).reset_index(drop=True)
    return result_frame, best_config, best_result


def save_pickle(path: str | Path, payload: object) -> None:
    """Persist a Python object with pickle."""
    with Path(path).open("wb") as handle:
        pickle.dump(payload, handle)
