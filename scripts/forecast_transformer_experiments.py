from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.session1_intro_utils import derive_fractional_row_limit
from scripts.session1_modeling import (
    build_reviewed_target_frame,
    load_full_row_level_frame,
    split_frame_by_strategy,
)


DATASET_PROFILES = {
    "conductivity_plugs": {
        "cache_stem": "conductivity_plugs_session1",
        "target_flag": "ml_label",
        "task_mode": "multiclass",
        "good_labels": [0],
        "issue_labels": [1, 2, 3, 4],
        "target_column": "cond_value_ctd",
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
    },
    "ctd_conductivity": {
        "cache_stem": "ctd_session1",
        "target_flag": "Conductivity QC Flag",
        "task_mode": "multiclass",
        "good_labels": [1],
        "issue_labels": [3, 4, 9],
        "target_column": "Conductivity (S/m)",
        "measurement_columns": [
            "Conductivity (S/m)",
            "Density (kg/m3)",
            "Depth (m)",
            "Practical Salinity (psu)",
            "Pressure (decibar)",
            "Sigma-t (kg/m3)",
            "Sigma-theta (0 dbar) (kg/m3)",
            "Sound Speed (m/s)",
            "Temperature (C)",
        ],
    },
}


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    include_target_history: bool = True
    lookback: int = 96
    max_gap_seconds: float = 5.0
    train_windows: int = 20_000
    eval_windows: int = 20_000
    eval_sampling: str = "issue_enriched"
    train_sampling: str = "spread"
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.15
    epochs: int = 8
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    loss: str = "mse"
    patience: int = 3
    min_delta: float = 1e-4
    gradient_clip_norm: float = 1.0


def evenly_spaced(values: np.ndarray, limit: int | None) -> np.ndarray:
    if limit is None or len(values) <= int(limit):
        return values
    positions = np.linspace(0, len(values) - 1, num=int(limit), dtype=int)
    return values[positions]


def random_spread(values: np.ndarray, limit: int | None, rng: np.random.Generator) -> np.ndarray:
    if limit is None or len(values) <= int(limit):
        return values
    selected = rng.choice(values, size=int(limit), replace=False)
    return np.sort(selected)


def choose_candidate_ends(
    candidate_ends: np.ndarray,
    labels: np.ndarray,
    *,
    issue_labels: set[int],
    limit: int | None,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select candidate forecast endpoints while keeping the resulting windows valid."""

    if limit is None or len(candidate_ends) <= int(limit):
        return candidate_ends

    if mode == "spread":
        return evenly_spaced(candidate_ends, int(limit))
    if mode == "random":
        return random_spread(candidate_ends, int(limit), rng)
    if mode != "issue_enriched":
        raise ValueError(f"Unsupported endpoint sampling mode: {mode}")

    end_labels = labels[candidate_ends]
    issue_ends = candidate_ends[np.isin(end_labels, list(issue_labels))]
    good_ends = candidate_ends[~np.isin(end_labels, list(issue_labels))]

    # Keep all issue examples when possible, then fill the rest with good rows
    # spread over time so the threshold source still covers different regimes.
    if len(issue_ends) >= int(limit):
        return evenly_spaced(issue_ends, int(limit))
    good_budget = int(limit) - len(issue_ends)
    selected_good = evenly_spaced(good_ends, good_budget)
    return np.sort(np.concatenate([issue_ends, selected_good]))


def contiguous_true_ends(mask: np.ndarray, link_mask: np.ndarray, lookback: int) -> np.ndarray:
    """Return end indices where the previous rows are valid and locally contiguous."""

    # For an end index e, context is [e-lookback, e), and e is the target row.
    required = lookback + 1
    if len(mask) < required:
        return np.empty((0,), dtype=int)
    valid_int = mask.astype(np.int16)
    valid_window_sum = np.convolve(valid_int, np.ones(required, dtype=np.int16), mode="valid")

    # link_mask[i] says whether row i is close enough in time to row i - 1.
    # For an end index e, every link from start+1 through e must be valid.
    link_int = link_mask[1:].astype(np.int16)
    link_window_sum = np.convolve(link_int, np.ones(lookback, dtype=np.int16), mode="valid")

    valid_starts = np.flatnonzero((valid_window_sum == required) & (link_window_sum == lookback))
    return valid_starts + lookback


def build_windows_from_source_frame(
    source_frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    label_column: str,
    good_labels: set[int],
    issue_labels: set[int],
    lookback: int,
    max_gap_seconds: float,
    max_windows: int | None,
    require_good_window: bool,
    sampling_mode: str,
    seed: int,
) -> dict[str, object]:
    """Build selected next-value forecast windows from one already-filtered source file."""

    source_frame = source_frame.sort_values("Time UTC").reset_index(drop=True)
    if len(source_frame) <= lookback:
        return empty_window_bundle(lookback, len(feature_columns))

    values = source_frame[feature_columns].to_numpy(dtype=np.float32)
    target_values = source_frame[target_column].to_numpy(dtype=np.float32)
    labels = source_frame[label_column].astype(int).to_numpy()
    time_values = pd.to_datetime(source_frame["Time UTC"], utc=True)
    time_gaps = time_values.diff().dt.total_seconds().to_numpy()
    link_mask = np.ones(len(source_frame), dtype=bool)
    link_mask[1:] = np.isfinite(time_gaps[1:]) & (time_gaps[1:] <= float(max_gap_seconds))
    finite_rows = np.isfinite(values).all(axis=1) & np.isfinite(target_values)
    if require_good_window:
        valid_rows = finite_rows & np.isin(labels, list(good_labels))
    else:
        valid_rows = finite_rows

    candidate_ends = contiguous_true_ends(valid_rows, link_mask, lookback)
    rng = np.random.default_rng(seed)
    selected_ends = choose_candidate_ends(
        candidate_ends,
        labels,
        issue_labels=issue_labels,
        limit=max_windows,
        mode=sampling_mode,
        rng=rng,
    )
    if len(selected_ends) == 0:
        return empty_window_bundle(lookback, len(feature_columns))

    X = np.empty((len(selected_ends), lookback, len(feature_columns)), dtype=np.float32)
    y = np.empty((len(selected_ends),), dtype=np.float32)
    metadata_rows = []
    for out_index, end_index in enumerate(selected_ends.tolist()):
        start_index = end_index - lookback
        X[out_index] = values[start_index:end_index]
        y[out_index] = target_values[end_index]
        label = int(labels[end_index])
        metadata_rows.append(
            {
                "Time UTC": source_frame.loc[end_index, "Time UTC"],
                "source_file": source_frame.loc[end_index, "source_file"],
                "target_value": float(target_values[end_index]),
                "qc_label": label,
                "is_good": label in good_labels,
                "is_issue": label in issue_labels,
            }
        )

    return {"X": X, "y": y, "metadata": pd.DataFrame(metadata_rows)}


def empty_window_bundle(lookback: int, feature_count: int) -> dict[str, object]:
    return {
        "X": np.empty((0, lookback, feature_count), dtype=np.float32),
        "y": np.empty((0,), dtype=np.float32),
        "metadata": pd.DataFrame(columns=["Time UTC", "source_file", "target_value", "qc_label", "is_good", "is_issue"]),
    }


def concat_bundles(bundles: list[dict[str, object]], *, max_windows: int | None = None) -> dict[str, object]:
    nonempty = [bundle for bundle in bundles if len(bundle["X"]) > 0]
    if not nonempty:
        if bundles:
            first = bundles[0]
            return empty_window_bundle(first["X"].shape[1], first["X"].shape[2])
        return empty_window_bundle(1, 1)
    X = np.concatenate([bundle["X"] for bundle in nonempty], axis=0)
    y = np.concatenate([bundle["y"] for bundle in nonempty], axis=0)
    metadata = pd.concat([bundle["metadata"] for bundle in nonempty], ignore_index=True)
    if max_windows is not None and len(X) > int(max_windows):
        keep = evenly_spaced(np.arange(len(X)), int(max_windows))
        X = X[keep]
        y = y[keep]
        metadata = metadata.iloc[keep].reset_index(drop=True)
    else:
        metadata = metadata.reset_index(drop=True)
    return {"X": X, "y": y, "metadata": metadata}


class NextValueTransformer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        lookback: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, lookback, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = x + self.position_embedding[:, : x.size(1)]
        x = self.encoder(x)
        x = self.norm(x[:, -1, :])
        return self.head(x).squeeze(-1)


def score_model(
    model: nn.Module,
    bundle: dict[str, object],
    *,
    batch_size: int,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> pd.DataFrame:
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(bundle["X_scaled"])),
        torch.from_numpy(bundle["y_scaled"]),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    predictions = []
    targets = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=device.type == "cuda")
            predictions.append(model(batch_x).detach().cpu().numpy())
            targets.append(batch_y.detach().cpu().numpy())
    predicted_scaled = np.concatenate(predictions) if predictions else np.array([], dtype=np.float32)
    target_scaled = np.concatenate(targets) if targets else np.array([], dtype=np.float32)
    result = bundle["metadata"].reset_index(drop=True).copy()
    result["predicted_value"] = predicted_scaled * target_std + target_mean
    result["actual_value"] = target_scaled * target_std + target_mean
    result["forecast_error"] = result["actual_value"] - result["predicted_value"]
    result["forecast_abs_error_scaled"] = np.abs(target_scaled - predicted_scaled)
    return result


def threshold_metrics(scores: pd.DataFrame, threshold: float, *, split: str) -> dict[str, object]:
    y_true = scores["is_issue"].astype(int).to_numpy()
    y_pred = (scores["forecast_abs_error_scaled"].to_numpy() >= threshold).astype(int)
    row = {
        "split": split,
        "threshold": float(threshold),
        "rows": int(len(scores)),
        "issue_rows": int(y_true.sum()),
        "predicted_anomaly_rows": int(y_pred.sum()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "median_abs_error_scaled": float(scores["forecast_abs_error_scaled"].median()),
        "p95_abs_error_scaled": float(scores["forecast_abs_error_scaled"].quantile(0.95)),
    }
    for label in sorted(scores.loc[scores["is_issue"], "qc_label"].dropna().astype(int).unique().tolist()):
        label_mask = scores["qc_label"].astype(int).to_numpy() == int(label)
        row[f"label_{label}_rows"] = int(label_mask.sum())
        row[f"label_{label}_recall"] = float(y_pred[label_mask].mean()) if label_mask.any() else 0.0
    return row


def threshold_sweep(valid_scores: pd.DataFrame, test_scores: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    good_valid = valid_scores.loc[valid_scores["is_good"], "forecast_abs_error_scaled"]
    if good_valid.empty:
        good_valid = valid_scores["forecast_abs_error_scaled"]
    quantiles = [0.90, 0.95, 0.97, 0.98, 0.99, 0.995]
    rows = []
    thresholds = {}
    for quantile in quantiles:
        threshold = float(good_valid.quantile(quantile))
        thresholds[f"good_q{quantile:g}"] = threshold
        rows.append({"threshold_source": f"good_q{quantile:g}", **threshold_metrics(valid_scores, threshold, split="validation")})
        rows.append({"threshold_source": f"good_q{quantile:g}", **threshold_metrics(test_scores, threshold, split="test")})

    candidate_thresholds = np.unique(np.quantile(valid_scores["forecast_abs_error_scaled"], np.linspace(0.50, 0.999, 250)))
    best_threshold = float(candidate_thresholds[0])
    best_f1 = -1.0
    for threshold in candidate_thresholds:
        value = threshold_metrics(valid_scores, float(threshold), split="validation")["f1"]
        if value > best_f1:
            best_f1 = value
            best_threshold = float(threshold)
    thresholds["validation_f1_optimized"] = best_threshold
    rows.append({"threshold_source": "validation_f1_optimized", **threshold_metrics(valid_scores, best_threshold, split="validation")})
    rows.append({"threshold_source": "validation_f1_optimized", **threshold_metrics(test_scores, best_threshold, split="test")})
    return pd.DataFrame(rows), thresholds


def ranking_metrics(scores: pd.DataFrame, *, split: str) -> dict[str, object]:
    y_true = scores["is_issue"].astype(int).to_numpy()
    y_score = scores["forecast_abs_error_scaled"].to_numpy()
    result = {
        "split": split,
        "rows": int(len(scores)),
        "issue_rows": int(y_true.sum()),
        "average_precision": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
    }
    return result


def load_split_feature_frames(
    *,
    part_paths: list[Path],
    split_frames: dict[str, pd.DataFrame],
    columns: list[str],
) -> dict[str, pd.DataFrame]:
    """Load split rows with feature columns while preserving real local adjacency."""

    loaded_by_split: dict[str, list[pd.DataFrame]] = {name: [] for name in split_frames}
    key_columns = ["source_file", "Time UTC"]
    split_keys = {
        name: frame[key_columns + ["model_target", "issue"]].copy()
        for name, frame in split_frames.items()
    }
    for name, frame in split_keys.items():
        frame["Time UTC"] = pd.to_datetime(frame["Time UTC"], utc=True)

    for path in part_paths:
        source_frame = pd.read_parquet(path, columns=columns)
        source_frame["Time UTC"] = pd.to_datetime(source_frame["Time UTC"], utc=True)
        for column in columns:
            if column not in {"Time UTC", "source_file"}:
                source_frame[column] = pd.to_numeric(source_frame[column], errors="coerce")
        if "source_file" not in source_frame.columns:
            source_frame["source_file"] = path.name

        for split_name, keys in split_keys.items():
            relevant = keys[keys["source_file"].isin(source_frame["source_file"].unique())]
            if relevant.empty:
                continue
            merged = source_frame.merge(relevant, on=key_columns, how="inner", sort=False)
            if len(merged):
                loaded_by_split[split_name].append(merged.sort_values(["source_file", "Time UTC"]).reset_index(drop=True))
        del source_frame

    return {
        split_name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
        for split_name, frames in loaded_by_split.items()
    }


def build_forecast_bundles(
    *,
    split_feature_frames: dict[str, pd.DataFrame],
    config: ExperimentConfig,
    feature_columns: list[str],
    target_column: str,
    label_column: str,
    good_labels: set[int],
    issue_labels: set[int],
    seed: int,
) -> dict[str, dict[str, object]]:
    bundles = {}
    split_settings = {
        "train": ("train", config.train_windows, True, config.train_sampling),
        # Model selection should evaluate normal behaviour only. We keep this
        # separate from the issue-enriched validation set used for anomaly scoring.
        "validation_good": ("validation", config.eval_windows, True, "spread"),
        "validation": ("validation", config.eval_windows, False, config.eval_sampling),
        "test": ("test", config.eval_windows, False, config.eval_sampling),
    }
    for bundle_name, (split_name, limit, require_good_window, sampling_mode) in split_settings.items():
        source_bundles = []
        frame = split_feature_frames[split_name]
        for source_index, (_, source_frame) in enumerate(frame.groupby("source_file", sort=False, observed=False)):
            source_bundles.append(
                build_windows_from_source_frame(
                    source_frame,
                    feature_columns=feature_columns,
                    target_column=target_column,
                    label_column=label_column,
                    good_labels=good_labels,
                    issue_labels=issue_labels,
                    lookback=config.lookback,
                    max_gap_seconds=config.max_gap_seconds,
                    max_windows=None if limit is None else max(1, math.ceil(int(limit) / max(frame["source_file"].nunique(), 1))),
                    require_good_window=require_good_window,
                    sampling_mode=sampling_mode,
                    seed=seed + source_index,
                )
            )
        bundles[bundle_name] = concat_bundles(source_bundles, max_windows=limit)
    return bundles


def train_one_experiment(
    *,
    config: ExperimentConfig,
    split_feature_frames: dict[str, pd.DataFrame],
    profile: dict[str, object],
    seed: int,
    device: torch.device,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision("high")

    target_column = str(profile["target_column"])
    feature_columns = list(profile["measurement_columns"])
    if not config.include_target_history:
        feature_columns = [column for column in feature_columns if column != target_column]
    if not feature_columns:
        raise ValueError("No feature columns remain after applying include_target_history=False.")

    start = time.perf_counter()
    bundles = build_forecast_bundles(
        split_feature_frames=split_feature_frames,
        config=config,
        feature_columns=feature_columns,
        target_column=target_column,
        label_column="model_target",
        good_labels=set(int(label) for label in profile["good_labels"]),
        issue_labels=set(int(label) for label in profile["issue_labels"]),
        seed=seed,
    )
    window_seconds = time.perf_counter() - start

    if (
        len(bundles["train"]["X"]) == 0
        or len(bundles["validation_good"]["X"]) == 0
        or len(bundles["validation"]["X"]) == 0
        or len(bundles["test"]["X"]) == 0
    ):
        raise RuntimeError(f"{config.name} produced an empty train/validation/test bundle.")

    feature_mean = bundles["train"]["X"].mean(axis=(0, 1), keepdims=True)
    feature_std = bundles["train"]["X"].std(axis=(0, 1), keepdims=True) + 1e-6
    target_mean = float(bundles["train"]["y"].mean())
    target_std = float(bundles["train"]["y"].std() + 1e-6)
    for bundle in bundles.values():
        bundle["X_scaled"] = ((bundle["X"] - feature_mean) / feature_std).astype(np.float32, copy=False)
        bundle["y_scaled"] = ((bundle["y"] - target_mean) / target_std).astype(np.float32, copy=False)

    train_dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(bundles["train"]["X_scaled"])),
        torch.from_numpy(bundles["train"]["y_scaled"]),
    )
    valid_dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(bundles["validation_good"]["X_scaled"])),
        torch.from_numpy(bundles["validation_good"]["y_scaled"]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    model = NextValueTransformer(
        input_dim=len(feature_columns),
        lookback=config.lookback,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    ).to(device)
    loss_fn: nn.Module
    if config.loss == "huber":
        loss_fn = nn.SmoothL1Loss(beta=0.5)
    elif config.loss == "mae":
        loss_fn = nn.L1Loss()
    else:
        loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_state = None
    best_loss = np.inf
    best_epoch = 0
    patience_counter = 0
    history_rows = []

    def run_epoch(loader: DataLoader, training: bool) -> float:
        model.train(training)
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=device.type == "cuda")
            batch_y = batch_y.to(device, non_blocking=device.type == "cuda")
            with torch.set_grad_enabled(training):
                prediction = model(batch_x)
                loss = loss_fn(prediction, batch_y)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if config.gradient_clip_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                    optimizer.step()
            total_loss += float(loss.item()) * len(batch_x)
        return total_loss / max(len(loader.dataset), 1)

    fit_start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(train_loader, training=True)
        valid_loss = run_epoch(valid_loader, training=False)
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        print(f"{config.name}: epoch {epoch:02d} train={train_loss:.5f} valid={valid_loss:.5f}", flush=True)
        if valid_loss < best_loss - config.min_delta:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break
    fit_seconds = time.perf_counter() - fit_start
    if best_state is not None:
        model.load_state_dict(best_state)

    valid_scores = score_model(
        model,
        bundles["validation"],
        batch_size=config.batch_size,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )
    test_scores = score_model(
        model,
        bundles["test"],
        batch_size=config.batch_size,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )
    threshold_frame, thresholds = threshold_sweep(valid_scores, test_scores)
    ranking_frame = pd.DataFrame([ranking_metrics(valid_scores, split="validation"), ranking_metrics(test_scores, split="test")])

    summary = {
        **asdict(config),
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "train_windows_actual": int(len(bundles["train"]["X"])),
        "validation_good_windows_actual": int(len(bundles["validation_good"]["X"])),
        "validation_windows_actual": int(len(bundles["validation"]["X"])),
        "test_windows_actual": int(len(bundles["test"]["X"])),
        "validation_issue_rows": int(valid_scores["is_issue"].sum()),
        "test_issue_rows": int(test_scores["is_issue"].sum()),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "window_seconds": float(window_seconds),
        "fit_seconds": float(fit_seconds),
        "thresholds": thresholds,
        "ranking": ranking_frame.to_dict(orient="records"),
        "history": history_rows,
    }
    return summary, threshold_frame, pd.concat(
        [
            valid_scores.assign(experiment=config.name, split="validation"),
            test_scores.assign(experiment=config.name, split="test"),
        ],
        ignore_index=True,
    )


def build_experiment_configs(preset: str) -> list[ExperimentConfig]:
    if preset == "quick":
        return [
            ExperimentConfig(name="baseline_target_history_20k"),
            ExperimentConfig(name="exogenous_only_20k", include_target_history=False),
            ExperimentConfig(name="target_history_100k", train_windows=100_000, eval_windows=30_000, epochs=10),
            ExperimentConfig(name="exogenous_only_100k", include_target_history=False, train_windows=100_000, eval_windows=30_000, epochs=10),
            ExperimentConfig(name="exogenous_huber_100k", include_target_history=False, train_windows=100_000, eval_windows=30_000, epochs=10, loss="huber"),
        ]
    if preset == "extended":
        return [
            ExperimentConfig(name="target_history_20k"),
            ExperimentConfig(name="target_history_100k", train_windows=100_000, eval_windows=30_000, epochs=10),
            ExperimentConfig(name="target_history_200k", train_windows=200_000, eval_windows=50_000, epochs=10),
            ExperimentConfig(name="exogenous_only_20k", include_target_history=False),
            ExperimentConfig(name="exogenous_only_100k", include_target_history=False, train_windows=100_000, eval_windows=30_000, epochs=10),
            ExperimentConfig(name="exogenous_huber_100k", include_target_history=False, train_windows=100_000, eval_windows=30_000, epochs=10, loss="huber"),
            ExperimentConfig(name="exogenous_bigger_100k", include_target_history=False, train_windows=100_000, eval_windows=30_000, d_model=96, nhead=4, num_layers=3, dim_feedforward=256, epochs=10),
            ExperimentConfig(name="exogenous_lookback192_100k", include_target_history=False, lookback=192, train_windows=100_000, eval_windows=30_000, epochs=10),
        ]
    raise ValueError(f"Unsupported preset: {preset}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local next-value transformer anomaly experiments.")
    parser.add_argument("--dataset", choices=sorted(DATASET_PROFILES), default="conductivity_plugs")
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/session1"))
    parser.add_argument("--data-fraction", type=float, default=0.9)
    parser.add_argument("--preset", choices=["quick", "extended"], default="quick")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/forecast_transformer_experiments"))
    args = parser.parse_args()

    profile = DATASET_PROFILES[args.dataset]
    row_dir = args.cache_root / f"{profile['cache_stem']}_row_level"
    part_paths = sorted(row_dir.glob("*.parquet"))
    if not part_paths:
        raise FileNotFoundError(f"No parquet parts found in {row_dir}")

    label_columns = ["Time UTC", "source_file", profile["target_flag"]]
    print(f"Loading label columns from {len(part_paths)} parquet parts...", flush=True)
    label_frame = load_full_row_level_frame(part_paths, columns=label_columns)
    label_frame[profile["target_flag"]] = pd.to_numeric(label_frame[profile["target_flag"]], errors="coerce")
    if "source_file" in label_frame.columns:
        label_frame["source_file"] = label_frame["source_file"].astype("category")

    reviewed_mask_count = int(
        label_frame[profile["target_flag"]]
        .dropna()
        .astype(int)
        .isin([*profile["good_labels"], *profile["issue_labels"]])
        .sum()
    )
    model_row_limit = derive_fractional_row_limit(reviewed_mask_count, args.data_fraction)
    reviewed_model_df, active_labels = build_reviewed_target_frame(
        label_frame,
        target_flag=profile["target_flag"],
        task_mode=profile["task_mode"],
        good_labels=profile["good_labels"],
        issue_labels=profile["issue_labels"],
        model_row_limit=model_row_limit,
    )
    del label_frame

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "data_fraction": args.data_fraction,
                "reviewed_rows_before_limit": reviewed_mask_count,
                "reviewed_rows_after_limit": len(reviewed_model_df),
                "active_labels": active_labels,
            },
            indent=2,
        ),
        flush=True,
    )

    split_frames = split_frame_by_strategy(
        reviewed_model_df,
        train_fraction=0.70,
        validation_fraction=0.15,
        strategy="episode_aware",
        block_rows=1024,
        target_column="model_target",
        issue_column="issue",
        issue_labels=profile["issue_labels"],
        episode_context_rows=512,
        episode_merge_gap_rows=128,
        purge_gap_rows=256,
    )
    print(
        json.dumps(
            {
                split: {
                    "rows": int(len(frame)),
                    "issue_rows": int(frame["issue"].sum()),
                    "issue_share_pct": round(100 * float(frame["issue"].mean()), 4) if len(frame) else 0.0,
                }
                for split, frame in split_frames.items()
            },
            indent=2,
        ),
        flush=True,
    )

    columns = list(dict.fromkeys(["Time UTC", "source_file", profile["target_flag"], *profile["measurement_columns"]]))
    print("Loading feature rows for split frames...", flush=True)
    split_feature_frames = load_split_feature_frames(part_paths=part_paths, split_frames=split_frames, columns=columns)
    del reviewed_model_df
    for split, frame in split_feature_frames.items():
        frame["model_target"] = pd.to_numeric(frame["model_target"], errors="coerce").astype(int)
        frame["issue"] = pd.to_numeric(frame["issue"], errors="coerce").astype(int)
        print(
            f"{split}: loaded {len(frame):,} rows, issue rows={int(frame['issue'].sum()):,}",
            flush=True,
        )

    output_dir = args.output_dir / f"{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    summaries = []
    threshold_frames = []
    for config in build_experiment_configs(args.preset):
        print(f"\n=== {config.name} ===", flush=True)
        summary, threshold_frame, scores = train_one_experiment(
            config=config,
            split_feature_frames=split_feature_frames,
            profile=profile,
            seed=args.seed,
            device=device,
        )
        summaries.append(summary)
        threshold_frame.insert(0, "experiment", config.name)
        threshold_frames.append(threshold_frame)
        scores.to_parquet(output_dir / f"{config.name}_scores.parquet", index=False)
        print(json.dumps({k: v for k, v in summary.items() if k not in {"history", "thresholds", "ranking", "feature_columns"}}, indent=2), flush=True)

    summary_path = output_dir / "summary.json"
    threshold_path = output_dir / "threshold_metrics.csv"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    pd.concat(threshold_frames, ignore_index=True).to_csv(threshold_path, index=False)
    print(json.dumps({"summary": str(summary_path), "threshold_metrics": str(threshold_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
