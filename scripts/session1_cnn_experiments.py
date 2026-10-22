"""Window-size, input, and target comparisons for the per-timestamp CNN in the advanced Session 1 notebook.

Every variant cuts its windows from the same training rows and is scored on the
same validation and test rows, so differences come from the window size, the
input features, or the target definition alone. Scores use the same definitions
as ``session1_score_notes``: macro F1 over the labels present, and issue-vs-good
F1 on the original labels with the missing-data label left out.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd

from .session1_modeling import (
    add_indicator_channels,
    add_temporal_context_features,
    apply_target_strategy,
    build_classification_loss,
    build_sequence_split_bundle_from_frames,
    materialize_reviewed_split_frames,
    sample_frame_by_strategy,
    time_gap_segment_ids,
)
from .session1_score_notes import SCORE_COLUMNS, split_scores, target_strategy_labels
from .session1_tree_experiments import _good_and_missing, _raw_issue_f1

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - PyTorch is optional until the CNN sections
    torch = None
    nn = None

FEATURE_SETS = ("measurements", "measurements_plus_deltas", "measurements_plus_deltas_plus_rolling_time")
TIME_FEATURE_COLUMNS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]


def add_sequence_features(frame: pd.DataFrame, measurement_columns: list[str]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Add the per-timestamp CNN's engineered inputs and return the columns of each feature set.

    The inputs match the per-timestamp CNN section: ``abs_delta`` changes, a 5-step
    rolling mean and standard deviation, and cyclical hour and day-of-year values.
    Each source file is handled on its own, and changes and rolling windows restart
    after time gaps. The first reading of a stretch has no spread yet, so its rolling
    standard deviation is 0 rather than missing; every feature set then keeps the same rows.
    """
    parts, context_columns = [], []
    groups = frame.groupby("source_file", sort=False) if "source_file" in frame.columns else [(None, frame)]
    for _, part in groups:
        part = part.sort_values("Time UTC").reset_index(drop=True)
        segment_ids = time_gap_segment_ids(part)
        for column in measurement_columns:
            part[f"{column} abs_delta"] = part[column].groupby(segment_ids).diff().abs().fillna(0.0)
        context_frame, context_columns = add_temporal_context_features(
            part[["Time UTC", *measurement_columns]], measurement_columns=measurement_columns, lag_steps=(), rolling_windows=(5,)
        )
        for column in context_columns:
            part[column] = context_frame[column].to_numpy()
            if " roll_std_" in column:
                part[column] = part[column].fillna(0.0)
        parts.append(part)
    work = pd.concat(parts, ignore_index=True)

    radians_hour = 2.0 * np.pi * work["Time UTC"].dt.hour / 24.0
    radians_doy = 2.0 * np.pi * work["Time UTC"].dt.dayofyear / 366.0
    work["hour_sin"], work["hour_cos"] = np.sin(radians_hour), np.cos(radians_hour)
    work["doy_sin"], work["doy_cos"] = np.sin(radians_doy), np.cos(radians_doy)

    deltas = [f"{column} abs_delta" for column in measurement_columns]
    feature_sets = {
        "measurements": list(measurement_columns),
        "measurements_plus_deltas": [*measurement_columns, *deltas],
        "measurements_plus_deltas_plus_rolling_time": [*measurement_columns, *deltas, *context_columns, *TIME_FEATURE_COLUMNS],
    }
    return work, feature_sets


def build_sequence_comparison_frames(
    ml_state: dict,
    *,
    train_rows: int,
    eval_rows: int,
    block_rows: int,
    indicator_channels=None,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    """Load one set of train/validation/test rows with every per-timestamp CNN input.

    Training rows follow the active training-subset strategy ("time_spread" for
    "full_train"); validation and test rows are evenly spaced blocks. The preset's
    0/1 range-check channels (``indicator_channels``) are added to every feature set.
    Returns the frames and the input columns of each feature set.
    """
    flag = ml_state["TARGET_FLAG"]
    labels = {"good_labels": ml_state["GOOD_LABELS"], "issue_labels": ml_state["ISSUE_LABELS"]}
    train_strategy = ml_state["TRAIN_SUBSET_STRATEGY"] if ml_state["TRAIN_SUBSET_STRATEGY"] != "full_train" else "time_spread"
    selections = {
        "train": sample_frame_by_strategy(
            ml_state["train_full_df"], rows_limit=train_rows, sample_strategy=train_strategy, target_flag=flag,
            issue_rows=ml_state["ISSUE_ROWS_PER_FILE"], balanced_issue_share=ml_state["BALANCED_REVIEWED_TARGET_ISSUE_SHARE"],
            block_rows=block_rows, **labels,
        ),
        "validation": sample_frame_by_strategy(
            ml_state["valid_df"], rows_limit=eval_rows, sample_strategy="time_spread", target_flag=flag, block_rows=block_rows,
        ),
        "test": sample_frame_by_strategy(
            ml_state["test_df"], rows_limit=eval_rows, sample_strategy="time_spread", target_flag=flag, block_rows=block_rows,
        ),
    }
    rows = materialize_reviewed_split_frames(
        ml_state["selected_paths"], selections, columns=ml_state["ROW_USE_COLUMNS"], target_flag=flag,
        task_mode=ml_state["task_mode"], **labels,
    )
    frames = {}
    for split_name, frame in rows.items():
        frames[split_name], feature_sets = add_sequence_features(frame, ml_state["measurement_columns"])
        frames[split_name], indicator_columns = add_indicator_channels(frames[split_name], indicator_channels)
    return frames, {name: columns + indicator_columns for name, columns in feature_sets.items()}


if nn is not None:

    class SequenceLabelCNN(nn.Module):
        """The per-timestamp CNN of the advanced notebook: Conv1D blocks, then one class score per time step."""

        def __init__(self, channels: int, output_dim: int, conv_channels=(64, 128, 256), dropout: float = 0.15) -> None:
            super().__init__()
            blocks, in_channels = [], channels
            for index, out_channels in enumerate(conv_channels):
                kernel_size = (7, 5, 3)[min(index, 2)]
                blocks += [
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
                in_channels = out_channels
            self.features = nn.Sequential(*blocks)
            self.head = nn.Conv1d(in_channels, output_dim, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.features(x))


def _strategy_mapping(raw_labels, strategy: str, issue_labels) -> tuple[dict[int, int], str]:
    """Map each original label to its label under ``strategy`` (see ``apply_target_strategy``)."""
    frame = pd.DataFrame({"label": sorted({int(label) for label in raw_labels})})
    mapped, _, average = apply_target_strategy(frame, "label", strategy, issue_labels=issue_labels)
    unmapped = sorted(set(frame["label"]) - set(mapped["label"]))
    if unmapped:
        raise ValueError(f"Target {strategy!r} has no label for {unmapped} in this dataset; remove it from the list.")
    return dict(zip(mapped["label"].astype(int), mapped["strategy_target"].astype(int))), average


def _predict(model, X: np.ndarray, *, binary: bool, device, batch_size: int) -> np.ndarray:
    """Return one predicted class index (or 0/1) per time step, shaped like the windows."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            logits = model(torch.from_numpy(X[start : start + batch_size]).to(device))
            if binary:
                predictions.append((torch.sigmoid(logits.squeeze(1)) >= 0.5).long().cpu().numpy())
            else:
                predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def compare_sequence_cnns(
    frames: dict[str, pd.DataFrame],
    feature_columns: dict[str, list[str]],
    *,
    window_sizes,
    feature_sets,
    target_strategies,
    ml_state: dict,
    epochs: int,
    loss: str,
    seed: int,
    batch_rows: int = 4096,
    learning_rate: float = 5e-4,
) -> pd.DataFrame:
    """Train one per-timestamp CNN per window size, feature set, and target; one result row each.

    Every variant uses the same network, optimiser, loss option, and seed, and keeps
    the epoch with the best validation macro F1. Each training batch holds about
    ``batch_rows`` time steps (``batch_rows // window_size`` windows), so every window
    size gets the same number of weight updates per epoch. Macro F1 is computed over each
    target's own labels, so it rises when classes are merged. Issue-vs-good F1 is
    computed on the original labels for every target, so it can be compared across
    target definitions.
    """
    if torch is None:
        raise ImportError("PyTorch is required for the CNN comparison.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    good, missing = _good_and_missing(ml_state)
    all_columns = list(dict.fromkeys(column for name in feature_sets for column in feature_columns[name]))
    rows = []
    for window_size in window_sizes:
        # Cut windows once per size from every input column, so all feature sets and targets share them.
        bundle = build_sequence_split_bundle_from_frames(
            frames, measurement_columns=all_columns, target_column="model_target", task_mode="multiclass",
            output_mode="per_timestep", window_size=window_size, label_reduction="worst",
        )
        windows = {"train": bundle.X_train, "validation": bundle.X_valid, "test": bundle.X_test}
        if min(len(values) for values in windows.values()) == 0:
            print(f"Window size {window_size} skipped: at least one split has no full windows of {window_size} rows.")
            continue
        batch_size = max(1, batch_rows // window_size)
        raw = {
            split_name: np.asarray(bundle.class_labels, dtype=int)[values]
            for split_name, values in (("train", bundle.y_train), ("validation", bundle.y_valid), ("test", bundle.y_test))
        }
        for feature_set in feature_sets:
            channel_indexes = [all_columns.index(column) for column in feature_columns[feature_set]]
            # Channel-first windows, scaled with the training windows' mean and spread.
            inputs = {name: np.transpose(values[:, :, channel_indexes], (0, 2, 1)) for name, values in windows.items()}
            mean = inputs["train"].mean(axis=(0, 2), keepdims=True)
            std = inputs["train"].std(axis=(0, 2), keepdims=True) + 1e-6
            inputs = {name: ((values - mean) / std).astype(np.float32) for name, values in inputs.items()}
            for strategy in target_strategies:
                row = {
                    "window_size": window_size, "features": feature_set, "target": strategy,
                    "channels": len(channel_indexes), "train_windows": len(inputs["train"]),
                }
                mapping, average = _strategy_mapping(
                    np.concatenate([values.ravel() for values in raw.values()]), strategy, ml_state["ISSUE_LABELS"]
                )
                binary = average == "binary"
                targets = {name: np.vectorize(mapping.get)(values) for name, values in raw.items()}
                class_labels = [0, 1] if binary else sorted(set().union(*(set(np.unique(values).tolist()) for values in targets.values())))
                to_index = {label: index for index, label in enumerate(class_labels)}
                train_y = np.vectorize(to_index.get)(targets["train"])
                strategy_good, strategy_missing = ([0], []) if binary else target_strategy_labels(strategy, good, missing)

                torch.manual_seed(seed)
                model = SequenceLabelCNN(len(channel_indexes), 1 if binary else len(class_labels)).to(device)
                loss_fn, _ = build_classification_loss(
                    train_y, task_mode="binary" if binary else "multiclass", class_labels=class_labels, loss_name=loss, device=device
                )
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
                loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(inputs["train"]),
                        torch.from_numpy(train_y.astype(np.float32) if binary else train_y.astype(np.int64)),
                    ),
                    batch_size=batch_size,
                    shuffle=True,
                )

                started = time.time()
                best_valid, best_epoch, best_state = -1.0, 0, None
                for epoch in range(1, epochs + 1):
                    model.train()
                    for batch_x, batch_y in loader:
                        logits = model(batch_x.to(device))
                        batch_loss = loss_fn(logits.squeeze(1) if binary else logits, batch_y.to(device))
                        optimizer.zero_grad(set_to_none=True)
                        batch_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    predicted = _predict(model, inputs["validation"], binary=binary, device=device, batch_size=batch_size)
                    valid_f1 = split_scores(
                        targets["validation"].ravel(), np.asarray(class_labels)[predicted].ravel(),
                        labels=class_labels, good=strategy_good, missing=strategy_missing,
                    )[SCORE_COLUMNS["present"]]
                    if valid_f1 > best_valid:
                        best_valid, best_epoch, best_state = valid_f1, epoch, copy.deepcopy(model.state_dict())
                model.load_state_dict(best_state)
                row.update({"best_epoch": best_epoch, "fit_seconds": round(time.time() - started, 1)})

                for split_name in ("validation", "test"):
                    predicted = _predict(model, inputs[split_name], binary=binary, device=device, batch_size=batch_size)
                    predicted = np.asarray(class_labels)[predicted].ravel()
                    scores = split_scores(
                        targets[split_name].ravel(), predicted, labels=class_labels, good=strategy_good, missing=strategy_missing
                    )
                    row[f"{split_name} macro F1 (labels present)"] = round(float(scores[SCORE_COLUMNS["present"]]), 3)
                    row[f"{split_name} issue-vs-good F1"] = round(
                        _raw_issue_f1(raw[split_name].ravel(), ~np.isin(predicted, strategy_good), good=good, missing=missing), 3
                    )
                rows.append(row)
                del model, optimizer, loader, best_state
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    return pd.DataFrame(rows)
