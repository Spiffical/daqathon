"""Tree-family, feature-set, and target comparisons for the advanced Session 1 notebook.

Every variant trains on the same training rows and is scored on the same
validation and test rows, so differences come from the model family, the input
features, or the target definition alone. Scores use the same definitions as
``session1_score_notes``: macro F1 over the labels present, and issue-vs-good F1
with the missing-data label left out.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from .session1_modeling import (
    add_temporal_context_features,
    apply_target_strategy,
    build_model_frame,
    materialize_reviewed_split_frames,
    sample_frame_by_strategy,
)
from .session1_score_notes import SCORE_COLUMNS, split_scores, target_strategy_labels

TREE_FAMILIES = ("random_forest", "extra_trees", "gradient_boosting")
FEATURE_SETS = ("base", "base_plus_context")


def make_tree_classifier(family: str, *, seed: int, n_estimators: int = 200) -> Pipeline:
    """Return an imputer plus one tree-ensemble classifier.

    - ``random_forest``: each tree sees a bootstrap sample and picks the best split.
    - ``extra_trees``: each split threshold is drawn at random, then the best of those is kept.
    - ``gradient_boosting``: trees are built one after another, each correcting the previous errors.
    """
    if family == "random_forest":
        model = RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
        )
    elif family == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=n_estimators, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
        )
    elif family == "gradient_boosting":
        # Built-in early stopping holds out a stratified sample, which fails when a
        # label has a single training row, so it is switched off.
        model = HistGradientBoostingClassifier(
            max_iter=n_estimators, class_weight="balanced", early_stopping=False, random_state=seed,
        )
    else:
        raise ValueError(f"Unknown tree family {family!r}; choose from {TREE_FAMILIES}")
    return Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("model", model)])


def build_comparison_frames(
    ml_state: dict,
    *,
    train_rows: int,
    eval_rows: int,
    block_rows: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    """Load one set of train/validation/test rows with base and temporal context features.

    Training rows follow the active training-subset strategy ("time_spread" for
    "full_train"); validation and test rows are evenly spaced blocks. Returns the
    frames and the feature columns of each feature set.
    """
    flag = ml_state["TARGET_FLAG"]
    labels = {"good_labels": ml_state["GOOD_LABELS"], "issue_labels": ml_state["ISSUE_LABELS"]}
    train_strategy = ml_state["TRAIN_SUBSET_STRATEGY"] if ml_state["TRAIN_SUBSET_STRATEGY"] != "full_train" else "time_spread"
    selections = {
        "train": sample_frame_by_strategy(
            ml_state["train_df"], rows_limit=train_rows, sample_strategy=train_strategy, target_flag=flag,
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
        model_df, base_columns, _ = build_model_frame(
            frame, target_flag=flag, measurement_columns=ml_state["measurement_columns"],
            task_mode=ml_state["task_mode"], model_row_limit=None, **labels,
        )
        frames[split_name], context_columns = add_temporal_context_features(
            model_df, measurement_columns=ml_state["measurement_columns"]
        )
    return frames, {"base": base_columns, "base_plus_context": base_columns + context_columns}


def _good_and_missing(ml_state: dict) -> tuple[list[int], list[int]]:
    if ml_state["task_mode"] == "binary":
        return [0], []
    return [int(label) for label in ml_state["GOOD_LABELS"]], [int(label) for label in ml_state["FLAG_EXAMPLE_AVOID_CONTEXT_LABELS"]]


def _raw_issue_f1(raw_labels, predicted_issue, *, good, missing) -> float:
    """Issue-vs-good F1 on the original labels, missing-data label left out."""
    keep = ~np.isin(raw_labels, missing)
    true_issue = ~np.isin(raw_labels[keep], good)
    if not true_issue.any():
        return np.nan
    return float(f1_score(true_issue, predicted_issue[keep], zero_division=0))


def compare_tree_models(
    frames: dict[str, pd.DataFrame],
    feature_columns: dict[str, list[str]],
    *,
    families,
    feature_sets,
    ml_state: dict,
    seed: int,
) -> pd.DataFrame:
    """Train every family and feature-set combination on the same rows; one result row each."""
    good, missing = _good_and_missing(ml_state)
    targets = {name: frame["model_target"].astype(int).to_numpy() for name, frame in frames.items()}
    labels = sorted(set().union(*(set(values.tolist()) for values in targets.values())))
    rows = []
    for family in families:
        for feature_set in feature_sets:
            columns = feature_columns[feature_set]
            started = time.time()
            model = make_tree_classifier(family, seed=seed).fit(frames["train"][columns], targets["train"])
            row = {"family": family, "features": feature_set, "feature_count": len(columns), "fit_seconds": round(time.time() - started, 1)}
            for split_name in ("validation", "test"):
                scores = split_scores(
                    targets[split_name], model.predict(frames[split_name][columns]), labels=labels, good=good, missing=missing
                )
                row[f"{split_name} macro F1 (labels present)"] = round(float(scores[SCORE_COLUMNS["present"]]), 3)
                row[f"{split_name} issue-vs-good F1"] = round(float(scores[SCORE_COLUMNS["issue"]]), 3)
            rows.append(row)
    return pd.DataFrame(rows)


def compare_target_strategies(
    frames: dict[str, pd.DataFrame],
    columns: list[str],
    *,
    strategies,
    family: str,
    ml_state: dict,
    seed: int,
) -> pd.DataFrame:
    """Train one model per target definition on the same rows and inputs; one result row each.

    Macro F1 is computed over each target's own labels, so it rises when classes are
    merged. The issue-vs-good F1 is computed on the original labels for every target,
    so it can be compared across target definitions.
    """
    good, missing = _good_and_missing(ml_state)
    rows = []
    for strategy in strategies:
        mapped = {
            name: apply_target_strategy(frame, "model_target", strategy, issue_labels=ml_state["ISSUE_LABELS"])[0]
            for name, frame in frames.items()
        }
        strategy_good, strategy_missing = (
            ([0], []) if strategy == "binary_issue" else target_strategy_labels(strategy, good, missing)
        )
        started = time.time()
        model = make_tree_classifier(family, seed=seed).fit(mapped["train"][columns], mapped["train"]["strategy_target"])
        targets = {name: frame["strategy_target"].to_numpy() for name, frame in mapped.items()}
        labels = sorted(set().union(*(set(values.tolist()) for values in targets.values())))
        row = {"target": strategy, "labels": labels, "fit_seconds": round(time.time() - started, 1)}
        for split_name in ("validation", "test"):
            predicted = model.predict(mapped[split_name][columns])
            scores = split_scores(targets[split_name], predicted, labels=labels, good=strategy_good, missing=strategy_missing)
            row[f"{split_name} macro F1 (labels present)"] = round(float(scores[SCORE_COLUMNS["present"]]), 3)
            row[f"{split_name} issue-vs-good F1"] = round(
                _raw_issue_f1(
                    mapped[split_name]["model_target"].astype(int).to_numpy(),
                    ~np.isin(predicted, strategy_good),
                    good=good,
                    missing=missing,
                ),
                3,
            )
        rows.append(row)
    return pd.DataFrame(rows)
