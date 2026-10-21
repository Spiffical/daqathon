"""Plain-language explanations for the QC model scores in the Session 1 notebooks.

These helpers never change a model. They look at the labels, predictions, and
inputs behind a score and say why it looks the way it does: labels missing from
a split, labels with too few training examples, over-called labels, and missed
rows that look just like good data. ``log_experiment`` keeps one row per
evaluated model so runs with different settings can be compared side by side.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from sklearn.metrics import f1_score, precision_recall_fscore_support

from .session1_modeling import CLOCK_FEATURE_COLUMNS

# A label with fewer training examples than this is called rare.
RARE_TRAINING_EXAMPLES = 50
# A label with fewer evaluation examples than this gets a "scores can swing" note.
SMALL_EVALUATION_EXAMPLES = 30
# Over-called: the model catches most of the label's rows, but most of its predictions of it are wrong.
OVER_CALLED_PRECISION = 0.5
OVER_CALLED_RECALL = 0.8
# Mostly missed: the model catches fewer than this share of the label's rows.
MISSED_RECALL = 0.3
# Good-data value ranges run from this lower to this upper quantile of good training rows.
GOOD_RANGE_QUANTILES = (0.01, 0.99)
# Missed rows "look like good data" when their in-range share is at least this fraction of the good test rows' share.
LOOKS_GOOD_SHARE = 0.5
# Below this share of good test rows inside the good training ranges, the test period looks different from training.
SHIFTED_GOOD_SHARE = 0.25

SCORE_COLUMNS = {
    "all": "macro F1 (all labels)",
    "present": "macro F1 (labels present)",
    "issue": "issue-vs-good F1",
}


def _good_and_missing(task_mode: str, good_labels, missing_labels) -> tuple[list[int], list[int]]:
    """Return the good and missing-data labels in the model's label space."""
    if task_mode == "binary":
        return [0], []
    return [int(label) for label in good_labels], [int(label) for label in missing_labels]


def target_strategy_labels(strategy: str, good_labels, missing_labels) -> tuple[list[int], list[int]]:
    """Return the good and missing-data labels after a target strategy from ``apply_target_strategy``."""
    if strategy == "collapsed_1_34_9":
        return [1], [9]
    if strategy in {"collapsed_12_34_9", "oxygen_collapse_12_34_9"}:
        return [12], [9]
    if strategy == "binary_issue":
        return [0], []
    return [int(label) for label in good_labels], [int(label) for label in missing_labels]


def indexes_to_labels(values, class_labels, task_mode: str) -> np.ndarray:
    """Turn CNN/Transformer class indexes back into the labels they stand for.

    Negative values mark padded positions and are dropped. Binary models already
    predict 0 (good) or 1 (issue), so their values are returned unchanged.
    """
    array = np.asarray(values).ravel()
    array = array[array >= 0].astype(int)
    if task_mode == "binary":
        return array
    return np.asarray(class_labels)[array]


def label_counts(labels) -> dict[int, int]:
    """Count how many examples carry each label."""
    series = pd.Series(np.asarray(labels).ravel()).dropna().astype(int)
    return {int(label): int(count) for label, count in series.value_counts().sort_index().items()}


def good_value_ranges(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    good_labels,
    task_mode: str,
    target_column: str = "model_target",
) -> pd.DataFrame:
    """Return the usual range (1st to 99th percentile) of every input on good training rows."""
    good, _ = _good_and_missing(task_mode, good_labels, [])
    good_rows = frame.loc[pd.to_numeric(frame[target_column], errors="coerce").isin(good), feature_columns]
    low, high = GOOD_RANGE_QUANTILES
    return pd.DataFrame({"low": good_rows.quantile(low), "high": good_rows.quantile(high)})


def _inside_good_ranges(rows: pd.DataFrame, ranges: pd.DataFrame) -> pd.DataFrame:
    """True where an input sits inside its good-data range (missing values count as inside)."""
    columns = [column for column in ranges.index if column in rows.columns]
    values = rows[columns]
    return (values.ge(ranges.loc[columns, "low"]) & values.le(ranges.loc[columns, "high"])) | values.isna()


def _share_inside_good_ranges(rows: pd.DataFrame, ranges: pd.DataFrame) -> float:
    """Share of rows whose every input sits inside the good-data range."""
    return float(_inside_good_ranges(rows, ranges).all(axis=1).mean())


def split_scores(y_true, y_pred, *, labels, good, missing) -> dict[str, float]:
    """Score one split three ways: macro F1 over all labels, over the labels present, and issue-vs-good F1."""
    present = sorted(set(y_true.tolist()))
    keep = ~np.isin(y_true, missing)
    true_issue = ~np.isin(y_true[keep], good)
    predicted_issue = ~np.isin(y_pred[keep], good)
    return {
        SCORE_COLUMNS["all"]: f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        SCORE_COLUMNS["present"]: f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
        SCORE_COLUMNS["issue"]: f1_score(true_issue, predicted_issue, zero_division=0) if true_issue.any() else np.nan,
    }


def explain_scores(
    *,
    splits: dict[str, tuple],
    train_counts: dict[int, int],
    labels,
    good_labels,
    missing_labels=(),
    task_mode: str = "multiclass",
    label_name: str = "label",
    example_name: str = "rows",
    feature_importances: pd.Series | None = None,
    test_inputs: pd.DataFrame | None = None,
    good_ranges: pd.DataFrame | None = None,
    dataset_notes=(),
) -> tuple[pd.DataFrame, list[str]]:
    """Score each split three ways and list plain-language notes about the results.

    ``splits`` maps a split name to ``(y_true, y_pred)`` in label space.
    ``train_counts`` counts the training examples of each label. With
    ``test_inputs`` (the model inputs of the test rows, in the same order as the
    test labels) and ``good_ranges`` (from ``good_value_ranges``), the notes also
    say whether missed test rows look like good training data.
    """
    good, missing = _good_and_missing(task_mode, good_labels, missing_labels)
    labels = [int(label) for label in labels]
    issue_labels = [label for label in labels if label not in good]
    table_rows, notes = [], []

    # Training data first: a label the model rarely or never saw cannot be learned well.
    evaluated = set().union(*(set(np.asarray(y_true).astype(int).tolist()) for y_true, _ in splits.values()))
    for label in issue_labels:
        count = train_counts.get(label, 0)
        if label in evaluated and count == 0:
            notes.append(f"{label_name} {label} never appears in the training {example_name}, so the model cannot predict it.")
        elif label in evaluated and count < RARE_TRAINING_EXAMPLES and label not in missing:
            examples = example_name if count != 1 else example_name.rstrip("s")
            notes.append(f"{label_name} {label} has only {count} training {examples}, so it may be missed.")

    # Do good test rows fit the usual range of good training rows? If most do not, the test
    # period differs from training, and comparing missed rows with that range says little.
    good_share, shift_note = None, None
    if "test" in splits and test_inputs is not None and good_ranges is not None:
        test_true = np.asarray(splits["test"][0]).astype(int)
        good_inside = _inside_good_ranges(test_inputs.iloc[np.flatnonzero(np.isin(test_true, good))], good_ranges)
        if len(good_inside):
            good_share = float(good_inside.all(axis=1).mean())
        if good_share is not None and good_share < SHIFTED_GOOD_SHARE:
            outside = (~good_inside).mean().sort_values(ascending=False)
            shift_note = (
                f"Test: only {good_share:.0%} of the good test rows have every input inside the usual range of good "
                f"training rows, so the test period looks different from training. The input most often outside its "
                f"training range is `{outside.index[0]}` ({outside.iloc[0]:.0%} of good test rows)."
            )

    for split_name, (y_true, y_pred) in splits.items():
        y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
        scores = split_scores(y_true, y_pred, labels=labels, good=good, missing=missing)
        table_rows.append({"split": split_name, **{name: round(float(value), 3) for name, value in scores.items()}})

        absent = [label for label in labels if label not in set(y_true.tolist())]
        if absent:
            names = " and ".join(str(label) for label in absent)
            verb = "has" if len(absent) == 1 else "have"
            notes.append(
                f"{split_name.title()}: {label_name} {names} {verb} no {example_name} in this split. sklearn's `macro avg` "
                f"counts each missing label as an F1 of 0, so its macro F1 is {scores[SCORE_COLUMNS['all']]:.2f}; "
                f"over the labels that are present it is {scores[SCORE_COLUMNS['present']]:.2f}."
            )

        precision, recall, _, support = precision_recall_fscore_support(
            y_true, y_pred, labels=issue_labels, zero_division=0
        )
        for label, label_precision, label_recall, label_support in zip(issue_labels, precision, recall, support):
            if label_support == 0 or label in missing:
                continue
            if label_support < SMALL_EVALUATION_EXAMPLES:
                notes.append(
                    f"{split_name.title()}: {label_name} {label} has only {label_support} "
                    f"{example_name if label_support != 1 else example_name.rstrip('s')}, so a few predictions move its F1 a lot."
                )
                continue
            if label_precision < OVER_CALLED_PRECISION and label_recall >= OVER_CALLED_RECALL:
                notes.append(
                    f"{split_name.title()}: {label_name} {label} is over-called. The model catches {label_recall:.0%} "
                    f"of these {example_name}, but only {label_precision:.0%} of its {label_name} {label} predictions "
                    "are right. Lighter class weights can help; if the model uses no class weights, compare the inputs in this split with the training rows."
                )
            elif label_recall < MISSED_RECALL:
                note = (
                    f"{split_name.title()}: the model misses most {label_name} {label} {example_name} "
                    f"(it catches {label_recall:.0%} of {label_support})."
                )
                if split_name == "test" and good_share is not None:
                    missed_share = _share_inside_good_ranges(
                        test_inputs.iloc[np.flatnonzero((y_true == label) & (y_pred != label))], good_ranges
                    )
                    note += (
                        f" {missed_share:.0%} of the missed rows have every input inside the usual range of good "
                        f"training rows (for comparison, {good_share:.0%} of the good test rows do)."
                    )
                    if shift_note is not None:
                        note += " Most good test rows are outside that range too (see the note on the test period), so this comparison says little here."
                    elif missed_share >= LOOKS_GOOD_SHARE * good_share:
                        note += (
                            " Most missed rows look like good data, so these inputs barely separate them: "
                            "different inputs would likely help more than a different model."
                        )
                    else:
                        note += (
                            " The missed rows stand out from good data, so the inputs do carry a signal: more training "
                            "examples of this label (for example the `balanced_reviewed` training subset) or different "
                            "class weights may help."
                        )
                notes.append(note)

    if shift_note is not None:
        notes.append(shift_note)

    if feature_importances is not None:
        top_features = feature_importances.sort_values(ascending=False).head(3).index.tolist()
        clock_features = [feature for feature in top_features if feature in CLOCK_FEATURE_COLUMNS]
        if clock_features:
            notes.append(
                f"`{clock_features[0]}` is one of the three most important features. In a dataset that covers about "
                "one year, the forest can use clock features to memorise when problems happened instead of learning "
                "what they look like, and those dates may not appear in the test period."
            )

    if not notes:
        notes.append("Nothing stands out: every label has enough examples and no label is badly over-called or missed.")
    notes.extend(f"About this dataset: {note}" for note in dataset_notes)
    return pd.DataFrame(table_rows), notes


def show_score_notes(notes: list[str]) -> None:
    """Display score notes as a bulleted list."""
    display(Markdown("\n".join(f"- {note}" for note in notes)))


def log_experiment(
    log_path: str | Path,
    *,
    model_name: str,
    ml_state: dict,
    settings: dict[str, object],
    score_table: pd.DataFrame,
) -> pd.DataFrame:
    """Add one row for this evaluated model to a CSV log and return the whole log."""
    scores = score_table.set_index("split")
    row = {
        "logged_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": ml_state["DATASET_PROFILE_ID"],
        "model": model_name,
        "split_strategy": ml_state["SPLIT_STRATEGY"],
        "train_subset": ml_state["TRAIN_SUBSET_STRATEGY"],
        "data_fraction": ml_state["DATA_FRACTION"],
        "settings": "; ".join(f"{name}={value}" for name, value in settings.items()),
    }
    for split_name in scores.index:
        row[f"{split_name}_macro_f1_present"] = scores.loc[split_name, SCORE_COLUMNS["present"]]
        row[f"{split_name}_issue_f1"] = scores.loc[split_name, SCORE_COLUMNS["issue"]]
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    log.to_csv(log_path, index=False)
    return log
