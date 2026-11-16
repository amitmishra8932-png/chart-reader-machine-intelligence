"""Leakage-aware baseline classifier evaluation for Phase 8.6.

Fits preprocessing and estimators on training dates only. Validation and test
sets are held out by calendar date; this module does not tune thresholds or
claim that classification metrics imply tradable performance.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CLASS_LABELS = (-1, 0, 1)
SPLIT_NAMES = ("train", "validation", "test")


def _validate_splits(frame: pd.DataFrame) -> None:
    required = {"timestamp", "validation_split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required split columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Cannot evaluate an empty ML dataset.")

    if frame["validation_split"].isna().any():
        raise ValueError("Every row must have a chronological validation_split.")
    unknown = set(frame["validation_split"].astype(str)) - set(SPLIT_NAMES)
    if unknown:
        raise ValueError(f"Unknown validation_split values: {sorted(unknown)}")
    present = set(frame["validation_split"].astype(str))
    absent = set(SPLIT_NAMES) - present
    if absent:
        raise ValueError(f"All train/validation/test splits are required; missing: {sorted(absent)}")

    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("All timestamps must be parseable for split integrity checks.")
    work = pd.DataFrame({
        "date": timestamps.dt.strftime("%Y-%m-%d"),
        "split": frame["validation_split"].astype(str).to_numpy(),
    })
    assignments_per_date = work.groupby("date")["split"].nunique()
    if (assignments_per_date > 1).any():
        raise ValueError("Calendar dates must not be shared across train/validation/test splits.")

    date_ranges = {
        name: sorted(work.loc[work["split"] == name, "date"].unique().tolist())
        for name in SPLIT_NAMES
    }
    if not (max(date_ranges["train"]) < min(date_ranges["validation"])):
        raise ValueError("Training dates must precede validation dates.")
    if not (max(date_ranges["validation"]) < min(date_ranges["test"])):
        raise ValueError("Validation dates must precede test dates.")


def _split_summary(frame: pd.DataFrame, name: str) -> dict[str, Any]:
    rows = frame[frame["validation_split"].astype(str) == name]
    dates = pd.to_datetime(rows["timestamp"]).dt.strftime("%Y-%m-%d")
    return {
        "rows": int(len(rows)),
        "sessions": int(dates.nunique()),
        "first_date": str(dates.min()),
        "last_date": str(dates.max()),
    }


def _metric_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(CLASS_LABELS), zero_division=0
    )
    return {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_recall_fscore_support(
            y_true, y_pred, labels=list(CLASS_LABELS), average="macro", zero_division=0
        )[0]),
        "macro_recall": float(precision_recall_fscore_support(
            y_true, y_pred, labels=list(CLASS_LABELS), average="macro", zero_division=0
        )[1]),
        "macro_f1": float(f1_score(
            y_true, y_pred, labels=list(CLASS_LABELS), average="macro", zero_division=0
        )),
        "per_class": {
            str(label): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(CLASS_LABELS)
        },
        "confusion_matrix_labels": list(CLASS_LABELS),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(CLASS_LABELS)
        ).astype(int).tolist(),
    }


def _make_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
        )),
    ])


def evaluate_baseline_models(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (3, 6, 12),
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Fit majority and logistic-regression baselines on train dates only.

    Feature preprocessing (median imputation and scaling) is fitted inside a
    sklearn Pipeline, so validation/test values cannot affect learned transforms.
    Test results are reported once by this deterministic evaluation call.
    """
    _validate_splits(frame)
    if not horizons or any(isinstance(h, bool) or int(h) != h or int(h) <= 0 for h in horizons):
        raise ValueError("Horizons must be a non-empty sequence of positive integers.")

    candidates = feature_columns or [
        col for col in frame.columns
        if col.startswith("feature_") and pd.api.types.is_numeric_dtype(frame[col])
    ]
    if not candidates:
        raise ValueError("No numeric feature_ columns are available for training.")

    train_mask = frame["validation_split"].astype(str).eq("train")
    # Determine usable features using training data only.
    usable_features: list[str] = []
    dropped_features: list[str] = []
    for column in candidates:
        if column not in frame.columns or not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"Feature {column!r} is missing or not numeric.")
        values = pd.to_numeric(frame.loc[train_mask, column], errors="coerce")
        finite = values.replace([np.inf, -np.inf], np.nan).dropna()
        if finite.empty:
            dropped_features.append(column)
        else:
            usable_features.append(column)
    if not usable_features:
        raise ValueError("No feature has finite training observations.")

    x_all = frame[usable_features].apply(pd.to_numeric, errors="coerce")
    x_all = x_all.replace([np.inf, -np.inf], np.nan)

    report: dict[str, Any] = {
        "method": "majority-class DummyClassifier vs class-weighted LogisticRegression",
        "feature_columns": usable_features,
        "dropped_all_missing_in_train": dropped_features,
        "fit_scope": "training dates only",
        "preprocessing": "training-fitted median imputation and standard scaling inside Pipeline",
        "class_labels": {"-1": "Down", "0": "Neutral", "1": "Up"},
        "splits": {name: _split_summary(frame, name) for name in SPLIT_NAMES},
        "horizons": {},
        "limitations": [
            "This is a baseline classification experiment, not a trading strategy backtest.",
            "The logistic-regression model is intentionally simple and is not hyperparameter-tuned.",
            "Validation and test sets are not used to fit preprocessing or model parameters.",
            "Classification metrics do not include fees, slippage, liquidity, turnover or execution.",
            "No profitability, deployability or predictive edge is established by this report.",
            "Small or narrow-universe datasets can produce unstable metrics; broader clean data is required.",
        ],
    }

    for horizon in horizons:
        target_col = f"target_{int(horizon)}"
        if target_col not in frame.columns:
            raise ValueError(f"Missing target column {target_col!r}.")

        labelled_mask = frame[target_col].notna()
        y_numeric = pd.to_numeric(frame[target_col], errors="coerce")
        observed_targets = y_numeric[labelled_mask]
        if observed_targets.isna().any() or not np.isfinite(observed_targets.to_numpy(dtype=float)).all():
            raise ValueError(f"{target_col} contains non-finite or non-numeric labels.")
        if not np.equal(observed_targets.to_numpy(dtype=float), np.floor(observed_targets.to_numpy(dtype=float))).all():
            raise ValueError(f"{target_col} must contain integer class labels; fractional labels are invalid.")
        if not set(observed_targets.astype(int).unique()).issubset(set(CLASS_LABELS)):
            raise ValueError(f"{target_col} must contain only -1, 0 and +1.")

        horizon_report: dict[str, Any] = {
            "target_column": target_col,
            "class_distribution": {},
            "models": {},
        }
        split_data: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
        for name in SPLIT_NAMES:
            mask = frame["validation_split"].astype(str).eq(name) & labelled_mask
            subset_x = x_all.loc[mask]
            subset_y = y_numeric.loc[mask].astype(int).to_numpy()
            if len(subset_y) == 0:
                raise ValueError(f"{target_col} has no labelled rows in {name} split.")
            horizon_report["class_distribution"][name] = {
                str(label): int(count)
                for label, count in pd.Series(subset_y).value_counts().sort_index().items()
            }
            split_data[name] = (subset_x, subset_y)

        x_train, y_train = split_data["train"]
        if len(np.unique(y_train)) < 2:
            raise ValueError(
                f"{target_col} training split has fewer than two classes; "
                "collect more representative training dates before fitting."
            )

        majority = DummyClassifier(strategy="most_frequent")
        majority.fit(x_train, y_train)
        model = _make_model()
        model.fit(x_train, y_train)

        for model_name, estimator in (("majority_baseline", majority), ("logistic_regression", model)):
            model_report: dict[str, Any] = {}
            for split_name in ("validation", "test"):
                x_eval, y_eval = split_data[split_name]
                prediction = estimator.predict(x_eval)
                model_report[split_name] = _metric_report(y_eval, prediction)
            horizon_report["models"][model_name] = model_report

        horizon_report["training_rows"] = int(len(y_train))
        horizon_report["preprocessing_fit_rows"] = int(len(x_train))
        horizon_report["training_classes"] = sorted(int(v) for v in np.unique(y_train))
        report["horizons"][str(horizon)] = horizon_report

    return report
