"""Reproduce Phase 8.20 cost-aware event diagnostics from a Phase 8.13 CSV.

Example:
python scripts/phase_8_20_cost_aware_backtest.py phase_8_13_output/phase_8_13_chronological_research_dataset.csv --manifest-json phase_8_13_output/phase_8_13_dataset_manifest.json

The model is frozen to the Phase 8.17/8.19 documented logistic-regression
protocol. Training and preprocessing use training rows only. Test labels are
used only to restrict rows with known outcomes; they are never model inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from engine.ml_cost_aware_backtest import (
    simulate_fixed_horizon_trades,
    summarize_trades,
)

VOLATILITY_FEATURES = [
    "feature_atr_14",
    "feature_range_atr",
    "feature_body_atr",
    "feature_realized_volatility_20",
]
PRICE_ACTION_FEATURES = [
    "feature_candle_range",
    "feature_body_abs",
    "feature_body_to_range",
    "feature_upper_wick_to_range",
    "feature_lower_wick_to_range",
    "feature_close_location",
    "feature_candle_direction",
    "feature_gap_pct",
    "feature_return_1",
    "feature_return_3",
    "feature_return_6",
    "feature_return_12",
]
VOLUME_FEATURES = [
    "feature_relative_volume_20",
    "feature_volume_zscore_20",
]
HORIZON_FEATURES = {
    3: VOLATILITY_FEATURES + PRICE_ACTION_FEATURES + VOLUME_FEATURES,
    6: VOLATILITY_FEATURES,
    12: VOLATILITY_FEATURES,
}
COST_SCENARIOS_BPS = (0, 5, 10, 20, 30, 50)
MODEL_CONFIG = {
    "estimator": "sklearn.linear_model.LogisticRegression",
    "class_weight": "balanced",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 250,
    "random_state": 42,
    "preprocessing": [
        "SimpleImputer(strategy=median, add_indicator=True)",
        "StandardScaler",
    ],
    "fit_scope": "labelled training rows only",
}


def _validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "timestamp", "symbol", "timeframe", "open", "close",
        "validation_split", *[f"target_{h}" for h in HORIZON_FEATURES],
        *{c for cols in HORIZON_FEATURES.values() for c in cols},
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Dataset is empty.")

    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    if data["timestamp"].isna().any():
        raise ValueError("Dataset contains invalid timestamps.")
    if data[["symbol", "timeframe", "validation_split"]].isna().any().any():
        raise ValueError("Symbol, timeframe and split fields cannot be missing.")
    for column in ("symbol", "timeframe", "validation_split"):
        if data[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} cannot contain blank values.")
    if not data["timeframe"].astype(str).str.lower().eq("5m").all():
        raise ValueError("Phase 8.20 requires the Phase 8.13 5-minute dataset only.")
    if not data["validation_split"].astype(str).isin(
        ["train", "validation", "test"]
    ).all():
        raise ValueError("Unknown validation_split value.")
    if data.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("Duplicate symbol/timestamp rows are not allowed.")
    for column in ("open", "close"):
        values = pd.to_numeric(data[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all() or (values <= 0).any():
            raise ValueError(f"{column} must contain finite positive prices.")
        data[column] = values

    dates = data["timestamp"].dt.strftime("%Y-%m-%d")
    split_by_date = pd.DataFrame({
        "date": dates,
        "split": data["validation_split"].astype(str),
    }).groupby("date")["split"].nunique()
    if (split_by_date > 1).any():
        raise ValueError("A calendar date appears in more than one split.")
    ranges = {
        name: sorted(dates[data["validation_split"].astype(str).eq(name)].unique())
        for name in ("train", "validation", "test")
    }
    if any(not ranges[name] for name in ranges):
        raise ValueError("Train, validation and test partitions must all be present.")
    if not max(ranges["train"]) < min(ranges["validation"]):
        raise ValueError("Training dates must precede validation dates.")
    if not max(ranges["validation"]) < min(ranges["test"]):
        raise ValueError("Validation dates must precede test dates.")

    for horizon in HORIZON_FEATURES:
        target = f"target_{horizon}"
        numeric = pd.to_numeric(data[target], errors="coerce")
        if numeric[data[target].notna()].isna().any():
            raise ValueError(f"{target} contains non-numeric labels.")
        observed = numeric.dropna()
        if not np.isfinite(observed.to_numpy(dtype=float)).all():
            raise ValueError(f"{target} contains non-finite labels.")
        if not np.equal(observed.to_numpy(float), np.floor(observed.to_numpy(float))).all():
            raise ValueError(f"{target} contains fractional class labels.")
        if not set(observed.astype(int).unique()).issubset({-1, 0, 1}):
            raise ValueError(f"{target} must contain only -1, 0, +1 or missing.")
        data[target] = numeric

    return data


def _make_frozen_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            C=1.0,
            solver="lbfgs",
            max_iter=250,
            random_state=42,
        )),
    ])


def fit_frozen_test_predictions(frame: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Fit one frozen horizon model on train rows and predict labelled test rows."""
    data = _validate_dataset(frame)
    outputs: dict[int, pd.DataFrame] = {}
    train_mask = data["validation_split"].astype(str).eq("train")
    test_mask = data["validation_split"].astype(str).eq("test")

    for horizon, features in HORIZON_FEATURES.items():
        target = f"target_{horizon}"
        train = data.loc[train_mask & data[target].notna()].copy()
        test = data.loc[test_mask & data[target].notna()].copy()
        if train.empty or test.empty:
            raise ValueError(f"Horizon {horizon} has no labelled train or test rows.")
        y_train = train[target].astype(int)
        if y_train.nunique() < 2:
            raise ValueError(f"Horizon {horizon} training data has fewer than two classes.")
        x_train = train[features].apply(pd.to_numeric, errors="coerce")
        x_test = test[features].apply(pd.to_numeric, errors="coerce")
        for column in features:
            train_values = x_train[column].replace([np.inf, -np.inf], np.nan)
            if train_values.notna().sum() == 0:
                raise ValueError(
                    f"Horizon {horizon} feature {column} is entirely missing in training."
                )
        x_train = x_train.replace([np.inf, -np.inf], np.nan)
        x_test = x_test.replace([np.inf, -np.inf], np.nan)

        model = _make_frozen_model()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        result = test[[
            "timestamp", "symbol", "open", "close", target
        ]].copy()
        result["session_date"] = result["timestamp"].dt.strftime("%Y-%m-%d")
        result["prediction"] = predictions.astype(int)
        result["horizon_bars"] = int(horizon)
        outputs[horizon] = result.reset_index(drop=True)
    return outputs


def _group_summary(trades: pd.DataFrame, column: str) -> dict[str, Any]:
    if trades.empty:
        return {}
    result: dict[str, Any] = {}
    for key, group in trades.groupby(column, sort=True):
        result[str(key)] = {
            "trades": int(len(group)),
            "gross_mean_bps": float(group["gross_return"].mean() * 10000),
            "net_mean_bps": float(group["net_return"].mean() * 10000),
            "net_win_rate_pct": float((group["net_return"] > 0).mean() * 100),
        }
    return result


def run_experiment(
    frame: pd.DataFrame,
    *,
    cost_scenarios_bps: tuple[int, ...] = COST_SCENARIOS_BPS,
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame], dict[str, Any]]:
    """Run frozen test predictions, event audit and cost sensitivity."""
    data = _validate_dataset(frame)
    predictions = fit_frozen_test_predictions(data)
    audit_outputs: dict[int, pd.DataFrame] = {}
    report: dict[str, Any] = {
        "phase": "8.20_corrected_cost_aware_event_simulation",
        "model_config": MODEL_CONFIG,
        "sklearn_version": sklearn.__version__,
        "test_dates": sorted(
            data.loc[data["validation_split"].astype(str).eq("test"), "timestamp"]
            .dt.strftime("%Y-%m-%d").unique().tolist()
        ),
        "horizons": {},
        "guardrails": [
            "The test period was already consumed by Phase 8.19 and is not independent validation.",
            "Model and preprocessing are fitted on labelled training rows only.",
            "Validation rows are not used for fitting or threshold tuning.",
            "Test target values are not model inputs; they are retained only for audit.",
            "Events enter at the next bar open and exit at the close after exactly h full bars.",
            "Costs are sensitivity assumptions, not broker-specific fee calculations.",
            "No live signals, portfolio sizing, or order placement is produced.",
        ],
    }

    for horizon, predicted in predictions.items():
        horizon_report: dict[str, Any] = {
            "test_rows_with_known_target": int(len(predicted)),
            "prediction_counts": {
                str(k): int(v) for k, v in predicted["prediction"].value_counts().sort_index().items()
            },
            "actual_target_counts": {
                str(k): int(v) for k, v in predicted[f"target_{horizon}"].value_counts().sort_index().items()
            },
            "cost_sensitivity": {},
        }
        audit_frame = None
        for cost_bps in cost_scenarios_bps:
            trades, current_audit = simulate_fixed_horizon_trades(
                predicted,
                horizon_bars=horizon,
                interval_minutes=5,
                round_trip_cost_bps=float(cost_bps),
                return_audit=True,
            )
            if audit_frame is None:
                audit_frame = current_audit
            key = str(cost_bps)
            horizon_report["cost_sensitivity"][key] = {
                **summarize_trades(trades),
                "long_trades": int((trades["side"] == "LONG").sum()) if not trades.empty else 0,
                "short_trades": int((trades["side"] == "SHORT").sum()) if not trades.empty else 0,
                "by_symbol": _group_summary(trades, "symbol"),
                "by_date": _group_summary(trades, "session_date"),
            }
        assert audit_frame is not None
        audit_outputs[horizon] = audit_frame
        horizon_report["event_audit_reason_counts"] = {
            str(reason): int(count)
            for reason, count in audit_frame["reason"].value_counts().sort_index().items()
        }
        horizon_report["accepted_events"] = int((audit_frame["status"] == "accepted").sum())
        horizon_report["skipped_events"] = int((audit_frame["status"] == "skipped").sum())
        report["horizons"][str(horizon)] = horizon_report
    return predictions, audit_outputs, report


def validate_phase_8_19_manifest(manifest: dict[str, Any]) -> None:
    """Fail closed unless Phase 8.13 target bands match the frozen report."""
    bands = manifest.get("neutral_bands_pct")
    if not isinstance(bands, dict):
        raise ValueError("Phase 8.13 manifest must include neutral_bands_pct.")
    for horizon in ("3", "6", "12"):
        if horizon not in bands:
            raise ValueError(f"Phase 8.13 manifest is missing the {horizon}-bar neutral band.")
        try:
            value = float(bands[horizon])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid neutral band for horizon {horizon}.") from exc
        if not np.isclose(value, 0.20, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"Phase 8.19 protocol requires a 0.20% neutral band for {horizon} bars; "
                f"manifest contains {value}%. Do not rerun until the protocol is reconciled."
            )

    expected_splits = {
        "train": ("2026-05-19", "2026-07-09", 36),
        "validation": ("2026-07-10", "2026-07-21", 8),
        "test": ("2026-07-22", "2026-07-31", 8),
    }
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("Phase 8.13 manifest must include chronological split metadata.")
    for name, (first_date, last_date, sessions) in expected_splits.items():
        item = splits.get(name)
        if not isinstance(item, dict):
            raise ValueError(f"Phase 8.13 manifest is missing the {name} split.")
        observed = (item.get("first_date"), item.get("last_date"), item.get("sessions"))
        if observed != (first_date, last_date, sessions):
            raise ValueError(
                f"{name} split does not match frozen Phase 8.19 dates/session count: "
                f"expected {(first_date, last_date, sessions)}, found {observed}."
            )

    if manifest.get("selected_complete_5m_rows") != 192450:
        raise ValueError("Phase 8.13 manifest row count does not match the frozen 192,450-candle dataset.")
    if manifest.get("selected_sessions") != 2566:
        raise ValueError("Phase 8.13 manifest session count does not match the frozen 2,566 sessions.")
    symbols = manifest.get("selected_symbols")
    if not isinstance(symbols, list) or len(set(symbols)) != 50:
        raise ValueError("Phase 8.13 manifest must identify the frozen 50-symbol universe.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_csv", help="Phase 8.13 chronological research CSV")
    parser.add_argument("--manifest-json", required=True,
                        help="Matching phase_8_13_dataset_manifest.json")
    parser.add_argument("--output-dir", default="phase_8_20_output")
    args = parser.parse_args()

    source = Path(args.dataset_csv)
    manifest_source = Path(args.manifest_json)
    raw_bytes = source.read_bytes()
    manifest_bytes = manifest_source.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    validate_phase_8_19_manifest(manifest)
    frame = pd.read_csv(source)
    predictions, audit_outputs, report = run_experiment(frame)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for horizon, predicted in predictions.items():
        predicted.to_csv(output / f"phase_8_20_test_predictions_{horizon}bar.csv", index=False)
        audit_outputs[horizon].to_csv(
            output / f"phase_8_20_event_audit_{horizon}bar.csv", index=False
        )
    report["input"] = {
        "filename": source.name,
        "sha256": sha256,
        "manifest_filename": manifest_source.name,
        "manifest_sha256": manifest_sha256,
        "manifest_neutral_bands_pct": manifest["neutral_bands_pct"],
        "rows": int(len(frame)),
        "symbols": sorted(frame["symbol"].astype(str).unique().tolist()),
    }
    report["code_revision"] = os.environ.get("GITHUB_SHA", "not-provided")
    (output / "phase_8_20_corrected_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
