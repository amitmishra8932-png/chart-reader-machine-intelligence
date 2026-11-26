"""Evidence-first audit for 5paisa provider timestamps and candle alignment.

This tool never infers a timezone for naive provider timestamps and never
rewrites or repairs source candles. It emits a row-level audit CSV and JSON
summary. Wall-clock session labels are candidates only until timezone semantics
and the instrument's session eligibility are independently established.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

REQUIRED = {
    "timestamp", "provider_timestamp_raw", "open", "high", "low", "close",
    "volume", "symbol", "timeframe",
}
OFFSET_RE = re.compile(r"(?:Z|[+-]\d{2}:\d{2})$", re.IGNORECASE)


def _raw_has_offset(value: object) -> bool:
    return bool(OFFSET_RE.search(str(value).strip()))


def _same_instant_or_wall_time(raw: pd.Series, parsed: pd.Series) -> pd.Series:
    """Compare raw and parsed values without localizing naive timestamps."""
    results = []
    for raw_value, parsed_value in zip(raw, parsed):
        try:
            raw_ts = pd.Timestamp(str(raw_value))
            parsed_ts = pd.Timestamp(parsed_value)
            if pd.isna(raw_ts) or pd.isna(parsed_ts):
                results.append(False)
            elif (raw_ts.tz is None) != (parsed_ts.tz is None):
                results.append(False)
            elif raw_ts.tz is None:
                results.append(raw_ts == parsed_ts)
            else:
                results.append(raw_ts.tz_convert("UTC") == parsed_ts.tz_convert("UTC"))
        except (TypeError, ValueError, OverflowError):
            results.append(False)
    return pd.Series(results, index=raw.index, dtype=bool)


def _aggregate_5m(group: pd.DataFrame) -> dict:
    ordered = group.sort_values("timestamp", kind="mergesort")
    return {
        "open": float(ordered.iloc[0]["open"]),
        "high": float(ordered["high"].max()),
        "low": float(ordered["low"].min()),
        "close": float(ordered.iloc[-1]["close"]),
        "volume": float(ordered["volume"].sum()),
    }


def audit_frame(
    source: pd.DataFrame,
    special_dates: set[str] | None = None,
    *,
    provider_labeling_confirmed: bool = False,
) -> tuple[pd.DataFrame, dict]:
    missing = REQUIRED.difference(source.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    data = source.copy().reset_index(drop=True)
    data["provider_timestamp_raw"] = data["provider_timestamp_raw"].astype("string")
    raw = data["provider_timestamp_raw"]
    data["timestamp_parsed"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["raw_timestamp_parsed"] = pd.to_datetime(raw, errors="coerce", format="mixed")
    data["raw_has_explicit_offset"] = raw.map(_raw_has_offset)
    offset_count = int(data["raw_has_explicit_offset"].sum())
    if offset_count == 0:
        timezone_status = "unverified_naive"
    elif offset_count == len(data):
        timezone_status = "explicit_offset_present"
    else:
        timezone_status = "mixed_offset_and_naive"

    data["raw_parse_matches_timestamp"] = _same_instant_or_wall_time(
        raw, data["timestamp_parsed"]
    )
    data["timestamp_parse_ok"] = data["timestamp_parsed"].notna()
    data["ohlcv_numeric_ok"] = True
    for col in ("open", "high", "low", "close", "volume"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data["ohlcv_numeric_ok"] &= data[col].notna() & data[col].map(
            lambda value: pd.notna(value) and float("-inf") < float(value) < float("inf")
        )
    data["ohlc_geometry_ok"] = (
        (data["high"] >= data[["open", "close", "low"]].max(axis=1))
        & (data["low"] <= data[["open", "close", "high"]].min(axis=1))
        & (data[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (data["volume"] >= 0)
    )
    data["duplicate_key"] = data.duplicated(
        ["symbol", "timeframe", "timestamp"], keep=False
    )

    # Use the literal clock portion from provider text for candidate labels.
    # This is deliberately not an IST conversion or timezone assertion.
    raw_clock = data["raw_timestamp_parsed"]
    data["provider_wall_date"] = raw_clock.map(
        lambda value: value.strftime("%Y-%m-%d") if pd.notna(value) else ""
    )
    data["provider_wall_time"] = raw_clock.map(
        lambda value: value.strftime("%H:%M:%S") if pd.notna(value) else ""
    )
    special_dates = special_dates or set()
    data["special_session_candidate"] = data["provider_wall_date"].isin(special_dates)
    data["outside_regular_continuous_session_candidate"] = False
    for idx, row in data.iterrows():
        ts = row["raw_timestamp_parsed"]
        if pd.isna(ts) or row["special_session_candidate"]:
            continue
        minute = ts.hour * 60 + ts.minute
        tf = str(row["timeframe"]).strip().lower()
        # Candidate boundaries for ordinary NSE continuous sessions only.
        if tf == "5m":
            inside = 9 * 60 + 15 <= minute <= 15 * 60 + 10
        elif tf == "15m":
            inside = 9 * 60 + 15 <= minute <= 15 * 60
        else:
            inside = False
        data.at[idx, "outside_regular_continuous_session_candidate"] = not inside

    data["row_audit_status"] = "review"
    data.loc[~data["timestamp_parse_ok"], "row_audit_status"] = "invalid_timestamp"
    data.loc[~data["raw_parse_matches_timestamp"], "row_audit_status"] = "raw_parsed_timestamp_mismatch"
    data.loc[~data["ohlcv_numeric_ok"], "row_audit_status"] = "invalid_ohlcv"
    data.loc[~data["ohlc_geometry_ok"], "row_audit_status"] = "invalid_ohlc_geometry"
    data.loc[data["duplicate_key"], "row_audit_status"] = "duplicate_key"
    data.loc[data["special_session_candidate"], "row_audit_status"] = "special_session_review"
    data.loc[
        data["outside_regular_continuous_session_candidate"],
        "row_audit_status",
    ] = "outside_regular_session_review"

    # Compare each 15m start-labelled candle against its exact three 5m bars.
    five = data[data["timeframe"].astype(str).str.lower().eq("5m")].copy()
    fifteen = data[data["timeframe"].astype(str).str.lower().eq("15m")].copy()
    comparisons = []
    fields = ("open", "high", "low", "close", "volume")
    for _, bar15 in fifteen.iterrows():
        start = bar15["raw_timestamp_parsed"]
        if pd.isna(start) or bar15["special_session_candidate"]:
            continue
        end = start + pd.Timedelta(minutes=15)
        subset = five[
            (five["symbol"].astype(str) == str(bar15["symbol"]))
            & (five["provider_wall_date"] == bar15["provider_wall_date"])
            & (five["raw_timestamp_parsed"] >= start)
            & (five["raw_timestamp_parsed"] < end)
            & (~five["special_session_candidate"])
        ]
        if len(subset) != 3:
            comparisons.append({"status": "incomplete_group"})
            continue
        try:
            aggregate = _aggregate_5m(subset)
            matches = all(
                abs(aggregate[field] - float(bar15[field])) <= 1e-8
                for field in fields
            )
            comparisons.append({"status": "match" if matches else "mismatch"})
        except (TypeError, ValueError):
            comparisons.append({"status": "invalid_values"})
    comparison_counts = pd.Series(
        [item["status"] for item in comparisons], dtype="string"
    ).value_counts().to_dict()

    counts = {
        "rows": int(len(data)),
        "symbols": sorted(data["symbol"].dropna().astype(str).unique().tolist()),
        "timeframes": {
            str(key): int(value)
            for key, value in data["timeframe"].astype(str).value_counts().items()
        },
        "timezone_evidence": timezone_status,
        "rows_with_explicit_offset": offset_count,
        "raw_parsed_timestamp_mismatches": int((~data["raw_parse_matches_timestamp"]).sum()),
        "invalid_timestamps": int((~data["timestamp_parse_ok"]).sum()),
        "invalid_ohlcv": int((~data["ohlcv_numeric_ok"]).sum()),
        "invalid_ohlc_geometry": int((~data["ohlc_geometry_ok"]).sum()),
        "duplicate_keys": int(data["duplicate_key"].sum()),
        "special_session_candidate_rows": int(data["special_session_candidate"].sum()),
        "outside_regular_session_candidate_rows": int(
            data["outside_regular_continuous_session_candidate"].sum()
        ),
        "five_to_fifteen_minute_comparison": {
            str(key): int(value) for key, value in comparison_counts.items()
        },
    }
    # This is intentionally strict: candidate labels and timezone evidence
    # are not silently upgraded to verified facts.
    blockers = []
    if timezone_status != "explicit_offset_present":
        blockers.append("provider timestamps do not all carry explicit timezone offsets")
    if not provider_labeling_confirmed:
        blockers.append("provider candle timestamp start/end semantics have not been independently confirmed")
    if counts["raw_parsed_timestamp_mismatches"]:
        blockers.append("raw provider timestamps do not match parsed timestamps")
    if counts["invalid_timestamps"] or counts["invalid_ohlcv"] or counts["invalid_ohlc_geometry"]:
        blockers.append("timestamp or OHLCV integrity defects are present")
    if counts["duplicate_keys"]:
        blockers.append("duplicate symbol/timeframe/timestamp keys are present")
    if counts["special_session_candidate_rows"]:
        blockers.append("special-session candidate rows require separate review")
    if counts["outside_regular_session_candidate_rows"]:
        blockers.append("outside-regular-session candidate rows require instrument/session classification")
    if comparison_counts.get("mismatch", 0) or comparison_counts.get("invalid_values", 0):
        blockers.append("5m-to-15m OHLCV aggregation mismatches are present")
    counts["replay_timestamp_gate"] = "pass" if not blockers else "blocked"
    counts["blockers"] = blockers
    counts["limitations"] = [
        "Explicit offsets show the timestamp's stated offset, not that the provider labels candle starts or ends correctly.",
        "Provider candle start-versus-end labeling remains unverified until independently confirmed against provider documentation or known-time observations.",
        "Session labels are wall-clock candidates only; timezone and instrument session eligibility require independent confirmation.",
        "Incomplete 5m-to-15m groups are reported, not synthesized or repaired.",
        "This audit does not establish liquidity, execution quality, strategy profitability, or live-feed correctness.",
    ]
    return data, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output-prefix", type=Path, default=Path("provider_timestamp_audit"))
    parser.add_argument(
        "--special-date", action="append", default=[],
        help="Provider wall-clock date to flag for manual special-session review (YYYY-MM-DD). Repeatable.",
    )
    parser.add_argument(
        "--confirm-provider-candle-label-semantics", action="store_true",
        help="Explicitly attest that provider start/end candle-label semantics were independently confirmed. Does not confirm timezone or session eligibility.",
    )
    args = parser.parse_args()
    source = pd.read_csv(args.input_csv)
    audited, report = audit_frame(
        source, set(args.special_date),
        provider_labeling_confirmed=args.confirm_provider_candle_label_semantics,
    )
    csv_path = args.output_prefix.with_suffix(".rows.csv")
    json_path = args.output_prefix.with_suffix(".json")
    audited.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nWrote {csv_path} and {json_path}")
    return 0 if report["replay_timestamp_gate"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
