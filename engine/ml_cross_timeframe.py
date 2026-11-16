"""Compare Phase 8.9 validation-only pattern associations across timeframes.

This is an aggregate robustness diagnostic. It does not match individual 5m
candles to 15m candles, fit a model, inspect held-out test outcomes, or create
trading signals.
"""
from __future__ import annotations

from typing import Any


def _timeframe(report: dict[str, Any]) -> str:
    summary = report.get("dataset_summary", {})
    values = summary.get("timeframes", [])
    if len(values) != 1 or values[0] not in {"5m", "15m"}:
        raise ValueError("Each report must contain exactly one supported timeframe: 5m or 15m.")
    return values[0]


def _neutral_bands(report: dict[str, Any]) -> dict[str, float]:
    bands = report.get("dataset_summary", {}).get("neutral_bands_pct")
    if not isinstance(bands, dict) or not bands:
        raise ValueError("Report is missing neutral_bands_pct metadata.")
    return {str(k): float(v) for k, v in bands.items()}


def _directional_effect(comparison: dict[str, Any], min_effect_pp: float) -> dict[str, Any]:
    if not comparison.get("eligible"):
        return {"eligible": False, "direction": "insufficient_support", "effect_pp": None}
    differences = comparison.get("rate_difference_pattern_minus_comparison", {})
    if "1" not in differences or "-1" not in differences:
        return {"eligible": False, "direction": "missing_rates", "effect_pp": None}
    effect_pp = (float(differences["1"]) - float(differences["-1"])) * 100.0
    if effect_pp >= min_effect_pp:
        direction = "bullish"
    elif effect_pp <= -min_effect_pp:
        direction = "bearish"
    else:
        direction = "weak_or_neutral"
    return {"eligible": True, "direction": direction, "effect_pp": round(effect_pp, 3)}


def _compare_pair(left: dict[str, Any], right: dict[str, Any], min_effect_pp: float) -> dict[str, Any]:
    a = _directional_effect(left, min_effect_pp)
    b = _directional_effect(right, min_effect_pp)
    if not a["eligible"] or not b["eligible"]:
        status = "insufficient_support"
    elif a["direction"] in {"bullish", "bearish"} and b["direction"] in {"bullish", "bearish"}:
        status = "same_direction" if a["direction"] == b["direction"] else "direction_conflict"
    else:
        # A weak/neutral result is not a directional conflict with a bullish
        # or bearish result; it means the pair has no clear shared direction.
        status = "no_clear_direction"
    return {"5m": a, "15m": b, "status": status}


def analyze_cross_timeframe_robustness(
    report_5m: dict[str, Any],
    report_15m: dict[str, Any],
    *,
    min_effect_pp: float = 2.0,
) -> dict[str, Any]:
    """Compare 5m and 15m validation-only pattern effects.

    A directional effect is the pattern-present minus pattern-absent change
    in (up-class rate minus down-class rate), expressed in percentage points.
    A direction is only assigned at or beyond min_effect_pp. Existing Phase
    8.9 support eligibility is respected for overall, per-symbol and context
    comparisons.
    """
    if _timeframe(report_5m) != "5m" or _timeframe(report_15m) != "15m":
        raise ValueError("Pass the 5m report first and the 15m report second.")
    if _neutral_bands(report_5m) != _neutral_bands(report_15m):
        raise ValueError("Reports must use identical neutral-band settings.")
    if not isinstance(min_effect_pp, (int, float)) or min_effect_pp < 0:
        raise ValueError("min_effect_pp must be a non-negative number.")

    symbols_5m = set(report_5m.get("symbols", []))
    symbols_15m = set(report_15m.get("symbols", []))
    common_symbols = sorted(symbols_5m & symbols_15m)
    patterns_5m = set(report_5m.get("pattern_columns", []))
    patterns_15m = set(report_15m.get("pattern_columns", []))
    common_patterns = sorted(patterns_5m & patterns_15m)
    horizons_5m = set(report_5m.get("horizons", {}))
    horizons_15m = set(report_15m.get("horizons", {}))
    common_horizons = sorted(horizons_5m & horizons_15m, key=int)
    if not common_patterns or not common_horizons or not common_symbols:
        raise ValueError("Reports have no common patterns, horizons or symbols.")

    output: dict[str, Any] = {
        "experiment": "phase_8_10_cross_timeframe_pattern_robustness",
        "timeframes": ["5m", "15m"],
        "neutral_bands_pct": _neutral_bands(report_5m),
        "min_directional_effect_pp": float(min_effect_pp),
        "common_symbols": common_symbols,
        "common_symbol_count": len(common_symbols),
        "common_patterns": common_patterns,
        "common_pattern_count": len(common_patterns),
        "horizons": {},
        "guardrails": [
            "Only Phase 8.9 validation aggregates are compared; no held-out test outcomes are read.",
            "This compares timeframe-level aggregates, not matched or synchronized candle events.",
            "Directional effect is the change in (up rate minus down rate) for pattern-present versus pattern-absent rows.",
            "Existing Phase 8.9 support eligibility is preserved; unsupported comparisons are never ranked.",
            "Same-direction association is not proof of causality, predictive skill, profitability or a trading edge.",
            "The source dataset covers a short validation window; results require longer historical confirmation.",
        ],
    }

    for horizon in common_horizons:
        h5 = report_5m["horizons"][horizon]["patterns"]
        h15 = report_15m["horizons"][horizon]["patterns"]
        horizon_patterns = {}
        for pattern in common_patterns:
            p5, p15 = h5[pattern], h15[pattern]
            overall = _compare_pair(p5["overall"], p15["overall"], float(min_effect_pp))

            symbol_results = {}
            eligible_symbols = []
            same_direction_symbols = []
            for symbol in common_symbols:
                s5 = p5.get("by_symbol", {}).get(symbol, {})
                s15 = p15.get("by_symbol", {}).get(symbol, {})
                pair = _compare_pair(s5, s15, float(min_effect_pp))
                symbol_results[symbol] = pair
                if pair["status"] != "insufficient_support":
                    eligible_symbols.append(symbol)
                    if pair["status"] == "same_direction":
                        same_direction_symbols.append(symbol)

            context_names = sorted(set(p5.get("by_context", {})) & set(p15.get("by_context", {})))
            context_results = {
                name: _compare_pair(
                    p5["by_context"][name], p15["by_context"][name], float(min_effect_pp)
                )
                for name in context_names
            }
            horizon_patterns[pattern] = {
                "occurrences": {"5m": int(p5.get("occurrences", 0)), "15m": int(p15.get("occurrences", 0))},
                "overall": overall,
                "per_symbol": {
                    "comparisons": symbol_results,
                    "eligible_common_symbols": len(eligible_symbols),
                    "same_direction_symbols": len(same_direction_symbols),
                    "same_direction_symbol_names": same_direction_symbols,
                },
                "by_context": context_results,
            }
        output["horizons"][horizon] = {"patterns": horizon_patterns}
    return output
