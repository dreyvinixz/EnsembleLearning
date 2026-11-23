from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "paired_session_max_t", ROOT / "scripts/paired_session_max_t.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _predictions(probabilities: list[float], *, june: bool = False) -> pd.DataFrame:
    dates = ["2026-02-02", "2026-03-02", "2026-04-01", "2026-05-04", "2026-05-05"]
    rows = []
    for index, day in enumerate(dates):
        for scale, label in (("5m", index % 2), ("10m", (index + 1) % 2)):
            rows.append({
                "instrument": "WING26", "session_date": "2026-06-01" if june else day,
                "decision_time": f"{day}T13:00:00-03:00", "scale": scale,
                "fold": "fold04" if index > 2 else f"fold{index + 1:02d}",
                "y_true": label, "p_up": probabilities[index],
            })
    return pd.DataFrame(rows)


def test_paired_session_max_t_uses_exact_pairs_and_familywise_intervals():
    results = MODULE.max_t_comparisons({
        "model_a": _predictions([.8, .2, .7, .3, .6]),
        "model_b": _predictions([.2, .8, .4, .9, .1]),
    }, iterations=1000, seed=7)
    assert set(results.period) == {"oof_feb_apr", "oof_mar_apr", "validation_may"}
    assert set(results.metric) == {"mcc", "accuracy"}
    assert set(results.loc[results.period.eq("oof_feb_apr"), "n_sessions"]) == {3}
    assert set(results.loc[results.period.ne("oof_feb_apr"), "n_sessions"]) == {2}
    assert results.simultaneous_ci95_low.le(results.difference_left_minus_right).all()
    assert results.simultaneous_ci95_high.ge(results.difference_left_minus_right).all()
    assert results.max_t_critical_95.gt(0).all()


def test_paired_session_max_t_rejects_june_and_label_disagreement():
    with pytest.raises(ValueError, match="June is forbidden"):
        MODULE.align_methods({"a": _predictions([.8, .2, .7, .3, .6], june=True),
                              "b": _predictions([.7, .3, .6, .4, .5], june=True)})
    changed = _predictions([.7, .3, .6, .4, .5])
    changed.loc[0, "y_true"] = 1
    with pytest.raises(ValueError, match="y_true differs"):
        MODULE.align_methods({"a": _predictions([.8, .2, .7, .3, .6]), "b": changed})


def test_select_ensemble_predictions_filters_one_named_member():
    source = _predictions([.8, .2, .7, .3, .6])
    source["ensemble"] = "seed_a"
    other = source.copy()
    other["ensemble"] = "seed_b"
    combined = pd.concat([source, other], ignore_index=True)

    selected = MODULE.select_ensemble_predictions(combined, "seed_b")

    assert len(selected) == len(source)
    assert selected.ensemble.eq("seed_b").all()
    assert selected["p_up"].tolist() == source["p_up"].tolist()


def test_select_ensemble_predictions_requires_exact_key():
    source = _predictions([.8, .2, .7, .3, .6])
    source["ensemble"] = "available"
    with pytest.raises(ValueError, match="No predictions found"):
        MODULE.select_ensemble_predictions(source, "missing")


def test_february_april_uses_methods_with_first_oof_fold_only():
    method_a = _predictions([.8, .2, .7, .3, .6])
    method_b = method_a.loc[method_a.fold.ne("fold01")].reset_index(drop=True)
    method_c = _predictions([.7, .3, .6, .4, .5])

    results = MODULE.max_t_comparisons(
        {"a": method_a, "b_stacking": method_b, "c": method_c},
        iterations=1000,
        seed=11,
    )

    feb_apr = results.loc[results.period.eq("oof_feb_apr")]
    mar_apr = results.loc[results.period.eq("oof_mar_apr")]
    assert set(feb_apr.method_count) == {2}
    assert set(feb_apr.left_method) | set(feb_apr.right_method) == {"a", "c"}
    assert set(mar_apr.method_count) == {3}
