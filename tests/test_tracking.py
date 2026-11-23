from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import matthews_corrcoef

from ensemble_learning.contracts import validate_frame
from ensemble_learning.tracking import _session_bootstrap_ci, run_walk_forward, walk_forward_folds


def test_session_bootstrap_matches_resampled_session_mcc():
    frame = pd.DataFrame({
        "session_date": ["2026-01-02"] * 3 + ["2026-01-03"] * 2,
        "y_true": [0, 1, 1, 0, 1],
        "p_up": [.1, .8, .4, .7, .9],
    })
    rng = np.random.default_rng(23)
    sessions = frame.session_date.unique()
    scores = []
    for sampled in (rng.choice(sessions, size=len(sessions), replace=True) for _ in range(40)):
        sample = pd.concat([frame.loc[frame.session_date == day] for day in sampled])
        scores.append(matthews_corrcoef(sample.y_true, (sample.p_up >= .5).astype(int)))
    assert _session_bootstrap_ci(frame, seed=23, iterations=40) == [
        float(np.quantile(scores, .025)), float(np.quantile(scores, .975))
    ]


def _development_frame():
    rows = []
    for month in range(1, 6):
        for day in (2, 3):
            session = f"2026-{month:02d}-{day:02d}"
            for minute in range(8):
                t = datetime.fromisoformat(f"{session}T14:{minute:02d}:00+00:00")
                row = {"instrument": "SYNTH", "session_date": session,
                    "decision_time": t.isoformat(), "features_available_at": t.isoformat(),
                    "x_signal": float(minute % 2), "x_noise": float(day + minute)}
                for horizon in (5, 10, 15):
                    row[f"y_{horizon}m"] = minute % 2
                    row[f"label_info_start_{horizon}m"] = f"{session}T09:00:00+00:00"
                    row[f"label_info_end_{horizon}m"] = f"{session}T18:01:00+00:00"
                    row[f"label_known_at_{horizon}m"] = f"{session}T18:02:00+00:00"
                rows.append(row)
    return pd.DataFrame(rows)


def test_walk_forward_creates_reproducible_artifacts_without_june(tmp_path):
    raw = _development_frame()
    csv = tmp_path / "dev.csv"
    raw.to_csv(csv, index=False)
    frame = validate_frame(pd.read_csv(csv))
    folds = walk_forward_folds(frame)
    assert [fold["fold"] for fold in folds] == ["fold01", "fold02", "fold03", "fold04"]
    assert all(fold["oof"]["session_date"].str.slice(5, 7).max() <= "05" for fold in folds)
    output = run_walk_forward(frame, data_path=csv, results_root=tmp_path / "results", seed=19)
    assert {"config.yaml", "manifest.json", "metrics.json", "metrics_by_horizon.csv",
            "metrics_by_fold_horizon.csv", "metrics_by_session.csv", "metrics_by_instrument.csv",
            "predictions.parquet", "training_history.csv", "checkpoints"}.issubset(
                {path.name for path in output.iterdir()})
    predictions = pd.read_parquet(output / "predictions.parquet")
    assert set(predictions.scale) == {"5m", "10m", "15m"}
    assert predictions.session_date.max() <= "2026-05-31"
    horizon_metrics = pd.read_csv(output / "metrics_by_horizon.csv")
    assert set(horizon_metrics.evaluation_set) == {"development_oof_feb_apr", "may_validation"}
    with pytest.raises(FileExistsError):
        run_walk_forward(frame, data_path=csv, results_root=tmp_path / "results",
                         seed=19, experiment_id=output.name)


def test_walk_forward_rejects_june_rows():
    frame = _development_frame()
    june = frame.iloc[[0]].copy()
    june["session_date"] = "2026-06-01"
    with pytest.raises(ValueError, match="Jan-May"):
        from ensemble_learning.tracking import run_walk_forward
        run_walk_forward(pd.concat([frame, june]), data_path=__file__, results_root="unused")
