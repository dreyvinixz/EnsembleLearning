from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ensemble_learning.contracts import validate_sequence_frame
from ensemble_learning.neural_training import run_neural_experiment

TZ = ZoneInfo("America/Sao_Paulo")


def _neural_development_frame():
    rows = []
    for month in range(1, 6):
        for day in (5, 6):
            session = f"2026-{month:02d}-{day:02d}"
            start = datetime.fromisoformat(f"{session}T09:01:00").replace(tzinfo=TZ)
            for minute in range(70):
                decision = start + timedelta(minutes=minute)
                label = minute % 2
                row = {
                    "instrument": "SYNTH", "session_date": session,
                    "decision_time": decision, "features_available_at": decision,
                    "x_signal": float((minute % 7) - 3),
                }
                for horizon in (5, 10, 15):
                    row[f"y_{horizon}m"] = label
                    row[f"label_info_start_{horizon}m"] = datetime.fromisoformat(
                        f"{session}T09:00:00"
                    ).replace(tzinfo=TZ)
                    row[f"label_info_end_{horizon}m"] = datetime.fromisoformat(
                        f"{session}T18:01:00"
                    ).replace(tzinfo=TZ)
                    row[f"label_known_at_{horizon}m"] = datetime.fromisoformat(
                        f"{session}T18:02:00"
                    ).replace(tzinfo=TZ)
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("strategy", ["independent", "mtl"])
def test_neural_walk_forward_records_b0_predictions_and_artifacts(tmp_path, strategy):
    raw = _neural_development_frame()
    data_path = tmp_path / "development.csv"
    raw.to_csv(data_path, index=False)
    frame = validate_sequence_frame(pd.read_csv(data_path))
    output = run_neural_experiment(
        frame, architecture="mlp", strategy=strategy, data_path=data_path,
        results_root=tmp_path / "results", seed=17, device_name="cpu",
        epochs=1, patience=1, batch_size=16, hidden_size=4,
    )
    required = {
        "config.yaml", "manifest.json", "metrics.json", "metrics_by_horizon.csv",
        "metrics_by_fold_horizon.csv", "metrics_by_session.csv",
        "metrics_by_instrument.csv", "predictions.parquet", "training_history.csv",
        "training_summary.json", "checkpoints",
    }
    assert required <= {path.name for path in output.iterdir()}
    predictions = pd.read_parquet(output / "predictions.parquet")
    manifest = __import__("json").loads((output / "manifest.json").read_text())
    assert set(predictions.scale) == {"5m", "10m", "15m"}
    assert set(predictions.fold) == {"fold01", "fold02", "fold03", "fold04"}
    assert predictions.session_date.astype(str).max() <= "2026-05-31"
    assert predictions.target_protocol.eq("b0_retrospective").all()
    assert predictions.p_up.between(0, 1).all()
    assert predictions.p_down.between(0, 1).all()
    assert manifest["test_june_accessed"] is False
    assert manifest["target_columns_used_as_features"] is False
    assert len(list((output / "checkpoints").glob("*_best.pt"))) == (
        12 if strategy == "independent" else 4
    )


def test_neural_runner_rejects_june_before_training(tmp_path):
    frame = _neural_development_frame()
    june = frame.iloc[[0]].copy()
    june["session_date"] = "2026-06-01"
    with pytest.raises(ValueError, match="June is reserved"):
        run_neural_experiment(
            pd.concat([frame, june], ignore_index=True), architecture="mlp",
            strategy="independent", data_path=__file__, results_root=tmp_path,
            epochs=1, patience=1,
        )
