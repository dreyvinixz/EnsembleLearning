from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ensemble_learning.contracts import validate_sequence_frame
from ensemble_learning.neural_training import (
    _training_validation_split,
    _WindowDataset,
    run_neural_experiment,
)
from ensemble_learning.pairing import paired_metrics
from ensemble_learning.sequences import CausalWindows
from ensemble_learning.tracking import run_walk_forward

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


def test_window_dataset_materializes_causal_slots_and_missing_mask():
    windows = CausalWindows(
        row_indices=np.array([[0, -1, 2], [1, 2, 3]], dtype=np.int64),
        anchors=pd.DataFrame({"y_5m": [1, 0]}),
        observed_candles=np.array([2, 3]),
    )
    values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
                      dtype=np.float32)
    dataset = _WindowDataset(windows, values, ["y_5m"])
    features, target = dataset[0]
    np.testing.assert_array_equal(features.numpy(), np.array([
        [1.0, 10.0, 1.0], [0.0, 0.0, 0.0], [3.0, 30.0, 1.0],
    ], dtype=np.float32))
    assert target.item() == 1
    assert dataset.feature_tensor_bytes == 2 * 3 * 3 * np.dtype(np.float32).itemsize


def test_purge_keeps_context_but_removes_labels_at_validation_boundary():
    frame = validate_sequence_frame(_neural_development_frame())
    frame = frame.loc[frame.session_date.str.startswith("2026-01")].copy()
    boundary = frame.loc[frame.session_date.eq("2026-01-06"), "decision_time"].min()
    fit_mask = frame.session_date.eq("2026-01-05")
    for horizon in (5, 10, 15):
        frame.loc[fit_mask, f"label_known_at_{horizon}m"] = boundary
    fit, validation = _training_validation_split(frame, validation_fraction=.5)
    assert len(fit) == 70 and len(validation) == 70
    assert fit[["y_5m", "y_10m", "y_15m"]].isna().all().all()


@pytest.mark.parametrize("architecture", ["mlp", "cnn1d", "lstm", "gru"])
@pytest.mark.parametrize("strategy", ["independent", "mtl"])
def test_neural_walk_forward_records_b0_predictions_and_artifacts(
    tmp_path, architecture, strategy,
):
    raw = _neural_development_frame()
    data_path = tmp_path / "development.csv"
    raw.to_csv(data_path, index=False)
    frame = validate_sequence_frame(pd.read_csv(data_path))
    output = run_neural_experiment(
        frame, architecture=architecture, strategy=strategy, data_path=data_path,
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
    training_summary = __import__("json").loads(
        (output / "training_summary.json").read_text()
    )
    assert set(predictions.scale) == {"5m", "10m", "15m"}
    assert set(predictions.fold) == {"fold01", "fold02", "fold03", "fold04"}
    assert predictions.session_date.astype(str).max() <= "2026-05-31"
    assert predictions.target_protocol.eq("b0_retrospective").all()
    assert predictions.p_up.between(0, 1).all()
    assert predictions.p_down.between(0, 1).all()
    assert manifest["test_june_accessed"] is False
    assert manifest["target_columns_used_as_features"] is False
    assert all(
        fit["fit_end"] < fit["inner_validation_start"]
        <= fit["inner_validation_end"] < fit["oof_start"]
        for fit in training_summary["fold_fits"]
    )
    assert len(list((output / "checkpoints").glob("*_best.pt"))) == (
        12 if strategy == "independent" else 4
    )


def test_same_seed_reproduces_and_pairs_independent_with_mtl(tmp_path):
    raw = _neural_development_frame()
    raw.loc[(raw.decision_time.dt.hour == 9) & (raw.decision_time.dt.minute < 30),
            ["y_5m", "y_10m", "y_15m"]] = np.nan
    data_path = tmp_path / "development.csv"
    raw.to_csv(data_path, index=False)
    frame = validate_sequence_frame(pd.read_csv(data_path))
    outputs = {}
    for strategy in ("independent", "mtl", "independent"):
        outputs[strategy, len(outputs)] = run_neural_experiment(
            frame, architecture="mlp", strategy=strategy, data_path=data_path,
            results_root=tmp_path / "results", seed=29, device_name="cpu",
            epochs=1, patience=1, batch_size=16, hidden_size=4,
        )
    first = pd.read_parquet(outputs[("independent", 0)] / "predictions.parquet")
    joint = pd.read_parquet(outputs[("mtl", 1)] / "predictions.parquet")
    repeat = pd.read_parquet(outputs[("independent", 2)] / "predictions.parquet")
    classical = run_walk_forward(frame, data_path=data_path, results_root=tmp_path / "results")
    baseline = pd.read_parquet(classical / "predictions.parquet")
    paired, comparison = paired_metrics(baseline, first)
    assert len(paired) == len(baseline) == len(first)
    assert (comparison.available_left == comparison.available_right).all()
    keys = ["sample_id", "fold", "scale", "seed"]
    assert first.sort_values(keys)[keys].reset_index(drop=True).equals(
        joint.sort_values(keys)[keys].reset_index(drop=True)
    )
    pd.testing.assert_series_equal(
        first.sort_values(keys).p_up.reset_index(drop=True),
        repeat.sort_values(keys).p_up.reset_index(drop=True),
        check_exact=True,
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
