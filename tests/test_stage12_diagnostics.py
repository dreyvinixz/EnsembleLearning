from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_stage11_histories_requires_all_runs_and_hashes_sources(tmp_path):
    exporter = load_script("export_stage11_histories", "scripts/export_stage11_histories.py")
    results = tmp_path / "results"
    for run_id in sorted(exporter.EXPECTED_RUNS):
        run_dir = results / run_id
        run_dir.mkdir(parents=True)
        pd.DataFrame([{
            "fold": "fold01", "architecture": "mlp", "strategy": "independent",
            "seed": 42, "horizon": 5, "epoch": 1, "train_loss": .5,
            "validation_loss": .6, "validation_mcc": .2,
            "validation_accuracy": .6, "phase": None,
        }]).to_csv(run_dir / "training_history.csv", index=False)

    output, manifest_path = tmp_path / "history.csv", tmp_path / "manifest.json"
    exporter.export(results, output, manifest_path)

    history = pd.read_csv(output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(history) == 16
    assert history.run_id.nunique() == 16
    assert manifest["run_count"] == 16
    assert manifest["contains_predictions"] is False
    assert manifest["June_accessed"] is False
    assert len(manifest["source_sha256_by_run"]) == 16


def test_diagnostic_summary_averages_seeds_and_refuses_overwrite(tmp_path):
    pytest.importorskip("matplotlib")
    plotting = load_script("plot_stage12_diagnostics", "scripts/plot_stage12_diagnostics.py")
    rows = []
    for period in ("oof_feb_apr", "validation_may"):
        for scale in ("5m", "10m", "15m"):
            for seed, mcc, accuracy in ((42, .4, .7), (43, .5, .8)):
                rows.append({"architecture": "mlp", "strategy": "independent",
                             "seed": seed, "scale": scale, "period": period,
                             "mcc": mcc, "accuracy": accuracy, "n": 100})
    frame = pd.DataFrame(rows)
    output = tmp_path / "summary.csv"
    plotting.summarize_metrics(frame, output)
    summary = pd.read_csv(output)
    assert len(summary) == 6
    assert summary.n_seeds.eq(2).all()
    assert summary.mcc_mean.eq(.45).all()
    assert summary.accuracy_mean.eq(.75).all()
    with pytest.raises(FileExistsError):
        plotting.summarize_metrics(frame, output)


def test_diagnostic_plots_include_learning_curves_mcc_and_accuracy(tmp_path):
    pytest.importorskip("matplotlib")
    plotting = load_script("plot_stage12_diagnostics", "scripts/plot_stage12_diagnostics.py")
    history_rows = []
    for architecture in ("mlp", "cnn1d", "lstm", "gru"):
        for strategy in ("independent", "mtl"):
            for epoch in (1, 2):
                history_rows.append({
                    "architecture": architecture, "strategy": strategy, "seed": 42,
                    "fold": "fold01", "horizon": 5, "epoch": epoch,
                    "train_loss": .6 - epoch * .02, "validation_loss": .6 - epoch * .01,
                    "validation_mcc": .2 + epoch * .03,
                    "validation_accuracy": .6 + epoch * .02, "phase": None,
                })
    history = pd.DataFrame(history_rows)
    plotting.plot_curves(history, tmp_path)

    metrics = pd.DataFrame([
        {"architecture": architecture, "strategy": strategy, "seed": seed,
         "scale": scale, "period": period, "mcc": .4, "accuracy": .7, "n": 100}
        for architecture, strategy in (("LogisticRegression", "INDEPENDENT"),
                                       ("MLP", "INDEPENDENT"), ("MLP", "MTL"))
        for seed in (42, 43)
        for scale in ("5m", "10m", "15m")
        for period in ("oof_feb_apr", "validation_may")
    ])
    plotting.plot_performance(metrics, tmp_path, "mcc")
    plotting.plot_performance(metrics, tmp_path, "accuracy")
    assert (tmp_path / "neural-learning-curves.png").is_file()
    assert (tmp_path / "neural-mcc.png").is_file()
    assert (tmp_path / "neural-accuracy.png").is_file()


def test_complementarity_summary_enforces_exact_keys_and_labels():
    complementarity = load_script(
        "summarize_stage12_complementarity",
        "scripts/summarize_stage12_complementarity.py",
    )
    frames = {}
    for name, probability in (("LogisticRegression", .65),
                              ("MLP_INDEPENDENT_seed42", .66), ("Stacking", .7)):
        rows = []
        for fold in ("fold01", "fold02", "fold03", "fold04"):
            for scale in ("5m", "10m", "15m"):
                for day, label in (("2026-03-02", 0), ("2026-05-04", 1)):
                    rows.append({
                        "instrument": "WING26", "session_date": day,
                        "decision_time": pd.Timestamp(f"{day} 13:00:00", tz="UTC"),
                        "scale": scale, "fold": fold, "y_true": label,
                        "p_up": probability if label else 1 - probability,
                    })
        frames[name] = pd.DataFrame(rows)
    summary = complementarity.summarize(frames)
    assert set(summary.period) == {"oof_feb_apr", "oof_mar_apr", "validation_may"}
    assert summary.n.gt(0).all()
    assert summary.prediction_agreement.eq(1).all()

    incomplete = {name: frame.copy() for name, frame in frames.items()}
    missing_index = incomplete["Stacking"].index[
        incomplete["Stacking"].fold.eq("fold02")
    ][0]
    incomplete["Stacking"] = incomplete["Stacking"].drop(index=missing_index)
    with pytest.raises(ValueError, match="common sample keys"):
        complementarity.summarize(incomplete)
