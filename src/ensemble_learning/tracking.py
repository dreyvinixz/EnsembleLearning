"""Immutable local experiment artifacts and walk-forward predictions."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ensemble_learning import HORIZONS
from ensemble_learning.baselines import make_classical_model
from ensemble_learning.contracts import feature_columns, latest_label_confirmation


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def walk_forward_folds(frame: pd.DataFrame, *, start_month: int = 2, end_month: int = 5):
    """Expanding Jan-Apr OOF folds followed by May external validation."""
    dates = pd.to_datetime(frame.session_date)
    folds = []
    for month in range(start_month, end_month + 1):
        train = frame.loc[dates.dt.month < month].copy()
        test = frame.loc[dates.dt.month == month].copy()
        if train.empty or test.empty:
            continue
        cutoff = test.decision_time.min()
        train = train.loc[latest_label_confirmation(train) < cutoff]
        if train.empty:
            continue
        folds.append({"fold": f"fold{month-1:02d}", "train": train, "oof": test,
            "train_start": train.session_date.min(), "train_end": train.session_date.max(),
            "oof_start": test.session_date.min(), "oof_end": test.session_date.max()})
    if not folds:
        raise ValueError("No walk-forward folds have both train and OOF observations")
    return folds


def classification_metrics(y, p):
    y = np.asarray(y, dtype=int); p = np.clip(np.asarray(p, dtype=float), 1e-7, 1-1e-7)
    pred = (p >= .5).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=[0,1], zero_division=0)
    bins=np.linspace(0,1,11); bin_ids=np.minimum(np.digitize(p,bins[1:-1]),9)
    reliability=[]; ece=0.0
    for index in range(10):
        selected=bin_ids==index
        if selected.any():
            confidence=float(p[selected].mean()); accuracy=float(y[selected].mean())
            reliability.append({"bin":index,"n":int(selected.sum()),"confidence":confidence,"accuracy":accuracy})
            ece += float(selected.mean())*abs(confidence-accuracy)
    result={"n": len(y), "mcc": float(matthews_corrcoef(y,pred)),
        "accuracy": float(accuracy_score(y,pred)), "balanced_accuracy": float(balanced_accuracy_score(y,pred)),
        "precision_by_class": precision.tolist(), "recall_by_class": recall.tolist(),
        "f1_by_class": f1.tolist(), "macro_f1": float(f1_score(y,pred,average="macro",zero_division=0)),
        "log_loss": float(log_loss(y,p,labels=[0,1])), "brier": float(brier_score_loss(y,p)),
        "ece_10_bins":float(ece),"reliability_bins":reliability,
        "support_by_class": support.tolist(), "confusion_matrix": confusion_matrix(y,pred,labels=[0,1]).tolist()}
    result["roc_auc"]=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None
    result["average_precision"]=float(average_precision_score(y,p)) if len(np.unique(y))==2 else None
    return result


def _session_bootstrap_ci(frame, *, seed=42, iterations=500):
    grouped = frame.assign(
        _pred=(frame.p_up.to_numpy() >= .5).astype(int),
        _true=frame.y_true.to_numpy(dtype=int),
        _session=frame.session_date.astype(str).to_numpy(),
    )
    sessions = grouped["_session"].unique()
    if len(sessions) < 2:
        return [None, None]
    counts = []
    for session in sessions:
        part = grouped.loc[grouped["_session"] == session]
        truth, pred = part["_true"].to_numpy(), part["_pred"].to_numpy()
        counts.append((
            np.count_nonzero((truth == 1) & (pred == 1)),
            np.count_nonzero((truth == 0) & (pred == 0)),
            np.count_nonzero((truth == 0) & (pred == 1)),
            np.count_nonzero((truth == 1) & (pred == 0)),
        ))
    counts = np.asarray(counts, dtype=np.int64)
    rng = np.random.default_rng(seed)
    scores = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled = counts[rng.integers(0, len(sessions), size=len(sessions))].sum(axis=0)
        tp, tn, fp, fn = sampled
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        scores[index] = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return [float(np.quantile(scores, .025)), float(np.quantile(scores, .975))]


def run_walk_forward(frame: pd.DataFrame, *, model_name="logreg", seed=42,
                     data_path: Path, results_root: Path, experiment_id: str | None = None):
    exp_id = experiment_id or ("EXP_" + uuid.uuid4().hex[:10].upper())
    dates = frame.session_date.astype(str)
    if dates.min() < "2026-01-01" or dates.max() > "2026-05-31":
        raise ValueError("Experiment input must contain development dates Jan-May 2026 only")
    exp_dir = results_root / exp_id
    if exp_dir.exists():
        raise FileExistsError(f"Experiment already exists: {exp_dir}")
    results_root=Path(results_root); data_path=Path(data_path)
    results_root.mkdir(parents=True,exist_ok=True)
    work_dir = results_root / ("." + exp_id + ".building")
    work_dir.mkdir(parents=True, exist_ok=False)
    cols = feature_columns(frame); rows=[]; summaries=[]
    started=time.perf_counter()
    history=[]; model_params={}
    model_dir=work_dir/"checkpoints"; model_dir.mkdir()
    folds = walk_forward_folds(frame)
    for fold in folds:
        fold_rows=[]
        for horizon in HORIZONS:
            label=f"y_{horizon}m"; train=fold["train"]; oof=fold["oof"]
            if set(train[label].unique()) != {0,1}:
                raise ValueError(f"Fold {fold['fold']} {label} lacks both train classes")
            model=make_classical_model(model_name,seed); t0=time.perf_counter()
            model.fit(train[cols],train[label]); fit_seconds=time.perf_counter()-t0
            model_params.setdefault(f"{horizon}m",model.get_params(deep=True))
            t0=time.perf_counter(); p=model.predict_proba(oof[cols])[:,1]; infer=time.perf_counter()-t0
            pred=(p>=.5).astype(int)
            part=pd.DataFrame({"sample_id":oof.instrument.astype(str)+"|"+oof.decision_time.astype(str),
                "model":model_name,"fold":fold["fold"],"seed":seed,"instrument":oof.instrument,
                "session_date":oof.session_date,
                "decision_time":oof.decision_time,"scale":f"{horizon}m","y_true":oof[label].astype(int),
                "p_up":p,"y_pred":pred})
            rows.append(part); fold_rows.append({"horizon":horizon,"metrics":classification_metrics(oof[label],p),
                "fit_seconds":fit_seconds,"inference_seconds":infer,"n_train":len(train),"n_oof":len(oof)})
            history.append({"fold":fold["fold"],"horizon":horizon,"train_rows":len(train),
                "fit_seconds":fit_seconds,"inference_seconds":infer})
            import joblib
            joblib.dump(model,model_dir/f"{fold['fold']}_{horizon}m.joblib")
        summaries.append({"fold":fold["fold"],"train_start":fold["train_start"],"train_end":fold["train_end"],
            "oof_start":fold["oof_start"],"oof_end":fold["oof_end"],"horizons":fold_rows})
    predictions=pd.concat(rows,ignore_index=True); metrics=[]; detailed=[]; robust=[]
    for (fold_name,scale),group in predictions.groupby(["fold","scale"],sort=True):
        metric=classification_metrics(group.y_true,group.p_up)
        detailed.append({"fold":fold_name,"scale":scale,**metric})
        for (day),session in group.groupby("session_date"):
            robust.append({"fold":fold_name,"scale":scale,"session_date":day,
                **classification_metrics(session.y_true,session.p_up)})
    pd.DataFrame(detailed).to_csv(work_dir/"metrics_by_fold_horizon.csv",index=False)
    pd.DataFrame(robust).to_csv(work_dir/"metrics_by_session.csv",index=False)
    by_instrument=[]
    for (scale,instrument),group in predictions.groupby(["scale","instrument"],sort=True):
        by_instrument.append({"scale":scale,"instrument":instrument,**classification_metrics(group.y_true,group.p_up)})
    pd.DataFrame(by_instrument).to_csv(work_dir/"metrics_by_instrument.csv",index=False)
    for scale,g in predictions.groupby("scale",sort=True):
        metrics.append({"scale":scale,**classification_metrics(g.y_true,g.p_up),
            "mcc_session_bootstrap_95ci":_session_bootstrap_ci(g,seed=seed)})
    predictions.to_parquet(work_dir/"predictions.parquet",index=False)
    pd.DataFrame(metrics).to_csv(work_dir/"metrics_by_horizon.csv",index=False)
    pd.DataFrame(history).to_csv(work_dir/"training_history.csv",index=False)
    duration=time.perf_counter()-started
    config={"model":model_name,"seed":seed,"folds":"expanding_jan_apr_oof_plus_may_validation",
        "features":cols,"primary_metric":"MCC (provisional)","threshold":.5,
        "classical_hyperparameters":model_params,"early_stopping":False}
    (work_dir/"config.yaml").write_text(json.dumps(config,indent=2,default=str),encoding="utf-8")
    try: commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError): commit=None
    versions={}
    for package in ("numpy","pandas","scikit-learn","xgboost","pyarrow"):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    manifest={"experiment_id":exp_id,"created_at_utc":datetime.now(UTC).isoformat(),"git_commit":commit,
        "data_file":str(data_path),"data_sha256":_sha(data_path),"python":platform.python_version(),
        "platform":platform.platform(),"cpu":platform.processor(),"device":"CPU classical model",
        "versions":versions,"model":model_name,"seed":seed,"features":cols,
        "folds": [{k:v for k,v in f.items() if k not in ("train","oof")} for f in folds],
        "duration_seconds":duration,"test_june_accessed":False,
        "metric_protocol":{"primary":"MCC provisional","threshold":.5,"ece_bins":10,
            "bootstrap":"500 resamples of whole sessions; diagnostic 95% percentile interval"}}
    (work_dir/"manifest.json").write_text(json.dumps(manifest,indent=2,default=str),encoding="utf-8")
    (work_dir/"metrics.json").write_text(json.dumps({"primary_metric":"mcc_provisional","by_fold":summaries,
        "aggregate_by_horizon":metrics,"duration_seconds":duration},indent=2),encoding="utf-8")
    os.replace(work_dir,exp_dir)
    return exp_dir
