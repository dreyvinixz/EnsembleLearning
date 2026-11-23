"""Tracked causal PyTorch training for independent and multiscale B0 tasks."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ensemble_learning import HORIZONS
from ensemble_learning.contracts import (
    feature_columns,
    latest_label_confirmation,
    validate_sequence_frame,
)
from ensemble_learning.sequence_models import (
    NEURAL_ARCHITECTURES,
    IndependentSequenceClassifier,
    MultiTaskSequenceClassifier,
)
from ensemble_learning.sequences import CausalWindows, build_causal_windows, summarize_eligibility
from ensemble_learning.tracking import (
    _session_bootstrap_ci,
    classification_metrics,
    walk_forward_folds,
)

STRATEGIES = ("independent", "mtl")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _training_validation_split(
    training_context: pd.DataFrame,
    *,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sessions = sorted(training_context.session_date.astype(str).unique())
    if len(sessions) < 2 or not 0 < validation_fraction < 1:
        raise ValueError("Internal early-stopping split needs >=2 sessions and a fraction in (0,1)")
    validation_count = max(1, int(np.ceil(len(sessions) * validation_fraction)))
    fit_sessions, validation_sessions = sessions[:-validation_count], sessions[-validation_count:]
    fit = training_context.loc[training_context.session_date.astype(str).isin(fit_sessions)].copy()
    validation = training_context.loc[
        training_context.session_date.astype(str).isin(validation_sessions)
    ].copy()
    first_validation_decision = validation.decision_time.min()
    unavailable = latest_label_confirmation(fit).ge(first_validation_decision)
    fit.loc[unavailable, [f"y_{h}m" for h in HORIZONS]] = np.nan
    if fit.session_date.nunique() == 0 or validation.session_date.nunique() == 0:
        raise ValueError("Purging leaves an empty internal train or validation partition")
    return fit, validation


def _fit_normalizer(frame: pd.DataFrame, columns: list[str]) -> dict[str, list[float]]:
    values = frame[columns].to_numpy(dtype=np.float64, na_value=np.nan)
    medians = np.nanmedian(values, axis=0)
    if not np.isfinite(medians).all():
        missing = [columns[index] for index, value in enumerate(medians) if not np.isfinite(value)]
        raise ValueError(f"Features entirely missing in a training fold: {missing}")
    imputed = np.where(np.isfinite(values), values, medians)
    means = imputed.mean(axis=0)
    scales = imputed.std(axis=0)
    scales[scales == 0] = 1.0
    return {"median": medians.tolist(), "mean": means.tolist(), "scale": scales.tolist()}


def _transform_features(frame: pd.DataFrame, columns: list[str], normalizer) -> np.ndarray:
    values = frame[columns].to_numpy(dtype=np.float64, na_value=np.nan)
    medians = np.asarray(normalizer["median"], dtype=np.float64)
    values = np.where(np.isfinite(values), values, medians)
    values = (values - np.asarray(normalizer["mean"])) / np.asarray(normalizer["scale"])
    if not np.isfinite(values).all():
        raise ValueError("Feature normalization produced non-finite values")
    return values.astype(np.float32)


class _WindowDataset(Dataset):
    def __init__(
        self,
        windows: CausalWindows,
        feature_values: np.ndarray,
        targets: list[str],
    ):
        self.windows = windows.row_indices
        self.y = windows.anchors[targets].to_numpy(dtype=np.int64)
        self.features = np.zeros(
            (len(self.windows), self.windows.shape[1], feature_values.shape[1] + 1),
            dtype=np.float32,
        )
        for slot in range(self.windows.shape[1]):
            row_ids = self.windows[:, slot]
            present = row_ids >= 0
            self.features[present, slot, :-1] = feature_values[row_ids[present]]
            self.features[present, slot, -1] = 1.0
        self.feature_tensor = torch.from_numpy(self.features)
        self.target_tensor = torch.from_numpy(self.y)
        self.feature_tensor_bytes = self.features.nbytes

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        labels = self.target_tensor[index]
        if labels.numel() == 1:
            labels = labels[0]
        return self.feature_tensor[index], labels


def _build_dataset(
    context: pd.DataFrame,
    *,
    columns: list[str],
    anchor_targets: list[str],
    output_targets: list[str],
    normalizer,
    window_minutes: int,
    min_observed: int,
) -> tuple[_WindowDataset, CausalWindows]:
    windows = build_causal_windows(
        context,
        feature_columns=columns,
        target_columns=anchor_targets,
        window_minutes=window_minutes,
        min_observed=min_observed,
    )
    if not len(windows.anchors):
        raise ValueError("No B0 target anchors pass the causal window and coverage gate")
    values = _transform_features(context, columns, normalizer)
    return _WindowDataset(windows, values, output_targets), windows


def _class_weights(dataset: _WindowDataset, task_index: int) -> torch.Tensor:
    labels = dataset.y if dataset.y.ndim == 1 else dataset.y[:, task_index]
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Every training target needs both B0 classes")
    return torch.tensor(len(labels) / (2.0 * counts), dtype=torch.float32)


def _prediction_map(output, strategy: str, target_list: list[int]):
    if strategy == "mtl":
        return {horizon: output[horizon] for horizon in target_list}
    return {target_list[0]: output}


@torch.no_grad()
def _evaluate(model, loader, *, strategy: str, target_list: list[int], device):
    model.eval()
    probabilities = {horizon: [] for horizon in target_list}
    truths = {horizon: [] for horizon in target_list}
    losses = {horizon: [] for horizon in target_list}
    for features, labels in loader:
        features = features.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        with _autocast_context(device, getattr(model, "_ensemble_precision", "fp32")):
            outputs = _prediction_map(model(features), strategy, target_list)
        for index, horizon in enumerate(target_list):
            truth = labels if strategy == "independent" else labels[:, index]
            logits = outputs[horizon]
            losses[horizon].append(F.cross_entropy(logits, truth, reduction="sum").item())
            probabilities[horizon].extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
            truths[horizon].extend(truth.cpu().tolist())
    count = len(next(iter(truths.values())))
    return {
        horizon: {
            "y_true": np.asarray(truths[horizon], dtype=np.int64),
            "p_up": np.asarray(probabilities[horizon], dtype=np.float64),
            "loss": float(sum(losses[horizon]) / count),
        }
        for horizon in target_list
    }


def _autocast_context(device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _new_model(architecture, strategy, *, input_size, hidden_size, window_minutes):
    options = {"sequence_length": window_minutes} if architecture == "mlp" else {}
    if strategy == "independent":
        return IndependentSequenceClassifier(
            architecture, input_size, hidden_size=hidden_size, **options
        )
    return MultiTaskSequenceClassifier(
        architecture, input_size, hidden_size=hidden_size, **options
    )


def _train_one_fold(
    *,
    architecture: str,
    strategy: str,
    fold_name: str,
    fit_dataset: _WindowDataset,
    early_validation_dataset: _WindowDataset,
    prediction_dataset: _WindowDataset,
    target_list: list[int],
    input_size: int,
    hidden_size: int,
    window_minutes: int,
    seed: int,
    device,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    checkpoint_dir: Path,
    columns: list[str],
    normalizer,
    precision: str,
    num_workers: int,
    compile_model: bool,
    allow_tf32: bool,
):
    _seed(seed)
    model = _new_model(
        architecture, strategy, input_size=input_size, hidden_size=hidden_size,
        window_minutes=window_minutes,
    ).to(device)
    model._ensemble_precision = precision
    uncompiled_model = model
    compile_seconds = 0.0
    if compile_model:
        compile_started = time.perf_counter()
        model = torch.compile(model)
        compile_seconds = time.perf_counter() - compile_started
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    weights = {horizon: _class_weights(fit_dataset, index).to(device)
               for index, horizon in enumerate(target_list)}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
        fused=device.type == "cuda",
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and precision == "fp16"
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        fit_dataset, batch_size=batch_size, shuffle=True, generator=generator,
        num_workers=num_workers, pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0, drop_last=False,
    )
    validation_loader = DataLoader(
        early_validation_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    prediction_loader = DataLoader(
        prediction_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    target_suffix = "_".join(f"{h}m" for h in target_list)
    checkpoint_base = f"{fold_name}_{architecture}_{strategy}_{target_suffix}"
    best_path = checkpoint_dir / f"{checkpoint_base}_best.pt"
    last_path = checkpoint_dir / f"{checkpoint_base}_last.pt"
    history = []
    best_loss, best_epoch, stale_epochs = float("inf"), 0, 0
    best_state = None
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        train_loss_sum, train_count = 0.0, 0
        for features, labels in train_loader:
            features = features.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, precision):
                outputs = _prediction_map(model(features), strategy, target_list)
                task_losses = []
                for index, horizon in enumerate(target_list):
                    truth = labels if strategy == "independent" else labels[:, index]
                    task_losses.append(F.cross_entropy(
                        outputs[horizon].float(), truth, weight=weights[horizon]
                    ))
                loss = torch.stack(task_losses).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size_actual = len(features)
            train_loss_sum += loss.item() * batch_size_actual
            train_count += batch_size_actual

        validation = _evaluate(
            model, validation_loader, strategy=strategy, target_list=target_list, device=device
        )
        mean_validation_loss = float(np.mean([result["loss"] for result in validation.values()]))
        train_loss = train_loss_sum / max(train_count, 1)
        epoch_seconds = time.perf_counter() - epoch_start
        improved = mean_validation_loss < best_loss - 1e-8
        if improved:
            best_loss, best_epoch, stale_epochs = mean_validation_loss, epoch, 0
            best_state = {name: value.detach().cpu().clone()
                          for name, value in uncompiled_model.state_dict().items()}
            torch.save({
                "state_dict": best_state, "architecture": architecture, "strategy": strategy,
                "target_protocol": "b0_retrospective", "target_list": target_list,
                "feature_columns": columns, "normalizer": normalizer,
                "window_minutes": window_minutes, "minimum_observed": 48,
                "best_epoch": best_epoch, "seed": seed,
            }, best_path)
        else:
            stale_epochs += 1
        last_state = {name: value.detach().cpu()
                      for name, value in uncompiled_model.state_dict().items()}
        torch.save({"state_dict": last_state, "epoch": epoch}, last_path)

        for horizon, result in validation.items():
            metric = classification_metrics(result["y_true"], result["p_up"])
            history.append({
                "fold": fold_name, "architecture": architecture, "strategy": strategy,
                "seed": seed, "horizon": horizon, "epoch": epoch,
                "train_loss": train_loss, "validation_loss": result["loss"],
                "validation_mcc": metric["mcc"],
                "validation_accuracy": metric["accuracy"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": epoch_seconds, "best_epoch_so_far": best_epoch,
            })
        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("Early stopping did not produce a best checkpoint")
    uncompiled_model.load_state_dict(best_state)
    uncompiled_model.to(device)
    predictions = _evaluate(
        model, prediction_loader, strategy=strategy, target_list=target_list, device=device
    )
    elapsed = time.perf_counter() - started
    return predictions, history, {
        "fold": fold_name, "architecture": architecture, "strategy": strategy,
        "best_epoch": best_epoch, "epochs_run": epoch, "fit_seconds": elapsed,
        "best_inner_validation_loss": best_loss, "parameter_count": parameter_count,
        "compile_setup_seconds": compile_seconds, "precision": precision,
        "loader_workers": num_workers, "pin_memory": device.type == "cuda",
        "tf32_allowed": allow_tf32 and device.type == "cuda",
        "fused_adamw": device.type == "cuda",
        "checkpoints": [best_path.name, last_path.name],
    }


def _device_info(device) -> dict:
    info = {"device": str(device), "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info.update({"gpu_name": properties.name, "gpu_total_memory_bytes": properties.total_memory,
                     "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)})
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        info["process_peak_rss_bytes"] = int(rss * (1024 if platform.system() == "Linux" else 1))
    except (ImportError, AttributeError):
        info["process_peak_rss_bytes"] = None
    return info


def run_neural_experiment(
    frame: pd.DataFrame,
    *,
    architecture: str,
    strategy: str,
    data_path: Path,
    results_root: Path,
    seed: int = 42,
    experiment_id: str | None = None,
    device_name: str = "cuda",
    window_minutes: int = 60,
    min_observed: int = 48,
    validation_fraction: float = 0.2,
    hidden_size: int = 32,
    epochs: int = 40,
    patience: int = 6,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    precision: str = "fp32",
    num_workers: int = 0,
    compile_model: bool = False,
    allow_tf32: bool = False,
) -> Path:
    """Train one neural architecture/strategy over the Stage 10 folds.

    The frame must contain all observed candles in each session, including
    unlabeled context bars, and must end no later than May 2026. B0 labels are
    supervision only; all feature selection is limited to ``x_`` columns.
    """
    if architecture not in NEURAL_ARCHITECTURES or strategy not in STRATEGIES:
        raise ValueError("Unknown architecture or learning strategy")
    if min_observed != 48 or window_minutes != 60:
        raise ValueError("The current Stage 11 protocol fixes windows at 60 minutes and 48 candles")
    if epochs < 1 or patience < 1 or batch_size < 1 or hidden_size < 1:
        raise ValueError("Training sizes, epochs and patience must be positive")
    if precision not in {"fp32", "bf16", "fp16"} or num_workers < 0:
        raise ValueError("precision must be fp32/bf16/fp16 and num_workers cannot be negative")
    frame = validate_sequence_frame(frame)
    dates = frame.session_date.astype(str)
    if dates.min() < "2026-01-01" or dates.max() > "2026-05-31":
        raise ValueError("Neural development input must contain Jan-May 2026 only; June is reserved")
    columns = feature_columns(frame)
    target_columns = [f"y_{horizon}m" for horizon in HORIZONS]
    if not frame[target_columns].notna().all(axis=1).any():
        raise ValueError("No common complete B0 target rows exist for the three scales")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type != "cuda" and (precision != "fp32" or allow_tf32):
        raise ValueError("bf16/fp16 and TF32 options require CUDA")
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not support bfloat16")
    if compile_model and shutil.which("cc") is None and shutil.which("gcc") is None:
        raise RuntimeError("--compile needs a C compiler (cc or gcc) for the Triton backend")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

    exp_id = experiment_id or f"EXP_{uuid.uuid4().hex[:10].upper()}"
    root = Path(results_root)
    final_dir = root / exp_id
    if final_dir.exists():
        raise FileExistsError(final_dir)
    root.mkdir(parents=True, exist_ok=True)
    building = root / f".{exp_id}.building"
    building.mkdir(exist_ok=False)
    checkpoint_dir = building / "checkpoints"
    checkpoint_dir.mkdir()
    started = time.perf_counter()
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        folds = walk_forward_folds(frame)
        target_list = list(HORIZONS)
        predictions, histories, fits, coverage = [], [], [], []

        for fold in folds:
            fit_context, inner_validation_context = _training_validation_split(
                fold["train"], validation_fraction=validation_fraction
            )
            normalizer = _fit_normalizer(fit_context, columns)
            target_runs = [[horizon] for horizon in HORIZONS] if strategy == "independent" else [target_list]
            for run_targets in target_runs:
                output_targets = [f"y_{horizon}m" for horizon in run_targets]
                fit_dataset, fit_windows = _build_dataset(
                    fit_context, columns=columns, anchor_targets=target_columns,
                    output_targets=output_targets, normalizer=normalizer,
                    window_minutes=window_minutes, min_observed=min_observed,
                )
                validation_dataset, validation_windows = _build_dataset(
                    inner_validation_context, columns=columns, anchor_targets=target_columns,
                    output_targets=output_targets, normalizer=normalizer,
                    window_minutes=window_minutes, min_observed=min_observed,
                )
                prediction_dataset, prediction_windows = _build_dataset(
                    fold["oof"], columns=columns, anchor_targets=target_columns,
                    output_targets=output_targets, normalizer=normalizer,
                    window_minutes=window_minutes, min_observed=min_observed,
                )
                if run_targets == target_runs[0]:
                    for partition, context, windows in (
                        ("fit", fit_context, fit_windows),
                        ("inner_validation", inner_validation_context, validation_windows),
                        ("oof", fold["oof"], prediction_windows),
                    ):
                        coverage.append({"fold": fold["fold"], "partition": partition,
                            **summarize_eligibility(context, windows, target_columns)})
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                fold_predictions, fold_history, fit_manifest = _train_one_fold(
                    architecture=architecture, strategy=strategy, fold_name=fold["fold"],
                    fit_dataset=fit_dataset, early_validation_dataset=validation_dataset,
                    prediction_dataset=prediction_dataset, target_list=run_targets,
                    input_size=len(columns) + 1, hidden_size=hidden_size,
                    window_minutes=window_minutes, seed=seed, device=device,
                    epochs=epochs, patience=patience, batch_size=batch_size,
                    learning_rate=learning_rate, weight_decay=weight_decay,
                    checkpoint_dir=checkpoint_dir, columns=columns, normalizer=normalizer,
                    precision=precision, num_workers=num_workers,
                    compile_model=compile_model, allow_tf32=allow_tf32,
                )
                fit_manifest["fit_start"] = fit_context.session_date.min()
                fit_manifest["fit_end"] = fit_context.session_date.max()
                histories.extend(fold_history)
                fits.append({
                    **fit_manifest, "trained_horizons": run_targets,
                    "train_start": fold["train_start"], "train_end": fold["train_end"],
                    "inner_validation_start": inner_validation_context.session_date.min(),
                    "inner_validation_end": inner_validation_context.session_date.max(),
                    "oof_start": fold["oof_start"], "oof_end": fold["oof_end"],
                    "n_fit_anchors": len(fit_dataset),
                    "n_inner_validation_anchors": len(validation_dataset),
                    "n_prediction_anchors": len(prediction_dataset),
                    "materialized_feature_bytes": (
                        fit_dataset.feature_tensor_bytes
                        + validation_dataset.feature_tensor_bytes
                        + prediction_dataset.feature_tensor_bytes
                    ),
                })
                for horizon, result in fold_predictions.items():
                    anchors = prediction_windows.anchors
                    predictions.append(pd.DataFrame({
                        "sample_id": anchors.instrument.astype(str) + "|" + anchors.decision_time.astype(str),
                        "architecture": architecture, "strategy": strategy,
                        "target_protocol": "b0_retrospective", "fold": fold["fold"],
                        "seed": seed, "instrument": anchors.instrument,
                        "session_date": anchors.session_date, "decision_time": anchors.decision_time,
                        "scale": f"{horizon}m", "y_true": result["y_true"],
                        "p_up": result["p_up"], "p_down": 1.0 - result["p_up"],
                        "y_pred": (result["p_up"] >= 0.5).astype(int),
                        "observed_candles": prediction_windows.observed_candles,
                    }))
                del (
                    fit_dataset, validation_dataset, prediction_dataset,
                    prediction_windows, fold_predictions, fold_history,
                )

        prediction_frame = pd.concat(predictions, ignore_index=True)
        pd.DataFrame(coverage).to_csv(building / "coverage.csv", index=False)
        by_fold, by_session, by_instrument = [], [], []
        for (fold_name, scale), group in prediction_frame.groupby(["fold", "scale"], sort=True):
            by_fold.append({"fold": fold_name, "scale": scale,
                            **classification_metrics(group.y_true, group.p_up)})
            for day, session in group.groupby("session_date", sort=True):
                by_session.append({"fold": fold_name, "scale": scale, "session_date": day,
                                   **classification_metrics(session.y_true, session.p_up)})
        for (scale, instrument), group in prediction_frame.groupby(
            ["scale", "instrument"], sort=True
        ):
            by_instrument.append({"scale": scale, "instrument": instrument,
                                  **classification_metrics(group.y_true, group.p_up)})

        by_horizon = []
        evaluation_sets = {
            "development_oof_feb_apr": prediction_frame.loc[prediction_frame.fold != "fold04"],
            "may_validation": prediction_frame.loc[prediction_frame.fold == "fold04"],
        }
        for evaluation_set, evaluation_frame in evaluation_sets.items():
            for scale, group in evaluation_frame.groupby("scale", sort=True):
                by_horizon.append({
                    "evaluation_set": evaluation_set, "scale": scale,
                    **classification_metrics(group.y_true, group.p_up),
                    "mcc_session_bootstrap_95ci": _session_bootstrap_ci(group, seed=seed),
                })

        prediction_frame.to_parquet(building / "predictions.parquet", index=False)
        pd.DataFrame(by_fold).to_csv(building / "metrics_by_fold_horizon.csv", index=False)
        pd.DataFrame(by_session).to_csv(building / "metrics_by_session.csv", index=False)
        pd.DataFrame(by_instrument).to_csv(building / "metrics_by_instrument.csv", index=False)
        pd.DataFrame(by_horizon).to_csv(building / "metrics_by_horizon.csv", index=False)
        pd.DataFrame(histories).to_csv(building / "training_history.csv", index=False)
        (building / "training_summary.json").write_text(
            json.dumps({"fold_fits": fits}, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        config = {
            "target_protocol": "b0_retrospective",
            "class_mapping": {"0": "DOWNTREND", "1": "UPTREND"},
            "architecture": architecture, "strategy": strategy, "seed": seed,
            "window_minutes": window_minutes, "minimum_observed_candles": min_observed,
            "first_decision_local": "10:00", "features": columns,
            "mask_channel": "1 if source candle observed in slot, otherwise 0",
            "horizons": list(HORIZONS), "hidden_size": hidden_size,
            "epochs": epochs, "early_stopping_patience": patience,
            "internal_validation_fraction_by_session": validation_fraction,
            "batch_size": batch_size, "learning_rate": learning_rate,
            "weight_decay": weight_decay, "device_requested": device_name,
            "precision": precision, "num_workers": num_workers,
            "pin_memory": device.type == "cuda", "compile_model": compile_model,
            "allow_tf32": allow_tf32, "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "optimizer_fused_on_cuda": device.type == "cuda",
            "primary_metric": "MCC (provisional)", "probability_threshold": 0.5,
            "fold_protocol": "Expanding Jan-Apr OOF; May external validation; June reserved",
        }
        (building / "config.yaml").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        input_path = Path(data_path)
        versions = {}
        for package in ("numpy", "pandas", "scikit-learn", "pyarrow", "torch"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                continue
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            commit = None
        manifest = {
            "experiment_id": exp_id, "created_at_utc": datetime.now(UTC).isoformat(),
            "git_commit": commit, "data_file": str(input_path),
            "data_sha256": _sha256(input_path), "python": platform.python_version(),
            "platform": platform.platform(), "cpu": platform.processor(),
            "environment": _device_info(device), "versions": versions,
            "architecture": architecture, "strategy": strategy, "seed": seed,
            "target_protocol": "b0_retrospective", "target_class_mapping": {
                "0": "DOWNTREND", "1": "UPTREND",
            },
            "feature_columns": columns, "feature_columns_are_causal": True,
            "target_columns_used_as_features": False,
            "folds": fits, "duration_seconds": time.perf_counter() - started,
            "test_june_accessed": False,
            "label_release_assumption": "Conservative simulated post-session B0 availability; not historically observed.",
            "metric_protocol": {
                "primary": "MCC provisional", "threshold": 0.5, "ece_bins": 10,
                "bootstrap": "500 resamples of whole sessions; diagnostic percentile interval",
            },
        }
        (building / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        (building / "metrics.json").write_text(json.dumps({
            "primary_metric": "mcc_provisional", "by_fold_horizon": by_fold,
            "by_evaluation_set_and_horizon": by_horizon,
            "duration_seconds": manifest["duration_seconds"],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(building, final_dir)
    except Exception:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return final_dir
