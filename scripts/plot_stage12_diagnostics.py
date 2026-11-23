"""Render neural training and paired-population diagnostic figures.

The script consumes aggregate Stage 11 metrics committed to the repository and
training histories from the Linux experiment host. It never loads predictions
or accesses the reserved June holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ARCHITECTURES = ["MLP", "CNN1D", "LSTM", "GRU"]
STRATEGIES = ["INDEPENDENT", "MTL"]
SCALES = ["5m", "10m", "15m"]
COLORS = {"INDEPENDENT": "#0072B2", "MTL": "#D55E00", "Logistic Regression": "#30343B"}
SCALE_COLORS = {"5m": "#0072B2", "10m": "#009E73", "15m": "#7B61A8"}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.with_suffix(".png").exists() or path.with_suffix(".pdf").exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plot_curves(history: pd.DataFrame, output: Path) -> None:
    required = {"architecture", "strategy", "seed", "fold", "horizon", "epoch",
                "train_loss", "validation_loss", "validation_mcc", "validation_accuracy",
                "phase"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"History CSV lacks columns: {sorted(missing)}")
    selection = history.loc[history.phase.isna() | history.phase.eq("")].copy()
    selection = selection.loc[selection.epoch.notna()]
    if selection.empty:
        raise ValueError("No early-stopping selection epochs found")
    for col in ("epoch", "horizon", "train_loss", "validation_loss", "validation_mcc",
                "validation_accuracy"):
        selection[col] = pd.to_numeric(selection[col], errors="coerce")
    metrics = [
        ("train_loss", "Loss de treino"),
        ("validation_loss", "Loss de validação interna"),
        ("validation_mcc", "MCC de validação interna"),
        ("validation_accuracy", "Acurácia de validação interna"),
    ]
    fig, axes = plt.subplots(4, 4, figsize=(16, 11), sharex=True)
    for row, architecture in enumerate(ARCHITECTURES):
        for col, (metric, label) in enumerate(metrics):
            ax = axes[row, col]
            for strategy in STRATEGIES:
                part = selection.loc[
                    selection.strategy.str.upper().eq(strategy)
                    & selection.architecture.str.upper().eq(architecture)
                ]
                grouped = part.groupby("epoch", observed=True)[metric].agg(["mean", "std"])
                grouped = grouped.dropna(subset=["mean"])
                if grouped.empty:
                    continue
                x = grouped.index.to_numpy(dtype=int)
                mean = grouped["mean"].to_numpy()
                spread = grouped["std"].fillna(0).to_numpy()
                ax.plot(x, mean, marker="o", linewidth=1.8, color=COLORS[strategy],
                        label=strategy)
                ax.fill_between(x, mean - spread, mean + spread,
                                color=COLORS[strategy], alpha=.13)
            ax.set_title(f"{architecture} · {label}" if row == 0 else architecture,
                         loc="left", fontsize=9, fontweight="bold")
            ax.set_xticks(range(1, 6))
            ax.grid(axis="y", color="#D9E0E8", linewidth=.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel("Valor médio")
            if row == 3:
                ax.set_xlabel("Época")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, .95), ncol=2,
               frameon=False)
    fig.suptitle("Curvas de seleção neural — Independent vs MTL", x=.08, y=.995,
                 ha="left", fontsize=16, fontweight="bold", color="#172B4D")
    fig.text(.08, .015,
             "Média ± 1 desvio-padrão descritivo; máximo de 5 épocas, paciência 2. "
             "Refit final omitido: não tem métricas de validação.", fontsize=9, color="#526274")
    fig.tight_layout(rect=(0, .055, 1, .92), h_pad=1.5, w_pad=1.0)
    _save(fig, output / "neural-learning-curves")


def plot_performance(metrics: pd.DataFrame, output: Path, metric: str) -> None:
    expected = {"architecture", "strategy", "seed", "scale", "period", "mcc", "accuracy"}
    missing = expected - set(metrics.columns)
    if missing:
        raise ValueError(f"Metric CSV lacks columns: {sorted(missing)}")
    selected_periods = [("oof_feb_apr", "OOF fev–abr"),
                        ("validation_may", "Validação maio")]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    offsets = {"5m": -.22, "10m": 0.0, "15m": .22}
    for ax, (period, period_label) in zip(axes, selected_periods, strict=True):
        data = metrics.loc[metrics.period.eq(period)].copy()
        data["model"] = data.architecture.fillna("").astype(str).str.upper()
        data.loc[data.model.eq("LOGISTICREGRESSION"), "model"] = "Logistic Regression"
        data.loc[data.model.ne("Logistic Regression"), "model"] = (
            data.loc[data.model.ne("Logistic Regression"), "model"] + " "
            + data.loc[data.model.ne("Logistic Regression"), "strategy"].astype(str).str.upper()
        )
        if data.empty:
            raise ValueError(f"No metric records for {period}")
        ordered_models = ["Logistic Regression"] + [
            f"{architecture} {strategy}" for architecture in ARCHITECTURES
            for strategy in STRATEGIES
        ]
        for model_i, model in enumerate(ordered_models):
            subset = data.loc[data.model.eq(model)]
            if subset.empty:
                continue
            # Plot each seed as a faint point; bold diamond is the two-seed mean.
            for scale in SCALES:
                rows = subset.loc[subset.scale.eq(scale)]
                x = model_i + offsets[scale]
                color = SCALE_COLORS[scale]
                ax.scatter([x] * len(rows), rows[metric], s=21, color=color, alpha=.42,
                           edgecolors="none", zorder=2)
                if not rows.empty:
                    ax.scatter([x], [rows[metric].mean()], s=44, marker="D", color=color,
                               edgecolors="white", linewidths=.6, zorder=3)
        ax.set_title(period_label + " · losangos = média; pontos = execuções/seeds",
                     loc="left", fontsize=10, color="#374151")
        ax.set_ylabel("MCC" if metric == "mcc" else "Acurácia")
        ax.grid(axis="y", color="#D9E0E8", linewidth=.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xticks(range(len(ordered_models)), [
        "LR" if model == "Logistic Regression" else model.replace(" ", "\n", 1)
        for model in ordered_models
    ], rotation=25, ha="right")
    axes[-1].set_xlabel("Modelo e estratégia")
    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], marker="D", linestyle="", color=color, label=scale,
                     markersize=7) for scale, color in SCALE_COLORS.items()]
    fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(.97, .985),
               title="Escala B0", frameon=False, ncol=3)
    fig.suptitle(f"Redes neurais e baseline — {('MCC' if metric == 'mcc' else 'Acurácia')}",
                 x=.08, y=.99, ha="left", fontsize=16, fontweight="bold", color="#172B4D")
    fig.text(.08, .015,
             "OOF fev–abr e validação maio são exibidos separadamente. "
             "Métricas descritivas, sem IC; duas seeds. Junho não acessado.",
             fontsize=9, color="#526274")
    fig.tight_layout(rect=(0, .11, 1, .93))
    _save(fig, output / f"neural-{metric}")


def summarize_metrics(metrics: pd.DataFrame, output: Path) -> Path:
    selected = metrics.loc[metrics.period.isin(["oof_feb_apr", "validation_may"])].copy()
    selected["architecture"] = selected.architecture.fillna("LogisticRegression")
    group_cols = ["architecture", "strategy", "period", "scale"]
    summary = selected.groupby(group_cols, dropna=False, observed=True).agg(
        n_seeds=("seed", "nunique"),
        mcc_mean=("mcc", "mean"), mcc_sd=("mcc", "std"),
        accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        n_min=("n", "min"), n_max=("n", "max"),
    ).reset_index()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite summary {output}")
    summary.to_csv(output, index=False)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-csv", type=Path, required=True,
                        help="Aggregate training histories exported from Linux runs")
    parser.add_argument("--metrics-csv", type=Path, required=True,
                        help="Committed Stage 11 aggregate metrics CSV")
    parser.add_argument("--source-manifest", type=Path, required=True,
                        help="Committed Stage 11 manifest for snapshot/commit provenance")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    history, metrics = pd.read_csv(args.history_csv), pd.read_csv(args.metrics_csv)
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    plot_curves(history, args.output_dir)
    plot_performance(metrics, args.output_dir, "mcc")
    plot_performance(metrics, args.output_dir, "accuracy")
    summary_path = summarize_metrics(metrics, args.summary_output)
    artifacts = sorted(args.output_dir.glob("neural-*.png")) + sorted(
        args.output_dir.glob("neural-*.pdf")
    )
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite manifest {manifest_path}")
    manifest = {
        "source_metrics_csv": str(args.metrics_csv),
        "source_metrics_sha256": _sha256(args.metrics_csv),
        "source_history_csv": str(args.history_csv),
        "source_history_sha256": _sha256(args.history_csv),
        "source_manifest": str(args.source_manifest),
        "source_manifest_sha256": _sha256(args.source_manifest),
        "snapshot_sha256": source_manifest.get("snapshot_sha256"),
        "source_commits": source_manifest.get("source_commits", []),
        "summary_csv": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "figure_sha256": {path.name: _sha256(path) for path in artifacts},
        "periods": ["OOF February-April", "May development validation"],
        "metrics_included": ["MCC", "accuracy", "train loss", "validation loss"],
        "seeds": [42, 43],
        "June_accessed": False,
        "uncertainty": "No confidence intervals in neural comparison figures; metrics are descriptive.",
        "warning": "OOF February-April neural figure is not on the March-April-only Stage 12 stacking population.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Created 6 figure files in {args.output_dir}; summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
