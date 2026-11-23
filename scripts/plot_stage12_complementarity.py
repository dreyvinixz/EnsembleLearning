"""Plot paired probability/error diversity and two-model gains versus Logistic Regression."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

SCALES = ["5m", "10m", "15m"]
SCALE_COLORS = {"5m": "#0072B2", "10m": "#009E73", "15m": "#7B61A8"}
PERIODS = [("oof_mar_apr", "OOF março–abril"),
           ("validation_may", "Validação maio")]
METRICS = [
    ("probability_pearson", "Correlação p(alta)"),
    ("prediction_agreement", "Acordo de classe"),
    ("error_jaccard", "Jaccard de erros"),
    ("delta_mcc", "Δ MCC vs LR"),
    ("delta_accuracy", "Δ acurácia vs LR"),
]


def prepare(summary: pd.DataFrame) -> pd.DataFrame:
    required = {"period", "scale", "model_left", "model_right", "probability_pearson",
                "prediction_agreement", "error_jaccard", "mean_probability_mcc", "left_mcc",
                "mean_probability_accuracy", "left_accuracy", "n"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Complementarity CSV lacks columns: {sorted(missing)}")
    paired = summary.loc[
        summary.model_left.eq("LogisticRegression")
        & summary.period.isin([period for period, _ in PERIODS])
        & ~summary.model_right.astype(str).str.startswith("SeedEnsemble_")
    ].copy()
    paired["delta_mcc"] = paired.mean_probability_mcc - paired.left_mcc
    paired["delta_accuracy"] = paired.mean_probability_accuracy - paired.left_accuracy
    paired["model_group"] = paired.model_right.astype(str).str.replace(
        r"_seed\d+$", "", regex=True
    ).str.replace("_", " ")
    metric_columns = [name for name, _ in METRICS]
    paired = paired.groupby(["period", "scale", "model_group"], observed=True)[
        metric_columns
    ].mean().reset_index().rename(columns={"model_group": "model_right"})
    for period, _ in PERIODS:
        part = paired.loc[paired.period.eq(period)]
        if set(part.scale.astype(str)) != set(SCALES):
            raise ValueError(f"Incomplete scale coverage for {period}")
    if paired.empty:
        raise ValueError("No paired Logistic Regression rows to plot")
    return paired


def render(summary: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    if output.with_suffix(".png").exists() or output.with_suffix(".pdf").exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    paired = prepare(summary)
    model_order = ["Voting", "BlockBagging", "Stacking"] + sorted(
        model for model in paired.model_right.astype(str).unique()
        if model not in {"Voting", "BlockBagging", "Stacking"}
    )
    models = model_order
    y_positions = {model: index for index, model in enumerate(models)}
    offsets = {"5m": -.18, "10m": 0.0, "15m": .18}
    fig, axes = plt.subplots(2, len(METRICS), figsize=(17, 8.5), sharey="row")
    for row, (period, period_label) in enumerate(PERIODS):
        data = paired.loc[paired.period.eq(period)]
        for col, (metric, title) in enumerate(METRICS):
            ax = axes[row, col]
            for scale in SCALES:
                part = data.loc[data.scale.eq(scale)]
                ys = [y_positions[name] + offsets[scale] for name in part.model_right]
                ax.scatter(part[metric], ys, color=SCALE_COLORS[scale], s=25,
                           alpha=.88, edgecolors="white", linewidths=.4, zorder=3)
            if metric.startswith("delta_"):
                ax.axvline(0, color="#606B78", linestyle="--", linewidth=1)
            ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
            ax.grid(axis="x", color="#D9E0E8", linewidth=.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlabel("pontos" if metric.startswith("delta_") else "proporção")
            if col == 0:
                ax.set_yticks(range(len(models)), models)
                ax.tick_params(axis="y", labelsize=8)
            else:
                ax.tick_params(axis="y", labelleft=False)
            ax.invert_yaxis()
        fig.text(.005, .725 if row == 0 else .295, period_label, rotation=90,
                 ha="center", va="center", fontsize=10, color="#374151",
                 fontweight="bold")
    legend = [Line2D([0], [0], marker="o", linestyle="", color=color,
                     label=scale, markersize=7) for scale, color in SCALE_COLORS.items()]
    fig.legend(handles=legend, title="Escala B0", loc="upper center",
               bbox_to_anchor=(.56, .975), ncol=3, frameon=False)
    fig.suptitle("Diversidade e média de probabilidades — comparações pareadas com LR",
                 x=.08, y=1.015, ha="left", fontsize=15,
                 fontweight="bold", color="#172B4D")
    fig.text(.08, .015,
             "Jaccard = interseção/união de erros. Δ mostra MCC/acurácia da média das duas "
             "probabilidades menos LR. Descritivo, sem IC ou correção de multiplicidade; "
             "populações exatas por período/escala.", fontsize=8.5, color="#526274")
    fig.tight_layout(rect=(.035, .08, 1, .90), h_pad=1.6, w_pad=1.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    png, pdf = output.with_suffix(".png"), output.with_suffix(".pdf")
    fig.savefig(png, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.output.with_name(f"{args.output.stem}-manifest.json")
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {manifest_path}")
    png, pdf = render(pd.read_csv(args.summary_csv), args.output)
    manifest = {
        "summary_csv": str(args.summary_csv),
        "summary_sha256": sha256(args.summary_csv),
        "png_sha256": sha256(png), "pdf_sha256": sha256(pdf),
        "periods": ["OOF March-April", "May development validation"],
        "comparison": "Exact paired Logistic Regression predictions by scale and period",
        "June_accessed": False,
        "inference": "Descriptive only; no confidence intervals or multiplicity correction.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Created {png} and {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
