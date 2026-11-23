"""Render paired-population Stage 12 MCC and accuracy figures."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

METHODS = ["Logistic Regression", "Voting", "Block Bagging", "Stacking"]
METHOD_LABELS = {
    "Logistic Regression": "Regressão Logística",
    "Voting": "Voting",
    "Block Bagging": "Block Bagging",
    "Stacking": "Stacking",
}
COLORS = {
    "Logistic Regression": "#374151",
    "Voting": "#0072B2",
    "Block Bagging": "#E69F00",
    "Stacking": "#8B3FB0",
}
PERIODS = ["OOF March-April", "May validation"]
PERIOD_LABELS = {
    "OOF March-April": "OOF março–abril",
    "May validation": "Validação de maio",
}
SCALES = ["5m", "10m", "15m"]


def plot_metric(frame: pd.DataFrame, metric: str, output: Path) -> tuple[Path, Path]:
    low_col, high_col = f"{metric}_ci95_low", f"{metric}_ci95_high"
    if output.with_suffix(".png").exists() or output.with_suffix(".pdf").exists():
        raise FileExistsError(f"Refusing to overwrite figure {output}")
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.5), sharex=True)
    offsets = [-0.24, -0.08, 0.08, 0.24]
    for ax, period in zip(axes, PERIODS, strict=True):
        subset = frame.loc[frame["period"].eq(period)]
        sessions = int(subset["n_sessions"].iloc[0])
        sample_sizes = [int(subset.loc[subset.scale.eq(scale), "n"].iloc[0])
                        for scale in SCALES]
        ax.set_title(
            f"{PERIOD_LABELS[period]} · {sessions} sessões · n por escala: "
            + " / ".join(map(str, sample_sizes)),
            loc="left", fontsize=10, color="#374151", pad=8,
        )
        for method_index, method in enumerate(METHODS):
            values = subset.loc[subset["method"].eq(method)].set_index("scale")
            point = values.loc[SCALES, metric].to_numpy()
            lower = point - values.loc[SCALES, low_col].to_numpy()
            upper = values.loc[SCALES, high_col].to_numpy() - point
            ax.errorbar(
                [i + offsets[method_index] for i in range(len(SCALES))], point,
                yerr=[lower, upper], fmt="o", markersize=6,
                color=COLORS[method], ecolor=COLORS[method], elinewidth=1.25,
                capsize=3, label=METHOD_LABELS[method], zorder=3,
            )
        ax.grid(axis="y", color="#D9E0E8", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#AAB4C0")
        ax.set_ylabel("MCC" if metric == "mcc" else "Acurácia")
        ax.set_ylim((0.40, 0.56) if metric == "mcc" else (0.70, 0.78))
    axes[-1].set_xticks(range(len(SCALES)), SCALES)
    axes[-1].set_xlabel("Rótulo B0")
    fig.suptitle(
        f"Desempenho por escala — {('MCC' if metric == 'mcc' else 'Acurácia')}",
        fontsize=16, fontweight="bold", color="#172B4D", x=0.08, ha="left",
    )
    fig.text(
        0.08, 0.925,
        "Mesmas amostras por método · IC 95% por bootstrap de sessões (5.000 reamostragens)",
        fontsize=9.5, color="#526274",
    )
    fig.text(
        0.08, 0.02,
        "OOF restrito a março–abril para incluir Stacking; maio é validação. "
        "Classificação dos rótulos B0, não previsão de retorno ou PnL.",
        fontsize=8.5, color="#526274",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.895),
               ncol=4, frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.12, hspace=0.34)
    output.parent.mkdir(parents=True, exist_ok=True)
    png_path, pdf_path = output.with_suffix(".png"), output.with_suffix(".pdf")
    fig.savefig(png_path, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.summary_csv)
    required = {"period", "scale", "method", "n", "n_sessions"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing required summary columns: {sorted(required - set(frame.columns))}")
    for metric in ("mcc", "accuracy"):
        needed = {metric, f"{metric}_ci95_low", f"{metric}_ci95_high"}
        if not needed.issubset(frame.columns):
            raise ValueError(f"Missing {metric} or bootstrap interval columns")
        for period in PERIODS:
            for scale in SCALES:
                if frame.loc[frame.period.eq(period) & frame.scale.eq(scale)].method.nunique() != len(METHODS):
                    raise ValueError(f"Incomplete method set for {period}/{scale}")
        png, pdf = plot_metric(frame, metric, args.output_dir / f"stage12-{metric}-2026-11-23")
        print(f"{png}\n{pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
