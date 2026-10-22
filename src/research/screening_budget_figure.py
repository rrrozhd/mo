"""Render the retrospective CYP3A4 screening comparison from its saved summary."""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SURFACE = "#fcfcfa"
INK = "#182b31"
SECONDARY = "#59676b"
GRID = "#dfe5e3"
ARMS = {
    "classifier": ("Trained classifier", "#087f7e"),
    "smiles": ("Input-only LLM", "#cc692f"),
}


def render(summary: pd.DataFrame, output: Path, population: int = 2425) -> None:
    """Write PNG, PDF, and SVG figures; all result labels come from ``summary``.

    ``summary`` contains one row per arm, policy, and assay budget. Policies are
    ``score_only`` and ``diversity_cap``; the latter uses a 0.8 similarity cap.
    """
    output.mkdir(parents=True, exist_ok=True)
    rows = summary.set_index(["arm", "policy", "budget"]).sort_index()
    budgets = sorted(summary.budget.unique())
    largest = int(max(budgets))

    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": SECONDARY,
        "xtick.color": SECONDARY,
        "ytick.color": SECONDARY,
        "svg.hashsalt": "cyp3a4-screening-budget",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }):
        figure, (hits_ax, coverage_ax) = plt.subplots(
            1, 2, figsize=(13.4, 7.2), facecolor=SURFACE,
            gridspec_kw={"width_ratios": [1, 1.08]},
        )
        figure.subplots_adjust(left=0.07, right=0.97, bottom=0.28, top=0.72, wspace=0.3)
        figure.text(0.07, 0.95, "CYP3A4 screening: hits and chemical coverage",
                    fontsize=21, weight="bold", va="top")
        figure.text(0.07, 0.892, "Frozen predictions turned into ranked, budget-limited assay batches",
                    fontsize=12, color=SECONDARY)
        figure.legend(
            handles=[Line2D([], [], color=color, marker="o", linewidth=2.5,
                            markersize=7, label=label) for label, color in ARMS.values()],
            loc="upper left", bbox_to_anchor=(0.062, 0.861), ncol=2,
            frameon=False, handlelength=2, columnspacing=2.5,
        )

        for axes in (hits_ax, coverage_ax):
            axes.set_facecolor(SURFACE)
            axes.spines[["top", "right"]].set_visible(False)
            for side in ("left", "bottom"):
                axes.spines[side].set_color(GRID)
            axes.tick_params(length=0, pad=8)
            axes.grid(axis="y", color=GRID, linewidth=0.8)
            axes.set_axisbelow(True)

        hits_ax.set_title("A   Observed hits by assay budget", loc="left",
                          fontsize=13, weight="bold", pad=27)
        hits_ax.text(0, 1.035, "Both predictors use the same 0.8 similarity cap",
                     transform=hits_ax.transAxes, fontsize=10, color=SECONDARY)
        for arm, (_, color) in ARMS.items():
            selected = rows.loc[(arm, "diversity_cap")].sort_index()
            hits_ax.plot(selected.index, selected.hits, color=color, linewidth=2.6,
                         marker="o", markersize=8, markeredgecolor=SURFACE,
                         markeredgewidth=1.5, zorder=3)
            for budget, row in selected.iterrows():
                hits_ax.annotate(
                    f"{int(row.hits)}/{int(row.selected_count)}",
                    (budget, row.hits), xytext=(0, 13 if arm == "classifier" else -21),
                    textcoords="offset points", ha="center", fontsize=11,
                    weight="bold", color=color,
                )
        hits_ax.set(xlim=(min(budgets) - 10, largest + 12), ylim=(0, largest * 1.16),
                    xticks=budgets, xlabel="Assay budget (compounds)", ylabel="Observed assay hits")
        hits_ax.set_yticks(range(0, largest + 1, 24))
        hits_ax.text(0, -0.23, "Point labels: hits / selected compounds",
                     transform=hits_ax.transAxes, fontsize=10, color=SECONDARY)

        coverage_ax.set_title(f"B   Diversity trade-off at budget {largest}", loc="left",
                              fontsize=13, weight="bold", pad=27)
        coverage_ax.text(0, 1.035, "Hollow: score only    →    Filled: similarity cap 0.8",
                         transform=coverage_ax.transAxes, fontsize=10, color=SECONDARY)
        maximum_budget = summary.loc[summary.budget == largest]
        for arm, (_, color) in ARMS.items():
            before = rows.loc[(arm, "score_only", largest)]
            after = rows.loc[(arm, "diversity_cap", largest)]
            start = (before.unique_scaffolds, before.hits)
            end = (after.unique_scaffolds, after.hits)
            coverage_ax.annotate("", xy=end, xytext=start,
                                 arrowprops={"arrowstyle": "->", "color": color,
                                             "lw": 2, "shrinkA": 8, "shrinkB": 8}, zorder=2)
            for row, filled, label_offset in ((before, False, 19), (after, True, -37)):
                coverage_ax.scatter(row.unique_scaffolds, row.hits, s=105,
                                    facecolors=color if filled else SURFACE,
                                    edgecolors=color, linewidths=2, zorder=3)
                coverage_ax.annotate(
                    f"{int(row.unique_scaffolds)} scaffolds\n{int(row.hits)} hits",
                    (row.unique_scaffolds, row.hits), xytext=(0, label_offset),
                    textcoords="offset points", ha="center", fontsize=10.5,
                    color=color, weight="bold", linespacing=1.35,
                )
        coverage_ax.set(
            xlim=(maximum_budget.unique_scaffolds.min() - 12,
                  maximum_budget.unique_scaffolds.max() + 12),
            ylim=(maximum_budget.hits.min() - 10, maximum_budget.hits.max() + 11),
            xlabel="Distinct Bemis–Murcko scaffolds", ylabel="Observed assay hits",
        )
        coverage_ax.text(0, -0.23, "Arrows show the effect of adding the diversity constraint",
                         transform=coverage_ax.transAxes, fontsize=10, color=SECONDARY)

        figure.text(0.07, 0.15,
                    f"Retrospective exploratory holdout · {population:,} labeled compounds · "
                    "fixed compound-ID tie-break",
                    fontsize=10.5, weight="bold")
        figure.text(0.07, 0.112,
                    "Selection uses frozen scores and structures; assay labels are joined after selection.",
                    fontsize=10, color=SECONDARY)
        figure.text(0.07, 0.074,
                    "Descriptive counts without statistical intervals; no prospective validation "
                    "or claim of expected experimental yield.", fontsize=10, color=SECONDARY)
        for extension in ("png", "pdf", "svg"):
            metadata = {"Date": None} if extension == "svg" else None
            if extension == "pdf":
                metadata = {"CreationDate": None, "ModDate": None}
            figure.savefig(output / f"screening-budget.{extension}", dpi=240,
                           facecolor=SURFACE, metadata=metadata)
        plt.close(figure)
