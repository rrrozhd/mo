"""Render the two audit figures for the write-up and the video.

Dot plots rather than bars: both charts encode position on a truncated scale,
which is honest for a point mark and misleading for a bar.
"""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
OTHER, CYP = "#2a78d6", "#ec4d0e"          # categorical slots 1 and 2
BANDS = {"<0.4": "#86b6ef", "0.4-0.6": "#2a78d6", ">=0.6": "#104281"}  # ordinal ramp
LABELS = {"<0.4": "novel (<0.4)", "0.4-0.6": "moderate (0.4-0.6)",
          ">=0.6": "near-duplicate (>=0.6)"}


def _frame(axes):
    axes.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
        axes.spines[side].set_linewidth(0.8)
    axes.tick_params(colors=MUTED, labelsize=8, length=3)


def spectrum(path, output):
    """47 TDC benchmarks ranked by how novel their test molecules really are."""
    rows = pd.read_csv(path)
    rows = rows.loc[rows.status == "ok"].sort_values("median_novelty").reset_index(drop=True)
    is_cyp = rows.dataset.str.startswith("cyp") & rows.dataset.str.endswith("veith")
    colors = [CYP if flag else OTHER for flag in is_cyp]

    figure, axes = plt.subplots(figsize=(7.6, 9.2), facecolor=SURFACE)
    axes.hlines(rows.index, 0, rows.median_novelty, color=GRID, linewidth=1.2, zorder=1)
    axes.scatter(rows.median_novelty, rows.index, s=46, c=colors, zorder=3,
                 edgecolors=SURFACE, linewidths=1.2)
    axes.set_yticks(rows.index)
    axes.set_yticklabels([f"{d[:40]}  (n={int(n)})" for d, n in
                          zip(rows.dataset, rows.n_test)], fontsize=7.5, color=INK_SECONDARY)
    axes.set_xlim(0, 0.78)
    axes.set_ylim(-1, len(rows))
    axes.set_xlabel("median test-set novelty  (max Tanimoto to training)",
                    fontsize=9, color=INK_SECONDARY)
    axes.xaxis.grid(True, color=GRID, linewidth=0.7)
    axes.set_axisbelow(True)
    _frame(axes)

    for label, color in (("TDC benchmark", OTHER), ("CYP set used here", CYP)):
        axes.scatter([], [], s=46, c=color, label=label)
    axes.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)

    figure.tight_layout(rect=(0, 0, 1, 0.962))
    figure.text(0.012, 0.988, "One split protocol, four-fold difference in difficulty",
                fontsize=12.5, color=INK, ha="left", va="top", weight="bold")
    figure.text(0.012, 0.969, "TDC scaffold split, seed 42, Morgan r=2 2048 bits · "
                f"{len(rows)} datasets · herg and herg_karim are the same endpoint",
                fontsize=8.5, color=MUTED, ha="left", va="top")
    figure.savefig(output, dpi=220, facecolor=SURFACE)
    return output


def decomposition(path, output):
    """What one reported AUC averages over, per enzyme."""
    rows = pd.read_csv(path)
    rows = rows.loc[rows.metric == "auc"]
    if "arm" in rows and rows.arm.nunique() > 1:
        raise ValueError("Choose one prediction arm before plotting its novelty bands")
    enzymes = [e for e in rows.enzyme.unique()]
    offsets = {"<0.4": 0.26, "0.4-0.6": 0.0, ">=0.6": -0.26}

    figure, axes = plt.subplots(figsize=(7.6, 4.8), facecolor=SURFACE)
    for position, enzyme in enumerate(enzymes):
        block = rows.loc[rows.enzyme == enzyme]
        reported = block.loc[block.band == "all"].iloc[0]
        axes.plot([reported.value, reported.value],
                  [position - 0.38, position + 0.38],
                  color=MUTED, linewidth=1.4, zorder=2)
        for band, offset in offsets.items():
            row = block.loc[block.band == band]
            if row.empty:
                continue
            row = row.iloc[0]
            y = position + offset
            axes.hlines(y, row.ci_low, row.ci_high, color=BANDS[band],
                        linewidth=2, zorder=3)
            axes.scatter(row.value, y, s=52, c=BANDS[band], zorder=4,
                         edgecolors=SURFACE, linewidths=1.2)
        # the low band labels go left, clear of the reported-value rule
        for band, side in (("<0.4", "right"), (">=0.6", "left")):
            row = block.loc[block.band == band]
            if row.empty:
                continue
            row = row.iloc[0]
            x = row.ci_low - 0.004 if side == "right" else row.ci_high + 0.004
            axes.text(x, position + offsets[band], f"{row.value:.3f}", ha=side,
                      va="center", fontsize=7.5, color=INK_SECONDARY)

    axes.set_yticks(range(len(enzymes)))
    axes.set_yticklabels(enzymes, fontsize=9, color=INK)
    axes.set_ylim(-0.7, len(enzymes) - 0.3)
    axes.set_xlim(0.70, 1.0)
    axes.invert_yaxis()
    axes.set_xlabel("test AUC  (point and 95% scaffold-bootstrap interval)",
                    fontsize=9, color=INK_SECONDARY)
    axes.xaxis.grid(True, color=GRID, linewidth=0.7)
    axes.set_axisbelow(True)
    _frame(axes)

    for band, color in BANDS.items():
        axes.scatter([], [], s=52, c=color, label=LABELS[band])
    axes.plot([], [], color=MUTED, linewidth=1.4, label="reported (whole test set)")
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False,
                fontsize=8, ncol=4, columnspacing=1.4, labelcolor=INK_SECONDARY)

    axes.set_title("A reported AUC is an average over difficulty it never states",
                   fontsize=12, color=INK, loc="left", pad=26, weight="bold")
    axes.text(0, 1.035, "Same frozen models, same test set, split by each molecule's "
              "similarity to training", transform=axes.transAxes,
              fontsize=8.5, color=MUTED)
    figure.tight_layout()
    figure.savefig(output, dpi=220, facecolor=SURFACE)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("audit"))
    args = parser.parse_args()
    print(spectrum(args.audit / "novelty_spectrum.csv",
                   args.audit / "fig_spectrum.png"))
    print(decomposition(args.audit / "novelty_decomposition.csv",
                        args.audit / "fig_decomposition.png"))


if __name__ == "__main__":
    main()
