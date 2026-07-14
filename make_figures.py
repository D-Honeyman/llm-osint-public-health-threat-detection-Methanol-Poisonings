"""
Figure generation for the toxic alcohol and methanol poisoning LLM evaluation.

Produces Figures 3 to 7 of the manuscript:

    Figure 3  Inter-annotator agreement (Cohen's kappa with 95% CIs)
    Figure 4  Precision, recall and F1 with Clopper-Pearson 95% CIs
    Figure 5  Confusion matrix heatmap (TP, FP, FN, TN)
    Figure 6  Entity-level F1-scores, ranked
    Figure 7  Distribution of false positives and false negatives

By default the script reads the outputs of the two analysis scripts:

    python evaluator_methanol.py     ->  evaluation_results.csv
    python iaa_methanol.py           ->  inter_annotator_agreement.csv
    python make_figures.py           ->  figures/

If those files are absent, the script falls back to the values reported in the
manuscript, so it can be run standalone.

All figures share a single colour palette (see PALETTE below) so that a given
colour carries the same meaning across every figure.

Author: Damian Honeyman et al.
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


# --------------------------------------------------------------------------
# Shared palette. A colour means the same thing in every figure.
# --------------------------------------------------------------------------
PALETTE = {
    "precision": "#1F3864",   # dark blue
    "recall": "#2E7D6B",      # teal
    "f1": "#C77D26",          # amber
    "true_positive": "#1F3864",
    "false_positive": "#C77D26",
    "false_negative": "#A03E3E",   # red
    "true_negative": "#9AA5B1",    # grey
    "kappa": "#1F3864",
    "grid": "#DDDDDD",
    "note": "#555555",
    "reference_line": "#BBBBBB",
}

RC = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#1a1a1a",
    "text.color": "#1a1a1a",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "figure.facecolor": "white",
}

# Display order and line-wrapped labels used on every figure.
ENTITY_ORDER = [
    "Event identification",
    "Fatality count",
    "Location",
    "Date",
    "Case number",
    "Temporal expressions",
]

ENTITY_LABELS = {
    "Event identification": "Event\nidentification",
    "Fatality count": "Fatality\ncount",
    "Location": "Location",
    "Date": "Date",
    "Case number": "Case\nnumber",
    "Temporal expressions": "Temporal\nexpressions",
}

# Maps the raw entity keys emitted by evaluator_methanol.py to reporting names.
EVALUATOR_KEYS = {
    "Event identification": "Event identification",
    "FATALITY COUNT": "Fatality count",
    "COUNTRY": "Location",
    "date": "Date",
    "CASE NUMBER": "Case number",
    "Dates": "Temporal expressions",
}


# --------------------------------------------------------------------------
# Fallback values (as reported in the manuscript)
# --------------------------------------------------------------------------
FALLBACK_METRICS = pd.DataFrame([
    # entity, precision, p_lo, p_hi, recall, r_lo, r_hi, f1, TP, FP, FN, TN
    ("Event identification",                 0.98, 0.93, 1.00, 0.98, 0.93, 1.00, 0.98, 98, 2, 2,  0),
    ("Fatality count",                       0.87, 0.69, 0.96, 0.96, 0.81, 1.00, 0.91, 26, 4, 1, 17),
    ("Location",                             0.94, 0.86, 0.98, 0.98, 0.92, 1.00, 0.96, 64, 4, 1,  3),
    ("Date",                                 0.80, 0.63, 0.92, 0.93, 0.78, 0.99, 0.86, 28, 7, 2,  7),
    ("Case number",                          0.91, 0.59, 1.00, 0.91, 0.59, 1.00, 0.91, 10, 1, 1, 25),
    ("Temporal expressions", 1.00, 0.91, 1.00, 1.00, 0.91, 1.00, 1.00, 37, 0, 0,  6),
], columns=["entity", "precision", "p_lo", "p_hi", "recall", "r_lo", "r_hi",
            "f1", "TP", "FP", "FN", "TN"])

FALLBACK_IAA = pd.DataFrame([
    # entity, n, per cent agreement, kappa, k_lo, k_hi
    ("Event identification",                 100, 98.00, 0.95,  0.88, 1.00),
    ("Fatality count",                        28, 78.57, 0.54,  0.22, 0.87),
    ("Location",                              28, 85.71, 0.29, -0.35, 0.93),
    ("Date",                                  28, 75.00, 0.42,  0.04, 0.79),
    ("Case number",                           28, 78.57, 0.13, -0.48, 0.75),
    ("Temporal expressions",  28, 96.43, 0.92,  0.77, 1.00),
], columns=["entity", "n", "agreement", "kappa", "k_lo", "k_hi"])


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_metrics(path):
    """Load evaluation_results.csv, or fall back to manuscript values."""
    if not os.path.exists(path):
        print("  %s not found, using fallback values" % path)
        return FALLBACK_METRICS.copy()

    raw = pd.read_csv(path)
    raw = raw[raw["ENTITY_TYPE"].isin(EVALUATOR_KEYS)].copy()
    raw["entity"] = raw["ENTITY_TYPE"].map(EVALUATOR_KEYS)

    out = pd.DataFrame({
        "entity": raw["entity"],
        "precision": raw["precision"],
        "p_lo": raw["precision_ci_low"],
        "p_hi": raw["precision_ci_high"],
        "recall": raw["recall"],
        "r_lo": raw["recall_ci_low"],
        "r_hi": raw["recall_ci_high"],
        "f1": raw["fmeasure"],
        "TP": raw["TP"],
        "FP": raw["FP"],
        "FN": raw["FN"],
        "TN": raw["TN"],
    })

    print("  loaded %s" % path)
    return out


def parse_ci(text):
    """Parse a '0.88 to 1.00' or '-0.48 to 0.75' confidence interval string."""
    found = re.findall(r"-?\d+\.?\d*", str(text))
    if len(found) < 2:
        return np.nan, np.nan
    return float(found[0]), float(found[1])


def load_iaa(path):
    """Load inter_annotator_agreement.csv, or fall back to manuscript values."""
    if not os.path.exists(path):
        print("  %s not found, using fallback values" % path)
        return FALLBACK_IAA.copy()

    raw = pd.read_csv(path)

    kappa_col = [c for c in raw.columns if "kappa" in c.lower()][0]
    agree_col = [c for c in raw.columns if "agreement" in c.lower()][0]
    ci_col = [c for c in raw.columns if "CI" in c][0]

    lows, highs = zip(*[parse_ci(v) for v in raw[ci_col]])

    out = pd.DataFrame({
        "entity": raw["Entity"],
        "n": raw["n"],
        "agreement": raw[agree_col],
        "kappa": raw[kappa_col],
        "k_lo": lows,
        "k_hi": highs,
    })

    print("  loaded %s" % path)
    return out


def order_rows(df):
    """Sort into the canonical entity order used throughout the manuscript."""
    df = df.copy()
    df["_order"] = df["entity"].apply(
        lambda e: ENTITY_ORDER.index(e) if e in ENTITY_ORDER else 99
    )
    return df.sort_values("_order").drop(columns="_order").reset_index(drop=True)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def style_axes(ax):
    ax.yaxis.grid(True, color=PALETTE["grid"], lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def add_note(fig, text, y=-0.09, size=7.8):
    fig.text(0.008, y, text, fontsize=size, color=PALETTE["note"],
             ha="left", va="top")


def save(fig, outdir, name):
    for ext in ("png", "pdf"):
        path = os.path.join(outdir, "%s.%s" % (name, ext))
        fig.savefig(path)
    plt.close(fig)
    print("  wrote %s.png and %s.pdf" % (name, name))


# --------------------------------------------------------------------------
# Figure 3: inter-annotator agreement
# --------------------------------------------------------------------------
def figure_3(iaa, outdir):

    labels = [ENTITY_LABELS[e] for e in iaa["entity"]]
    x = np.arange(len(iaa))

    kappa = iaa["kappa"].values
    err = [kappa - iaa["k_lo"].values, iaa["k_hi"].values - kappa]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))

    bars = ax.bar(x, kappa, width=0.58, color=PALETTE["kappa"],
                  edgecolor="white", lw=0.6, zorder=3)

    ax.errorbar(x, kappa, yerr=err, fmt="none", ecolor="black",
                elinewidth=1.1, capsize=4, capthick=1.1, zorder=4)

    for bar, k in zip(bars, kappa):
        ax.text(bar.get_x() + bar.get_width() / 2, max(k, 0) + 0.055,
                "%.2f" % k, ha="center", va="bottom",
                fontsize=9.5, fontweight="bold")

    # Landis and Koch interpretation thresholds
    for threshold, label in [(0.81, "Almost perfect"), (0.61, "Substantial"),
                             (0.41, "Moderate"), (0.21, "Fair")]:
        ax.axhline(threshold, color=PALETTE["reference_line"],
                   ls=(0, (4, 3)), lw=0.7, zorder=1)
        ax.text(len(iaa) - 0.42, threshold + 0.015, label, fontsize=7.5,
                color="#888888", ha="right", va="bottom")

    ax.axhline(0, color="#333333", lw=0.9, zorder=2)

    ax.set_ylim(-0.60, 1.18)
    ax.set_xlim(-0.6, len(iaa) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["%s\n(n=%d)" % (lab, n) for lab, n in zip(labels, iaa["n"])],
        fontsize=9
    )
    ax.set_ylabel("Cohen's \u03ba", fontsize=11)
    ax.set_title("Inter-annotator agreement by entity type",
                 fontsize=12.5, fontweight="bold", pad=14, loc="left")

    style_axes(ax)

    add_note(fig,
        "Note: Black error bars represent 95% confidence intervals for Cohen's \u03ba. Event identification was assessed across all 100 annotated\n"
        "articles; all other entities across the 28 event-positive articles. Intervals for location and case number span zero, indicating substantial\n"
        "uncertainty. \u03ba is unstable at high prevalence and should be interpreted alongside per cent agreement.",
        y=-0.10, size=7.6)

    save(fig, outdir, "Figure3_inter_annotator_agreement")


# --------------------------------------------------------------------------
# Figure 4: precision, recall, F1 with Clopper-Pearson CIs
# --------------------------------------------------------------------------
def figure_4(metrics, outdir):

    labels = [ENTITY_LABELS[e] for e in metrics["entity"]]
    x = np.arange(len(metrics))
    width = 0.26

    precision = metrics["precision"].values
    recall = metrics["recall"].values
    f1 = metrics["f1"].values

    p_err = [precision - metrics["p_lo"].values, metrics["p_hi"].values - precision]
    r_err = [recall - metrics["r_lo"].values, metrics["r_hi"].values - recall]

    fig, ax = plt.subplots(figsize=(9.4, 5.0))

    b1 = ax.bar(x - width, precision, width, color=PALETTE["precision"],
                label="Precision", edgecolor="white", lw=0.6, zorder=3)
    b2 = ax.bar(x, recall, width, color=PALETTE["recall"],
                label="Recall", edgecolor="white", lw=0.6, zorder=3)
    b3 = ax.bar(x + width, f1, width, color=PALETTE["f1"],
                label="F\u2081-score", edgecolor="white", lw=0.6, zorder=3)

    ax.errorbar(x - width, precision, yerr=p_err, fmt="none", ecolor="black",
                elinewidth=1.1, capsize=3.5, capthick=1.1, zorder=4)
    ax.errorbar(x, recall, yerr=r_err, fmt="none", ecolor="black",
                elinewidth=1.1, capsize=3.5, capthick=1.1, zorder=4)

    for bars, values in [(b1, precision), (b2, recall), (b3, f1)]:
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, 0.022, "%.2f" % value,
                    ha="center", va="bottom", fontsize=7.6, color="white",
                    fontweight="bold", rotation=90, zorder=5)

    ax.set_ylim(0, 1.16)
    ax.set_xlim(-0.62, len(metrics) - 0.38)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score", fontsize=11)

    # Title padded so the legend sits between the title and the plot.
    ax.set_title("GPT-4o entity recognition performance",
                 fontsize=12.5, fontweight="bold", pad=34, loc="left")
    ax.legend(frameon=False, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.005), fontsize=9.5,
              handlelength=1.5, handleheight=0.9, columnspacing=2.2)

    style_axes(ax)

    add_note(fig,
        "Note: Black error bars represent 95% confidence intervals (CIs) for precision and recall estimates, calculated using the Clopper-Pearson\n"
        "exact binomial method. Wide intervals for case number reflect the small number of positive examples (10 true positives).")

    save(fig, outdir, "Figure4_precision_recall_f1")


# --------------------------------------------------------------------------
# Figure 5: confusion matrix heatmap
# --------------------------------------------------------------------------
def figure_5(metrics, outdir):

    matrix = metrics[["TP", "FP", "FN", "TN"]].values
    labels = [e for e in metrics["entity"]]

    fig, ax = plt.subplots(figsize=(7.4, 5.0))

    cmap = LinearSegmentedColormap.from_list(
        "navy", ["#FFFFFF", "#9FB4D4", PALETTE["true_positive"]]
    )

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=matrix.max())

    ax.set_xticks(range(4))
    ax.set_xticklabels(["True\npositive", "False\npositive",
                        "False\nnegative", "True\nnegative"], fontsize=9.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9.5)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, str(int(value)), ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if value > matrix.max() * 0.55 else "#1a1a1a")

    ax.set_title("Entity-level prediction outcomes",
                 fontsize=12.5, fontweight="bold", pad=14, loc="left")

    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.03)
    cbar.set_label("Count", fontsize=9.5)
    cbar.outline.set_visible(False)

    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)

    add_note(fig,
        "Note: Counts are reported at the document level within a pre-filtered, domain-relevant OSINT corpus. A true negative is recorded only where\n"
        "neither the model nor the annotator recorded the entity. Event identification has no true negatives by construction.",
        y=-0.06)

    save(fig, outdir, "Figure5_confusion_matrix")


# --------------------------------------------------------------------------
# Figure 6: entity-level F1, ranked
# --------------------------------------------------------------------------
def figure_6(metrics, outdir):

    df = metrics.copy()
    df["support"] = df["TP"] + df["FN"]
    df = df.sort_values("f1").reset_index(drop=True)

    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    bars = ax.barh(y, df["f1"], height=0.6, color=PALETTE["f1"],
                   edgecolor="white", lw=0.6, zorder=3)

    for bar, value, support in zip(bars, df["f1"], df["support"]):
        ax.text(value + 0.012, bar.get_y() + bar.get_height() / 2,
                "%.2f" % value, va="center", fontsize=10, fontweight="bold")
        ax.text(0.012, bar.get_y() + bar.get_height() / 2,
                "n=%d" % support, va="center", fontsize=8, color="white")

    ax.set_yticks(y)
    ax.set_yticklabels(df["entity"], fontsize=10)
    ax.set_xlim(0, 1.10)
    ax.set_xlabel("F\u2081-score", fontsize=11)

    ax.xaxis.grid(True, color=PALETTE["grid"], lw=0.6, zorder=0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    ax.set_title("Entity-level F\u2081-scores",
                 fontsize=12.5, fontweight="bold", pad=14, loc="left")

    add_note(fig,
        "Note: Performance is shown for comparative purposes. n denotes the number of gold-standard entity instances (true positives plus false\n"
        "negatives). Estimates for infrequent entities should be interpreted with caution due to limited sample support.",
        y=-0.08)

    save(fig, outdir, "Figure6_f1_scores")


# --------------------------------------------------------------------------
# Figure 7: false positives vs false negatives
# --------------------------------------------------------------------------
def figure_7(metrics, outdir):

    labels = [ENTITY_LABELS[e] for e in metrics["entity"]]
    x = np.arange(len(metrics))
    width = 0.36

    fp = metrics["FP"].values
    fn = metrics["FN"].values

    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    b1 = ax.bar(x - width / 2, fp, width, color=PALETTE["false_positive"],
                label="False positives", edgecolor="white", lw=0.6, zorder=3)
    b2 = ax.bar(x + width / 2, fn, width, color=PALETTE["false_negative"],
                label="False negatives", edgecolor="white", lw=0.6, zorder=3)

    for bars in (b1, b2):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.12,
                    "%d" % int(height), ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_ylim(0, max(list(fp) + list(fn)) + 1.4)

    ax.set_title("Distribution of false positives and false negatives",
                 fontsize=12.5, fontweight="bold", pad=14, loc="left")
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=9.5)

    style_axes(ax)

    add_note(fig,
        "Note: False positives exceed false negatives for every entity type, indicating a consistent tendency toward over-extraction. Over-extraction\n"
        "is most pronounced for exact dates, where publication timestamps and historical references were misidentified as event dates.")

    save(fig, outdir, "Figure7_false_positives_negatives")


# --------------------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser(
        description="Generate Figures 3 to 7 for the methanol LLM evaluation."
    )
    parser.add_argument("--metrics", default="evaluation_results.csv",
                        help="Output of evaluator_methanol.py")
    parser.add_argument("--iaa", default="inter_annotator_agreement.csv",
                        help="Output of iaa_methanol.py")
    parser.add_argument("--outdir", default="figures",
                        help="Directory to write figures into")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    plt.rcParams.update(RC)

    print("Loading data:")
    metrics = order_rows(load_metrics(args.metrics))
    iaa = order_rows(load_iaa(args.iaa))

    print("\nGenerating figures:")
    figure_3(iaa, args.outdir)
    figure_4(metrics, args.outdir)
    figure_5(metrics, args.outdir)
    figure_6(metrics, args.outdir)
    figure_7(metrics, args.outdir)

    print("\nDone. Figures written to: %s/" % args.outdir)


if __name__ == "__main__":
    main()
