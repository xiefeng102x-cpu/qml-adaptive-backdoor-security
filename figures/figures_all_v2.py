#!/usr/bin/env python3
"""
All non-concept figures (2–5, Supplementary 1–4) redesign – NC publication style.
Saves as *_v2.{svg,pdf,png} to avoid overwriting originals.
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
# Repo root = parent of this script's directory (figures/).
# Override with environment variables if your layout differs:
#   DATA_DIR   : directory containing the Zenodo metric CSVs (default: repo_root/data)
#   RESULTS_DIR: directory containing training results (default: repo_root/results)
#   OUT_DIR    : figure output directory (default: repo_root/figures_output)
import os as _os
_REPO = Path(__file__).resolve().parents[1]
DATA  = Path(_os.environ.get("DATA_DIR",     str(_REPO / "data")))
FULL  = Path(_os.environ.get("RESULTS_DIR",  str(_REPO / "results")))
OUT   = Path(_os.environ.get("OUT_DIR",      str(_REPO / "figures_output")))
LOUT  = OUT
OUT.mkdir(parents=True, exist_ok=True)

# ── semantic palette ───────────────────────────────────────────────────────────
BLUE      = "#2166AC"   # XYZ AUC / QML / main detection
TEAL      = "#35978F"   # TPR / CA / clean-side metrics
AMBER     = "#F4A11D"   # Joint evasion / key failure metric
CORAL     = "#D6604D"   # Classical Bloch / ASR / attacker-side
DKGREEN   = "#1B7837"   # Unconstrained control
PURPLE    = "#7B3F9E"   # Post-hoc / alternative condition
BLACK     = "#1A1A1A"
PALE_GRID = "#E2E2E2"
GREY      = "#888888"

# λ_r sequential: light→dark blue  (for scatter plots with 4 λ levels)
LAM_COLORS = {0.0: "#D1E5F0", 0.5: "#92C5DE", 1.0: "#4393C3", 2.0: BLUE}

# Measurement-set sequential: single/double/triple axis
MSET_COLORS = {"Z": "#D1E5F0", "X": "#D1E5F0", "Y": "#D1E5F0",
               "XZ": "#4393C3", "XY": "#4393C3", "YZ": "#4393C3",
               "XYZ": BLUE}
MSET_EDGE   = {"Z": "#4393C3", "X": "#4393C3", "Y": "#4393C3",
               "XZ": "#1a5c8a", "XY": "#1a5c8a", "YZ": "#1a5c8a",
               "XYZ": "#0d3a5c"}

WIDTH_IN = 183 / 25.4   # Nature double-column
SUPP_W   = 183 / 25.4


# ── global style ──────────────────────────────────────────────────────────────
def setup_style() -> None:
    mpl.rcParams.update({
        "font.family":        "Liberation Sans",
        "font.sans-serif":    ["Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size":          7,
        "axes.labelsize":     8,
        "axes.labelweight":   "bold",
        "axes.linewidth":     0.7,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.edgecolor":     BLACK,
        "xtick.labelsize":    7,
        "ytick.labelsize":    7,
        "xtick.major.width":  0.65,
        "ytick.major.width":  0.65,
        "xtick.major.size":   3,
        "ytick.major.size":   3,
        "legend.fontsize":    6.5,
        "legend.frameon":     False,
        "lines.linewidth":    1.6,
        "lines.markersize":   5,
        "patch.linewidth":    0.4,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "svg.fonttype":       "none",
        "savefig.facecolor":  "white",
        "savefig.edgecolor":  "white",
    })


def ygrid(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=PALE_GRID, linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", pad=2)


def panel_label(ax: plt.Axes, lbl: str) -> None:
    ax.text(-0.16, 1.06, lbl, transform=ax.transAxes,
            fontsize=10, fontweight="bold",
            ha="left", va="bottom", clip_on=False)


def mean_sd_by(df: pd.DataFrame, group: str, cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, part in df.groupby(group, sort=True):
        row = {group: key}
        for c in cols:
            row[f"{c}_mean"] = part[c].mean()
            row[f"{c}_sd"]   = part[c].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def confidence_band(ax, x, mu, sd, color, alpha=0.14, clip=(0, 1)):
    lo = np.clip(mu - sd, *clip)
    hi = np.clip(mu + sd, *clip)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0, zorder=2)


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("svg", "pdf", "png"):
        kw = {"bbox_inches": "tight"}
        if ext == "png":
            kw["dpi"] = 1200
        fig.savefig(OUT / f"{stem}.{ext}", **kw)
    plt.close(fig)
    print(f"  saved {stem}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 – QML Attack Paths
# ══════════════════════════════════════════════════════════════════════════════
def figure_3() -> None:
    ic     = pd.read_csv(DATA / "core_adaptive_comparison/indep_cal_mnist_metrics.csv")
    posthoc = json.loads((DATA / "multiseed_posthoc_l12.json").read_text())
    trig_m = pd.read_csv(DATA / "trigger_space/mnist_trigger_space_raw.csv")
    trig_b = pd.read_csv(DATA / "trigger_space/blood_trigger_space_raw.csv")

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_IN, 5.2))

    # ── a: XYZ AUC / ASR / CA vs λ_r with confidence bands ──────────────────
    ax = axes[0, 0]
    agg = mean_sd_by(ic, "lambda_r", ["auc_all", "asr", "ca"])
    for col, color, marker, lbl in [
        ("auc_all", BLUE,  "o", "XYZ AUC"),
        ("asr",     CORAL, "s", "ASR"),
        ("ca",      TEAL,  "^", "CA"),
    ]:
        mu = agg[f"{col}_mean"].values
        sd = agg[f"{col}_sd"].values
        lx = agg["lambda_r"].values
        ax.plot(lx, mu, color=color, marker=marker, label=lbl, zorder=3)
        confidence_band(ax, lx, mu, sd, color, clip=(0, 1.05))

    ax.axhline(0.9, color=PALE_GRID, linestyle="--", linewidth=0.8, zorder=1)
    ax.axhline(0.7, color=PALE_GRID, linestyle=":",  linewidth=0.8, zorder=1)
    # Feasible zone: CA≥0.9 and ASR≥0.7
    ax.fill_between([0, 2], 0.7, 0.9, color=TEAL, alpha=0.07, linewidth=0, zorder=0)
    ax.text(0.95, 0.78, "Feasible\nzone", fontsize=5.5, color=TEAL,
            ha="center", va="center", style="italic")
    # Trade-off annotation
    ax.annotate("CA–AUC\ntrade-off", xy=(1.5, 0.73), fontsize=5.5,
                color=GREY, ha="center", style="italic")

    ax.set_xticks([0, 0.5, 1, 2])
    ax.set_xlabel(r"Detector-evasion weight $\lambda_r$")
    ax.set_ylabel("Metric")
    ax.set_ylim(0.35, 1.06)
    ax.legend(ncol=3, loc="lower left",
              handlelength=1.2, handletextpad=0.3, columnspacing=0.6,
              borderpad=0, labelspacing=0.3)
    ygrid(ax)
    panel_label(ax, "a")

    # ── b: XYZ AUC vs Clean accuracy scatter, coloured by λ_r ────────────────
    ax = axes[0, 1]
    for lam, part in ic.groupby("lambda_r"):
        ax.scatter(part["ca"], part["auc_all"],
                   s=22, color=LAM_COLORS[lam],
                   edgecolors=BLACK, linewidths=0.3,
                   label=rf"$\lambda_r={lam:g}$", zorder=3)

    ax.axvline(0.9, color=BLACK, linestyle="--", linewidth=0.8, zorder=2)
    ax.text(0.905, 0.715, "Accuracy\nthreshold",
            fontsize=5.5, color=BLACK, va="bottom", style="italic")

    ax.set_xlabel("Clean accuracy")
    ax.set_ylabel("XYZ AUC")
    ax.set_xlim(0.38, 1.02)
    ax.set_ylim(0.70, 1.01)
    ax.legend(ncol=2, loc="lower left",
              handlelength=0.8, handletextpad=0.3, columnspacing=0.5,
              borderpad=0, labelspacing=0.2,
              scatterpoints=1, markerscale=0.9)
    ygrid(ax)
    panel_label(ax, "b")

    # ── c: 3D grouped bars – Seed × Before/After × AUC ──────────────────────────
    axes[1, 0].remove()
    ax = fig.add_subplot(2, 2, 3, projection="3d")

    ordered    = sorted(posthoc, key=lambda z: z["seed"])
    seeds_num  = [item["seed"]                for item in ordered]
    baselines  = [item["baseline"]["auc_xyz"] for item in ordered]
    finals_auc = [item["final"]["auc_xyz"]    for item in ordered]
    deltas     = [f - b for f, b in zip(finals_auc, baselines)]

    z_floor = 0.90
    x_pos   = np.arange(len(ordered), dtype=float)
    bw, bd  = 0.28, 0.26
    y_base, y_final = 0.0, 1.0

    # Pastel colours (lighter than semantic palette)
    C_BASE     = "#C8E6F5"   # pastel blue
    C_IMPROVED = "#82C9C4"   # pastel teal
    C_DEGRADED = "#F5B8A8"   # pastel coral

    for i, (b, f, d) in enumerate(zip(baselines, finals_auc, deltas)):
        fc = C_IMPROVED if d >= 0 else C_DEGRADED

        ax.bar3d(x_pos[i] - bw / 2, y_base, z_floor,
                 bw, bd, b - z_floor,
                 color=C_BASE, alpha=0.92, edgecolor="white", linewidth=0.5)

        ax.bar3d(x_pos[i] - bw / 2, y_final, z_floor,
                 bw, bd, f - z_floor,
                 color=fc, alpha=0.92, edgecolor="white", linewidth=0.5)

    # Δ summary as compact 2D text block (avoids 3D crowding)
    delta_lines = [f"Seed {s}: Δ{d:+.3f}" for s, d in zip(seeds_num, deltas)]
    ax.text2D(0.97, 0.02,
              "\n".join(delta_lines),
              transform=ax.transAxes,
              fontsize=5.6, color=BLACK, va="bottom", ha="right",
              linespacing=1.45,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                        edgecolor=PALE_GRID, linewidth=0.5, alpha=0.85))

    # Threshold note
    ax.text2D(0.03, 0.02,
              "Threshold = 0.80\n(below axis)",
              transform=ax.transAxes,
              fontsize=5.6, color=AMBER, style="italic", va="bottom")

    # Axis styling
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(s) for s in seeds_num], fontsize=6.5)
    ax.set_yticks([y_base + bd / 2, y_final + bd / 2])
    ax.set_yticklabels(["Baseline", "Step 200"], fontsize=6.5)
    ax.set_zlim(z_floor, 1.015)
    ax.set_zlabel("XYZ AUC", fontsize=7.5, labelpad=5)
    ax.set_xlabel("Seed", fontsize=7.5, labelpad=5)
    ax.zaxis.set_tick_params(labelsize=6)
    ax.view_init(elev=24, azim=-42)

    # Clean panes
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(PALE_GRID)

    # Match panel-a x-extent: explicitly set axes position
    ax.set_position([0.09, 0.09, 0.40, 0.42])

    # Legend
    from matplotlib.patches import Patch
    leg_h = [Patch(facecolor=C_BASE,     label="Baseline"),
             Patch(facecolor=C_IMPROVED, label="↑ Improved (Step 200)"),
             Patch(facecolor=C_DEGRADED, label="↓ Degraded (Step 200)")]
    ax.legend(handles=leg_h, loc="upper left", bbox_to_anchor=(-0.05, 1.08),
              fontsize=5.8, frameon=False, borderpad=0, labelspacing=0.25)

    ax.text2D(-0.08, 1.06, "c", transform=ax.transAxes,
              fontsize=10, fontweight="bold", ha="left", va="bottom")

    # ── d: Trigger-space trade-off – λ on x-axis, gradient curves ────────────
    # x = evenly-spaced lambda index for clean label spacing
    ax = axes[1, 1]
    all_lam   = [0.0, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0]
    lam_idx   = np.arange(len(all_lam))
    lam_ticks = [str(v) for v in all_lam]

    blue_cmap = LinearSegmentedColormap.from_list("", ["#92C5DE", BLUE])
    red_cmap  = LinearSegmentedColormap.from_list("", ["#F4A582", CORAL])

    def _gradient_line(ax_in, xi, yi, cmap, lw=1.8, zorder=3):
        """Draw a line whose color grades from light (start) to dark (end)."""
        pts  = np.array([xi, yi]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        fracs = np.linspace(0.35, 1.0, len(segs))
        cols = [cmap(f) for f in fracs]
        lc = LineCollection(segs, colors=cols, linewidth=lw, zorder=zorder)
        ax_in.add_collection(lc)

    series_cfg = [
        (trig_m, blue_cmap, BLUE,  "o", "MNIST",       "auc",      "-"),
        (trig_m, blue_cmap, BLUE,  "o", "MNIST",       "asr_test", "--"),
        (trig_b, red_cmap,  CORAL, "s", "BloodMNIST",  "auc",      "-"),
        (trig_b, red_cmap,  CORAL, "s", "BloodMNIST",  "asr_test", "--"),
    ]

    # twin axis: AUC left, ASR right
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(axis="y", labelsize=7, direction="out", pad=2)
    ax2.set_ylabel("Trigger-space ASR", fontsize=8, fontweight="bold", color=BLACK)

    legend_handles = []
    legend_labels  = []

    for df, cmap, base_color, marker, lbl, metric, ls in series_cfg:
        q = df[(df["attack_type"] == "qml_adaptive") &
               (df["feature_space"] == "XYZ") &
               df["lambda"].isin(all_lam)]
        g  = q.groupby("lambda")[metric].agg(["mean", lambda v: v.std(ddof=0)])
        g.columns = ["mean", "sd"]
        g  = g.reindex(all_lam)
        mu = g["mean"].values
        sd = g["sd"].values

        target_ax = ax if metric == "auc" else ax2

        # Gradient line
        _gradient_line(target_ax, lam_idx.astype(float), mu, cmap,
                       lw=1.8, zorder=3)
        # Confidence band
        target_ax.fill_between(
            lam_idx,
            mu - sd, mu + sd,
            color=base_color, alpha=0.12, linewidth=0, zorder=2)

        # Gradient markers
        for i, (xi, yi) in enumerate(zip(lam_idx, mu)):
            frac = 0.35 + 0.65 * i / (len(lam_idx) - 1)
            target_ax.scatter(xi, yi, color=cmap(frac), s=32,
                              marker=marker, zorder=5,
                              edgecolors="white", linewidths=0.5)

        # Legend entries (one per dataset × metric)
        from matplotlib.lines import Line2D
        metric_lbl = "AUC" if metric == "auc" else "ASR"
        h = Line2D([0], [0], color=base_color, linestyle=ls,
                   linewidth=1.6, marker=marker, markersize=4,
                   label=f"{lbl} {metric_lbl}")
        legend_handles.append(h)
        legend_labels.append(f"{lbl} {metric_lbl}")

    ax.set_xticks(lam_idx, lam_ticks)
    ax.set_xlabel(r"Trigger weight $\lambda$")
    ax.set_ylabel("XYZ AUC", color=BLACK)
    ax.autoscale_view()
    ax2.autoscale_view()

    # Add light separator between AUC / ASR y-axes visually via right spine colour
    ax2.spines["right"].set_edgecolor("#888888")
    ax2.spines["right"].set_linewidth(0.7)

    # Deduplicate legend
    seen = set()
    uh, ul = [], []
    for h, l in zip(legend_handles, legend_labels):
        if l not in seen:
            seen.add(l); uh.append(h); ul.append(l)

    ax.legend(uh, ul, ncol=2, loc="lower left",
              handlelength=1.4, handletextpad=0.3, columnspacing=0.7,
              borderpad=0, labelspacing=0.3, fontsize=6.0)
    ygrid(ax)
    panel_label(ax, "d")

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.11, top=0.97,
                        wspace=0.40, hspace=0.48)
    save(fig, "Figure_3_qml_attack_paths_NC_v2")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 – Mechanism Interventions
# ══════════════════════════════════════════════════════════════════════════════
def figure_4() -> None:
    from matplotlib.patches import Patch

    ladder = pd.read_csv(DATA / "mechanism_interventions/pauli_ladder/ladder_results.csv")
    e2_raw = pd.read_csv(DATA / "mechanism_interventions/nullspace/e2_raw.csv")
    e2     = pd.read_csv(DATA / "mechanism_interventions/nullspace/e2_summary.csv")
    gains  = pd.read_csv(DATA / "mechanism_interventions/dim_matched_control"
                               "/measurement_basis_expansion_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_IN, 5.5))

    # ── a: Full measurement ladder – AUC across all 7 feature sets, 4 lines ───
    # Shows systematic improvement: single-axis → double-axis → XYZ (triple)
    # Both datasets and both conditions (baseline dashed, adaptive solid).
    ax = axes[0, 0]
    fs_order = ["Z", "X", "Y", "XZ", "XY", "YZ", "XYZ"]
    x_idx    = np.arange(len(fs_order))

    series = [
        ("MNIST",      "baseline", "#92C5DE", "--", "o", "MNIST baseline"),
        ("MNIST",      "adaptive", BLUE,      "-",  "o", "MNIST adaptive"),
        ("BloodMNIST", "baseline", "#F4A582", "--", "s", "BloodMNIST baseline"),
        ("BloodMNIST", "adaptive", CORAL,     "-",  "s", "BloodMNIST adaptive"),
    ]
    for ds, cond, color, ls, mk, lbl in series:
        sub  = ladder[(ladder["dataset"]==ds) & (ladder["condition"]==cond)]
        mu   = np.array([sub[sub["feature_set"]==f]["auc"].mean() for f in fs_order])
        sd   = np.array([sub[sub["feature_set"]==f]["auc"].std(ddof=0) for f in fs_order])
        ax.plot(x_idx, mu, color=color, linestyle=ls, marker=mk,
                markersize=4.5, linewidth=1.5, label=lbl, zorder=3)
        ax.fill_between(x_idx,
                        np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1),
                        color=color, alpha=0.10, linewidth=0, zorder=2)

    # Group dividers: single / double / triple
    ax.axvline(2.5, color=PALE_GRID, linewidth=0.8, zorder=1)
    ax.axvline(5.5, color=PALE_GRID, linewidth=0.8, zorder=1)
    for xc, lbl in [(1.0, "Single axis"), (4.0, "Double axis"), (6.0, "Triple")]:
        ax.text(xc, 0.30, lbl, fontsize=5.5, color="#888888",
                ha="center", style="italic")

    ax.set_xticks(x_idx, fs_order)
    ax.set_xlabel("Measured feature set")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.27, 1.08)
    ax.legend(ncol=2, loc="upper left",
              handlelength=1.2, handletextpad=0.3, columnspacing=0.5,
              borderpad=0, labelspacing=0.25, fontsize=5.8)
    ygrid(ax)
    panel_label(ax, "a")

    # ── b: Dimension-matched control – actual AUC for XYZ/Z/Z_expanded ────────
    # Uses adaptive_frontier data at λ=50, showing three detector types side-by-side.
    # "Z expanded" is the dimension-matched baseline: extra Z features ≠ XYZ gain.
    ax = axes[0, 1]
    front   = gains[gains["scope"] == "adaptive_frontier"]
    datasets_b  = ["MNIST", "BloodMNIST"]
    detectors   = ["XYZ", "Z", "Z_expanded"]
    det_colors  = [BLUE, "#5BA3C9", "#92C5DE"]
    det_labels  = ["XYZ", "Z only", "Z expanded\n(dim-matched)"]
    bw_b = 0.22
    offs = [-bw_b, 0, bw_b]
    x_ds = np.arange(len(datasets_b))

    for det, color, lbl, off in zip(detectors, det_colors, det_labels, offs):
        vals = []
        for ds in datasets_b:
            row = front[(front["dataset"]==ds) & (front["detector"]==det)]
            vals.append(float(row["auc"].values[0]) if len(row) > 0 else np.nan)
        ax.bar(x_ds + off, vals, bw_b, color=color, label=lbl,
               edgecolor="white", linewidth=0.3, zorder=2)
        # Label ALL bars with their AUC value (black, consistent font size)
        for xi, v in zip(x_ds + off, vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.002, f"{v:.3f}",
                        ha="center", va="bottom",
                        fontsize=5.3, color=BLACK)

    ax.set_xticks(x_ds, datasets_b)
    ax.set_ylabel("AUC  (adaptive attack, λ = 50)")
    ax.set_ylim(0.88, 1.015)   # extra headroom for labels
    ax.legend(ncol=3, loc="lower right",
              handlelength=1.0, handletextpad=0.3, columnspacing=0.4,
              borderpad=0, labelspacing=0.25)
    ygrid(ax)
    panel_label(ax, "b")

    # ── c: Nullspace XYZ drift – projected ≈ 0 vs random ≈ 1.3 ──────────────
    # 300 per-sample points per group; shows the stark structural contrast.
    # Projected perturbations stay in the nullspace (zero XYZ drift);
    # random perturbations cause large drift and are detected.
    ax = axes[1, 0]
    rng = np.random.RandomState(0)

    # Colors: Projected = blue family (light=baseline, dark=adaptive)
    #         Random    = amber/coral family (light=baseline, dark=adaptive)
    # This gives clear 2×2 contrast with no duplicates.
    C_PROJ_BASE = "#92C5DE"   # light blue
    C_PROJ_ADAP = BLUE        # dark blue   #2166AC
    C_RAND_BASE = "#F9C77E"   # light amber
    C_RAND_ADAP = CORAL       # dark coral  #D6604D

    grps = [
        (0.0, e2_raw[e2_raw["condition"]=="baseline"]["xyz_drift_proj"],
         C_PROJ_BASE, "Projected (baseline)"),
        (1.1, e2_raw[e2_raw["condition"]=="baseline"]["xyz_drift_rand"],
         C_RAND_BASE, "Random (baseline)"),
        (2.5, e2_raw[e2_raw["condition"]=="adaptive"]["xyz_drift_proj"],
         C_PROJ_ADAP, "Projected (adaptive)"),
        (3.6, e2_raw[e2_raw["condition"]=="adaptive"]["xyz_drift_rand"],
         C_RAND_ADAP, "Random (adaptive)"),
    ]
    for pos, data, color, lbl in grps:
        vals = data.values
        jit = rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(pos + jit, vals, s=1.8, color=color,
                   alpha=0.20, linewidth=0, zorder=2)
        q25, q75 = np.percentile(vals, 25), np.percentile(vals, 75)
        ax.fill_betweenx([q25, q75], pos - 0.24, pos + 0.24,
                          color=color, alpha=0.40, linewidth=0, zorder=3)
        ax.plot([pos - 0.24, pos + 0.24], [vals.mean()] * 2,
                color=color, linewidth=2.2, zorder=4)

    # Condition separator
    ax.axvline(1.8, color=PALE_GRID, linewidth=0.8, zorder=1)
    ax.text(0.55, 1.78, "Baseline condition",
            fontsize=5.8, ha="center", color=BLACK, style="italic")
    ax.text(3.05, 1.78, "Adaptive condition",
            fontsize=5.8, ha="center", color=BLACK, style="italic")

    # Key annotation – BLACK text
    ax.text(0.55, 0.12, "Drift = 0\n(in nullspace)",
            fontsize=5.8, color=BLACK, ha="center", style="italic")

    # Single legend — sole labeling system (no duplicate bottom text labels)
    # Blue family = Projected; Amber/Coral family = Random
    leg_h = [Patch(facecolor=C_PROJ_BASE, label="Projected (baseline)"),
             Patch(facecolor=C_PROJ_ADAP, label="Projected (adaptive)"),
             Patch(facecolor=C_RAND_BASE, label="Random (baseline)"),
             Patch(facecolor=C_RAND_ADAP, label="Random (adaptive)")]
    ax.legend(handles=leg_h, ncol=2, loc="upper right",
              fontsize=5.5, frameon=True, framealpha=0.9,
              edgecolor=PALE_GRID, handlelength=0.9, handletextpad=0.3,
              columnspacing=0.5, labelspacing=0.25, borderpad=0.4)

    ax.set_xlim(-0.55, 4.2)
    ax.set_ylim(-0.42, 1.92)
    # x-axis ticks label each group once (no duplication with legend)
    ax.set_xticks([0.0, 1.1, 2.5, 3.6],
                  ["Projected\n(baseline)", "Random\n(baseline)",
                   "Projected\n(adaptive)", "Random\n(adaptive)"],
                  fontsize=5.5)
    ax.set_ylabel("XYZ drift magnitude")
    ygrid(ax)
    panel_label(ax, "c")

    # ── d: Per-seed evasion rates – projected vs random (adaptive) ───────────
    # evades_rand = 0.00 for ALL seeds: random perturbations never evade detection.
    # evades_proj varies: nullspace-aligned perturbations can partially evade,
    # but r_hidden → 0 shows the nullspace has no exploitable hidden capacity.
    ax = axes[1, 1]
    e2a = e2[e2["condition"] == "adaptive"].sort_values("seed")
    seeds_d = e2a["seed"].values
    x_d     = np.arange(len(seeds_d))
    bw_d    = 0.30

    # Harmonious blue-family pair: both from same hue, differ in lightness
    C_PROJ_D = BLUE       # dark blue  – projected CAN partially evade
    C_RAND_D = "#92C5DE"  # light blue – random NEVER evades (null result)

    for off, col, color, lbl in [
        (-bw_d / 2, "evades_proj", C_PROJ_D, "Projected (nullspace-aligned)"),
        (+bw_d / 2, "evades_rand", C_RAND_D, "Random"),
    ]:
        vals = e2a[col].values
        ax.bar(x_d + off, vals, bw_d, color=color, label=lbl,
               edgecolor="white", linewidth=0.3, zorder=2)
        # Value annotations in BLACK
        for xi, v in zip(x_d + off, vals):
            if v > 0.01:
                ax.text(xi, v + 0.015, f"{v:.2f}",
                        ha="center", va="bottom",
                        fontsize=5.5, color=BLACK, fontweight="bold")

    # annotation: random always = 0
    ax.text((len(seeds_d) - 1) / 2, 0.02,
            "Random: 0.00 (all seeds)",
            fontsize=5.5, color=BLACK, ha="center", va="bottom", style="italic")

    ax.set_xticks(x_d, [str(s) for s in seeds_d])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Evasion fraction  (adaptive condition)")
    ax.set_ylim(0, 0.72)
    ax.legend(ncol=1, loc="upper right",
              handlelength=1.0, handletextpad=0.4, borderpad=0, labelspacing=0.3)
    ygrid(ax)
    panel_label(ax, "d")

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.13, top=0.97,
                        wspace=0.42, hspace=0.55)
    save(fig, "Figure_4_mechanism_interventions_NC_v2")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 – Sampling and Hardware (redesigned)
# ══════════════════════════════════════════════════════════════════════════════
def figure_5() -> None:
    shots_df = pd.read_csv(DATA / "shot_sweep/adaptive_shot_sweep_summary.csv")
    noise_df = pd.read_csv(DATA / "shot_sweep/finite_shot_noise_xyz_degradation.csv")

    import json
    hw_mnist = json.load(open(DATA / "hardware/ibm_v3_MNIST_5q_layer12_amplitude_ibm_kingston.json"))
    hw_blood = json.load(open(DATA / "hardware/ibm_blood_layer8_amplitude_ibm_kingston.json"))

    shot_order = ["128", "512", "1024", "2048", "4096", "exact"]
    xl_shots   = ["128", "512", "1,024", "2,048", "4,096", "Exact"]
    x_shots    = np.arange(len(shot_order))

    # λ sweep colours: sequential blue family
    LAM_SPEC = [
        (0.0, "#92C5DE", "o", "λ = 0"),
        (0.5, "#4393C3", "s", "λ = 0.5"),
        (1.0, BLUE,      "^", "λ = 1"),
        (2.0, "#0a3a6e", "D", "λ = 2"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_IN, 5.5))

    # ── a: AUC vs shots, all λ – gradient lines ──────────────────────────────
    # Each λ curve uses a LineCollection so colour fades from pale→full
    # left→right, visually encoding the "progressive convergence" to Exact.
    ax = axes[0, 0]

    # Helper: make N-segment LineCollection with alpha ramp pale→full colour
    def gradient_line(ax, xs, ys, hex_color, n_seg=200, lw=1.5):
        from matplotlib.colors import to_rgba
        pts   = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segs  = np.concatenate([pts[:-1], pts[1:]], axis=1)
        t     = np.linspace(0, 1, len(segs))
        r, g_c, b, _ = to_rgba(hex_color)
        colors = [(r, g_c, b, 0.18 + 0.82 * ti) for ti in t]
        lc = LineCollection(segs, colors=colors, linewidth=lw, zorder=3)
        ax.add_collection(lc)
        return lc

    legend_handles = []
    for lam, color, marker, lbl in LAM_SPEC:
        sub = shots_df[shots_df["lambda_r"] == lam].copy()
        g   = sub.groupby("shots")[["auc_mean"]].mean().reindex(shot_order)
        mu  = g["auc_mean"].values
        xs_finite = x_shots[:-1].astype(float)
        ys_finite = mu[:-1]

        # Gradient line through finite-shot points
        gradient_line(ax, xs_finite, ys_finite, color, lw=1.6)

        # Markers on each finite-shot point, darkening towards right
        from matplotlib.colors import to_rgba
        r, g_c, b, _ = to_rgba(color)
        for xi, yi, ti in zip(xs_finite, ys_finite,
                               np.linspace(0.3, 1.0, len(xs_finite))):
            ax.scatter(xi, yi, s=14, color=(r, g_c, b, ti),
                       marker=marker, zorder=4, linewidths=0)

        # Exact: hollow marker at rightmost position, full colour
        ax.scatter(x_shots[-1], mu[-1], s=22, facecolors="none",
                   edgecolors=color, linewidths=1.1,
                   marker=marker, zorder=5)
        # Dashed convergence arrow from last finite to exact
        ax.annotate("", xy=(x_shots[-1], mu[-1]),
                    xytext=(xs_finite[-1], ys_finite[-1]),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=0.7, mutation_scale=5,
                                    linestyle="dashed"))

        # Legend proxy (solid line + marker)
        from matplotlib.lines import Line2D
        legend_handles.append(
            Line2D([0], [0], color=color, marker=marker,
                   markersize=4, linewidth=1.4, label=lbl))

    # Vertical dashed separator before "Exact"
    ax.axvline(x_shots[-1] - 0.5, color=PALE_GRID,
               linewidth=0.8, linestyle="--", zorder=1)
    ax.text(x_shots[-1], 0.838, "Exact", fontsize=5.5, color=BLACK,
            ha="center", va="bottom", style="italic")

    ax.set_xticks(x_shots, xl_shots, rotation=25, ha="right")
    ax.set_xlim(-0.5, x_shots[-1] + 0.5)
    ax.set_xlabel("Shots per observable")
    ax.set_ylabel("XYZ AUC")
    ax.set_ylim(0.835, 0.975)
    ax.legend(handles=legend_handles, ncol=2, loc="lower right",
              handlelength=1.2, handletextpad=0.3,
              columnspacing=0.5, labelspacing=0.25, borderpad=0.3)
    ygrid(ax)
    panel_label(ax, "a")

    # ── b: TPR@5%FPR vs shot count – all λ, mirrors panel a ─────────────────
    ax = axes[0, 1]

    legend_handles_b = []
    for lam, color, marker, lbl in LAM_SPEC:
        sub = shots_df[shots_df["lambda_r"] == lam].copy()
        g   = sub.groupby("shots")[["tpr5_mean"]].mean().reindex(shot_order)
        mu  = g["tpr5_mean"].values
        xs_finite = x_shots[:-1].astype(float)
        ys_finite = mu[:-1]

        gradient_line(ax, xs_finite, ys_finite, color, lw=1.6)

        from matplotlib.colors import to_rgba
        r, g_c, b, _ = to_rgba(color)
        for xi, yi, ti in zip(xs_finite, ys_finite,
                               np.linspace(0.3, 1.0, len(xs_finite))):
            ax.scatter(xi, yi, s=14, color=(r, g_c, b, ti),
                       marker=marker, zorder=4, linewidths=0)

        ax.scatter(x_shots[-1], mu[-1], s=22, facecolors="none",
                   edgecolors=color, linewidths=1.1,
                   marker=marker, zorder=5)
        ax.annotate("", xy=(x_shots[-1], mu[-1]),
                    xytext=(xs_finite[-1], ys_finite[-1]),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=0.7, mutation_scale=5,
                                    linestyle="dashed"))

        from matplotlib.lines import Line2D
        legend_handles_b.append(
            Line2D([0], [0], color=color, marker=marker,
                   markersize=4, linewidth=1.4, label=lbl))

    ax.axvline(x_shots[-1] - 0.5, color=PALE_GRID,
               linewidth=0.8, linestyle="--", zorder=1)
    ax.text(x_shots[-1], 0.445, "Exact", fontsize=5.5, color=BLACK,
            ha="center", va="bottom", style="italic")

    ax.set_xticks(x_shots, xl_shots, rotation=25, ha="right")
    ax.set_xlim(-0.5, x_shots[-1] + 0.5)
    ax.set_xlabel("Shots per observable")
    ax.set_ylabel("TPR@5% FPR")
    ax.set_ylim(0.44, 0.84)
    ax.legend(handles=legend_handles_b, ncol=2, loc="lower right",
              handlelength=1.2, handletextpad=0.3,
              columnspacing=0.5, labelspacing=0.25, borderpad=0.3)
    ygrid(ax)
    panel_label(ax, "b")

    # ── c: AUC change vs noise scenario ─────────────────────────────────────
    ax = axes[1, 0]
    sc_order = ["shot512",
                "shot512_readout3pct",
                "shot512_depol3pct",
                "shot512_readout3pct_depol3pct"]
    xl_c = ["No noise", "Readout\n3%", "Depol.\n3%", "Both\n3%"]

    for ds, color, marker, lbl in [
        ("MNIST_5q",         BLUE,  "o", "MNIST"),
        ("Blood_8q_eps0.8",  CORAL, "s", "BloodMNIST"),
    ]:
        d = (noise_df[noise_df["dataset"] == ds]
             .set_index("scenario").reindex(sc_order))
        ax.plot(np.arange(4), d["delta_auc_vs_exact"],
                color=color, marker=marker, markersize=4,
                linewidth=1.2, label=lbl, zorder=3)
        # value labels at each point
        for xi, v in enumerate(d["delta_auc_vs_exact"].values):
            if not np.isnan(v):
                ax.text(xi, v - 0.003 if v < 0 else v + 0.002,
                        f"{v:+.3f}", ha="center",
                        va="top" if v < 0 else "bottom",
                        fontsize=5.0, color=color)

    ax.axhline(0, color=BLACK, linewidth=0.8, zorder=2)
    ax.set_xticks(np.arange(4), xl_c)
    ax.set_ylabel("ΔAUC vs exact statevector")
    ax.legend(loc="lower left",
              handlelength=1.2, handletextpad=0.35, borderpad=0.3, labelspacing=0.3)
    ygrid(ax)
    panel_label(ax, "c")

    # ── d: per-axis AUC on IBM hardware – MNIST vs BloodMNIST ───────────────
    ax = axes[1, 1]
    axes_lbl = ["XYZ", "X", "Y", "Z"]
    hw_key   = {"XYZ": "XYZ", "X": "X_only", "Y": "Y_only", "Z": "Z_only"}

    ds_specs_d = [
        ("MNIST",      hw_mnist["hardware"], BLUE,  "o"),
        ("BloodMNIST", hw_blood["hardware"], CORAL, "s"),
    ]
    bw_d2 = 0.32
    x_ax = np.arange(len(axes_lbl))
    offs_d2 = [-bw_d2/2, bw_d2/2]

    for (lbl, hw_data, color, marker), off in zip(ds_specs_d, offs_d2):
        vals = [hw_data[hw_key[a]]["AUC"] for a in axes_lbl]
        ax.bar(x_ax + off, vals, bw_d2, color=color,
               label=lbl, edgecolor="white", linewidth=0.3,
               alpha=0.88, zorder=2)
        for xi, v in zip(x_ax + off, vals):
            ax.text(xi, v + 0.008, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=5.3, color=BLACK)

    ax.axhline(1.0, color=BLACK, linewidth=0.7, linestyle="--",
               zorder=1, label="Simulator (exact)")
    ax.text(3.5, 1.003, "Sim.", fontsize=5.3, color=BLACK, ha="right", va="bottom")

    ax.set_xticks(x_ax, axes_lbl)
    ax.set_xlabel("Measurement axis")
    ax.set_ylabel("AUC on IBM hardware")
    ax.set_ylim(0.30, 1.08)
    ax.legend(ncol=1, loc="lower left", handlelength=1.0,
              handletextpad=0.3, labelspacing=0.25, borderpad=0.3)
    ygrid(ax)
    panel_label(ax, "d")

    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.97,
                        wspace=0.50, hspace=0.52)
    save(fig, "Figure_5_sampling_and_hardware_NC_v2")


# ══════════════════════════════════════════════════════════════════════════════
# Supplementary Figure 1 – Classical full grid
# ══════════════════════════════════════════════════════════════════════════════
def supplementary_1() -> None:
    controls = pd.read_csv(DATA / "classical_controls/training_time_summary_classical.csv")
    specs = [
        ("mnist", "sep_xyz",  "Separation",                   "a"),
        ("mnist", "mahal_d",  r"Mahalanobis $D$",             "b"),
        ("blood", "sep_xyz",  "Separation",                   "c"),
        ("blood", "mahal_d",  r"Mahalanobis $D$",             "d"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(SUPP_W, 5.0))

    for ax, (ds, metric, ylabel, lbl) in zip(axes.flat, specs):
        for control, color, marker, name in [
            ("bloch",       CORAL,   "o", "Classical Bloch"),
            ("unconstrained", DKGREEN, "s", "Unconstrained"),
        ]:
            source = f"{control}_{ds}"
            d = mean_sd_by(controls[controls["source"] == source],
                           "lambda_r", [metric])
            mu = d[f"{metric}_mean"].values
            sd = d[f"{metric}_sd"].values
            lx = d["lambda_r"].values
            ax.plot(lx, mu, color=color, marker=marker,
                    linewidth=1.5, label=name, zorder=3)
            if metric != "mahal_d":
                confidence_band(ax, lx, mu, sd, color, clip=(0, 1.05))
            else:
                ax.fill_between(lx,
                                np.maximum(mu - sd, 1e-2),
                                mu + sd,
                                color=color, alpha=0.13, linewidth=0, zorder=2)

        ax.set_xlabel(r"Adaptive weight $\lambda_r$")
        ax.set_ylabel(ylabel)
        if metric == "mahal_d":
            ax.set_yscale("log")
        if lbl == "a":
            ax.legend(handlelength=1.3, handletextpad=0.4, borderpad=0, labelspacing=0.3)
        ygrid(ax)
        panel_label(ax, lbl)

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.97,
                        wspace=0.38, hspace=0.46)
    save(fig, "Supplementary_Figure_1_classical_full_grid_NC_v2")


# ══════════════════════════════════════════════════════════════════════════════
# Supplementary Figure 2 – QML checkpoint trajectories (mean ± band)
# ══════════════════════════════════════════════════════════════════════════════
def supplementary_2() -> None:
    history_dir = FULL / "results/training_time/new_version"
    lam_list    = [0.0, 0.5, 1.0, 2.0]
    labels      = ["a", "b", "c", "d"]

    fig, axes = plt.subplots(2, 2, figsize=(SUPP_W, 5.0),
                             sharex=True, sharey=True)

    for ax, lam, lbl in zip(axes.flat, lam_list, labels):
        epochs_all, auc_all = [], []
        for seed in range(42, 47):
            p = history_dir / f"history_seed={seed}_L12_lr{lam:.1f}.json"
            h = pd.DataFrame(json.loads(p.read_text()))
            epochs_all.append(h["epoch"].values)
            auc_all.append(h["auc_xyz"].values)

        # Align on common epoch grid
        epochs = epochs_all[0]
        mat    = np.vstack(auc_all)          # shape (5, T)
        mu     = mat.mean(axis=0)
        sd     = mat.std(axis=0, ddof=0)

        ax.plot(epochs, mu, color=BLUE, linewidth=1.6, zorder=3)
        ax.fill_between(epochs,
                        np.clip(mu - sd, 0.68, 1.02),
                        np.clip(mu + sd, 0.68, 1.02),
                        color=BLUE, alpha=0.18, linewidth=0, zorder=2)

        # Individual seed traces – very faint
        for auc_s in auc_all:
            ax.plot(epochs, auc_s,
                    color=BLUE, alpha=0.18, linewidth=0.5, zorder=1)

        ax.text(0.05, 0.08, rf"$\lambda_r={lam:g}$",
                transform=ax.transAxes, fontsize=7.5, color=BLACK)
        ax.set_ylim(0.70, 1.01)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("XYZ AUC")
        ygrid(ax)
        panel_label(ax, lbl)

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.97,
                        wspace=0.30, hspace=0.40)
    save(fig, "Supplementary_Figure_2_qml_checkpoint_trajectories_v2")


# ══════════════════════════════════════════════════════════════════════════════
# Supplementary Figure 3 – Measurement ladder full (2×2)
# Row = metric (AUC / TPR@5%FPR); Col = condition (baseline / adaptive)
# Dataset: MNIST only (BloodMNIST adaptive data not available)
# ══════════════════════════════════════════════════════════════════════════════
def supplementary_3() -> None:
    ladder     = pd.read_csv(DATA / "mechanism_interventions/pauli_ladder/ladder_results.csv")
    feat_order = ["Z", "X", "Y", "XZ", "XY", "YZ", "XYZ"]
    clrs       = [MSET_COLORS[f] for f in feat_order]
    edges      = [MSET_EDGE[f]   for f in feat_order]
    mnist      = ladder[ladder["dataset"] == "MNIST"]

    # layout: [row0=AUC, row1=TPR] × [col0=baseline, col1=adaptive]
    specs = [
        ("baseline", "AUC",        "auc",      (0.35, 1.08), "a"),
        ("adaptive",  "AUC",       "auc",      (0.35, 1.08), "b"),
        ("baseline", "TPR@5% FPR", "tpr_5fpr", (0.00, 1.05), "c"),
        ("adaptive",  "TPR@5% FPR","tpr_5fpr", (0.00, 1.05), "d"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(SUPP_W, 5.0), sharex=True)

    for ax, (cond, ylabel, metric, ylim, lbl) in zip(axes.flat, specs):
        d = mnist[mnist["condition"] == cond]
        means = [d[d["feature_set"] == f][metric].mean() for f in feat_order]
        sds   = [d[d["feature_set"] == f][metric].std(ddof=0) for f in feat_order]

        ax.bar(np.arange(7), means, yerr=sds,
               color=clrs, edgecolor=edges, linewidth=0.5,
               capsize=2.5, zorder=2,
               error_kw=dict(linewidth=0.7, capthick=0.7))

        # Value label on every bar
        for xi, m, s in zip(np.arange(7), means, sds):
            ax.text(xi, m + s + 0.015, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=4.8, color=BLACK)

        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        if lbl in ("a", "b"):
            cond_label = "Baseline" if cond == "baseline" else "Adaptive attack"
            ax.set_title(cond_label, fontsize=7, pad=3, color=BLACK)
        if lbl in ("c", "d"):
            ax.set_xticks(np.arange(7), feat_order, rotation=35, ha="right")
        ygrid(ax)
        panel_label(ax, lbl)

    from matplotlib.patches import Patch
    handles = [
        Patch(color="#D1E5F0", edgecolor="#4393C3", linewidth=0.5, label="Single axis"),
        Patch(color="#4393C3", edgecolor="#1a5c8a", linewidth=0.5, label="Double axis"),
        Patch(color=BLUE,      edgecolor="#0d3a5c", linewidth=0.5, label="Triple axis (XYZ)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               handlelength=1.0, handletextpad=0.4,
               borderpad=0.3, labelspacing=0.3,
               bbox_to_anchor=(0.5, 0.0))

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.95,
                        wspace=0.32, hspace=0.28)
    save(fig, "Supplementary_Figure_3_measurement_ladder_full_v2")


# ══════════════════════════════════════════════════════════════════════════════
# Supplementary Figure 4 – Post-hoc trajectories (mean ± band)
# ══════════════════════════════════════════════════════════════════════════════
def supplementary_4() -> None:
    posthoc_all = json.loads(
        (FULL / "results/full_results.json").read_text())
    layers = [4, 8, 12]
    labels = ["a", "b", "c"]
    # colour per layer: light→dark
    layer_colors = {4: "#92C5DE", 8: "#4393C3", 12: BLUE}

    fig, axes = plt.subplots(1, 3, figsize=(SUPP_W, 2.8),
                             sharex=True, sharey=True)

    for ax, layer, lbl in zip(axes, layers, labels):
        items = [x for x in posthoc_all if x["layer"] == layer]
        steps_all, auc_all = [], []
        for item in items:
            h = pd.DataFrame(item["history"])
            steps_all.append(h["step"].values)
            auc_all.append(h["auc_xyz"].values)

        if steps_all:
            steps = steps_all[0]
            mat   = np.vstack(auc_all)
            mu    = mat.mean(axis=0)
            sd    = mat.std(axis=0, ddof=0)
            color = layer_colors[layer]

            ax.plot(steps, mu, color=color, linewidth=1.6, zorder=3)
            ax.fill_between(steps,
                            np.clip(mu - sd, 0.68, 1.02),
                            np.clip(mu + sd, 0.68, 1.02),
                            color=color, alpha=0.18, linewidth=0, zorder=2)
            # Faint individual traces
            for auc_s in auc_all:
                ax.plot(steps, auc_s,
                        color=color, alpha=0.20, linewidth=0.5, zorder=1)

        ax.axhline(0.8, color=AMBER, linestyle="--", linewidth=1.0, zorder=4)
        if lbl == "a":
            ax.text(5, 0.81, "Detection threshold",
                    fontsize=5.5, color=AMBER, va="bottom", style="italic")

        ax.set_xlabel(f"Post-hoc step\n(L={layer})")
        ax.set_ylim(0.70, 1.01)
        if lbl == "a":
            ax.set_ylabel("XYZ AUC")
        ygrid(ax)
        panel_label(ax, lbl)

    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.22, top=0.97, wspace=0.18)
    save(fig, "Supplementary_Figure_4_posthoc_trajectories_v2")


# ── Figure 2 ──────────────────────────────────────────────────────────────────
def figure_2():
    DATA2 = DATA / "classical_controls"
    adap  = pd.read_csv(DATA2 / "ss_k1_adap2s_lr1_5seeds.csv")
    ctrl  = pd.read_csv(DATA2 / "training_time_summary_classical.csv")
    mn    = ctrl[ctrl["source"].isin(["bloch_mnist", "unconstrained_mnist"])]

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_IN, 5.2))

    # ── a: attack escalation – jser per seed for fix / rev / two-sided ────────
    ax    = axes[0, 0]
    seeds = adap["seed"].values
    x     = np.arange(len(seeds))
    bw    = 0.24
    stages = [
        ("jser_fix", "#F9C77E", "One-sided (fix)"),
        ("jser_rev", AMBER,     "One-sided (rev)"),
        ("jser_2s",  CORAL,     "Two-sided"),
    ]
    for (col, color, lbl), off in zip(stages, [-bw, 0, bw]):
        ax.bar(x + off, adap[col].values, bw, color=color,
               label=lbl, edgecolor="white", linewidth=0.3, zorder=2)

    mean_2s = adap["jser_2s"].mean()
    ax.axhline(mean_2s, color=BLUE, linestyle="--", linewidth=1.6, zorder=3, alpha=0.8)
    ax.text(len(seeds) - 0.35, mean_2s + 0.015, f"mean = {mean_2s:.2f}",
            fontsize=6.0, color=BLUE, va="bottom", fontweight="bold")
    ax.set_xticks(x, seeds.astype(str))
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Joint evasion rate")
    ax.set_ylim(0, 1.12)
    ax.legend(ncol=1, loc="upper left",
              handlelength=1.0, handletextpad=0.35, borderpad=0.3, labelspacing=0.25)
    ygrid(ax); panel_label(ax, "a")

    # ── b: escalating attack strategy – mean TPR & JSER ───────────────────────
    ax = axes[0, 1]
    tpr_mean_2s  = float(adap["tpr_2s_at_5pct_fpr"].mean())
    jser_mean_2s = float(adap["jser_2s"].mean())
    tpr_vals  = [0.589, 0.674, tpr_mean_2s]
    jser_vals = [0.411, 0.326, jser_mean_2s]
    labels    = ["Standard", "One-sided", "Two-sided"]
    xx  = np.arange(3)
    bw2 = 0.35
    ax.bar(xx - bw2/2, tpr_vals,  bw2, color=TEAL,  label="TPR@5% FPR",   zorder=2)
    ax.bar(xx + bw2/2, jser_vals, bw2, color=AMBER, label="Joint evasion", zorder=2)
    for i, (tv, jv) in enumerate(zip(tpr_vals, jser_vals)):
        ax.text(i - bw2/2, tv  + 0.02, f"{tv:.2f}",
                ha="center", va="bottom", fontsize=5.8, color=BLACK)
        ax.text(i + bw2/2, jv  + 0.02, f"{jv:.2f}",
                ha="center", va="bottom", fontsize=5.8, color=BLACK)
    ax.set_xticks(xx, labels)
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1.15)
    ax.legend(ncol=2, loc="upper left",
              handlelength=1.2, handletextpad=0.35,
              columnspacing=0.7, borderpad=0, labelspacing=0.3)
    ygrid(ax); panel_label(ax, "b")

    # ── c: jser + Mahalanobis vs λ (twin axes) ────────────────────────────────
    ax  = axes[1, 0]
    ax2 = ax.twinx()
    cond_specs = [
        ("bloch_mnist",         CORAL,   "o", "Classical Bloch"),
        ("unconstrained_mnist", DKGREEN, "s", "Unconstrained"),
    ]
    lam_vals = [0.0, 0.5, 1.0, 2.0]
    lx = np.array(lam_vals)

    def _mean_sd(df, col):
        grp = df.groupby("lambda_r")[col]
        return grp.mean().reindex(lam_vals).values, grp.std(ddof=0).reindex(lam_vals).values

    for source, color, marker, lbl in cond_specs:
        sub = mn[mn["source"] == source]
        mu_j, sd_j = _mean_sd(sub, "jser_xyz_5fpr")
        mu_m, _    = _mean_sd(sub, "mahal_d")
        ax.plot(lx, mu_j, color=color, marker=marker, linewidth=1.6, label=lbl, zorder=3)
        ax.fill_between(lx, np.clip(mu_j - sd_j, 0, 1), np.clip(mu_j + sd_j, 0, 1),
                        color=color, alpha=0.12, linewidth=0, zorder=2)
        ax2.plot(lx, mu_m, color=color, marker=marker,
                 linewidth=1.0, linestyle="--", alpha=0.65, zorder=2)

    ax.set_xlabel(r"Adaptive weight $\lambda_r$")
    ax.set_ylabel("Joint evasion rate")
    ax.set_ylim(-0.06, 1.12)
    ax.set_xticks(lam_vals)
    ax2.set_ylabel("Mahalanobis distance", color=BLACK)
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelsize=6.5)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.7)

    from matplotlib.lines import Line2D as _L2D
    leg_h = [
        _L2D([0],[0], color=CORAL,   marker="o", linewidth=1.4, label="Bloch – JSER"),
        _L2D([0],[0], color=DKGREEN, marker="s", linewidth=1.4, label="Unconstr. – JSER"),
        _L2D([0],[0], color=CORAL,   linestyle="--", linewidth=1.0, alpha=0.7, label="Bloch – Mahal"),
        _L2D([0],[0], color=DKGREEN, linestyle="--", linewidth=1.0, alpha=0.7, label="Unconstr. – Mahal"),
    ]
    ax.legend(handles=leg_h, ncol=2, loc="center right",
              handlelength=1.2, handletextpad=0.35,
              columnspacing=0.5, borderpad=0.3, labelspacing=0.25)
    ygrid(ax); panel_label(ax, "c")

    # ── d: jser + auc_xyz vs λ for both conditions (grouped bars) ─────────────
    ax      = axes[1, 1]
    lam_list = [0.0, 0.5, 1.0, 2.0]
    n_lam    = len(lam_list)
    x_lam    = np.arange(n_lam)
    bw_d     = 0.20
    bar_specs = [
        ("bloch_mnist",         "jser_xyz_5fpr", CORAL,     "",    "Bloch – JSER"),
        ("bloch_mnist",         "auc_xyz",        "#F4A582", "//",  "Bloch – AUC"),
        ("unconstrained_mnist", "jser_xyz_5fpr", DKGREEN,   "",    "Unconstr. – JSER"),
        ("unconstrained_mnist", "auc_xyz",        "#74C476", "//",  "Unconstr. – AUC"),
    ]
    offs_d = [-1.5*bw_d, -0.5*bw_d, 0.5*bw_d, 1.5*bw_d]
    for (source, col, color, hatch, lbl), off in zip(bar_specs, offs_d):
        d = mn[mn["source"] == source].groupby("lambda_r")[col].mean().reindex(lam_list)
        ax.bar(x_lam + off, d.values, bw_d, color=color, label=lbl,
               hatch=hatch, edgecolor="white", linewidth=0.3, zorder=2)
    ax.set_xticks(x_lam, [str(l) for l in lam_list])
    ax.set_xlabel(r"Adaptive weight $\lambda_r$")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.14)
    ax.legend(ncol=2, loc="upper right",
              handlelength=1.0, handletextpad=0.3,
              columnspacing=0.5, borderpad=0.3, labelspacing=0.25)
    ygrid(ax); panel_label(ax, "d")

    fig.subplots_adjust(left=0.10, right=0.93, bottom=0.11, top=0.97,
                        wspace=0.52, hspace=0.48)
    save(fig, "Figure_2_classical_adaptive_failure_NC_v2")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    setup_style()
    print("Figure 2 …");      figure_2()
    print("Figure 3 …");      figure_3()
    print("Figure 4 …");      figure_4()
    print("Figure 5 …");      figure_5()
    print("Supp Figure 1 …"); supplementary_1()
    print("Supp Figure 2 …"); supplementary_2()
    print("Supp Figure 3 …"); supplementary_3()
    print("Supp Figure 4 …"); supplementary_4()
    print("\nAll done.")
