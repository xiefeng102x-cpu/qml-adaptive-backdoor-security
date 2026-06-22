"""Recompute core manuscript numbers from the reviewer package CSV/JSON files.

Intentionally lightweight: does not rerun QNN training or regenerate features.
Reads the processed CSV files and IBM hardware JSON files in data/ and
writes a table of key manuscript numbers to reproduction_outputs/key_numbers_recomputed.csv.

Run from the repository root:`n    python reproduce_key_numbers.py

Dependencies: numpy, pandas, scipy (no additional packages required)

SD convention note
------------------
ALL Table 1 values (including indep_cal TPR/JSER) use POPULATION std (ddof=0).
The within-ladder JSER comparison in the text (0.267±0.177) is an exception:
it uses the true_jser column with SAMPLE std (ddof=1, N=5).

Float-rounding note
-------------------
The within-ladder XYZ AUC (C16, SI Table S3) has mean 0.9515 across 5 seeds.
Python float64 stores 0.9515 as 0.9514999..., so round(0.9515, 3) = 0.951 not
0.952; the SI value 0.952 uses conventional arithmetic rounding (0.9515 → 0.952).
This applies to C16 only. Table 1 Panel A AUC = 0.944 is from indep_cal_mnist_metrics.csv
auc_all column (independent-calibration protocol), not ladder_results.csv.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parent
OUT  = ROOT / "reproduction_outputs"


def add(rows, claim, value, source, note=""):
    rows.append({"claim": claim, "value": value,
                 "source": str(Path(source).relative_to(ROOT)), "note": note})


def sep_series(auc_series):
    return auc_series.apply(lambda x: max(x, 1 - x))


def main():
    rows = []

    # ── Table 1 Panel A: QML MNIST L=12  AUC / Sep / ASR (indep-cal protocol) ──
    # Source: indep_cal_mnist_metrics.csv, lambda_r=1.0, seeds 42-46
    # auc_all = threshold-free AUC using R-fitted LW detector on all samples.
    # SD convention: POPULATION std (ddof=0) — matches manuscript Table 1.
    # Note: within-ladder AUC (ladder_results.csv, condition=adaptive, XYZ) = 0.9515±0.024
    #       is kept separately for the ladder-comparison figure (Supplementary Table S3).
    lad_path = ROOT / "data/mechanism_interventions/pauli_ladder/ladder_results.csv"
    lad = pd.read_csv(lad_path)
    mnist_xyz = lad[(lad["dataset"] == "MNIST") &
                    (lad["layer"] == 12) &
                    (lad["condition"] == "adaptive") &
                    (lad["feature_set"] == "XYZ")].copy()

    # Panel A AUC from indep_cal (unified protocol)
    mnist_indep_path = ROOT / "data/core_metrics/indep_cal_mnist_metrics.csv"
    mnist_indep = pd.read_csv(mnist_indep_path)
    m1 = mnist_indep[mnist_indep["lambda_r"] == 1.0]

    auc_mean = round(m1["auc_all"].mean(), 3)
    auc_std  = round(m1["auc_all"].std(ddof=0), 3)   # population std
    sep_mean = round(sep_series(m1["auc_all"]).mean(), 3)
    asr_mean = round(m1["asr"].mean(), 3)

    add(rows, "QML MNIST L12 λr=1 AUC (mean±pop_sd; indep-cal unified protocol)",
        f"{auc_mean}±{auc_std}", mnist_indep_path,
        "Table 1 col 5; indep-cal auc_all; population std; 0.944±0.030")
    add(rows, "QML MNIST L12 λr=1 Sep (mean±pop_sd)",
        f"{sep_mean}±{auc_std}", mnist_indep_path,
        "Table 1 col 6; sep=max(auc_all,1-auc_all); all seeds>0.5 so sep=auc_all")
    add(rows, "QML MNIST L12 λr=1 ASR (mean, no sd in manuscript)",
        asr_mean, mnist_indep_path,
        "Table 1 col 3; mean over seeds [1.0,1.0,1.0,0.95,1.0]")

    add(rows, "QML MNIST L12 λr=1 CA (mean, no sd in manuscript)",
        round(m1["ca"].mean(), 3), mnist_indep_path,
        "Table 1 col 4; 0.6175→0.618; from indep-cal split")

    # ── Table 1 Panel B: QML MNIST L=12  TPR / JSER (indep-cal, pop std) ──────
    # The independent-calibration protocol (R=300/C=120/E=120) gives the
    # primary operational metric in Table 1 Panel B.  Uses POPULATION std
    # (ddof=0) consistent with the rest of Table 1.
    tpr_mean  = round(m1["tpr_5fpr"].mean(), 3)
    tpr_std   = round(m1["tpr_5fpr"].std(ddof=0), 3)   # population std
    # JSER mean at 4dp: 0.3385 → manuscript rounds to 0.339; Python float gives 0.338
    jser_mean = round(m1["jser"].mean(), 4)
    jser_std  = round(m1["jser"].std(ddof=0), 3)       # population std
    fpr_mean  = round(m1["fpr_e"].mean(), 3)
    fpr_std   = round(m1["fpr_e"].std(ddof=0), 3)      # population std (manuscript ≈3.9%)

    add(rows, "QML MNIST L12 λr=1 TPR@5%FPR (mean±pop_sd)",
        f"{tpr_mean}±{tpr_std}", mnist_indep_path,
        "Table 1 Panel B col 7; indep-cal; population std (ddof=0)")
    add(rows, "QML MNIST L12 λr=1 JSER (mean±pop_sd, 4dp; manuscript rounds to 0.339±0.206)",
        f"{jser_mean}±{jser_std}", mnist_indep_path,
        "Table 1 Panel B col 8; primary operational metric; population std (ddof=0)")
    add(rows, "QML MNIST L12 λr=1 empirical FPR (mean±pop_sd)",
        f"{fpr_mean}±{fpr_std}", mnist_indep_path,
        "Methods: FPR bias note; nominal 5%; empirical ≈5.3%±3.9% (population std)")

    # 95% CI for JSER uses sample std / sqrt(N) as SEM (ddof=1; manuscript §3.4 states (0.05, 0.63))
    jser_vals     = m1["jser"].values
    jser_samp_sem = np.std(jser_vals, ddof=1) / np.sqrt(len(jser_vals))
    ci = st.t.interval(0.95, df=len(jser_vals) - 1,
                       loc=np.mean(jser_vals), scale=jser_samp_sem)
    add(rows, "MNIST JSER 95% CI (Student-t, df=4, sample-std SEM ddof=1)",
        f"({round(ci[0], 2)}, {round(ci[1], 2)})", mnist_indep_path,
        "Results §3.4 text claim (0.05, 0.63); sample SD ddof=1; ASSERTION: expect (0.05, 0.63)")

    # ── Within-ladder MNIST JSER (text / Fig 4) ───────────────────────────────
    # Uses the true_jser column (unconditional JSER) with SAMPLE std (ddof=1).
    # This value (0.267±0.177) differs from Table 1 Panel B indep-cal (0.339±0.206)
    # because they use different calibration references.
    lad_jser_mean = round(mnist_xyz["true_jser"].mean(), 3)
    lad_jser_std  = round(mnist_xyz["true_jser"].std(ddof=1), 3)  # sample std
    add(rows, "MNIST within-ladder JSER (mean±samp_sd, text/Fig 4; uses true_jser col)",
        f"{lad_jser_mean}±{lad_jser_std}", lad_path,
        "Within-ladder calibration; NOT Table 1; Z→XYZ comparison in §3.5")

    # ── Table 1 Panel A: QML BloodMNIST L=8  AUC / ASR / CA (indep-cal unified) ─
    # Source: indep_cal_bloodmnist_metrics.csv, lambda_r=1.0, seeds 42-46
    # auc_all = threshold-free AUC using R-fitted LW detector on all samples.
    # SD convention: POPULATION std (ddof=0) — matches manuscript Table 1.
    # Note: a1b protocol AUC (a1b_bloodmnist_metrics.csv) = 0.993±0.005 (shared-fit).
    blood_path = ROOT / "data/core_metrics/indep_cal_bloodmnist_metrics.csv"
    blood = pd.read_csv(blood_path)
    b1 = blood[blood["lambda_r"] == 1.0]

    add(rows, "QML BloodMNIST L8 λr=1 AUC (mean±pop_sd; indep-cal unified protocol)",
        f"{round(b1['auc_all'].mean(), 3)}±{round(b1['auc_all'].std(ddof=0), 3)}",
        blood_path, "Table 1 Panel A col 5; indep-cal auc_all; population std; 0.999±0.000")
    add(rows, "QML BloodMNIST L8 λr=1 Sep (mean±pop_sd)",
        f"{round(sep_series(b1['auc_all']).mean(), 3)}±{round(b1['auc_all'].std(ddof=0), 3)}",
        blood_path, "Table 1 col 6; sep=auc_all since all seeds>0.5")
    add(rows, "QML BloodMNIST L8 λr=1 ASR (mean±pop_sd)",
        f"{round(b1['asr'].mean(), 3)}±{round(b1['asr'].std(ddof=0), 3)}",
        blood_path, "Table 1 col 3")
    add(rows, "QML BloodMNIST L8 λr=1 CA (mean±pop_sd)",
        f"{b1['ca'].mean():.3f}±{b1['ca'].std(ddof=0):.3f}",
        blood_path, "Table 1 col 4; max CA across all lambda_r ~0.74 (at lambda_r=0)")

    # ── Table 1 Panel B: QML BloodMNIST L=8  TPR / JSER (same indep-cal source) ─
    add(rows, "QML BloodMNIST L8 λr=1 TPR@5%FPR (mean±pop_sd, §S)",
        f"{round(b1['tpr_5fpr'].mean(), 3)}±{round(b1['tpr_5fpr'].std(ddof=0), 3)}",
        blood_path, "Table 1 Panel B §S; indep-cal; population std; CA-constrained")
    add(rows, "QML BloodMNIST L8 λr=1 JSER (mean±pop_sd, §S)",
        f"{round(b1['jser'].mean(), 3)}±{round(b1['jser'].std(ddof=0), 3)}",
        blood_path, "Table 1 Panel B §S; reflects CA-constraint (CA<0.9), not detection alone")
    add(rows, "QML BloodMNIST L8 λr=1 empirical FPR (mean±pop_sd)",
        f"{round(b1['fpr_e'].mean(), 3)}±{round(b1['fpr_e'].std(ddof=0), 3)}",
        blood_path, "Methods: FPR bias note; BloodMNIST ≈8.2%±5.5% (population std)")

    # ── Table 1: Classical controls at lambda_r=1 ────────────────────────────
    cls_path = ROOT / "data/classical_controls/training_time_summary_classical.csv"
    cls = pd.read_csv(cls_path)

    for system, label in [("bloch_mnist",         "Classical Bloch MNIST L12"),
                           ("bloch_blood",         "Classical Bloch BloodMNIST L8"),
                           ("unconstrained_mnist", "Classical Unconst MNIST L12"),
                           ("unconstrained_blood", "Classical Unconst BloodMNIST L8")]:
        sub = cls[(cls["source"] == system) & (cls["lambda_r"] == 1.0)]
        sep_val  = round(sep_series(sub["auc_xyz"]).mean(), 3)
        asr_val  = round(sub["asr"].mean(), 3)
        ca_val   = round(sub["ca"].mean(), 3)
        jser_val = round(sub["jser_xyz_5fpr"].mean(), 3)
        add(rows, f"{label} λr=1 Sep",  sep_val,  cls_path, "Table 1 Panel A")
        add(rows, f"{label} λr=1 ASR",  asr_val,  cls_path, "Table 1")
        add(rows, f"{label} λr=1 CA",   ca_val,   cls_path, "Table 1")
        add(rows, f"{label} λr=1 JSER", jser_val, cls_path, "Table 1 Panel B; all seeds exact")

    # ── Capacity-matched ablation (Classical Bloch small, 111 params) ────────
    abl_path = ROOT / "data/robustness_checks/ablation_summary.csv"
    abl = pd.read_csv(abl_path)
    a1 = abl[abl["lambda_r"] == 1.0].iloc[0]
    add(rows, "Capacity ablation (111-param) λr=1 CA (mean)",
        round(a1["ca_mean"], 3), abl_path, "Table 2; Discussion §capacity")
    add(rows, "Capacity ablation (111-param) λr=1 ASR (mean)",
        round(a1["asr_mean"], 3), abl_path, "Table 2; Discussion §capacity")
    add(rows, "Capacity ablation (111-param) λr=1 JSER (mean)",
        round(a1["jser_mean"], 3), abl_path, "Table 2; Discussion §capacity")
    add(rows, "Capacity ablation (111-param) λr=1 Sep (mean±sd)",
        f"{round(a1['sep_mean'], 3)}±{round(a1['sep_sd'], 3)}", abl_path,
        "Discussion: classically constrained classifier still evades fully")

    # ── Null-space intervention ───────────────────────────────────────────────
    ns_path = ROOT / "data/mechanism_interventions/nullspace/e2_summary.csv"
    ns = pd.read_csv(ns_path)
    ad_ns = ns[ns["condition"] == "adaptive"]
    add(rows, "Null-space r_dim (structural kernel fraction, adaptive seeds)",
        round(ad_ns["r_dim"].iloc[0], 4), ns_path,
        "Results §3.5; 17/32 = 0.5312; constant across seeds; SVD of circuit Jacobian")
    add(rows, "Null-space r_hidden max across adaptive seeds",
        float(ad_ns["r_hidden"].max()), ns_path,
        "Stored as 0.0 (pandas round(4) truncation); true value ≈4×10^-12 from e2_raw.csv")
    add(rows, "Null-space evades_proj (mean across adaptive seeds)",
        round(float(ad_ns["evades_proj"].mean()), 3), ns_path,
        "0/600 samples show logit change; acceptance fraction ≈ baseline (1−TPR); NOT partial evasion — near-zero perturbation, acceptance reflects detector baseline")

    # ── Pauli ladder: MNIST XYZ adaptive AUC (ladder_summary for SI Table S3) ─
    lad_sum_path = ROOT / "data/mechanism_interventions/pauli_ladder/ladder_summary.csv"
    lad_sum_raw = pd.read_csv(lad_sum_path, skiprows=3, header=None)
    lad_sum_raw.columns = ["dataset", "layer", "condition", "feature_set",
                            "auc_mean", "auc_std", "sep_mean", "sep_std",
                            "tpr5_mean", "tpr5_std", "jser5_mean", "jser5_std",
                            "true_jser_mean", "true_jser_std",
                            "mahal_mean", "mahal_std"]
    lad_sum_raw["layer"] = pd.to_numeric(lad_sum_raw["layer"])

    xyz_adapt = lad_sum_raw[(lad_sum_raw["dataset"] == "MNIST") &
                             (lad_sum_raw["layer"] == 12) &
                             (lad_sum_raw["condition"] == "adaptive") &
                             (lad_sum_raw["feature_set"] == "XYZ")].iloc[0]
    z_adapt   = lad_sum_raw[(lad_sum_raw["dataset"] == "MNIST") &
                             (lad_sum_raw["layer"] == 12) &
                             (lad_sum_raw["condition"] == "adaptive") &
                             (lad_sum_raw["feature_set"] == "Z")].iloc[0]

    add(rows, "Ladder MNIST XYZ adaptive AUC (mean±sd, SI Table S3)",
        f"{round(float(xyz_adapt['auc_mean']), 4)}±{round(float(xyz_adapt['auc_std']), 4)}",
        lad_sum_path, "Results §3.5 / SI S3; within-ladder calibration")
    add(rows, "Ladder MNIST Z-only adaptive AUC (mean±sd, SI Table S3)",
        f"{round(float(z_adapt['auc_mean']), 4)}±{round(float(z_adapt['auc_std']), 4)}",
        lad_sum_path, "Results §3.5; XYZ gain over Z confirms multi-axis information")

    # ── Hardware AUC (IBM Kingston, seed=42, 4096 shots) ─────────────────────
    # Data from IBM Kingston job runs on 2026-05-27, parsed from JSON.
    # These are the ONLY hardware results in the manuscript; TianYan176 runs
    # (1024 shots, June 2026) are excluded from manuscript claims.
    hw_mnist_json = ROOT / "data/hardware/ibm_v3_MNIST_5q_layer12_amplitude_ibm_kingston.json"
    hw_blood_json = ROOT / "data/hardware/ibm_blood_layer8_amplitude_ibm_kingston.json"

    with open(hw_mnist_json) as f:
        hw_m = json.load(f)
    with open(hw_blood_json) as f:
        hw_b = json.load(f)

    mnist_hw_auc   = hw_m["hardware"]["XYZ"]["AUC"]           # 0.7731
    mnist_hw_fcorr = hw_m["hardware_vs_simulator"]["feature_correlation"]  # 0.2473
    blood_hw_auc   = hw_b["hardware"]["XYZ"]["AUC"]           # 0.9697
    blood_hw_fcorr = hw_b["hardware_vs_simulator"]["feature_correlation"]  # 0.523

    add(rows, "Hardware MNIST AUC (IBM Kingston, 4096 shots, seed=42, XYZ)",
        f"{mnist_hw_auc:.3f}", hw_mnist_json,
        "Results §3.6 / Fig 5d; hardware.XYZ.AUC; job_id d8b9auh59p8c73bm6mtg; 2026-05-27")
    add(rows, "Hardware BloodMNIST AUC (IBM Kingston, 4096 shots, seed=42, XYZ)",
        f"{blood_hw_auc:.3f}", hw_blood_json,
        "Results §3.6 / Fig 5d; hardware.XYZ.AUC; job_id d8be0btmdd1s73b9e2bg; 2026-05-27")
    add(rows, "Hardware MNIST XYZ feature correlation sim↔hw",
        round(mnist_hw_fcorr, 4), hw_mnist_json,
        "hardware_vs_simulator.feature_correlation; lower than BloodMNIST due to deeper circuit")
    add(rows, "Hardware BloodMNIST XYZ feature correlation sim↔hw",
        round(blood_hw_fcorr, 3), hw_blood_json,
        "hardware_vs_simulator.feature_correlation; ~27% theoretical fidelity")

    # ── Shot sweep: MNIST at lambda_r=1 ──────────────────────────────────────
    shot_path = ROOT / "data/robustness_checks/adaptive_shot_sweep_summary.csv"
    shot = pd.read_csv(shot_path)
    shot_num = shot[(shot["lambda_r"] == 1.0) & (shot["shots"] != "exact")].copy()
    shot_num["shots"] = shot_num["shots"].astype(int)
    shot_agg = shot_num.groupby("shots")[["auc_mean", "jser_mean"]].mean()
    add(rows, "Shot sweep MNIST λr=1 AUC at 128 shots (mean across seeds)",
        round(float(shot_agg.loc[128, "auc_mean"]), 3), shot_path,
        "Results §3.6; Fig 5a; 0.899±0.040 in text")
    add(rows, "Shot sweep MNIST λr=1 AUC at 4096 shots (mean across seeds)",
        round(float(shot_agg.loc[4096, "auc_mean"]), 3), shot_path,
        "Results §3.6; Fig 5a; plateau confirms shot-efficient detection")
    add(rows, "Shot sweep MNIST λr=1 JSER at 128 shots (mean across seeds)",
        round(float(shot_agg.loc[128, "jser_mean"]), 3), shot_path,
        "Results §3.6; Fig 5a")
    add(rows, "Shot sweep MNIST λr=1 JSER at 4096 shots (mean across seeds)",
        round(float(shot_agg.loc[4096, "jser_mean"]), 3), shot_path,
        "Results §3.6; Fig 5a")

    # ── C29: Depth scan — layer-selection justification ──────────────────────
    depth_path = ROOT / "data/depth_scan/best_layer_by_mean_auc_summary.csv"
    depth = pd.read_csv(depth_path)
    q29 = depth[(depth["slice"] == "pr_0.1") &
                (depth["protocol"] == "Phase0_diagnostic_full_fit") &
                (depth["axis"] == "XYZ")]
    mnist_peak = q29[q29["dataset"] == "MNIST_5q"].iloc[0]
    blood_peak = q29[q29["dataset"] == "Blood_8q_eps0.8"].iloc[0]
    add(rows, "Depth scan: MNIST XYZ AUC peak layer (pr=0.1, Phase0)",
        int(mnist_peak["best_layer_by_mean_auc"]), depth_path,
        "C29; Methods §layer choice; expects L=12; best_mean_auc≈0.950")
    add(rows, "Depth scan: MNIST XYZ AUC at peak layer",
        round(float(mnist_peak["best_mean_auc"]), 3), depth_path,
        "C29; Methods §layer choice; mean over seeds at L=12")
    add(rows, "Depth scan: BloodMNIST XYZ AUC peak layer (pr=0.1, Phase0)",
        int(blood_peak["best_layer_by_mean_auc"]), depth_path,
        "C29; Methods §layer choice; expects L=8; best_mean_auc≈0.941")
    add(rows, "Depth scan: BloodMNIST XYZ AUC at peak layer",
        round(float(blood_peak["best_mean_auc"]), 3), depth_path,
        "C29; Methods §layer choice; mean over seeds at L=8")

    # ── C30: Directional entropy H_dir ────────────────────────────────────────
    hdir_path = ROOT / "data/mechanism_interventions/hdir/hdir_by_dataset.csv"
    hdir = pd.read_csv(hdir_path)
    mnist_hdir = hdir[hdir["dataset"] == "MNIST_5q"].iloc[0]
    blood_hdir = hdir[hdir["dataset"] == "Blood_8q_eps0.8"].iloc[0]
    add(rows, "H_dir (directional entropy): MNIST",
        round(float(mnist_hdir["directional_entropy_norm"]), 3), hdir_path,
        "C30; Results §3.5 / Fig 4 Hdir panel; expects 0.981 (multi-axis)")
    add(rows, "H_dir (directional entropy): BloodMNIST",
        round(float(blood_hdir["directional_entropy_norm"]), 3), hdir_path,
        "C30; Results §3.5 / Fig 4 Hdir panel; expects 0.482 (Z-dominant)")

    # ── C31: Calibration-split stability (SI Table S12) ──────────────────────
    rsplit_path = ROOT / "data/robustness_checks/repeated_splits_summary.csv"
    rs = pd.read_csv(rsplit_path)
    N_SPLITS = 50
    # Pooled within-seed variance (mean of per-seed population variances)
    within_var = float((rs["jser_std"] ** 2).mean())
    # Between-seed variance: grand mean across seed means, then SS_between / N_total
    grand_mean = float(rs["jser_mean"].mean())
    ss_between = float(N_SPLITS * ((rs["jser_mean"] - grand_mean) ** 2).sum())
    ss_within  = float(N_SPLITS * (rs["jser_std"] ** 2).sum())
    ss_total   = ss_between + ss_within
    between_pct = round(100 * ss_between / ss_total)
    within_pct  = round(100 * ss_within  / ss_total)
    pooled_jser = round(float(rs["jser_mean"].mean()), 3)
    add(rows, "Repeated splits: pooled JSER mean (250 evals, lambda_r=1, MNIST)",
        pooled_jser, rsplit_path,
        "C31; SI Table S12 / Note 9; 50 splits x 5 seeds; expects ~0.322")
    add(rows, "Repeated splits: between-seed variance fraction (%)",
        f"{between_pct}%", rsplit_path,
        "C31; SI Note 9; expects ~76%; dominant source is model heterogeneity")
    add(rows, "Repeated splits: within-seed variance fraction (%)",
        f"{within_pct}%", rsplit_path,
        "C31; SI Note 9; expects ~24%; calibration sampling contribution")
    add(rows, "Repeated splits: within-seed JSER std range",
        f"{float(rs['jser_std'].min()):.3f}--{float(rs['jser_std'].max()):.3f}", rsplit_path,
        "C31; SI Note 9; expects 0.043--0.122")
    assert abs(pooled_jser - 0.322) < 0.005, \
        f"Repeated splits pooled JSER {pooled_jser} != expected 0.322"
    assert between_pct == 76, \
        f"Between-seed variance fraction {between_pct}% != expected 76%"

    # ── Executable assertions for P0 claims ──────────────────────────────────
    assert abs(auc_mean - 0.944) <= 0.005, \
        f"QML MNIST AUC {auc_mean} deviates from manuscript 0.944 by >{0.005}"
    assert round(ci[0], 2) == 0.05, \
        f"JSER 95% CI lower {ci[0]:.4f} != manuscript 0.05"
    assert round(ci[1], 2) == 0.63, \
        f"JSER 95% CI upper {ci[1]:.4f} != manuscript 0.63"
    assert abs(round(float(ad_ns["r_dim"].iloc[0]), 4) - 0.5312) < 0.0001, \
        f"Null-space r_dim {ad_ns['r_dim'].iloc[0]:.6f} != expected 0.5312"
    assert int(mnist_peak["best_layer_by_mean_auc"]) == 12, \
        f"MNIST depth-scan peak layer {int(mnist_peak['best_layer_by_mean_auc'])} != 12"
    assert int(blood_peak["best_layer_by_mean_auc"]) == 8, \
        f"BloodMNIST depth-scan peak layer {int(blood_peak['best_layer_by_mean_auc'])} != 8"
    assert abs(round(float(mnist_hdir["directional_entropy_norm"]), 3) - 0.981) < 0.002, \
        f"MNIST H_dir {float(mnist_hdir['directional_entropy_norm']):.4f} != 0.981"
    assert abs(round(float(blood_hdir["directional_entropy_norm"]), 3) - 0.482) < 0.002, \
        f"BloodMNIST H_dir {float(blood_hdir['directional_entropy_norm']):.4f} != 0.482"

    # ── C32: JSER_dc  /  C33: already_target  /  C34: TPR_dc ─────────────────
    # Source: indep_cal_mnist_jser_dc_metrics.csv, lambda_r=1.0, seeds 42-46
    # Prop-1-aligned metrics added 2026-06-15 (p05_jser_dc.py)
    dc_path = ROOT / "data/core_metrics/indep_cal_mnist_jser_dc_metrics.csv"
    dc_df   = pd.read_csv(dc_path)
    dc1     = dc_df[dc_df["lambda_r"] == 1.0]
    dc0     = dc_df[dc_df["lambda_r"] == 0.0]

    jser_dc_mean  = round(float(dc1["jser_dc"].mean()), 3)
    jser_dc_std   = round(float(dc1["jser_dc"].std(ddof=0)), 3)
    at1_mean      = round(float(dc1["already_target"].mean()), 3)
    at1_std       = round(float(dc1["already_target"].std(ddof=0)), 3)
    at0_mean      = round(float(dc0["already_target"].mean()), 3)
    at0_std       = round(float(dc0["already_target"].std(ddof=0)), 3)
    tpr_dc_vals   = dc1["dc_tpr"].dropna()
    tpr_dc_mean   = round(float(tpr_dc_vals.mean()), 3)
    tpr_dc_std    = round(float(tpr_dc_vals.std(ddof=0)), 3)

    add(rows, "C32: MNIST JSER_dc lambda_r=1 (mean+/-pop_sd; Prop-1-aligned)",
        f"{jser_dc_mean}+/-{jser_dc_std}", dc_path,
        "Table 1 footnote / Results s3.4 / SI Note 12; expects 0.130+/-0.093")
    add(rows, "C33: MNIST already_target lambda_r=0 (mean+/-pop_sd)",
        f"{at0_mean}+/-{at0_std}", dc_path,
        "Results s3.4 / SI Table S14; expects 0.070+/-0.036")
    add(rows, "C33: MNIST already_target lambda_r=1 (mean+/-pop_sd)",
        f"{at1_mean}+/-{at1_std}", dc_path,
        "Results s3.4 / SI Table S14; expects 0.563+/-0.171")
    add(rows, "C34: MNIST TPR_dc lambda_r=1 (mean+/-pop_sd; genuine decision-change detection rate)",
        f"{tpr_dc_mean}+/-{tpr_dc_std}", dc_path,
        "SI Table S13 / main text ~line 717; expects 0.684+/-0.181")

    assert abs(jser_dc_mean - 0.130) < 0.005, \
        f"C32 JSER_dc {jser_dc_mean} != expected 0.130"
    assert abs(at1_mean - 0.563) < 0.005, \
        f"C33 already_target lambda_r=1: {at1_mean} != expected 0.563"
    assert abs(at0_mean - 0.070) < 0.005, \
        f"C33 already_target lambda_r=0: {at0_mean} != expected 0.070"
    assert abs(tpr_dc_mean - 0.684) < 0.005, \
        f"C34 TPR_dc {tpr_dc_mean} != expected 0.684"

    print("All assertions passed.")

    # ── Write output ──────────────────────────────────────────────────────────
    OUT.mkdir(exist_ok=True)
    out_path = OUT / "key_numbers_recomputed.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path.relative_to(ROOT)}")
    print()
    # Use ASCII-safe output to avoid terminal encoding issues on non-UTF-8 systems
    summary = df[["claim", "value"]].to_string(index=False, max_colwidth=70)
    print(summary.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()

