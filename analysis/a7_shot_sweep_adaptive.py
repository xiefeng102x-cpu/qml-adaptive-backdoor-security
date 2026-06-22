"""
A7  Finite-shot sweep for adaptive attack context  (MNIST L=12)
================================================================
Loads saved attacked-model weight files, computes exact Pauli XYZ features,
adds binomial shot noise, and evaluates detector performance at 128/512/1024
shots.  Calibration uses exact features (mimicking a high-shot reference set);
test samples receive shot noise.

Metrics per (seed, lambda, shot_count):
  auc_mean / auc_std          -- Mahalanobis score AUC (orientation-free sep)
  tpr5_mean / tpr5_std        -- TPR at 5% FPR
  jser_mean / jser_std        -- joint successful evasion rate
                                 = ASR_exact x (1 - TPR_5pct_noisy)
  asr_exact                   -- ASR from training-time JSON (no shot noise)

Usage:
    python a7_shot_sweep_adaptive.py
    python a7_shot_sweep_adaptive.py --n_repeats 50
    python a7_shot_sweep_adaptive.py --lambdas 1.0
"""

from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

ROOT     = Path(__file__).resolve().parents[1]
# ── reviewer-package guard ───────────────────────────────────────────────────
if not (ROOT / 'results').exists():
    import sys as _sys
    print('=' * 68)
    print('This script requires external training data (model weights, raw')
    print('dataset samples) NOT included in the reviewer package.')
    print('Pre-computed results are available in the Zenodo deposit (https://doi.org/10.5281/zenodo.20700154).')
    print()
    print('To verify all key manuscript numbers, run from the package root:')
    print('    python reproduce_key_numbers.py')
    print()
    print('Download training artefacts from the Zenodo deposit or regenerate via training/ scripts.')
    print('=' * 68)
    _sys.exit(0)
# ────────────────────────────────────────────────────────────────────────────

CODE_DIR = Path(__file__).parent
sys.path.insert(0, str(CODE_DIR))

from train_adaptive_qnn_fast import (
    xyz_feats, z_feats, load_config,
    FixedClassifier,
)

WEIGHTS_DIR = ROOT / "results/training_time/new_version"
JSON_DIR    = ROOT / "results/training_time/new_version"
OUT_DIR     = ROOT / "results/shot_sweep_adaptive"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS   = [42, 43, 44, 45, 46]
LAMBDAS = [0.0, 0.5, 1.0, 2.0]
SHOTS   = [128, 512, 1024, 2048, 4096]   # None = exact (included automatically)
N_QUBITS, N_LAYERS = 5, 12
PAIR, PR = "t7_vs_t0", 0.1


# ── shot noise simulation ─────────────────────────────────────────────────────

def add_shot_noise(exact: np.ndarray, n_shots: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate binomial shot noise on Pauli expectation values in [-1, 1]."""
    p = (1.0 + exact) / 2.0          # probability of +1 outcome
    p = np.clip(p, 0.0, 1.0)
    counts = rng.binomial(n_shots, p)
    return 2.0 * counts / n_shots - 1.0


# ── detector evaluation ───────────────────────────────────────────────────────

def eval_detector(xyz_clean: np.ndarray,
                  xyz_poison: np.ndarray,
                  mu: np.ndarray,
                  prec: np.ndarray,
                  fpr_target: float = 0.05) -> dict:
    """Compute AUC, TPR@fpr_target, JSER given exact/noisy feature arrays.

    Calibration (mu, prec) is passed in from exact features.
    Returns orientation-free separation = max(AUC, 1-AUC).
    """
    def mahal_score(X):
        diff = X - mu
        return np.sqrt(np.einsum("ni,ij,nj->n", diff, prec, diff))

    scores_clean  = mahal_score(xyz_clean)
    scores_poison = mahal_score(xyz_poison)
    labels = np.array([0] * len(scores_clean) + [1] * len(scores_poison))
    scores = np.concatenate([scores_clean, scores_poison])

    auc = roc_auc_score(labels, scores)
    sep = max(auc, 1.0 - auc)

    # Threshold at fpr_target on clean scores
    threshold = np.quantile(scores_clean, 1.0 - fpr_target)
    tpr = float(np.mean(scores_poison > threshold))

    return {"auc": sep, "tpr5": tpr}


def compute_asr(weights: np.ndarray,
                clf: FixedClassifier,
                x_poison: np.ndarray) -> float:
    """ASR using exact Z features (no shot noise)."""
    z_p = z_feats(x_poison, weights, N_LAYERS, N_QUBITS)
    logits = clf.forward(z_p)
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == 1))   # target class = 1


# ── main sweep ────────────────────────────────────────────────────────────────

def run_sweep(seeds, lambdas, shot_counts, n_repeats):
    rng = np.random.default_rng(42)
    rows = []

    for seed in seeds:
        _, clf, x_ct, x_cnt, x_p = load_config(seed, N_LAYERS, PAIR, PR)

        for lam in lambdas:
            tag = f"seed={seed}_L{N_LAYERS}_lr{lam}"
            wfile = WEIGHTS_DIR / f"weights_{tag}.npy"
            jfile = JSON_DIR    / f"result_{tag}.json"

            if not wfile.exists():
                print(f"  SKIP {tag}: weight file not found")
                continue

            weights = np.load(wfile)
            asr_exact = None
            if jfile.exists():
                jdata = json.loads(jfile.read_text())
                ep_data = jdata.get("final") or jdata.get("latest", {})
                asr_exact = ep_data.get("asr")

            t0 = time.time()
            print(f"\n  {tag}  (asr_exact={asr_exact})")

            # Exact features (used for calibration throughout)
            xyz_ct_exact  = xyz_feats(x_ct,  weights, N_LAYERS, N_QUBITS)  # clean target
            xyz_p_exact   = xyz_feats(x_p,   weights, N_LAYERS, N_QUBITS)  # poison

            # Calibrate detector on exact clean-target features
            # Negative class = clean target (same class as attack target)
            # Positive class = poison — matches detection_auc() in training code
            lw    = LedoitWolf().fit(xyz_ct_exact)
            mu    = lw.location_
            prec  = lw.get_precision()

            # Exact baseline
            res_exact = eval_detector(xyz_ct_exact, xyz_p_exact, mu, prec)
            jser_exact = (asr_exact * (1.0 - res_exact["tpr5"])) if asr_exact else None
            print(f"    exact  auc={res_exact['auc']:.4f}  tpr5={res_exact['tpr5']:.4f}"
                  f"  jser={jser_exact:.4f}" if jser_exact else "")
            rows.append({
                "seed": seed, "lambda_r": lam, "shots": "exact",
                "auc_mean": res_exact["auc"],  "auc_std": 0.0,
                "tpr5_mean": res_exact["tpr5"], "tpr5_std": 0.0,
                "jser_mean": jser_exact or "",  "jser_std": 0.0,
                "asr_exact": asr_exact or "",
            })

            # Shot-noisy sweeps
            for n_shots in shot_counts:
                aucs, tprs, jsers = [], [], []
                for _ in range(n_repeats):
                    xyz_ct_n = add_shot_noise(xyz_ct_exact, n_shots, rng)
                    xyz_p_n  = add_shot_noise(xyz_p_exact,  n_shots, rng)
                    res = eval_detector(xyz_ct_n, xyz_p_n, mu, prec)
                    aucs.append(res["auc"])
                    tprs.append(res["tpr5"])
                    jser = (asr_exact * (1.0 - res["tpr5"])) if asr_exact else None
                    if jser is not None:
                        jsers.append(jser)

                auc_m, auc_s  = np.mean(aucs), np.std(aucs)
                tpr_m, tpr_s  = np.mean(tprs), np.std(tprs)
                jser_m = np.mean(jsers) if jsers else ""
                jser_s = np.std(jsers)  if jsers else 0.0
                print(f"    {n_shots:>4} shots  auc={auc_m:.4f}±{auc_s:.4f}"
                      f"  tpr5={tpr_m:.4f}±{tpr_s:.4f}"
                      + (f"  jser={jser_m:.4f}±{jser_s:.4f}" if jsers else ""))
                rows.append({
                    "seed": seed, "lambda_r": lam, "shots": n_shots,
                    "auc_mean": round(auc_m, 6),  "auc_std": round(auc_s, 6),
                    "tpr5_mean": round(tpr_m, 6), "tpr5_std": round(tpr_s, 6),
                    "jser_mean": round(jser_m, 6) if jsers else "",
                    "jser_std":  round(jser_s, 6) if jsers else 0.0,
                    "asr_exact": asr_exact or "",
                })

            print(f"    elapsed {time.time()-t0:.1f}s")

    return rows


def write_csv(rows, path):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {path}")


def print_summary(rows):
    import collections
    by_lam = collections.defaultdict(list)
    for r in rows:
        if r["shots"] != "exact":
            by_lam[r["lambda_r"]].append(r)

    print("\n" + "="*72)
    print(f"{'λ':>4}  {'shots':>5}  {'AUC(mean±std)':>16}  "
          f"{'TPR5(mean±std)':>16}  {'JSER(mean±std)':>16}")
    print("-"*72)
    for lam in sorted(by_lam.keys()):
        # group by shots
        by_shots = collections.defaultdict(list)
        for r in by_lam[lam]:
            by_shots[r["shots"]].append(r)
        for shots in sorted(by_shots.keys(), key=lambda x: int(x)):
            aucs  = [r["auc_mean"]  for r in by_shots[shots]]
            tprs  = [r["tpr5_mean"] for r in by_shots[shots]]
            jsers = [r["jser_mean"] for r in by_shots[shots] if r["jser_mean"] != ""]
            print(f"{lam:>4}  {shots:>5}  "
                  f"{np.mean(aucs):.4f}±{np.std(aucs):.4f}  "
                  f"{np.mean(tprs):.4f}±{np.std(tprs):.4f}  "
                  + (f"{np.mean(jsers):.4f}±{np.std(jsers):.4f}" if jsers else "  n/a"))


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds",     nargs="+", type=int,   default=SEEDS)
    ap.add_argument("--lambdas",   nargs="+", type=float, default=LAMBDAS)
    ap.add_argument("--n_repeats", type=int,  default=30)
    args = ap.parse_args()

    print(f"Seeds: {args.seeds}  Lambdas: {args.lambdas}  "
          f"Shots: {SHOTS}  Repeats: {args.n_repeats}")

    rows = run_sweep(args.seeds, args.lambdas, SHOTS, args.n_repeats)
    out  = OUT_DIR / "adaptive_shot_sweep_summary.csv"
    write_csv(rows, out)
    print_summary(rows)


if __name__ == "__main__":
    main()

