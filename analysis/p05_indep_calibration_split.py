"""
p05_indep_calibration_split.py
===============================
Re-evaluates TPR@5%FPR and JSER using properly split R / C / E sets.

Current protocol (R=C):
  The same ~60 clean target samples are used for BOTH LedoitWolf fitting
  and 95th-percentile threshold calibration.

Independent-split protocol:
  540 clean target samples split (seed 0, fixed) into:
    R = 300  →  fit LedoitWolf (mu, Sigma)
    C = 120  →  set 5%-FPR threshold (95th percentile of C Mahal scores)
    E = 120  →  verify empirical FPR (should ≈ 5%)
  Poisoned samples (60) → TPR, JSER

Key difference from existing a1b script:
  - R and C are disjoint (proper calibration isolation)
  - LedoitWolf is fit on R=300, not R=60 (better covariance estimate)
  - JSER = ASR × (1 - TPR)  [true joint rate, not conditional]

Datasets:
  MNIST       n=5 qubits, L=12, pair t7_vs_t0, epsilon=0.8
  BloodMNIST  n=8 qubits, L=8,  pair t6_vs_t0, epsilon=0.8

Outputs:
  results/audit/indep_cal_mnist_metrics.csv
  results/audit/indep_cal_bloodmnist_metrics.csv
"""

import sys, json, csv
from pathlib import Path
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
import torch

ROOT = Path(__file__).resolve().parents[1]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fast_circuit import _make_evals

OUT_DIR = ROOT / "results/audit"
OUT_DIR.mkdir(exist_ok=True)

# ── split sizes (must sum to 540) ─────────────────────────────────────────────
R_SIZE    = 300   # LedoitWolf reference
C_SIZE    = 120   # calibration (threshold)
E_SIZE    = 120   # clean evaluation (FPR check)
SPLIT_SEED = 0    # fixed for reproducibility

# ── circuit helpers ───────────────────────────────────────────────────────────
_H_gate  = np.array([[1, 1],[1,-1]], dtype=np.complex128) / np.sqrt(2)
_HS_gate = np.array([[1,-1j],[1,1j]], dtype=np.complex128) / np.sqrt(2)

def _apply_1q(psi, gate, q):
    dim = psi.shape[1]
    step = dim >> (q + 1)
    groups = dim // (2 * step)
    return np.einsum("ij,bkjl->bkil", gate,
                     psi.reshape(-1, groups, 2, step),
                     optimize=True).reshape(-1, dim)

def get_xyz_feats(samples, weights, n_layers, n_qubits):
    """Single-pass XYZ features [N, 3*n_qubits] via basis rotation."""
    z_ev, _, _ = _make_evals(n_qubits)
    norms = np.linalg.norm(samples, axis=1, keepdims=True).clip(min=1e-12)
    psi   = (samples / norms).astype(np.complex128)
    wr    = weights.reshape(n_layers, n_qubits, 2)

    for l in range(n_layers):
        for q in range(n_qubits):
            ct, st = np.cos(wr[l, q, 0] / 2), np.sin(wr[l, q, 0] / 2)
            psi = _apply_1q(psi, np.array([[ct, -st],[st, ct]], dtype=np.complex128), q)
            ep  = np.exp(1j * wr[l, q, 1] / 2)
            psi = _apply_1q(psi, np.array([[1/ep, 0],[0, ep]], dtype=np.complex128), q)
        for q in range(n_qubits):
            dim = psi.shape[1]; n = int(np.log2(dim))
            cb, tb = n-1-q, n-1-((q+1) % n_qubits)
            states = np.arange(dim, dtype=np.int64)
            psi = psi[:, np.where((states >> cb) & 1, states ^ (1 << tb), states)]

    probs = np.abs(psi) ** 2
    Z = (probs @ z_ev.T).astype(np.float64)
    X = np.stack([(np.abs(_apply_1q(psi, _H_gate,  q)) ** 2 @ z_ev[q])
                  for q in range(n_qubits)], axis=1)
    Y = np.stack([(np.abs(_apply_1q(psi, _HS_gate, q)) ** 2 @ z_ev[q])
                  for q in range(n_qubits)], axis=1)
    return np.hstack([X, Y, Z]).astype(np.float64)

# ── calibration-split evaluation ──────────────────────────────────────────────
def mahal_sq(xyz, mu, prec):
    d = xyz - mu
    return np.einsum("ni,ij,nj->n", d, prec, d)

def evaluate_indep(weights, x_clean, x_poison, n_layers, n_qubits, asr):
    """Full R/C/E independent-split evaluation."""
    assert len(x_clean) == R_SIZE + C_SIZE + E_SIZE, \
        f"Expected {R_SIZE+C_SIZE+E_SIZE} clean samples, got {len(x_clean)}"

    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(len(x_clean))
    x_r = x_clean[idx[:R_SIZE]]
    x_c = x_clean[idx[R_SIZE:R_SIZE + C_SIZE]]
    x_e = x_clean[idx[R_SIZE + C_SIZE:]]

    xyz_r = get_xyz_feats(x_r.astype(np.float64), weights, n_layers, n_qubits)
    xyz_c = get_xyz_feats(x_c.astype(np.float64), weights, n_layers, n_qubits)
    xyz_e = get_xyz_feats(x_e.astype(np.float64), weights, n_layers, n_qubits)
    xyz_p = get_xyz_feats(x_poison.astype(np.float64), weights, n_layers, n_qubits)

    lw   = LedoitWolf().fit(xyz_r)
    mu   = lw.location_
    prec = lw.get_precision()

    sc = mahal_sq(xyz_c, mu, prec)  # C scores (for threshold)
    se = mahal_sq(xyz_e, mu, prec)  # E scores (FPR check)
    sp = mahal_sq(xyz_p, mu, prec)  # poisoned scores

    tau = float(np.percentile(sc, 95))   # 5% FPR threshold from C

    tpr   = float(np.mean(sp > tau))
    fpr_e = float(np.mean(se > tau))     # empirical FPR on held-out E
    jser  = float(asr * (1 - tpr))       # true joint rate

    # AUC: E (clean) vs poisoned, threshold-free
    y_true  = np.concatenate([np.zeros(len(xyz_e)), np.ones(len(xyz_p))])
    y_score = np.concatenate([se, sp])
    auc_indep = float(roc_auc_score(y_true, y_score))

    # AUC: all clean (R+C+E) vs poisoned (for comparison with original)
    xyz_all = np.vstack([xyz_r, xyz_c, xyz_e])
    s_all   = mahal_sq(xyz_all, mu, prec)
    y_all   = np.concatenate([np.zeros(len(xyz_all)), np.ones(len(xyz_p))])
    s_cat   = np.concatenate([s_all, sp])
    auc_all = float(roc_auc_score(y_all, s_cat))

    mahal_d = float(np.sqrt(sp).mean())

    return dict(
        auc_indep = round(auc_indep, 4),   # AUC on E vs poisoned (unbiased)
        auc_all   = round(auc_all,   4),   # AUC on all-clean vs poisoned
        tpr_5fpr  = round(tpr,       4),
        fpr_e     = round(fpr_e,     4),   # should ≈ 0.05
        jser      = round(jser,      4),
        mahal_d   = round(mahal_d,   4),
        tau       = round(tau,       4),
    )

# ── dataset configurations ────────────────────────────────────────────────────
DATASETS = {
    "mnist": dict(
        data_dir  = ROOT / "data/Mnist/detail_single_samle",
        wgt_dir   = ROOT / "results/training_time/new_version",
        res_dir   = ROOT / "results/training_time/new_version",
        n_qubits  = 5,
        n_layers  = 12,
        pair      = "t7_vs_t0",
        pr        = "0.1",
        seeds     = [42, 43, 44, 45, 46],
        lambdas   = [0.0, 0.5, 1.0, 2.0],
        out_csv   = OUT_DIR / "indep_cal_mnist_metrics.csv",
    ),
    "bloodmnist": dict(
        data_dir  = ROOT / "data/MedMnist/single_detect_20260411/epsilon_0.8",
        wgt_dir   = ROOT / "results/training_time_bloodmnist",
        res_dir   = ROOT / "results/training_time_bloodmnist",
        n_qubits  = 8,
        n_layers  = 8,
        pair      = "t6_vs_t0",
        pr        = "0.1",
        seeds     = [42, 43, 44, 45, 46],
        lambdas   = [0.0, 0.5, 1.0, 2.0],
        out_csv   = OUT_DIR / "indep_cal_bloodmnist_metrics.csv",
    ),
}

# ── run ───────────────────────────────────────────────────────────────────────
def run_dataset(name, cfg):
    print(f"\n{'='*72}")
    print(f"Dataset: {name.upper()}  "
          f"(n={cfg['n_qubits']} qubits, L={cfg['n_layers']}, "
          f"ε={'0.8' if 'blood' in name else '0.8'})")
    print(f"  R={R_SIZE}  C={C_SIZE}  E={E_SIZE}  split_seed={SPLIT_SEED}")
    print(f"{'='*72}")
    print(f"{'seed':>5}  {'λ_r':>5}  {'ep':>5}  {'AUC(E)':>8}  {'AUC(all)':>9}  "
          f"{'ASR':>7}  {'TPR5':>7}  {'FPR_E':>7}  {'JSER':>7}  {'Mahal':>8}")
    print("-" * 80)

    rows = []
    for lam in cfg["lambdas"]:
        aucs_e, aucs_a, asrs, tprs, jsers, mahals = [], [], [], [], [], []
        for seed in cfg["seeds"]:
            pr_s = f"{float(cfg['pr']):.12g}"
            wgt  = cfg["wgt_dir"] / f"weights_seed={seed}_L{cfg['n_layers']}_lr{lam}.npy"
            res  = cfg["res_dir"] / f"result_seed={seed}_L{cfg['n_layers']}_lr{lam}.json"
            meta_path = (cfg["data_dir"]
                         / f"seed_{seed}"
                         / f"layer_{cfg['n_layers']}_grid"
                         / cfg["pair"]
                         / f"pr_{pr_s}"
                         / f"poisoned_samples_meta_seed_{seed}_pr_{pr_s}.pt")

            if not wgt.exists():
                print(f"  SKIP (no weights): {wgt.name}"); continue
            if not meta_path.exists():
                print(f"  SKIP (no meta): {meta_path}"); continue

            weights = np.load(wgt).astype(np.float64)
            meta    = torch.load(meta_path, map_location="cpu", weights_only=False)
            x_ct    = meta["target_clean_data"].numpy().astype(np.float32)
            x_p     = meta["poisoned_non_target_data"].numpy().astype(np.float32)

            # Check sample count
            if len(x_ct) != R_SIZE + C_SIZE + E_SIZE:
                print(f"  WARNING: expected 540 clean, got {len(x_ct)} — skipping")
                continue

            ep, asr, ca = 0, 1.0, 0.0
            if res.exists():
                r    = json.load(open(res))
                final = r.get("final", r.get("latest", {}))
                ep   = final.get("epoch", 0)
                asr  = final.get("asr",   1.0)
                ca   = final.get("ca",    0.0)

            m = evaluate_indep(weights, x_ct, x_p,
                               cfg["n_layers"], cfg["n_qubits"], asr)

            print(f"{seed:>5}  {lam:>5}  {ep:>5}  "
                  f"{m['auc_indep']:>8.4f}  {m['auc_all']:>9.4f}  "
                  f"{asr:>7.4f}  {m['tpr_5fpr']:>7.4f}  "
                  f"{m['fpr_e']:>7.4f}  {m['jser']:>7.4f}  "
                  f"{m['mahal_d']:>8.3f}")

            rows.append(dict(seed=seed, lambda_r=lam, epoch=ep,
                             auc_indep=m["auc_indep"], auc_all=m["auc_all"],
                             asr=round(asr, 4), ca=round(ca, 4),
                             tpr_5fpr=m["tpr_5fpr"], fpr_e=m["fpr_e"],
                             jser=m["jser"], mahal_d=m["mahal_d"],
                             tau=m["tau"]))
            aucs_e.append(m["auc_indep"]); aucs_a.append(m["auc_all"])
            asrs.append(asr); tprs.append(m["tpr_5fpr"])
            jsers.append(m["jser"]); mahals.append(m["mahal_d"])

        if aucs_e:
            print(f"  MEAN  {lam:>5}        "
                  f"{np.mean(aucs_e):>8.4f}  {np.mean(aucs_a):>9.4f}  "
                  f"{np.mean(asrs):>7.4f}  {np.mean(tprs):>7.4f}  "
                  f"  ----   {np.mean(jsers):>7.4f}  {np.mean(mahals):>8.3f}")
            print(f"   STD               "
                  f"{np.std(aucs_e):>8.4f}  {np.std(aucs_a):>9.4f}  "
                  f"{np.std(asrs):>7.4f}  {np.std(tprs):>7.4f}  "
                  f"         {np.std(jsers):>7.4f}  {np.std(mahals):>8.3f}")
        print()

    if rows:
        with open(cfg["out_csv"], "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved → {cfg['out_csv']}")

    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist", "bloodmnist", "both"],
                    default="both")
    args = ap.parse_args()

    targets = (["mnist", "bloodmnist"] if args.dataset == "both"
               else [args.dataset])
    for name in targets:
        run_dataset(name, DATASETS[name])

