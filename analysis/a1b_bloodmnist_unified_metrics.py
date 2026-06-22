"""
A1b: Unified TPR@5%FPR and JSER for QML BloodMNIST L=8
=======================================================
Mirrors p0_unified_metrics.py protocol applied to BloodMNIST L=8.
Uses weight files from training_time_bloodmnist/.

Output:
  results/audit/a1b_bloodmnist_metrics.csv
"""
import sys, json
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
from fast_circuit import run_circuit_batch, _make_evals

WDIR     = ROOT / "results/training_time_bloodmnist"
DATA_DIR = ROOT / "data/MedMnist/single_detect_20260411/epsilon_0.8"
OUT_DIR  = ROOT / "results/audit"
OUT_DIR.mkdir(exist_ok=True)

N_QUBITS = 8
N_LAYERS = 8
PAIR     = "t6_vs_t0"
PR       = "0.1"
SEEDS    = [42, 43, 44, 45, 46]
LAMBDAS  = [0.0, 0.5, 1.0, 2.0]
FPR      = 0.05

# ── gate helpers for X/Y measurements ────────────────────────────────────────
import numpy as np_
_H   = np_.array([[1,1],[1,-1]], dtype=np_.complex128)/np_.sqrt(2)
_HS  = np_.array([[1,-1j],[1,1j]], dtype=np_.complex128)/np_.sqrt(2)

def _apply_1q(psi, gate, q):
    dim=psi.shape[1]; step=dim>>(q+1); groups=dim//(2*step)
    return np_.einsum("ij,bkjl->bkil",gate,
                      psi.reshape(-1,groups,2,step),optimize=True).reshape(-1,dim)

def xyz_feats(samples, weights):
    z_ev, _, _ = _make_evals(N_QUBITS)
    out = run_circuit_batch(samples.astype(np_.float64), weights, N_LAYERS, N_QUBITS)
    # out[:, :N_QUBITS] = Z features
    # Need to re-run for X,Y via basis rotation
    norms = np_.linalg.norm(samples, axis=1, keepdims=True).clip(min=1e-12)
    psi = (samples / norms).astype(np_.complex128)
    wr = weights.reshape(N_LAYERS, N_QUBITS, 2)
    for l in range(N_LAYERS):
        for q in range(N_QUBITS):
            ct,st = np_.cos(wr[l,q,0]/2), np_.sin(wr[l,q,0]/2)
            psi = _apply_1q(psi, np_.array([[ct,-st],[st,ct]],dtype=np_.complex128), q)
            ep = np_.exp(1j*wr[l,q,1]/2)
            psi = _apply_1q(psi, np_.array([[1/ep,0],[0,ep]],dtype=np_.complex128), q)
        for q in range(N_QUBITS):
            dim=psi.shape[1]; n=int(np_.log2(dim))
            cb,tb = n-1-q, n-1-((q+1)%N_QUBITS)
            states = np_.arange(dim,dtype=np_.int64)
            psi = psi[:,np_.where((states>>cb)&1, states^(1<<tb), states)]
    probs = np_.abs(psi)**2
    Z = (probs @ z_ev.T).astype(np_.float64)
    X = np_.stack([(np_.abs(_apply_1q(psi,_H,q))**2@z_ev[q]) for q in range(N_QUBITS)],axis=1)
    Y = np_.stack([(np_.abs(_apply_1q(psi,_HS,q))**2@z_ev[q]) for q in range(N_QUBITS)],axis=1)
    return np_.hstack([X,Y,Z]).astype(np_.float64)

def load_data(seed):
    pr_s = f"{float(PR):.12g}"
    base = DATA_DIR / f"seed_{seed}/layer_{N_LAYERS}_grid/{PAIR}/pr_{pr_s}"
    meta = torch.load(base / f"poisoned_samples_meta_seed_{seed}_pr_{pr_s}.pt",
                      map_location="cpu", weights_only=False)
    x_ct  = meta["target_clean_data"].numpy().astype(np_.float32)
    x_cnt = meta["non_target_clean_data"].numpy().astype(np_.float32)
    x_p   = meta["poisoned_non_target_data"].numpy().astype(np_.float32)
    return x_ct, x_cnt, x_p

def compute_metrics(weights, x_ct, x_cnt, x_p):
    feat_c = xyz_feats(x_ct, weights)
    feat_p = xyz_feats(x_p,  weights)

    n = min(len(feat_c), len(feat_p), 300)
    rng = np_.random.default_rng(0)
    fc = feat_c[rng.choice(len(feat_c), n, replace=False)]
    fp = feat_p[rng.choice(len(feat_p), n, replace=False)]

    lw = LedoitWolf().fit(fc)
    mu, prec = lw.location_, lw.get_precision()
    sc = np_.einsum("ni,ij,nj->n", fc-mu, prec, fc-mu)
    sp = np_.einsum("ni,ij,nj->n", fp-mu, prec, fp-mu)

    labels = np_.array([0]*n + [1]*n)
    scores = np_.concatenate([sc, sp])
    auc = float(roc_auc_score(labels, scores))
    sep = max(auc, 1-auc)

    t5  = float(np_.percentile(sc, 95))
    tpr = float(np_.mean(sp > t5))
    jser = float(1 - tpr)

    mahal_d = float(np_.sqrt(sp).mean())
    return dict(auc=round(auc,4), sep=round(sep,4),
                tpr_5fpr=round(tpr,4), jser=round(jser,4),
                mahal_d=round(mahal_d,4))

# ── Main ──────────────────────────────────────────────────────────────────────
rows = []
print(f"{'seed':>5}  {'lam':>5}  {'ep':>5}  {'AUC':>7}  {'Sep':>7}  "
      f"{'ASR':>7}  {'CA':>7}  {'TPR5':>7}  {'JSER':>7}  {'Mahal':>8}")
print("-"*80)

for lam in LAMBDAS:
    aucs, seps, asrs, cas, tprs, jsers, mahals = [],[],[],[],[],[],[]
    for seed in SEEDS:
        wgt_path = WDIR / f"weights_seed={seed}_L{N_LAYERS}_lr{lam}.npy"
        res_path = WDIR / f"result_seed={seed}_L{N_LAYERS}_lr{lam}.json"
        if not wgt_path.exists():
            print(f"  SKIP weights: {wgt_path.name}"); continue

        weights = np_.load(wgt_path).astype(np_.float64)
        x_ct, x_cnt, x_p = load_data(seed)

        m = compute_metrics(weights, x_ct, x_cnt, x_p)

        # ASR/CA/epoch from result JSON
        ep, asr, ca = 0, 0.0, 0.0
        if res_path.exists():
            with open(res_path) as f: r = json.load(f)
            final = r.get('final', r.get('latest', {}))
            ep  = final.get('epoch', 0)
            asr = final.get('asr', 0.0)
            ca  = final.get('ca',  0.0)

        print(f"{seed:>5}  {lam:>5}  {ep:>5}  {m['auc']:>7.4f}  {m['sep']:>7.4f}  "
              f"{asr:>7.4f}  {ca:>7.4f}  {m['tpr_5fpr']:>7.4f}  {m['jser']:>7.4f}  "
              f"{m['mahal_d']:>8.3f}")

        row = dict(seed=seed, lambda_r=lam, epoch=ep,
                   auc=m['auc'], sep=m['sep'], asr=asr, ca=ca,
                   tpr_5fpr=m['tpr_5fpr'], jser=m['jser'], mahal_d=m['mahal_d'])
        rows.append(row)
        aucs.append(m['auc']); seps.append(m['sep']); asrs.append(asr); cas.append(ca)
        tprs.append(m['tpr_5fpr']); jsers.append(m['jser']); mahals.append(m['mahal_d'])

    if aucs:
        print(f"  MEAN  {lam:>5}        "
              f"{np_.mean(aucs):>7.4f}  {np_.mean(seps):>7.4f}  "
              f"{np_.mean(asrs):>7.4f}  {np_.mean(cas):>7.4f}  "
              f"{np_.mean(tprs):>7.4f}  {np_.mean(jsers):>7.4f}  "
              f"{np_.mean(mahals):>8.3f}")
        print(f"   STD               "
              f"{np_.std(aucs):>7.4f}  {np_.std(seps):>7.4f}  "
              f"{np_.std(asrs):>7.4f}  {np_.std(cas):>7.4f}  "
              f"{np_.std(tprs):>7.4f}  {np_.std(jsers):>7.4f}  "
              f"{np_.std(mahals):>8.3f}")
    print()

import csv
out_path = OUT_DIR / "a1b_bloodmnist_metrics.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"Saved → {out_path}")

