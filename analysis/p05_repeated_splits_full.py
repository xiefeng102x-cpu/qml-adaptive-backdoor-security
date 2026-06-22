"""
p05_repeated_splits_full.py
============================
Repeated calibration splits: assess JSER/TPR stability across 50 random
R/C/E assignments using existing ep200 weights (lambda_r=1.0, MNIST only).

Protocol per split_seed s:
  - permute 540 clean samples with rng(s)
  - R=300 -> LedoitWolf fit
  - C=120 -> 95th-percentile threshold tau
  - E=120 -> empirical FPR
  - 60 poisoned samples -> TPR, JSER

Output files (written incrementally):
  results/repeated_splits/repeated_splits_raw.csv    -- one row per (seed, split_seed)
  results/repeated_splits/repeated_splits_run.log    -- progress log, flushed each line

Summary printed at end and written to:
  results/repeated_splits/repeated_splits_summary.csv
"""

import sys, json, time
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

# ── config ────────────────────────────────────────────────────────────────────
N_SPLITS  = 50
LAM       = 1.0
N_LAYERS  = 12
N_QUBITS  = 5
PAIR      = "t7_vs_t0"
PR        = "0.1"
SEEDS     = [42, 43, 44, 45, 46]
R_SIZE, C_SIZE, E_SIZE = 300, 120, 120

WGT_DIR  = ROOT / "results/training_time/new_version"
DATA_DIR = ROOT / "data/Mnist/detail_single_samle"
OUT_DIR  = ROOT / "results/repeated_splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV     = OUT_DIR / "repeated_splits_raw.csv"
LOG_FILE    = OUT_DIR / "repeated_splits_run.log"
SUMMARY_CSV = OUT_DIR / "repeated_splits_summary.csv"

# ── circuit helpers ───────────────────────────────────────────────────────────
_H_gate  = np.array([[1, 1],[1,-1]], dtype=np.complex128) / np.sqrt(2)
_HS_gate = np.array([[1,-1j],[1,1j]], dtype=np.complex128) / np.sqrt(2)

def _apply_1q(psi, gate, q):
    dim = psi.shape[1]; step = dim >> (q+1); groups = dim // (2*step)
    return np.einsum("ij,bkjl->bkil", gate,
                     psi.reshape(-1, groups, 2, step), optimize=True).reshape(-1, dim)

def get_xyz_feats(samples, weights):
    z_ev, _, _ = _make_evals(N_QUBITS)
    norms = np.linalg.norm(samples, axis=1, keepdims=True).clip(min=1e-12)
    psi   = (samples / norms).astype(np.complex128)
    wr    = weights.reshape(N_LAYERS, N_QUBITS, 2)
    for l in range(N_LAYERS):
        for q in range(N_QUBITS):
            ct, st = np.cos(wr[l,q,0]/2), np.sin(wr[l,q,0]/2)
            psi = _apply_1q(psi, np.array([[ct,-st],[st,ct]], dtype=np.complex128), q)
            ep  = np.exp(1j*wr[l,q,1]/2)
            psi = _apply_1q(psi, np.array([[1/ep,0],[0,ep]], dtype=np.complex128), q)
        for q in range(N_QUBITS):
            dim = psi.shape[1]; n = int(np.log2(dim))
            cb, tb = n-1-q, n-1-((q+1) % N_QUBITS)
            states = np.arange(dim, dtype=np.int64)
            psi = psi[:, np.where((states>>cb)&1, states^(1<<tb), states)]
    probs = np.abs(psi)**2
    Z = (probs @ z_ev.T).astype(np.float64)
    X = np.stack([(np.abs(_apply_1q(psi,_H_gate, q))**2 @ z_ev[q]) for q in range(N_QUBITS)], axis=1)
    Y = np.stack([(np.abs(_apply_1q(psi,_HS_gate,q))**2 @ z_ev[q]) for q in range(N_QUBITS)], axis=1)
    return np.hstack([X, Y, Z]).astype(np.float64)

def mahal_sq(xyz, mu, prec):
    d = xyz - mu
    return np.einsum("ni,ij,nj->n", d, prec, d)

def evaluate_one(weights, x_clean, x_poison, asr, split_seed):
    rng = np.random.default_rng(split_seed)
    idx = rng.permutation(len(x_clean))
    x_r = x_clean[idx[:R_SIZE]]
    x_c = x_clean[idx[R_SIZE:R_SIZE+C_SIZE]]
    x_e = x_clean[idx[R_SIZE+C_SIZE:]]

    xyz_r = get_xyz_feats(x_r.astype(np.float64), weights)
    xyz_c = get_xyz_feats(x_c.astype(np.float64), weights)
    xyz_e = get_xyz_feats(x_e.astype(np.float64), weights)
    xyz_p = get_xyz_feats(x_poison.astype(np.float64), weights)

    lw   = LedoitWolf().fit(xyz_r)
    mu   = lw.location_; prec = lw.get_precision()
    sc   = mahal_sq(xyz_c, mu, prec)
    se   = mahal_sq(xyz_e, mu, prec)
    sp   = mahal_sq(xyz_p, mu, prec)
    tau  = float(np.percentile(sc, 95))
    tpr  = float(np.mean(sp > tau))
    fpr  = float(np.mean(se > tau))
    jser = float(asr * (1 - tpr))
    D    = float(np.sqrt(sp).mean())

    y_true  = np.concatenate([np.zeros(len(se)), np.ones(len(sp))])
    y_score = np.concatenate([se, sp])
    auc     = float(roc_auc_score(y_true, y_score))

    return dict(tau=round(tau,4), sqrt_tau=round(tau**0.5,4),
                tpr=round(tpr,4), fpr=round(fpr,4),
                jser=round(jser,4), D=round(D,4), auc=round(auc,4))

# ── load weights and meta for all seeds ───────────────────────────────────────
seed_data = {}
for seed in SEEDS:
    wgt_path  = WGT_DIR / f"weights_seed={seed}_L{N_LAYERS}_lr{LAM}.npy"
    meta_path = (DATA_DIR / f"seed_{seed}"
                 / f"layer_{N_LAYERS}_grid" / PAIR / f"pr_{PR}"
                 / f"poisoned_samples_meta_seed_{seed}_pr_{PR}.pt")
    res_path  = WGT_DIR / f"result_seed={seed}_L{N_LAYERS}_lr{LAM}.json"

    assert wgt_path.exists(),  f"MISSING weight: {wgt_path}"
    assert meta_path.exists(), f"MISSING meta:   {meta_path}"

    weights = np.load(wgt_path).astype(np.float64)
    meta    = torch.load(meta_path, map_location="cpu", weights_only=False)
    x_ct    = meta["target_clean_data"].numpy().astype(np.float32)
    x_p     = meta["poisoned_non_target_data"].numpy().astype(np.float32)

    asr = 1.0
    if res_path.exists():
        r   = json.load(open(res_path))
        fin = r.get("final", r.get("latest", {}))
        asr = fin.get("asr", 1.0)

    seed_data[seed] = dict(weights=weights, x_ct=x_ct, x_p=x_p, asr=asr)

# ── open output files ─────────────────────────────────────────────────────────
raw_f   = open(RAW_CSV, "w", buffering=1)   # line-buffered
log_f   = open(LOG_FILE, "w", buffering=1)  # line-buffered

raw_header = "split_seed,model_seed,tau,sqrt_tau,tpr,fpr,jser,D,auc\n"
raw_f.write(raw_header)
raw_f.flush()

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_f.write(line + "\n")
    log_f.flush()

log(f"START: {N_SPLITS} splits × {len(SEEDS)} seeds  lambda_r={LAM}  MNIST L={N_LAYERS}")
log(f"Output dir: {OUT_DIR}")
log("-" * 72)

# ── main loop ─────────────────────────────────────────────────────────────────
t_global = time.perf_counter()
all_rows = []

for ss in range(N_SPLITS):
    split_results = []
    for seed in SEEDS:
        d = seed_data[seed]
        m = evaluate_one(d["weights"], d["x_ct"], d["x_p"], d["asr"], split_seed=ss)
        row = dict(split_seed=ss, model_seed=seed, **m)
        all_rows.append(row)
        split_results.append(m)

        # write raw CSV row immediately
        raw_f.write(f"{ss},{seed},{m['tau']},{m['sqrt_tau']},"
                    f"{m['tpr']},{m['fpr']},{m['jser']},{m['D']},{m['auc']}\n")
        raw_f.flush()

    # per-split summary log line
    jsers = [r["jser"] for r in split_results]
    tprs  = [r["tpr"]  for r in split_results]
    taus  = [r["tau"]  for r in split_results]
    jser_vals = " ".join("%.3f" % r["jser"] for r in split_results)
    log(f"split {ss:02d}: "
        f"tau={np.mean(taus):.2f}+/-{np.std(taus):.2f}  "
        f"TPR={np.mean(tprs):.3f}+/-{np.std(tprs):.3f}  "
        f"JSER={np.mean(jsers):.3f}+/-{np.std(jsers):.3f}  "
        f"[seeds: {jser_vals}]")

raw_f.close()

elapsed = time.perf_counter() - t_global
log("-" * 72)
log(f"DONE: {len(all_rows)} evaluations in {elapsed:.1f}s")

# ── global summary ────────────────────────────────────────────────────────────
log("")
log("=== GLOBAL SUMMARY (across all splits, all seeds) ===")
all_jser = [r["jser"] for r in all_rows]
all_tpr  = [r["tpr"]  for r in all_rows]
all_tau  = [r["tau"]  for r in all_rows]
all_D    = [r["D"]    for r in all_rows]
all_sqrt = [r["sqrt_tau"] for r in all_rows]
log(f"  JSER : mean={np.mean(all_jser):.4f}  std={np.std(all_jser):.4f}  "
    f"min={np.min(all_jser):.4f}  max={np.max(all_jser):.4f}")
log(f"  TPR  : mean={np.mean(all_tpr):.4f}  std={np.std(all_tpr):.4f}  "
    f"min={np.min(all_tpr):.4f}  max={np.max(all_tpr):.4f}")
log(f"  tau  : mean={np.mean(all_tau):.3f}  std={np.std(all_tau):.3f}  "
    f"min={np.min(all_tau):.3f}  max={np.max(all_tau):.3f}")
log(f"  sqrt_tau: mean={np.mean(all_sqrt):.3f}  std={np.std(all_sqrt):.3f}  "
    f"min={np.min(all_sqrt):.3f}  max={np.max(all_sqrt):.3f}")
log(f"  D    : mean={np.mean(all_D):.4f}  std={np.std(all_D):.4f}  "
    f"min={np.min(all_D):.4f}  max={np.max(all_D):.4f}")

log("")
log("=== PER-MODEL-SEED SUMMARY (across 50 splits) ===")
sum_rows = []
for seed in SEEDS:
    seed_rows = [r for r in all_rows if r["model_seed"] == seed]
    js = [r["jser"] for r in seed_rows]
    tp = [r["tpr"]  for r in seed_rows]
    ta = [r["tau"]  for r in seed_rows]
    Ds = [r["D"]    for r in seed_rows]
    log(f"  seed {seed}: JSER={np.mean(js):.4f}±{np.std(js):.4f} "
        f"[{np.min(js):.4f},{np.max(js):.4f}]  "
        f"TPR={np.mean(tp):.4f}±{np.std(tp):.4f}  "
        f"tau={np.mean(ta):.2f}±{np.std(ta):.2f}  "
        f"D={np.mean(Ds):.4f}±{np.std(Ds):.4f}")
    sum_rows.append(dict(
        model_seed=seed,
        jser_mean=round(np.mean(js),4), jser_std=round(np.std(js),4),
        jser_min=round(np.min(js),4),   jser_max=round(np.max(js),4),
        tpr_mean=round(np.mean(tp),4),  tpr_std=round(np.std(tp),4),
        tau_mean=round(np.mean(ta),4),  tau_std=round(np.std(ta),4),
        D_mean=round(np.mean(Ds),4),    D_std=round(np.std(Ds),4),
    ))

# write summary CSV
with open(SUMMARY_CSV, "w") as f:
    f.write("model_seed,jser_mean,jser_std,jser_min,jser_max,"
            "tpr_mean,tpr_std,tau_mean,tau_std,D_mean,D_std\n")
    for r in sum_rows:
        f.write(f"{r['model_seed']},{r['jser_mean']},{r['jser_std']},"
                f"{r['jser_min']},{r['jser_max']},"
                f"{r['tpr_mean']},{r['tpr_std']},"
                f"{r['tau_mean']},{r['tau_std']},"
                f"{r['D_mean']},{r['D_std']}\n")

log("")
log(f"Raw CSV    : {RAW_CSV}")
log(f"Summary CSV: {SUMMARY_CSV}")
log(f"Log file   : {LOG_FILE}")
log_f.close()

