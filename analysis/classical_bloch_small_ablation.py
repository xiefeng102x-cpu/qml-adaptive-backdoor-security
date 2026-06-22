"""
classical_bloch_small_ablation.py
==================================
~120-parameter Classical Bloch-like ablation for the QML adaptive paper.

Purpose
-------
Disentangle geometric-constraint vs. scalar-capacity explanations for
QML's three-goal incompatibility (Corollary 1).

Design
------
BlochHead(input_dim=32, hidden=2, constrained=True)
  Linear(32, 2, bias=True)  =  66 params
  Tanh
  Linear( 2,15, bias=True)  =  45 params
  Bloch-ball projection
  ─────────────────────────────────────────
  Total adaptive params      = 111   ← compare to QML MNIST L=12: 120

ClassicalClassifier(n_qubits=5)
  Linear(5,16) + Tanh + Linear(16,2)  = 130 params — FROZEN during adaptive

Protocol
--------
Same as main paper (Methods §"Detection scores and calibration"):
  R=300  → LedoitWolf fit (mu, Sigma)
  C=120  → 95th-pct threshold  (5% FPR calibration)
  E=120  → TPR@5%FPR, JSER evaluation
  SPLIT_SEED = 0 (fixed)

Grid
----
  Seeds    : 42–46  (same 5 seeds as main experiment)
  λr       : 0, 0.5, 1, 2
  Epochs   : 150 pretrain + 200 adaptive  (same as main Classical Bloch)
  Pair     : t7_vs_t0  (MNIST), L=12, pr=0.1

Output
------
  results/classical_bloch_small/
    ablation_results_<timestamp>.csv       ← per-seed-λr metrics
    ablation_summary_<timestamp>.csv       ← mean±SD across seeds per λr

Resource control
----------------
  --cpu-threads N   (default 4)  leave remaining cores for other tasks
  --device cpu|cuda              (default cpu; model is tiny, cpu is fine)

Usage
-----
  python classical_bloch_small_ablation.py                    # full grid
  python classical_bloch_small_ablation.py --dry-run          # time estimate
  python classical_bloch_small_ablation.py --lambda-r 1.0     # single λr
  python classical_bloch_small_ablation.py --seeds 42 43      # subset seeds
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
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

DATA_ROOT = ROOT / "data/Mnist/detail_single_samle"
OUT_ROOT  = ROOT / "results/classical_bloch_small"

# ── calibration split sizes (must sum to 540 = total clean target samples) ───
R_SIZE    = 300
C_SIZE    = 120
E_SIZE    = 120
SPLIT_SEED = 0


# ── architecture ──────────────────────────────────────────────────────────────

class BlochHeadSmall(nn.Module):
    """
    Capacity-matched Bloch encoder: input_dim → hidden → 3Q.

    hidden=2 → 111 adaptive params (vs. QML MNIST L=12: 120).
    hidden=64 (original) → 3,087 adaptive params.
    """
    def __init__(self, input_dim: int, n_qubits: int,
                 hidden: int = 2, constrained: bool = True):
        super().__init__()
        self.n_qubits    = n_qubits
        self.constrained = constrained
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3 * n_qubits),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z  = self.net(x)
        z3 = z.view(-1, self.n_qubits, 3)
        if self.constrained:
            norms = z3.norm(dim=-1, keepdim=True).clamp(min=1.0)
            z3    = z3 / norms
        x_v = z3[:, :, 0]
        y_v = z3[:, :, 1]
        z_v = z3[:, :, 2]
        return torch.cat([x_v, y_v, z_v], dim=1)  # [N, 3Q]


class FixedClassifier(nn.Module):
    """2-layer MLP on Z-component. Frozen during adaptive phase."""
    def __init__(self, n_qubits: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_qubits, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ── feature helpers ───────────────────────────────────────────────────────────

def xyz_np(head: BlochHeadSmall, x: np.ndarray) -> np.ndarray:
    head.eval()
    with torch.no_grad():
        return head(torch.tensor(x, dtype=torch.float32)).numpy()


def z_np(head: BlochHeadSmall, x: np.ndarray) -> np.ndarray:
    return xyz_np(head, x)[:, 2 * head.n_qubits:]


# ── data loading ──────────────────────────────────────────────────────────────

def load_data(seed: int, layer: int = 12, pair: str = "t7_vs_t0",
              pr: float = 0.1):
    pr_s = f"{pr:.12g}"
    base = DATA_ROOT / f"seed_{seed}" / f"layer_{layer}_grid" / pair / f"pr_{pr_s}"
    meta_path = base / f"poisoned_samples_meta_seed_{seed}_pr_{pr_s}.pt"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta: {meta_path}")
    try:
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    except TypeError:
        meta = torch.load(meta_path, map_location="cpu")
    x_ct  = meta["target_clean_data"].numpy().astype(np.float32)
    x_cnt = meta["non_target_clean_data"].numpy().astype(np.float32)
    x_p   = meta["poisoned_non_target_data"].numpy().astype(np.float32)
    return x_ct, x_cnt, x_p


# ── R/C/E calibration split (mirroring paper Methods) ─────────────────────────

def make_rce_split(x_ct: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split 540 clean target samples into R(300) / C(120) / E(120)."""
    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(len(x_ct))
    r = x_ct[idx[:R_SIZE]]
    c = x_ct[idx[R_SIZE:R_SIZE + C_SIZE]]
    e = x_ct[idx[R_SIZE + C_SIZE:R_SIZE + C_SIZE + E_SIZE]]
    return r, c, e


def mahal_scores(feats: np.ndarray, mu: np.ndarray,
                 prec: np.ndarray) -> np.ndarray:
    diff = feats - mu
    return np.einsum("ni,ij,nj->n", diff, prec, diff)  # D² per sample


def detection_metrics(head: BlochHeadSmall,
                      x_ct: np.ndarray,
                      x_p: np.ndarray,
                      asr: float) -> dict:
    """
    Full calibrated detection metrics using R/C/E split.
    Returns: auc, sep, tpr_5fpr, jser
    """
    r, c, e = make_rce_split(x_ct)

    xyz_r  = xyz_np(head, r)
    xyz_c  = xyz_np(head, c)
    xyz_e  = xyz_np(head, e)
    xyz_p  = xyz_np(head, x_p)

    # Fit on R
    lw = LedoitWolf().fit(xyz_r)
    mu    = lw.location_
    prec  = lw.get_precision()

    sc_r  = mahal_scores(xyz_r, mu, prec)
    sc_c  = mahal_scores(xyz_c, mu, prec)
    sc_e  = mahal_scores(xyz_e, mu, prec)
    sc_p  = mahal_scores(xyz_p, mu, prec)

    # AUC: clean (R+C+E combined) vs. poison
    all_clean_scores = np.concatenate([sc_r, sc_c, sc_e])
    n_c = len(all_clean_scores)
    n_p = len(sc_p)
    labels = np.concatenate([np.zeros(n_c), np.ones(n_p)])
    scores = np.concatenate([all_clean_scores, sc_p])
    auc = float(roc_auc_score(labels, scores))
    sep = max(auc, 1.0 - auc)

    # Threshold from C (5% FPR)
    tau = float(np.percentile(sc_c, 95))

    # TPR@5%FPR on poison samples
    tpr = float(np.mean(sc_p > tau))

    # Empirical FPR on E (diagnostic only)
    fpr_e = float(np.mean(sc_e > tau))

    # JSER = ASR × (1 - TPR)  [joint successful evasion rate]
    jser = asr * (1.0 - tpr)

    # Mean Mahalanobis D (sqrt of D²) for poison
    mahal_d = float(np.sqrt(sc_p.mean()))

    return {
        "auc":      round(auc,    4),
        "sep":      round(sep,    4),
        "tpr_5fpr": round(tpr,    4),
        "jser":     round(jser,   4),
        "fpr_e":    round(fpr_e,  4),
        "mahal_d":  round(mahal_d, 4),
    }


# ── phase 1: pre-training ─────────────────────────────────────────────────────

def pretrain(head: BlochHeadSmall, clf: FixedClassifier,
             x_ct: np.ndarray, x_cnt: np.ndarray, x_p: np.ndarray,
             epochs: int = 150, lr: float = 1e-3,
             verbose: bool = True) -> None:
    x_tr = np.vstack([x_ct, x_cnt, x_p])
    y_tr = np.concatenate([
        np.ones(len(x_ct),  dtype=np.int64),
        np.zeros(len(x_cnt), dtype=np.int64),
        np.ones(len(x_p),  dtype=np.int64),
    ])
    opt = optim.Adam(list(head.parameters()) + list(clf.parameters()), lr=lr)
    rng = np.random.default_rng(42)
    ce  = nn.CrossEntropyLoss()

    head.train(); clf.train()
    for ep in range(epochs):
        idx = rng.permutation(len(x_tr))
        for i in range(0, len(idx), 32):
            bi  = idx[i:i+32]
            xb  = torch.tensor(x_tr[bi], dtype=torch.float32)
            yb  = torch.tensor(y_tr[bi], dtype=torch.long)
            z_b = head(xb)[:, 2 * head.n_qubits:]  # Z component
            loss = ce(clf(z_b), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    head.eval(); clf.eval()
    with torch.no_grad():
        z_c   = torch.tensor(z_np(head, x_ct),  dtype=torch.float32)
        z_cnt = torch.tensor(z_np(head, x_cnt), dtype=torch.float32)
        z_p   = torch.tensor(z_np(head, x_p),   dtype=torch.float32)
        asr_pre = float((clf(z_p).argmax(1) == 1).float().mean())
        n_correct = int((clf(z_c).argmax(1) == 1).sum()) + \
                    int((clf(z_cnt).argmax(1) == 0).sum())
    ca_pre = float(n_correct / (len(x_ct) + len(x_cnt)))
    if verbose:
        print(f"    Pretrain done: CA={ca_pre:.4f}  ASR={asr_pre:.4f}")


# ── phase 2: adaptive attack ──────────────────────────────────────────────────

def mahal_loss_torch(feats: torch.Tensor,
                     mu: torch.Tensor,
                     prec: torch.Tensor) -> torch.Tensor:
    diff = feats - mu
    return (diff @ prec * diff).sum(dim=1).mean()


def adaptive_attack(head: BlochHeadSmall, clf: FixedClassifier,
                    x_ct: np.ndarray, x_cnt: np.ndarray, x_p: np.ndarray,
                    epochs: int = 200, lr: float = 1e-2,
                    lambda_r: float = 1.0,
                    eval_every: int = 40,
                    verbose: bool = True) -> dict:
    """
    Fine-tune head only (clf frozen):
        L = CE(z_features, labels) + lambda_r * Mahal_XYZ(poison, clean_ref)
    Reference (mu, prec) fixed from pre-adaptive x_ct XYZ features.
    Returns final metrics dict.
    """
    # Reference distribution (fixed before adaptive training)
    xyz_ct   = xyz_np(head, x_ct)
    mu_t     = torch.tensor(xyz_ct.mean(0), dtype=torch.float32)
    lw_ref   = LedoitWolf().fit(xyz_ct)
    prec_t   = torch.tensor(lw_ref.get_precision(), dtype=torch.float32)

    x_tr = np.vstack([x_ct, x_cnt, x_p])
    y_tr = np.concatenate([
        np.ones(len(x_ct),  dtype=np.int64),
        np.zeros(len(x_cnt), dtype=np.int64),
        np.ones(len(x_p),  dtype=np.int64),
    ])

    # Freeze classifier
    for param in clf.parameters():
        param.requires_grad_(False)

    opt = optim.Adam(head.parameters(), lr=lr)
    rng = np.random.default_rng(42)
    ce  = nn.CrossEntropyLoss()
    xp_t = torch.tensor(x_p, dtype=torch.float32)

    head.train()
    for epoch in range(1, epochs + 1):
        idx = rng.permutation(len(x_tr))
        for i in range(0, len(idx), 32):
            bi      = idx[i:i+32]
            xb      = torch.tensor(x_tr[bi], dtype=torch.float32)
            yb      = torch.tensor(y_tr[bi], dtype=torch.long)
            feats   = head(xb)
            z_b     = feats[:, 2 * head.n_qubits:]
            loss_ce = ce(clf(z_b), yb)
            if lambda_r > 0:
                xyz_p  = head(xp_t)
                loss_r = mahal_loss_torch(xyz_p, mu_t, prec_t)
            else:
                loss_r = torch.tensor(0.0)
            loss = loss_ce + lambda_r * loss_r
            opt.zero_grad()
            loss.backward()
            opt.step()

        if verbose and epoch % eval_every == 0:
            head.eval()
            with torch.no_grad():
                z_p   = torch.tensor(z_np(head, x_p),   dtype=torch.float32)
                z_c   = torch.tensor(z_np(head, x_ct),  dtype=torch.float32)
                z_cnt = torch.tensor(z_np(head, x_cnt), dtype=torch.float32)
                asr_e = float((clf(z_p).argmax(1) == 1).float().mean())
                n_cor = int((clf(z_c).argmax(1) == 1).sum()) + \
                        int((clf(z_cnt).argmax(1) == 0).sum())
            ca_e = float(n_cor / (len(x_ct) + len(x_cnt)))
            print(f"      ep {epoch:3d}: CA={ca_e:.4f}  ASR={asr_e:.4f}", end="  ")
            dm = detection_metrics(head, x_ct, x_p, asr_e)
            print(f"AUC={dm['auc']:.4f}  JSER={dm['jser']:.4f}")
            head.train()

    # Final evaluation
    head.eval()
    with torch.no_grad():
        z_p   = torch.tensor(z_np(head, x_p),   dtype=torch.float32)
        z_c   = torch.tensor(z_np(head, x_ct),  dtype=torch.float32)
        z_cnt = torch.tensor(z_np(head, x_cnt), dtype=torch.float32)
        asr_f = float((clf(z_p).argmax(1) == 1).float().mean())
        n_cor = int((clf(z_c).argmax(1) == 1).sum()) + \
                int((clf(z_cnt).argmax(1) == 0).sum())
    ca_f = float(n_cor / (len(x_ct) + len(x_cnt)))
    dm   = detection_metrics(head, x_ct, x_p, asr_f)

    # Unfreeze classifier
    for param in clf.parameters():
        param.requires_grad_(True)

    return {"asr": round(asr_f, 4), "ca": round(ca_f, 4), **dm}


# ── single-run entry point ────────────────────────────────────────────────────

def run_one(seed: int, lambda_r: float,
            hidden: int = 2, n_qubits: int = 5,
            layer: int = 12, pair: str = "t7_vs_t0", pr: float = 0.1,
            pretrain_epochs: int = 150, adaptive_epochs: int = 200,
            lr: float = 1e-2, constrained: bool = True,
            verbose: bool = True) -> dict:

    x_ct, x_cnt, x_p = load_data(seed, layer, pair, pr)
    input_dim = x_ct.shape[1]

    head = BlochHeadSmall(input_dim, n_qubits, hidden=hidden,
                          constrained=constrained)
    clf  = FixedClassifier(n_qubits)

    n_adaptive = count_params(head)
    n_frozen   = count_params(clf)

    if verbose:
        print(f"\n  seed={seed} λr={lambda_r} | "
              f"head params: {n_adaptive} (adaptive)  "
              f"clf params: {n_frozen} (frozen during adaptive)")
        print(f"  Phase 1: pretrain ({pretrain_epochs} ep)...")

    pretrain(head, clf, x_ct, x_cnt, x_p,
             epochs=pretrain_epochs, lr=1e-3, verbose=verbose)

    if verbose:
        print(f"  Phase 2: adaptive attack (λr={lambda_r}, {adaptive_epochs} ep)...")

    metrics = adaptive_attack(
        head, clf, x_ct, x_cnt, x_p,
        epochs=adaptive_epochs, lr=lr, lambda_r=lambda_r,
        eval_every=40, verbose=verbose,
    )

    return {
        "seed":          seed,
        "lambda_r":      lambda_r,
        "hidden":        hidden,
        "n_adaptive":    n_adaptive,
        "n_frozen":      n_frozen,
        "constrained":   constrained,
        **metrics,
    }


# ── summary statistics across seeds ──────────────────────────────────────────

def summarise(rows: list[dict]) -> list[dict]:
    metrics_to_agg = ["asr", "ca", "auc", "sep", "tpr_5fpr", "jser",
                      "fpr_e", "mahal_d"]
    summary = []
    lambda_rs = sorted(set(r["lambda_r"] for r in rows))
    for lam in lambda_rs:
        sub = [r for r in rows if r["lambda_r"] == lam]
        rec = {
            "lambda_r":   lam,
            "n_seeds":    len(sub),
            "hidden":     sub[0]["hidden"],
            "n_adaptive": sub[0]["n_adaptive"],
        }
        for m in metrics_to_agg:
            vals = [r[m] for r in sub if m in r]
            if vals:
                rec[f"{m}_mean"] = round(float(np.mean(vals)), 4)
                rec[f"{m}_sd"]   = round(float(np.std(vals, ddof=0)), 4)
        # Jointly feasible? CA≥0.9 AND ASR≥0.7 for ALL seeds
        ca_ok  = all(r["ca"]  >= 0.9 for r in sub)
        asr_ok = all(r["asr"] >= 0.7 for r in sub)
        rec["jointly_feasible"] = "Yes" if (ca_ok and asr_ok) else "No"
        summary.append(rec)
    return summary


# ── CSV writers ───────────────────────────────────────────────────────────────

RESULT_FIELDS = [
    "seed", "lambda_r", "hidden", "n_adaptive", "n_frozen", "constrained",
    "asr", "ca", "auc", "sep", "tpr_5fpr", "jser", "fpr_e", "mahal_d",
]

SUMMARY_FIELDS = [
    "lambda_r", "n_seeds", "hidden", "n_adaptive",
    "asr_mean", "asr_sd",
    "ca_mean",  "ca_sd",
    "auc_mean", "auc_sd",
    "sep_mean", "sep_sd",
    "tpr_5fpr_mean", "tpr_5fpr_sd",
    "jser_mean", "jser_sd",
    "fpr_e_mean", "fpr_e_sd",
    "mahal_d_mean", "mahal_d_sd",
    "jointly_feasible",
]


def write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  → Wrote {len(rows)} rows to {path.name}")


# ── dry-run estimate ──────────────────────────────────────────────────────────

def dry_run_estimate(seeds: list[int], lambda_rs: list[float],
                     pretrain_ep: int, adaptive_ep: int) -> None:
    n_runs = len(seeds) * len(lambda_rs)
    n_epochs_total = n_runs * (pretrain_ep + adaptive_ep)
    # Rough empirical estimate: ~0.5 s per epoch for tiny net on MNIST CPU
    t_est_min = n_epochs_total * 0.5 / 60
    print(f"\n  DRY RUN estimate")
    print(f"  Seeds: {seeds}")
    print(f"  λr values: {lambda_rs}")
    print(f"  Runs: {n_runs}  (pretrain {pretrain_ep} + adaptive {adaptive_ep} ep each)")
    print(f"  Total epochs: {n_epochs_total}")
    print(f"  Estimated time: ~{t_est_min:.0f}–{t_est_min*2:.0f} min on CPU "
          f"(rough; tiny network)")
    print(f"  Peak memory: <100 MB (safe to run alongside other tasks)")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="~120-param Classical Bloch ablation for QML paper")
    ap.add_argument("--seeds",     type=int, nargs="+", default=list(range(42, 47)),
                    help="Random seeds (default: 42 43 44 45 46)")
    ap.add_argument("--lambda-r",  type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 2.0],
                    help="λr values (default: 0 0.5 1 2)")
    ap.add_argument("--hidden",    type=int, default=2,
                    help="Hidden size (default 2 → 111 adaptive params)")
    ap.add_argument("--pretrain-epochs", type=int, default=150)
    ap.add_argument("--adaptive-epochs", type=int, default=200)
    ap.add_argument("--lr",        type=float, default=0.01)
    ap.add_argument("--layer",     type=int, default=12)
    ap.add_argument("--pair",      default="t7_vs_t0")
    ap.add_argument("--pr",        type=float, default=0.1)
    ap.add_argument("--unconstrained", action="store_true",
                    help="Remove Bloch-ball projection (further ablation)")
    ap.add_argument("--cpu-threads", type=int, default=4,
                    help="CPU threads for PyTorch (default 4, leaves others free)")
    ap.add_argument("--dry-run",   action="store_true",
                    help="Print time estimate and exit without running")
    ap.add_argument("--quiet",     action="store_true",
                    help="Suppress per-epoch progress (only print run-level summary)")
    args = ap.parse_args()

    torch.set_num_threads(args.cpu_threads)
    print(f"PyTorch threads: {torch.get_num_threads()} / "
          f"{torch.get_num_interop_threads()} interop")

    constrained = not args.unconstrained
    seeds     = args.seeds
    lambda_rs = args.lambda_r
    verbose   = not args.quiet

    if args.dry_run:
        dry_run_estimate(seeds, lambda_rs,
                         args.pretrain_epochs, args.adaptive_epochs)
        # Quick parameter count check
        head_tmp = BlochHeadSmall(32, 5, hidden=args.hidden, constrained=constrained)
        clf_tmp  = FixedClassifier(5)
        print(f"\n  hidden={args.hidden} → "
              f"adaptive params: {count_params(head_tmp)}  "
              f"frozen clf params: {count_params(clf_tmp)}")
        print(f"  QML MNIST L=12: 120 adaptive params  (reference)")
        return 0

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    result_csv  = OUT_ROOT / f"ablation_results_{ts}.csv"
    summary_csv = OUT_ROOT / f"ablation_summary_{ts}.csv"

    print(f"\nClassical Bloch Small Ablation  (hidden={args.hidden})")
    print(f"  Seeds  : {seeds}")
    print(f"  λr     : {lambda_rs}")
    print(f"  Epochs : {args.pretrain_epochs} pretrain + {args.adaptive_epochs} adaptive")
    print(f"  Output : {OUT_ROOT}")
    print("=" * 60)

    all_rows: list[dict] = []
    t_total = time.time()

    for seed in seeds:
        for lam in lambda_rs:
            t0 = time.time()
            print(f"\n[seed={seed} λr={lam}]")
            try:
                row = run_one(
                    seed=seed, lambda_r=lam,
                    hidden=args.hidden, n_qubits=5,
                    layer=args.layer, pair=args.pair, pr=args.pr,
                    pretrain_epochs=args.pretrain_epochs,
                    adaptive_epochs=args.adaptive_epochs,
                    lr=args.lr, constrained=constrained,
                    verbose=verbose,
                )
                elapsed = time.time() - t0
                row["elapsed_s"] = round(elapsed)
                all_rows.append(row)
                print(f"  Done: ASR={row['asr']:.4f}  CA={row['ca']:.4f}  "
                      f"AUC={row['auc']:.4f}  Sep={row['sep']:.4f}  "
                      f"TPR@5%={row['tpr_5fpr']:.4f}  JSER={row['jser']:.4f}  "
                      f"({elapsed:.0f}s)")
            except Exception as exc:
                print(f"  ERROR: {exc}")
                all_rows.append({
                    "seed": seed, "lambda_r": lam, "hidden": args.hidden,
                    "error": str(exc),
                })

    # Write per-run results
    write_csv(all_rows, result_csv, RESULT_FIELDS + ["elapsed_s"])

    # Write summary
    valid = [r for r in all_rows if "asr" in r]
    if valid:
        summary = summarise(valid)
        write_csv(summary, summary_csv, SUMMARY_FIELDS)

        print("\n" + "=" * 60)
        print(f"SUMMARY  (hidden={args.hidden}, "
              f"n_adaptive={valid[0]['n_adaptive']} vs. QML 120)")
        hdr = f"{'λr':>5}  {'CA':>8}  {'ASR':>8}  {'AUC':>8}  {'Sep':>8}  "
        hdr += f"{'JSER':>8}  Jointly_feasible?"
        print(hdr)
        print("-" * 62)
        for s in summary:
            print(f"  {s['lambda_r']:>3}  "
                  f"{s['ca_mean']:.4f}±{s['ca_sd']:.4f}  "
                  f"{s['asr_mean']:.4f}±{s['asr_sd']:.4f}  "
                  f"{s['auc_mean']:.4f}±{s['auc_sd']:.4f}  "
                  f"{s['sep_mean']:.4f}±{s['sep_sd']:.4f}  "
                  f"{s['jser_mean']:.4f}±{s['jser_sd']:.4f}  "
                  f"{s['jointly_feasible']}")

    total_elapsed = time.time() - t_total
    print(f"\nTotal time: {total_elapsed/60:.1f} min")
    print(f"Results  : {result_csv}")
    print(f"Summary  : {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


