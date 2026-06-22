"""
classical_bloch_control.py
==========================
E1: Classical Bloch-ball matched control experiment.

Architecture: MLP(input_dim, hidden, 3*Q) -> per-qubit Bloch-ball projection.
Same XYZ Mahalanobis detector + same adaptive attack objective as QML.

Key comparisons (to be tabulated against QML results):
  - Classical-unconstrained + XYZ Mahalanobis   (constrained=False)
  - Classical-Bloch         + XYZ Mahalanobis   (constrained=True)  <-- E1 focus
  - QML                     + XYZ Mahalanobis   (reference, from train_adaptive_qnn_fast.py)

Usage:
  python classical_bloch_control.py --seed 42 --layer 4 --lambda_r 1.0
  python classical_bloch_control.py --seed 42 --layer 4 --mode sweep
  python classical_bloch_control.py --mode full --layers 4 8 12
"""

from __future__ import annotations

import argparse
import json
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
OUT_ROOT  = ROOT / "results/classical_bloch"


# ── models ────────────────────────────────────────────────────────────────────

class BlochHead(nn.Module):
    """
    Classical encoder: input_dim → hidden → 3Q → Bloch-ball projection per triplet.

    Output [N, 3Q] in order [X_0..X_{Q-1}, Y_0..Y_{Q-1}, Z_0..Z_{Q-1}]
    to match xyz_feats() from fast_circuit.py.

    Set constrained=False for the unconstrained ablation (no ball projection).
    """
    def __init__(self, input_dim: int, n_qubits: int,
                 hidden: int = 64, constrained: bool = True):
        super().__init__()
        self.n_qubits    = n_qubits
        self.constrained = constrained
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3 * n_qubits),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z  = self.net(x)                            # [N, 3Q]
        z3 = z.view(-1, self.n_qubits, 3)           # [N, Q, 3]  triplets: (X_i, Y_i, Z_i)
        if self.constrained:
            norms = z3.norm(dim=-1, keepdim=True).clamp(min=1.0)
            z3 = z3 / norms                         # project each triplet onto Bloch ball
        # Reorder from [N, Q, (X,Y,Z)] to [N, 3Q] = [X_0..X_{Q-1}, Y..., Z...]
        x_v = z3[:, :, 0]                           # [N, Q]
        y_v = z3[:, :, 1]
        z_v = z3[:, :, 2]
        return torch.cat([x_v, y_v, z_v], dim=1)   # [N, 3Q]


class ClassicalClassifier(nn.Module):
    """
    2-layer linear classifier on Z-only features [N, Q].
    Mirrors the FixedClassifier used in the QML pipeline.
    """
    def __init__(self, n_qubits: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_qubits, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def ce_loss(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return nn.CrossEntropyLoss()(self.forward(z), labels)


# ── feature helpers ───────────────────────────────────────────────────────────

def xyz_features_np(head: BlochHead, x_np: np.ndarray) -> np.ndarray:
    head.eval()
    with torch.no_grad():
        return head(torch.tensor(x_np, dtype=torch.float32)).numpy().astype(np.float64)


def z_features_np(head: BlochHead, x_np: np.ndarray) -> np.ndarray:
    """Z-only features = last Q columns of xyz output."""
    return xyz_features_np(head, x_np)[:, 2 * head.n_qubits:]


def mahal_loss_torch(feats: torch.Tensor,
                     mu: torch.Tensor, prec: torch.Tensor) -> torch.Tensor:
    diff = feats - mu
    return (diff @ prec * diff).sum(dim=1).mean()


def compute_detection_metrics(feat_c: np.ndarray, feat_p: np.ndarray,
                              succ_p: np.ndarray,
                              n: int = 300) -> dict:
    """
    Full detection metrics for one feature space (Z or XYZ).
    succ_p: bool array [len(feat_p)] — True if that poison sample is classified
            as target class (successful backdoor). Used for JSER.
    Returns: auc, sep, tpr_5fpr, tpr_1fpr, jser_5fpr, jser_1fpr
    """
    rng  = np.random.default_rng(0)
    nc   = min(len(feat_c), n);  np_ = min(len(feat_p), n)
    ic   = rng.choice(len(feat_c), nc, replace=False)
    ip   = rng.choice(len(feat_p), np_, replace=False)
    fc   = feat_c[ic];  fp = feat_p[ip]
    nn_  = min(nc, np_)
    fc, fp = fc[:nn_], fp[:nn_]
    succ = succ_p[ip[:nn_]]

    lw    = LedoitWolf().fit(fc)
    mu, P = lw.location_, lw.get_precision()
    sc = np.einsum("ni,ij,nj->n", fc - mu, P, fc - mu)   # D² per sample
    sp = np.einsum("ni,ij,nj->n", fp - mu, P, fp - mu)   # D² per sample

    auc = float(roc_auc_score(np.array([0]*nn_ + [1]*nn_),
                              np.concatenate([sc, sp])))
    sep = max(auc, 1.0 - auc)

    t5  = float(np.percentile(sc, 95))   # threshold at 5% FPR on clean
    t1  = float(np.percentile(sc, 99))   # threshold at 1% FPR on clean
    tpr_5 = float(np.mean(sp > t5))
    tpr_1 = float(np.mean(sp > t1))

    # JSER: Pr[evasion | U_a=1] — conditional on attack success
    # (more conservative than joint Pr[U_a AND evasion] = ASR × JSER_cond)
    if np.any(succ):
        jser_5 = float(np.mean(sp[succ] <= t5))
        jser_1 = float(np.mean(sp[succ] <= t1))
    else:
        jser_5 = jser_1 = None

    # mahal_d: mean Mahalanobis distance D (sqrt of quadratic form) for poison samples
    # matches theoretical D_Σ = sqrt(δᵀ Σ⁻¹ δ), consistent with pauli_ladder reporting
    mahal_d = round(float(np.sqrt(sp).mean()), 4)

    return {
        "auc":      round(auc, 4),
        "sep":      round(sep, 4),
        "tpr_5fpr": round(tpr_5, 4),
        "tpr_1fpr": round(tpr_1, 4),
        "jser_5fpr": round(jser_5, 4) if jser_5 is not None else None,
        "jser_1fpr": round(jser_1, 4) if jser_1 is not None else None,
        "mahal_d":  mahal_d,   # detection-time mean D (not D²)
    }


# ── data loading ──────────────────────────────────────────────────────────────

def pr_token(pr: float) -> str:
    return f"{float(pr):.12g}"


def load_data(seed: int, layer: int, pair: str, pr: float):
    pr_s = pr_token(pr)
    base = DATA_ROOT / f"seed_{seed}" / f"layer_{layer}_grid" / pair / f"pr_{pr_s}"
    meta = torch.load(
        base / f"poisoned_samples_meta_seed_{seed}_pr_{pr_s}.pt",
        map_location="cpu", weights_only=False,
    )
    x_ct  = meta["target_clean_data"].numpy().astype(np.float32)
    x_cnt = meta["non_target_clean_data"].numpy().astype(np.float32)
    x_p   = meta["poisoned_non_target_data"].numpy().astype(np.float32)
    return x_ct, x_cnt, x_p


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_bloch(head: BlochHead, clf: ClassicalClassifier,
                   x_ct: np.ndarray, x_cnt: np.ndarray, x_p: np.ndarray,
                   mu_xyz: np.ndarray, prec_xyz: np.ndarray) -> dict:
    n_q = head.n_qubits
    head.eval(); clf.eval()

    xyz_c  = xyz_features_np(head, x_ct)
    xyz_p  = xyz_features_np(head, x_p)
    z_c    = xyz_c[:, 2*n_q:]
    z_p    = xyz_p[:, 2*n_q:]
    z_cnt  = z_features_np(head, x_cnt)

    with torch.no_grad():
        preds_p = clf(torch.tensor(z_p, dtype=torch.float32)).argmax(dim=1).numpy()
        asr     = float(np.mean(preds_p == 1))
        correct = (
            np.sum(clf(torch.tensor(z_c,   dtype=torch.float32)).argmax(dim=1).numpy() == 1) +
            np.sum(clf(torch.tensor(z_cnt, dtype=torch.float32)).argmax(dim=1).numpy() == 0))
    ca = float(correct / (len(z_c) + len(z_cnt)))

    # succ_p[i]=True when poison sample i is successfully classified as target
    succ_p = (preds_p == 1)

    # Full detection metrics for Z-only and XYZ feature spaces
    mz  = compute_detection_metrics(z_c,   z_p,   succ_p)
    mxyz = compute_detection_metrics(xyz_c, xyz_p, succ_p)

    # Mahalanobis of poison from reference clean mean (training-time monitor)
    # mahal_poison = mean D (sqrt of quadratic form), consistent with theoretical D_Σ
    diff_p  = xyz_p - mu_xyz
    mahal_p = round(float(np.sqrt(np.einsum("ni,ij,nj->n", diff_p, prec_xyz, diff_p)).mean()), 4)

    return {
        # Z-only features (matches QML classifier input)
        "auc_z":          mz["auc"],
        "sep_z":          mz["sep"],
        "tpr_z_5fpr":     mz["tpr_5fpr"],
        "jser_z_5fpr":    mz["jser_5fpr"],
        # XYZ features (main detection channel)
        "auc_xyz":        mxyz["auc"],
        "sep_xyz":        mxyz["sep"],
        "tpr_xyz_5fpr":   mxyz["tpr_5fpr"],
        "tpr_xyz_1fpr":   mxyz["tpr_1fpr"],
        "jser_xyz_5fpr":  mxyz["jser_5fpr"],
        "jser_xyz_1fpr":  mxyz["jser_1fpr"],
        # Task performance
        "asr":            round(asr, 4),
        "ca":             round(ca,  4),
        "mahal_poison":   mahal_p,
        "n_features":     xyz_p.shape[1],
    }


# ── phase 1: pre-training (plant backdoor) ────────────────────────────────────

def pretrain(head: BlochHead, clf: ClassicalClassifier,
             x_ct: np.ndarray, x_cnt: np.ndarray, x_p: np.ndarray,
             epochs: int = 150, lr: float = 1e-3) -> None:
    """
    Train Bloch head + classifier jointly on backdoored data.
    Poisoned samples are treated as target class (=1) — plants the backdoor.
    This mirrors how the QML clean_model_seed_*_epoch_200.pt was trained.
    """
    x_tr = np.vstack([x_ct, x_cnt, x_p])
    y_tr = np.concatenate([
        np.ones(len(x_ct),  dtype=np.int64),
        np.zeros(len(x_cnt), dtype=np.int64),
        np.ones(len(x_p),  dtype=np.int64),   # backdoor: trigger → target class
    ])

    opt = optim.Adam(list(head.parameters()) + list(clf.parameters()), lr=lr)
    rng = np.random.default_rng(42)

    head.train(); clf.train()
    for ep in range(epochs):
        idx = rng.permutation(len(x_tr))
        for i in range(0, len(idx), 32):
            bi = idx[i:i+32]
            xb = torch.tensor(x_tr[bi], dtype=torch.float32)
            yb = torch.tensor(y_tr[bi], dtype=torch.long)
            feats = head(xb)
            z_b   = feats[:, 2 * head.n_qubits:]  # Z component for classifier
            loss  = clf.ce_loss(z_b, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Report baseline clean accuracy + ASR
    head.eval(); clf.eval()
    with torch.no_grad():
        z_c   = torch.tensor(z_features_np(head, x_ct),  dtype=torch.float32)
        z_cnt = torch.tensor(z_features_np(head, x_cnt), dtype=torch.float32)
        z_p   = torch.tensor(z_features_np(head, x_p),   dtype=torch.float32)
        asr = float(np.mean(clf(z_p).argmax(dim=1).numpy() == 1))
        correct = (np.sum(clf(z_c).argmax(dim=1).numpy() == 1) +
                   np.sum(clf(z_cnt).argmax(dim=1).numpy() == 0))
    ca = float(correct / (len(x_ct) + len(x_cnt)))
    print(f"  Pre-training done  CA={ca:.4f}  ASR={asr:.4f}")


# ── phase 2: adaptive attack ──────────────────────────────────────────────────

def train_adaptive(head: BlochHead, clf: ClassicalClassifier,
                   x_ct: np.ndarray, x_cnt: np.ndarray, x_p: np.ndarray,
                   epochs: int, lr: float, lambda_r: float,
                   eval_every: int = 20,
                   out_path: Path | None = None) -> tuple[list, BlochHead]:
    """
    Fine-tune Bloch head (classifier FROZEN) with:
        L = CE(z_features, labels) + lambda_r * Mahal_XYZ(poison, clean_ref)

    Reference distribution (mu_xyz, prec_xyz) is fit on x_ct BEFORE training
    and kept fixed — same as QML adaptive training.
    """
    # Fix reference XYZ distribution on clean target (never updated)
    xyz_ct   = xyz_features_np(head, x_ct)
    mu_xyz   = torch.tensor(xyz_ct.mean(axis=0), dtype=torch.float32)
    lw       = LedoitWolf().fit(xyz_ct)
    prec_xyz = torch.tensor(lw.get_precision(), dtype=torch.float32)

    # Training set: target(1) + non-target(0) + poison(1)
    x_tr = np.vstack([x_ct, x_cnt, x_p])
    y_tr = np.concatenate([
        np.ones(len(x_ct),  dtype=np.int64),
        np.zeros(len(x_cnt), dtype=np.int64),
        np.ones(len(x_p),  dtype=np.int64),
    ])

    # Freeze classifier
    for p in clf.parameters():
        p.requires_grad_(False)

    opt = optim.Adam(head.parameters(), lr=lr)
    rng = np.random.default_rng(42)
    history: list[dict] = []

    # Baseline (epoch 0)
    det0 = evaluate_bloch(head, clf, x_ct, x_cnt, x_p,
                          mu_xyz.numpy(), lw.get_precision())
    det0["epoch"] = 0
    history.append(det0)
    print(f"  Epoch    0 : Z={det0['auc_z']:.4f}  XYZ={det0['auc_xyz']:.4f}  "
          f"ASR={det0['asr']:.4f}  CA={det0['ca']:.4f}  Mahal={det0['mahal_poison']:.2f}")

    xp_t = torch.tensor(x_p, dtype=torch.float32)

    head.train()
    for epoch in range(1, epochs + 1):
        idx = rng.permutation(len(x_tr))
        for i in range(0, len(idx), 32):
            bi  = idx[i:i+32]
            xb  = torch.tensor(x_tr[bi], dtype=torch.float32)
            yb  = torch.tensor(y_tr[bi], dtype=torch.long)
            feats   = head(xb)
            z_b     = feats[:, 2 * head.n_qubits:]
            loss_ce = clf.ce_loss(z_b, yb)
            if lambda_r > 0 and len(x_p) > 0:
                xyz_p  = head(xp_t)
                loss_r = mahal_loss_torch(xyz_p, mu_xyz, prec_xyz)
            else:
                loss_r = torch.tensor(0.0)
            loss = loss_ce + lambda_r * loss_r
            opt.zero_grad()
            loss.backward()
            opt.step()

        if epoch % eval_every == 0:
            head.eval()
            det = evaluate_bloch(head, clf, x_ct, x_cnt, x_p,
                                 mu_xyz.numpy(), lw.get_precision())
            det["epoch"] = epoch
            history.append(det)
            print(f"  Epoch {epoch:4d} : Z={det['auc_z']:.4f}  XYZ={det['auc_xyz']:.4f}  "
                  f"ASR={det['asr']:.4f}  CA={det['ca']:.4f}  Mahal={det['mahal_poison']:.2f}")
            if out_path:
                with open(out_path, "w") as f:
                    json.dump(history, f, indent=2)
            head.train()

    for p in clf.parameters():
        p.requires_grad_(True)

    return history, head


# ── run_one ───────────────────────────────────────────────────────────────────

def run_one(seed: int, layer: int, pair: str, pr: float,
            lambda_r: float, epochs: int, lr: float,
            n_qubits: int = 5, hidden: int = 64,
            constrained: bool = True) -> dict:

    constraint_tag = "bloch" if constrained else "unconstrained"
    tag         = f"seed={seed}_L{layer}_lr{lambda_r}_{constraint_tag}"
    out_summary = OUT_ROOT / f"result_{tag}.json"
    out_weights = OUT_ROOT / f"weights_{tag}.pt"

    if out_summary.exists() and out_weights.exists():
        print(f"\n  Skipping {tag}")
        with open(out_summary) as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"Classical {constraint_tag} control | {tag}")
    t0 = time.time()

    x_ct, x_cnt, x_p = load_data(seed, layer, pair, pr)
    input_dim = x_ct.shape[1]   # 2^n_qubits (amplitude-encoded state vector)

    head = BlochHead(input_dim, n_qubits, hidden, constrained=constrained)
    clf  = ClassicalClassifier(n_qubits)

    # Phase 1: pre-train on backdoored data (plant the backdoor)
    print("  Phase 1: pre-training (150 epochs)...")
    pretrain(head, clf, x_ct, x_cnt, x_p, epochs=150, lr=1e-3)

    # Phase 2: adaptive attack (freeze clf, optimize head)
    print("  Phase 2: adaptive attack...")
    out_path = OUT_ROOT / f"history_{tag}.json"
    history, head = train_adaptive(
        head, clf, x_ct, x_cnt, x_p,
        epochs=epochs, lr=lr, lambda_r=lambda_r,
        eval_every=20, out_path=out_path,
    )

    torch.save({"head": head.state_dict(), "clf": clf.state_dict()}, out_weights)

    elapsed = time.time() - t0
    base    = history[0]
    final   = history[-1]
    print(f"\n  Final: Z={final['auc_z']:.4f}  XYZ={final['auc_xyz']:.4f}  "
          f"ASR={final['asr']:.4f}  CA={final['ca']:.4f}  ({elapsed:.0f}s)")

    result = {
        "seed": seed, "layer": layer, "pair": pair, "pr": pr,
        "lambda_r": lambda_r, "epochs": epochs, "lr": lr,
        "n_qubits": n_qubits, "hidden": hidden,
        "constrained": constrained, "input_dim": input_dim,
        "baseline": base, "final": final,
        "history": history, "elapsed_s": round(elapsed),
        "weights_file": out_weights.name,
    }
    with open(out_summary, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → {out_summary.name}  +  {out_weights.name}")
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="E1: Classical Bloch-ball matched control")
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--layer",        type=int,   default=4)
    ap.add_argument("--pair",         default="t7_vs_t0")
    ap.add_argument("--pr",           type=float, default=0.1)
    ap.add_argument("--lambda_r",     type=float, default=1.0)
    ap.add_argument("--epochs",       type=int,   default=200)
    ap.add_argument("--lr",           type=float, default=0.01)
    ap.add_argument("--hidden",       type=int,   default=64)
    ap.add_argument("--unconstrained", action="store_true",
                    help="Run unconstrained ablation instead of Bloch-ball control")
    ap.add_argument("--mode",    choices=["single", "sweep", "full"], default="single")
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--layers",  nargs="+", type=int,   default=[4, 8, 12])
    ap.add_argument("--dataset", choices=["mnist", "bloodmnist"], default="mnist")
    args = ap.parse_args()

    global DATA_ROOT, OUT_ROOT
    constrained = not args.unconstrained
    n_qubits    = 5

    if args.dataset == "bloodmnist":
        DATA_ROOT = ROOT / "data/MedMnist/single_detect_20260411/epsilon_0.8"
        suffix    = "unconstrained" if not constrained else "bloch"
        OUT_ROOT  = ROOT / f"results/classical_{suffix}_bloodmnist"
        n_qubits  = 8
        if args.pair == "t7_vs_t0":
            args.pair = "t6_vs_t0"
        if args.layers == [4, 8, 12]:
            args.layers = [8]
    else:
        suffix   = "unconstrained" if not constrained else "bloch"
        OUT_ROOT = ROOT / f"results/classical_{suffix}"

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"DATA_ROOT  : {DATA_ROOT}")
    print(f"OUT_ROOT   : {OUT_ROOT}")
    print(f"Constrained: {constrained}")

    if args.mode == "single":
        run_one(args.seed, args.layer, args.pair, args.pr,
                args.lambda_r, args.epochs, args.lr,
                n_qubits, args.hidden, constrained)

    elif args.mode == "sweep":
        rows = []
        for lam in args.lambdas:
            r = run_one(args.seed, args.layer, args.pair, args.pr,
                        lam, args.epochs, args.lr,
                        n_qubits, args.hidden, constrained)
            rows.append(r)
        print(f"\n{'='*58}")
        print(f"{'lambda':>7}  {'Z_base':>7} {'XYZ_base':>9}  "
              f"{'Z_final':>8} {'XYZ_final':>10}")
        for r in rows:
            b, f = r["baseline"], r["final"]
            print(f"  {r['lambda_r']:>5}  {b['auc_z']:>7.4f} {b['auc_xyz']:>9.4f}  "
                  f"{f['auc_z']:>8.4f} {f['auc_xyz']:>10.4f}")

    elif args.mode == "full":
        for layer in args.layers:
            for lam in args.lambdas:
                run_one(args.seed, layer, args.pair, args.pr,
                        lam, args.epochs, args.lr,
                        n_qubits, args.hidden, constrained)


if __name__ == "__main__":
    main()

