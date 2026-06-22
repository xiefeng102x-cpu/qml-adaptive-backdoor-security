"""
train_adaptive_qnn_fast.py  (Linux server version)
===================================================
Training-time adaptive backdoor attack against XYZ Pauli detection.
Uses fast_circuit.py with manual parameter-shift — no PennyLane, no GPU needed.

Server directory layout assumed:
  <repo_root>/
    training/
      train_adaptive_qnn_fast.py   <- this file
    fast_circuit.py                <- shared circuit engine (repo root)
    results/
      full_grid_mnist/             <- MNIST model .pt files
      full_grid_bloodmnist/          <- BloodMNIST model .pt files
      training_time_mnist/           <- output (auto-created)
      training_time_bloodmnist/      <- output (auto-created)

Usage:
  # Single config
  python train_adaptive_qnn_fast.py --seed 42 --layer 4 --lambda_r 1.0

  # Full grid per seed (run in parallel, one process per seed)
  for seed in 42 43 44 45 46; do
    nohup python train_adaptive_qnn_fast.py --seed $seed --mode full \
      > logs/mnist_seed${seed}.log 2>&1 &
  done

  # BloodMNIST
  for seed in 42 43 44 45 46; do
    nohup python train_adaptive_qnn_fast.py --seed $seed --mode full \
      --dataset bloodmnist > logs/blood_seed${seed}.log 2>&1 &
  done
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

# ── paths ─────────────────────────────────────────────────────────────────────
# Walk up until we find the directory that contains BOTH experiments/ and
_HERE    = Path(__file__).resolve()
_REPO    = _HERE.parent.parent   # training/ -> repo root

DATA_ROOT: Path = Path(".")   # overridden in main()
OUT_ROOT:  Path = Path(".")   # overridden in main()

# fast_circuit.py must be placed in the same directory as this script
sys.path.insert(0, str(_HERE.parent.parent))  # repo root, where fast_circuit.py lives

from fast_circuit import run_circuit_batch, _make_evals


# ── gate helpers (inline, no PennyLane) ──────────────────────────────────────

_H_GATE  = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
_HS_GATE = np.array([[1, -1j], [1, 1j]], dtype=np.complex128) / np.sqrt(2)


def _apply_1q(psi: np.ndarray, gate: np.ndarray, q: int) -> np.ndarray:
    dim = psi.shape[1]; step = dim >> (q + 1); groups = dim // (2 * step)
    return np.einsum("ij,bkjl->bkil", gate,
                     psi.reshape(-1, groups, 2, step),
                     optimize=True).reshape(-1, dim)


def _apply_cnot(psi: np.ndarray, ctrl: int, tgt: int) -> np.ndarray:
    dim = psi.shape[1]; n = int(np.log2(dim))
    cb, tb = n - 1 - ctrl, n - 1 - tgt
    states  = np.arange(dim, dtype=np.int64)
    ctrl_on = (states >> cb) & 1
    return psi[:, np.where(ctrl_on, states ^ (1 << tb), states)]


def _run_circuit_get_psi(samples: np.ndarray, w: np.ndarray,
                         n_layers: int, n_qubits: int) -> np.ndarray:
    """Run the QNN circuit and return the final state vector [N, 2^Q]."""
    dim = 2 ** n_qubits
    norms = np.linalg.norm(samples, axis=1, keepdims=True).clip(min=1e-12)
    psi   = (samples / norms).astype(np.complex128)
    wr    = w.reshape(n_layers, n_qubits, 2)
    for l in range(n_layers):
        for q in range(n_qubits):
            ct, st = np.cos(wr[l,q,0]/2), np.sin(wr[l,q,0]/2)
            psi = _apply_1q(psi, np.array([[ct,-st],[st,ct]], dtype=np.complex128), q)
            ep  = np.exp(1j * wr[l,q,1] / 2)
            psi = _apply_1q(psi, np.array([[1/ep,0],[0,ep]], dtype=np.complex128), q)
        for q in range(n_qubits):
            psi = _apply_cnot(psi, q, (q + 1) % n_qubits)
    return psi


# ── feature helpers ───────────────────────────────────────────────────────────

def z_feats(samples: np.ndarray, w: np.ndarray, n_layers: int,
            n_qubits: int = 5) -> np.ndarray:
    """Z-only features [N, Q]. Uses fast_circuit (fastest path)."""
    out = run_circuit_batch(samples.astype(np.float64), w, n_layers, n_qubits)
    return out[:, :n_qubits].astype(np.float64)


def xyz_feats(samples: np.ndarray, w: np.ndarray, n_layers: int,
              n_qubits: int = 5) -> np.ndarray:
    """
    Single-qubit XYZ features [N, 3Q] in order [X..., Y..., Z...].
    Fast path: runs circuit once, applies basis rotations per qubit.
    """
    z_ev, _, _ = _make_evals(n_qubits)
    psi   = _run_circuit_get_psi(samples.astype(np.float64), w, n_layers, n_qubits)
    probs = np.abs(psi) ** 2
    z_vals = (probs @ z_ev.T).astype(np.float64)                  # [N, Q]
    x_vals = np.stack([
        (np.abs(_apply_1q(psi, _H_GATE, q)) ** 2 @ z_ev[q])
        for q in range(n_qubits)], axis=1).astype(np.float64)     # [N, Q]
    y_vals = np.stack([
        (np.abs(_apply_1q(psi, _HS_GATE, q)) ** 2 @ z_ev[q])
        for q in range(n_qubits)], axis=1).astype(np.float64)     # [N, Q]
    return np.concatenate([x_vals, y_vals, z_vals], axis=1)       # [N, 3Q]


def zxyz_feats(samples: np.ndarray, w: np.ndarray, n_layers: int,
               n_qubits: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return Z [N,Q] and XYZ [N,3Q] in a single circuit pass."""
    z_ev, _, _ = _make_evals(n_qubits)
    psi   = _run_circuit_get_psi(samples.astype(np.float64), w, n_layers, n_qubits)
    probs = np.abs(psi) ** 2
    z_vals = (probs @ z_ev.T).astype(np.float64)
    x_vals = np.stack([
        (np.abs(_apply_1q(psi, _H_GATE, q)) ** 2 @ z_ev[q])
        for q in range(n_qubits)], axis=1).astype(np.float64)
    y_vals = np.stack([
        (np.abs(_apply_1q(psi, _HS_GATE, q)) ** 2 @ z_ev[q])
        for q in range(n_qubits)], axis=1).astype(np.float64)
    xyz = np.concatenate([x_vals, y_vals, z_vals], axis=1)
    return z_vals, xyz


# ── classifier (fixed, loaded from paper2) ────────────────────────────────────

class FixedClassifier:
    """Linear(Q->32)->ReLU->Linear(32->2), weights fixed from paper2."""
    def __init__(self, w0: np.ndarray, b0: np.ndarray,
                 w3: np.ndarray, b3: np.ndarray):
        self.w0, self.b0 = w0, b0   # [32, Q], [32]
        self.w3, self.b3 = w3, b3   # [2, 32], [2]

    def forward(self, z: np.ndarray) -> np.ndarray:
        """z: [N, Q] -> logits: [N, 2]"""
        h = np.maximum(0, z @ self.w0.T + self.b0)   # ReLU
        return h @ self.w3.T + self.b3

    def ce_loss(self, z: np.ndarray, labels: np.ndarray) -> float:
        logits = self.forward(z)
        logits -= logits.max(1, keepdims=True)
        log_sm  = logits - np.log(np.exp(logits).sum(1, keepdims=True))
        return float(-log_sm[np.arange(len(labels)), labels].mean())


# ── Mahalanobis loss ──────────────────────────────────────────────────────────

def mahal_loss(xyz: np.ndarray, mu: np.ndarray, prec: np.ndarray) -> float:
    diff = xyz - mu
    return float(np.einsum("ni,ij,nj->n", diff, prec, diff).mean())


# ── parameter-shift gradient ──────────────────────────────────────────────────

def compute_grad(weights: np.ndarray,
                 x_batch: np.ndarray, y_batch: np.ndarray,
                 x_poison: np.ndarray,
                 clf: FixedClassifier,
                 mu_xyz: np.ndarray, prec_xyz: np.ndarray,
                 lambda_r: float, n_layers: int, n_qubits: int,
                 shift: float = np.pi / 2) -> np.ndarray:
    """
    Combined parameter-shift gradient:
      d(CE + lambda * Mahal_XYZ) / d(theta_k)  for each weight k.
    """
    n_params = len(weights)
    grad     = np.zeros(n_params)

    for k in range(n_params):
        w_p = weights.copy(); w_p[k] += shift
        w_m = weights.copy(); w_m[k] -= shift

        # CE gradient (uses Z features on training batch)
        ce_p = clf.ce_loss(z_feats(x_batch, w_p, n_layers, n_qubits), y_batch)
        ce_m = clf.ce_loss(z_feats(x_batch, w_m, n_layers, n_qubits), y_batch)
        grad_ce = (ce_p - ce_m) / 2.0

        # Repr-matching gradient (uses XYZ features on poisoned samples)
        if lambda_r > 0 and len(x_poison) > 0:
            xyz_p = xyz_feats(x_poison, w_p, n_layers, n_qubits)
            xyz_m = xyz_feats(x_poison, w_m, n_layers, n_qubits)
            grad_r = (mahal_loss(xyz_p, mu_xyz, prec_xyz)
                    - mahal_loss(xyz_m, mu_xyz, prec_xyz)) / 2.0
        else:
            grad_r = 0.0

        grad[k] = grad_ce + lambda_r * grad_r

    return grad


# ── detection AUC helper ──────────────────────────────────────────────────────

def detection_auc(fc: np.ndarray, fp: np.ndarray, n: int = 300) -> float:
    rng = np.random.default_rng(0)
    nc  = min(len(fc), n);  np_ = min(len(fp), n)
    ic  = rng.choice(len(fc), nc, replace=False)
    ip  = rng.choice(len(fp), np_, replace=False)
    fc2, fp2 = fc[ic], fp[ip];  nn_ = min(nc, np_)
    fc2, fp2 = fc2[:nn_], fp2[:nn_]
    lw    = LedoitWolf().fit(fc2)
    diff  = np.vstack([fc2, fp2]) - lw.location_
    scores = np.einsum("ni,ij,nj->n", diff, lw.get_precision(), diff)
    return float(roc_auc_score(np.array([0]*nn_+[1]*nn_), scores))


def evaluate(weights: np.ndarray, x_clean: np.ndarray, x_clean_nt: np.ndarray,
             x_poison: np.ndarray, clf: FixedClassifier,
             mu_xyz: np.ndarray, prec_xyz: np.ndarray,
             n_layers: int, n_qubits: int) -> dict:
    z_c,  xyz_c = zxyz_feats(x_clean,   weights, n_layers, n_qubits)
    z_p,  xyz_p = zxyz_feats(x_poison,  weights, n_layers, n_qubits)
    z_cnt        = z_feats(x_clean_nt,  weights, n_layers, n_qubits)
    # ASR: fraction of poisoned samples classified as target (class 1)
    asr = float(np.mean(np.argmax(clf.forward(z_p),   axis=1) == 1))
    # CA: overall clean accuracy on target + non-target
    correct = (np.sum(np.argmax(clf.forward(z_c),   axis=1) == 1) +
               np.sum(np.argmax(clf.forward(z_cnt), axis=1) == 0))
    ca = float(correct / (len(z_c) + len(z_cnt)))
    mahal_p = round(float(mahal_loss(xyz_p, mu_xyz, prec_xyz)), 4)
    return {
        "auc_z":        round(detection_auc(z_c,  z_p),   4),
        "auc_xyz":      round(detection_auc(xyz_c, xyz_p), 4),
        "asr":          round(asr, 4),
        "ca":           round(ca,  4),
        "mahal_poison": mahal_p,
    }


# ── data loading ──────────────────────────────────────────────────────────────

def pr_token(pr: float) -> str:
    return f"{float(pr):.12g}"


def load_config(seed: int, layer: int, pair: str, pr: float):
    import torch
    pr_s  = pr_token(pr)
    base  = DATA_ROOT / f"seed_{seed}" / f"layer_{layer}_grid" / pair / f"pr_{pr_s}"
    meta  = torch.load(base / f"poisoned_samples_meta_seed_{seed}_pr_{pr_s}.pt",
                       map_location="cpu", weights_only=False)
    # MNIST:      clean_model_seed_{seed}_epoch_200.pt
    # BloodMNIST: clean_model_seed_200.pt
    model_path = base / "models" / f"clean_model_seed_{seed}_epoch_200.pt"
    if not model_path.exists():
        model_path = base / "models" / "clean_model_seed_200.pt"
    state = torch.load(model_path, map_location="cpu", weights_only=False)

    weights = state["weights"].detach().numpy().copy().astype(np.float64)
    clf = FixedClassifier(
        w0=state["classifier.0.weight"].numpy(),
        b0=state["classifier.0.bias"].numpy(),
        w3=state["classifier.3.weight"].numpy(),
        b3=state["classifier.3.bias"].numpy(),
    )

    x_clean_target    = meta["target_clean_data"].numpy().astype(np.float64)
    x_clean_nontarget = meta["non_target_clean_data"].numpy().astype(np.float64)
    x_poison          = meta["poisoned_non_target_data"].numpy().astype(np.float64)

    return weights, clf, x_clean_target, x_clean_nontarget, x_poison


# ── training loop ─────────────────────────────────────────────────────────────

def train(weights_init: np.ndarray,
          clf: FixedClassifier,
          x_clean_target: np.ndarray,
          x_clean_nontarget: np.ndarray,
          x_poison: np.ndarray,
          n_layers: int, n_qubits: int,
          epochs: int, lr: float, lambda_r: float,
          batch_size: int = 32,
          eval_every: int = 20,
          out_path: Path | None = None,
          on_checkpoint=None) -> tuple[list, np.ndarray]:

    weights = weights_init.copy()
    rng     = np.random.default_rng(42)

    # Build training set
    x_tr = np.vstack([x_clean_target, x_clean_nontarget, x_poison])
    y_tr = np.concatenate([
        np.ones(len(x_clean_target),    dtype=int),   # target class = 1
        np.zeros(len(x_clean_nontarget),dtype=int),   # non-target   = 0
        np.ones(len(x_poison),          dtype=int),   # backdoor     = 1
    ])

    # Fit reference XYZ distribution on clean target (fixed throughout)
    xyz_clean_ref = xyz_feats(x_clean_target, weights, n_layers, n_qubits)
    lw_ref        = LedoitWolf().fit(xyz_clean_ref)
    mu_xyz        = lw_ref.location_
    prec_xyz      = lw_ref.get_precision()

    history = []

    # Baseline
    det = evaluate(weights, x_clean_target, x_clean_nontarget, x_poison,
                   clf, mu_xyz, prec_xyz, n_layers, n_qubits)
    det["epoch"] = 0
    history.append(det)
    print(f"  Baseline : Z={det['auc_z']:.4f}  XYZ={det['auc_xyz']:.4f}  "
          f"ASR={det['asr']:.4f}  CA={det['ca']:.4f}  Mahal={det['mahal_poison']:.2f}")

    for epoch in range(1, epochs + 1):
        idx = rng.permutation(len(x_tr))
        for start in range(0, len(x_tr), batch_size):
            bi   = idx[start:start + batch_size]
            xb   = x_tr[bi]
            yb   = y_tr[bi]
            grad = compute_grad(weights, xb, yb, x_poison,
                                clf, mu_xyz, prec_xyz,
                                lambda_r, n_layers, n_qubits)
            weights -= lr * grad

        if epoch % eval_every == 0:
            det = evaluate(weights, x_clean_target, x_clean_nontarget, x_poison,
                           clf, mu_xyz, prec_xyz, n_layers, n_qubits)
            det["epoch"] = epoch
            history.append(det)
            print(f"  Epoch {epoch:4d} : Z={det['auc_z']:.4f}  XYZ={det['auc_xyz']:.4f}  "
                  f"ASR={det['asr']:.4f}  CA={det['ca']:.4f}  Mahal={det['mahal_poison']:.2f}  "
                  f"Delta={det['auc_xyz']-det['auc_z']:+.4f}")

            if out_path:
                _save_incremental(out_path, history)
            if on_checkpoint:
                on_checkpoint(history)

    return history, weights


def _save_incremental(path: Path, history: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


# ── main ──────────────────────────────────────────────────────────────────────

def run_one(seed: int, layer: int, pair: str, pr: float,
            lambda_r: float, epochs: int, lr: float,
            n_qubits: int = 5) -> dict:

    tag         = f"seed={seed}_L{layer}_lr{lambda_r}"
    out_summary = OUT_ROOT / f"result_{tag}.json"
    out_weights = OUT_ROOT / f"weights_{tag}.npy"

    # Skip only when BOTH result JSON and weights file exist.
    # If result exists but weights are missing (older run), re-train to save them.
    if out_summary.exists() and out_weights.exists():
        print(f"\n  Skipping {tag} (result + weights already exist)")
        with open(out_summary) as f:
            return json.load(f)

    if out_summary.exists() and not out_weights.exists():
        print(f"\n  Re-running {tag} to save weights (result exists, weights missing)")

    print(f"\n{'='*60}")
    print(f"Training-time adaptive | {tag}")
    t0 = time.time()

    weights, clf, x_ct, x_cnt, x_p = load_config(seed, layer, pair, pr)

    # Partial result written after each eval checkpoint so progress is visible.
    meta = {
        "seed": seed, "layer": layer, "pair": pair, "pr": pr,
        "lambda_r": lambda_r, "epochs": epochs, "lr": lr,
    }

    def _on_checkpoint(history: list) -> None:
        partial = {
            **meta,
            "status":        "running",
            "current_epoch": history[-1]["epoch"],
            "elapsed_s":     round(time.time() - t0),
            "baseline":      history[0],
            "latest":        history[-1],
            "history":       history,
        }
        with open(out_summary, "w") as f:
            json.dump(partial, f, indent=2)

    out_path = OUT_ROOT / f"history_{tag}.json"
    history, w_final = train(
        weights, clf, x_ct, x_cnt, x_p,
        n_layers=layer, n_qubits=n_qubits,
        epochs=epochs, lr=lr, lambda_r=lambda_r,
        eval_every=20, out_path=out_path,
        on_checkpoint=_on_checkpoint,
    )

    # Save final adapted weights for downstream E3 Pauli ladder audit
    np.save(out_weights, w_final)

    elapsed = time.time() - t0
    base    = history[0]
    final   = history[-1]
    print(f"\n  Final: Z={final['auc_z']:.4f}  XYZ={final['auc_xyz']:.4f}  "
          f"ASR={final['asr']:.4f}  CA={final['ca']:.4f}  ({elapsed:.0f}s)")

    result = {
        **meta,
        "status":       "done",
        "baseline":     base,
        "final":        final,
        "history":      history,
        "elapsed_s":    round(elapsed),
        "weights_file": out_weights.name,
    }
    with open(out_summary, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved -> {out_summary.name}  +  {out_weights.name}")
    return result


def main():
    ap = argparse.ArgumentParser(
        description="Training-time adaptive QNN attack (fast_circuit, CPU-ready)")
    ap.add_argument("--seed",     type=int,   default=42)
    ap.add_argument("--layer",    type=int,   default=4)
    ap.add_argument("--pair",     default="t7_vs_t0")
    ap.add_argument("--pr",       type=float, default=0.1)
    ap.add_argument("--lambda_r", type=float, default=1.0,
                    help="Repr-matching loss weight")
    ap.add_argument("--epochs",   type=int,   default=200)
    ap.add_argument("--lr",       type=float, default=0.01)
    ap.add_argument("--mode",     choices=["single", "sweep", "full"],
                    default="single",
                    help="single: one config | sweep: lambda sweep | full: all layers")
    ap.add_argument("--lambdas",  nargs="+",  type=float,
                    default=[0.0, 0.5, 1.0, 2.0],
                    help="Lambda values for sweep/full mode")
    ap.add_argument("--layers",   nargs="+",  type=int,
                    default=[4, 8, 12])
    ap.add_argument("--dataset",  choices=["mnist", "bloodmnist"],
                    default="mnist")
    ap.add_argument("--data-root", type=Path, default=None,
                    help="Override DATA_ROOT (local testing; server uses auto-detect)")
    ap.add_argument("--out-root",  type=Path, default=None,
                    help="Override OUT_ROOT  (local testing; server uses auto-detect)")
    args = ap.parse_args()

    global DATA_ROOT, OUT_ROOT
    if args.dataset == "bloodmnist":
        DATA_ROOT = args.data_root if args.data_root else _REPO / "results/baseline_bloodmnist"
        OUT_ROOT  = args.out_root  if args.out_root  else _REPO / "results/training_time_bloodmnist"
        n_qubits  = 8
        if args.pair == "t7_vs_t0":
            args.pair = "t6_vs_t0"
        if args.layers == [4, 8, 12]:
            args.layers = [8]
    else:
        DATA_ROOT = args.data_root if args.data_root else _REPO / "results/baseline_mnist"
        OUT_ROOT  = args.out_root  if args.out_root  else _REPO / "results/training_time_mnist"
        n_qubits  = 5

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"DATA_ROOT : {DATA_ROOT}")
    print(f"OUT_ROOT  : {OUT_ROOT}")

    if args.mode == "single":
        run_one(args.seed, args.layer, args.pair, args.pr,
                args.lambda_r, args.epochs, args.lr, n_qubits)

    elif args.mode == "sweep":
        all_rows = []
        for lam in args.lambdas:
            row = run_one(args.seed, args.layer, args.pair, args.pr,
                          lam, args.epochs, args.lr, n_qubits)
            all_rows.append(row)

        print(f"\n{'='*58}")
        print(f"{'lambda':>7}  {'Z_base':>7} {'XYZ_base':>9}  "
              f"{'Z_final':>8} {'XYZ_final':>10}")
        for r in all_rows:
            b, f = r["baseline"], r["final"]
            print(f"  {r['lambda_r']:>5}  {b['auc_z']:>7.4f} {b['auc_xyz']:>9.4f}  "
                  f"{f['auc_z']:>8.4f} {f['auc_xyz']:>10.4f}")

        out = OUT_ROOT / f"sweep_s{args.seed}_L{args.layer}.json"
        with open(out, "w") as f:
            json.dump(all_rows, f, indent=2)
        print(f"Saved -> {out}")

    elif args.mode == "full":
        all_rows = []
        for layer in args.layers:
            for lam in args.lambdas:
                row = run_one(args.seed, layer, args.pair, args.pr,
                              lam, args.epochs, args.lr, n_qubits)
                all_rows.append(row)

        out = OUT_ROOT / f"full_s{args.seed}.json"
        with open(out, "w") as f:
            json.dump(all_rows, f, indent=2)
        print(f"\nFull results saved -> {out}")


if __name__ == "__main__":
    main()

