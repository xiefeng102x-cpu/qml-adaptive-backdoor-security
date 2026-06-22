"""
e2_nullspace_attack.py
======================
E2 experiment: Null-space projection attack on the QML XYZ detector.

For each poison sample, computes:
  1. J_M = d(XYZ_features) / d(input)   [vectorized numerical Jacobian]
  2. rank(J_M), hidden_ratio = dim(ker J_M) / D
  3. alignment  = ||Proj_{ker J_M}(g)|| / ||g||   where g = grad(target logit)
  4. Projection attack: T gradient ascent steps on target logit within ker J_M
  5. Random-direction control: same budget, random subspace of same dimension

No model retraining. Uses existing λ=0.0 (baseline) and λ=1.0 (adaptive) weights.

Usage:
  python e2_nullspace_attack.py                     # MNIST L=12, all seeds
  python e2_nullspace_attack.py --condition baseline
  python e2_nullspace_attack.py --condition adaptive

Output:
  results/e2_nullspace/e2_raw.csv      per-sample rows
  results/e2_nullspace/e2_summary.csv  per-seed aggregates
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

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

# ── gate helpers ───────────────────────────────────────────────────────────────
_H_GATE  = np.array([[1,  1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
_HS_GATE = np.array([[1, -1j], [1, 1j]], dtype=np.complex128) / np.sqrt(2)


def _apply_1q(psi: np.ndarray, gate: np.ndarray, q: int) -> np.ndarray:
    dim = psi.shape[1]; step = dim >> (q + 1); groups = dim // (2 * step)
    return np.einsum("ij,bkjl->bkil", gate,
                     psi.reshape(-1, groups, 2, step), optimize=True).reshape(-1, dim)


def _apply_cnot(psi: np.ndarray, ctrl: int, tgt: int) -> np.ndarray:
    dim = psi.shape[1]; n = int(np.log2(dim))
    cb, tb = n - 1 - ctrl, n - 1 - tgt
    states = np.arange(dim, dtype=np.int64)
    return psi[:, np.where((states >> cb) & 1, states ^ (1 << tb), states)]


def _run_psi(samples: np.ndarray, weights: np.ndarray,
             n_layers: int, n_qubits: int) -> np.ndarray:
    norms = np.linalg.norm(samples, axis=1, keepdims=True).clip(min=1e-12)
    psi = (samples / norms).astype(np.complex128)
    wr = weights.reshape(n_layers, n_qubits, 2)
    for l in range(n_layers):
        for q in range(n_qubits):
            ct, st = np.cos(wr[l, q, 0] / 2), np.sin(wr[l, q, 0] / 2)
            psi = _apply_1q(psi, np.array([[ct, -st], [st, ct]], dtype=np.complex128), q)
            ep = np.exp(1j * wr[l, q, 1] / 2)
            psi = _apply_1q(psi, np.array([[1/ep, 0], [0, ep]], dtype=np.complex128), q)
        for q in range(n_qubits):
            psi = _apply_cnot(psi, q, (q + 1) % n_qubits)
    return psi


# ── feature extraction ─────────────────────────────────────────────────────────

def _extract_xyz_raw(samples: np.ndarray, weights: np.ndarray,
                     n_layers: int, n_qubits: int) -> np.ndarray:
    """Returns (N, 3*n_qubits) XYZ feature matrix."""
    psi   = _run_psi(samples.astype(np.float64), weights, n_layers, n_qubits)
    probs = np.abs(psi) ** 2
    z_ev, _, _ = _make_evals(n_qubits)
    Z = (probs @ z_ev.T).astype(np.float64)
    X = np.stack([(np.abs(_apply_1q(psi, _H_GATE,  q)) ** 2 @ z_ev[q])
                  for q in range(n_qubits)], axis=1).astype(np.float64)
    Y = np.stack([(np.abs(_apply_1q(psi, _HS_GATE, q)) ** 2 @ z_ev[q])
                  for q in range(n_qubits)], axis=1).astype(np.float64)
    return np.hstack([X, Y, Z])


def _target_logit_raw(samples: np.ndarray, weights: np.ndarray,
                      n_layers: int, n_qubits: int) -> np.ndarray:
    """
    Target logit proxy: mean ⟨Z_q⟩ across all qubits.
    For binary QNN (class 7 vs 0): higher = more like target class.
    """
    psi   = _run_psi(samples.astype(np.float64), weights, n_layers, n_qubits)
    probs = np.abs(psi) ** 2
    z_ev, _, _ = _make_evals(n_qubits)
    Z = probs @ z_ev.T   # (N, n_qubits)
    return Z.mean(axis=1)  # (N,)


# ── Jacobian & gradient (vectorized single-batch) ──────────────────────────────

def compute_jm_and_gradient(sample: np.ndarray, weights: np.ndarray,
                             n_layers: int, n_qubits: int,
                             eps: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """
    Single batched forward pass → J_M and target logit gradient.

    Returns:
        J_M      (3*n_qubits, D)  feature Jacobian w.r.t. raw input
        g_logit  (D,)             target logit gradient w.r.t. raw input
    """
    D = len(sample)
    # Row 0 = base; rows 1..D = one dimension perturbed each
    X_batch = np.tile(sample, (D + 1, 1))
    for i in range(D):
        X_batch[i + 1, i] += eps

    xyz_all   = _extract_xyz_raw(X_batch, weights, n_layers, n_qubits)   # (D+1, 3n)
    logit_all = _target_logit_raw(X_batch, weights, n_layers, n_qubits)  # (D+1,)

    J_M     = (xyz_all[1:] - xyz_all[0]).T / eps   # (3n, D)
    g_logit = (logit_all[1:] - logit_all[0]) / eps  # (D,)
    return J_M, g_logit


# ── null space projector ───────────────────────────────────────────────────────

def null_projector(J_M: np.ndarray,
                   svd_thresh: float = 1e-6) -> tuple[np.ndarray, int, np.ndarray]:
    """
    SVD-based projection onto ker(J_M).

    Returns:
        P_null  (D, D)  projection matrix  = I − J^+ J
        rank    int     numerical rank of J_M
        S       (min(m,D),)  singular values (descending)
    """
    _, S, Vt = np.linalg.svd(J_M, full_matrices=True)
    rank    = int(np.sum(S > svd_thresh))
    V_null  = Vt[rank:].T          # (D, D-rank) — basis for ker(J_M)
    P_null  = V_null @ V_null.T    # (D, D)
    return P_null, rank, S


# ── attack runner ──────────────────────────────────────────────────────────────

def _run_attack(sample: np.ndarray, weights: np.ndarray,
                n_layers: int, n_qubits: int,
                P_proj: np.ndarray,
                n_steps: int, alpha: float, eps: float) -> tuple[np.ndarray, float, float]:
    """
    Generic projected gradient ascent on target logit.
    P_proj: fixed projection matrix (either P_null or P_rand)

    Returns:
        x_adv        final adversarial sample
        logit_delta  logit_final − logit_initial
        xyz_drift    ||XYZ(x_adv) − XYZ(x_0)||₂
    """
    x       = sample.copy()
    xyz_0   = _extract_xyz_raw(sample[None], weights, n_layers, n_qubits)[0]
    D = len(x)

    for _ in range(n_steps):
        # Recompute target logit gradient at current x
        X_pert    = np.tile(x, (D + 1, 1))
        for i in range(D):
            X_pert[i + 1, i] += eps
        logit_all = _target_logit_raw(X_pert, weights, n_layers, n_qubits)
        g         = (logit_all[1:] - logit_all[0]) / eps   # (D,)

        # Project and step
        v      = P_proj @ g
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-8:   # threshold well above machine precision (~1e-12)
            break
        x = x + alpha * (v / v_norm)

    xyz_final   = _extract_xyz_raw(x[None], weights, n_layers, n_qubits)[0]
    logit_init  = _target_logit_raw(sample[None], weights, n_layers, n_qubits)[0]
    logit_final = _target_logit_raw(x[None], weights, n_layers, n_qubits)[0]

    return x, float(logit_final - logit_init), float(np.linalg.norm(xyz_final - xyz_0))


# ── Mahalanobis helper ─────────────────────────────────────────────────────────

def _fit_mahal(feat_clean: np.ndarray):
    lw   = LedoitWolf().fit(feat_clean)
    prec = lw.get_precision()
    mu   = lw.location_

    def score(feat: np.ndarray) -> np.ndarray:
        d = feat - mu
        return np.einsum("ni,ij,nj->n", d, prec, d)

    return score, float(lw.shrinkage_)


# ── per-seed runner ────────────────────────────────────────────────────────────

def run_seed(seed: int, layer: int, n_qubits: int, pair: str,
             data_root: Path, weights_npy: Path,
             condition: str,
             n_steps: int = 100,
             step_frac: float = 0.02,
             eps: float = 1e-4,
             svd_thresh: float = 1e-6) -> list[dict]:
    """
    Run E2 analysis for one (seed, condition).
    Returns list of per-sample result dicts.
    """
    import torch

    # Load data
    pr_s  = "0.1"
    base  = data_root / f"seed_{seed}" / f"layer_{layer}_grid" / pair / f"pr_{pr_s}"
    meta  = torch.load(base / f"poisoned_samples_meta_seed_{seed}_pr_{pr_s}.pt",
                       map_location="cpu", weights_only=False)
    x_clean  = meta["target_clean_data"].numpy().astype(np.float64)
    x_poison = meta["poisoned_non_target_data"].numpy().astype(np.float64)

    weights = np.load(weights_npy).astype(np.float64)

    D = x_poison.shape[1]

    # Fit Mahalanobis on clean XYZ features
    xyz_clean  = _extract_xyz_raw(x_clean, weights, layer, n_qubits)
    mahal_fn, shrinkage = _fit_mahal(xyz_clean)
    thresh_95  = float(np.percentile(mahal_fn(xyz_clean), 95))

    # Per-sample loop
    rows = []
    rng  = np.random.default_rng(seed + 1000)

    for i, sample in enumerate(x_poison):
        alpha = step_frac * float(np.linalg.norm(sample))

        # ── 1. Structural analysis ──────────────────────────────────────────
        J_M, g_logit = compute_jm_and_gradient(sample, weights, layer, n_qubits, eps)
        P_null, rank, S = null_projector(J_M, svd_thresh)

        null_dim        = D - rank
        # r_dim: structural null-space fraction, does NOT depend on g
        r_dim           = null_dim / D
        g_norm          = float(np.linalg.norm(g_logit))
        v_proj          = P_null @ g_logit                       # Proj_{ker J_M}(g)
        v_proj_norm     = float(np.linalg.norm(v_proj))          # ||P_null @ g||
        # r_hidden: theoretical gradient hidden ratio (user definition)
        r_hidden        = v_proj_norm / (g_norm + 1e-30)
        # projection_residual: ||J_M @ v|| / (||J_M||_F * ||v|| + eps)
        # verifies v actually lies in ker(J_M)
        J_norm          = float(np.linalg.norm(J_M))
        if v_proj_norm > 1e-30:
            proj_residual = float(np.linalg.norm(J_M @ v_proj)) / (J_norm * v_proj_norm + 1e-30)
        else:
            proj_residual = 0.0
        S_min_nonzero   = float(S[rank - 1]) if rank > 0 else 0.0
        S_max           = float(S[0])

        # ── 2. Baseline Mahalanobis ─────────────────────────────────────────
        xyz_base  = _extract_xyz_raw(sample[None], weights, layer, n_qubits)[0]
        mahal_0   = float(mahal_fn(xyz_base[None])[0])
        logit_0   = float(_target_logit_raw(sample[None], weights, layer, n_qubits)[0])

        # ── 3. Projection attack (ker J_M) ──────────────────────────────────
        x_proj, dlogit_proj, xyz_drift_proj = _run_attack(
            sample, weights, layer, n_qubits, P_null, n_steps, alpha, eps)
        xyz_proj  = _extract_xyz_raw(x_proj[None], weights, layer, n_qubits)[0]
        mahal_proj = float(mahal_fn(xyz_proj[None])[0])

        # ── 4. Random-direction control (same null_dim) ──────────────────────
        Q, _ = np.linalg.qr(rng.standard_normal((D, max(null_dim, 1))))
        P_rand = Q @ Q.T
        x_rand, dlogit_rand, xyz_drift_rand = _run_attack(
            sample, weights, layer, n_qubits, P_rand, n_steps, alpha, eps)
        xyz_rand  = _extract_xyz_raw(x_rand[None], weights, layer, n_qubits)[0]
        mahal_rand = float(mahal_fn(xyz_rand[None])[0])

        rows.append(dict(
            seed=seed, condition=condition, sample_idx=i,
            D=D, n_qubits=n_qubits, layer=layer,
            shrinkage=shrinkage, thresh_95=thresh_95,
            # structural (no g)
            rank_JM=rank, null_dim=null_dim,
            r_dim=round(r_dim, 6),                     # (D-rank)/D — structural
            S_max=round(S_max, 6), S_min_nonzero=round(S_min_nonzero, 6),
            # gradient-based (contains g)
            g_norm=round(g_norm, 8),
            v_proj_norm=v_proj_norm,                   # ||P_null@g||
            r_hidden=r_hidden,                         # ||P_null@g||/||g|| — theoretical hidden ratio
            proj_residual=round(proj_residual, 8),     # ||J_M v||/(||J_M||*||v||) — verify ker quality
            # baseline
            mahal_before=round(mahal_0, 4),
            logit_before=round(logit_0, 6),
            detected_before=int(mahal_0 > thresh_95),
            # projection attack
            logit_delta_proj=round(dlogit_proj, 6),
            xyz_drift_proj=round(xyz_drift_proj, 6),
            mahal_after_proj=round(mahal_proj, 4),
            detected_after_proj=int(mahal_proj > thresh_95),
            evades_proj=int(mahal_proj <= thresh_95),
            # random control
            logit_delta_rand=round(dlogit_rand, 6),
            xyz_drift_rand=round(xyz_drift_rand, 6),
            mahal_after_rand=round(mahal_rand, 4),
            detected_after_rand=int(mahal_rand > thresh_95),
            evades_rand=int(mahal_rand <= thresh_95),
        ))

    return rows


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["baseline", "adaptive", "both"],
                        default="both")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--step-frac", type=float, default=0.02)
    parser.add_argument("--eps", type=float, default=1e-4)
    args = parser.parse_args()

    OUT_DIR    = ROOT / "results/e2_nullspace"
    DATA_ROOT  = ROOT / "data/Mnist/detail_single_samle"
    ADAP_DIR   = ROOT / "results/training_time/new_version"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    N_QUBITS = 5
    PAIR     = "t7_vs_t0"

    conditions = ["baseline", "adaptive"] if args.condition == "both" else [args.condition]

    all_rows: list[dict] = []
    t0 = time.time()

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"  Condition: {condition}  L={args.layer}")
        print(f"{'='*60}")

        for seed in args.seeds:
            wgt = ADAP_DIR / f"weights_seed={seed}_L{args.layer}_lr{'0.0' if condition == 'baseline' else '1.0'}.npy"
            if not wgt.exists():
                print(f"  seed={seed}  SKIP (weights not found: {wgt.name})")
                continue

            print(f"  seed={seed}  n_steps={args.n_steps}  step_frac={args.step_frac}...",
                  end=" ", flush=True)
            t1 = time.time()
            rows = run_seed(seed, args.layer, N_QUBITS, PAIR, DATA_ROOT, wgt,
                            condition=condition,
                            n_steps=args.n_steps,
                            step_frac=args.step_frac,
                            eps=args.eps)
            dt = time.time() - t1
            all_rows.extend(rows)

            # Quick summary for this seed
            r_d  = np.mean([r["r_dim"]            for r in rows])
            r_h  = np.mean([r["r_hidden"]         for r in rows])
            vn   = np.mean([r["v_proj_norm"]       for r in rows])
            pr   = np.mean([r["proj_residual"]     for r in rows])
            dl_p = np.mean([r["logit_delta_proj"]  for r in rows])
            dr_p = np.mean([r["xyz_drift_proj"]    for r in rows])
            ev_p = np.mean([r["evades_proj"]        for r in rows])
            ev_r = np.mean([r["evades_rand"]        for r in rows])
            print(f"OK  ({dt:.1f}s)  "
                  f"r_dim={r_d:.4f}  r_hidden={r_h:.2e}  "
                  f"||P@g||={vn:.2e}  proj_res={pr:.2e}  "
                  f"Δlogit={dl_p:+.6f}  xyz_drift={dr_p:.6f}  "
                  f"JSER_proj={ev_p:.3f}  JSER_rand={ev_r:.3f}")

    if not all_rows:
        print("No rows collected.")
        return

    df = pd.DataFrame(all_rows)
    raw_path = OUT_DIR / "e2_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nRaw data → {raw_path}  ({len(df)} rows)")

    # ── Summary table ──────────────────────────────────────────────────────────
    cols = ["r_dim", "r_hidden", "v_proj_norm", "proj_residual", "rank_JM",
            "mahal_before", "mahal_after_proj", "mahal_after_rand",
            "logit_delta_proj", "logit_delta_rand",
            "xyz_drift_proj", "xyz_drift_rand",
            "evades_proj", "evades_rand", "detected_before"]
    grp  = df.groupby(["condition", "seed"])
    agg  = grp[cols].mean().round(4)
    sum_path = OUT_DIR / "e2_summary.csv"
    agg.to_csv(sum_path)
    print(f"Summary    → {sum_path}")

    # ── Console digest ─────────────────────────────────────────────────────────
    print(f"\n{'Cond':10} {'Seed':5}  "
          f"{'r_dim':7} {'r_hidden':10} {'proj_res':10} "
          f"{'Δlogit':10} {'xyz_drift':10} {'JSER_proj':10} {'JSER_rand':10}")
    print("-" * 100)
    for (cond, seed), grp_df in df.groupby(["condition", "seed"]):
        rd   = grp_df["r_dim"].mean()
        rh   = grp_df["r_hidden"].mean()
        pr   = grp_df["proj_residual"].mean()
        dl   = grp_df["logit_delta_proj"].mean()
        dr   = grp_df["xyz_drift_proj"].mean()
        ev_p = grp_df["evades_proj"].mean()
        ev_r = grp_df["evades_rand"].mean()
        print(f"{cond:10} {seed:5}  "
              f"{rd:.4f}  {rh:.3e}  {pr:.3e}  "
              f"{dl:+.6f}  {dr:.6f}  {ev_p:.4f}     {ev_r:.4f}")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
    print("\nKey interpretation:")
    print("  hidden_ratio  > 0   → ker(J_M) exists (structural)")
    print("  alignment     ≈ 0   → target gradient ⊥ ker(J_M) → projection attack ineffective")
    print("  Δlogit_proj   ≈ 0   → logit immovable within ker(J_M)")
    print("  xyz_drift_proj≈ 0   → XYZ features stable during projection attack (linearization holds)")
    print("  JSER_proj vs JSER_rand → null-space vs random control evasion rates")


if __name__ == "__main__":
    main()

