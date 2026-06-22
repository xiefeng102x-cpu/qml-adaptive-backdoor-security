"""
pauli_ladder_ablation.py
========================
Pauli measurement ladder ablation for the QML backdoor detection paper.

Feature sets: Z, X, Y, XZ, XY, YZ, XYZ
Metrics: AUC, sep (=max(AUC,1-AUC)), TPR@5%FPR, JSER@5%FPR

Two conditions per (dataset, seed, layer):
  baseline  — original backdoor model weights from full_grid
  adaptive  — final weights after training-time attack λ=1.0 (loaded if .npy exists)

MNIST:      seeds 42-46, layers [4, 12], 5 qubits, pair t7_vs_t0, pr=0.1
BloodMNIST: seeds 42-46, layer  [8],    8 qubits, pair t6_vs_t0, pr=0.1

Usage:
  python pauli_ladder_ablation.py               # all datasets
  python pauli_ladder_ablation.py --dataset mnist
  python pauli_ladder_ablation.py --dataset blood

Output:
  results/pauli_ladder/ladder_results.csv    (per-seed raw)
  results/pauli_ladder/ladder_summary.csv    (mean±std)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

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

sys.path.insert(0, str(Path(__file__).resolve().parent))  # local code/ dir
from fast_circuit import _make_evals

# ── gate helpers ──────────────────────────────────────────────────────────────
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


# ── feature extraction ────────────────────────────────────────────────────────

def extract_all_features(samples: np.ndarray, weights: np.ndarray,
                          n_layers: int, n_qubits: int) -> dict[str, np.ndarray]:
    """
    Single circuit pass → all 7 Pauli feature sets.
    Returns dict: Z, X, Y, XZ, XY, YZ, XYZ  (float64 arrays [N, k])
    """
    psi   = _run_psi(samples.astype(np.float64), weights, n_layers, n_qubits)
    probs = np.abs(psi) ** 2
    z_ev, _, _ = _make_evals(n_qubits)

    Z = (probs @ z_ev.T).astype(np.float64)
    X = np.stack([(np.abs(_apply_1q(psi, _H_GATE,  q)) ** 2 @ z_ev[q])
                  for q in range(n_qubits)], axis=1).astype(np.float64)
    Y = np.stack([(np.abs(_apply_1q(psi, _HS_GATE, q)) ** 2 @ z_ev[q])
                  for q in range(n_qubits)], axis=1).astype(np.float64)

    return {
        "Z":   Z,
        "X":   X,
        "Y":   Y,
        "XZ":  np.hstack([X, Z]),
        "XY":  np.hstack([X, Y]),
        "YZ":  np.hstack([Y, Z]),
        "XYZ": np.hstack([X, Y, Z]),
    }


# ── fixed classifier (same architecture as paper2, weights fixed during training) ──

class FixedClassifier:
    """Linear(Q→32)→ReLU→Linear(32→2). Weights loaded from baseline .pt."""
    def __init__(self, w0: np.ndarray, b0: np.ndarray,
                 w3: np.ndarray, b3: np.ndarray):
        self.w0, self.b0 = w0, b0
        self.w3, self.b3 = w3, b3

    def forward(self, z: np.ndarray) -> np.ndarray:
        h = np.maximum(0, z @ self.w0.T + self.b0)
        return h @ self.w3.T + self.b3


# ── detection metrics ─────────────────────────────────────────────────────────

def detection_metrics(feat_clean: np.ndarray, feat_poison: np.ndarray,
                       u_a: np.ndarray | None = None,
                       fpr: float = 0.05) -> dict:
    """
    LedoitWolf Mahalanobis detector.

    u_a: optional per-sample attack-success indicator (bool array, len = len(feat_poison)).
         If provided, also computes true_jser = Pr[U_a AND evasion].

    Returns auc, sep, tpr_5fpr, jser_5fpr, true_jser, mahal_poison.
    mahal_poison = mean D (Mahalanobis distance, sqrt of quadratic form), matching
    the theoretical definition D_Σ = sqrt(δᵀ Σ⁻¹ δ).
    """
    lw   = LedoitWolf().fit(feat_clean)
    prec = lw.get_precision()
    mu   = lw.location_

    def mahal(f: np.ndarray) -> np.ndarray:
        d = f - mu
        return np.einsum("ni,ij,nj->n", d, prec, d)  # returns D², then sqrt below

    sc = mahal(feat_clean)
    sp = mahal(feat_poison)

    y    = np.array([0] * len(sc) + [1] * len(sp))
    auc  = float(roc_auc_score(y, np.concatenate([sc, sp])))
    sep  = max(auc, 1.0 - auc)

    thresh = np.percentile(sc, (1.0 - fpr) * 100)
    tpr    = float(np.mean(sp > thresh))
    jser   = 1.0 - tpr

    evades = sp <= thresh  # per-sample evasion indicator

    true_jser = None
    if u_a is not None:
        true_jser = round(float(np.mean(u_a & evades)), 4)

    return dict(
        auc=round(auc, 4),
        sep=round(sep, 4),
        tpr_5fpr=round(tpr, 4),
        jser_5fpr=round(jser, 4),
        true_jser=true_jser,
        mahal_poison=round(float(np.sqrt(sp).mean()), 4),  # mean D (not D²)
    )


# ── data loading ──────────────────────────────────────────────────────────────

def load_data(seed: int, layer: int, pair: str, data_root: Path,
              pr: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray, "FixedClassifier"]:
    """
    Load baseline backdoor model and data.
    Returns: (weights [P], x_clean_target [N,D], x_poison [M,D], clf)

    clf is the fixed MLP classifier (weights never change during adaptive training).
    """
    import torch
    pr_s  = f"{float(pr):.12g}"
    base  = data_root / f"seed_{seed}" / f"layer_{layer}_grid" / pair / f"pr_{pr_s}"
    meta  = torch.load(base / f"poisoned_samples_meta_seed_{seed}_pr_{pr_s}.pt",
                       map_location="cpu", weights_only=False)

    # Handle two naming conventions across datasets
    model_path = base / "models" / f"clean_model_seed_{seed}_epoch_200.pt"
    if not model_path.exists():
        model_path = base / "models" / "clean_model_seed_200.pt"

    state   = torch.load(model_path, map_location="cpu", weights_only=False)
    weights = state["weights"].detach().numpy().copy().astype(np.float64)

    clf = FixedClassifier(
        w0=state["classifier.0.weight"].numpy(),
        b0=state["classifier.0.bias"].numpy(),
        w3=state["classifier.3.weight"].numpy(),
        b3=state["classifier.3.bias"].numpy(),
    )

    x_clean  = meta["target_clean_data"].numpy().astype(np.float64)
    x_poison = meta["poisoned_non_target_data"].numpy().astype(np.float64)
    return weights, x_clean, x_poison, clf


# ── main per-config runner ────────────────────────────────────────────────────

FEATURE_ORDER = ["Z", "X", "Y", "XZ", "XY", "YZ", "XYZ"]


def load_asr_ca(result_json: Path | None, condition: str) -> dict:
    """Read ASR and CA from training-time result JSON for given condition."""
    na = dict(asr=None, ca=None, auc_z_train=None, auc_xyz_train=None)
    if result_json is None or not result_json.exists():
        return na
    import json
    d = json.loads(result_json.read_text())
    key = "final" if condition == "adaptive" else "baseline"
    block = d.get(key)
    if not block:
        return na
    return dict(
        asr=block.get("asr"),
        ca=block.get("ca"),
        auc_z_train=block.get("auc_z"),
        auc_xyz_train=block.get("auc_xyz"),
    )


def run_config(seed: int, layer: int, n_qubits: int, pair: str,
               data_root: Path, adap_weights_path: Path | None,
               condition: str, dataset: str,
               result_json: Path | None = None) -> list[dict]:
    weights_base, x_clean, x_poison, clf = load_data(seed, layer, pair, data_root)

    if condition == "adaptive":
        if adap_weights_path is None or not adap_weights_path.exists():
            return []
        weights = np.load(adap_weights_path).astype(np.float64)
    else:
        weights = weights_base

    feats_c = extract_all_features(x_clean,  weights, layer, n_qubits)
    feats_p = extract_all_features(x_poison, weights, layer, n_qubits)
    extra   = load_asr_ca(result_json, condition)

    # U_a: per-sample attack success indicator (uses Z features + fixed clf)
    # clf is fixed (paper2 weights), QNN weights differ between baseline/adaptive
    z_p = feats_p["Z"]
    preds = np.argmax(clf.forward(z_p), axis=1)
    u_a = (preds == 1)  # target class = 1

    rows = []
    for fs in FEATURE_ORDER:
        m = detection_metrics(feats_c[fs], feats_p[fs], u_a=u_a)
        rows.append(dict(dataset=dataset, layer=layer, n_qubits=n_qubits,
                         seed=seed, condition=condition, feature_set=fs,
                         **m, **extra))
    return rows


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "blood", "all"], default="all")
    parser.add_argument("--mnist-layer", type=int, default=12,
                        help="MNIST layer to run (default: 12; use 4 for L=4)")
    args = parser.parse_args()

    OUT_DIR    = ROOT / "results/pauli_ladder"
    MNIST_ROOT = ROOT / "data/Mnist/detail_single_samle"
    BLOOD_ROOT = ROOT / "data/MedMnist/single_detect_20260411/epsilon_0.8"
    ADAP_MNIST = ROOT / "results/training_time/new_version"
    ADAP_BLOOD = ROOT / "results/training_time_bloodmnist"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    SEEDS = [42, 43, 44, 45, 46]

    configs: list[dict] = []
    if args.dataset in ("mnist", "all"):
        for layer in [args.mnist_layer]:
            configs.append(dict(dataset="MNIST",      data_root=MNIST_ROOT,
                                layer=layer, n_qubits=5, pair="t7_vs_t0",
                                adap_dir=ADAP_MNIST))
    if args.dataset in ("blood", "all"):
        configs.append(dict(dataset="BloodMNIST", data_root=BLOOD_ROOT,
                            layer=8, n_qubits=8, pair="t6_vs_t0",
                            adap_dir=ADAP_BLOOD))

    all_rows: list[dict] = []

    for cfg in configs:
        ds = cfg["dataset"]
        print(f"\n{'='*60}")
        print(f"  {ds}  L={cfg['layer']}  ({cfg['n_qubits']}q)")
        print(f"{'='*60}")

        for seed in SEEDS:
            # result JSON for ASR/CA (λ=1.0 only)
            result_json = cfg["adap_dir"] / f"result_seed={seed}_L{cfg['layer']}_lr1.0.json"

            for condition in ("baseline", "adaptive"):
                adap_path = None
                if condition == "adaptive":
                    adap_path = (cfg["adap_dir"]
                                 / f"weights_seed={seed}_L{cfg['layer']}_lr1.0.npy")

                print(f"  seed={seed}  {condition:10s}...", end=" ", flush=True)
                rows = run_config(seed, cfg["layer"], cfg["n_qubits"], cfg["pair"],
                                  cfg["data_root"], adap_path, condition, ds,
                                  result_json=result_json)
                if rows:
                    all_rows.extend(rows)
                    print(f"OK  ({len(rows)} feature sets)")
                else:
                    print("SKIP (no adaptive weights)")

    if not all_rows:
        print("\nNo data collected.")
        return

    df = pd.DataFrame(all_rows)

    # Merge with any existing rows covering datasets NOT in this run,
    # so a partial run (--dataset mnist) doesn't wipe BloodMNIST rows.
    raw_path = OUT_DIR / "ladder_results.csv"
    if raw_path.exists():
        old = pd.read_csv(raw_path)
        new_ds = df["dataset"].unique().tolist()
        kept   = old[~old["dataset"].isin(new_ds)]
        df     = pd.concat([kept, df], ignore_index=True)

    df.to_csv(raw_path, index=False)
    print(f"\nRaw results → {raw_path}")

    # Summary: mean ± std across seeds
    grp  = df.groupby(["dataset", "layer", "condition", "feature_set"])
    cols = ["auc", "sep", "tpr_5fpr", "jser_5fpr", "true_jser", "mahal_poison"]
    # true_jser may be None for configs missing clf; dropna before aggregating
    agg  = grp[cols].agg(["mean", "std"]).round(4)
    sum_path = OUT_DIR / "ladder_summary.csv"
    agg.to_csv(sum_path)
    print(f"Summary      → {sum_path}")

    # ── console table ─────────────────────────────────────────────────────────
    print(f"\n{'Dataset':12} {'L':3} {'Cond':10} {'Features':7}  "
          f"{'AUC':7} {'sep':7} {'TPR@5%':7} {'JSER_FNR':8} {'JSER_true':9} {'Mahal_D':8}")
    print("-" * 86)

    means = grp[cols].mean().reset_index()
    for _, r in means.iterrows():
        tj = f"{r['true_jser']:.4f}" if pd.notna(r["true_jser"]) else "  N/A  "
        print(f"{r['dataset']:12} {r['layer']:<3} {r['condition']:10} {r['feature_set']:7}  "
              f"{r['auc']:.4f}  {r['sep']:.4f}  {r['tpr_5fpr']:.4f}   "
              f"{r['jser_5fpr']:.4f}   {tj}   {r['mahal_poison']:.3f}")


if __name__ == "__main__":
    main()


