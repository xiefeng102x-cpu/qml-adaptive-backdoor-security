"""
parse_ibm_hw_results.py
=======================
Parse IBM Quantum job archives (downloaded from the Zenodo deposit) and
recompute the per-axis Gate-2 AUC metrics reported in the paper.

Does NOT require IBM Quantum credentials — works entirely from the local
ZIP files. No special dependencies beyond numpy and scikit-learn.

Usage
-----
  # MNIST job (job-d8b9auh59p8c73bm6mtg.zip):
  python hardware/parse_ibm_hw_results.py --dataset mnist

  # BloodMNIST job (job-d8be0btmdd1s73b9e2bg.zip):
  python hardware/parse_ibm_hw_results.py --dataset blood

  # Both:
  python hardware/parse_ibm_hw_results.py

Input
-----
  Place the job ZIP files from the Zenodo deposit in the same directory
  as this script (hardware/) before running, or pass --data-dir:

    python hardware/parse_ibm_hw_results.py --data-dir /path/to/zenodo/data/hardware/

Expected outputs
----------------
  hardware/ibm_mnist_per_axis.json
  hardware/ibm_blood_per_axis.json

Reference hardware results (see Supplementary Table S12):
  MNIST  : X-AUC=0.668, Y-AUC=0.745, Z-AUC=0.669, XYZ-AUC=0.773
  Blood  : XYZ-AUC=0.970

Backend: IBM Quantum ibm_kingston (Heron r2, 156 qubits)
API    : qiskit-ibm-runtime 0.47.0, EstimatorV2
Shots  : 4096, Pauli twirling randomisations: 32
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Job configuration
# ---------------------------------------------------------------------------

CONFIGS = {
    "mnist": {
        "job_id":    "d8b9auh59p8c73bm6mtg",
        "zip_name":  "job-d8b9auh59p8c73bm6mtg.zip",
        "n_qubits":  5,
        "n_samples": 60,
        # 15 clean-target + 15 clean-nontarget + 15 poisoned-target + 15 poisoned-nontarget
        # Label 1 = poisoned (backdoor-triggered), 0 = clean
        "labels": np.array([0] * 15 + [0] * 15 + [1] * 15 + [1] * 15),
        "description": "MNIST 5q Layer-12 amplitude encoding, seed=42, λ_r=1.0",
        "sim_auc_baseline": {"X": 0.771, "Y": 0.768, "Z": 0.767, "XYZ": 0.882},
    },
    "blood": {
        "job_id":    "d8be0btmdd1s73b9e2bg",
        "zip_name":  "job-d8be0btmdd1s73b9e2bg.zip",
        "n_qubits":  8,
        "n_samples": 60,
        "labels": np.array([0] * 15 + [0] * 15 + [1] * 15 + [1] * 15),
        "description": "BloodMNIST 8q Layer-8 angle encoding, seed=42, λ_r=1.0",
        "sim_auc_baseline": {"XYZ": 0.970},
    },
}


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------

def decode_ndarray(obj: dict) -> np.ndarray:
    """Decode a Qiskit-serialised numpy array (base64 + zlib)."""
    import zlib
    raw = base64.b64decode(obj["__value__"])
    raw = zlib.decompress(raw)
    return np.load(io.BytesIO(raw), allow_pickle=False)


def load_phi_hw(zip_path: Path, job_id: str, n_qubits: int) -> np.ndarray:
    """
    Extract per-sample Pauli expectation-value matrix from an IBM job ZIP.

    Returns
    -------
    phi : ndarray, shape (n_samples, 3 * n_qubits)
        Columns: [X_q0..X_qN, Y_q0..Y_qN, Z_q0..Z_qN]
    """
    with zipfile.ZipFile(zip_path) as zf:
        result_name = f"{job_id}-result.json"
        data = json.loads(zf.read(result_name))

    pubs = data["__value__"]["pub_results"]
    n = len(pubs)
    phi = np.full((n, 3 * n_qubits), np.nan)

    for i, pub in enumerate(pubs):
        fields = pub["__value__"]["data"]["__value__"]["fields"]
        evs = decode_ndarray(fields["evs"]).ravel()
        if evs.shape == (3 * n_qubits,):
            phi[i] = evs

    n_valid = np.sum(~np.isnan(phi[:, 0]))
    print(f"  Loaded {n_valid}/{n} valid pub_results ({3 * n_qubits} features each)")
    return phi


# ---------------------------------------------------------------------------
# Gate-2 detector evaluation
# ---------------------------------------------------------------------------

def gate2_eval(
    phi: np.ndarray,
    labels: np.ndarray,
    target_fpr: float = 0.05,
    rng_seed: int = 0,
) -> dict:
    """
    Logistic-regression Gate-2 detector on a random 50/50 split.

    Parameters
    ----------
    phi    : (N, d) feature matrix
    labels : (N,) binary labels — 1 = poisoned, 0 = clean
    target_fpr : FPR operating point for TPR / precision / F1

    Returns
    -------
    dict with AUC, TPR@FPR, Precision, Recall, F1, Realised_FPR
    """
    idx = np.random.default_rng(rng_seed).permutation(len(phi))
    split = len(phi) // 2
    tr, te = idx[:split], idx[split:]

    scaler = StandardScaler()
    phi_tr = scaler.fit_transform(phi[tr])
    phi_te = scaler.transform(phi[te])

    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(phi_tr, labels[tr])

    scores_tr = clf.predict_proba(phi_tr)[:, 1]
    threshold = float(np.percentile(scores_tr[labels[tr] == 0], 100 * (1 - target_fpr)))
    scores_te = clf.predict_proba(phi_te)[:, 1]

    auc = float(roc_auc_score(labels[te], scores_te))
    y_pred = (scores_te >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (labels[te] == 1)))
    fp = int(np.sum((y_pred == 1) & (labels[te] == 0)))
    fn = int(np.sum((y_pred == 0) & (labels[te] == 1)))
    tn = int(np.sum((y_pred == 0) & (labels[te] == 0)))
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0

    return {
        "AUC":          auc,
        "TPR_at_5FPR":  recall,
        "Precision":    prec,
        "F1":           f1,
        "Realised_FPR": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def process_dataset(name: str, data_dir: Path, out_dir: Path) -> None:
    cfg = CONFIGS[name]
    zip_path = data_dir / cfg["zip_name"]

    if not zip_path.exists():
        print(f"[{name}] ZIP not found: {zip_path} — skipping")
        return

    print(f"\n{'='*60}")
    print(f"Dataset : {name.upper()}")
    print(f"Config  : {cfg['description']}")
    print(f"ZIP     : {zip_path}")

    phi_hw = load_phi_hw(zip_path, cfg["job_id"], cfg["n_qubits"])
    labels = cfg["labels"]
    nq     = cfg["n_qubits"]

    axes = {
        "X":   phi_hw[:, :nq],
        "Y":   phi_hw[:, nq:2*nq],
        "Z":   phi_hw[:, 2*nq:],
        "XYZ": phi_hw,
    }

    results = {}
    sim_base = cfg.get("sim_auc_baseline", {})

    header = f"{'Axis':<6} {'HW AUC':>10} {'Sim base':>10} {'Delta':>8} {'TPR@5%FPR':>10}"
    print("\n" + header)
    print("-" * 50)

    for ax in ["X", "Y", "Z", "XYZ"]:
        r = gate2_eval(axes[ax], labels)
        results[ax] = r
        sim = sim_base.get(ax, float("nan"))
        delta = r["AUC"] - sim if not np.isnan(sim) else float("nan")
        print(
            f"{ax:<6} {r['AUC']:>10.4f} {sim:>10.4f} {delta:>+8.4f}"
            f" {r['TPR_at_5FPR']:>10.4f}"
        )

    out = {
        "job_id":      cfg["job_id"],
        "config":      {k: v for k, v in cfg.items()
                        if k not in ("labels", "sim_auc_baseline", "zip_name")},
        "hardware":    results,
        "sim_baseline_reported": sim_base,
        "notes": (
            "AUC computed on a random 50/50 split (rng_seed=0). "
            "Hardware results include Pauli twirling (32 randomisations). "
            "The large-scale simulation AUC (6000-config mean over all seeds/λ) "
            "is higher than the single-seed-42 subsample used here."
        ),
    }

    out_path = out_dir / f"ibm_{name}_per_axis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=lambda x: x.item() if hasattr(x, "item") else x)
    print(f"\nSaved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Parse IBM Quantum job ZIPs from Zenodo deposit")
    parser.add_argument(
        "--dataset", choices=["mnist", "blood", "both"], default="both",
        help="Which dataset to process (default: both)"
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the job ZIP files (default: same as this script)"
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).resolve().parent,
        help="Output directory for JSON results (default: same as this script)"
    )
    args = parser.parse_args()

    datasets = ["mnist", "blood"] if args.dataset == "both" else [args.dataset]
    for name in datasets:
        process_dataset(name, args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
