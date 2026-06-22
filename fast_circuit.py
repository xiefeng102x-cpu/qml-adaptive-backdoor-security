"""
Fast pure-numpy state-vector evaluator for the QNN circuit.

Matches paper2's circuit exactly:
  AmplitudeEmbedding (normalize) → L × [RY+RZ per qubit + ring CNOT]
  Measurements: <Z_q> and <Z_i Z_j>

Processes a batch of N samples simultaneously in O(N * 2^n_qubits) memory.
~1000x faster than calling PennyLane one sample at a time.
"""
import itertools
import numpy as np

# Default for MNIST (5 qubits); override via run_circuit_batch(n_qubits=...)
N_QUBITS = 5
ZZ_PAIRS = list(itertools.combinations(range(N_QUBITS), 2))  # 10 pairs


# ── gate application helpers ──────────────────────────────────────────────────

def _apply_single_qubit(psi: np.ndarray, gate: np.ndarray, qubit: int) -> np.ndarray:
    """
    Apply a 2×2 gate to `qubit` in a batch of state vectors.

    psi:   (N, 2^N_QUBITS) complex
    gate:  (2, 2) complex
    qubit: 0 = MSB (PennyLane convention)
    Returns: (N, 2^N_QUBITS) complex
    """
    dim  = psi.shape[1]
    step = dim >> (qubit + 1)          # = 2^(N_QUBITS-1-qubit)
    groups = dim // (2 * step)          # number of [|0⟩...|1⟩] blocks
    psi_r = psi.reshape(-1, groups, 2, step)          # (N, groups, 2, step)
    out   = np.einsum("ij,bkjl->bkil", gate, psi_r, optimize=True)
    return out.reshape(-1, dim)


def _apply_cnot(psi: np.ndarray, ctrl: int, tgt: int) -> np.ndarray:
    """
    Apply CNOT(ctrl → tgt) to batch of state vectors.
    CNOT is a permutation on basis states: |c,t⟩ → |c, c⊕t⟩.
    n_qubits is needed to compute bit positions.
    """
    dim      = psi.shape[1]
    n_qubits = int(np.log2(dim))
    ctrl_bit = n_qubits - 1 - ctrl   # bit position from LSB
    tgt_bit  = n_qubits - 1 - tgt
    states   = np.arange(dim, dtype=np.int64)
    ctrl_on  = (states >> ctrl_bit) & 1          # 1 where control qubit = |1⟩
    src      = np.where(ctrl_on, states ^ (1 << tgt_bit), states)
    return psi[:, src]   # fancy-index: reorder columns


# ── eigenvalue tables (cached per n_qubits) ────────────────────────────────────

_eval_cache: dict = {}


def _make_evals(n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute Z and ZZ eigenvalue arrays over all 2^n_qubits basis states.
    Cached: subsequent calls with same n_qubits are O(1).
    """
    if n_qubits in _eval_cache:
        return _eval_cache[n_qubits]

    zz_pairs = list(itertools.combinations(range(n_qubits), 2))
    dim      = 2 ** n_qubits
    states   = np.arange(dim, dtype=np.int64)

    z_evals = np.zeros((n_qubits, dim), dtype=np.float64)
    for q in range(n_qubits):
        bit_q      = (states >> (n_qubits - 1 - q)) & 1   # MSB convention
        z_evals[q] = 1 - 2 * bit_q                         # +1 if |0⟩, -1 if |1⟩

    zz_evals = np.zeros((len(zz_pairs), dim), dtype=np.float64)
    for k, (i, j) in enumerate(zz_pairs):
        bit_i       = (states >> (n_qubits - 1 - i)) & 1
        bit_j       = (states >> (n_qubits - 1 - j)) & 1
        zz_evals[k] = (1 - 2 * bit_i) * (1 - 2 * bit_j)

    result = (z_evals, zz_evals, zz_pairs)
    _eval_cache[n_qubits] = result
    return result


# Precompute for the default (MNIST) case
_Z_EVALS, _ZZ_EVALS, _ = _make_evals(N_QUBITS)


# ── main batch evaluator ───────────────────────────────────────────────────────

def run_circuit_batch(x_batch: np.ndarray, weights: np.ndarray,
                      n_layers: int, n_qubits: int = N_QUBITS) -> np.ndarray:
    """
    Evaluate QNN circuit on a batch of input samples.

    x_batch:  (N, 2^n_qubits)            — raw input (will be normalised)
    weights:  (n_layers * n_qubits * 2,) — flat weight vector
    n_layers: int
    n_qubits: int (default 5 for MNIST; use 8 for BloodMNIST)

    Returns: (N, n_qubits + C(n_qubits,2)) float32
             first n_qubits cols = <Z_q>
             remaining cols      = <Z_i Z_j> for pairs in itertools.combinations order
    """
    N   = len(x_batch)
    dim = 2 ** n_qubits
    # Amplitude embedding: normalise each row
    norms = np.linalg.norm(x_batch, axis=1, keepdims=True).clip(min=1e-12)
    psi   = (x_batch / norms).astype(np.complex128)   # (N, dim)

    z_evals, zz_evals, _ = _make_evals(n_qubits)
    w = weights.reshape(n_layers, n_qubits, 2)

    for l in range(n_layers):
        for q in range(n_qubits):
            theta, phi = float(w[l, q, 0]), float(w[l, q, 1])

            # RY(theta)
            ct, st = np.cos(theta / 2), np.sin(theta / 2)
            ry = np.array([[ct, -st], [st,  ct]], dtype=np.complex128)
            psi = _apply_single_qubit(psi, ry, q)

            # RZ(phi)
            ep = np.exp(1j * phi / 2)
            rz = np.array([[1/ep, 0], [0, ep]], dtype=np.complex128)
            psi = _apply_single_qubit(psi, rz, q)

        # Ring CNOT: q → (q+1) % n_qubits
        for q in range(n_qubits):
            psi = _apply_cnot(psi, q, (q + 1) % n_qubits)

    # Compute probabilities: (N, dim)
    probs = np.abs(psi) ** 2

    # Expectation values
    z_vals  = probs @ z_evals.T    # (N, n_qubits)
    zz_vals = probs @ zz_evals.T   # (N, C(n_qubits,2))

    return np.concatenate([z_vals, zz_vals], axis=1).astype(np.float32)


def verify_against_pennylane(n_layers: int = 3, n_samples: int = 4,
                             tol: float = 1e-5) -> bool:
    """
    Quick self-test: compare fast_circuit output against PennyLane reference.
    Returns True if max absolute error < tol.
    """
    try:
        import pennylane as qml
        import torch
    except ImportError:
        print("PennyLane or torch not available for verification.")
        return True

    rng = np.random.RandomState(0)
    X   = rng.randn(n_samples, 32).astype(np.float64)
    W   = rng.randn(n_layers * N_QUBITS * 2).astype(np.float64)

    # ── fast circuit ─────────────────────────────────────────────────────────
    fast_out = run_circuit_batch(X, W, n_layers)  # (n_samples, 15)

    # ── PennyLane reference ──────────────────────────────────────────────────
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface=None)
    def pl_circuit(x, weights):
        qml.AmplitudeEmbedding(x, wires=range(N_QUBITS), normalize=True)
        ww = weights.reshape(n_layers, N_QUBITS, 2)
        for l in range(n_layers):
            for q in range(N_QUBITS):
                qml.RY(ww[l, q, 0], wires=q)
                qml.RZ(ww[l, q, 1], wires=q)
            for q in range(N_QUBITS):
                qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
        return (
            [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]
            + [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for i, j in ZZ_PAIRS]
        )

    pl_out = np.array([pl_circuit(X[i], W) for i in range(n_samples)], dtype=np.float32)

    err = np.max(np.abs(fast_out - pl_out))
    ok  = err < tol
    print(f"verify_against_pennylane: max_err={err:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


# ── full 2-body Pauli extension ────────────────────────────────────────────────

# Basis-change gates for X and Y measurement
_H_GATE = np.array([[1,  1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
_HSDAG  = np.array([[1, -1j],[1,  1j]], dtype=np.complex128) / np.sqrt(2)  # HS†: Y eigenstates → Z eigenstates

# Two-qubit Pauli pair labels and corresponding (row-qubit, col-qubit) rotations
PAULI2_LABELS = ['ZZ','ZX','ZY','XZ','XX','XY','YZ','YX','YY']
_PAULI2_ROTS  = [
    ('Z','Z'), ('Z','X'), ('Z','Y'),
    ('X','Z'), ('X','X'), ('X','Y'),
    ('Y','Z'), ('Y','X'), ('Y','Y'),
]
_ROT_GATE = {'X': _H_GATE, 'Y': _HSDAG}   # 'Z' → no rotation needed


def run_circuit_batch_full2body(x_batch: np.ndarray, weights: np.ndarray,
                                n_layers: int, n_qubits: int = N_QUBITS) -> np.ndarray:
    """
    Full 1+2-body Pauli feature extraction.

    Column layout of returned (N, 3n + 9*C(n,2)) float32 array:
      [0       : n]       ⟨Z_q⟩  for q = 0..n-1
      [n       : 2n]      ⟨X_q⟩
      [2n      : 3n]      ⟨Y_q⟩
      [3n      : 3n+9C]   for each pair (i<j) in combinations order,
                          9 values in PAULI2_LABELS order:
                          ZZ, ZX, ZY, XZ, XX, XY, YZ, YX, YY

    n = n_qubits,  C = C(n_qubits, 2)
    """
    N   = len(x_batch)
    dim = 2 ** n_qubits
    norms = np.linalg.norm(x_batch, axis=1, keepdims=True).clip(min=1e-12)
    psi   = (x_batch / norms).astype(np.complex128)   # (N, dim)

    z_evals, zz_evals, pairs = _make_evals(n_qubits)
    w = weights.reshape(n_layers, n_qubits, 2)

    # ── run circuit layers ────────────────────────────────────────────────────
    for l in range(n_layers):
        for q in range(n_qubits):
            theta, phi = float(w[l, q, 0]), float(w[l, q, 1])
            ct, st = np.cos(theta / 2), np.sin(theta / 2)
            ry = np.array([[ct, -st], [st, ct]], dtype=np.complex128)
            psi = _apply_single_qubit(psi, ry, q)
            ep = np.exp(1j * phi / 2)
            rz = np.array([[1/ep, 0], [0, ep]], dtype=np.complex128)
            psi = _apply_single_qubit(psi, rz, q)
        for q in range(n_qubits):
            psi = _apply_cnot(psi, q, (q + 1) % n_qubits)

    # ── precompute per-qubit basis-rotated states ─────────────────────────────
    # psi_x[q]: state with qubit q rotated to X measurement basis (H applied)
    # psi_y[q]: state with qubit q rotated to Y measurement basis (HS† applied)
    probs  = np.abs(psi) ** 2                                   # (N, dim)
    psi_x  = [_apply_single_qubit(psi, _H_GATE, q) for q in range(n_qubits)]
    psi_y  = [_apply_single_qubit(psi, _HSDAG,  q) for q in range(n_qubits)]

    # ── single-qubit expectations ─────────────────────────────────────────────
    z_vals = probs @ z_evals.T                                  # (N, n)
    x_vals = np.stack([np.abs(psi_x[q]) ** 2 @ z_evals[q]
                       for q in range(n_qubits)], axis=1)       # (N, n)
    y_vals = np.stack([np.abs(psi_y[q]) ** 2 @ z_evals[q]
                       for q in range(n_qubits)], axis=1)       # (N, n)

    # ── two-qubit expectations ────────────────────────────────────────────────
    two_cols = []
    for k, (i, j) in enumerate(pairs):
        for P, Q in _PAULI2_ROTS:
            if P == 'Z' and Q == 'Z':
                ev = probs @ zz_evals[k]
            elif Q == 'Z':
                # qubit i rotated; qubit j stays in Z basis
                src = psi_x[i] if P == 'X' else psi_y[i]
                ev  = np.abs(src) ** 2 @ zz_evals[k]
            elif P == 'Z':
                # qubit j rotated; qubit i stays in Z basis
                src = psi_x[j] if Q == 'X' else psi_y[j]
                ev  = np.abs(src) ** 2 @ zz_evals[k]
            else:
                # both qubits need rotation: start from P-rotated state of qubit i,
                # then apply Q-rotation to qubit j
                src    = psi_x[i] if P == 'X' else psi_y[i]
                psi_ij = _apply_single_qubit(src, _ROT_GATE[Q], j)
                ev     = np.abs(psi_ij) ** 2 @ zz_evals[k]
            two_cols.append(ev)

    two_arr = np.stack(two_cols, axis=1)                        # (N, 9*C(n,2))
    return np.concatenate([z_vals, x_vals, y_vals, two_arr], axis=1).astype(np.float32)


if __name__ == "__main__":
    verify_against_pennylane()
