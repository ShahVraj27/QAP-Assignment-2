"""Optional bonus: A9 and A12 on a noisy fake IBM backend.

Both U^(2^k) methods are transpiled for the same fake backend and simulated with
its noise model (AerSimulator.from_backend). The comparison shows how the deeper
repetition circuit loses more probability to noise than angle multiplication.
"""

import importlib

import numpy as np
import pandas as pd
from qiskit import transpile
from qiskit_aer import AerSimulator

from .analysis import bits, counts_to_probabilities, exact_probabilities
from .core import build_U, qpe_circuit
from .experiments import SEED, SHOTS
from .states import basis

# 7-qubit fake backends (A12 needs 3 counting + 4 target = 7 qubits); first one found is used
FAKE_BACKENDS = ["FakeLagosV2", "FakeNairobiV2", "FakeJakartaV2", "FakePerth", "FakeOslo", "FakeGuadalupeV2"]

# run id, n, x, m, outcomes counted as "correct" (A12 has two equally likely nearest estimates)
BONUS_RUNS = [
    ("A9", 3, 5, 3, [5]),        # |101>,  phi = 5/8      -> 101 with probability 1
    ("A12", 4, 11, 3, [5, 6]),   # |1011>, phi = 11/16    -> 101 or 110, 0.41 each
]
METHODS = ["repeat", "angle"]


def load_fake_backend(names=FAKE_BACKENDS):
    """Return the first available fake backend from ``names``."""
    module = importlib.import_module("qiskit_ibm_runtime.fake_provider")
    for name in names:
        cls = getattr(module, name, None)
        if cls is not None:
            return cls()
    raise ImportError(f"None of {names} found in qiskit_ibm_runtime.fake_provider")


def two_qubit_gate_count(qc):
    """Number of two-qubit gates (barriers and measurements excluded)."""
    return sum(1 for inst in qc.data
               if len(inst.qubits) == 2 and inst.operation.name not in ("barrier", "measure"))


def transpile_for(qc, backend, seed=SEED):
    """Map the circuit onto the backend's qubits, connectivity and native gates."""
    return transpile(qc, backend=backend, optimization_level=1, seed_transpiler=seed)


def noisy_counts(tqc, backend, shots=SHOTS, seed=SEED):
    """Simulate a transpiled circuit with the backend's noise model."""
    sim = AerSimulator.from_backend(backend)
    return sim.run(tqc, shots=shots, seed_simulator=seed).result().get_counts()


def total_variation_distance(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def bonus_runs(backend, shots=SHOTS, seed=SEED):
    """Run A9 and A12 with both power methods on the noisy backend."""
    runs = []
    for run_id, n, x, m, correct in BONUS_RUNS:
        U = build_U(n)
        for method in METHODS:
            qc = qpe_circuit(U, n, m, basis(x), method=method)
            ideal = exact_probabilities(qc, range(m))
            tqc = transpile_for(qc, backend, seed)
            counts = noisy_counts(tqc, backend, shots, seed)
            noisy = counts_to_probabilities(counts, m)
            runs.append({
                "run": run_id, "method": method, "n": n, "x": x, "m": m, "correct": correct,
                "ideal": ideal, "noisy": noisy, "counts": dict(counts),
                "blocks": (2**m - 1) if method == "repeat" else m,
                "depth": tqc.depth(), "two_qubit": two_qubit_gate_count(tqc),
                "ops": dict(tqc.count_ops()),
            })
    return runs


def bonus_table(runs):
    """Depth, two-qubit gate count and how much of the ideal answer survives the noise."""
    rows = []
    for r in runs:
        y = int(np.argmax(r["noisy"]))
        rows.append({
            "Run": r["run"], "method": r["method"],
            "correct outcome(s)": " / ".join(bits(c, r["m"]) for c in r["correct"]),
            "controlled-U blocks": r["blocks"],
            "transpiled depth": r["depth"],
            "2-qubit gates": r["two_qubit"],
            "P(correct) ideal": round(float(sum(r["ideal"][c] for c in r["correct"])), 4),
            "P(correct) noisy": round(float(sum(r["noisy"][c] for c in r["correct"])), 4),
            "most likely y (noisy)": bits(y, r["m"]),
            "TVD noisy vs ideal": round(total_variation_distance(r["noisy"], r["ideal"]), 4),
        })
    return pd.DataFrame(rows)
