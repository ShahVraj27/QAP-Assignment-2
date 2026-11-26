"""Part D: deliberately broken QPE circuits (D1 to D6).

Every variant is built by qpe_variant, which is qpe_circuit with switches that
introduce exactly one mistake. With all switches at their defaults it builds the
correct circuit.
"""

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit

from .analysis import bits, counts_to_probabilities, exact_probabilities
from .core import build_U, run
from .experiments import SEED, SHOTS
from .powers import append_controlled_power
from .states import basis

N_D = 3          # U_3
M_D = 3          # three counting qubits
X_D = 5          # |101>, phi = 5/8
X_D6 = 3         # D6 intends |011>, phi = 3/8


def inverse_qft_variant(qc, qubits, swaps=True):
    """The starter inverse QFT, with the option to leave out the swap gates (D2)."""
    m = len(qubits)
    if swaps:
        for i in range(m // 2):
            qc.swap(qubits[i], qubits[m - 1 - i])
    for j in range(m):
        for k in range(j):
            qc.cp(-np.pi / 2**(j - k), qubits[k], qubits[j])
        qc.h(qubits[j])


def qft(qc, qubits):
    """Forward QFT: the exact inverse of the starter inverse_qft (gates reversed, angles negated)."""
    m = len(qubits)
    for j in reversed(range(m)):
        qc.h(qubits[j])
        for k in reversed(range(j)):
            qc.cp(np.pi / 2**(j - k), qubits[k], qubits[j])
    for i in range(m // 2):
        qc.swap(qubits[i], qubits[m - 1 - i])


def prepare_basis_state_reversed(qc, target, x):
    """D6 bug: X gates applied with the bit order reversed (bit j of x goes to qubit n-1-j)."""
    n = len(target)
    for j, q in enumerate(target):
        if (x >> (n - 1 - j)) & 1:
            qc.x(q)


def qpe_variant(U, n, m, prepare, *, hadamards=True, power_index=None,
                transform="inverse", swaps=True, method="angle", measure=True):
    """QPE with optional mistakes.

    hadamards   : False leaves out the H gates on the counting qubits (D4)
    power_index : function k -> j so that counting qubit k controls U^(2^j);
                  default j = k. D1 uses j = 0, D3 uses j = m-1-k.
    transform   : "inverse" (correct) or "forward" (D5)
    swaps       : False removes the swaps from the inverse QFT (D2)
    """
    if power_index is None:
        power_index = lambda k: k
    qc = QuantumCircuit(m + n, m)
    count = list(range(m))
    target = list(range(m, m + n))
    prepare(qc, target)
    qc.barrier()
    if hadamards:
        for q in count:
            qc.h(q)
    for k in range(m):
        append_controlled_power(qc, U, power_index(k), count[k], target, method=method)
    qc.barrier()
    if transform == "inverse":
        inverse_qft_variant(qc, count, swaps=swaps)
    elif transform == "forward":
        qft(qc, count)
    else:
        raise ValueError("transform must be 'inverse' or 'forward'")
    if measure:
        qc.barrier()
        for k in range(m):
            qc.measure(count[k], k)
    return qc


def _reversed_prep(qc, target):
    prepare_basis_state_reversed(qc, target, X_D6)


# id -> (description, keyword arguments for qpe_variant, preparer)
PART_D_VARIANTS = {
    "correct": ("Correct circuit, |101>", {}, basis(X_D)),
    "D1": ("Every counting qubit controls U once",
           {"power_index": lambda k: 0}, basis(X_D)),
    "D2": ("Swap gates removed from the inverse QFT",
           {"swaps": False}, basis(X_D)),
    "D3": ("Counting qubit k controls U^(2^(m-1-k))",
           {"power_index": lambda k: M_D - 1 - k}, basis(X_D)),
    "D4": ("H gates on the counting qubits left out",
           {"hadamards": False}, basis(X_D)),
    "D5": ("Forward QFT instead of inverse QFT",
           {"transform": "forward"}, basis(X_D)),
    "D6": ("|011> intended, X gates applied in reversed bit order",
           {}, _reversed_prep),
    "D2+D3": ("D2 and D3 combined (no swaps AND reversed powers)",
              {"swaps": False, "power_index": lambda k: M_D - 1 - k}, basis(X_D)),
}


def part_d_runs(shots=SHOTS, seed=SEED):
    """Run the correct circuit, D1..D6 and the D2+D3 combination with U_3, m = 3."""
    U = build_U(N_D)
    runs = []
    for run_id, (description, kwargs, prepare) in PART_D_VARIANTS.items():
        counts = run(qpe_variant(U, N_D, M_D, prepare, **kwargs), shots=shots, seed=seed)
        exact = exact_probabilities(qpe_variant(U, N_D, M_D, prepare, measure=False, **kwargs), range(M_D))
        runs.append({"run": run_id, "description": description, "counts": dict(counts),
                     "measured": counts_to_probabilities(counts, M_D), "exact": exact, "m": M_D})
    return runs


def part_d_table(runs):
    """One row per variant: exact distribution and the most likely outcome."""
    rows = []
    for r in runs:
        exact = r["exact"]
        support = ", ".join(f"{bits(y, M_D)}: {exact[y]:.3f}" for y in range(2**M_D) if exact[y] > 1e-9)
        y = int(np.argmax(r["measured"]))
        rows.append({"Run": r["run"], "change": r["description"],
                     "most likely y (measured)": bits(y, M_D),
                     "P measured": round(float(r["measured"][y]), 4),
                     "exact distribution (nonzero)": support})
    return pd.DataFrame(rows)


def kickback_phases(power_index, phi, m):
    """Relative phase (in turns) written on each counting qubit for a given power choice."""
    return [float(np.mod(2**power_index(k) * phi, 1)) for k in range(m)]
