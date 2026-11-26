"""Experiment runners for Part A (core grid) and Part B (phase 1/3)."""

import numpy as np
import pandas as pd
from qiskit import transpile

from .analysis import (
    bits,
    counts_to_probabilities,
    exact_probabilities,
    is_exact,
    phase_to_binary,
    top_outcomes,
)
from .core import build_U, qpe_circuit, run
from .oracles import build_mcp_U
from .states import all_ones, basis

SHOTS = 2048
SEED = 1234
M_VALUES = [1, 2, 3]

# Part A states (n, x): |1>, |11>, |101>, |1011>, |1000>
PART_A_STATES = [(1, 1), (2, 3), (3, 5), (4, 11), (4, 8)]


def bits_needed(phi, max_digits=16):
    """Number of binary digits phi needs (0 for phi = 0)."""
    for d in range(max_digits + 1):
        if predict_exact(phi, d):
            return d
    return None


def predict_exact(phi, m):
    """QPE is exact iff phi has at most m binary digits, i.e. 2^m * phi is an integer."""
    scaled = phi * 2**m
    return bool(np.isclose(scaled, np.round(scaled)))


def run_qpe(U, n, m, prepare, method="angle", shots=SHOTS, seed=SEED):
    """Run one QPE experiment: sampled counts plus exact probabilities."""
    counts = run(qpe_circuit(U, n, m, prepare, method=method), shots=shots, seed=seed)
    exact = exact_probabilities(qpe_circuit(U, n, m, prepare, method=method, measure=False), range(m))
    return {
        "counts": dict(counts),
        "measured": counts_to_probabilities(counts, m),
        "exact": exact,
    }


# ------------------------------------------------------------------ Part A
def part_a_predictions():
    """Prediction column for A1..A15, computed from the binary expansion only."""
    rows, run_id = [], 1
    for n, x in PART_A_STATES:
        phi = x / 2**n
        for m in M_VALUES:
            rows.append({
                "Run": f"A{run_id}", "n": n, "|v>": "|" + bits(x, n) + ">",
                "true phi": phase_to_binary(phi, n), "bits needed": bits_needed(phi),
                "m": m, "prediction: exact?": "yes" if predict_exact(phi, m) else "no",
            })
            run_id += 1
    return pd.DataFrame(rows)


def part_a_runs(method="angle", shots=SHOTS, seed=SEED):
    """Run all 15 Part A experiments with U_n."""
    runs, run_id = [], 1
    for n, x in PART_A_STATES:
        phi = x / 2**n
        for m in M_VALUES:
            result = run_qpe(build_U(n), n, m, basis(x), method=method, shots=shots, seed=seed)
            result.update({"run": f"A{run_id}", "n": n, "x": x, "phi": phi, "m": m,
                           "prediction": predict_exact(phi, m)})
            runs.append(result)
            run_id += 1
    return runs


def part_a_table(runs):
    """The completed Part A table of the assignment."""
    rows = []
    for r in runs:
        m = r["m"]
        y = int(np.argmax(r["measured"]))
        rows.append({
            "Run": r["run"], "n": r["n"], "|v>": "|" + bits(r["x"], r["n"]) + ">",
            "true phi": phase_to_binary(r["phi"], r["n"]), "m": m,
            "prediction: exact?": "yes" if r["prediction"] else "no",
            "most likely y (bits)": bits(y, m),
            "phi_tilde": y / 2**m,
            "P(most likely) measured": round(float(r["measured"][y]), 4),
            "P(most likely) exact": round(float(r["exact"][y]), 4),
            "exact?": "yes" if is_exact(r["exact"]) else "no",
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Part B
PART_B_PHI = 1 / 3


def part_b_runs(method="angle", shots=SHOTS, seed=SEED):
    """Run the 12 Part B experiments: MCP(2*pi/3) on |1...1>, n = 1..4, m = 1..3."""
    runs, run_id = [], 1
    for n in [1, 2, 3, 4]:
        for m in M_VALUES:
            result = run_qpe(build_mcp_U(n), n, m, all_ones(n), method=method, shots=shots, seed=seed)
            result.update({"run": f"B{run_id}", "n": n, "x": 2**n - 1, "phi": PART_B_PHI, "m": m})
            runs.append(result)
            run_id += 1
    return runs


def part_b_extra_run(method="angle", shots=SHOTS, seed=SEED):
    """Part B question 3: n = 4, |v> = |1110> (x = 14), m = 3. True phi = 0."""
    result = run_qpe(build_mcp_U(4), 4, 3, basis(14), method=method, shots=shots, seed=seed)
    result.update({"run": "B13", "n": 4, "x": 14, "phi": 0.0, "m": 3})
    return result


def part_b_table(runs):
    """Two most likely outcomes, their probabilities and the error check."""
    rows = []
    for r in runs:
        m = r["m"]
        (y1, p1), (y2, p2) = top_outcomes(r["measured"], 2)
        error = abs(r["phi"] - y1 / 2**m)
        bound = 1 / 2**(m + 1)
        rows.append({
            "Run": r["run"], "n": r["n"], "|v>": "|" + bits(r["x"], r["n"]) + ">", "m": m,
            "1st y": bits(y1, m), "P(1st) measured": round(p1, 4),
            "P(1st) exact": round(float(r["exact"][y1]), 4),
            "2nd y": bits(y2, m), "P(2nd) measured": round(p2, 4),
            "P(2nd) exact": round(float(r["exact"][y2]), 4),
            "phi_tilde (1st)": y1 / 2**m,
            "|phi - phi_tilde|": round(error, 4),
            "bound 1/2^(m+1)": bound,
            "error <= bound": bool(error <= bound + 1e-12),
        })
    return pd.DataFrame(rows)


def circuit_cost(U, n, m, prepare, method="angle"):
    """Transpiled depth and CX count of the measurement-free QPE circuit."""
    qc = qpe_circuit(U, n, m, prepare, method=method, measure=False)
    tqc = transpile(qc, basis_gates=["u", "p", "cx"], optimization_level=0)
    ops = tqc.count_ops()
    return {"depth": tqc.depth(), "CX count": ops.get("cx", 0), "total gates": sum(ops.values())}
