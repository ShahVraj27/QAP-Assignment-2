"""Part C: QPE when |v> is not an eigenstate of U."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import ClassicalRegister
from qiskit.quantum_info import Statevector

from .analysis import bits, counts_to_probabilities, exact_probabilities
from .core import build_U, qpe_circuit, run
from .experiments import SEED, SHOTS
from .oracles import build_X
from .states import prepare_minus, prepare_plus, prepare_plus_plus, prepare_zero

# run id -> (U name, n, |v> label, preparer, m, eigenstate description)
PART_C_RUNS = [
    ("C1", "X", 1, "|->", prepare_minus, 2, "yes, phi = 1/2"),
    ("C2", "X", 1, "|+>", prepare_plus, 2, "yes, phi = 0"),
    ("C3", "X", 1, "|0>", prepare_zero, 2, "no"),
    ("C4", "U2", 2, "|++>", prepare_plus_plus, 2, "no"),
    ("C5", "U2", 2, "|++>", prepare_plus_plus, 1, "no"),
]


def build_unitary(name):
    return build_X() if name == "X" else build_U(2)


def qpe_joint_circuit(U, n, m, prepare, method="repeat"):
    """QPE circuit that measures the counting register AND the target register.

    Counting qubit k -> classical bit k (register 'c');
    target qubit j  -> classical bit m + j (register 't').
    """
    qc = qpe_circuit(U, n, m, prepare, method=method, measure=False)
    qc.add_register(ClassicalRegister(n, "t"))
    qc.barrier()
    for k in range(m):
        qc.measure(k, k)
    for j in range(n):
        qc.measure(m + j, m + j)
    return qc


def split_joint_key(key, m, n):
    """Split an Aer key into (counting bits y, target bits x). Target bits are on the left."""
    s = key.replace(" ", "")
    return s[-m:], s[:n]


def joint_table(counts, exact_joint, m, n):
    """Joint (counting, target) outcomes: measured frequency and exact probability."""
    shots = sum(counts.values())
    measured = {}
    for key, value in counts.items():
        measured[split_joint_key(key, m, n)] = value / shots
    rows = []
    for x in range(2**n):
        for y in range(2**m):
            p_exact = float(exact_joint[y + 2**m * x])
            p_meas = measured.get((bits(y, m), bits(x, n)), 0.0)
            if p_exact > 1e-9 or p_meas > 0:
                rows.append({"counting y": bits(y, m), "target": bits(x, n),
                             "P exact": round(p_exact, 4), "measured freq": round(p_meas, 4)})
    return pd.DataFrame(rows)


def conditional_target_states(U, n, m, prepare, method="repeat"):
    """State of the target register after reading y, for every y with P(y) > 0."""
    qc = qpe_circuit(U, n, m, prepare, method=method, measure=False)
    psi = np.asarray(Statevector(qc).data).reshape(2**n, 2**m)   # [target x, counting y]
    out = {}
    for y in range(2**m):
        column = psi[:, y]
        norm = np.linalg.norm(column)
        if norm > 1e-9:
            state = column / norm
            first = state[np.argmax(np.abs(state) > 1e-9)]
            out[bits(y, m)] = state * (abs(first) / first)   # remove global phase
    return out


def format_state(state, n):
    """Write a target state as a sum of basis kets, e.g. '0.707|0> + 0.707|1>'."""
    terms = []
    for x, a in enumerate(state):
        if abs(a) > 1e-9:
            a = complex(np.round(a, 3))
            coeff = f"{a.real:.3f}" if abs(a.imag) < 1e-9 else f"({a.real:.3f}{a.imag:+.3f}i)"
            terms.append(f"{coeff}|{bits(x, n)}>")
    return " + ".join(terms)


def part_c_runs(method="repeat", shots=SHOTS, seed=SEED):
    """Run C1..C5. Each run measures the counting register; a joint version also measures the target."""
    runs = []
    for run_id, uname, n, label, prepare, m, eigen in PART_C_RUNS:
        U = build_unitary(uname)
        counts = run(qpe_circuit(U, n, m, prepare, method=method), shots=shots, seed=seed)
        exact = exact_probabilities(qpe_circuit(U, n, m, prepare, method=method, measure=False), range(m))
        joint_qc = qpe_joint_circuit(U, n, m, prepare, method=method)
        joint_counts = run(joint_qc, shots=shots, seed=seed)
        joint_exact = exact_probabilities(joint_qc, range(m + n))
        runs.append({
            "run": run_id, "U": uname, "n": n, "|v>": label, "m": m, "eigenstate": eigen,
            "counts": dict(counts), "measured": counts_to_probabilities(counts, m), "exact": exact,
            "joint_counts": dict(joint_counts), "joint_exact": joint_exact,
            "conditional": conditional_target_states(U, n, m, prepare, method=method),
        })
    return runs


def part_c_table(runs):
    """The completed Part C table."""
    rows = []
    for r in runs:
        m = r["m"]
        observed = ", ".join(
            f"{bits(y, m)}: {r['measured'][y]:.3f} (exact {r['exact'][y]:.3f})"
            for y in range(2**m) if r["exact"][y] > 1e-9 or r["measured"][y] > 0
        )
        rows.append({"Run": r["run"], "U": r["U"], "n": r["n"], "|v>": r["|v>"], "m": m,
                     "eigenstate of U?": r["eigenstate"],
                     "observed outcomes (measured, exact)": observed})
    return pd.DataFrame(rows)


def plot_joint(runs, filename=None):
    """Bar charts of the joint (counting, target) distributions."""
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 3.8), squeeze=False)
    for ax, r in zip(axes[0], runs):
        table = joint_table(r["joint_counts"], r["joint_exact"], r["m"], r["n"])
        labels = [f"y={a}\nt={b}" for a, b in zip(table["counting y"], table["target"])]
        x = np.arange(len(table))
        ax.bar(x, table["measured freq"], color="#4C72B0", alpha=0.85, label="measured (shots)")
        ax.scatter(x, table["P exact"], color="#DD8452", marker="D", s=30, zorder=3, label="exact")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 0.6)
        ax.set_ylabel("probability")
        ax.set_title(f"{r['run']}: U = {r['U']}, |v> = {r['|v>']}, m = {r['m']}\n"
                     "joint outcome (counting y, target t)", fontsize=10)
    axes[0][0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
    return fig
