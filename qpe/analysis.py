"""Exact probabilities, the analytic P(y) formula and helper tables."""

from fractions import Fraction

import numpy as np
import pandas as pd
from qiskit.quantum_info import Statevector

from .powers import reduce_angle


def bits(y, m):
    """Integer y as an m-bit string (classical bit 0 on the right)."""
    return format(int(y), f"0{m}b")


def phase_to_binary(phi, digits):
    """Binary expansion 0.b1b2... of phi, truncated to ``digits`` digits."""
    out = []
    frac = float(phi)
    for _ in range(digits):
        frac *= 2
        bit = int(frac)
        out.append(str(bit))
        frac -= bit
    return "0." + "".join(out)


def exact_probabilities(qc, qubits):
    """Exact outcome probabilities of ``qubits`` (index = y, qubits[0] is the LSB).

    Final measurements are removed first, as Section 4 suggests.
    """
    clean = qc.remove_final_measurements(inplace=False)
    return np.asarray(Statevector(clean).probabilities(list(qubits)))


def theoretical_probabilities(phi, m):
    """Analytic P(y) = |(1/2^m) sum_j exp(2*pi*i*j*(phi - y/2^m))|^2 for y = 0..2^m-1."""
    N = 2**m
    j = np.arange(N)
    y = np.arange(N)
    amplitudes = np.exp(2j * np.pi * np.outer(phi - y / N, j)).sum(axis=1) / N
    return np.abs(amplitudes) ** 2


def counts_to_probabilities(counts, m):
    """Convert an Aer counts dict into an array of frequencies indexed by y."""
    shots = sum(counts.values())
    probs = np.zeros(2**m)
    for key, value in counts.items():
        probs[int(key.replace(" ", ""), 2)] = value / shots
    return probs


def top_outcomes(probs, k=2):
    """The k most likely outcomes as a list of (y, probability), largest first."""
    order = np.argsort(-np.asarray(probs), kind="stable")[:k]
    return [(int(y), float(probs[y])) for y in order]


def is_exact(probs, tol=1e-9):
    """True if a single outcome has probability 1."""
    return bool(np.max(probs) > 1 - tol)


def format_angle(theta):
    """Show an angle as a multiple of pi, e.g. 'pi/4' or '0 (identity)'."""
    theta = reduce_angle(theta)
    if theta == 0.0:
        return "0 (I)"
    frac = Fraction(theta / np.pi).limit_denominator(64)
    num, den = frac.numerator, frac.denominator
    top = "π" if num == 1 else f"{num}π"
    return top if den == 1 else f"{top}/{den}"


def identity_table(n, m):
    """Hint 3: angle of P(2^k * theta_j) on each target qubit j for every k.

    Rows: counting qubit k (which applies U^(2^k)). Columns: target qubit j.
    Entries marked 'I' are multiples of 2*pi, i.e. the gate is the identity.
    """
    rows = {}
    for k in range(m):
        rows[f"k={k}  (U^{2**k})"] = {
            f"qubit {j}": format_angle(2**k * 2 * np.pi * 2**j / 2**n) for j in range(n)
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def probability_comparison_table(phi, m, exact, measured):
    """Table of analytic P(y), exact Statevector P(y) and measured frequency."""
    theory = theoretical_probabilities(phi, m)
    return pd.DataFrame(
        {
            "y (bits)": [bits(y, m) for y in range(2**m)],
            "phi_tilde": [y / 2**m for y in range(2**m)],
            "P(y) formula": np.round(theory, 4),
            "P(y) statevector": np.round(exact, 4),
            "measured freq": np.round(measured, 4),
        }
    )
