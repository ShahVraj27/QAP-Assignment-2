"""Core Quantum Phase Estimation (QPE) building blocks.

build_U, prepare_basis_state, inverse_qft, run and decode are the starter
helpers from Section 4 of the assignment. They are unchanged, except that
run() also takes an optional ``seed`` so histograms are reproducible.

qpe_circuit is the starter skeleton with its four TODOs completed. It
also takes ``method`` (how U^(2^k) is built) and ``measure`` (set False to
get a measurement-free circuit for exact Statevector probabilities).
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from .powers import append_controlled_power


def build_U(n):
    """Binary-fraction phase oracle: U|x> = exp(2*pi*i*x/2^n)|x>."""
    qc = QuantumCircuit(n, name=f"U{n}")
    for j in range(n):                       # j = 0 is the least-significant bit
        qc.p(2 * np.pi * 2**j / 2**n, j)
    return qc


def prepare_basis_state(qc, target, x):
    """Apply X to target qubit j whenever bit j of x is 1."""
    for j, q in enumerate(target):
        if (x >> j) & 1:
            qc.x(q)


def inverse_qft(qc, qubits):
    """Inverse QFT (starter code). Assumes counting qubit k controlled U^(2^k)."""
    m = len(qubits)
    for i in range(m // 2):
        qc.swap(qubits[i], qubits[m - 1 - i])
    for j in range(m):
        for k in range(j):
            qc.cp(-np.pi / 2**(j - k), qubits[k], qubits[j])
        qc.h(qubits[j])


def qpe_circuit(U, n, m, prepare, method="repeat", measure=True):
    """Build the QPE circuit.

    Parameters
    ----------
    U : QuantumCircuit
        n-qubit unitary whose eigenphase we estimate.
    n : int
        Number of target qubits.
    m : int
        Number of counting qubits.
    prepare : callable(qc, target)
        Prepares |v> on the target qubits.
    method : {"repeat", "angle"}
        "repeat": counting qubit k applies controlled-U 2^k times (works for any U).
        "angle":  counting qubit k applies one controlled U^(2^k) built by
                  multiplying every phase angle by 2^k (U made of P / MCP gates only).
    measure : bool
        If True, measure counting qubit k into classical bit k.
    """
    qc = QuantumCircuit(m + n, m)
    count = list(range(m))           # counting qubits 0 .. m-1
    target = list(range(m, m + n))   # target qubits m .. m+n-1
    prepare(qc, target)
    qc.barrier()

    # TODO 1: H on every counting qubit
    for q in count:
        qc.h(q)

    # TODO 2: counting qubit k controls U^(2^k)    (phase kickback)
    for k in range(m):
        append_controlled_power(qc, U, k, count[k], target, method=method)
    qc.barrier()

    # TODO 3: inverse_qft on the counting register
    inverse_qft(qc, count)

    # TODO 4: measure counting qubit k into classical bit k
    if measure:
        qc.barrier()
        for k in range(m):
            qc.measure(count[k], k)
    return qc


def run(qc, shots=2048, seed=None):
    """Sample the circuit on AerSimulator and return the counts dictionary."""
    sim = AerSimulator()
    if seed is None:
        job = sim.run(transpile(qc, sim), shots=shots)
    else:
        job = sim.run(transpile(qc, sim), shots=shots, seed_simulator=seed)
    return job.result().get_counts()


def decode(bitstring):
    """Convert a measured bit string y into the phase estimate y / 2^m."""
    return int(bitstring, 2) / 2**len(bitstring)
