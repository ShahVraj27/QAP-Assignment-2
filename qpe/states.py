"""State-preparation functions with the signature prepare(qc, target).

They are passed directly as the ``prepare`` argument of qpe_circuit.
"""

from .core import prepare_basis_state


def basis(x):
    """Return a preparer for the basis state |x> (bit j of x on target qubit j)."""
    def prepare(qc, target):
        prepare_basis_state(qc, target, x)
    prepare.__name__ = f"basis_{x}"
    return prepare


def all_ones(n):
    """Return a preparer for |1...1> on n qubits (x = 2^n - 1)."""
    return basis(2**n - 1)


def prepare_plus(qc, target):
    """|+> on every target qubit: H."""
    for q in target:
        qc.h(q)


def prepare_minus(qc, target):
    """|-> on every target qubit: X then H."""
    for q in target:
        qc.x(q)
        qc.h(q)


def prepare_zero(qc, target):
    """|0...0>: nothing to do."""
    return None


def prepare_plus_plus(qc, target):
    """|++> = (H (x) H)|00> on a 2-qubit target."""
    prepare_plus(qc, target)
