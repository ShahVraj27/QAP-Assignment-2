"""Unitaries used in the assignment.

* build_U(n)       : binary-fraction phase oracle U_n (Part A; U_2 is also used in Part C)
* build_mcp_U(n)   : multi-controlled phase MCP(2*pi/3) on n qubits (Part B)
* build_X()        : single-qubit X (Part C)
"""

import numpy as np
from qiskit import QuantumCircuit

from .core import build_U

PART_B_THETA = 2 * np.pi / 3


def build_mcp_U(n, theta=PART_B_THETA):
    """MCP(theta) on n qubits: adds phase theta only when every qubit is 1.

    For n = 1 it is the single-qubit phase gate P(theta).
    """
    qc = QuantumCircuit(n, name=f"MCP{n}")
    if n == 1:
        qc.p(theta, 0)
    else:
        qc.mcp(theta, list(range(n - 1)), n - 1)
    return qc


def build_X():
    """Single-qubit Pauli X as a circuit (eigenstates |+> with phi=0, |-> with phi=1/2)."""
    qc = QuantumCircuit(1, name="X")
    qc.x(0)
    return qc


__all__ = ["build_U", "build_mcp_U", "build_X", "PART_B_THETA"]
