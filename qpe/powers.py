"""Two ways of building the controlled U^(2^k) used in QPE step 3 (Hint 2).

* Repetition: append controlled-U 2^k times. Works for any U, but the circuit
  needs 2^m - 1 controlled-U calls in total.
* Angle multiplication: for a U made only of phase gates P(theta) and
  multi-controlled phase gates MCP(theta), U^(2^k) is the same circuit with
  every angle multiplied by 2^k. Same unitary, constant depth in k.
"""

import numpy as np
from qiskit import QuantumCircuit

PHASE_GATE_NAMES = ("p", "mcphase")
TWO_PI = 2 * np.pi


def reduce_angle(theta):
    """Map an angle into [0, 2*pi). Multiples of 2*pi become exactly 0."""
    angle = float(np.mod(theta, TWO_PI))
    if np.isclose(angle, TWO_PI) or np.isclose(angle, 0.0):
        return 0.0
    return angle


def power_by_angles(U, k):
    """Return U^(2^k) as a new circuit by multiplying every phase angle by 2^k.

    Only P and MCP gates are allowed in U (true for U_n and for MCP(theta)).
    Angles are reduced modulo 2*pi, so a gate that becomes the identity is
    shown as P(0) (Hint 3).
    """
    out = QuantumCircuit(U.num_qubits, name=f"{U.name}^{2**k}")
    for instruction in U.data:
        op = instruction.operation
        qubits = [U.find_bit(q).index for q in instruction.qubits]
        if op.name not in PHASE_GATE_NAMES:
            raise ValueError(
                f"Angle multiplication only works for P/MCP gates, found '{op.name}'. "
                "Use method='repeat' for this U."
            )
        angle = reduce_angle(2**k * float(op.params[0]))
        if op.name == "p":
            out.p(angle, qubits[0])
        else:
            out.mcp(angle, qubits[:-1], qubits[-1])
    return out


def controlled_U(U, label="U"):
    """Controlled-U as a gate (control is the first qubit), as in Hint 1."""
    return U.to_gate(label=label).control(1)


def controlled_power_angle(U, k):
    """Controlled U^(2^k) built from multiplied angles, labelled U^(2^k)."""
    return power_by_angles(U, k).to_gate(label=f"U^{2**k}").control(1)


def append_power_repeat(qc, U, k, control, target):
    """Append controlled-U 2^k times (repetition method)."""
    cu = controlled_U(U)
    for _ in range(2**k):
        qc.append(cu, [control] + list(target))


def append_power_angle(qc, U, k, control, target):
    """Append one controlled U^(2^k) (angle-multiplication method)."""
    qc.append(controlled_power_angle(U, k), [control] + list(target))


def append_controlled_power(qc, U, k, control, target, method="repeat"):
    """Append controlled U^(2^k) to qc using the chosen method."""
    if method == "repeat":
        append_power_repeat(qc, U, k, control, target)
    elif method == "angle":
        append_power_angle(qc, U, k, control, target)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'repeat' or 'angle'.")


def controlled_power_circuit(U, k, method="repeat"):
    """Return a (1 + n)-qubit circuit holding only controlled U^(2^k).

    Qubit 0 is the control; qubits 1..n are the target. Used to compare the
    two methods as unitaries and by circuit depth.
    """
    n = U.num_qubits
    qc = QuantumCircuit(1 + n)
    append_controlled_power(qc, U, k, 0, list(range(1, 1 + n)), method=method)
    return qc
