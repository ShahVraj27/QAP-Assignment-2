"""Tests for the core QPE implementation (Commit 1)."""

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from qpe import (
    SEED,
    basis,
    bits,
    build_mcp_U,
    build_U,
    controlled_power_circuit,
    decode,
    exact_probabilities,
    identity_table,
    phase_to_binary,
    power_by_angles,
    prepare_zero,
    qpe_circuit,
    run,
    theoretical_probabilities,
)
from qiskit import QuantumCircuit

METHODS = ["repeat", "angle"]


def exact_counting_probs(U, n, m, prepare, method):
    qc = qpe_circuit(U, n, m, prepare, method=method, measure=False)
    return exact_probabilities(qc, range(m))


# ---------------------------------------------------------------- sanity check
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("m", [1, 2, 3])
def test_sanity_all_zero_state_returns_zero(n, m, method):
    """|0...0> has phi = 0 for every U_n, so y = 0 with probability 1."""
    probs = exact_counting_probs(build_U(n), n, m, prepare_zero, method)
    assert probs[0] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("m", [1, 2, 3])
def test_sanity_sampled_all_zeros(m):
    qc = qpe_circuit(build_U(3), 3, m, prepare_zero, method="angle")
    counts = run(qc, shots=512, seed=SEED)
    assert counts == {"0" * m: 512}


# ----------------------------------------------------- the two power methods
@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("k", [0, 1, 2])
def test_power_methods_same_unitary_Un(n, k):
    U = build_U(n)
    rep = Operator(controlled_power_circuit(U, k, method="repeat"))
    ang = Operator(controlled_power_circuit(U, k, method="angle"))
    assert rep.equiv(ang)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("k", [0, 1, 2])
def test_power_methods_same_unitary_mcp(n, k):
    U = build_mcp_U(n)
    rep = Operator(controlled_power_circuit(U, k, method="repeat"))
    ang = Operator(controlled_power_circuit(U, k, method="angle"))
    assert rep.equiv(ang)


def test_power_by_angles_matches_matrix_power():
    U = build_U(3)
    for k in range(3):
        expected = np.linalg.matrix_power(Operator(U).data, 2**k)
        assert np.allclose(Operator(power_by_angles(U, k)).data, expected)


def test_angle_method_rejects_non_phase_gates():
    qc = QuantumCircuit(1)
    qc.x(0)
    with pytest.raises(ValueError):
        power_by_angles(qc, 1)


def test_n3_m3_both_methods_same_histogram():
    """Hint 2: both versions give the same histogram for n = 3, m = 3, |101>."""
    U = build_U(3)
    probs_rep = exact_counting_probs(U, 3, 3, basis(5), "repeat")
    probs_ang = exact_counting_probs(U, 3, 3, basis(5), "angle")
    assert np.allclose(probs_rep, probs_ang)
    counts_rep = run(qpe_circuit(U, 3, 3, basis(5), method="repeat"), shots=1024, seed=SEED)
    counts_ang = run(qpe_circuit(U, 3, 3, basis(5), method="angle"), shots=1024, seed=SEED)
    assert counts_rep == counts_ang == {"101": 1024}


# ------------------------------------------------ correctness of the estimate
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n, x", [(1, 1), (2, 3), (3, 5), (4, 11), (4, 8)])
@pytest.mark.parametrize("m", [1, 2, 3])
def test_statevector_matches_formula(n, x, m, method):
    """Exact probabilities from the circuit equal the analytic P(y) of Section 1."""
    probs = exact_counting_probs(build_U(n), n, m, basis(x), method)
    assert np.allclose(probs, theoretical_probabilities(x / 2**n, m), atol=1e-9)


def test_basis_state_bit_order():
    """|101> (x=5) puts X on target qubits 0 and 2, matching Qiskit order."""
    qc = QuantumCircuit(3)
    basis(5)(qc, [0, 1, 2])
    probs = exact_probabilities(qc, [0, 1, 2])
    assert probs[5] == pytest.approx(1.0)


def test_exact_case_returns_true_bits():
    qc = qpe_circuit(build_U(3), 3, 3, basis(5), method="angle")
    counts = run(qc, shots=256, seed=SEED)
    assert list(counts) == ["101"]
    assert decode("101") == 5 / 8


# ------------------------------------------------------------------ helpers
def test_formula_sums_to_one():
    for phi in [0.0, 1 / 3, 0.6875, 0.75]:
        for m in [1, 2, 3]:
            assert theoretical_probabilities(phi, m).sum() == pytest.approx(1.0)


def test_formula_lower_bound_closest_estimate():
    """The closest estimate has probability at least 4/pi^2 (Section 1)."""
    for phi in np.linspace(0, 1, 97, endpoint=False):
        for m in [1, 2, 3]:
            p = theoretical_probabilities(phi, m)
            closest = int(np.round(phi * 2**m)) % 2**m
            assert p[closest] >= 4 / np.pi**2 - 1e-9


def test_helpers():
    assert bits(5, 3) == "101"
    assert bits(1, 3) == "001"
    assert phase_to_binary(11 / 16, 4) == "0.1011"
    assert phase_to_binary(1 / 3, 6) == "0.010101"
    assert decode("11") == 0.75


def test_identity_table_n3():
    """For n = 3: k=0 -> T,S,Z ; k=1 -> S,Z,I ; k=2 -> Z,I,I."""
    table = identity_table(3, 3)
    assert list(table.iloc[0]) == ["π/4", "π/2", "π"]
    assert list(table.iloc[1]) == ["π/2", "π", "0 (I)"]
    assert list(table.iloc[2]) == ["π", "0 (I)", "0 (I)"]
