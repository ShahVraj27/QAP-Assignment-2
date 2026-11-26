"""Tests for Part C (non-eigenstates) and Part D (broken circuits) (Commit 3)."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qpe import basis, build_U, exact_probabilities
from qpe.part_c import joint_table, part_c_runs
from qpe.part_d import inverse_qft_variant, part_d_runs, qft, qpe_variant


@pytest.fixture(scope="module")
def c_runs():
    return {r["run"]: r for r in part_c_runs()}


@pytest.fixture(scope="module")
def d_runs():
    return {r["run"]: r for r in part_d_runs()}


# ------------------------------------------------------------------ Part C
def test_c1_minus_gives_half(c_runs):
    assert c_runs["C1"]["exact"][2] == pytest.approx(1.0)          # y = 10 -> 1/2


def test_c2_plus_gives_zero(c_runs):
    assert c_runs["C2"]["exact"][0] == pytest.approx(1.0)          # y = 00 -> 0


def test_c3_zero_splits(c_runs):
    assert np.allclose(c_runs["C3"]["exact"], [0.5, 0, 0.5, 0], atol=1e-9)


def test_c3_target_collapses_to_plus_or_minus(c_runs):
    cond = c_runs["C3"]["conditional"]
    assert np.allclose(cond["00"], [1 / np.sqrt(2), 1 / np.sqrt(2)])
    assert np.allclose(cond["10"], [1 / np.sqrt(2), -1 / np.sqrt(2)])


def test_c4_uniform_and_perfectly_correlated(c_runs):
    r = c_runs["C4"]
    assert np.allclose(r["exact"], 0.25)
    table = joint_table(r["joint_counts"], r["joint_exact"], 2, 2)
    assert (table["counting y"] == table["target"]).all()
    assert np.allclose(table["P exact"], 0.25)
    for y, state in r["conditional"].items():
        expected = np.zeros(4)
        expected[int(y, 2)] = 1
        assert np.allclose(np.abs(state), expected)


def test_c5_one_qubit_cannot_separate_four_phases(c_runs):
    r = c_runs["C5"]
    assert np.allclose(r["exact"], [0.5, 0.5])
    # P(y=0 | x) = cos^2(pi * x / 4): 1, 1/2, 0, 1/2 -> joint 1/4, 1/8, 0, 1/8
    joint = r["joint_exact"].reshape(4, 2)          # [x, y]
    assert np.allclose(joint[:, 0], [0.25, 0.125, 0.0, 0.125])
    assert np.allclose(joint[:, 1], [0.0, 0.125, 0.25, 0.125])


# ------------------------------------------------------------------ Part D
def test_forward_qft_is_inverse_of_starter():
    qc = QuantumCircuit(3)
    qft(qc, [0, 1, 2])
    inverse_qft_variant(qc, [0, 1, 2])
    assert np.allclose(Operator(qc).data, np.eye(8))


def test_correct_circuit(d_runs):
    assert d_runs["correct"]["exact"][5] == pytest.approx(1.0)


def test_d1_matches_formula(d_runs):
    """Every qubit carries phi: amplitude of y is (1/8) sum_j exp(2 pi i (phi*popcount(j) - j*y/8))."""
    phi = 5 / 8
    j = np.arange(8)
    pop = np.array([bin(v).count("1") for v in j])
    expected = [abs(np.exp(2j * np.pi * (phi * pop - j * y / 8)).sum() / 8) ** 2 for y in range(8)]
    assert np.allclose(d_runs["D1"]["exact"], expected)


def test_d2_equals_d3(d_runs):
    assert np.allclose(d_runs["D2"]["exact"], d_runs["D3"]["exact"])
    assert d_runs["D2"]["exact"][5] == pytest.approx((4 + 2 * np.sqrt(2)) ** 2 / 64)


def test_d2_plus_d3_is_correct(d_runs):
    assert d_runs["D2+D3"]["exact"][5] == pytest.approx(1.0)


def test_d2_is_not_a_plain_bit_reversal():
    """For |001> the no-swap circuit peaks at the reversed string 100, but not with certainty."""
    p = exact_probabilities(qpe_variant(build_U(3), 3, 3, basis(1), swaps=False, measure=False), range(3))
    assert int(np.argmax(p)) == 4
    assert p[4] < 0.5


def test_d4_uniform(d_runs):
    assert np.allclose(d_runs["D4"]["exact"], 1 / 8)


def test_d5_negated_phase(d_runs):
    assert d_runs["D5"]["exact"][3] == pytest.approx(1.0)          # 011 = 3/8 = 1 - 5/8


def test_d6_prepares_110(d_runs):
    assert d_runs["D6"]["exact"][6] == pytest.approx(1.0)          # 110 = 6/8
