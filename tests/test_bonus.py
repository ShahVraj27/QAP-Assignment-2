"""Tests for the optional bonus (noisy fake backend)."""

import numpy as np
import pytest

pytest.importorskip("qiskit_ibm_runtime")

from qpe.bonus import bonus_runs, bonus_table, load_fake_backend  # noqa: E402


@pytest.fixture(scope="module")
def backend():
    return load_fake_backend()


@pytest.fixture(scope="module")
def runs(backend):
    return bonus_runs(backend, shots=1024)


def test_backend_is_large_enough(backend):
    assert backend.num_qubits >= 7


def test_four_runs(runs):
    assert [(r["run"], r["method"]) for r in runs] == [
        ("A9", "repeat"), ("A9", "angle"), ("A12", "repeat"), ("A12", "angle")]


def test_ideal_distributions_unchanged(runs):
    for r in runs:
        if r["run"] == "A9":
            assert r["ideal"][5] == pytest.approx(1.0)
        else:
            assert r["ideal"][5] == pytest.approx(r["ideal"][6])


def test_noisy_counts_are_complete(runs):
    for r in runs:
        assert sum(r["counts"].values()) == 1024
        assert r["noisy"].sum() == pytest.approx(1.0)


def test_angle_method_is_shallower(runs):
    by = {(r["run"], r["method"]): r for r in runs}
    for run_id in ["A9", "A12"]:
        assert by[(run_id, "angle")]["depth"] < by[(run_id, "repeat")]["depth"]
        assert by[(run_id, "angle")]["two_qubit"] < by[(run_id, "repeat")]["two_qubit"]


def test_table_columns(runs):
    table = bonus_table(runs)
    assert len(table) == 4
    assert (table["TVD noisy vs ideal"] >= 0).all()
    assert np.all(table["P(correct) noisy"] <= 1.0)
