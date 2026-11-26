"""Tests for Part A (core grid) and Part B (phase 1/3) (Commit 2)."""

import numpy as np
import pytest

from qpe.analysis import is_exact, theoretical_probabilities
from qpe.experiments import (
    bits_needed,
    part_a_predictions,
    part_a_runs,
    part_a_table,
    part_b_extra_run,
    part_b_runs,
    part_b_table,
    predict_exact,
)


@pytest.fixture(scope="module")
def a_runs():
    return part_a_runs()


@pytest.fixture(scope="module")
def b_runs():
    return part_b_runs()


# ------------------------------------------------------------------ Part A
def test_bits_needed():
    assert bits_needed(1 / 2) == 1
    assert bits_needed(3 / 4) == 2
    assert bits_needed(5 / 8) == 3
    assert bits_needed(11 / 16) == 4
    assert bits_needed(0.0) == 0


def test_part_a_has_15_runs(a_runs):
    assert [r["run"] for r in a_runs] == [f"A{i}" for i in range(1, 16)]


def test_predictions_match_outcomes(a_runs):
    """Prediction (phi has <= m binary digits) agrees with the exact result for every run."""
    for r in a_runs:
        assert r["prediction"] == is_exact(r["exact"]), r["run"]


def test_prediction_column():
    pred = part_a_predictions()
    exact_runs = set(pred.loc[pred["prediction: exact?"] == "yes", "Run"])
    assert exact_runs == {"A1", "A2", "A3", "A5", "A6", "A9", "A13", "A14", "A15"}


def test_exact_runs_return_true_phase(a_runs):
    table = part_a_table(a_runs)
    for _, row in table[table["exact?"] == "yes"].iterrows():
        run = next(r for r in a_runs if r["run"] == row["Run"])
        assert row["phi_tilde"] == pytest.approx(run["phi"])
        assert row["P(most likely) measured"] == pytest.approx(1.0)


def test_non_exact_runs_match_formula_and_bound(a_runs):
    for r in a_runs:
        if not r["prediction"]:
            theory = theoretical_probabilities(r["phi"], r["m"])
            assert np.allclose(r["exact"], theory, atol=1e-9)
            assert theory.max() >= 4 / np.pi**2 - 1e-9


def test_tied_runs(a_runs):
    """A4 splits 50/50; A8 and A12 have two equally likely nearest estimates."""
    by_id = {r["run"]: r for r in a_runs}
    assert np.allclose(by_id["A4"]["exact"], [0.5, 0.5])
    assert by_id["A8"]["exact"][2] == pytest.approx(by_id["A8"]["exact"][3])
    assert by_id["A12"]["exact"][5] == pytest.approx(by_id["A12"]["exact"][6])


def test_shot_noise_is_small(a_runs):
    for r in a_runs:
        assert np.max(np.abs(r["measured"] - r["exact"])) < 0.05, r["run"]


# ------------------------------------------------------------------ Part B
def test_part_b_has_12_runs(b_runs):
    assert [r["run"] for r in b_runs] == [f"B{i}" for i in range(1, 13)]


def test_part_b_never_exact(b_runs):
    for r in b_runs:
        assert not predict_exact(r["phi"], r["m"])
        assert not is_exact(r["exact"])


def test_part_b_error_bound(b_runs):
    table = part_b_table(b_runs)
    assert table["error <= bound"].all()


def test_part_b_independent_of_n(b_runs):
    """For a fixed m the exact distribution is identical for n = 1..4."""
    for m in [1, 2, 3]:
        dists = [r["exact"] for r in b_runs if r["m"] == m]
        for d in dists[1:]:
            assert np.allclose(d, dists[0])
        assert np.allclose(dists[0], theoretical_probabilities(1 / 3, m))


def test_part_b_1110_returns_zero():
    r = part_b_extra_run()
    assert r["exact"][0] == pytest.approx(1.0)
    assert r["counts"] == {"000": 2048}
