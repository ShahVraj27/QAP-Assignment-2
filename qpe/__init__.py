"""QPE assignment package: Quantum Fourier Transform & Phase Estimation."""

from .core import build_U, decode, inverse_qft, prepare_basis_state, qpe_circuit, run
from .powers import (
    append_controlled_power,
    controlled_power_circuit,
    controlled_U,
    power_by_angles,
)
from .oracles import build_mcp_U, build_X
from .states import all_ones, basis, prepare_minus, prepare_plus, prepare_plus_plus, prepare_zero
from .analysis import (
    bits,
    counts_to_probabilities,
    exact_probabilities,
    identity_table,
    is_exact,
    phase_to_binary,
    probability_comparison_table,
    theoretical_probabilities,
    top_outcomes,
)
from .plotting import plot_side_by_side

SHOTS = 2048
SEED = 1234
