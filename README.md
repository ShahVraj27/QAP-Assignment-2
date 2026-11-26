# QAP Assignment 2: Quantum Fourier Transform & Phase Estimation

Quantum Phase Estimation (QPE) in Qiskit on eigenstates of 1 to 4 qubits with
m = 1, 2 and 3 counting qubits. Every result is explained from the math.

| Contributor | Student ID |
|---|---|
| Vraj Shah | 2023A7PS0478G |
| Soumya Dedhia | 2023A7PS0381G |

## Status

| Part | Contents | Status |
|---|---|---|
| Implementation | completed `qpe_circuit`, both U^(2^k) methods, \|0...0> sanity check, n=2 m=2 diagram | done |
| Part A | core grid n = 1..4, m = 1..3: predictions, 15 runs, histograms, P(y) comparison | done |
| Part B | phase 1/3 with MCP(2π/3): 12 runs, error bound, n-comparison, \|1110> run | done |
| Part C | non-eigenstates | to do |
| Part D | broken circuits D1–D6 | to do |
| Essay and PDF | inference essay, final PDF export | to do |

## Technologies

- Python 3.11 or 3.12
- Qiskit 2.x and Qiskit Aer 0.17 (`AerSimulator`)
- NumPy, pandas, Matplotlib (with `pylatexenc` for circuit drawings)
- pytest for tests, Jupyter (VS Code) for the notebook

## Project structure

```
QAP-Assignment-2/
├── qpe/                      # all circuit code (single source of truth)
│   ├── core.py               # starter helpers + completed qpe_circuit
│   ├── powers.py             # U^(2^k): repetition and angle multiplication
│   ├── oracles.py            # U_n, MCP(2π/3), X
│   ├── states.py             # state preparers: basis, |+>, |->, |++>
│   ├── analysis.py           # exact probabilities, analytic P(y), tables
│   ├── experiments.py        # Part A and Part B runners and result tables
│   └── plotting.py           # side-by-side histograms
├── tests/
│   ├── test_core.py          # sanity check, power-method equivalence, formula checks
│   └── test_part_a_b.py      # Part A predictions/results, Part B error bound
├── notebooks/
│   └── Assignment2_QPE.ipynb # the graded notebook
├── results/
│   ├── figures/              # PNG figures written by the notebook
│   └── tables/               # CSV tables written by the notebook
├── submission/               # final PDF export
├── requirements.txt
├── pytest.ini
└── .gitignore
```

## Installation (Windows, PowerShell)

```powershell
git clone https://github.com/ShahVraj27/QAP-Assignment-2.git
cd QAP-Assignment-2
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell refuses to run `Activate.ps1`, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## Running the tests

From the project root with the virtual environment active:

```powershell
python -m pytest
```

## Running the notebook

1. Open the `QAP-Assignment-2` folder in VS Code.
2. Open `notebooks/Assignment2_QPE.ipynb`.
3. Click **Select Kernel**, then **Python Environments**, then `.venv`.
4. Click **Restart**, then **Run All**.

The notebook writes its figures to `results/figures/` and its tables to `results/tables/`.

## Conventions

- Counting qubits are `0 .. m-1`, target qubits are `m .. m+n-1`.
- Counting qubit k controls U^(2^k) and is measured into classical bit k, so
  `int(bitstring, 2)` is y directly and the estimate is y / 2^m.
- Kets are written in Qiskit order (highest qubit on the left).
- All sampled runs use 2048 shots and simulator seed 1234.
