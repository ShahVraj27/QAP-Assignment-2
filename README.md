# QAP Assignment 2: Quantum Fourier Transform & Phase Estimation

Quantum Phase Estimation (QPE) in Qiskit on eigenstates of 1 to 4 qubits with
m = 1, 2 and 3 counting qubits. Every result is explained from the math.

| Contributor | Student ID |
|---|---|
| Vraj Shah | 2023A7PS0478G |
| Soumya Dedhia | 2023A7PS0381G |

## Submission files

| File | Contents |
|---|---|
| `notebooks/Assignment2_QPE.ipynb` | the graded notebook, executed, with all outputs |
| `submission/Assignment2_QPE.pdf` | PDF export of the executed notebook |

## What is in the notebook

| Section | Contents |
|---|---|
| 1–2 | background, completed `qpe_circuit`, both U^(2^k) methods |
| 3 | \|0...0> sanity check for n = 1..4, m = 1..3 |
| 4–5 | repetition vs angle multiplication for n = 3, m = 3; which gates vanish in U^(2^k) |
| 6 | circuit diagram for n = 2, m = 2 with the kickback gates labelled |
| 7 | state vector vs analytic P(y) for every Part A configuration |
| 8 | Part A: predictions, 15 runs, histograms, P(y) comparison for non-exact runs |
| 9 | Part B: phase 1/3, 12 runs, error bound, comparison across n, \|1110> run |
| 10 | Part C: non-eigenstates C1–C5, eigenstate decompositions, joint outcomes |
| 11 | Part D: broken circuits D1–D6 and the D2+D3 combination |
| 12 | inference essay |
| 13 | deliverables checklist |

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
│   ├── part_c.py             # Part C: non-eigenstates, joint measurement
│   ├── part_d.py             # Part D: broken-circuit variants D1–D6
│   └── plotting.py           # side-by-side histograms
├── tests/
│   ├── test_core.py          # sanity check, power-method equivalence, formula checks
│   ├── test_part_a_b.py      # Part A predictions/results, Part B error bound
│   └── test_part_c_d.py      # Part C distributions, Part D bug signatures
├── notebooks/
│   └── Assignment2_QPE.ipynb # the graded notebook
├── results/
│   ├── figures/              # PNG figures written by the notebook
│   └── tables/               # CSV tables written by the notebook
├── submission/
│   └── Assignment2_QPE.pdf   # PDF export of the notebook
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

The suite has 118 tests covering the sanity check, the equivalence of the two
U^(2^k) methods, the analytic P(y) formula, every Part A/B result, the Part C
distributions and the signature of every Part D bug.

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
- Exact probabilities come from `Statevector` with the measurements removed;
  differences from the histograms are shot noise.

## Commit history

| Commit | Contributor | Contents |
|---|---|---|
| 1 | Vraj Shah | project setup, `qpe_circuit`, both power methods, sanity check, n=2 m=2 diagram |
| 2 | Soumya Dedhia | Part A and Part B |
| 3 | Vraj Shah | Part C and Part D |
| 4 | Soumya Dedhia | inference essay, final README, PDF export |
