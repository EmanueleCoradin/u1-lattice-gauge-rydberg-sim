# U(1) Lattice Gauge Theory in Rydberg Atom Simulators

This repository contains code for simulating **U(1) lattice gauge theories** and **string dynamics** in Rydberg atom quantum simulators, following recent approaches in **quantum simulation** and **quantum link models**.

The project implements:

- Exact diagonalization of lattice Hamiltonians  
- Blockade effects in Rydberg chains  
- Tools for analyzing string breaking, quarkonium dynamics, and periodicity  
- Visualization scripts for generating figures used in presentations/reports  

---

## 📂 Project Structure

```
u1-lattice-gauge-rydberg-sim/
│
├── main.ipynb              # Jupyter notebook for running experiments interactively
├── points_selector.py      # Script to preselect points from the scatterplot of the overlaps vs eigenvalues
├── plot_selected.py        # Script for plotting the results of the points_selector
├── lattice.yml             # To quickly setup the conda environment
├── requirements.txt        # Python dependencies
├── README.md               # (this file)
├── .gitignore
│
└── quantkit/               # Core simulation library
    ├── __init__.py
    ├── blockade.py         # Implements blockade constraints in Rydberg chains
    ├── dynamics.py         # Time evolution and exact diagonalization routines
    ├── information.py      # Entanglement entropy, overlaps, and info-theoretic quantities
    ├── operators.py        # Hamiltonians, ladder operators, Pauli matrices, observables
    ├── plotting.py         # Visualization helpers (string dynamics, eigenstate overlaps)
    ├── states.py           # State preparation (vacuum, quarkonium, excitations)
```

---

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone <repo-url>
cd u1-lattice-gauge-rydberg-sim
pip install -r requirements.txt
```
or use conda

```bash
git clone <repo-url>
cd u1-lattice-gauge-rydberg-sim
conda env create -f lattice.yml
conda activate lattice
```

---

## 🧑‍💻 Usage


**Explore interactively:**

Open `main.ipynb` in JupyterLab or VSCode for step-by-step simulations.

---

## 📚 References

This code builds on methods from:

- F. M. Surace *et al.*, *Lattice Gauge Theories and String Dynamics in Rydberg Atom Quantum Simulators*, *Phys. Rev. X* **10**, 021041 (2020)
- https://baltig.infn.it/qpd/quantum_information_computing_2425
