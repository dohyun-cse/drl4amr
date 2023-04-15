This package provides a machine-learning framework for dynamic anticipatory mesh optimization (DynAMO).

Currently, it has been tested for `Python@3.10.6`, `ray@2.3.1`, and `PyMFEM@4.5.2` on Linux (Ubuntu@22.04 LTS)
This package is based on general hyperbolic conservation laws implementation in PyMFEM.

---

# Algorithm Preamble

## Overview

## Solver

## Environment

## Train

---

# Installation
If your system is fresh and has never installed PyMFEM before, please install the followings:
```bash
sudo apt install swig build-essential cmake chrpath mpich python-is-python3 python3-pip
```
> **_NOTE:_**  Please make sure that you have `swig@4.1.1` installed on your system.

To install dependent packages, run
```bash
python -m pip install -r requirements.txt
```

You also need PyMFEM at a branch `HCL-refactor-flux-Jacobian`:
```bash
python -m pip install git+https://github.com/mfem/pymfem.git@HCL-refactor-flux-Jacobian
```
<!-- You can also install a parallel version by
```bash
python -m pip install git+https://github.com/mfem/pymfem.git@HCL-refactor-flux-Jacobian --install-option="--with-parallel"
```
Please note that only the solver part is implemented in parallel.
Running `ray[rllib]` with `MPI` is future work. -->

---

# Solver Test

At the top directory,
```bash
python ./run_solvers.py -solver <solver_name>
```
where `<solver_name> = 'advection', 'burgers', or 'euler'`.

# Environment Test

TBD

# Training

TBD

# Evaluating

TBD