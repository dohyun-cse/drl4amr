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
sudo apt install swig build-essential cmake chrpath mpich zlib1g-dev libffi-dev
```
> **_NOTE:_**  Please make sure that you have `swig@4.1.1` installed on your system.

Install `python@3.8.2` if you don't have.
```bash
wget https://www.python.org/ftp/python/3.8.2/Python-3.8.2.tgz
tar xzf Python-3.8.2.tgz
cd Python-3.8.2
sudo ./configure --prefix=/opt/python/3.8.2/ --enable-optimizations --with-lto --with-computed-gotos --with-system-ffi
sudo make -j "$(nproc)"
sudo make altinstall
```

Create a virtual environment if you need. Then, the following will install the `hcl` package along with dependencies except `mfem`.
```bash
python -m pip install -e .
```

Then, install `PyMFEM` using
```bash
python -m pip install git+https://github.com/mfem/pymfem.git@HCL-refactor-flux-Jacobian
```
> **_NOTE:_** On a fresh Ubuntu@22.04, we need to restart the system before installing `mfem`.


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