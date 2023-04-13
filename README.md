This package provides machine learning framework for dynamic anticipatory mesh optimization (DynAMO).

Currently it has been tested for `Python@3.10.6`, `ray@2.3.1`, and `PyMFEM@4.5.2` on linux (Ubuntu@22.04 LTS)
This package is based on general hyperbolic conservation laws implementation in PyMFEM.

# Installation
Before you install PyMFEM, please install:
```bash
sudo apt install swig build-essential cmake chrpath mpich
```
and
```bash
pip install swig numba mpi4py scipy numpy
```

For installation, please install PyMFEM at a branch `HCL-refactor-flux-Jacobian` by
```bash
python -m pip install git+https://github.com/mfem/pymfem.git@HCL-refactor-flux-Jacobian
```
You can also install parallel version by
```bash
python -m pip install git+https://github.com/mfem/pymfem.git@HCL-refactor-flux-Jacobian --install-option="--with-parallel"
```
Please note that only solver part is implemented in parallel.
Running `ray[rllib]` with `MPI` is future work.

Also, install `ray[rllib]` and `tensorflow`
```bash
pip install 'ray[rllib]' tensorflow
```

# Solver Test

At the top directory,
```bash
python ./run_solvers -solver <solver_name>
```
where `<solver_name>='advection', 'burgers', or 'euler'`.

# Environment Test

TBD

# Training

TBD

# Evaluating

TBD