import sys
if 'mfem.ser' in sys.modules:
    MFEM_USE_MPI = False
    import mfem.ser as mfem
    from mfem.ser import \
        getAdvectionEquation, getBurgersEquation, getShallowWaterEquation, getEulerSystem, \
        RusanovFlux, RiemannSolver, DGHyperbolicConservationLaws
else:
    MFEM_USE_MPI = True
    from mpi4py import MPI
    import mfem.par as mfem
    from mfem.par import \
        getAdvectionEquation, getBurgersEquation, getShallowWaterEquation, getEulerSystem, \
        RusanovFlux, RiemannSolver, DGHyperbolicConservationLaws

import numpy as np

class Solver:
    def __init__(self, mesh: mfem.Mesh, order: int, num_equations: int, refinement_mode: str, ode_solver: mfem.ODESolver, cfl, terminal_time, **kwargs):
        self.order = order
        self.max_order = order
        self.num_equations = num_equations
        self.refinement_mode = refinement_mode
        self.sdim = mesh.Dimension()
        self.vdim = num_equations
        self.fec = mfem.DG_FECollection(order, self.sdim)
        self._isParallel = False
        if MFEM_USE_MPI:
            if isinstance(mesh, mfem.ParMesh):
                if not mesh.Nonconforming():
                    raise ValueError(
                        "The provided parallel mesh is a conforming mesh. Please provide a non-conforming parallel mesh.")
                self._isParallel = True
                self.fespace = mfem.ParFiniteElementSpace(
                    mesh, self.fec, self.vdim)
                self._sol = mfem.ParGridFunction(self.fespace)
            else:
                self.fespace = mfem.FiniteElementSpace(
                    mesh, self.fec, self.vdim)
                self._sol = mfem.GridFunction(self.fespace)
        else:
            self.fespace = mfem.FiniteElementSpace(
                mesh, self.fec, self.vdim)
            self._sol = mfem.GridFunction(self.fespace)
        self.rsolver = RusanovFlux()
        self.ode_solver = ode_solver
        self.getSystem(IntOrderOffset=3, **kwargs)
        self.ode_solver.Init(self.HCL)
        self.CFL = cfl
        self.terminal_time = terminal_time

    def init(self, u0: mfem.VectorFunctionCoefficient):
        self.t = 0.0
        self._sol.ProjectCoefficient(u0)
        self.initial_condition = u0
        if self._isParallel:
            dummy = mfem.ParGridFunction(self._sol)
        else:
            dummy = mfem.GridFunction(self._sol)
        self.HCL.Mult(self._sol, dummy)

    def reset(self):
        if self._isParallel:
            self.fespace = mfem.ParFiniteElementSpace(
                self._initial_mesh, self.fec, self.vdim)
            self._sol = mfem.ParGridFunction(self._fespace)
        else:
            self.fespace = mfem.FiniteElementSpace(
                self._initial_mesh, self.fec, self.vdim)
            self._sol = mfem.GridFunction(self._fespace)
        self._sol.ProjectCoefficient(self.initial_condition)

    def getSystem(self, **kwargs):
        raise NotImplementedError(
            "getSystem should be implemented in the subclass")

    def step(self, big_time_step=None):
        """Advance PDE solver in time. If big_time_step is not provided, then it advance single time-step.
        If big_time_step>0 is provided, then it advance multi time-steps so that sum of dt = big_time_step

        Args:
            big_time_step (np.double, optional): Target time step size. Defaults to None.

        Raises:
            RuntimeError: When time step size is negative

        Returns:
            bool: Whether the solver reaches to terminal time or not.
        """
        
        if big_time_step is None:  # single step
            # single step
            dt = self.compute_timestep()
            real_dt = min(self.terminal_time - self.t, dt)
            self.ode_solver.Step(self._sol, self.t, real_dt)
            self.t += real_dt
            return (self.terminal_time - self.t) < dt*1.e-06
        
        big_time_step = min(big_time_step, self.terminal_time - self.t)
        while big_time_step > 0:
            dt = self.compute_timestep()
            real_dt = min(dt, big_time_step)
            if real_dt < 0:
                    raise RuntimeError(
                        f"dt is negative: Either computed time step is negative or time exceeded target time.\n\tdt = {dt}, \n\tcurrent time = {self.t}")
            self.ode_solver.Step(self._sol, self.t, real_dt)
            self.t += real_dt
            big_time_step -= real_dt
        
        return (self.terminal_time - self.t) < dt*1.e-06
        

    def compute_timestep(self):
        dt = self.CFL * self.min_h / self._HCL.getMaxCharSpeed() / (2*self.max_order + 1)
        if self._isParallel:
            reduced_dt = MPI.COMM_WORLD.allreduce(dt, op=MPI.MIN)
            dt = reduced_dt
        return dt

    def render(self):
        raise NotImplementedError("render should be implemented in the subclass")

    def refine(self, marked: mfem.intArray, coarsening: bool = False):
        if self.refinement_mode == 'h':
            if coarsening:
                self.hDerefine(marked)
            else:
                self.hRefine(marked)
        elif self.refinement_mode == 'p':
            self.pRefine(marked)
        else:
            raise ValueError(
                f"Refinement mode should be either 'h' or 'p', but {self.refinement_mode} is provided.")

    def hRefine(self, marked):
        self.update_min_h()
        pass

    def hDerefine(self, marked):
        self.update_min_h()
        pass

    def pRefine(self, marked):
        if self._isParallel:
            old_fes = mfem.ParFiniteElementSpace(self.fespace)
            old_sol = mfem.ParGridFunction(old_fes)
        else:
            old_fes = mfem.FiniteElementSpace(self.fespace)
            old_sol = mfem.GridFunction(old_fes)
        old_sol.Assign(self._sol)
        for i in range(self._mesh.GetNE()):
            self.fespace.SetElementOrder(i, marked[i])
        self.fespace.Update(False)
        self._sol.Update()
        if self.t == 0:
            self._sol.ProjectCoefficient(self.u0)
        else:
            transfer = mfem.PRefinementTransferOperator(old_fes, self.fespace)
            transfer.Mult(old_sol, self._sol)

    def update_min_h(self):
        self.min_h = min([self._mesh.GetElementSize(i, 1)
                         for i in range(self._mesh.GetNE())])
        if MFEM_USE_MPI and self._isParallel:
            hmin = MPI.COMM_WORLD.allreduce(self.min_h, op=MPI.MIN)
            self.min_h = hmin

    def update_max_order(self):
        self.max_order = self._fespace.GetMaxElementOrder()
        if MFEM_USE_MPI and self._isParallel:
            pmax = MPI.COMM_WORLD.allreduce(self.max_order, op=MPI.MAX, )
            self.max_order = pmax

    def init_renderer(self):
        raise NotImplementedError(
            "init_renderer should be implemented in the subclass")

    # PROPERTIES

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, new_mesh: mfem.Mesh):
        raise ValueError(
            'Mesh should not be modified by solver.mesh = mesh. Either create a new solver, or change finite element space.')

    @property
    def fespace(self):
        return self._fespace

    @fespace.setter
    def fespace(self, new_fespace: mfem.FiniteElementSpace):
        if MFEM_USE_MPI:  # if parallel mfem is used
            if self._isParallel != isinstance(new_fespace, mfem.ParFiniteElementSpace):
                if self._isParallel:
                    raise ValueError(
                        'The solver is initialized with parallel mesh, but tried to overwrite FESpace with serial FESpace.')
                else:
                    raise ValueError(
                        'The solver is initialized with serial mesh, but tried to overwrite FESpace with parallel FESpace.')
        self._mesh: mfem.Mesh = new_fespace.GetMesh()
        self._fespace = new_fespace
        if self._isParallel:
            self._pmesh: mfem.ParMesh = self._fespace.GetParMesh()
            self._initial_mesh = mfem.ParMesh(self._mesh)
        else:
            self._initial_mesh = mfem.Mesh(self._mesh)
        self.update_min_h()
        self.update_max_order()

    @property
    def sol(self):
        return self._sol

    @sol.setter
    def sol(self, gf: mfem.GridFunction):
        if gf.FESpace() is not self.fespace:
            raise ValueError(
                'The provided grid function is not a function in the current finite element space.')
        self._sol = gf

    @property
    def HCL(self):
        return self._HCL

    @HCL.setter
    def HCL(self, new_HCL: DGHyperbolicConservationLaws):
        self._HCL = new_HCL

    @property
    def renderer_space(self):
        return self._renderer_space

    @renderer_space.setter
    def renderer_space(self, subspace: mfem.FiniteElementSpace):
        self._renderer_space = subspace


class AdvectionSolver(Solver):
    def getSystem(self, IntOrderOffset=3, **kwargs):
        self.b = kwargs.get('b')
        self.HCL: DGHyperbolicConservationLaws = getAdvectionEquation(
            self._fespace, self.rsolver, self.b, IntOrderOffset)

    def render(self):
        self.sout.precision(8)
        self.sout << "solution\n" << self._mesh << self._sol
        self.sout.flush()

    def init_renderer(self):
        self.sout = mfem.socketstream("Dohyuns-Macbook", 19916)
        print(self.sout.good())
        self.sout.precision(8)
        self.sout.send_text("view 0 0")
        self.sout.send_text("keys jl")
        self.sout.send_solution(self._mesh, self._sol)
        self.sout.flush()


class BurgersSolver(Solver):
    def getSystem(self, IntOrderOffset=3, **kwargs):
        self.HCL: DGHyperbolicConservationLaws = getBurgersEquation(
            self._fespace, self.rsolver, IntOrderOffset)

    def render(self):
        self.sout.precision(8)
        self.sout << "solution\n" << self._mesh << self._sol
        self.sout.flush()

    def init_renderer(self):
        self.sout = mfem.socketstream("Dohyuns-Macbook", 19916)
        print(self.sout.good())
        self.sout.precision(8)
        self.sout.send_text("view 0 0")
        self.sout.send_text("keys jl")
        self.sout.send_solution(self._mesh, self._sol)
        self.sout.flush()


class EulerSolver(Solver):
    def getSystem(self, IJntOrderOffset=3, **kwargs):
        self.gas_constant = kwargs.get('gas_constant', 1.0)
        self.specific_heat_ratio = kwargs.get('specific_heat_ratio', 1.4)
        self.HCL: DGHyperbolicConservationLaws = getEulerSystem(
            self._fespace, self.rsolver, self.specific_heat_ratio, 3)

    def render(self):
        if self._isParallel:
            self.sout.send_text("parallel " + str(MPI.COMM_WORLD.size) +
                           " " + str(MPI.COMM_WORLD.rank))
            self.sout.send_solution()

    def render(self):
        self.sout.precision(8)
        self.sout << "solution\n" << self._mesh << self._sol
        self.sout.flush()

    def init_renderer(self):
        self.sout = mfem.socketstream("localhost", 19916)
        self.sout.send_text("view 0 0")
        self.sout.send_text("keys jl")
        self.sout.flush()
