import sys
if 'mfem.ser' in sys.modules:
    MFEM_USE_MPI = False
    import mfem.ser as mfem
    from mfem.ser import \
        getAdvectionEquation, getBurgersEquation, getShallowWaterEquation, getEulerSystem, \
        RusanovFlux, RiemannSolver, DGHyperbolicConservationLaws, HyperbolicFormIntegrator, \
           AdvectionFormIntegrator, BurgersFormIntegrator, ShallowWaterFormIntegrator, EulerFormIntegrator
else:
    MFEM_USE_MPI = True
    from mpi4py import MPI
    import mfem.par as mfem
    from mfem.par import \
        getAdvectionEquation, getBurgersEquation, getShallowWaterEquation, getEulerSystem, \
        RusanovFlux, RiemannSolver, DGHyperbolicConservationLaws, HyperbolicFormIntegrator, \
           AdvectionFormIntegrator, BurgersFormIntegrator, ShallowWaterFormIntegrator, EulerFormIntegrator

import numpy as np

class Solver:
    def __init__(self, mesh: mfem.Mesh, order: int, num_equations: int, refinement_mode: str, ode_solver: mfem.ODESolver, cfl, **kwargs):
        self.order = order
        self.max_order = order
        self.num_equations = num_equations
        self.refinement_mode = refinement_mode
        self.sdim = mesh.Dimension()
        self.vdim = num_equations
        self.fec = mfem.DG_FECollection(order, self.sdim)
        self.fecP0 = mfem.DG_FECollection(0, self.sdim)
        self._isParallel = False
        if MFEM_USE_MPI:
            if isinstance(mesh, mfem.ParMesh):
                if not mesh.Nonconforming():
                    raise ValueError(
                        "The provided parallel mesh is a conforming mesh. Please provide a non-conforming parallel mesh.")
                self._isParallel = True
                self.fespace = mfem.ParFiniteElementSpace(
                    mesh, self.fec, self.vdim, mfem.Ordering.byNODES)
                self.constant_space = mfem.ParFiniteElementSpace(
                    mesh, self.fecP0, self.vdim, mfem.Ordering.byNODES)
                self._sol = mfem.ParGridFunction(self.fespace)
            else:
                self.fespace = mfem.FiniteElementSpace(
                    mesh, self.fec, self.vdim, mfem.Ordering.byNODES)
                self.constant_space = mfem.FiniteElementSpace(
                    mesh, self.fecP0, self.vdim, mfem.Ordering.byNODES)
                self._sol = mfem.GridFunction(self.fespace)
        else:
            self.fespace = mfem.FiniteElementSpace(
                mesh, self.fec, self.vdim, mfem.Ordering.byNODES)
            self.constant_space = mfem.FiniteElementSpace(
                mesh, self.fecP0, self.vdim, mfem.Ordering.byNODES)
            self._sol = mfem.GridFunction(self.fespace)
        self.rsolver = RusanovFlux()
        self.ode_solver = ode_solver
        self.getSystem(IntOrderOffset=3, **kwargs)
        self.ode_solver.Init(self.HCL)
        self.CFL = cfl
        
        self.element_geometry = self.mesh.GetElementGeometry(0)

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
        self.t = 0.0

    def getSystem(self, **kwargs):
        raise NotImplementedError(
            "getSystem should be implemented in the subclass")

    def step(self):
        """Advance FE solution one time step where dt is from CFL condition.

        Returns:
            bool: Whether the solver reaches to terminal time or not.
        """
        dt = self.compute_timestep()
        real_dt = min(dt, self.terminal_time - self.t)
        if real_dt <= 0:
            return (True, real_dt)
        # single step
        self.ode_solver.Step(self._sol, self.t, real_dt)
        self.t += real_dt
        return ((self.terminal_time - self.t) < dt*1.e-04, real_dt)
        
    def compute_timestep(self):
        dt = self.CFL * self.min_h / self._HCL.getMaxCharSpeed() / (2*self.max_order + 1)
        if self._isParallel:
            reduced_dt = MPI.COMM_WORLD.allreduce(dt, op=MPI.MIN)
            dt = reduced_dt
        return dt
    def compute_L2_errors(self, exact:mfem.VectorFunctionCoefficient):
        self.initial_condition.SetTime(self.t)
        errors = mfem.GridFunction(self.constant_space)
        self.sol.ComputeElementL2Errors(exact, errors)
        return (np.sqrt(np.dot(errors, errors)), errors)

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

    def ComputeElementAverageFluxJacobian(self):
        average_state = mfem.GridFunction(self.constant_space)
        self.sol.GetElementAverages(average_state)
        intrule:mfem.IntegrationRule = mfem.IntRules.Get(self.element_geometry, 0)
        ip = intrule.IntPoint(0)
        
        Jacobians = mfem.DenseTensor(self.vdim, self.vdim, self.sdim*self.mesh.GetNE())
        memory_J = Jacobians.GetMemory()
        eigs = mfem.DenseMatrix(self.vdim, self.sdim*self.mesh.GetNE())
        for i in range(self.mesh.GetNE()):
            current_state = mfem.Vector(average_state, self.vdim*i, self.vdim)
            current_J = mfem.DenseTensor(Jacobians.GetData(self.sdim*i), self.vdim, self.vdim, self.sdim)
            current_eigs = mfem.DenseMatrix(eigs.GetColumn(self.sdim*i), self.vdim, self.sdim)
            trans = self.mesh.GetElementTransformation(i)
            trans.SetIntPoint(ip)
            self.formIntegrator.ComputeFluxJacobian(current_state, trans, current_J, current_eigs)
        
        return (Jacobians, eigs)
        
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
        
    def set_estimator(self, estimator):
        pass
    
    def estimate(self):
        pass

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, new_mesh: mfem.Mesh):
        raise ValueError(
            'Mesh should not be modified by solver.mesh = mesh. Either create a new solver, or change finite element space.')

    @property
    def fespace(self) -> mfem.FiniteElementSpace:
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
        self._fespace = new_fespace
        if self._isParallel:
            self._mesh: mfem.ParMesh = self._fespace.GetParMesh()
            self._initial_mesh = mfem.ParMesh(self._mesh, True)
        else:
            self._mesh: mfem.Mesh = self._fespace.GetMesh()
            self._initial_mesh = mfem.Mesh(self._mesh)
        self.update_min_h()
        self.update_max_order()

    @property
    def sol(self) -> mfem.GridFunction:
        return self._sol

    @sol.setter
    def sol(self, gf: mfem.GridFunction):
        if gf.FESpace() is not self.fespace:
            raise ValueError(
                'The provided grid function is not a function in the current finite element space.')
        self._sol = gf
        
    @property
    def formIntegrator(self) -> mfem.HyperbolicFormIntegrator:
        return self._formIntegrator
    
    @formIntegrator.setter
    def formIntegrator(self, fi:mfem.HyperbolicFormIntegrator):
        self._formIntegrator = fi
        
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
        
    @property
    def terminal_time(self) -> float:
        return self._terminal_time
    @terminal_time.setter
    def terminal_time(self, tf:float):
        self._terminal_time = tf


class AdvectionSolver(Solver):
    def getSystem(self, IntOrderOffset=3, **kwargs):
        self.b = kwargs.get('b')
        self.formIntegrator = AdvectionFormIntegrator(self.rsolver, self.sdim, self.b, 3)
        self.HCL = DGHyperbolicConservationLaws(self.fespace, self.formIntegrator, self.vdim)

    def render(self):
        self.sout.precision(8)
        self.sout << "solution\n" << self._mesh << self._sol
        self.sout.flush()

    def init_renderer(self):
        self.sout = mfem.socketstream("localhost", 19916)
        print(self.sout.good())
        self.sout.precision(8)
        self.sout.send_text("view 0 0")
        self.sout.send_text("keys jl")
        self.sout.send_solution(self._mesh, self._sol)
        self.sout.flush()


class BurgersSolver(Solver):
    def getSystem(self, IntOrderOffset=3, **kwargs):
        self.formIntegrator = BurgersFormIntegrator(self.rsolver, self.sdim, 3)
        self.HCL = DGHyperbolicConservationLaws(self.fespace, self.formIntegrator, self.vdim)

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
        self.formIntegrator = EulerFormIntegrator(self.rsolver, self.sdim, self.specific_heat_ratio, 3)
        self.HCL = DGHyperbolicConservationLaws(self.fespace, self.formIntegrator, self.vdim)

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
