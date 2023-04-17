import sys
# if 'mfem.ser' in sys.modules:
    # MFEM_USE_MPI = False
import mfem.ser as mfem
from mfem.ser import \
    RusanovFlux, RiemannSolver, DGHyperbolicConservationLaws, HyperbolicFormIntegrator, \
        AdvectionFormIntegrator, BurgersFormIntegrator, ShallowWaterFormIntegrator, EulerFormIntegrator
from typing import Optional, Tuple
# else:
#     MFEM_USE_MPI = True
#     from mpi4py import MPI
#     import mfem.par as mfem
#     from mfem.par import \
#         getAdvectionEquation, getBurgersEquation, getShallowWaterEquation, getEulerSystem, \
#         RusanovFlux, RiemannSolver, DGHyperbolicConservationLaws, HyperbolicFormIntegrator, \
#            AdvectionFormIntegrator, BurgersFormIntegrator, ShallowWaterFormIntegrator, EulerFormIntegrator

import numpy as np

class Solver:
    def __init__(self, mesh: mfem.Mesh, order: int, num_equations: int, refinement_mode: str, ode_solver: mfem.ODESolver, cfl, **kwargs):
        self.visualization = False # Set True when init_render is called
        self.order = order
        self.max_order = order
        self.num_equations = num_equations
        self.refinement_mode = refinement_mode
        self.sdim = mesh.Dimension()
        self.vdim = num_equations
        self.fec = mfem.DG_FECollection(order, self.sdim)
        self.fecP0 = mfem.DG_FECollection(0, self.sdim)
        self._isParallel = False
        # if MFEM_USE_MPI:
        #     if isinstance(mesh, mfem.ParMesh):
        #         if not mesh.Nonconforming():
        #             raise ValueError(
        #                 "The provided parallel mesh is a conforming mesh. Please provide a non-conforming parallel mesh.")
        #         self._isParallel = True
        #         self.fespace = mfem.ParFiniteElementSpace(
        #             mesh, self.fec, self.vdim, mfem.Ordering.byNODES)
        #         self.constant_space = mfem.ParFiniteElementSpace(
        #             mesh, self.fecP0, self.vdim, mfem.Ordering.byNODES)
        #         self._sol = mfem.ParGridFunction(self.fespace)
        #     else:
        #         self.fespace = mfem.FiniteElementSpace(
        #             mesh, self.fec, self.vdim, mfem.Ordering.byNODES)
        #         self.constant_space = mfem.FiniteElementSpace(
        #             mesh, self.fecP0, self.vdim, mfem.Ordering.byNODES)
        #         self._sol = mfem.GridFunction(self.fespace)
        # else:
        self.fespace = mfem.FiniteElementSpace(
            mesh, self.fec, self.vdim, mfem.Ordering.byNODES)
        self.constant_space = mfem.FiniteElementSpace(
            mesh, self.fecP0, self.vdim, mfem.Ordering.byNODES)
        
        self._sol = mfem.GridFunction(self.fespace)
        self.rsolver = RusanovFlux()
        self.ode_solver = ode_solver
        self.solver_args = kwargs
        self.getSystem(IntOrderOffset=3, **self.solver_args)
        self.ode_solver.Init(self.HCL)
        self.CFL = cfl
        
        self.element_geometry = self.mesh.GetElementGeometry(0)
        self.has_estimator =  False # Use set_estimator

    def init(self, u0: mfem.VectorFunctionCoefficient):
        self.t = 0.0
        self._sol.ProjectCoefficient(u0)
        self.initial_condition = u0
        # if self._isParallel:
        #     dummy = mfem.ParGridFunction(self._sol)
        # else:
        dummy = mfem.GridFunction(self._sol)
        self.HCL.Mult(self._sol, dummy)

    def reset(self):
        # if self._isParallel:
        #     self.fespace = mfem.ParFiniteElementSpace(
        #         self._initial_mesh, self.fec, self.vdim)
        #     self._sol = mfem.ParGridFunction(self._fespace)
        # else:
        self.fespace = mfem.FiniteElementSpace(
            self._initial_mesh, self.fec, self.vdim)
        self._sol = mfem.GridFunction(self._fespace)
        self.getSystem(**self.solver_args)
        self.t = 0.0
        self.initial_condition.SetTime(0.0)
        self._sol.ProjectCoefficient(self.initial_condition)

    def getSystem(self, **kwargs):
        raise NotImplementedError(
            "getSystem should be implemented in the subclass")

    def step(self) -> tuple[bool, float]:
        """Advance FE solution one time step where dt is from CFL condition.

        Returns:
            bool: Whether the solver reaches to terminal time or not.
            float: time step size
        """
        done = False
        
        dt = self.compute_timestep()
        time_remaining = self.terminal_time - self.t
        if (time_remaining - dt) / dt < 1e-04:
            done = True
            dt = time_remaining
        if dt <= 0:
            return (True, dt)
        # single step
        self.ode_solver.Step(self._sol, self.t, dt)
        self.t += dt
        return (done, self.t)
        
    def compute_timestep(self):
        dt = self.CFL * self.min_h / self._HCL.getMaxCharSpeed() / (2*self.max_order + 1)
        # if self._isParallel:
        #     reduced_dt = MPI.COMM_WORLD.allreduce(dt, op=MPI.MIN)
        #     dt = reduced_dt
        return dt
    
    def compute_L2_errors(self, exact:mfem.VectorFunctionCoefficient):
        exact.SetTime(self.t)
        errors = mfem.GridFunction(self.constant_space)
        self.sol.ComputeElementL2Errors(exact, errors)
        return (errors.Norml2(), errors)

    def render(self):
        raise NotImplementedError("render should be implemented in the subclass")
    
    def save(self, postfix:int, prefix:str='./'):
        """Save mesh and solution with <solver_name>-<postfix:06>.mesh/gf.
        Perform ProlongToMaxOrder if variable order.

        Args:
            postfix (int): postfix for filename. Should be less than or equal to 6 digits
        """
        self.mesh.Print(f'{prefix}{self.solver_name}-{postfix:06}.mesh')
        if self.fespace.IsVariableOrder():
            mfem.ProlongToMaxOrder(self.sol).Save(f'{prefix}{self.solver_name}-{postfix:06}.gf')
        else:
            self.sol.Save(f'{prefix}{self.solver_name}-{postfix:06}.gf')

    def refine(self, marked: mfem.intArray, coarsening: bool = False):
        """Perform general h/p mesh refinement
        For h-refinement, marked elements will be divided into 2^dim elements
        For p-refinement, order[i] = base_order + marked[i]

        Args:
            marked (mfem.intArray): Refine elements where marked[i]=True
            coarsening (bool, optional): Used only when refinement_mode='h'. Defaults to False.

        Raises:
            ValueError: Check whether refinement_mode is correctly set or not
        """
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
        # TODO: Keep an array that remembers initial element index such as idx[i] = original_index
        # IDEA: Initialize idx = 0...N and transfer this to refined mesh by using transfer operator
        
        self.fespace.Update()
        self.sol.Update()
        self.HCL.Update()
        self.update_min_h()
        raise NotImplementedError("Not yet implemented")

    def hDerefine(self, marked):
        self.fespace.Update()
        self.sol.Update()
        self.HCL.Update()
        self.update_min_h()
        raise NotImplementedError("Not yet implemented")

    def pRefine(self, marked, base='base_order'):
        """Perform p-refinement with given marker

        Args:
            marked (_type_): Shift from the target base order
            base (str, optional): Either refinement will be performed with base_order or current_order.
            If base_order, order[i] = self.order + marked[i]. If current_order, order[i] += marked[i].
            Defaults to 'base_order'.

        Raises:
            ValueError: If provided base is neither 'base_order' nor 'current_order'
        """
        if self.t != 0: # Copying old fes is only necessary when t is non-zero
            old_fes = mfem.FiniteElementSpace(self.mesh, self.fec)
            for i in range(self.mesh.GetNE()):
                old_fes.SetElementOrder(i, self.fespace.GetElementOrder(i))
        
        # Update FESpace
        if base == 'base_order': # order[i] = base_order + marked[i]
            for i in range(self.mesh.GetNE()):
                self.fespace.SetElementOrder(i, self.order + marked[i])
        elif base == 'current_order': # order[i] = current_order[i] + marked[i]
            for i in range(self.mesh.GetNE()):
                self.fespace.SetElementOrder(i, self.fespace.GetElementOrder(i) + marked[i])
        else:
            raise ValueError(f"pRefine only supports 'base_order' or 'current_order' base, but {base} is given.")
        self.fespace.Update(False)
        
        # Update solution
        if self.t == 0: # if initial time
            self.sol.Update()
            self.sol.ProjectCoefficient(self.initial_condition) # reproject
        else: # otherwise, transfer from previous space to new space
            new_sol = mfem.GridFunction(self.fespace)
            
            op = mfem.PRefinementTransferOperator(old_fes, self.fespace)
            op.Mult(self.sol, new_sol)
            self.sol = new_sol
        
        # Update DGHyperbolic and notify fespace that update is finished
        self.HCL.Update()
        self.fespace.UpdatesFinished()

    def ComputeElementAverageFluxJacobian(self) -> tuple[mfem.DenseTensor, mfem.DenseMatrix]:
        """Compute element average state ū and compute ∂F(ū)/∂u

        Returns:
            mfem.DenseMatrix: Flux Jacobian of average state
        """
        average_state = mfem.GridFunction(self.constant_space)
        self.sol.GetElementAverages(average_state)
        
        # Set up integration point as center of element.
        # This is necessary when F depends on space.
        # e.g., Advection equation
        intrule:mfem.IntegrationRule = mfem.IntRules.Get(self.element_geometry, 0)
        ip = intrule.IntPoint(0)
        
        
        # Preallocate Jacobian(nvars, nvars, dim, nElem)
        Jacobians = mfem.DenseTensor(self.vdim, self.vdim, self.sdim*self.mesh.GetNE())
        # Preallocate eigs(nvars, dim, nElem)
        eigs = mfem.DenseMatrix(self.vdim, self.sdim*self.mesh.GetNE())
        
        # NOTE: Will it be much cheaper and effective enough to just consider
        #       normal directional Jacobian?
        for i in range(self.mesh.GetNE()):
            # Subvectors
            current_state = mfem.Vector(average_state, self.vdim*i, self.vdim)
            current_J = mfem.DenseTensor(Jacobians.GetData(self.sdim*i), self.vdim, self.vdim, self.sdim)
            current_eigs = mfem.DenseMatrix(eigs.GetColumn(self.sdim*i), self.vdim, self.sdim)
            
            # Compute flux!
            trans = self.mesh.GetElementTransformation(i)
            trans.SetIntPoint(ip)
            self.formIntegrator.ComputeFluxJacobian(current_state, trans, current_J, current_eigs)
        
        return (Jacobians, eigs)
        
    def update_min_h(self):
        self.min_h = min([self._mesh.GetElementSize(i, 1)
                         for i in range(self._mesh.GetNE())])
        # if MFEM_USE_MPI and self._isParallel:
        #     hmin = MPI.COMM_WORLD.allreduce(self.min_h, op=MPI.MIN)
        #     self.min_h = hmin

    def update_max_order(self):
        self.max_order = self._fespace.GetMaxElementOrder()
        # if MFEM_USE_MPI and self._isParallel:
        #     pmax = MPI.COMM_WORLD.allreduce(self.max_order, op=MPI.MAX, )
        #     self.max_order = pmax

    def init_renderer(self):
        raise NotImplementedError(
            "init_renderer should be implemented in the subclass")
        
    def set_estimator(self, estimator:mfem.ErrorEstimator):
        self.has_estimator = True
        self.estimator = estimator
    
    def estimate(self) -> tuple[float, mfem.Vector]:
        if self.has_estimator:
            errors = mfem.Vector(self.mesh.GetNE())
            self.estimator.GetLocalErrors(self.sol, errors)
            total_error = errors.Norml2()
        else:
            total_error, errors = self.compute_L2_errors(self.initial_condition)
        return (total_error, errors)

    @property
    def mesh(self) -> mfem.Mesh:
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
        # if MFEM_USE_MPI:  # if parallel mfem is used
        #     if self._isParallel != isinstance(new_fespace, mfem.ParFiniteElementSpace):
        #         if self._isParallel:
        #             raise ValueError(
        #                 'The solver is initialized with parallel mesh, but tried to overwrite FESpace with serial FESpace.')
        #         else:
        #             raise ValueError(
        #                 'The solver is initialized with serial mesh, but tried to overwrite FESpace with parallel FESpace.')
        self._fespace = new_fespace
        # if self._isParallel:
        #     self._mesh: mfem.ParMesh = self._fespace.GetParMesh()
        #     self._initial_mesh = mfem.ParMesh(self._mesh, True)
        # else:
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
    def HCL(self) -> DGHyperbolicConservationLaws:
        return self._HCL

    @HCL.setter
    def HCL(self, new_HCL: DGHyperbolicConservationLaws):
        self._HCL = new_HCL

    @property
    def renderer_space(self) -> mfem.FiniteElementSpace:
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
        self.solver_name = 'advection'
        self.b = kwargs.get('b')
        self.formIntegrator = AdvectionFormIntegrator(self.rsolver, self.sdim, self.b, 3)
        self.HCL = DGHyperbolicConservationLaws(self.fespace, self.formIntegrator, self.vdim)

    def render(self):
        if not self.visualization:
            return
        
        if not self.sout.is_open():
            print("GLVis is closed.")
            self.visualization = False
            return
        
        self.sout.precision(8)
        if self.fespace.IsVariableOrder():
            self.sout.send_solution(self._mesh, mfem.ProlongToMaxOrder(self._sol))
        else:
            self.sout.send_solution(self._mesh, self._sol)
        self.sout.flush()

    def init_renderer(self):
        self.visualization = True
        self.sout = mfem.socketstream("localhost", 19916)
        if not self.sout.good():
            print("Unable to open GLVis.")
            self.visualization = False
            return
        self.sout.precision(8)
        self.sout.send_solution(self._mesh, self._sol)
        self.sout.send_text("view 0 0\n")
        self.sout.send_text("keys jlm")
        self.sout.send_solution(self._mesh, self._sol)
        self.sout.flush()


class BurgersSolver(Solver):
    def getSystem(self, IntOrderOffset=3, **kwargs):
        self.solver_name = 'burgers'
        self.formIntegrator = BurgersFormIntegrator(self.rsolver, self.sdim, 3)
        self.HCL = DGHyperbolicConservationLaws(self.fespace, self.formIntegrator, self.vdim)

    def render(self):
        if not self.visualization:
            return
        
        if not self.sout.is_open():
            print("GLVis is closed.")
            self.visualization = False
            return
        
        self.sout.precision(8)
        if self.fespace.IsVariableOrder():
            self.sout.send_solution(self._mesh, mfem.ProlongToMaxOrder(self._sol))
        else:
            self.sout.send_solution(self._mesh, self._sol)
        self.sout.flush()

    def init_renderer(self):
        self.visualization = True
        self.sout = mfem.socketstream("localhost", 19916)
        if not self.sout.good():
            print("Unable to open GLVis.")
            self.visualization = False
            return
        
        self.sout.precision(8)
        self.sout.send_solution(self._mesh, self._sol)
        self.sout.send_text("view 0 0")
        self.sout.send_text("keys jlm")
        self.sout.flush()


class EulerSolver(Solver):
    def getSystem(self, IJntOrderOffset=3, **kwargs):
        self.solver_name = 'euler'
        self.gas_constant = kwargs.get('gas_constant', 1.0)
        self.specific_heat_ratio = kwargs.get('specific_heat_ratio', 1.4)
        self.formIntegrator = EulerFormIntegrator(self.rsolver, self.sdim, self.specific_heat_ratio, 3)
        self.HCL = DGHyperbolicConservationLaws(self.fespace, self.formIntegrator, self.vdim)

    def render(self):
        # TODO: Visualize momentum / density / pressure ... etc
        if not self.visualization:
            return
        
        if not self.sout.is_open():
            print("GLVis is closed.")
            self.visualization = False
            return
        
        self.sout.precision(8)
        if self.fespace.IsVariableOrder():
            self.sout.send_solution(self._mesh, mfem.ProlongToMaxOrder(self._sol))
        else:
            self.sout.send_solution(self._mesh, self._sol)
        self.sout.flush()

    def init_renderer(self):
        self.visualization = True
        self.sout = mfem.socketstream("localhost", 19916)
        if not self.sout.good():
            print("Unable to open GLVis.")
            self.visualization = False
            return
        self.sout.send_text("view 0 0")
        self.sout.send_text("keys jlm")
        self.sout.flush()
