import numpy as np
from typing import Tuple
import sys
from gymnasium.spaces import Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv

if 'mfem.ser' in sys.modules:
    import mfem.ser as mfem
    from mfem.ser import getAdvectionEquation, RusanovFlux, DGHyperbolicConservationLaws, ODESolver
else:
    import mfem.par as mfem
    from mfem.par import getAdvectionEquation, RusanovFlux, DGHyperbolicConservationLaws, ODESolver
    
from ray.rllib.utils.typing import (
    AgentID,
    EnvCreator,
    EnvID,
    EnvType,
    MultiAgentDict,
    MultiEnvDict
)


class BaseDynamicRefinementEnv(MultiAgentEnv):

    solver_name: str
    num_equations: int

    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.solver_name = kwargs.get('solver_name')
        self.solver = kwargs.get('solver')
        self.ode_solver_name = kwargs.get('ode_solver_name')
        self.ode_solver = kwargs.get('ode_solver')
        
        self.final_time = kwargs.get('final_time', -1)
        assert self.final_time > 0, print(f'Expected positive final time, but {self.final_time} is given')
        
        self.regrid_time = kwargs.get('regrid_time', -1)
        assert self.regrid_time > 0 & self.regrid_time < self.final_time, print(f'Expected positive regrid time smaller than the final time, but {self.regrid_time} is given')
        
        
        self.reward_type = kwargs.get('reward_type', 'local_threshold')
        assert self.reward_type in ['local_threshold', 'global'], print(
            f'Reward type {self.reward_type} is not supported')

        self.observation_type = kwargs.get('observation_type', 'local_neighborhood')
        assert self.observatoin_type in [
            'local_history', 'local_neighborhood', 'graph'], print(f'Observation type {self.observation_type} is not supported')

        self.refinement_mode = kwargs.get('ref_mode', 'p')
        assert self.refinement_mode in ['h', 'p'], print(
            f'Refinement mode - {self.refinement_mode} is not supported')

        self.observe_flux = kwargs.get('observe_flux', False)
        
        self.step_count = 0
        self.global_error_prvs = -1
        self.time = 0
        self.mesh = self.initial_mesh
        self.env_done = False
        self.env_truncated = False
        
        self.exact_solution = kwargs.get('exact_solution', None)
        
    @property
    def solver(self):
        return self._solver
    @solver.setter
    def solver(self, solver):
        assert isinstance(solver, DGHyperbolicConservationLaws)
        self._solver = solver
        
    @property
    def solution(self):
        return self._solution
    @solution.setter
    def solution(self, solution):
        assert isinstance(solution, mfem.GridFunction)
        self._solution = solution
    
    @property
    def ode_solver(self):
        return self._ode_solver
    @ode_solver.setter
    def ode_solver(self, ode_solver):
        assert isinstance(ode_solver, ODESolver)
        self._ode_solver = ode_solver
        
    def step(self, action_dict: MultiAgentDict) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """Perform refine/derefine and advance to next re-grid time

        Args:
            action_dict (MultiAgentDict): Refinement, derefinement action dict

        Returns:
            Tuple[ MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict ]: new observation, reward, terminated value, truncated value, info value for each agent
        """
        
        done_dict = dict.fromkeys(self._agent_ids, self.env_done)
        trunc_dict = dict.fromkeys(self._agent_ids, self.env_truncated)
        info_dict = {k: {} for k in self._agent_ids}
        
        errors = self.estimator.Estimate(self.solution)
