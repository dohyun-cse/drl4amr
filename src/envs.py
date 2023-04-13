import sys
import numpy as np
import math

import gymnasium as gym
import sys

from gymnasium import spaces, utils
from gymnasium.spaces import Box, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import logging
from typing import Callable, Dict, List, Tuple, Optional, Union, Set, Type

from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.annotations import (
    ExperimentalAPI,
    override,
    PublicAPI,
    DeveloperAPI,
)
from ray.rllib.utils.typing import (
    AgentID,
    EnvCreator,
    EnvID,
    EnvType,
    MultiAgentDict,
    MultiEnvDict,
)

import src.solvers as solver
if 'mfem.ser' in sys.modules:
    MFEM_USE_MPI = False
    import mfem.ser as mfem
else:
    MFEM_USE_MPI = True
    from mpi4py import MPI

class HyperbolicAMREnv(MultiAgentEnv):
    def __init__(self, 
                 solver_name:str, 
                 solver_args:Dict=None,
                 regrid_time=-1.0,
                 refine_mode='p',
                 window_size=3, 
                 seed:Optional[float]=None):
        if solver_name == 'advection':
            self.solver = solver.AdvectionSolver(**solver_args)
        elif solver_name == 'burgers':
            self.solver = solver.BurgersSolver(**solver_args)
        elif solver_name == 'euler':
            self.solver = solver.EulerSolver(**solver_args)
        else:
            raise ValueError(f'Unsupported solver name: {solver_name}.')
        
        if regrid_time < 0 or regrid_time > solver_args.get('terminal_time', 2):
            raise ValueError(f'Regrid time is not provided or larger than the terminal time: {regrid_time}')
        
        self.regrid_time = regrid_time
        self.refine_mode = refine_mode
        self.window_size = window_size
        self.randomize = seed is not None
        self.seed = seed
    
    def reset(self, *, seed: Optional[int]=None, options: Optional[dict] = None):
        self.solver.reset()
        
    
    def step(self, action_dict:MultiAgentDict) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """Returns observations from ready agents.

        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        Returns:
            Tuple containing 1) new observations for
            each ready agent, 2) reward values for each ready agent. If
            the episode is just started, the value will be None.
            3) Terminated values for each ready agent. The special key
            "__all__" (required) is used to indicate env termination.
            4) Truncated values for each ready agent.
            5) Info values for each agent id (may be empty dicts).
        """
        marked = self.convert_to_marked(action_dict)
        
        self.solver.refine(marked)
        if self.observation_norm in ['L2', 'Linfty']:
            errors = mfem.Vector(self.solver.mesh.GetNE())
            for i in range(errors.Size()):
                errors[i] = 0.0
            done = False
            while not done:
                done = self.solver.step()
                current_errors = self.estimate(self.solver.sol)
                if self.observation_norm == "L2":
                    for i in range(errors.Size()):
                        errors[i] += current_errors[i]**2
                else:
                    for i in range(errors.Size()):
                        errors[i] = max(errors[i], current_errors[i])
                
        elif self.observation_norm == 'at_regrid_time':
            self.solver.step(self.regrid_time)
            errors = self.estimate(self.solver.sol)
            average_flux_jacobian = self.solver.ComputeFluxJacobian()
        
    def render(self):
        self.solver.render()
    
    def build_obs_map(self):
        """obs[:,i] = element indices within the window size. Only uniform periodic rectangular mesh is supported.
        """
        sdim = self.solver.sdim
        obs_map = np.zeros(((self.window_size*2 + 1)**self.solver.sdim, self.solver.mesh()), dtype=int)
        idx = np.arange(np.prod(self.num_grids)).reshape(self.num_grids)
        if sdim == 1:
            for x_offset in range(self.window_size*2 + 1):
                obs_map[x_offset, :] = np.roll(idx, (-self.window_size + x_offset))
        elif sdim == 2:
            i = 0
            for y_offset in range(self.window_size*2 + 1):
                for x_offset in range(self.window_size*2 + 1):
                    i += 1
                    obs_map[i] = np.roll(idx, (-self.window_size + x_offset, -self.window_size + y_offset), axis=(0,1))
        elif sdim == 3:
            i = 0
            for z_offset in range(self.window_size*2 + 1):
                for y_offset in range(self.window_size*2 + 1):
                    for x_offset in range(self.window_size*2 + 1):
                        i += 1
                        obs_map[i] = np.roll(idx, (-self.window_size + x_offset, -self.window_size + y_offset, -self.window_size + z_offset), axis=(0,1,2))
        
        
        