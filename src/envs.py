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


class HyperbolicAMREnv(MultiAgentEnv):
    def __init__(self, 
                 solver_name:str, 
                 solver_args:Dict=None,
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
        self.refine_mode = refine_mode
        self.window_size = window_size
        self.randomize = seed is not None
        self.seed = seed
    
    def reset(self, *, seed: Optional[int]=None, options: Optional[dict] = None):
        pass
    
    def step(self, action_dict:MultiAgentDict) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        pass
    def render(self):
        pass
    
    