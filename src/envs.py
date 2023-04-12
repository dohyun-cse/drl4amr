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

class HyperbolicAMREnv(MultiAgentEnv):
    def __init__(self):
        pass
    
    def reset(self, *, seed: Optional[int]=None, options: Optional[dict] = None):
        pass
    
    def step(self, action_dict:MultiAgentDict) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        pass
    def render(self):
        pass
    
    