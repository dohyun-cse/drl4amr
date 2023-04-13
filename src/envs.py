import sys
import numpy as np
import math

import gymnasium as gym
import sys

from gymnasium import spaces, utils
from gymnasium.spaces import Box, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import logging
from typing import Callable, Dict, List, Tuple, Optional, Union, Set, Type, Sequence

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
                 num_grids:Sequence[int]=[10, 10],
                 domain_size:Sequence[float]=[1., 1.],
                 offsets:Sequence[float]=[0., 0.],
                 regrid_time:float=-1.0,
                 terminal_time:float= -1.0,
                 refine_mode:str='p',
                 window_size:int=3,
                 visualization:bool=False,
                 seed:Optional[int]=None,
                 solver_args:Dict=None):
        
        #region Create Mesh
        # Uniform tensor elements
        self.num_grids = num_grids
        if len(num_grids) == 1:
            self.mesh:mfem.Mesh = mfem.Mesh.MakeCartesian1D(n=num_grids[0], sx=domain_size[0])
            translations = (mfem.Vector([domain_size[0]]))
        elif len(num_grids) == 2:
            self.mesh:mfem.Mesh = mfem.Mesh.MakeCartesian2D(nx=num_grids[0], ny=num_grids[1], sx=domain_size[0], sy=domain_size[1], sfc_ordering=False, type=mfem.Element.QUADRILATERAL)
            translations = (mfem.Vector([domain_size[0], 0.]), mfem.Vector([0., domain_size[1]]))
        elif len(num_grids) == 3:
            self.mesh:mfem.Mesh = mfem.Mesh.MakeCartesian3D(nx=num_grids[0], ny=num_grids[1], nz=num_grids[2], sx=domain_size[0], sy=domain_size[1], sz=domain_size[2], sfc_ordering=False, type=mfem.Element.HEXAHEDRON)
            translations = (mfem.Vector([domain_size[0], 0., 0.]), mfem.Vector([0., domain_size[1], 0.]), mfem.Vector([0., 0., domain_size[2]]))
            
        # periodic boundary
        self.mesh.CreatePeriodicVertexMapping(translations)
        
        # Shift domain if necessary
        if not all(np.equal(offsets, 0)):
            if len(offsets) == 1:
                @mfem.jit.vector(vdim=1, interface="c++", sdim=1, dependency=(offsets))
                def Velocity(x, out):
                    out[0] = x[0] - offsets[0]
            elif len(offsets) == 2:
                @mfem.jit.vector(vdim=2, interface="c++", sdim=2, dependency=(offsets))
                def Velocity(x, out):
                    out[0] = x[0] - offsets[0]
                    out[1] = x[1] - offsets[1]
            elif len(offsets) == 3:
                @mfem.jit.vector(vdim=3, interface="c++", sdim=3, dependency=(offsets))
                def Velocity(x, out):
                    out[0] = x[0] - offsets[0]
                    out[1] = x[1] - offsets[1]
                    out[2] = x[2] - offsets[2]
            self.mesh.Transform(Velocity)
            
        # Make it non-conforming mesh
        self.mesh.EnsureNCMesh()
        
        # build mapping between elements
        self.obs_map = self.build_obs_map()
        
        #endregion
        
        
        #region Create Solver
        if solver_name == 'advection':
            self.solver = solver.AdvectionSolver(mesh=self.mesh, **solver_args)
        elif solver_name == 'burgers':
            self.solver = solver.BurgersSolver(mesh=self.mesh, **solver_args)
        elif solver_name == 'euler':
            self.solver = solver.EulerSolver(mesh=self.mesh, **solver_args)
        else:
            raise ValueError(f'Unsupported solver name: {solver_name}.')
        #endregion
        
        if regrid_time < 0 or regrid_time > solver_args.get('terminal_time', 2):
            raise ValueError(f'Regrid time is not provided or larger than the terminal time: {regrid_time}')
        
        self.observation_norm = 'L2'
        self.flux_observation = 'at_regrid_time'
        self.regrid_time = regrid_time
        self.terminal_time = terminal_time
        self.refine_mode = refine_mode
        self.window_size = window_size
        self.seed = seed
        self.visualization = visualization
        
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
        
        # Perform Actions
        self.solver.refine(marked)
        
        #region Advance in time        
        errors = np.zeros((self.solver.mesh.GetNE(),),np.double)
        total_error = 0.0
        for i in range(errors.Size()):
            errors[i] = 0.0
        
        # update solver terminal time
        self.solver.terminal_time = min(self.regrid_time, self.terminal_time - self.solver.t)
        done = False
        while not done:
            done, dt = self.solver.step() # advance time
            # if error should be measured at each time step
            if self.observation_norm in ["L2", "Linfty"]:
                # compute error
                current_total_error, current_errors = self.solver.compute_L2_errors()
                current_errors = current_errors.GetDataArray()
                if self.observation_norm == "L2":
                    # square it and weighted sum with dt
                    errors += current_errors**2*dt 
                    total_error += current_total_error**2*dt
                elif self.observation_norm == "Linfty":
                    # take element-wise maximum
                    errors = np.maximum(errors, current_errors)
                    total_error = max(total_error, current_total_error)
        
        if self.observation_norm == "at_regrid_time": # only measure at the regrid time
            # compute current error
            total_error, errors = self.solver.compute_L2_errors()
            errors = errors.GetDataArray()
        #endregion
        
        #region Logscale Error
        # take logarithm
        total_error = np.log(total_error)
        errors = np.log(errors)
        
        if self.observation_norm == "L2":
            # If L2 error, then it must be squared!
            # divide it by 2 because we took log
            # Also divide it by dt for reweighting
            errors = errors/2 - np.log(self.regrid_time)/2
            total_error = total_error/2 - np.log(self.regrid_time)/2
        #endregion
        
        #region FluxJacobian
        Jacobian, eigs = self.solver.ComputeElementAverageFluxJacobian()
        Jacobian = Jacobian.GetDataArray().reshape((self.solver.vdim**2*self.solver.sdim, self.solver.mesh.GetNE()))
        eigs = eigs.GetDataArray().reshape((self.solver.vdim*self.solver.sdim, self.solver.mesh.GetNE()))
        #endregion
        
        #region Make Map!        
        elementwise_observation = np.append(errors.reshape((1, 1, self.solver.mesh.GetNE())), Jacobian)
        
        #endregion
        
        
        
        
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
        
        
        