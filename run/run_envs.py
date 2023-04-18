import mfem.ser as mfem
from hcl import Envs
import numpy as np


def run_advection(meshfile, order, ode_solver_type, cfl, terminal_time, regrid_time=None):
    @mfem.jit.vector(td=True, vdim=1, interface="c++", sdim=2)
    def InitCond(x, t, out):
        out[0] = np.sin(np.pi*(x[0] + t))*np.sin(np.pi*(x[1] + t))

    @mfem.jit.vector(vdim=2, interface="c++", sdim=2)
    def Velocity(x, out):
        out[0] = 1.0
        out[1] = 1.0

    mesh = mfem.Mesh(meshfile)
    mesh.UniformRefinement()
    mesh.UniformRefinement()
    if ode_solver_type == 1:
        ode_solver = mfem.ForwardEulerSolver()
    elif ode_solver_type == 2:
        ode_solver = mfem.RK2Solver(1.0)
    elif ode_solver_type == 3:
        ode_solver = mfem.RK3SSolver()
    elif ode_solver_type == 4:
        ode_solver = mfem.RK4Solver()
    elif ode_solver_type == 6:
        ode_solver = mfem.RK6Solver()
    else:
        print("Unknown ODE solver type: " + str(ode_solver_type))
        exit

    InitCond.SetTime(0.0)
    advection = Envs.HyperbolicAMREnv(
        solver_name='advection',
        num_grids=[10, 10],
        domain_size=[1.0, 1.0],
        offsets=[0.0, 0.0],
        regrid_time=0.2,
        terminal_time=2.0,
        refine_mode='p',
        window_size=10,
        observation_norm='L2',
        allow_coarsening=False,
        seed=None,
        initial_condition=InitCond,
        solver_args={
            'order': order,
            'num_equations': 1,
            'refinement_mode': 'p',
            'ode_solver_type': ode_solver_type, 
            'cfl': cfl,
            'b': Velocity})

if __name__ == "__main__":
    from mfem.common.arg_parser import ArgParser
    parser = ArgParser(description='Run solver')
    parser.add_argument('-solver', '--solver_name',
                        default='advection',
                        action='store', type=str,
                        help="Solver name")
    parser.add_argument('-m', '--mesh',
                        default="./mesh/periodic-square-4x4.mesh",
                        action='store', type=str,
                        help='Mesh file to use.')
    parser.add_argument('-r', '--refine',
                        action='store', default=2, type=int,
                        help="Number of times to refine the mesh uniformly.")
    parser.add_argument('-o', '--order',
                        action='store', default=3, type=int,
                        help="Finite element order (polynomial degree)")
    parser.add_argument('-s', '--ode_solver',
                        action='store', default=4, type=int,
                        help="ODE solver: 1 - Forward Euler,\n\t" +
                        "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6.")
    parser.add_argument('-tf', '--t_final',
                        action='store', default=2.0, type=float,
                        help="Final time; start time is 0.")
    parser.add_argument("-dt", "--time_step",
                        action='store', default=-0.01, type=float,
                        help="Time step.")
    parser.add_argument("-rg", "--regrid-time",
                        action='store', default=0.5, type=float,
                        help="Regrid time, larger than time step")
    parser.add_argument('-c', '--cfl_number',
                        action='store', default=0.3, type=float,
                        help="CFL number for timestep calculation.")
    parser.add_argument('-vis', '--visualization',
                        action='store_true', default=True,
                        help='Enable GLVis visualization')
    parser.add_argument('-vs', '--visualization-steps',
                        action='store', default=1, type=float,
                        help="Visualize every n-th timestep.")
    args = parser.parse_args()
    parser.print_options(args)
    if args.solver_name == 'advection':
        run_advection(args.mesh, args.order, args.ode_solver, args.cfl_number, args.t_final, args.regrid_time)
    elif args.solver_name == 'burgers':
        run_burgers(args.mesh, args.order, args.ode_solver, args.cfl_number, args.t_final, args.regrid_time)
    elif args.solver_name == 'euler':
        run_euler(args.mesh, args.order, args.ode_solver, args.cfl_number, args.t_final, args.regrid_time)