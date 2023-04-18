import mfem.ser as mfem
from mfem.ser import ProlongToMaxOrder
from hcl import Solvers
import numpy as np
import os

def run_advection(meshfile, order, ode_solver_type, cfl, terminal_time, regrid_time=None):
    vx, vy = 1., -1.
    @mfem.jit.vector(vdim=2, interface="c++", sdim=2, params={'vx':vx, 'vy':vy})
    def Velocity(x, out):
        out[0] = vx
        out[1] = vy
        
    @mfem.jit.vector(td=True, vdim=1, interface="c++", sdim=2, params={'vx':vx, 'vy':vy})
    def InitCond(x, t, out):
        out[0] = np.sin(np.pi*(x[0] - vx*t))*np.sin(np.pi*(x[1] - vy*t))
    
    mesh = mfem.Mesh(meshfile)
    mesh.UniformRefinement()
    mesh.UniformRefinement()

    advection = Solvers.AdvectionSolver(
        mesh, order, 1, 'h', ode_solver_type, cfl, b=Velocity)
    InitCond.SetTime(0.0)
    advection.init(InitCond)
    advection.init_renderer()
    n_regrid = 0
    
    advection.save(n_regrid, './results/')
    while advection.t < terminal_time:
        n_regrid += 1
        advection.terminal_time = min(regrid_time*n_regrid, terminal_time)
        print(f'(t_start, t_end) = ({advection.t:.4f}, {advection.terminal_time:.4f})')
        done = False
        while not done:
            done, dt = advection.step()
        error, errors = advection.estimate()
        advection.save(n_regrid, './results/')
        advection.render()
        print(f'{error=}')
        
        advection.render()
    print(advection.t)
    
def run_burgers(meshfile, order, ode_solver_type, cfl, terminal_time, regrid_time=None):
    """run burgers solver

    Args:
        meshfile (str): mesh file
        order (int): order of polynomial
        ode_solver_type (int): ode solver type. 1: Euler, 2-4 and 6: RK
        cfl (float): CFL number
        terminal_time (float): terminal time
    """
    @mfem.jit.vector(vdim=1, interface="c++", sdim=2)
    def InitCond(x, out):
        out[0] = np.sin(np.pi*(x[0]))*np.sin(np.pi*x[1])

    mesh = mfem.Mesh(meshfile)
    mesh.UniformRefinement()
    mesh.UniformRefinement()

    burgers = Solvers.BurgersSolver(
        mesh, order, 1, 'h', ode_solver_type, cfl)
    InitCond.SetTime(0.0)
    burgers.init(InitCond)
    burgers.init_renderer()

    done = False

    n_regrid = 0
    burgers.save(n_regrid)
    while burgers.t < terminal_time:
        n_regrid += 1
        burgers.terminal_time = min(regrid_time*n_regrid, terminal_time)
        print(f'(t_start, t_end) = ({burgers.t}, {burgers.terminal_time})')
        done = False
        while not done:
            done, dt = burgers.step()
        error, errors = burgers.estimate()
        burgers.save(n_regrid)
        burgers.render()
        print(f'{error=}')
        
        burgers.render()
    print(burgers.t)
    
def run_euler(meshfile, order, ode_solver_type, cfl, terminal_time, regrid_time=None):
    @mfem.jit.vector(vdim=4, td=True, interface="c++")
    def InitCond(x, t, out):
        xt = x
        xt[0] -= t*0.5
        
        # "Fast vortex"
        radius = 0.2
        Minf = 0.5
        beta = 1. / 5.

        xc = 0.0
        yc = 0.0
        # Nice units
        vel_inf = 1.
        den_inf = 1.

        specific_heat_ratio = 1.4
        gas_constant = 1.0

        pres_inf = (den_inf / specific_heat_ratio) * \
            (vel_inf / Minf) * (vel_inf / Minf)
        temp_inf = pres_inf / (den_inf * gas_constant)

        r2rad = 0.0
        r2rad += (xt[0] - xc) * (xt[0] - xc)
        r2rad += (xt[1] - yc) * (xt[1] - yc)
        r2rad /= (radius * radius)

        shrinv1 = 1.0 / (specific_heat_ratio - 1.)

        velX = vel_inf * \
            (1 - beta * (xt[1] - yc) / radius * np.exp(-0.5 * r2rad))
        velY = vel_inf * beta * (xt[0] - xc) / radius * np.exp(-0.5 * r2rad)
        vel2 = velX * velX + velY * velY

        specific_heat = gas_constant * specific_heat_ratio * shrinv1

        temp = temp_inf - (0.5 * (vel_inf * beta) *
                    (vel_inf * beta) / specific_heat * np.exp(-r2rad))

        den = den_inf * (temp/temp_inf)**shrinv1
        pres = den * gas_constant * temp
        energy = shrinv1 * pres / den + 0.5 * vel2
        
        out[0] = den
        out[1] = den*velX
        out[2] = den*velY
        out[3] = den*energy

    mesh = mfem.Mesh(meshfile)
    mesh.UniformRefinement()
    mesh.UniformRefinement()

    euler = Solvers.EulerSolver(
        mesh, order, 4, 'h', ode_solver_type, cfl, specific_heat_ratio=1.4, gas_constant=1.0)
    euler.init(InitCond)
    euler.init_renderer()

    done = False
    n_regrid = 0
    euler.mesh.Print(f"euler-{n_regrid:06}.mesh", 8)
    euler.sol.Save(f"euler-{n_regrid:06}.gf", 8)
    while euler.t < terminal_time:
        n_regrid += 1
        euler.terminal_time = min(regrid_time*n_regrid, terminal_time)
        print(f'(t_start, t_end) = ({euler.t}, {euler.terminal_time})')
        done = False
        while not done:
            done, dt = euler.step()
        euler.mesh.Print(f"euler-{n_regrid:06}.mesh", 8)
        euler.sol.Save(f"euler-{n_regrid:06}.gf", 8)
        error, errors = euler.estimate()
        print(f'{error=}')
        
        euler.render()
    print(euler.t)


if __name__ == "__main__":
    from mfem.common.arg_parser import ArgParser
    parser = ArgParser(description='Run solver')
    parser.add_argument('-solver', '--solver_name',
                        default='advection',
                        action='store', type=str,
                        help="Solver name")
    parser.add_argument('-m', '--mesh',
                        default=os.path.dirname(os.path.realpath(__file__)) + "/../mesh/periodic-square-4x4.mesh",
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
                        action='store', default=0.1, type=float,
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