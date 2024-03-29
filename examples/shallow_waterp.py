'''
   MFEM example 18
      This is a version of Example 18 with a simple adaptive mesh
      refinement loop. 
      See c++ version in the MFEM library for more detail 
'''
from mfem.par import getShallowWaterEquation, RusanovFlux
from mfem.common.arg_parser import ArgParser
import mfem.par as mfem
from mfem.par import intArray
from os.path import expanduser, join, dirname
import numpy as np
from numpy import sqrt, pi, cos, sin, hypot, arctan2, exp
from scipy.special import erfc


from mpi4py import MPI
num_procs = MPI.COMM_WORLD.size
myid = MPI.COMM_WORLD.rank
smyid = '{:0>6d}'.format(myid)

@mfem.jit.vector(vdim=3, interface="c++")
def InitCond(x, out):
    maxval = 10.0
    minval = 6.0
    r_sigma = 0.05
    xc = 0.0
    yc = 0.0
    dx = x[0] - xc
    dy = x[1] - yc
    
    out[0] = (maxval - minval) * exp(-0.5 * r_sigma * (dx * dx + dy * dy)) + minval
    out[1] = 0.0
    out[2] = 0.0
    
    
@mfem.jit.vector(vdim=2, interface="c++")
def MeshTransform(x, out):
    
    out[0] = x[0]*25.0
    out[1] = x[1]*25.0
        

def run(problem=1,
        ser_ref_levels=0,
        par_ref_levels=4,
        order=3,
        IntOrderOffset=3,
        ode_solver_type=4,
        t_final=0.5,
        dt=-0.01,
        cfl=0.3,
        visualization=True,
        vis_steps=50,
        meshfile=''):
    g = 9.81
    # 2. Read the mesh from the given mesh file. This example requires a 2D
    #    periodic mesh, such as ../data/periodic-square.mesh.
    meshfile = expanduser(join(dirname(__file__), 'mesh', meshfile))
    mesh = mfem.Mesh(meshfile, 1, 1)
    dim = mesh.Dimension()
    num_equations = dim + 1
    mesh.Transform(MeshTransform)

    # 3. Define the ODE solver used for time integration. Several explicit
    #    Runge-Kutta methods are available.
    ode_solver = None
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

    # 4. Refine the mesh to increase the resolution. In this example we do
    #    'ref_levels' of uniform refinement, where 'ref_levels' is a
    #    command-line parameter.
    for lev in range(ser_ref_levels):
        mesh.UniformRefinement()

    pmesh = mfem.ParMesh(MPI.COMM_WORLD, mesh)
    del mesh
    for lev in range(par_ref_levels):
        pmesh.UniformRefinement()

    # 5. Define the discontinuous DG finite element space of the given
    #    polynomial order on the refined mesh.

    fec = mfem.DG_FECollection(order, dim)
    # Finite element space for a scalar (thermodynamic quantity)
    fes = mfem.ParFiniteElementSpace(pmesh, fec)
    # Finite element space for a mesh-dim vector quantity (momentum)
    dfes = mfem.ParFiniteElementSpace(pmesh, fec, dim, mfem.Ordering.byNODES)
    # Finite element space for all variables together (total thermodynamic state)
    vfes = mfem.ParFiniteElementSpace(
        pmesh, fec, num_equations, mfem.Ordering.byNODES)

    assert fes.GetOrdering() == mfem.Ordering.byNODES, "Ordering must be byNODES"
    if myid == 0:
        print("Number of unknowns: " + str(vfes.GetVSize()))

    # # 6. Define the initial conditions, save the corresponding mesh and grid
    # #    functions to a file. This can be opened with GLVis with the -gc option.
    # #    The solution u has components {density, x-momentum, y-momentum, energy}.
    # #    These are stored contiguously in the BlockVector u_block.

    offsets = [k*vfes.GetNDofs() for k in range(num_equations+1)]
    offsets = mfem.intArray(offsets)
    u_block = mfem.BlockVector(offsets)
    height = mfem.ParGridFunction(fes, u_block,  offsets[0])

    # #
    # #  Define coefficient using VecotrPyCoefficient and PyCoefficient
    # #  A user needs to define EvalValue method
    # #
    sol = mfem.ParGridFunction(vfes, u_block.GetData())
    sol.ProjectCoefficient(InitCond)

    # mesh.Print("vortex.mesh", 8)
    # for k in range(num_equations):
    #     uk = mfem.ParGridFunction(fes, u_block.GetBlock(k).GetData())
    #     sol_name = "vortex-" + str(k) + "-init.gf"
    #     uk.Save(sol_name, 8)

    #  7. Set up the nonlinear form corresponding to the DG discretization of the
    #     flux divergence, and assemble the corresponding mass matrix.
    numericalFlux = RusanovFlux()
    euler = getShallowWaterEquation(vfes, numericalFlux, g, IntOrderOffset)
    
    if (visualization):
        sout = mfem.socketstream("localhost", 19916)
        sout.precision(8)
        sout.send_text("parallel " + str(num_procs) + " " + str(myid))
        sout.send_solution(pmesh, height)
        sout << "view 0 0\n"
        sout << "keys jl********\n"
        sout << "valuerange 5 8\n"
        sout << "autoscale off\n"
        # sout << "pause\n"
        # sout << "valuerange 3 7\n"
        # sout << "autoscale off\n"
        sout.flush()
        if myid == 0:
            print("GLVis visualization paused.")
            print(" Press space (in the GLVis window) to resume it.")

    # Determine the minimum element size.
    my_hmin = 0
    if (cfl > 0):
        my_hmin = min([pmesh.GetElementSize(i, 1) for i in range(pmesh.GetNE())])
    hmin = MPI.COMM_WORLD.allreduce(my_hmin, op=MPI.MIN)

    t = 0.0
    euler.SetTime(t)
    ode_solver.Init(euler)
    if (cfl > 0):
        #  Find a safe dt, using a temporary vector. Calling Mult() computes the
        #  maximum char speed at all quadrature points on all faces.
        z = mfem.Vector(euler.Width())
        euler.Mult(sol, z)
        my_dt = cfl * hmin / euler.getMaxCharSpeed() / (2*vfes.GetMaxElementOrder() + 1)
        dt = MPI.COMM_WORLD.allreduce(my_dt, op=MPI.MIN)

    # Integrate in time.
    done = False
    ti = 0
    while not done:
        dt_real = min(dt, t_final - t)

        t, dt_real = ode_solver.Step(sol, t, dt_real)

        if (cfl > 0):
            my_dt = cfl * hmin / euler.getMaxCharSpeed() / (2*vfes.GetMaxElementOrder() + 1)
            dt = MPI.COMM_WORLD.allreduce(my_dt, op=MPI.MIN)
        ti = ti+1
        done = (t >= t_final - 1e-8*dt)
        if (done or ti % vis_steps == 0):
            if myid == 0:
                print("time step: " + str(ti) + ", time: " + "{:g}".format(t))
            if (visualization):
                sout.send_text("parallel " + str(num_procs) + " " + str(myid))
                sout.send_solution(pmesh, height)
                sout.flush()

    #  9. Save the final solution. This output can be viewed later using GLVis:
    #    "glvis -m vortex.mesh -g vortex-1-final.gf".
    for k in range(num_equations):
        uk = mfem.ParGridFunction(fes, u_block.GetBlock(k).GetData())
        sol_name = "euler-" + str(k) + "-final."+smyid
        uk.Save(sol_name, 8)
    if myid == 0:
        print(" done")
    # 10. Compute the L2 solution error summed for all components.
    if (t_final == 2.0):
        error = sol.ComputeLpError(2., InitCond)
        if myid == 0:
            print("Solution error: " + str(error))


if __name__ == "__main__":

    parser = ArgParser(description='Ex18')
    parser.add_argument('-m', '--mesh',
                        default='periodic-square-4x4.mesh',
                        action='store', type=str,
                        help='Mesh file to use.')
    parser.add_argument('-p', '--problem',
                        action='store', default=1, type=int,
                        help='Problem setup to use. See options in velocity_function().')
    parser.add_argument('-rs', '--refine_serial',
                        action='store', default=0, type=int,
                        help="Number of times to refine the mesh uniformly before parallel.")
    parser.add_argument('-rp', '--refine_parallel',
                        action='store', default=2, type=int,
                        help="Number of times to refine the mesh uniformly after parallel.")
    parser.add_argument('-o', '--order',
                        action='store', default=3, type=int,
                        help="Finite element order (polynomial degree)")
    parser.add_argument('-s', '--ode_solver',
                        action='store', default=4, type=int,
                        help="ODE solver: 1 - Forward Euler,\n\t" +
                        "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6.")
    parser.add_argument('-tf', '--t_final',
                        action='store', default=10.0, type=float,
                        help="Final time; start time is 0.")
    parser.add_argument("-dt", "--time_step",
                        action='store', default=-0.01, type=float,
                        help="Time step.")
    parser.add_argument('-c', '--cfl_number',
                        action='store', default=0.3, type=float,
                        help="CFL number for timestep calculation.")
    parser.add_argument('-vis', '--visualization',
                        action='store_true', default=True,
                        help='Enable GLVis visualization')
    parser.add_argument('-vs', '--visualization-steps',
                        action='store', default=50, type=float,
                        help="Visualize every n-th timestep.")

    args = parser.parse_args()

    if myid == 0:
        parser.print_options(args)

    run(problem=args.problem,
        ser_ref_levels=args.refine_serial,
        par_ref_levels=args.refine_parallel,
        order=args.order,
        ode_solver_type=args.ode_solver,
        t_final=args.t_final,
        dt=args.time_step,
        cfl=args.cfl_number,
        visualization=args.visualization,
        vis_steps=args.visualization_steps,
        meshfile=args.mesh)
