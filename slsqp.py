# ======================================================================
#         Import modules
# ======================================================================
# rst imports (beg)
import os
import numpy as np
from mpi4py import MPI
from baseclasses import AeroProblem
from pygeo import DVConstraints, DVGeometryCST
from pyoptsparse import Optimization, OPT
from multipoint import multiPointSparse
from cmplxfoil import CMPLXFOIL, AnimateAirfoilOpt
import AeroSolver as AS
import Optimization as op 
from scipy.optimize import Bounds

# rst imports (end)

# ======================================================================
#         Specify parameters for optimization
# ======================================================================
# rst params (beg)
mycl = 0.5  # lift coefficient constraint
alpha = 3.0  # initial angle of attack (zero if the target cl is zero)
mach = 0.06  # Mach number
Re = 200000.  # Reynolds number
T = 288.15  # 1976 US Standard Atmosphere temperature @ sea level (K)

solver = AS.AeroSolver("naca0012.dat",Re,alpha,mycl)
# rst params (end)

# ======================================================================
#         Create multipoint communication object
# ======================================================================
# rst procs (beg)

# ======================================================================
#         Functions:
# ======================================================================
# rst funcs (beg)
def cruiseFuncs(x):
    print(x)
    # Set design vars
    solver.CFDSolver.DVGeo.setDesignVars(x)
    solver.aero_problem.setDesignVars(x)
    # Run CFD
    solver.CFDSolver(solver.aero_problem)
    # Evaluate functions
    funcs = {}
    solver.CFDSolver.evalFunctions(solver.aero_problem, funcs)
    solver.CFDSolver.checkSolutionFailure(solver.aero_problem, funcs)
    if MPI.COMM_WORLD.rank == 0:
        print("functions:")
        for key, val in funcs.items():
            if key == "DVCon1_thickness_constraints_0":
                continue
            print(f"    {key}: {val}")
    return funcs

# rst funcs (end)

# ======================================================================
#         Optimization Problem Set-up
# ======================================================================
# rst optprob (beg)
# Create optimization problem

op_problem = op.Optimization(solver)

cl_const = lambda cst: op_problem.cl(cst) - mycl #0 = clfunc - mycl. This is eq
op_problem.add_con(cl_const,lambda cst: op_problem.cl_grad(cst),"eq")
thickness_const = lambda cst: op_problem.thickness(cst)
op_problem.add_con(thickness_const,lambda cst: op_problem.thickness_grad(cst),"ineq")
radius_const = lambda cst: op_problem.radius(cst)
op_problem.add_con(radius_const,lambda cst: op_problem.radius_grad(cst),"ineq")
bounds = Bounds([-.1,-.1,-.1,-.1,-.5,-.5,-.5,-.5],[.5,.5,.5,.5,.1,.1,.1,.1])
op_problem.slsqp(bounds)
print(np.array(op_problem.log))



'''
optProb = Optimization("opt", MP.obj)

# Add objective
#optProb.addObj("obj", scale=1e4)

# Add variables from the AeroProblem
#solver.aero_problem.addVariablesPyOpt(optProb)

# Add DVGeo variables
solver.dvGeo.addVariablesPyOpt(optProb)

# Add constraints
#DVCon.addConstraintsPyOpt(optProb)

# Add cl constraint
optProb.addCon("cl_con_" + solver.aero_problem.name, lower=0.0, upper=0.0, scale=1.0)

# Enforce first upper and lower CST coefficients to add to zero
# to maintain continuity at the leading edge
jac = np.zeros((1, solver.nCoeff), dtype=float)
jac[0, 0] = 1.0
optProb.addCon(
    "first_cst_coeff_match",
    lower=0.0,
    upper=0.0,
    linear=True,
    wrt=["upper_shape", "lower_shape"],
    jac={"upper_shape": jac, "lower_shape": jac},
)

# The MP object needs the 'obj' and 'sens' function for each proc set,
# the optimization problem and what the objcon function is:
MP.setProcSetObjFunc("cruise", cruiseFuncs)
MP.setProcSetSensFunc("cruise", cruiseFuncsSens)
MP.setObjCon(objCon)
MP.setOptProb(optProb)
optProb.printSparsity()
optProb.getDVConIndex()
# rst optprob (end)

# rst opt (beg)
# Run optimization
optOptions = {"IFILE": os.path.join(solver.output_dir, "SLSQP.out")}
opt = OPT("SLSQP", options=optOptions)
sol = opt(optProb, MP.sens, storeHistory=os.path.join(solver.output_dir, "opt.hst"))
if MPI.COMM_WORLD.rank == 0:
    print(sol)
# rst opt (end)
'''



# ======================================================================
#         Postprocessing
# ======================================================================
# rst postprocessing (beg)
# Save the final figure
solver.CFDSolver.airfoilAxs[1].legend(["Original", "Optimized"], labelcolor="linecolor")
solver.CFDSolver.airfoilFig.savefig(os.path.join(solver.output_dir, "OptFoil.pdf"))

# # Animate the optimization
AnimateAirfoilOpt(solver.output_dir, "fc").animate(
    outputFileName=os.path.join(solver.output_dir, "OptFoil"), fps=10, dpi=300, extra_args=["-vcodec", "libx264"]
)
# rst postprocessing (end)