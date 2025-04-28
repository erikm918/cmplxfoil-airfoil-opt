import AeroSolver as AS
import Optimization as op 
from scipy.optimize import Bounds
import animation
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-opt', help='Select which optimizer to use from list of: slsqp, penalty_grad, penalty_dfo.',
                    type=str, default='slsqp')
args = parser.parse_args()

mycl = 0.5    # Lift coefficient constraint, i.e. target lift coefficient
alpha = 3.0   # Angle of attack
mach = 0.06   # Mach number
Re = 200000.  # Reynolds number
T = 288.15    # 1976 US Standard Atmosphere temperature @ sea level (K)

solver = AS.AeroSolver("naca0012.dat",Re,alpha,mycl)

op_problem = op.Optimization(solver)

# Add cl constraint
cl_const = lambda cst: op_problem.cl(cst) - mycl #0 = clfunc - mycl. This is eq
op_problem.add_con(cl_const,lambda cst: op_problem.cl_grad(cst),"eq")
# Problem bounds for slsqp
bounds = Bounds([-.1,-.1,-.1,-.1,-.5,-.5,-.5,-.5],[.5,.5,.5,.5,.1,.1,.1,.1])
if args.opt == 'slsqp':
    op_problem.slsqp(bounds=bounds)
elif args.opt == 'penalty_grad':
    op_problem.penalty()
elif args.opt == 'penalty_dfo':
    op_problem.penalty(dfo=True, max_iter=20, tau_min=1e-3)
else:
    print('\n#############################################\n')
    print('Optimization option not found, quitting.')
    print('\n#############################################')
    
    exit(1)