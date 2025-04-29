import matplotlib.pyplot as plt
import numpy as np
import animation
import os
import pandas as pd

if not os.path.exists("images"):
    os.mkdir*("images")
airfoil = np.loadtxt('naca0012.dat')
# Make sure to change the file name when rerunning the optimization
slsqp = np.loadtxt('Results/SLSQP/airfoil_iter32')
pen_grad = np.loadtxt('Results/PENALTY_GRAD/airfoil_sub6iter3')
pen_dfo = np.loadtxt('Results/PENALTY_DFO/airfoil_sub10iter4')
single = np.loadtxt('output/fc_039.dat')

# SLSQP vs NACA0012
plt.figure(figsize=(8, 2))
plt.plot(airfoil[...,0], airfoil[...,1], color='k', label='Original')
plt.plot(slsqp[...,0], slsqp[...,1], color='b', label='SLSQP')
plt.legend(loc='upper right')
plt.xlim([-0.1, 1.3])
plt.ylim([-0.07, 0.07])
plt.xlabel('x / c')
plt.ylabel('y / c')
plt.savefig('images/slsqp.png')

# Penalty method with BFGS vs. NACA0012
plt.figure(figsize=(8, 2))
plt.plot(airfoil[...,0], airfoil[...,1], color='k', label='Original')
plt.plot(pen_grad[...,0], pen_grad[...,1], color='b', label='Penalty BFGS')
plt.legend(loc='upper right')
plt.xlim([-0.1, 1.3])
plt.ylim([-0.07, 0.07])
plt.xlabel('x / c')
plt.ylabel('y / c')
plt.savefig('images/bfgs.png')

# Penalty method with Nelder-Mead vs. NACA0012
plt.figure(figsize=(8, 2))
plt.plot(airfoil[...,0], airfoil[...,1], color='k', label='Original')
plt.plot(pen_dfo[...,0], pen_dfo[...,1], color='b', label='Penalty Nelder-Mead')
plt.legend(loc='upper right')
plt.xlim([-0.1, 1.5])
plt.ylim([-0.07, 0.07])
plt.xlabel('x / c')
plt.ylabel('y / c')
plt.savefig('images/nelder.png')

# Our optimziation methods vs. optimization example given by CMPLXFOIL
plt.figure(figsize=(8, 2))
plt.plot(slsqp[...,0], slsqp[...,1], color='b', label='SLSQP')
plt.plot(single[...,0], single[...,1], color='k', label='Single Point')
plt.legend(loc='upper right')
plt.xlim([-0.1, 1.5])
plt.ylim([-0.07, 0.07])
plt.xlabel('x / c')
plt.ylabel('y / c')
plt.savefig('images/comparison.png')

folders = os.listdir(f"Results")
cl_full = []
cl_iters = []
cl_leg = []
cd_full = []
cd_iters = []
cd_leg = []
grad_full = []
grad_iters = []
grad_leg = []
penalties = []
pen_iters = []
pen_leg = []
for folder in folders:
    path = f"Results/{folder}/outdata.csv"
    df = pd.read_csv(path)
    cl_full.append(df['Cl'].to_numpy()-.5)
    cl_iters.append(df.index.to_numpy())
    cl_leg.append(folder)
    cd_full.append(df['Cd'].to_numpy())
    cd_iters.append(df.index.to_numpy())
    cd_leg.append(folder)
    try: 
        grad_full.append(df['gradnorm'].to_numpy())
        grad_leg.append(folder)
        grad_iters.append(df.index.to_numpy())
    except:
        pass
    try: 
        penalties.append(df['Penalty'].to_numpy())
        pen_leg.append(folder)
        pen_iters.append(df.index.to_numpy())
    except:
        pass
plt.figure()
for x in range(len(cl_full)):
    plt.plot(cl_iters[x],cl_full[x])
plt.legend(cl_leg)
plt.xlabel("Iterations/Subproblems")
plt.ylabel("Cl Violation")
plt.title("Cl Violation vs Iterations/Subproblems")
plt.savefig("images/cl.png")
plt.figure()
for x in range(len(cd_full)):
    plt.plot(cd_iters[x],cd_full[x])
plt.legend(cd_leg)
plt.xlabel("Iterations/Subproblems")
plt.ylabel("Cd")
plt.title("Cd vs Iterations/Subproblems")
plt.savefig("images/cd.png")
plt.figure()
for x in range(len(penalties)):
    plt.plot(pen_iters[x],penalties[x])
plt.legend(pen_leg)
plt.xlabel("Subproblems")
plt.ylabel("Penalty Value")
plt.title("Penalty Value vs Subproblems")
plt.savefig("images/penalty.png")
plt.figure()
for x in range(len(grad_full)):
    plt.plot(grad_iters[x],grad_full[x])
plt.legend(grad_leg)
plt.xlabel("Iterations/Subproblems")
plt.ylabel("Gradient Norm of Cd/Penalty")
plt.title("Gradient Norm of Cd/Penalty")
plt.savefig("images/gradient.png")


animation.animate("PENALTY_DFO",fps=10)
animation.animate("PENALTY_GRAD",fps=3)
animation.animate("PENALTY_DFO",fps=2)
