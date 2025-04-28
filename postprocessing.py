import matplotlib.pyplot as plt
import numpy as np
import animation

airfoil = np.loadtxt('naca0012.dat')
# Make sure to change the file name when rerunning the optimization
slsqp = np.loadtxt('Results/SLSQP/airfoil_iter32')
pen_grad = np.loadtxt('Results/PENALTY_GRAD/airfoil_sub4iter14')
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