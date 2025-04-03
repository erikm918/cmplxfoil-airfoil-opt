import AeroSolver.py
from scipy.optimize import minimize
from scipy.optimize import Bounds
import numpy as np


def cl(solver,cst):
    alias = solver
    alias.updateCSTCoeff(cst)
    funcs = alias.solvePrimal()
    cl = funcs["fc_cl"]
    return cl

def cd(solver,cst):
    alias = solver
    alias.updateCSTCoeff(cst)
    funcs = alias.solvePrimal()
    cd = funcs["fc_cd"]
    return cd

def cd_grad(solver,cst):
    alias = solver
    alias.updateCSTCoeff(cst)
    grads = alias.findFunctionSens()
    fc_cd = grads['fc_cd']
    combined = fc_cd['upper_shape'].tolist() + fc_cd['lower_shape'].tolist()
    return np.array(combined)

def cl_grad(solver,cst):
    alias = solver
    alias.updateCSTCoeff(cst)
    grads = alias.findFunctionSens()
    fc_cl = grads['fc_cl']
    combined = fc_cl['upper_shape'].tolist() + fc_cl['lower_shape'].tolist()
    return np.array(combined)

class Optimization:
    def __init__(self, solver): #solver is a class of AeroSolver type
        self.constraints = []
        
    def add_con(self, func, con_type): #need to pass a lambda function or some function with inputs
        constraint = {'type': con_type,
             'fun': func}
        self.constraints.append(constraint)
       
    def add_func(self, function,x0):
        self.func = function
        self.x0 = x0

    def optimize(self,method):
        pass