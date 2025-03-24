import os
import numpy as np

from cmplxfoil import CMPLXFOIL
from baseclasses import AeroProblem
from multipoint import multiPointSparse
from pygeo import DVGeometryCST
from mpi4py import MPI

# Multipoint function definitiions
# I don't think we need these here (maybe not at all)
def cruiseFuncs(x):
    pass

def cruiseFuncsSens(x, funcs):
    pass

def objCon(funcs, printOK):
    pass

class AeroSolver:
    def __init__(self, airfoil, Re, alpha, clObj, T=288.15, M=0.1, output_dir="output"):
        # Initialize free-stream conditions
        self.airfoil = airfoil
        self.Re = Re
        self.alpha = alpha
        self.clObj = clObj
        self.clObj
        self.T = T
        self.M = M
        
        # Create output folder if it does not already exist
        if not os.path.exists(os.path.join(os.getcwd(), output_dir)):
            os.mkdir(output_dir)
        # Initialize output directory
        self.output_dir = os.path.join(os.getcwd(), output_dir)
        
        # Initialize aerodynamic problem
        self._init_aero()
        self._init_solver()
        self._set_geom()
        
        # Input aerodynamic problem into CMPLXFOIL
        self.solver(self.aero_problem)
        # Input geomtry to CMPLXFOIL
        self.solver.setDVGeo(self.dvGeo)
        
    # Define aerodynamic probelm from baseClasses
    def _init_aero(self):
        self.aero_problem = AeroProblem(
            name="fc",
            alpha=self.alpha,
            mach=self.M,
            reynolds=self.Re,
            reynoldsLength=0.15, # if something breaks i changed this from 1 to 0.15
            T=self.T,
            areaRef=1.0,
            chordRef=1.0,
            evalFuncs=["cl", "cd"],
        )
        
    # Set-up CMPLXFOIL solver
    def _init_solver(self):
        # CMPLXFOIL options
        self.solver_options = {
            "writeSolution": True,
            "writeSliceFile": True,
            "writeCoordinates": True,
            "plotAirfoil": True,
            "outputDirectory": self.output_dir,
        }
        
        # CMPLXFOIL solver
        self.solver = CMPLXFOIL(self.airfoil, self.solver_options)
    
    # Define geometry with pyGeo
    def _set_geom(self):
        # Number of CST coefficients
        nCoeff = 4
        # Airfoil geometry
        self.dvGeo = DVGeometryCST(self.airfoil, numCST=nCoeff)
        
        self.dvGeo.addDV("upper_shape", dvType="upper", lowerBound=-0.1, upperBound=0.5)
        self.dvGeo.addDV("lower_shape", dvType="lower", lowerBound=-0.5, upperBound=0.1)
        
    def solvePrimal(self):
        funcs = {}
        # Set functions we want to evaluate
        self.solver.evalFunctions(self.aero_problem, funcs=funcs)
        # Make sure CMPLXFOIL doesn't break
        self.solver.checkSolutionFailure(self.aero_problem, funcs=funcs)

        return funcs
    
test = AeroSolver("naca0012.dat", 200000., 3., 0.6)
test.solvePrimal()