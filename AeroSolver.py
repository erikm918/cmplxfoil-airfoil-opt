import os
import numpy as np

from cmplxfoil import CMPLXFOIL
from baseclasses import AeroProblem
from multipoint import multiPointSparse
from pygeo import DVGeometryCST
from mpi4py import MPI

class AeroSolver:
    def __init__(self, airfoil, Re, alpha, clObj, T=288.15, M=0.06, output_dir="output"): 
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
        self.CFDSolver(self.aero_problem)
        # Input geomtry to CMPLXFOIL
        self.CFDSolver.setDVGeo(self.dvGeo)
        print(self.dvGeo.getValues())
        
    # Define aerodynamic probelm from baseClasses
    def _init_aero(self):
        self.aero_problem = AeroProblem(
            name="fc",
            alpha=self.alpha,
            mach=self.M,
            reynolds=self.Re,
            reynoldsLength=1, # if something breaks i changed this from 1 to 0.15
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
        self.CFDSolver = CMPLXFOIL(self.airfoil, self.solver_options)
    
    # get geometry with pyGeo
    def _set_geom(self):
        # Number of CST coefficients
        self.nCoeff = 4
        # Airfoil geometry
        self.dvGeo = DVGeometryCST(self.airfoil, numCST=self.nCoeff)
        
        self.dvGeo.addDV("upper_shape", dvType="upper", lowerBound=-0.1, upperBound=0.5)
        self.dvGeo.addDV("lower_shape", dvType="lower", lowerBound=-0.5, upperBound=0.1)
        
        self.dvGeo.addPointSet(np.loadtxt(self.airfoil), 'airfoilPoints')
    
    def solvePrimal(self):
        funcs = {}
        # Set functions we want to evaluate
        self.CFDSolver.evalFunctions(self.aero_problem, funcs=funcs)
        # Make sure CMPLXFOIL doesn't break
        self.CFDSolver.checkSolutionFailure(self.aero_problem, funcs=funcs)
        
        return funcs
        
    def findFunctionSens(self):
        func_sens = {}
        self.CFDSolver.evalFunctionsSens(self.aero_problem, func_sens)
        
        return func_sens
    
    def updateCSTCoeff(self, new_CST, new_airfoil='updated_airfoil.dat'):
        self.dvGeo.setDesignVars(
            {
                "upper_shape": np.array([new_CST[0], new_CST[1], new_CST[2], new_CST[3]]),
                "lower_shape": np.array([new_CST[4], new_CST[5], new_CST[6], new_CST[7]]),
            }
        )
        
        self.dvGeo.update('airfoilPoints')
        self.dvGeo.update('cmplxfoil_fc_coords')
        
        updated_points = self.dvGeo.points.get('airfoilPoints')
        np.savetxt(new_airfoil, updated_points['points'])

        # Re-initialize CMPLXFOIL with the updated geometry
        self.CFDSolver = CMPLXFOIL(new_airfoil, self.solver_options)
        self.CFDSolver(self.aero_problem)
        self.CFDSolver.setDVGeo(self.dvGeo)

    def getValuesNp(self):
        cstdict = self.dvGeo.getValues()
        listcst = list(cstdict["upper_shape"]) + list(cstdict["lower_shape"])
        return np.array(listcst)