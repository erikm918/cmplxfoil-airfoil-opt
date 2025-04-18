import os
import numpy as np

from cmplxfoil import CMPLXFOIL
from baseclasses import AeroProblem
from multipoint import multiPointSparse
from pygeo import DVGeometryCST, DVConstraints
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
        
        self._init_aero()   # AeroProblem init
        self._init_solver() # CMPLXFOIL init
        self._set_geom()    # DVGeo Init
        self._init_dvcon()  # DVCon Init
        
        # Input aerodynamic problem into CMPLXFOIL
        self.CFDSolver(self.aero_problem)
        # Input geomtry to CMPLXFOIL
        self.CFDSolver.setDVGeo(self.DVGeo)
        print(self.DVGeo.getValues())
        
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
        self.DVGeo = DVGeometryCST(self.airfoil, numCST=self.nCoeff)
        
        # Bounds for the CST coefficients
        self.DVGeo.addDV("upper_shape", dvType="upper", lowerBound=-0.1, upperBound=0.5)
        self.DVGeo.addDV("lower_shape", dvType="lower", lowerBound=-0.5, upperBound=0.1)
        
        # 
        self.DVGeo.addPointSet(np.loadtxt(self.airfoil), 'airfoilPoints')
    
    def _init_dvcon(self):
        self.DVCon = DVConstraints()
        self.DVCon.setDVGeo(self.DVGeo)
        self.DVCon.setSurface(self.CFDSolver.getTriangulatedMeshSurface())

        # Thickness, volume, and leading edge radius constraints
        le = 0.0001
        wingtipSpacing = 0.1
        leList = [[le, 0, wingtipSpacing], [le, 0, 1.0 - wingtipSpacing]]
        teList = [[1.0 - le, 0, wingtipSpacing], [1.0 - le, 0, 1.0 - wingtipSpacing]]
        self.DVCon.addThicknessConstraints2D(leList, teList, 2, 100, lower=0.1, scaled=True)
        le = 0.01
        leList = [[le, 0, wingtipSpacing], [le, 0, 1.0 - wingtipSpacing]]
        self.DVCon.addLERadiusConstraints(leList, 2, axis=[0, 1, 0], chordDir=[-1, 0, 0], lower=0.85, scaled=True)
    
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
        self.DVGeo.setDesignVars(
            {
                "upper_shape": np.array([new_CST[0], new_CST[1], new_CST[2], new_CST[3]]),
                "lower_shape": np.array([new_CST[4], new_CST[5], new_CST[6], new_CST[7]]),
            }
        )
        
        self.DVGeo.update('airfoilPoints')
        
        updated_points = self.DVGeo.points.get('airfoilPoints')
        np.savetxt(new_airfoil, updated_points['points'])

        # Re-initialize CMPLXFOIL with the updated geometry
        self.CFDSolver = CMPLXFOIL(new_airfoil, self.solver_options)
        self.CFDSolver(self.aero_problem)
        self.CFDSolver.setDVGeo(self.DVGeo)
    
    def findConstraint(self, min_t=0.25, min_r=0.75):
        constraints = {}
        
        # Evaluate constraints
        # Constraints are returned as normalized values, based on the original values for the 
        # airfoil
        self.DVCon.evalFunctions(constraints)
        
        # Thickness and LE Radius constraint for current airfoil shape
        thickness_con = constraints['DVCon1_thickness_constraints_0']
        le_con = constraints['DVCon1_leradius_constraints_0']
        
        # Return constraints in canonical form for scipy
        thickness_con -= min_t # t - 0.25(t_0) >= 0
        le_con -= min_r  # r - 0.75(r_0) >= 0
        
        return thickness_con, le_con

    def findConSens(self):
        cons_sens = {}
        
        # Find constraint sensitivity
        self.DVCon.evalFunctionsSens(cons_sens)
        '''
        Since we are using CST points, each point defined within DVCon will have four total derivatives;
        one for each CST coefficient. For example, a given thickness constraint point will have the 
        derivatives: dT/dN1, dT/dN2, dT/dN3, dT/dN4. 
        
        Additionally, we have different values of CST for the upper and lower surfaces and therefore a 
        derivative is calculated for the upper CST coefficients and the lower CST coefficients. So, we
        two different dictionaries within our constraint dictionaries. One dicitionary corresponds to 
        the upper surface and the other to the lower surface. Concatinating these two provide the full 
        sensitivity to our thickness and LE radius constraints. 
        '''
        
        # Thickness sensitivity
        thickness_sens_upper = cons_sens['DVCon1_thickness_constraints_0']['upper_shape']
        thickness_sens_lower = cons_sens['DVCon1_thickness_constraints_0']['lower_shape']
        
        # LE radius sensitivity
        le_sens_upper = cons_sens['DVCon1_leradius_constraints_0']['upper_shape']
        le_sens_lower = cons_sens['DVCon1_leradius_constraints_0']['lower_shape']
        
        # Combining our sensitivities
        thickness_sens = np.concatenate((thickness_sens_upper.flatten(), thickness_sens_lower.flatten()))
        le_sens = np.concatenate((le_sens_upper.flatten(), le_sens_lower.flatten()))
        
        return thickness_sens, le_sens

    def getValuesNp(self):
        cstdict = self.DVGeo.getValues()
        listcst = list(cstdict["upper_shape"]) + list(cstdict["lower_shape"])
        return np.array(listcst)