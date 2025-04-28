from scipy.optimize import minimize, Bounds
import numpy as np
import pandas as pd
import os
import shutil

from AeroSolver import AeroSolver

class Optimization:
    def __init__(self, solver: AeroSolver):
        self.solver = solver
        self.constraints = []
        self.log = []
        self.add_geom_con()
        self.iters = 0 #Iteration count 
        if not os.path.exists("Results"): #Setting up file director
            os.mkdir("Results")

    def update(self,cst): #This function updates primals, etc, for use in other funcs
        if np.all(self.solver.getValuesNp() == cst): 
            pass
        else: #If hasn't been updated already, then update. This prevents needless updating of primals
            self.solver.updateCSTCoeff(cst)
            self.solver.solvePrimal()
            self.solver.findFunctionSens()
            self.solver.findConstraint()
            self.solver.findConSens()

    #Functions for getting cl, cd. All follow the same basic form: Updating, then grabbing the 
    # relevant data and returning it
    def cl(self,cst):
        self.update(cst)
        funcs = self.solver.funcs
        cl = funcs["fc_cl"]
        return cl

    def cd(self,cst):
        self.update(cst)
        funcs = self.solver.funcs
        cd = funcs["fc_cd"]
        return cd

    def cd_grad(self,cst):
        self.update(cst)
        grads = self.solver.func_sens
        fc_cd = grads['fc_cd']
        combined = fc_cd['upper_shape'].tolist() + fc_cd['lower_shape'].tolist()
        return np.array(combined)

    def cl_grad(self,cst):
        self.update(cst)
        grads = self.solver.func_sens
        fc_cl = grads['fc_cl']
        combined = fc_cl['upper_shape'].tolist() + fc_cl['lower_shape'].tolist()
        return np.array(combined)

    def thickness(self,cst):
        self.update(cst)
        funcs = self.solver.thickness
        return funcs #200 by 1

    def thickness_grad(self,cst):
        self.update(cst)
        grads = self.solver.thickness_sens
        return grads #200 by 8

    def radius(self,cst):
        self.update(cst)
        funcs = self.solver.le
        return funcs #2 by 1

    def radius_grad(self,cst):
        self.update(cst)
        grads = self.solver.le_sens
        return grads #2 by 8 

    #This function adds constraints to the optimizer
    def add_con(self, func, jac, con_type): #need to pass a lambda function or some function with inputs
        constraint = {'type': con_type,
             'fun': func,
             'jac': jac}
        self.constraints.append(constraint)
    
    #Implements thickness and geometric constraints
    def add_geom_con(self):
        for it in range(200): #Thickness. The thickness command returns a 200 by 1.
            def func(cst,index=it):
                return self.thickness(cst)[index]
            def jac(cst,index=it):
                return self.thickness_grad(cst)[index,:]
            self.add_con(func,jac,'ineq')
        for it in range(2): #LE constraints
            def func(cst,index=it):
                return self.radius(cst)[index]
            def jac(cst,index=it):
                return self.radius_grad(cst)[index,:]
            self.add_con(func,jac,'ineq')

    def results_df(self,cst,dfo = False, penalty = None, penaltygrad = None):
        cl = self.cl(cst)
        cd = self.cd(cst)

        if dfo:
            if penalty == None:
                dict_results = {"Cl": cl,"Cd": cd}
            else:
                dict_results = {"Cl": cl, "Cd": cd, "Penalty": penalty}
        else:
            if penalty == None:
                gradnorm = np.linalg.norm(self.cd_grad(cst),np.inf)
                dict_results = {"Cl": cl, "Cd": cd, "gradnorm":gradnorm}
            else:
                gradnorm = np.linalg.norm(penaltygrad, np.inf)
                dict_results = {"Cl": cl, "Cd": cd, "Penalty": penalty,"gradnorm":gradnorm}
        df = pd.DataFrame(dict_results,index=[self.iters])
        return df

    def slsqp(self, bounds:Bounds): #bounds must be Bounds object
        if os.path.exists("Results/SLSQP"):
            shutil.rmtree("Results/SLSQP")
        os.mkdir("Results/SLSQP")
        
        df = self.results_df(self.solver.getValuesNp())
        df.to_csv("Results/SLSQP/outdata.csv",mode='w')
        
        # Callback function to update iterations and results files
        def callback(cst):
            print(f"Cl constraint: {self.constraints[len(self.constraints)-1]['fun'](cst)}")
            self.iters = self.iters + 1      
            df = self.results_df(cst)
            df.to_csv("Results/SLSQP/outdata.csv",header=False,mode='a')
            shutil.copyfile("updated_airfoil.dat",f"Results/SLSQP/airfoil_iter{self.iters}")

        # Optimziation of airfoil using scipy's SLSQP
        res = minimize(self.cd, self.solver.getValuesNp(), method = "SLSQP", jac=self.cd_grad,
               constraints=self.constraints, options={'ftol':1e-6},
               bounds=bounds,callback = callback)

    def penalty(self, dfo=False, eta=0.5, rho=2., tau=1, mu=0.001, tau_min=1e-4, mu_max=10, max_iter=30):    
        # Get the starting CST coefficients from the solver 
        cst0 = self.solver.getValuesNp()
        
        # Define the penalty function for all constraints
        def Penalty(cst, mu):
            l = self.cd(cst)
            for index, con in enumerate(self.constraints):
                if con['type'] == "eq":
                    l = l + 1/2 * mu * con['fun'](cst)**2
                else:
                    l = l + 1/2 * mu * max(0, -con['fun'](cst))**2
            return l
        
        if dfo == False:
            if os.path.exists("Results/PENALTY_GRAD"):
                shutil.rmtree("Results/PENALTY_GRAD")
            os.mkdir("Results/PENALTY_GRAD")
           
            # Create the gradient of the quadratic penalty function
            def gradPenalty(cst,mu):
                lprime = self.cd_grad(cst)
                
                for index, con in enumerate(self.constraints):
                    if con['type'] == "eq":
                        lprime = lprime + mu * con['jac'](cst)*con['fun'](cst)
                    else:
                        lprime = lprime - max(0, -con['fun'](cst)) * con['jac'](cst)
                
                return lprime

            df = self.results_df(cst0,penalty=Penalty(cst0,mu),penaltygrad=gradPenalty(cst0,mu))
            df.to_csv("Results/PENALTY_GRAD/outdata.csv",mode='w')
            self.sub_iters = 0
            def callback(cst):
                self.sub_iters += 1
                print(self.sub_iters)
                shutil.copyfile("updated_airfoil.dat",f"Results/PENALTY_GRAD/airfoil_sub{self.iters}iter{self.sub_iters}")
            # Optimization problem
            while tau >= tau_min and mu <= mu_max:
                self.iters = self.iters + 1
                self.sub_iters = 0
                print(self.iters)
                # Optimization subproblem using BFGS
                res = minimize(lambda cst: Penalty(cst, mu), cst0, method="BFGS", 
                               jac=lambda cst: gradPenalty(cst, mu),
                               options={"maxiter":max_iter, "gtol":tau},callback=callback)
                
                cst0 = res.x
                
                # Update tau and mu based on additional parameters


                if res.nit < 5:
                    tau = tau * eta / 5
                    mu = mu * rho * 5
                else:
                    tau = tau * eta
                    mu = mu * rho   
                cd_val = res.fun
                # Update iteration
                
                print(f"Cl constraint: {self.constraints[len(self.constraints)-1]['fun'](cst0)}")
                
                # Update output files
                df = self.results_df(cst0,penalty=Penalty(cst0,mu),penaltygrad=gradPenalty(cst0,mu))
                df.to_csv("Results/PENALTY_GRAD/outdata.csv",header=False,mode='a')
        elif dfo == True:
            if os.path.exists("Results/PENALTY_DFO"):
                shutil.rmtree("Results/PENALTY_DFO")
            os.mkdir("Results/PENALTY_DFO")
            
            df = self.results_df(cst0,dfo=True,penalty=Penalty(cst0,mu))
            df.to_csv("Results/PENALTY_DFO/outdata.csv",mode='w')
            
            # Optimization problem
            self.sub_iters = 0
            def callback(cst):
                self.sub_iters += 1
                print(self.sub_iters)
                shutil.copyfile("updated_airfoil.dat",f"Results/PENALTY_DFO/airfoil_sub{self.iters}iter{self.sub_iters}")
            while tau >= tau_min and mu <= mu_max:
                # Subproblem using scipy's Nelder Mead method
                self.iters = self.iters + 1
                self.sub_iters = 0
                res = minimize(lambda cst: Penalty(cst,mu), cst0, method='Nelder-Mead',
                               options={"maxiter":max_iter, "fatol":tau},callback=callback)

                # new value for CST coefficients
                cst0 = res.x
                
                # Update values for tau, mu based on additioanl parameters
                if res.nit < 5:
                    tau = tau * eta / 5
                    mu = mu * rho * 5
                else:
                    tau = tau * eta
                    mu = mu * rho
                
                # Update iterations
                cd_val = res.fun                
                print(f"Cl constraint: {self.constraints[len(self.constraints)-1]['fun'](cst0)}")
                
                # Update output file
                df = self.results_df(cst0,dfo=True,penalty=Penalty(cst0,mu))
                df.to_csv("Results/PENALTY_DFO/outdata.csv",header=False,mode='a')
            
        print(f"Cd:{self.cd(cst0)}")