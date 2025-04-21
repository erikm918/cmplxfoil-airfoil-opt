from scipy.optimize import minimize
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
        self.iters = 0
        if not os.path.exists("Results"):
            os.mkdir("Results")
            os.mkdir("Results/SLSQP")
            os.mkdir("Results/TRUSTCR")
            os.mkdir("Results/AUGLAG")
            os.mkdir("Results/SINGLEPOINT")
    def update(self,cst):
        if np.all(self.solver.getValuesNp() == cst):
            pass
        else:
            self.solver.updateCSTCoeff(cst)
            self.solver.solvePrimal()
            self.solver.findFunctionSens()
            self.solver.findConstraint()
            self.solver.findConSens()

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

    def add_con(self, func, jac, con_type): #need to pass a lambda function or some function with inputs
        constraint = {'type': con_type,
             'fun': func,
             'jac': jac}
        self.constraints.append(constraint)
    
    def add_geom_con(self):
        for it in range(200):
            def func(cst,index=it):
                return self.thickness(cst)[index]
            def jac(cst,index=it):
                return self.thickness_grad(cst)[index,:]
            self.add_con(func,jac,'ineq')
        for it in range(2):
            def func(cst,index=it):
                return self.radius(cst)[index]
            def jac(cst,index=it):
                return self.radius_grad(cst)[index,:]
            self.add_con(func,jac,'ineq')

    def results_df(self,cst):
        cl = self.cl(cst)
        cd = self.cd(cst)
        dict_results = {"Cl": cl,"Cd": cd}
        df = pd.DataFrame(dict_results,index=[self.iters])
        return df

    def slsqp(self, bounds): #bounds must be Bounds object
        df = self.results_df(self.solver.getValuesNp())
        df.to_csv("./Results/SLSQP/outdata.csv",mode='w')
        def callback(cst):
            print(f"Cl constraint: {self.constraints[len(self.constraints)-1]['fun'](cst)}")
            self.iters = self.iters + 1      
            df = self.results_df(cst)
            df.to_csv("./Results/SLSQP/outdata.csv",header=False,mode='a')
            shutil.copyfile("updated_airfoil.dat",f"./Results/SLSQP/airfoil_iter{self.iters}")

        res = minimize(self.cd, self.solver.getValuesNp(), method = "SLSQP", jac=self.cd_grad,
               constraints=self.constraints, tol=1e-6,
               bounds=bounds,callback = callback)
    
    def trustcr(self,bounds):
        def callback(cst, state):
            self.log.append(cst)
            print(f"Cl: {self.constraints[len(self.constraints)-1]['fun'](cst)}")
            #for con in self.constraints:
                #print(cst)
                #print(f"Constraint: {con['fun'](cst)}")    
        res = minimize(self.cd, self.solver.getValuesNp(), method = "trust-constr", jac=self.cd_grad,
            constraints=self.constraints, tol=1e-6,
            bounds=bounds,callback = callback)
    
    def cobyqa(self,bounds):
        def callback(cst):
            print(f"Cl: {self.constraints[len(self.constraints)-1]['fun'](cst)}")
        res = minimize(self.cd, self.solver.getValuesNp(), method = "COBYQA",
            constraints=self.constraints, tol=1e-6,
            bounds=bounds,callback = callback)
    
    def aug_lagrange(self, bounds, max_iter=250):
        muk = 0.01
        eta = 0.5
        rho = 2.
        tauk = 1.
        
        cst0 = self.solver.getValuesNp()
        k = 0
        
        def Lagrange(x, lambdak, mu):
            l = self.cd
            
            for i, c in enumerate(self.constraints):
                l = l - lambdak[i] * c['fun'] + 0.5 * mu * max(0, -c['fun'])**2
            
            return l
        
        while tauk > 1e-4 or muk <= 10:
            pass