from scipy.optimize import minimize
import numpy as np

class Optimization:
    def __init__(self,solver):
        self.solver = solver
        self.constraints = []
        self.log = []
        self.add_geom_con()
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
    def slsqp(self, bounds): #bounds must be Bounds object
        def callback(cst):
            self.log.append(cst)
            print(f"Cl: {self.constraints[len(self.constraints)-1]['fun'](cst)}")
            #for con in self.constraints:
             #   print(f"Constraint: {con['fun'](cst)}")   
            #thicknesses = self.thickness(cst) 
            #for x in range(200):
            #    print(thicknesses[x])
                
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