from scipy.optimize import minimize
import numpy as np

class Optimization:
    def __init__(self,solver):
        self.solver = solver
        self.constraints = []
        self.log = []
        self.add_geom_con()
    def cl(self,cst):
        if np.all(self.solver.getValuesNp() == cst):
            funcs = self.solver.funcs
        else:
            self.solver.updateCSTCoeff(cst)
            funcs = self.solver.solvePrimal()
        cl = funcs["fc_cl"]
        return cl

    def cd(self,cst):
        if np.all(self.solver.getValuesNp() == cst):
            funcs = self.solver.funcs
        else:
            self.solver.updateCSTCoeff(cst)
            funcs = self.solver.solvePrimal()
        cd = funcs["fc_cd"]
        return cd

    def cd_grad(self,cst):
        if np.all(self.solver.getValuesNp() == cst):
            grads = self.solver.func_sens
        else:
            self.solver.updateCSTCoeff(cst)
            grads = self.solver.findFunctionSens()
        fc_cd = grads['fc_cd']
        combined = fc_cd['upper_shape'].tolist() + fc_cd['lower_shape'].tolist()
        return np.array(combined)

    def cl_grad(self,cst):
        if np.all(self.solver.getValuesNp() == cst):
            grads = self.solver.func_sens
        else:
            self.solver.updateCSTCoeff(cst)
            grads = self.solver.findFunctionSens()
        fc_cl = grads['fc_cl']
        combined = fc_cl['upper_shape'].tolist() + fc_cl['lower_shape'].tolist()
        return np.array(combined)

    def thickness(self,cst):
        if np.all(self.solver.getValuesNp() == cst):
            funcs = self.solver.thickness
        else:
            self.solver.updateCSTCoeff(cst)
            funcs = self.solver.findConstraint()[0]
        return funcs #200 by 1

    def thickness_grad(self,cst):
        if np.all(self.solver.getValuesNp() == cst):
            grads = self.solver.thickness_sens
        else:
            self.solver.updateCSTCoeff(cst)
            grads = self.solver.findConSens()[0]
        return grads #200 by 8

    def radius(self,cst):
        if np.all(self.solver.getValuesNp() == cst):
            funcs = self.solver.le
        else:
            self.solver.updateCSTCoeff(cst)
            funcs = self.solver.findConstraint()[1]
        return funcs #2 by 1

    def radius_grad(self,cst):
        if np.all(self.solver.getValuesNp() == cst):
            grads = self.solver.le_sens
        else:
            self.solver.updateCSTCoeff(cst)
            grads = self.solver.findConSens()[1]
        return grads #2 by 8 
    def add_con(self, func, jac, con_type): #need to pass a lambda function or some function with inputs
        constraint = {'type': con_type,
             'fun': func,
             'jac': jac}
        self.constraints.append(constraint)
    def add_geom_con(self):
        for iter in range(200):
            func = lambda cst: self.thickness(cst)[iter]
            jac = lambda cst: self.thickness_grad(cst)[iter,:]
            self.add_con(func,jac,'ineq')
        for iter in range(2):
            func = lambda cst: self.radius(cst)[iter]
            jac = lambda cst: self.radius_grad(cst)[iter,:]
            self.add_con(func,jac,'ineq')
    def slsqp(self, bounds): #bounds must be Bounds object
        def callback(cst):
            self.log.append(cst)
            #for con in self.constraints:
            #    print(cst)
            #    print(f"Constraint: {con['fun'](cst)}. Func:{self.cd(cst)}")
                
        res = minimize(self.cd, self.solver.getValuesNp(), method = "SLSQP", jac=self.cd_grad,
               constraints=self.constraints, tol=1e-6,
               bounds=bounds,callback = callback)
    def trustcr(self,bounds):
        def callback(cst, state):
            self.log.append(cst)
            #for con in self.constraints:
            #    print(cst)
            #    print(f"Constraint: {con['fun'](cst)}. Func:{self.cd(cst)}")
            
        res = minimize(self.cd, self.solver.getValuesNp(), method = "trust-constr", jac=self.cd_grad,
            constraints=self.constraints, tol=1e-6,
            bounds=bounds,callback = callback)
    def cobyqa(self,bounds):
        def callback(cst):
            pass #add stuff here
        res = minimize(self.cd, self.solver.getValuesNp(), method = "COBYQA",
            constraints=self.constraints, tol=1e-6,
            bounds=bounds,callback = callback)