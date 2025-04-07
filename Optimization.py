from scipy.optimize import minimize
import numpy as np




class Optimization:
    def __init__(self,solver):
        self.solver = solver
        self.constraints = []
        self.log = []
    def cl(self,cst):
        self.solver.updateCSTCoeff(cst)
        funcs = self.solver.solvePrimal()
        cl = funcs["fc_cl"]
        return cl

    def cd(self,cst):
        self.solver.updateCSTCoeff(cst)
        funcs = self.solver.solvePrimal()
        cd = funcs["fc_cd"]
        return cd

    def cd_grad(self,cst):
        self.solver.updateCSTCoeff(cst)
        grads = self.solver.findFunctionSens()
        fc_cd = grads['fc_cd']
        combined = fc_cd['upper_shape'].tolist() + fc_cd['lower_shape'].tolist()
        return np.array(combined)

    def cl_grad(self,cst):
        self.solver.updateCSTCoeff(cst)
        grads = self.solver.findFunctionSens()
        fc_cl = grads['fc_cl']
        combined = fc_cl['upper_shape'].tolist() + fc_cl['lower_shape'].tolist()
        return np.array(combined)
    def add_con(self, func, jac, con_type): #need to pass a lambda function or some function with inputs
        constraint = {'type': con_type,
             'fun': func,
             'jac': jac}
        self.constraints.append(constraint)
    def slsqp(self, bounds): #bounds must be Bounds object
        def callback(cst):
            self.log.append(cst)
            for con in self.constraints:
                print(cst)
                print(f"Constraint: {con['fun'](cst)}. Func:{self.cd(cst)}")
        res = minimize(self.cd, self.solver.getValuesNp(), method = "SLSQP", jac=self.cd_grad,
               constraints=self.constraints, tol=1e-6,
               bounds=bounds,callback = callback)

        
        

        