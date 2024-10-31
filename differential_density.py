import numpy as np
from scipy.optimize import fsolve
from utilities import CalculationContext, Constants


class Integrand:
    def __init__(self, cc: CalculationContext):
        self.constantTerm = 2 / (cc.transverseOverlapArea * cc.beta * cc.t21**2)

        self.detdy0old = 0.456 * 2 * cc.massNum * np.log(cc.comCollisionEnergy / 2.35)

        self.detdy0  = 1.25 * 0.308 * 2 * cc.massNum * np.log(cc.comCollisionEnergy / (2 * Constants.nucleonMass))**1.08 if cc.comCollisionEnergy <= 20.7 else self.detdy0old

        self.sigmab = 0.601 * (cc.comCollisionEnergy - 2 * Constants.nucleonMass)**0.121 * np.log(cc.comCollisionEnergy / (2 * Constants.nucleonMass))**0.241

        self.yb =  0.541 * (cc.comCollisionEnergy - 2 * Constants.nucleonMass)**0.196 * np.log(cc.comCollisionEnergy / (2 * Constants.nucleonMass))**0.392

        func = lambda x: cc.massNum * cc.comCollisionEnergy - self.detdy0 * np.exp(x**2 / 2) * np.sqrt(2 * np.pi) * x - Constants.nucleonMass * 2 * cc.massNum * np.exp(self.sigmab**2 / 2) * np.cosh(self.yb)

        self.sigmahad = fsolve(func, 0)[0]

        self.dnbdy0 = 2 * cc.massNum / (np.sqrt(2 * np.pi) * self.sigmab) * np.exp(-(self.yb / self.sigmab)**2 / 2)
    

    def get_data(self) -> list:
        return [self.constantTerm, self.detdy0old, self.detdy0, self.sigmab, self.yb, self.sigmahad, self.dnbdy0]


    def y0(self, z0, x, t):
        return np.arctanh(-z0 / (t - x))
    

    def detdy(self, y):
        return self.detdy0 * np.exp(-(y / self.sigmahad)**2 / 2)
    

    def dnbdy(self, y):
        return self.dnbdy0 * 0.5 * np.exp((self.yb / self.sigmab)**2 / 2) * (np.exp(-((y + self.yb) / self.sigmab)**2 / 2) + np.exp(-((y - self.yb) / self.sigmab)**2 / 2))
    

    def dmtdyhad(self, y):
        return self.detdy(y) + Constants.nucleonMass * self.dnbdy(y)
    
    def integrand(self, z0, x, t):
        pass

    def integrand1Var(self, y):
        pass


class EnergyDensity(Integrand):
    def __init__(self, cc: CalculationContext):
        super().__init__(cc)


    def integrand(self, z0, x, t):
        rapidity = super().y0(z0, x, t)
        return self.constantTerm / (t - x) * super().dmtdyhad(rapidity) * np.cosh(rapidity)**3
    
    def integrand1Var(self, y):
        # This is divided by a cosh(y) so we don't have to later
        return self.constantTerm * super().dmtdyhad(y) * np.cosh(y)**2


class NetBaryonDensity(Integrand):
    def __init__(self, cc: CalculationContext):
        super().__init__(cc)


    def integrand(self, z0, x, t):
        rapidity = super().y0(z0, x, t)
        return self.constantTerm / (t - x) * super().dnbdy(rapidity) * np.cosh(rapidity)**2
    

    def integrand1Var(self, y):
        # This is divided by a cosh(y) so we don't have to later
        return self.constantTerm * super().dnbdy(y) * np.cosh(y)
    

if __name__ == "__main__":
    cc = CalculationContext(79, 197, 200.0, 0.1, "quantum", 10)
    # integrand = Integrand(cc)

    # print(integrand.get_data())

    ed = EnergyDensity(cc)
    print(ed.integrand(0.0, 0.0, 0.2))

    nd = NetBaryonDensity(cc)
    print(nd.integrand(0.0, 0.0, 0.2))