from calc_profile import CalcProfile
from utilities import Constants
from densities.piecewise import PiecewiseSolution
import numpy as np


class DifferentialDensity:
    def __init__(self, cp: CalcProfile):
        self.cp = cp
        self.constantTerm = 2 / (cp.at * cp.beta * cp.tprod**2)

    def y0(self, z0, x, t):
        return np.arctanh(-z0 / (t - x))

    def detdy(self, y):
        return self.cp.detdy0 * np.exp(-(y / self.cp.sigmahadron)**2 / 2)

    def dnbdy(self, y):
        return self.cp.dnbdy0 * 0.5 * np.exp((self.cp.ybaryon / self.cp.sigmabaryon)**2 / 2) * (np.exp(-((y + self.cp.ybaryon) / self.cp.sigmabaryon)**2 / 2) + np.exp(-((y - self.cp.ybaryon) / self.cp.sigmabaryon)**2 / 2))

    def dmtdyhad(self, y):
        return self.detdy(y) + Constants.nucleonMass * self.dnbdy(y)
    
    def integrand(self, z0, x, t):
        pass

    def integrand1Var(self, y):
        pass


class EnergyDensity(DifferentialDensity):
    def __init__(self, cp: CalcProfile):
        super().__init__(cp)
        self.ps = PiecewiseSolution(cp, self.__integrand)
        self.densities = self.ps.calculate()

    def __integrand(self, z0, x, t):
        rapidity = super().y0(z0, x, t)
        return self.constantTerm / (t - x) * super().dmtdyhad(rapidity) * np.cosh(rapidity)**3
    
    # def __integrand1Var(self, y):
    #     # This is divided by a cosh(y) so we don't have to later
    #     return self.constantTerm * super().dmtdyhad(y) * np.cosh(y)**2


class NetBaryonDensity(DifferentialDensity):
    def __init__(self, cp: CalcProfile):
        super().__init__(cp)
        self.ps = PiecewiseSolution(cp, self.__integrand)
        self.densities = self.ps.calculate()

    def __integrand(self, z0, x, t):
        rapidity = super().y0(z0, x, t)
        return self.constantTerm / (t - x) * super().dnbdy(rapidity) * np.cosh(rapidity)**2

    # def __integrand1Var(self, y):
    #     # This is divided by a cosh(y) so we don't have to later
    #     return self.constantTerm * super().dnbdy(y) * np.cosh(y)
    

if __name__ == "__main__":
    cc = CalculationContext(79, 197, 200.0, 0.1, "quantum", 10)
    # integrand = Integrand(cc)

    # print(integrand.get_data())

    ed = EnergyDensity(cc)
    print(ed.integrand(0.0, 0.0, 0.2))

    nd = NetBaryonDensity(cc)
    print(nd.integrand(0.0, 0.0, 0.2))