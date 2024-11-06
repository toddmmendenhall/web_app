import numpy as np
from scipy.optimize import fsolve
from utilities import Constants
from calc_profile import CalcProfile


class NonInteractingMasslessQuantumEOSFullSolution:
    def __init__(self, cp: CalcProfile, es: np.ndarray, nBs: np.ndarray) -> None:
        self.es = es
        self.nBs = nBs

        # Define nQ_dens and nS_dens arrays
        self.nQs = cp.z / cp.a * nBs
        self.nSs = np.zeros(es.size)
        
        self.thermoVars = np.zeros((es.size, 3))

        self.calculate()

    # Calculate T, muB, muS, and muQ arrays
    def __equations(self, x, ep, nB, nQ):
        temp, muB, muS = x

        eq1 = 19 * Constants.piSquared / 12 * temp**4 + 3 / 2 * ((muB - 2 * muS)**2 + muS**2) * temp**2 + 3 / (4 * Constants.piSquared) * ((muB - 2 * muS)**4 + muS**4) - ep * Constants.hbarTimesC**3
        eq2 = 1 / 3 * (muB - muS) * temp**2 + 1 / (3 * Constants.piSquared) * ((muB - 2 * muS)**3 + muS**3) - nB * Constants.hbarTimesC**3
        eq3 = 1 / 3 * (2 * muB - 5 * muS) * temp**2 + 1 / (3 * Constants.piSquared) * (2 * (muB - 2 * muS)**3 - muS**3) - nQ * Constants.hbarTimesC**3

        return [eq1, eq2, eq3]

    def calculate(self):
        for i in range(self.es.size):
            args = (self.es[i], self.nBs[i], self.nSs[i])
            self.thermoVars[i] = fsolve(self.__equations, x0 = np.array([0.3, 0.9, 0.3]), args = args, maxfev = 1000)

        self.thermoVars = self.thermoVars * Constants.MeVPerGeV
        self.thermoVars = self.thermoVars.transpose()

        muQ = self.thermoVars[1] - 3 * self.thermoVars[2]

        self.thermoVars = np.insert(self.thermoVars, 2, muQ, axis=0)


class NonInteractingMasslessBoltzmannEOSFullSolution:
    def __init__(self, cp: CalcProfile, es: np.ndarray, nBs: np.ndarray) -> None:
        self.es = es
        self.nBs = nBs

        # Define nQ_dens and nS_dens arrays
        self.nQs = cp.z / cp.a * nBs
        self.nSs = np.zeros(es.size)
        
        self.thermoVars = np.zeros((es.size, 3))

        self.calculate()

    # Calculate T, muB, muS, and muQ arrays
    def __equations(self, x, ep, nB, nQ):
        temp, muB, muS = x

        eq1 = 12 / Constants.piSquared * temp**4 * (7 + 3 * np.cosh((muB - 2 * muS) / temp) + 3 * np.cosh(muS / temp)) - ep * Constants.hbarTimesC**3
        eq2 = 4 / Constants.piSquared * temp**3 * (np.sinh((muB - 2 * muS) / temp) + np.sinh(muS / temp)) - nB * Constants.hbarTimesC**3
        eq3 = 4 / Constants.piSquared * temp**3 * (2 * np.sinh((muB - 2 * muS) / temp) - np.sinh(muS / temp)) - nQ * Constants.hbarTimesC**3

        return [eq1, eq2, eq3]

    def calculate(self):
        for i in range(self.es.size):
            args = (self.es[i], self.nBs[i], self.nSs[i])
            self.thermoVars[i] = fsolve(self.__equations, x0 = np.array([0.3, 0.9, 0.3]), args = args, maxfev = 1000)

        self.thermoVars = self.thermoVars * Constants.MeVPerGeV
        self.thermoVars = self.thermoVars.transpose()

        muQ = self.thermoVars[1] - 3 * self.thermoVars[2]

        self.thermoVars = np.insert(self.thermoVars, 2, muQ, axis=0)


if __name__ == "__main__":
    from utilities import CalculationContext
    from densities.differential_density import EnergyDensity, NetBaryonDensity
    from densities.piecewise import PiecewiseSolution

    cc = CalculationContext(79, 197, 200, 0.3, "boltzmann", 10)

    ed = EnergyDensity(cc)
    nd = NetBaryonDensity(cc)

    pse = PiecewiseSolution(cc, ed)
    pse.calculate(cc)
    psn = PiecewiseSolution(cc, nd)
    psn.calculate(cc)

    # eos = NonInteractingMasslessQuantumEOSFullSolution(cc, pse, psn)
    eos = NonInteractingMasslessBoltzmannEOSFullSolution(cc, pse, psn)
    eos.calculate()
    print(eos.thermoVars)