import numpy as np
from scipy.integrate import dblquad


class PiecewiseSolution:
    from utilities import CalculationContext
    from differential_density import Integrand

    def __init__(self, cc: CalculationContext, density: Integrand) -> None:
        self.cc = cc

        self.conditions = [self.cc.times <= (self.cc.t1 + self.cc.partonFormationTime),
                           (self.cc.times > (self.cc.t1 + self.cc.partonFormationTime)) * (self.cc.times < self.cc.ta),
                           (self.cc.times >= self.cc.ta) * (self.cc.times < (self.cc.t2 + self.cc.partonFormationTime)),
                           self.cc.times >= (self.cc.t2 + self.cc.partonFormationTime)]
        
        self.functions = [self.__zerothPiece, self.__firstPiece, self.__secondPiece, self.__thirdPiece]

        self.integrand = density.integrand
        self.x1 = cc.x1
        self.x2 = cc.x2

    
    def __zerothPiece(self, times: np.ndarray) -> np.ndarray:
        return np.zeros(times.size)
    
    
    def __firstPiece(self, times: np.ndarray) -> np.ndarray:
        vals = np.zeros(times.size)

        lowerLeftZBound = lambda x: -self.cc.beta * (x - self.cc.t1)
        lowerRightZBound = lambda x: self.cc.beta * (x - self.cc.t1)

        for i, t in enumerate(times):
            upperLeftZBound = lambda x: -np.sqrt((t - x)**2 - self.cc.partonFormationTime**2)
            upperRightZBound = lambda x: np.sqrt((t - x)**2 - self.cc.partonFormationTime**2)

            upperXBound = self.x1(t)

            lowerIntegral, lowerError = dblquad(self.integrand, self.cc.t1, upperXBound, lowerLeftZBound, lowerRightZBound, args = (t,))
            upperIntegral, upperError = dblquad(self.integrand, upperXBound, t - self.cc.partonFormationTime, upperLeftZBound, upperRightZBound, args=(t,))

            vals[i] = lowerIntegral + upperIntegral

        return vals

    
    def __secondPiece(self, times: np.ndarray) -> np.ndarray:
        vals = np.zeros(times.size)

        lowerLeftZBound = lambda x: -self.cc.beta * (x - self.cc.t1)
        lowerRightZBound = lambda x: self.cc.beta * (x - self.cc.t1)
        middleLeftZBound = lambda x: -self.cc.beta * (self.cc.t2 - x)
        middleRightZBound = lambda x: self.cc.beta * (self.cc.t2 - x)

        for i, t in enumerate(times):
            upperLeftZBound = lambda x: -np.sqrt((t - x)**2 - cc.partonFormationTime**2)
            upperRightZBound = lambda x: np.sqrt((t - x)**2 - cc.partonFormationTime**2)

            upperXBound = self.x2(t)

            lowerIntegral, lowerError = dblquad(self.integrand, self.cc.t1, self.cc.tMid, lowerLeftZBound, lowerRightZBound, args = (t,))
            middleIntegral, middleError = dblquad(self.integrand, self.cc.tMid, upperXBound, middleLeftZBound, middleRightZBound, args=(t,))
            upperIntegral, upperError = dblquad(self.integrand, upperXBound, t - self.cc.partonFormationTime, upperLeftZBound, upperRightZBound, args=(t,))

            vals[i] = lowerIntegral + middleIntegral + upperIntegral
        
        return vals
    
    
    def __thirdPiece(self, times: np.ndarray):
        vals = np.zeros(times.size)

        lowerLeftZBound = lambda x: -self.cc.beta * (x - self.cc.t1)
        lowerRightZBound = lambda x: self.cc.beta * (x - self.cc.t1)
        upperLeftZBound = lambda x: -self.cc.beta * (self.cc.t2 - x)
        upperRightZBound = lambda x: self.cc.beta * (self.cc.t2 - x)

        for i, t in enumerate(times):
            lowerIntegral, lowerError = dblquad(self.integrand, self.cc.t1, self.cc.tMid, lowerLeftZBound, lowerRightZBound, args = (t,))
            upperIntegral, upperError = dblquad(self.integrand, self.cc.tMid, self.cc.t2, upperLeftZBound, upperRightZBound, args = (t,))

            vals[i] = lowerIntegral + upperIntegral

        return vals
    
    
    def calculate(self) -> None:
        self.densities = np.piecewise(self.cc.times, self.conditions, self.functions)


class PiecewiseSolutionSingleIntegral:
    from utilities import CalculationContext
    from differential_density import Integrand

    def __init__(self, cc: CalculationContext, density: Integrand) -> None:
        from scipy.integrate import quad
        from scipy.optimize import root_scalar
        self.integrator = quad
        self.rootFinder = root_scalar
        self.cc = cc
        self.__setConditions()
        self.functions = [self.__zerothPiece, self.__firstPiece, self.__secondPiece, self.__thirdPiece]
        self.integrand = density.integrand1Var
    

    def __setConditions(self) -> None:
        c0 = self.cc.times <= (self.cc.t1 + self.cc.partonFormationTime)
        c1 = (self.cc.times > (self.cc.t1 + self.cc.partonFormationTime)) * (self.cc.times < self.cc.ta)
        c2 = (self.cc.times >= self.cc.ta) * (self.cc.times < (self.cc.t2 + self.cc.partonFormationTime))
        c3 = self.cc.times >= (self.cc.t2 + self.cc.partonFormationTime)
        
        self.conditions = [c0, c1, c2, c3]


    def __rMaxLower(self, y: float, t: float) -> float:
        return self.cc.beta * (t - self.cc.t1) / (self.cc.beta * np.cosh(y) + np.sinh(y))

    
    def __rMinUpper(self, y: float, t: float) -> float:
        return self.cc.beta * (t - self.cc.t2) / (self.cc.beta * np.cosh(y) - np.sinh(y))
    

    def __yMaxEarly(self, t: float) -> float:
        func = lambda y, t: self.cc.beta * (t - self.cc.t1) - self.cc.partonFormationTime * (self.cc.beta * np.cosh(y) + np.sinh(y))
        return self.rootFinder(func, t, x0=0).root
    
    
    def __yMidMiddle(self, t: float) -> float:
        func = lambda y, t: self.cc.beta * (t - self.cc.t2) - self.cc.partonFormationTime * (self.cc.beta * np.cosh(y) - np.sinh(y))
        return self.rootFinder(func, t, x0=0).root

    
    def __yMaxLate(self, t: float) -> float:
        return np.arctanh(-self.cc.beta * (self.cc.tMid - self.cc.t2) / (t - self.cc.tMid))

    
    def __zerothPiece(self, times: np.ndarray) -> np.ndarray:
        return np.zeros(times.size)
    
    
    def __firstPiece(self, times: np.ndarray) -> np.ndarray:
        vals = np.zeros(times.size)

        integrand = lambda y, t: self.integrand(y) * (self.__rMaxLower(y, t) - self.cc.partonFormationTime)

        for i, t in enumerate(times):
            y1 = self.__yMaxEarly(t)
            integral, error = self.integrator(integrand, 0, y1, t)
            vals[i] = integral

        return 2 * vals

    
    def __secondPiece(self, times: np.ndarray) -> np.ndarray:
        vals = np.zeros(times.size)
        
        lowerIntegrand = lambda y, t: self.integrand(y) * (self.__rMaxLower(y, t) - self.cc.partonFormationTime)
        upperIntegrand = lambda y, t: self.integrand(y) * (self.__rMaxLower(y, t) - self.__rMinUpper(y, t))

        for i, t in enumerate(times):
            y1 = self.__yMidMiddle(t)
            y2 = self.__yMaxLate(t)
            lowerIntegral, lowerError = self.integrator(lowerIntegrand, 0, y1, t)
            upperIntegral, upperError = self.integrator(upperIntegrand, y1, y2, t)
            vals[i] = lowerIntegral + upperIntegral

        return 2 * vals
    
    
    def __thirdPiece(self, times: np.ndarray) -> np.ndarray:
        vals = np.zeros(times.size)

        integrand = lambda y, t: self.integrand(y) * (self.__rMaxLower(y, t) - self.__rMinUpper(y, t))

        for i, t in enumerate(times):
            y1 = self.__yMaxLate(t)
            integral, error = self.integrator(integrand, 0, y1, t)
            vals[i] = integral

        return 2 * vals
    
    
    def calculate(self) -> None:
        self.densities = np.piecewise(self.cc.times, self.conditions, self.functions)


if __name__ == "__main__":
    from utilities import CalculationContext
    from differential_density import Integrand, EnergyDensity, NetBaryonDensity


    cc = CalculationContext(79, 197, 2, 0.1, "boltzmann", 50)
    e = EnergyDensity(cc)

    # ps1 = PiecewiseSolutionSingleIntegral(cc, e)
    # ps1.calculate()
    
    ps2 = PiecewiseSolution(cc, e)
    ps2.calculate()
    
    # percentDifference = (ps1.densities - ps2.densities) / ps2.densities
    # print(percentDifference)

    import matplotlib.pyplot as plt
    plt.plot(cc.times, ps2.densities)
    # plt.plot(ps1.times[2:], percentDifference[2:])
    plt.xscale('log')
    plt.show()
