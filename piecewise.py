import numpy as np
from scipy.integrate import dblquad


class PiecewiseSolution:
    from utilities import CalculationContext
    from differential_density import Integrand

    def __init__(self, cc: CalculationContext, density: Integrand) -> None:
        tMin = cc.t1 + cc.partonFormationTime
        tMax = 3 * cc.t2 + cc.partonFormationTime

        if(tMax < 10):
            tMax = 10

        self.times = np.logspace(np.log10(tMin), np.log10(tMax), cc.numTimes)
        self.times = np.insert(self.times, 0, 0)

        self.conditions = [self.times <= (cc.t1 + cc.partonFormationTime),
                           (self.times > (cc.t1 + cc.partonFormationTime)) * (self.times < cc.ta),
                           (self.times >= cc.ta) * (self.times < (cc.t2 + cc.partonFormationTime)),
                           self.times >= (cc.t2 + cc.partonFormationTime)]
        
        self.functions = [self.__zerothPiece, self.__firstPiece, self.__secondPiece, self.__thirdPiece]

        self.integrand = density.integrand
        self.x1 = cc.x1
        self.x2 = cc.x2

    
    def __zerothPiece(self, times: np.ndarray, cc: CalculationContext) -> np.ndarray:
        return np.zeros(times.size)
    
    
    def __firstPiece(self, times: np.ndarray, cc: CalculationContext):
        vals = np.zeros(times.size)

        lowerLeftZBound = lambda x: -cc.beta * (x - cc.t1)
        lowerRightZBound = lambda x: cc.beta * (x - cc.t1)

        for i, t in enumerate(times):
            upperLeftZBound = lambda x: -np.sqrt((t - x)**2 - cc.partonFormationTime**2)
            upperRightZBound = lambda x: np.sqrt((t - x)**2 - cc.partonFormationTime**2)

            upperXBound = self.x1(t)

            lowerIntegral, lowerError = dblquad(self.integrand, cc.t1, upperXBound, lowerLeftZBound, lowerRightZBound, args = (t,))
            upperIntegral, upperError = dblquad(self.integrand, upperXBound, t - cc.partonFormationTime, upperLeftZBound, upperRightZBound, args=(t,))

            vals[i] = lowerIntegral + upperIntegral

        return vals

    
    def __secondPiece(self, times: np.ndarray, cc: CalculationContext):
        vals = np.zeros(times.size)

        lowerLeftZBound = lambda x: -cc.beta * (x - cc.t1)
        lowerRightZBound = lambda x: cc.beta * (x - cc.t1)
        middleLeftZBound = lambda x: -cc.beta * (cc.t2 - x)
        middleRightZBound = lambda x: cc.beta * (cc.t2 - x)

        for i, t in enumerate(times):
            upperLeftZBound = lambda x: -np.sqrt((t - x)**2 - cc.partonFormationTime**2)
            upperRightZBound = lambda x: np.sqrt((t - x)**2 - cc.partonFormationTime**2)

            upperXBound = self.x2(t)

            lowerIntegral, lowerError = dblquad(self.integrand, cc.t1, cc.tMid, lowerLeftZBound, lowerRightZBound, args = (t,))
            middleIntegral, middleError = dblquad(self.integrand, cc.tMid, upperXBound, middleLeftZBound, middleRightZBound, args=(t,))
            upperIntegral, upperError = dblquad(self.integrand, upperXBound, t - cc.partonFormationTime, upperLeftZBound, upperRightZBound, args=(t,))

            vals[i] = lowerIntegral + middleIntegral + upperIntegral
        
        return vals
    
    
    def __thirdPiece(self, times: np.ndarray, cc: CalculationContext):
        vals = np.zeros(times.size)

        lowerLeftZBound = lambda x: -cc.beta * (x - cc.t1)
        lowerRightZBound = lambda x: cc.beta * (x - cc.t1)
        upperLeftZBound = lambda x: -cc.beta * (cc.t2 - x)
        upperRightZBound = lambda x: cc.beta * (cc.t2 - x)

        for i, t in enumerate(times):
            lowerIntegral, lowerError = dblquad(self.integrand, cc.t1, cc.tMid, lowerLeftZBound, lowerRightZBound, args = (t,))
            upperIntegral, upperError = dblquad(self.integrand, cc.tMid, cc.t2, upperLeftZBound, upperRightZBound, args = (t,))

            vals[i] = lowerIntegral + upperIntegral

        return vals
    
    
    def calculate(self, cc: CalculationContext) -> np.ndarray:
        self.densities = np.piecewise(self.times, self.conditions, self.functions, cc)


class PiecewiseSolutionSingleIntegral:
    from utilities import CalculationContext
    from differential_density import Integrand

    def __init__(self, cc: CalculationContext, density: Integrand) -> None:
        from scipy.integrate import quad
        self.integrator = quad
        self.cc = cc
        self.__setTimes()
        self.__setConditions()
        self.functions = [self.__zerothPiece, self.__firstPiece, self.__secondPiece, self.__thirdPiece]
        self.integrand = density.integrand1Var


    def __setTimes(self):
        tMin = self.cc.t1 + self.cc.partonFormationTime
        tMax = 3 * self.cc.t2 + self.cc.partonFormationTime

        if(tMax < 10):
            tMax = 10

        self.times = np.logspace(np.log10(tMin), np.log10(tMax), self.cc.numTimes)
        self.times = np.insert(self.times, 0, 0)
    

    def __setConditions(self):
        c0 = self.times <= (self.cc.t1 + self.cc.partonFormationTime)
        c1 = (self.times > (self.cc.t1 + self.cc.partonFormationTime)) * (self.times < self.cc.ta)
        c2 = (self.times >= self.cc.ta) * (self.times < (self.cc.t2 + self.cc.partonFormationTime))
        c3 = self.times >= (self.cc.t2 + self.cc.partonFormationTime)
        
        self.conditions = [c0, c1, c2, c3]


    def __rMaxLower(self, y, t):
        return self.cc.beta * (t - self.cc.t1) / (self.cc.beta * np.cosh(y) + np.sinh(y))

    
    def __rMinUpper(self, y, t):
        return self.cc.beta * (t - self.cc.t2) / (self.cc.beta * np.cosh(y) - np.sinh(y))
    

    def __yMaxEarly(self, t):
        from scipy.optimize import root_scalar

        func = lambda y, t: self.cc.beta * (t - self.cc.t1) / (self.cc.beta * np.cosh(y) + np.sinh(y)) - self.cc.partonFormationTime

        x = root_scalar(func, t, x0=0)

        return x.root
    
    
    def __yMidMiddle(self, t):
        from scipy.optimize import root_scalar

        func = lambda y, t: self.cc.beta * (t - self.cc.t2) - self.cc.partonFormationTime * (self.cc.beta * np.cosh(y) - np.sinh(y))

        x = root_scalar(func, t, x0=0)

        return x.root

    
    def __yMaxLate(self, t):
        return np.arctanh(-self.cc.beta * (self.cc.tMid - self.cc.t2) / (t - self.cc.tMid))

    
    def __zerothPiece(self, times: np.ndarray) -> np.ndarray:
        return np.zeros(times.size)
    
    
    def __firstPiece(self, times: np.ndarray):
        vals = np.zeros(times.size)

        integrand = lambda y, t: self.integrand(y) * (self.__rMaxLower(y, t) - self.cc.partonFormationTime)

        for i, t in enumerate(times):
            y1 = self.__yMaxEarly(t)
            integral, error = self.integrator(integrand, 0, y1, t)
            vals[i] = integral

        return 2 * vals

    
    def __secondPiece(self, times: np.ndarray):
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
    
    
    def __thirdPiece(self, times: np.ndarray):
        vals = np.zeros(times.size)

        integrand = lambda y, t: self.integrand(y) * (self.__rMaxLower(y, t) - self.__rMinUpper(y, t))

        for i, t in enumerate(times):
            y1 = self.__yMaxLate(t)
            integral, error = self.integrator(integrand, 0, y1, t)
            vals[i] = integral

        return 2 * vals
    
    
    def calculate(self) -> np.ndarray:
        self.densities = np.piecewise(self.times, self.conditions, self.functions)


if __name__ == "__main__":
    from utilities import CalculationContext
    from differential_density import Integrand, EnergyDensity, NetBaryonDensity


    cc = CalculationContext(79, 197, 2, 0.1, "boltzmann", 500)
    e = EnergyDensity(cc)

    # ps1 = PiecewiseSolutionSingleIntegral(cc, e)
    # ps1.calculate()
    
    # ps2 = PiecewiseSolution(cc, e)
    # ps2.calculate(cc)
    
    # percentDifference = (ps1.densities - ps2.densities) / ps2.densities
    # print(percentDifference)

    # import matplotlib.pyplot as plt
    # plt.plot(ps1.times[2:], percentDifference[2:])
    # # plt.xscale('log')
    # plt.show()
