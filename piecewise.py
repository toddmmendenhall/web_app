import numpy as np
from scipy.integrate import dblquad


class PiecewiseSolution:
    from utilities import CalculationContext
    from differential_density import Integrand

    def __init__(self, cc: CalculationContext, density: Integrand) -> None:
        tMin = 1.1 * cc.t1 + cc.partonFormationTime
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


if __name__ == "__main__":
    from utilities import CalculationContext
    from differential_density import Integrand, EnergyDensity, NetBaryonDensity


    cc = CalculationContext(79, 197, 200, 0.3, "boltzmann", 10)
    e = EnergyDensity(cc)

    ps = PiecewiseSolution(cc, e)
    # print(ps.conditions)
    # print(ps.functions)
    ps.calculate(cc)
