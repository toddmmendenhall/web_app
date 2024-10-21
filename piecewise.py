import numpy as np
from scipy.integrate import dblquad
from density_calc import t1, ta, x1, beta, tmid, x2, t2
from utilities import CalculationContext
from differential_density import Integrand, EnergyDensity, NetBaryonDensity

def piecewise_solution(integrand, s, t, ra, at, a, tauf) -> np.ndarray:
    startTime = t1(s, ra)
    midTime = tmid(s, ra)
    finalTime = t2(s, ra)
    relativisticVelocity = beta(s)

    if t <= startTime + tauf:
        return np.array([0, 0])

    elif t > startTime + tauf and t < ta(s, ra, tauf):
        
        return np.sum(np.array([dblquad(integrand,
                                        startTime,
                                        x1(s, ra, t, tauf),
                                        lambda x: -relativisticVelocity * (x - startTime),
                                        lambda x: relativisticVelocity * (x - startTime),
                                        args = (s, t, ra, at, a)),
                                dblquad(integrand,
                                        x1(s, ra, t, tauf),
                                        t - tauf,
                                        lambda x: -np.sqrt((t - x)**2 - tauf**2),
                                        lambda x: np.sqrt((t - x)**2 - tauf**2),
                                        args = (s, t, ra, at, a))]).transpose(), axis = 1)

    elif t >= ta(s, ra, tauf) and t < finalTime + tauf:
        return np.sum(np.array([dblquad(integrand,
                                        startTime,
                                        midTime,
                                        lambda x: -relativisticVelocity * (x - startTime),
                                        lambda x: relativisticVelocity * (x - startTime),
                                        args = (s, t, ra, at, a)),
                                dblquad(integrand,
                                        midTime,
                                        x2(s, ra, t, tauf),
                                        lambda x: -relativisticVelocity * (finalTime - x),
                                        lambda x: relativisticVelocity * (finalTime - x),
                                        args = (s, t, ra, at, a)),
                                dblquad(integrand,
                                        x2(s, ra, t, tauf),
                                        t - tauf,
                                        lambda x: -np.sqrt((t - x)**2 - tauf**2),
                                        lambda x: np.sqrt((t - x)**2 - tauf**2),
                                        args = (s, t, ra, at, a))]).transpose(), axis = 1)

    else:
        return np.sum(np.array([dblquad(integrand,
                                        startTime,
                                        midTime,
                                        lambda x: -relativisticVelocity * (x - startTime),
                                        lambda x: relativisticVelocity * (x - startTime),
                                        args = (s, t, ra, at, a)),
                                dblquad(integrand,
                                        midTime,
                                        finalTime,
                                        lambda x: -relativisticVelocity * (finalTime - x),
                                        lambda x: relativisticVelocity * (finalTime - x),
                                        args = (s, t, ra, at, a))]).transpose(), axis = 1)

class PiecewiseSolution:
    def __init__(self, cc: CalculationContext, density: Integrand) -> None:
        tMin = 1.1 * t1(cc.comCollisionEnergy, cc.nuclearRadius) + cc.partonFormationTime
        tMax = 3 * t2(cc.comCollisionEnergy, cc.nuclearRadius) + cc.partonFormationTime

        if(tMax < 10):
            tMax = 10

        self.times = np.logspace(np.log10(tMin), np.log10(tMax), cc.numTimes)
        self.times = np.insert(self.times, 0, 0)

        self.conditions = [self.times <= (cc.t1 + cc.partonFormationTime)]#,
                        #    (self.times > (cc.t1 + cc.partonFormationTime)) * (self.times < cc.ta)]#,
                        #    (self.times >= cc.ta) * (self.times < (cc.t2 + cc.partonFormationTime)),
                        #    self.times >= (cc.t2 + cc.partonFormationTime)]
        
        self.functions = [self.__zerothPiece]#, self.__firstPiece]#, self.__secondPiece, self.__thirdPiece]

        self.integrand = density.integrand
        self.x1 = cc.x1
        self.x2 = cc.x2
    
    def __zerothPiece(self, times: np.ndarray, cc: CalculationContext) -> np.ndarray:
        return np.zeros(times.size)
    
    # def __firstPiece(self, times: np.ndarray, cc: CalculationContext):
    #     vals = np.zeros(times.size)
    #     lowerLeftZBound = lambda x: -cc.beta * (x - cc.t1)
    #     lowerRightZBound = lambda x: cc.beta * (x - cc.t1)
    #     upperLeftZBound = lambda x: -np.sqrt((t - x)**2 - cc.partonFormationTime**2)
    #     upperRightZBound = lambda x: np.sqrt((t - x)**2 - cc.partonFormationTime**2)

    #     upperXBound = self.x1(t)

    #     lowerIntegral, lowerError = dblquad(self.integrand, cc.t1, upperXBound, lowerLeftZBound, lowerRightZBound, args = (t,))
    #     upperIntegral, upperError = dblquad(self.integrand, upperXBound, t - cc.partonFormationTime, upperLeftZBound, upperRightZBound, args=(t,))

    #     return lowerIntegral + upperIntegral
    
    def __secondPiece(self, t: np.float64, cc: CalculationContext):
        lowerLeftZBound = lambda x: -cc.beta * (x - cc.t1)
        lowerRightZBound = lambda x: cc.beta * (x - cc.t1)
        middleLeftZBound = lambda x: -cc.beta * (cc.t2 - x)
        middleRightZBound = lambda x: cc.beta * (cc.t2 - x)
        upperLeftZBound = lambda x: -np.sqrt((t - x)**2 - cc.partonFormationTime**2)
        upperRightZBound = lambda x: np.sqrt((t - x)**2 - cc.partonFormationTime**2)

        upperXBound = self.x2(t)

        lowerIntegral, lowerError = dblquad(self.integrand, cc.t1, cc.tMid, lowerLeftZBound, lowerRightZBound, args = (t,))
        middleIntegral, middleError = dblquad(self.integrand, cc.tMid, upperXBound, middleLeftZBound, middleRightZBound, args=(t,))
        upperIntegral, upperError = dblquad(self.integrand, upperXBound, t - cc.partonFormationTime, upperLeftZBound, upperRightZBound, args=(t,))

        return lowerIntegral + middleIntegral + upperIntegral
    
    def __thirdPiece(self, t: np.float64, cc: CalculationContext):
        lowerLeftZBound = lambda x: -cc.beta * (x - cc.t1)
        lowerRightZBound = lambda x: cc.beta * (x - cc.t1)
        upperLeftZBound = lambda x: -cc.beta * (cc.t2 - x)
        upperRightZBound = lambda x: cc.beta * (cc.t2 - x)

        lowerIntegral, lowerError = dblquad(self.integrand, cc.t1, cc.tMid, lowerLeftZBound, lowerRightZBound, args = (t,))
        upperIntegral, upperError = dblquad(self.integrand, cc.tMid, cc.t2, upperLeftZBound, upperRightZBound, args = (t,))

        return lowerIntegral + upperIntegral
    
    def calculate(self, cc: CalculationContext) -> np.ndarray:
        arguments = (cc,)
        return np.piecewise(self.times, self.conditions, self.functions, arguments)


if __name__ == "__main__":
    cc = CalculationContext(79, 197, 200, 0.3, "boltzmann", 10)

    e = EnergyDensity(cc)

    ps = PiecewiseSolution(cc, e)
    print(ps.times)
    print(ps.conditions)
    print(ps.functions)
    print(ps.calculate(cc))