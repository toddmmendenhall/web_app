from utilities import CalculationContext
import numpy as np


class SpacetimeProduction:
    def __init__(self, cc: CalculationContext) -> None:
        self.cc = cc

    def g(self, z: float, x: float, t: float) -> float:
        pass


class UniformDistribution(SpacetimeProduction):
    def __init__(self, cc: CalculationContext) -> None:
        super().__init__(cc)
        self.constantTerm = 2 / (self.cc.transverseOverlapArea * self.cc.beta * self.cc.t21**2)

    def g(self, z: float, x: float, t: float) -> float:
        return 1


class SemiCircleDistribution(SpacetimeProduction):
    def __init__(self, cc: CalculationContext) -> None:
        super().__init__(cc)
        timeTerm = 1 / 3 * self.cc.tMid**3  - self.cc.tMid**2 * self.cc.t1 + self.cc.tMid * self.cc.t1**2
        self.constantTerm = 1 / (np.pi * self.cc.transverseOverlapArea * self.cc.beta**2 * timeTerm)

    def g(self, z: float, x: float, t: float) -> float:
        return np.sqrt((self.cc.beta * x)**2 - z**2)
    
if __name__ == "__main__":
    cc = CalculationContext(79, 197, 20, 0.1, "b", 10, "semicircle")

