import numpy as np


class Constants:
    nucleonMass = 0.94      # GeV / c^2
    nucleonRadius = 1.12    # fm
    hbarTimesC = 0.19732698 # GeV * fm
    MeVPerGeV = 1000
    piSquared = np.pi**2


class CalculationContext:
    def __init__(self, atomicNum: int, massNum: int, comCollisionEnergy: float,
                 partonFormationTime: float, eos: str, numTimes: int) -> None:
        self.atomicNum: int = atomicNum                         #
        self.massNum: int = massNum                             #
        self.comCollisionEnergy: float = comCollisionEnergy     # GeV
        self.partonFormationTime: float = partonFormationTime   # fm
        self.equationOfState: str = eos                         #
        self.numTimes: int = numTimes                           #

        self.ensure_valid_data()

        self.nuclearRadius: float = Constants.nucleonRadius * self.massNum**(1/3)
        self.transverseOverlapArea: float = np.pi * self.nuclearRadius**2

        self.projectileRapidity = np.arccosh(self.comCollisionEnergy / (2 * Constants.nucleonMass))

        self.beta = np.tanh(self.projectileRapidity)

        self.crossingTime = 2 * self.nuclearRadius / np.sinh(self.projectileRapidity)

        self.t1 = self.crossingTime / 6

        self.t2 = 5 / 6 * self.crossingTime

        self.t21 = self.t2 - self.t1

        self.tMid = (self.t1 + self.t2) / 2

        self.ta = self.tMid + np.sqrt(self.partonFormationTime**2 + (self.beta * self.t21 / 2)**2)

        self.__setTimes()

    def __setTimes(self):
        tMin = self.t1 + self.partonFormationTime
        tMax = 3 * self.t2 + self.partonFormationTime

        if(tMax < 10):
            tMax = 10

        self.times = np.logspace(np.log10(tMin), np.log10(tMax), self.numTimes)
        self.times = np.insert(self.times, 0, 0)


    def get_data(self) -> list:
        return [self.atomicNum, self.massNum, self.comCollisionEnergy,
                self.partonFormationTime, self.equationOfState, self.numTimes]
    
    def ensure_valid_data(self) -> None:
        if self.comCollisionEnergy < 2 * Constants.nucleonMass:
            self.comCollisionEnergy = 2 * Constants.nucleonMass

        if self.comCollisionEnergy > 200.0:
            self.comCollisionEnergy = 200.0

        if self.numTimes < 1:
            self.numTimes = 1

        if self.numTimes > 300:
            self.numTimes = 300
    
    def get_data_for_density_calc(self) -> list:
        return [self.massNum, self.comCollisionEnergy, self.partonFormationTime, self.numTimes]
    
    def get_data_for_thermo_calc(self) -> list:
        return [self.atomicNum] + self.get_data_for_density_calc()
    
    def x1(self, t):
        return (t - self.beta**2 * self.t1 - np.sqrt(self.beta**2 * ((t - self.t1)**2 - self.partonFormationTime**2) + self.partonFormationTime**2)) / (1 - self.beta**2)

    def x2(self, t):
            return (t - self.beta**2 * self.t2 - np.sqrt(self.beta**2 * ((t - self.t2)**2 - self.partonFormationTime**2) + self.partonFormationTime**2)) / (1 - self.beta**2)

if __name__ == "__main__":
    cc = CalculationContext(79, 197, 50, 0.3, "boltzmann", 10)

    print(cc.get_data())