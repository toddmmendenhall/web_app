class Constants:
    nucleonMass = 0.94      # GeV / c^2
    nucleonRadius = 1.12    # fm
    hbarTimesC = 0.19732698 # GeV * fm
    MeVPerGeV = 1000

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

        if self.numTimes > 100:
            self.numTimes = 100
    
    def get_data_for_density_calc(self) -> list:
        return [self.massNum, self.comCollisionEnergy, self.partonFormationTime, self.numTimes]
    
    def get_data_for_thermo_calc(self) -> list:
        return [self.atomicNum] + self.get_data_for_density_calc()

if __name__ == "__main__":
    cc = CalculationContext(79, 197, 50, 0.3, "boltzmann", 10)

    print(cc.get_data())