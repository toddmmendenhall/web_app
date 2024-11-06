from eos.thermo_calc import NonInteractingMasslessQuantumEOSFullSolution, NonInteractingMasslessBoltzmannEOSFullSolution
from densities.differential_density import EnergyDensity, NetBaryonDensity
from calc_profile import CalcProfile
import numpy as np


class EquationOfState:
    def __init__(self, cp: CalcProfile, ed: EnergyDensity, nB: NetBaryonDensity):
        match cp.eosType:
            case "quantum":
                self.eos = NonInteractingMasslessQuantumEOSFullSolution(cp, ed.densities, nB.densities)
            case "boltzmann":
                self.eos = NonInteractingMasslessBoltzmannEOSFullSolution(cp, ed.densities, nB.densities)
            case _:
                pass

    def get_data(self) -> np.ndarray:
        return self.eos.thermoVars