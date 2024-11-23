from eos.thermo_calc import NonInteractingMasslessQuantumEOSFullSolution, NonInteractingMasslessBoltzmannEOSFullSolution
from eos.lattice import Lattice
from densities.differential_density import EnergyDensity, NetBaryonDensity
from eos.ideal_gas_eos import IdealGasEOS
from eos.particle_data import ParticleData
from eos.phase_diagram import PhaseDiagram
from calc_profile import CalcProfile
import numpy as np


class EquationOfState:
    def __init__(self, cp: CalcProfile, ed: EnergyDensity, nB: NetBaryonDensity):
        match cp.eosType:
            case "quantum":
                self.eos = NonInteractingMasslessQuantumEOSFullSolution(cp, ed.densities, nB.densities)
            case "boltzmann":
                self.eos = NonInteractingMasslessBoltzmannEOSFullSolution(cp, ed.densities, nB.densities)
            case "lattice":
                self.eos = Lattice(cp, ed.densities, nB.densities)
            case "nhg":
                particle_data = ParticleData(['pi_zero', 'neutron', 'antineutron'])
                phase_diagram = PhaseDiagram([50, 400, 50, 0, 1200, 50])
                self.eos = IdealGasEOS(particle_data, phase_diagram, cp, ed.densities, nB.densities)
            case _:
                pass

    def get_data(self) -> np.ndarray:
        return self.eos.thermoVars