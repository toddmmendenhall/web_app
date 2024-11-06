from calc_profile import CalcProfile
import numpy as np

class Lattice:
    def __init__(self, cp: CalcProfile, es: np.ndarray, nBs: np.ndarray) -> None:
        self.es = es
        self.nBs = nBs

        # Define nQ_dens and nS_dens arrays
        self.nQs = cp.z / cp.a * nBs
        self.nSs = np.zeros(es.size)
        
        self.thermoVars = np.zeros((es.size, 3))

        self.calculate()
    
    def calculate(self):
        self.thermoVars = self.thermoVars.transpose()
