from utilities import Constants
import numpy as np
from scipy.optimize import fsolve

class CalcProfile:
    def __init__(self, atomicNumber: int, atomicMassNumber: int, collisionEnergy: float, formationTime: float, eosType: str, nTimes: int) -> None:
        precision = 12
        # Store the user input
        self.z: int = atomicNumber
        self.a: int = atomicMassNumber
        self.s: float = collisionEnergy
        self.tform: float = np.round(formationTime, precision)
        self.eosType: str = eosType
        self.ntimes: int = nTimes

        # Calculate quantities derived from user input
        self.ra: float = Constants.nucleonRadius * self.a**(1/3)
        self.at: float = np.pi * self.ra**2
        self.ycm = np.arccosh(self.s / (2 * Constants.nucleonMass))
        self.beta = np.tanh(self.ycm)
        self.dt = 2 * self.ra / np.sinh(self.ycm)

        # Set up quantities for the time evolution
        self.t1 = np.round(1 / 6 * self.dt, precision)
        self.t2 = np.round(5 / 6 * self.dt, precision)
        self.tprod = self.t2 - self.t1
        self.tmid = np.round((self.t1 + self.t2) / 2, precision)
        self.ta = np.round(self.tmid + np.sqrt(self.tform**2 + (self.beta * self.tprod / 2)**2), precision)

        # Set up the sampling times
        tmin = np.round(self.t1 + self.tform, precision)
        tmax = np.round(3 * self.t2 + self.tform, precision)
        if(tmax < 10):
            tmax = 10
        self.times = np.logspace(np.log10(tmin), np.log10(tmax), self.ntimes)
        self.times = np.insert(self.times, 0, 0)
        self.times = np.round(self.times, precision)

        # Set up values for the rapidity distributions
        self.detdy0old = 0.456 * 2 * self.a * np.log(self.s / 2.35)
        self.detdy0 = (1.25 * 0.308 * 2 * self.a * np.log(self.s / (2 * Constants.nucleonMass))**1.08) * (self.s <= 20.7) + self.detdy0old * (self.s > 20.7)
        self.sigmabaryon = 0.601 * (self.s - 2 * Constants.nucleonMass)**0.121 * np.log(self.s / (2 * Constants.nucleonMass))**0.241
        self.ybaryon =  0.541 * (self.s - 2 * Constants.nucleonMass)**0.196 * np.log(self.s / (2 * Constants.nucleonMass))**0.392

        # Calculate height of dN_netB/dy using conservation of baryon number
        self.dnbdy0 = 2 * self.a / (np.sqrt(2 * np.pi) * self.sigmabaryon) * np.exp(-(self.ybaryon / self.sigmabaryon)**2 / 2)

        # Calculate Gaussian width of dE_T/dy using conservation of energy
        func = lambda x: self.a * self.s - self.detdy0 * np.exp(x**2 / 2) * np.sqrt(2 * np.pi) * x - Constants.nucleonMass * 2 * self.a * np.exp(self.sigmabaryon**2 / 2) * np.cosh(self.ybaryon)
        self.sigmahadron = fsolve(func, 0)[0]
