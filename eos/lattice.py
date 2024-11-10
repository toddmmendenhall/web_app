from calc_profile import CalcProfile
from eos.contour_intersection_finder import ContourIntersectionFinder
import numpy as np
import math
import os

class Lattice:
    def __init__(self, cp: CalcProfile, es: np.ndarray, nBs: np.ndarray) -> None:
        """_summary_

        Args:
            cp (CalcProfile): _description_
            es (np.ndarray): _description_
            nBs (np.ndarray): _description_
        """
        self.es = es
        self.nBs = nBs

        # Define nQ_dens and nS_dens arrays
        self.nQs = cp.z / cp.a * nBs
        self.nSs = np.zeros(es.size)
        
        self.thermoVars = np.zeros((es.size, 4))

        # All combinations of i,j,k whose sum is less than or equal to 4
        self.indices = [[0,0,0],
                        [2,0,0], [0,2,0], [0,0,2],
                        [1,1,0], [1,0,1], [0,1,1],
                        [4,0,0], [0,4,0], [0,0,4],
                        [3,1,0], [3,0,1], [0,3,1],
                        [1,3,0], [1,0,3], [0,1,3],
                        [2,2,0], [2,0,2], [0,2,2],
                        [2,1,1], [1,2,1], [1,1,2]]

        self.__load_parameters()

        # Set up temperature dimension of QCD phase diagram
        self.dtype = np.float64
        dtemp = 5
        temp = np.arange(50, 400 + dtemp, dtemp, dtype=self.dtype)

        # Calculate the susceptibilities that are used in the Taylor series expansion of the pressure
        # and their derivatives
        self.__chi_ijk(temp)
        self.__d_chi_ijk_dT(temp)

        # Set up chemical potential dimensions of QCD phase diagram
        dmuB = 5
        muB = np.arange(0, 1200 + dmuB, dmuB, dtype=self.dtype)
        self.temp, self.muB = np.meshgrid(temp, muB)
        self.muQ = np.zeros(self.temp.shape, dtype=self.dtype)
        self.muS = np.zeros(self.temp.shape, dtype=self.dtype)
        
        # Calculate the Taylor series expansion of the pressure and its derivatives w.r.t. the QCD phase
        # diagram variables
        self.__sum()
        self.__dsumdT()
        self.__dsumdmuB()
        self.__dsumdmuQ()
        self.__dsumdmuS()

        # Calculate EoS over the QCD phase diagram
        self.pressure = self.temp**4 * self.sum
        self.entropy_density = 4 * self.temp**3 * self.sum + self.temp**4 * self.dsumdT
        self.net_baryon_density = self.temp**4 * self.dsumdmuB
        self.net_charge_density = self.temp**4 * self.dsumdmuQ
        self.net_strangeness_density = self.temp**4 * self.dsumdmuS
        self.energy_density = self.temp * self.entropy_density - self.pressure + self.muB * self.net_baryon_density + self.muQ * self.net_charge_density +  self.muS * self.net_strangeness_density

        self.calculate()


    def __chi_ijk(self, temp: np.ndarray) -> None:
        """_summary_

        Args:
            temp (np.ndarray): _description_
        """
        self.chi_ijk = np.zeros((len(self.indices), len(temp)))
        scaledTemp200 = temp / 200
        scaledTemp154 = temp / 154

        for m, ijk in enumerate(self.indices):
            if ijk == [2,0,0]:
                h1, h2, f3, f4, f5 = tuple(self.chi_2_B_coeff)
                t = scaledTemp200
                self.chi_ijk[m] = np.exp((-h1 / t) - (h2 / t**2)) * f3 * (1 + np.tanh((f4 * t) + f5))
            else:
                powers = np.arange(10)
                t = np.broadcast_to(scaledTemp154, (10, len(scaledTemp154))).T
                a = np.dot(t**(-powers), self.a_n_ijk.T)
                b = np.dot(t**(-powers), self.b_n_ijk.T)
                x = a / b + self.c_0_ijk
                n = m if m == 0 else m - 1
                self.chi_ijk[m] = x.T[n]


    def __d_chi_ijk_dT(self, temp: np.ndarray) -> None:
        """_summary_

        Args:
            temp (np.ndarray): _description_
        """
        self.d_chi_ijk_dT = np.zeros((len(self.indices), len(temp)))
        scaledTemp200 = temp / 200
        scaledTemp154 = temp / 154

        for m, ijk in enumerate(self.indices):
            if ijk == [2,0,0]:
                h1, h2, f3, f4, f5 = tuple(self.chi_2_B_coeff)
                t = scaledTemp200
                self.d_chi_ijk_dT[m] = f3 * np.exp((-h1 / t) - (h2 / t**2)) / 200 * ((h1 / t**2 + 2 * h2 / t**3) * (1 + np.tanh((f4 * t) + f5)) + f4 / np.cosh(f4 * t + f5)**2)
            else:
                powers = np.arange(10)
                t = np.broadcast_to(scaledTemp154, (10, len(scaledTemp154))).T

                top = np.dot(t**(-powers), self.a_n_ijk.T)
                bottom = np.dot(t**(-powers), self.b_n_ijk.T)
                dtop = np.dot(-powers * t**(-powers-1), self.a_n_ijk.T)
                dbottom = np.dot(-powers * t**(-powers - 1), self.b_n_ijk.T)

                x = (bottom * dtop - top * dbottom) / bottom**2 / 154

                n = m if m == 0 else m - 1
                self.d_chi_ijk_dT[m] = x.T[n]


    def __sum(self) -> None:
        """_summary_
        """
        self.sum: np.ndarray = np.zeros(self.temp.shape)
        for m, ijk in enumerate(self.indices):
            i, j, k = tuple(ijk)
            const = 1 / (math.factorial(i) * math.factorial(j) * math.factorial(k))
            self.sum += const * self.chi_ijk[m] * (self.muB / self.temp)**i * (self.muQ / self.temp)**j * (self.muS / self.temp)**k


    def __dsumdT(self) -> None:
        """_summary_
        """
        self.dsumdT: np.ndarray = np.zeros(self.temp.shape)
        for m, ijk in enumerate(self.indices):
            i, j, k = tuple(ijk)
            const = (self.muB**i * self.muQ**j * self.muS**k) / (math.factorial(i) * math.factorial(j) * math.factorial(k))
            self.dsumdT += const * (self.temp**(i + j + k) * self.d_chi_ijk_dT[m] - self.chi_ijk[m] * (i + j + k) * self.temp**((i + j + k) - 1)) / self.temp**(2 * (i + j + k))


    def __dsumdmuB(self) -> None:
        """_summary_
        """
        self.dsumdmuB: np.ndarray = np.zeros(self.temp.shape)
        for m, ijk in enumerate(self.indices):
            i, j, k = tuple(ijk)
            const = 0 if i == 0 else i * (self.muB**(i-1) * self.muQ**j * self.muS**k) / (math.factorial(i) * math.factorial(j) * math.factorial(k))
            self.dsumdmuB += const * self.chi_ijk[m] / self.temp**(i + j + k)


    def __dsumdmuQ(self) -> None:
        """_summary_
        """
        self.dsumdmuQ: np.ndarray = np.zeros(self.temp.shape)
        for m, ijk in enumerate(self.indices):
            i, j, k = tuple(ijk)
            const = 0 if j == 0 else j * (self.muB**i * self.muQ**(j-1) * self.muS**k) / (math.factorial(i) * math.factorial(j) * math.factorial(k))
            self.dsumdmuQ += const * self.chi_ijk[m] / self.temp**(i + j + k)


    def __dsumdmuS(self) -> None:
        """_summary_
        """
        self.dsumdmuS: np.ndarray = np.zeros(self.temp.shape)
        for m, ijk in enumerate(self.indices):
            i, j, k = tuple(ijk)
            const = 0 if k == 0 else k * (self.muB**i * self.muQ**j * self.muS**(k-1)) / (math.factorial(i) * math.factorial(j) * math.factorial(k))
            self.dsumdmuS += const * self.chi_ijk[m] / self.temp**(i + j + k)


    def __load_parameters(self) -> None:
        """_summary_
        """
        directory = os.getcwd() + '/eos/lattice_parameters/'
        self.a_n_ijk = np.loadtxt(directory + 'a_n_ijk.csv', delimiter=',')
        self.b_n_ijk = np.loadtxt(directory + 'b_n_ijk.csv', delimiter=',')
        self.c_0_ijk = np.loadtxt(directory + 'c_0_ijk.csv', delimiter=',')
        self.chi_2_B_coeff = np.loadtxt(directory + 'chi_2_B.csv', delimiter=',', dtype=np.float64)
    

    def calculate(self):
        """_summary_
        """
        cif = ContourIntersectionFinder(self.temp, self.muB, self.energy_density, self.net_baryon_density, self.es, self.nBs)

        self.thermoVars = self.thermoVars.transpose()
        self.thermoVars[1] = cif.intersections[0]
        self.thermoVars[0] = cif.intersections[1]