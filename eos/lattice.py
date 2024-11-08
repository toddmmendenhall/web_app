from calc_profile import CalcProfile
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
        
        self.thermoVars = np.zeros((es.size, 3))

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

        # Set up QCD phase diagram grid
        self.dtype = np.float64
        dtemp = 10
        dmuB = 10
        temp = np.arange(50, 300 + dtemp, dtemp, dtype=self.dtype)
        muB = np.arange(0, 1000 + dmuB, dmuB, dtype=self.dtype)
        self.temp, self.muB = np.meshgrid(temp, muB)
        self.muQ = np.zeros(self.temp.shape, dtype=self.dtype)
        self.muS = np.zeros(self.temp.shape, dtype=self.dtype)

        # Calculate terms that are used in the Taylor series expansion of the pressure
        self.__chi_ijk(self.temp)
        self.__chi_2_B(self.temp)
        self.__dchi_ijk_dT(self.temp)
        self.__dchi_2_B_dT(self.temp)
        
        # Calculate the Taylor series expansion of the pressure and some derivatives
        # so we can calculate thermodynamics
        self.__sum(self.temp, self.muB, self.muQ, self.muS)
        self.__dsumdT(self.temp, self.muB, self.muQ, self.muS)
        self.__dsumdmuB(self.temp, self.muB, self.muQ, self.muS)
        self.__dsumdmuQ(self.temp, self.muB, self.muQ, self.muS)
        self.__dsumdmuS(self.temp, self.muB, self.muQ, self.muS)

        # Calculate the thermodynamic quantities
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
        temp = temp / 154
        temp = np.tile(temp, 10).reshape((*temp.shape, 10))
        powers = np.arange(10)
        self.chi_ijk: np.ndarray = np.dot(temp**(-powers), self.a_n_ijk.T) / np.dot(temp**(-powers), self.b_n_ijk.T) + self.c_0_ijk
        self.chi_ijk = np.sum(self.chi_ijk, axis=2)


    def __chi_2_B(self, temp: np.ndarray) -> None:
        """_summary_

        Args:
            temp (np.ndarray): _description_
        """
        h1, h2, f3, f4, f5 = tuple(self.chi_2_B_coeff)
        temp = temp / 200
        self.chi_2_B: np.ndarray = np.exp((-h1 / temp) - (h2 / temp**2)) * f3 * (1 + np.tanh((f4 * temp) + f5))


    def __dchi_ijk_dT(self, temp: np.ndarray) -> None:
        """_summary_

        Args:
            temp (np.ndarray): _description_
        """
        temp = temp / 154
        temp = np.tile(temp, 10).reshape((*temp.shape, 10))
        powers = np.arange(10)

        top = np.dot(temp**(-powers), self.a_n_ijk.T)
        bottom = np.dot(temp**(-powers), self.b_n_ijk.T)
        dtop = np.dot(-powers * temp**(-powers-1), self.a_n_ijk.T)
        dbottom = np.dot(-powers * temp**(-powers - 1), self.b_n_ijk.T)

        self.dchi_ijk_dT: np.ndarray = (bottom * dtop - top * dbottom) / bottom**2 / 154
        self.dchi_ijk_dT = np.sum(self.dchi_ijk_dT, axis=2)


    def __dchi_2_B_dT(self, temp: np.ndarray) -> None:
        """_summary_

        Args:
            temp (np.ndarray): _description_
        """
        temp = temp / 200
        h1, h2, f3, f4, f5 = tuple(self.chi_2_B_coeff)
        self.dchi_2_B_dT: np.ndarray = f3 * np.exp((-h1 / temp) - (h2 / temp**2)) / 200 * ((h1 / temp**2 + 2 * h2 / temp**3) * (1 + np.tanh((f4 * temp) + f5)) + f4 / np.cosh(f4 * temp + f5)**2)


    def __sum(self, temp: np.ndarray, muB: np.ndarray, muQ: np.ndarray, muS: np.ndarray) -> None:
        """_summary_

        Args:
            temp (np.ndarray): _description_
            muB (np.ndarray): _description_
            muQ (np.ndarray): _description_
            muS (np.ndarray): _description_
        """
        self.sum: np.ndarray = np.zeros(temp.shape)
        for ijk in self.indices:
            i, j, k = tuple(ijk)
            const = 1 / (math.factorial(i) * math.factorial(j) * math.factorial(k))
            if ijk == [2,0,0]:
                chi = self.chi_2_B
            else:
                chi = self.chi_ijk
            self.sum += const * chi * (muB / temp)**i * (muQ / temp)**j * (muS / temp)**k


    def __dsumdT(self, temp: np.ndarray, muB: np.ndarray, muQ: np.ndarray, muS: np.ndarray) -> None:
        """_summary_

        Args:
            temp (np.ndarray): _description_
            muB (np.ndarray): _description_
            muQ (np.ndarray): _description_
            muS (np.ndarray): _description_
        """
        self.dsumdT: np.ndarray = np.zeros(temp.shape)
        for ijk in self.indices:
            i, j, k = tuple(ijk)

            const = (muB**i * muQ**j * muS**k) / (math.factorial(i) * math.factorial(j) * math.factorial(k))

            if ijk == [2,0,0]:
                chi = self.chi_2_B
                dchidT = self.dchi_2_B_dT
            else:
                chi = self.chi_ijk
                dchidT = self.dchi_ijk_dT

            m = i + j + k

            self.dsumdT += const * (temp**m * dchidT - chi * m * temp**(m - 1)) / temp**(2 * m)


    def __dsumdmuB(self, temp: np.ndarray, muB: np.ndarray, muQ: np.ndarray, muS: np.ndarray) -> None:
        """_summary_

        Args:
            temp (np.ndarray): _description_
            muB (np.ndarray): _description_
            muQ (np.ndarray): _description_
            muS (np.ndarray): _description_
        """
        self.dsumdmuB: np.ndarray = np.zeros(temp.shape)
        for ijk in self.indices:
            i, j, k = tuple(ijk)

            const = i * (muB**(i-1) * muQ**j * muS**k) / (math.factorial(i) * math.factorial(j) * math.factorial(k))

            if ijk == [2,0,0]:
                chi = self.chi_2_B
            else:
                chi = self.chi_ijk

            m = i + j + k

            self.dsumdmuB += const * chi / temp**m


    def __dsumdmuQ(self, temp: np.ndarray, muB: np.ndarray, muQ: np.ndarray, muS: np.ndarray) -> None:
        """_summary_

        Args:
            temp (np.ndarray): _description_
            muB (np.ndarray): _description_
            muQ (np.ndarray): _description_
            muS (np.ndarray): _description_
        """
        self.dsumdmuQ: np.ndarray = np.zeros(temp.shape)
        for ijk in self.indices:
            i, j, k = tuple(ijk)

            const = j * (muB**i * muQ**(j-1) * muS**k) / (math.factorial(i) * math.factorial(j) * math.factorial(k))

            if ijk == [2,0,0]:
                chi = self.chi_2_B
            else:
                chi = self.chi_ijk

            self.dsumdmuQ += const * chi / temp**(i + j + k)


    def __dsumdmuS(self, temp: np.ndarray, muB: np.ndarray, muQ: np.ndarray, muS: np.ndarray) -> None:
        """_summary_

        Args:
            temp (np.ndarray): _description_
            muB (np.ndarray): _description_
            muQ (np.ndarray): _description_
            muS (np.ndarray): _description_
        """
        self.dsumdmuS: np.ndarray = np.zeros(temp.shape)
        for ijk in self.indices:
            i, j, k = tuple(ijk)

            const = k * (muB**i * muQ**j * muS**(k-1)) / (math.factorial(i) * math.factorial(j) * math.factorial(k))

            if ijk == [2,0,0]:
                chi = self.chi_2_B
            else:
                chi = self.chi_ijk

            self.dsumdmuS += const * chi / temp**(i + j + k)


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
        self.thermoVars = self.thermoVars.transpose()
