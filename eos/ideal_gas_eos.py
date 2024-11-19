from eos.particle_data import ParticleData
from eos.phase_diagram import PhaseDiagram
from calc_profile import CalcProfile
from eos.contour_intersection_finder import ContourIntersectionFinder

import numpy as np
from scipy.integrate import quad


class IdealGasEOS:
    """Object that calculates the equation of state of an ideal gas of quantum particles
    """
    def __init__(self, particle_data: ParticleData, phase_diagram: PhaseDiagram, cp: CalcProfile, es: np.ndarray, nBs: np.ndarray) -> None:
        """Constructor that first calculates the pressure and then the other EOS variables

        Args:
            particle_data (ParticleData): Object that contains all the particles information
            phase_diagram (PhaseDiagram): Object that contains the grid over which the EOS is being calculated
        """
        self.es = es
        self.nBs = nBs
        
        self.thermoVars = np.zeros((es.size, 4))

        self._calculate_pressure(particle_data, phase_diagram)
        self.entropy_density = np.gradient(self.pressure, phase_diagram.temp, axis=1, edge_order=2)
        self.baryon_density = np.gradient(self.pressure, phase_diagram.muB, axis=0, edge_order=2)
        self.baryon_susceptibility = np.gradient(self.baryon_density, phase_diagram.muB, axis=0, edge_order=2)
        self.temp, self.muB = phase_diagram.grid()
        self.energy_density = self.temp * self.entropy_density - self.pressure + self.muB * self.baryon_density

        self.calculate()

    def _calculate_pressure(self, particle_data: ParticleData, phase_diagram: PhaseDiagram) -> None:
        """Private method that loops over all the provided particles and calculates their partial pressures

        Args:
            particle_data (ParticleData): Object that contains all the particles information
            phase_diagram (PhaseDiagram): Object that contains the grid over which the EOS is being calculated
        """
        self.pressure = np.zeros(phase_diagram.shape())
        for name in particle_data.names:
            species_data = particle_data.data(name)
            if species_data['type'] == 'boson':
                self._calculate_pressure_boson(phase_diagram, species_data)
            elif species_data['type'] == 'fermion':
                self._calculate_pressure_fermion(phase_diagram, species_data)
            else:
                raise

    def _integrand_boson(self, p: np.float64, temp: np.float64, mu: np.float64, m: np.float64) -> np.float64:
        """Private method that defines the integrand of the pressure for a boson over the momentum

        Args:
            p (np.float64): The momentum [MeV]
            temp (np.float64): The temperature [MeV]
            mu (np.float64): The chemical potential [MeV]
            m (np.float64): The mass [MeV]

        Returns:
            np.float64: The value of the integrand evaluated at arguments
        """
        return -p**2 * np.log(1 - np.exp(-(np.sqrt(p**2 + m**2) - mu) / temp))

    def _integrand_fermion(self, p: np.float64, temp: np.float64, mu: np.float64, m: np.float64) -> np.float64:
        """Private method that defines the integrand of the pressure for a boson over the momentum

        Args:
            p (np.float64): The momentum [MeV]
            temp (np.float64): The temperature [MeV]
            mu (np.float64): The chemical potential [MeV]
            m (np.float64): The mass [MeV]

        Returns:
            np.float64: The value of the integrand evaluated at arguments
        """
        return p**2 * np.log(1 + np.exp(-(np.sqrt(p**2 + m**2) - mu) / temp))
    
    def _calculate_pressure_boson(self, phase_diagram: PhaseDiagram, species_data: dict[str, any]) -> None:
        """Private method that calculates the partial pressure of a single boson

        Args:
            phase_diagram (PhaseDiagram): Object that contains the grid over which the EOS is being calculated
            species_data (dict[str, any]): Object that contains the boson's properties
        """
        for i, muB in enumerate(phase_diagram.muB):
            mu: np.float64 = species_data['baryon_number'] * muB
            for j, temp in enumerate(phase_diagram.temp):
                self.pressure[i][j] += quad(self._integrand_boson, 0, np.inf, args=(temp, mu, species_data['mass']))[0]
        self.pressure *= species_data['degeneracy'] / (2 * np.pi**2) * phase_diagram.temp

    def _calculate_pressure_fermion(self, phase_diagram: PhaseDiagram, species_data: dict[str, any]) -> None:
        """Private method that calculates the partial pressure of a single fermion

        Args:
            phase_diagram (PhaseDiagram): Object that contains the grid over which the EOS is being calculated
            species_data (dict[str, any]): Object that contains the fermion's properties
        """
        for i, muB in enumerate(phase_diagram.muB):
            mu: np.float64 = species_data['baryon_number'] * muB
            for j, temp in enumerate(phase_diagram.temp):
                self.pressure[i][j] += quad(self._integrand_fermion, 0, np.inf, args=(temp, mu, species_data['mass']))[0]
        self.pressure *= species_data['degeneracy'] / (2 * np.pi**2) * phase_diagram.temp
        
    def calculate(self):
        cif = ContourIntersectionFinder(self.temp, self.muB, self.energy_density, self.baryon_density, self.es, self.nBs)

        self.thermoVars = self.thermoVars.transpose()
        self.thermoVars[1] = cif.intersections[0]
        self.thermoVars[0] = cif.intersections[1]