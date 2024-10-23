import numpy as np
from scipy.optimize import fsolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utilities import Constants

from density_calc import t1, t2

################################################################################
# Definition of quantum statistics full solution

def quant_full_soln(z, a, sqrtsnn, tauf, ntimes):

    # Define parameters
    ra = Constants.nucleonRadius * a**(1 / 3)
    at = np.pi * ra**2

    # Set time array
    timesMin = 1.1 * t1(sqrtsnn, ra) + tauf
    timesMax = 3 * t2(sqrtsnn, ra) + tauf
    if(timesMax < 10):
        timesMax = 10
    times = np.logspace(np.log10(timesMin), np.log10(timesMax), ntimes)
    times = np.insert(times, 0, 0)

    # Import ep_dens and nB_dens data files into arrays
    e_file = 'results/e-dens-vs-t.dat'
    nB_file = 'results/nB-dens-vs-t.dat'

    e_dens = np.loadtxt(e_file, delimiter=',', usecols=1)
    nB_dens = np.loadtxt(nB_file, delimiter=',', usecols=1)

    # Define nQ_dens and nS_dens arrays
    nQ_dens = z / a * nB_dens
    nS_dens = np.zeros(len(times))

    # Calculate T, muB, muS, and muQ arrays
    def eqns(x, ep, nB, nQ):
        T, muB, muS = x

        eq1 = 19 * np.pi**2 * T**4 / 12 + 3 * ((muB - 2 * muS)**2 + muS**2) * T**2 / 2 + 3 * ((muB - 2 * muS)**4 +muS**4) / 4 / np.pi**2 - ep * Constants.hbarTimesC**3
        eq2 = (muB - muS) * T**2 / 3 + ((muB - 2 * muS)**3 + muS**3 ) / 3 / np.pi**2 - nB * Constants.hbarTimesC**3
        eq3 = (2 * muB - 5 * muS) * T**2 / 3 + (2 * (muB - 2 * muS)**3 - muS**3) / 3 / np.pi**2 - nQ * Constants.hbarTimesC**3

        return [eq1, eq2, eq3]

    traj = np.zeros((len(times), 3))

    for i, t in enumerate(times):
        traj[i] = fsolve(eqns, x0 = np.array([0.3, 0.9, 0.3]), args = (e_dens[i], nB_dens[i], nQ_dens[i]), maxfev = 1000)

    traj = traj * Constants.MeVPerGeV
    traj = traj.transpose()

    muQ = traj[1] - 3 * traj[2]

    # Save t, e_dens, T, muB, muS, muQ to data files
    output = np.vstack((times, e_dens, traj, muQ)).transpose()
    outfile = 'results/T-muB-muS-muQ-vs-t.dat'
    myHeader = 't (fm/c), e (GeV/fm^3), T (MeV), muB (MeV), muS (MeV), muQ (MeV)'

    np.savetxt(outfile, output, delimiter = ',', fmt = '%12.3f', header = myHeader)

    # Plot and save a T-muB trajectory figure
    fig, ax = plt.subplots()

    ax.plot(traj[1], traj[0], marker = '.')

    ax.tick_params(axis = 'both', which = 'both', direction = 'in', top = True, right = True)
    ax.grid(alpha = 0.5)

    ax.set_xlim(0)
    ax.set_ylim(0)

    ax.set_xlabel('$\mathrm{\mu}_B$ (MeV)')
    ax.set_ylabel('T (MeV)')

    plt.title('$\sqrt{\mathrm{s_{NN}}}$ = ' + str(sqrtsnn) + ' GeV, Z = ' + str(int(z)) + ', A = ' + str(int(a)) + ', $\\tau_F$ = ' + str(tauf) + ' fm/c, Quantum Statistics')

    figfile = 'results/results.pdf'

    plt.tight_layout()
    plt.savefig(figfile)

    return

################################################################################
# Definition of Boltzmann statistics full solution

def boltz_full_soln(z, a, sqrtsnn, tauf, ntimes):

    # Define parameters
    ra = Constants.nucleonRadius * a**(1 / 3)
    at = np.pi * ra ** 2.

    # Set time array
    timesMin = 1.1 * t1(sqrtsnn, ra) + tauf
    timesMax = 3 * t2(sqrtsnn, ra) + tauf
    if(timesMax < 10):
        timesMax = 10
    times = np.logspace(np.log10(timesMin), np.log10(timesMax), ntimes)
    times = np.insert(times, 0, 0)

    # Import ep_dens and nB_dens data files into arrays
    e_file = 'results/e-dens-vs-t.dat'
    nB_file = 'results/nB-dens-vs-t.dat'

    e_dens = np.loadtxt(e_file, delimiter=',', usecols=1)
    nB_dens = np.loadtxt(nB_file, delimiter=',', usecols=1)

    # Define nQ_dens and nS_dens arrays
    nQ_dens = z / a * nB_dens
    nS_dens = np.zeros(len(times))

    # Calculate T, muB, muS, and muQ arrays
    def eqns(x, ep, nB, nQ):
        T, muB, muS = x

        eq1 = 12 / np.pi**2 * T**4 * (7 + 3 * np.cosh((muB - 2 * muS) / T) + 3 * np.cosh(muS / T)) - ep * Constants.hbarTimesC**3
        eq2 = 4 / np.pi**2 * T**3 * (np.sinh((muB - 2 * muS) / T) + np.sinh(muS / T)) - nB * Constants.hbarTimesC**3
        eq3 = 4 / np.pi**2 * T**3 * (2 * np.sinh((muB - 2 * muS) / T) - np.sinh(muS / T)) - nQ * Constants.hbarTimesC**3

        return [eq1, eq2, eq3]


    traj = np.zeros((len(times), 3))

    for i, t in enumerate(times):
        traj[i] = fsolve(eqns, x0 = np.array([0.3, 0.9, 0.3]), args = (e_dens[i], nB_dens[i], nQ_dens[i]), maxfev = 1000)

    traj = traj * Constants.MeVPerGeV
    traj = traj.transpose()

    muQ = traj[1] - 3 * traj[2]

    # Save t, e_dens, T, muB, muS, muQ to data files
    output = np.vstack((times, e_dens, traj, muQ)).transpose()
    outfile = 'results/T-muB-muS-muQ-vs-t.dat'
    myHeader = 't (fm/c), e (GeV/fm^3), T (MeV), muB (MeV), muS (MeV), muQ (MeV)'

    np.savetxt(outfile, output, delimiter = ',', fmt = '%12.3f', header = myHeader)

    # Plot and save a T-muB trajectory figure
    fig, ax = plt.subplots()

    ax.plot(traj[1], traj[0], marker = '.')

    ax.tick_params(axis = 'both', which = 'both', direction = 'in', top = True, right = True)
    ax.grid(alpha = 0.5)

    ax.set_xlim(0)
    ax.set_ylim(0)

    ax.set_xlabel('$\mathrm{\mu}_B$ (MeV)')
    ax.set_ylabel('T (MeV)')

    plt.title('$\sqrt{\mathrm{s_{NN}}}$ = ' + str(sqrtsnn) + ' GeV, Z = ' + str(int(z)) + ', A = ' + str(int(a)) + ', $\\tau_F$ = ' + str(tauf) + ' fm/c, Boltzmann Statistics')

    figfile = 'results/results.pdf'

    plt.tight_layout()
    plt.savefig(figfile)

    return

class NonInteractingMasslessQuantumEOSFullSolution:
    from utilities import CalculationContext, Constants
    from piecewise import PiecewiseSolution

    def __init__(self, cc: CalculationContext, energySoln: PiecewiseSolution, netBaryonSoln: PiecewiseSolution) -> None:
        self.times = energySoln.times

        self.energyDensities = energySoln.densities
        self.netBaryonDensities = netBaryonSoln.densities

        # Define nQ_dens and nS_dens arrays
        self.netChargeDensities = cc.atomicNum / cc.massNum * netBaryonSoln.densities
        self.netStrangenessDensities = np.zeros(len(energySoln.times))
        
        self.thermoVars = np.zeros((len(energySoln.times), 3))

    # Calculate T, muB, muS, and muQ arrays
    def __equations(self, x, ep, nB, nQ):
        temp, muB, muS = x

        eq1 = 19 * Constants.piSquared / 12 * temp**4 + 3 / 2 * ((muB - 2 * muS)**2 + muS**2) * temp**2 + 3 / (4 * Constants.piSquared) * ((muB - 2 * muS)**4 + muS**4) - ep * Constants.hbarTimesC**3
        eq2 = 1 / 3 * (muB - muS) * temp**2 + 1 / (3 * Constants.piSquared) * ((muB - 2 * muS)**3 + muS**3) - nB * Constants.hbarTimesC**3
        eq3 = 1 / 3 * (2 * muB - 5 * muS) * temp**2 + 1 / (3 * Constants.piSquared) * (2 * (muB - 2 * muS)**3 - muS**3) - nQ * Constants.hbarTimesC**3

        return [eq1, eq2, eq3]

    def calculate(self):
        for i in range(self.times.size):
            args = (self.energyDensities[i], self.netBaryonDensities[i], self.netChargeDensities[i])
            self.thermoVars[i] = fsolve(self.__equations, x0 = np.array([0.3, 0.9, 0.3]), args = args, maxfev = 1000)

        self.thermoVars = self.thermoVars * Constants.MeVPerGeV
        self.thermoVars = self.thermoVars.transpose()

        muQ = self.thermoVars[1] - 3 * self.thermoVars[2]

        self.thermoVars = np.insert(self.thermoVars, 2, muQ, axis=0)


class NonInteractingMasslessBoltzmannEOSFullSolution:
    from utilities import CalculationContext, Constants
    from piecewise import PiecewiseSolution

    def __init__(self, cc: CalculationContext, energySoln: PiecewiseSolution, netBaryonSoln: PiecewiseSolution) -> None:
        self.times = energySoln.times

        self.energyDensities = energySoln.densities
        self.netBaryonDensities = netBaryonSoln.densities

        # Define nQ_dens and nS_dens arrays
        self.netChargeDensities = cc.atomicNum / cc.massNum * netBaryonSoln.densities
        self.netStrangenessDensities = np.zeros(len(energySoln.times))
        
        self.thermoVars = np.zeros((len(energySoln.times), 3))

    # Calculate T, muB, muS, and muQ arrays
    def __equations(self, x, ep, nB, nQ):
        temp, muB, muS = x

        eq1 = 12 / Constants.piSquared * temp**4 * (7 + 3 * np.cosh((muB - 2 * muS) / temp) + 3 * np.cosh(muS / temp)) - ep * Constants.hbarTimesC**3
        eq2 = 4 / Constants.piSquared * temp**3 * (np.sinh((muB - 2 * muS) / temp) + np.sinh(muS / temp)) - nB * Constants.hbarTimesC**3
        eq3 = 4 / Constants.piSquared * temp**3 * (2 * np.sinh((muB - 2 * muS) / temp) - np.sinh(muS / temp)) - nQ * Constants.hbarTimesC**3

        return [eq1, eq2, eq3]

    def calculate(self):
        for i in range(self.times.size):
            args = (self.energyDensities[i], self.netBaryonDensities[i], self.netChargeDensities[i])
            self.thermoVars[i] = fsolve(self.__equations, x0 = np.array([0.3, 0.9, 0.3]), args = args, maxfev = 1000)

        self.thermoVars = self.thermoVars * Constants.MeVPerGeV
        self.thermoVars = self.thermoVars.transpose()

        muQ = self.thermoVars[1] - 3 * self.thermoVars[2]

        self.thermoVars = np.insert(self.thermoVars, 2, muQ, axis=0)


if __name__ == "__main__":
    from utilities import CalculationContext
    from differential_density import EnergyDensity, NetBaryonDensity
    from piecewise import PiecewiseSolution

    cc = CalculationContext(79, 197, 200, 0.3, "boltzmann", 10)

    ed = EnergyDensity(cc)
    nd = NetBaryonDensity(cc)

    pse = PiecewiseSolution(cc, ed)
    pse.calculate(cc)
    psn = PiecewiseSolution(cc, nd)
    psn.calculate(cc)

    # eos = NonInteractingMasslessQuantumEOSFullSolution(cc, pse, psn)
    eos = NonInteractingMasslessBoltzmannEOSFullSolution(cc, pse, psn)
    eos.calculate()
    print(eos.thermoVars)