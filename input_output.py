import numpy as np
import os


class IO:
    from thermo_calc import EOS
    from utilities import CalculationContext

    def __init__(self, eos: EOS):
        self.data = np.vstack([eos.times, eos.energyDensities, eos.netBaryonDensities, *(eos.thermoVars[i] for i in range(4))])
        self.outputDir = os.getcwd() + "/results/"# + "/mysite/v2/results/"
        print("outputdir", self.outputDir)

    def write_output(self):
        header = 't (fm/c), eDens (GeV/fm^3), nbDens (fm^-3), temp (MeV), muB (MeV), muQ (MeV), muS(MeV)'
        outputFile = self.outputDir + 'time_evolution.csv'
        np.savetxt(outputFile, self.data.T, delimiter = ",", fmt = "%10.3e", header = header)

    def make_plots(self, cc:CalculationContext):
        self.__make_density_plot(cc)
        self.__make_thermo_plot(cc)

    def __make_density_plot(self, cc: CalculationContext):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        
        times = self.data[0]
        eDens = self.data[1]

        ax.plot(times, eDens, marker = ".", label = 'Semi-analytical result')
        ax.tick_params(axis = 'both', which = 'both', direction = 'in', top = True, right = True)
        ax.grid(alpha = 0.5)

        ax.set_xlim(0, np.max(times))
        ax.set_ylim(0)

        ax.set_xlabel("t (fm/c)")
        ax.set_ylabel("$\\rm{\epsilon}(t)$ (GeV/fm$^3$)")

        legendTitle = "$\sqrt{s_{\\rm NN}}$ = " + str(cc.comCollisionEnergy) + ' GeV, A = ' + str(int(cc.massNum)) + ', $\\tau_{\\rm F}$ = ' + str(cc.partonFormationTime) + ' fm/c'
        ax.legend(frameon=False, title = legendTitle)

        figfile = self.outputDir + 'e-vs-t.pdf'

        plt.tight_layout()
        plt.savefig(figfile)

    def __make_thermo_plot(self, cc: CalculationContext):
        import matplotlib.pyplot as plt

        # Plot and save a T-muB trajectory figure
        fig, ax = plt.subplots()

        temp = self.data[3]
        muB = self.data[4]

        ax.plot(muB, temp, marker = ".")

        ax.tick_params(axis = 'both', which = 'both', direction = 'in', top = True, right = True)
        ax.grid(alpha = 0.5)

        ax.set_xlim(0)
        ax.set_ylim(0)

        ax.set_xlabel('$\mu_{\\rm B}$ (MeV)')
        ax.set_ylabel('T (MeV)')

        plt.title('$\sqrt{s_{\\rm NN}}$ = ' + str(cc.comCollisionEnergy) + ' GeV, Z = ' + str(int(cc.atomicNum)) + ', A = ' + str(int(cc.massNum)) + ', $\\tau_{\\rm F}$ = ' + str(cc.partonFormationTime) + ' fm/c, Quantum Statistics')

        figfile = self.outputDir + 'phase_diagram_trajectory.pdf'

        plt.tight_layout()
        plt.savefig(figfile)

if __name__ == "__main__":
    from utilities import CalculationContext
    from differential_density import EnergyDensity, NetBaryonDensity
    from piecewise import PiecewiseSolution
    from thermo_calc import NonInteractingMasslessBoltzmannEOSFullSolution

    cc = CalculationContext(79, 197, 200, 0.3, "boltzmann", 50)

    ed = EnergyDensity(cc)
    nd = NetBaryonDensity(cc)

    pse = PiecewiseSolution(cc, ed)
    pse.calculate(cc)
    psn = PiecewiseSolution(cc, nd)
    psn.calculate(cc)

    eos = NonInteractingMasslessBoltzmannEOSFullSolution(cc, pse, psn)
    eos.calculate()

    io = IO(eos)
    io.write_output()
    io.make_plots(cc)