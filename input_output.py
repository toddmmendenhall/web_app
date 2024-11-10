from calc_profile import CalcProfile
from densities.differential_density import EnergyDensity, NetBaryonDensity
from eos.equation_of_state import EquationOfState
import numpy as np
from matplotlib.figure import Figure
import os


class IO:
    def __init__(self, cp: CalcProfile, e: EnergyDensity, nB: NetBaryonDensity, eos: EquationOfState, isOffline: bool) -> None:
        self.data = np.vstack([cp.times, e.densities, nB.densities, *(eos.get_data())])

        if isOffline:
            self.outputDir = os.getcwd() + "/results/"
        else:
            self.outputDir = os.getcwd() + "/mysite/v2/results/"

        self.write_output()
        self.__make_plots(cp)

    def write_output(self):
        header = '   t (fm/c), e (GeV/fm^3),   nB (fm^-3),   temp (MeV),    muB (MeV),    muQ (MeV),    muS (MeV)'
        outputFile = self.outputDir + 'time_evolution.csv'
        np.savetxt(outputFile, self.data.T, delimiter = ",", fmt = "%13.3e", header = header)

    def __make_plots(self, cp: CalcProfile):
        self.__make_density_plot(cp)
        self.__make_thermo_plot(cp)

    def __make_density_plot(self, cp: CalcProfile):
        fig = Figure()
        ax = fig.subplots()

        times = self.data[0]
        eDens = self.data[1]

        ax.plot(times, eDens, marker = ".", label = 'Semi-analytical result')
        ax.tick_params(axis = 'both', which = 'both', direction = 'in', top = True, right = True)
        ax.grid(alpha = 0.5)

        ax.set_xlim(0, np.max(times))
        ax.set_ylim(0)

        ax.set_xlabel("t (fm/c)")
        ax.set_ylabel("$\\rm{\epsilon}(t)$ (GeV/fm$^3$)")

        legendTitle = "$\sqrt{s_{\\rm NN}}$ = " + str(cp.s) + ' GeV, A = ' + str(int(cp.a)) + ', $\\tau_{\\rm F}$ = ' + str(cp.tform) + ' fm/c'
        ax.legend(frameon=False, title = legendTitle)

        figfile = self.outputDir + 'e-vs-t.png'

        fig.tight_layout()
        fig.savefig(figfile)

    def __make_thermo_plot(self, cp: CalcProfile):
        # Plot and save a T-muB trajectory figure
        fig = Figure()
        ax = fig.subplots()

        temp = self.data[3]
        muB = self.data[4]

        ax.plot(muB, temp, marker=".")

        ax.tick_params(axis='both', which='both',
                       direction='in', top=True, right=True)
        ax.grid(alpha=0.5)

        ax.set_xlim(0, 1200)
        ax.set_ylim(0, 400)

        ax.set_xlabel('$\mu_{\\rm B}$ (MeV)')
        ax.set_ylabel('T (MeV)')

        legendTitle = f'{(cp.eosType)} EoS' 
        ax.legend(frameon=False, title = legendTitle)

        fig.tight_layout()
        fig.savefig(figfile)


if __name__ == "__main__":
    from utilities import CalculationContext
    from densities.differential_density import EnergyDensity, NetBaryonDensity
    from densities.piecewise import PiecewiseSolution
    from eos.thermo_calc import NonInteractingMasslessBoltzmannEOSFullSolution

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
