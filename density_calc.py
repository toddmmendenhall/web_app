import numpy as np
from scipy.optimize import fsolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utilities import Constants

################################################################################
# Define functions of beam energy

def ycm(s):
    return np.arccosh(s / (2 * Constants.nucleonMass))

def beta(s):
    return np.tanh(ycm(s))

def dt(s, ra):
    return 2 * ra / np.sinh(ycm(s))

def t1(s, ra):
    return dt(s, ra) / 6

def t2(s, ra):
    return 5 * dt(s, ra) / 6

def t21(s, ra):
    return t2(s, ra) - t1(s, ra)

def tmid(s, ra):
    return (t1(s, ra) + t2(s, ra)) / 2

################################################################################
# Define functions for integrand

def y0(z0, x, t):
    return np.arctanh(-z0 / (t - x))

def detdy0old(s, a):
    return 0.456 * 2 * a * np.log(s / 2.35)

def detdy0(s, a):
    if (s <= 20.7):
        return 1.25 * 0.308 * 2 * a * np.log(s / (2 * Constants.nucleonMass))**1.08
    else:
        return detdy0old(s, a)

def sigmab(s):
    return 0.601 * (s - 2 * Constants.nucleonMass)**0.121 * np.log(s / (2 * Constants.nucleonMass))**0.241

def yb(s):
    return 0.541 * (s - 2 * Constants.nucleonMass)**0.196 * np.log(s / (2 * Constants.nucleonMass))**0.392

def sigmahad(s, a):
    func = lambda x, s, a: a * s - detdy0(s, a) * np.exp(x**2 / 2) * np.sqrt(2 * np.pi) * x - Constants.nucleonMass * 2 * a * np.exp(sigmab(s)**2 / 2) * np.cosh(yb(s))
    return fsolve(func, 0, args = (s, a))

def detdy(y, s, a):
    return detdy0(s, a) * np.exp(-(y / sigmahad(s, a))**2 / 2)

def dnbdy0(s, a):
    return 2 * a / (np.sqrt(2 * np.pi) * sigmab(s)) * np.exp(-(yb(s) / sigmab(s))**2 / 2)

def dnbdy(y, s, a):
    return dnbdy0(s, a) * 0.5 * np.exp((yb(s) / sigmab(s))**2 / 2) * (np.exp(-((y + yb(s)) / sigmab(s))**2 / 2) + np.exp(-((y - yb(s)) / sigmab(s))**2 / 2))

def dmtdyhad(y, s, a):
    return detdy(y, s, a) + Constants.nucleonMass * dnbdy(y, s, a)

def ep_integrand(z0, x, s, t, ra, at, a):
    return 2 / (at * beta(s) * t21(s, ra)**2) / (t - x) * dmtdyhad(y0(z0, x, t), s, a) * np.cosh(y0(z0, x, t))**3

def nb_integrand(z0, x, s, t, ra, at, a):
    return 2 / (at * beta(s) * t21(s, ra)**2) / (t - x) * dnbdy(y0(z0, x, t), s, a) * np.cosh(y0(z0, x, t))**2

################################################################################
# Define functions for limits of integration

def ta(s, ra, tauf):
    return tmid(s, ra) + np.sqrt(tauf**2 + (beta(s) * t21(s, ra) / 2)**2)

def x1(s, ra, t, tauf):
    return (t - beta(s)**2 * t1(s, ra) - np.sqrt(beta(s)**2 * ((t - t1(s, ra))**2 - tauf**2) + tauf**2)) / (1 - beta(s)**2)

def x2(s, ra, t, tauf):
    return (t - beta(s)**2 * t2(s, ra) - np.sqrt(beta(s)**2 * ((t - t2(s, ra))**2 - tauf**2) + tauf**2)) / (1 - beta(s)**2)

################################################################################
# Definition of energy density

def efullz(a, sqrtsnn, tauf, ntimes):
    from piecewise import piecewise_solution

    # Define parameters

    ra = Constants.nucleonRadius * a**(1 / 3)

    at = np.pi * ra**2

    # Calculate epsilon(t)

    # Set time array
    timesMin = 1.1 * t1(sqrtsnn, ra) + tauf
    timesMax = 3 * t2(sqrtsnn, ra) + tauf
    if(timesMax < 10):
        timesMax = 10
    times = np.logspace(np.log10(timesMin), np.log10(timesMax), ntimes)
    times = np.insert(times, 0, 0)

    # Declare empty arrays for densities and integration errors
    ep_dens = np.zeros(len(times))

    # Populate densities and errors arrays
    for i, t in enumerate(times):
        x = piecewise_solution(ep_integrand, sqrtsnn, times[i], ra, at, a, tauf)
        ep_dens[i] = x[0]

    # Write to file
    output = np.array([times, ep_dens]).transpose()

    myHeader = 'time (fm/c), energy density (GeV/fm^3)'

    outfile = 'results/e-dens-vs-t.dat'

    np.savetxt(outfile, output, delimiter = ',', fmt='%.4f', header = myHeader)

    # Make plot and save to file
    fig, ax = plt.subplots()

    plt.plot(times, ep_dens, marker = '.')

    plt.xlim(0, timesMax)
    plt.ylim(0)

    plt.xlabel('t (fm/c)')
    plt.ylabel('$\mathrm{\epsilon}$(t) (GeV/fm$^3$)')
    plt.title('$\sqrt{\mathrm{s_{NN}}}$ = ' + str(sqrtsnn) + ' GeV, A = ' \
        + str(int(a)) + ', $\\tau_F$ = ' + str(tauf) + ' fm/c')

    leglist = ['Semi-analytical result']

    plt.legend(leglist, frameon=False)

    figfile = 'results/e-vs-t.pdf'

    plt.tight_layout()
    plt.savefig(figfile)

    return

################################################################################
# Definition of net-Baryon density

def nbfullz(a, sqrtsnn, tauf, ntimes):
    from piecewise import piecewise_solution

    # Define parameters

    ra = Constants.nucleonRadius * a**(1 / 3)

    at = np.pi * ra**2

    # Calculate epsilon(t)

    # Set time array
    timesMin = 1.1 * t1(sqrtsnn, ra) + tauf
    timesMax = 3 * t2(sqrtsnn, ra) + tauf
    if(timesMax < 10):
        timesMax = 10
    times = np.logspace(np.log10(timesMin), np.log10(timesMax), ntimes)
    times = np.insert(times, 0, 0)

    # Declare empty arrays for densities and integration errors
    nb_dens = np.zeros(len(times))

    # Populate densities and errors arrays
    for i, t in enumerate(times):
        x = piecewise_solution(nb_integrand, sqrtsnn, times[i], ra, at, a, tauf)
        nb_dens[i] = x[0]

    # Write to file
    output = np.array([times, nb_dens]).transpose()

    myHeader = 'time (fm/c), net-Baryon density (GeV/fm^3)'

    outfile = 'results/nB-dens-vs-t.dat'

    np.savetxt(outfile, output, delimiter = ',', fmt='%.4f', header = myHeader)

    return
