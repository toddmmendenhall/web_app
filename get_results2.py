#!/usr/bin/python3

import numpy as np
from scipy.special import lambertw
from scipy.integrate import dblquad
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy.optimize import fsolve
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

def ep_pieces(s, t, ra, at, a, tauf):
    if t <= t1(s, ra) + tauf:
        return np.array([0, 0])
    elif t > t1(s, ra) + tauf and t < ta(s, ra, tauf):
        return np.sum(np.array([dblquad(ep_integrand,
                                        t1(s, ra),
                                        x1(s, ra, t, tauf),
                                        lambda x: -beta(s) * (x - t1(s, ra)),
                                        lambda x: beta(s) * (x - t1(s, ra)),
                                        args = (s, t, ra, at, a)),
                                dblquad(ep_integrand,
                                        x1(s, ra, t, tauf),
                                        t - tauf,
                                        lambda x: -np.sqrt((t - x)**2 - tauf**2),
                                        lambda x: np.sqrt((t - x)**2 - tauf**2),
                                        args = (s, t, ra, at, a))]).transpose(), axis = 1)
    elif t >= ta(s, ra, tauf) and t < t2(s, ra) + tauf:
        return np.sum(np.array([dblquad(ep_integrand,
                                        t1(s, ra),
                                        tmid(s, ra),
                                        lambda x: -beta(s) * (x - t1(s, ra)),
                                        lambda x: beta(s) * (x - t1(s, ra)),
                                        args = (s, t, ra, at, a)),
                                dblquad(ep_integrand,
                                        tmid(s, ra),
                                        x2(s, ra, t, tauf),
                                        lambda x: -beta(s) * (t2(s, ra) - x),
                                        lambda x: beta(s) * (t2(s, ra) - x),
                                        args = (s, t, ra, at, a)),
                                dblquad(ep_integrand,
                                        x2(s, ra, t, tauf),
                                        t - tauf,
                                        lambda x: -np.sqrt((t - x)**2 - tauf**2),
                                        lambda x: np.sqrt((t - x)**2 - tauf**2),
                                        args = (s, t, ra, at, a))]).transpose(), axis = 1)
    else:
        return np.sum(np.array([dblquad(ep_integrand,
                                        t1(s, ra),
                                        tmid(s, ra),
                                        lambda x: -beta(s) * (x - t1(s, ra)),
                                        lambda x: beta(s) * (x - t1(s, ra)),
                                        args = (s, t, ra, at, a)),
                                dblquad(ep_integrand,
                                        tmid(s, ra),
                                        t2(s, ra),
                                        lambda x: -beta(s) * (t2(s, ra) - x),
                                        lambda x: beta(s) * (t2(s, ra) - x),
                                        args = (s, t, ra, at, a))]).transpose(), axis = 1)

def efullz(a, sqrtsnn, tauf, ntimes):
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
        x = ep_pieces(sqrtsnn, times[i], ra, at, a, tauf)
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

def nb_pieces(s, t, ra, at, a, tauf):
    if t <= t1(s, ra) + tauf:
        return np.array([0, 0])
    elif t > t1(s, ra) + tauf and t < ta(s, ra, tauf):
        return np.sum(np.array([dblquad(nb_integrand,
                                        t1(s, ra),
                                        x1(s, ra, t, tauf),
                                        lambda x: -beta(s) * (x - t1(s, ra)),
                                        lambda x: beta(s) * (x - t1(s, ra)),
                                        args = (s, t, ra, at, a)),
                                dblquad(nb_integrand,
                                        x1(s, ra, t, tauf),
                                        t - tauf,
                                        lambda x: -np.sqrt((t - x)**2 - tauf**2),
                                        lambda x: np.sqrt((t - x)**2 - tauf**2),
                                        args = (s, t, ra, at, a))]).transpose(), axis = 1)
    elif t >= ta(s, ra, tauf) and t < t2(s, ra) + tauf:
        return np.sum(np.array([dblquad(nb_integrand,
                                        t1(s, ra),
                                        tmid(s, ra),
                                        lambda x: -beta(s) * (x - t1(s, ra)),
                                        lambda x: beta(s) * (x - t1(s, ra)),
                                        args = (s, t, ra, at, a)),
                                dblquad(nb_integrand,
                                        tmid(s, ra),
                                        x2(s, ra, t, tauf),
                                        lambda x: -beta(s) * (t2(s, ra) - x),
                                        lambda x: beta(s) * (t2(s, ra) - x),
                                        args = (s, t, ra, at, a)),
                                dblquad(nb_integrand,
                                        x2(s, ra, t, tauf),
                                        t - tauf,
                                        lambda x: -np.sqrt((t - x)**2 - tauf**2),
                                        lambda x: np.sqrt((t - x)**2 - tauf**2),
                                        args = (s, t, ra, at, a))]).transpose(), axis = 1)
    else:
        return np.sum(np.array([dblquad(nb_integrand,
                                        t1(s, ra),
                                        tmid(s, ra),
                                        lambda x: -beta(s) * (x - t1(s, ra)),
                                        lambda x: beta(s) * (x - t1(s, ra)),
                                        args = (s, t, ra, at, a)),
                                dblquad(nb_integrand,
                                        tmid(s, ra),
                                        t2(s, ra),
                                        lambda x: -beta(s) * (t2(s, ra) - x),
                                        lambda x: beta(s) * (t2(s, ra) - x),
                                        args = (s, t, ra, at, a))]).transpose(), axis = 1)

def nbfullz(a, sqrtsnn, tauf, ntimes):
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
        x = nb_pieces(sqrtsnn, times[i], ra, at, a, tauf)
        nb_dens[i] = x[0]

    # Write to file
    output = np.array([times, nb_dens]).transpose()

    myHeader = 'time (fm/c), net-Baryon density (GeV/fm^3)'

    outfile = 'results/nB-dens-vs-t.dat'

    np.savetxt(outfile, output, delimiter = ',', fmt='%.4f', header = myHeader)

    return

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
