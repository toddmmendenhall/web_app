#!/usr/bin/python3

import numpy as np
from scipy.special import lambertw
from scipy.integrate import dblquad
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
import os

########################################################################
# Remove old data files
# os.remove('static/results.dat')
# os.remove('static/results.pdf')

########################################################################
# Define functions of beam energy

# Rapidity of the cm frame
def ycm(s, mn):
    return np.arccosh(s / (2. * mn))

# beta
def beta(s, mn):
    return np.tanh(ycm(s, mn))

# Crossing time
def dt(s, mn, ra):
    return 2. * ra / np.sinh(ycm(s, mn))

# t1
def t1(s, mn, ra):
    return 0.2 * dt(s, mn, ra)

# t2
def t2(s, mn, ra):
    return 0.8 * dt(s, mn, ra)

# t21
def t21(s, mn, ra):
    return t2(s, mn, ra) - t1(s, mn, ra)

# tmid
def tmid(s, mn, ra):
    return (t1(s, mn, ra) + t2(s, mn, ra)) / 2.

########################################################################
# Define functions for integrand

# g(z_0,x)
def g(s, mn, ra):
    return 2. / (beta(s, mn) * t21(s, mn, ra) ** 2.)

# dmtdy0
def dmtdy0(s):
    return 168. * (s - 0.93) ** 0.348

# Gaussian width sigma
def sigma(s, a):
    return np.real(np.sqrt(lambertw((a * s / (np.sqrt(2. * np.pi) * \
    dmtdy0(s))) ** 2.)))

# Rapidity y0 of parton formed at (z0,x) that contributes to the energy
# density between -d,d of z = 0 at time t
def y0(z0, x, t):
    return np.arctanh(-z0 / (t - x))

# Full integrand - equation 19 in paper
def integrand(z0, x, s, t, mn, ra, at, a):
    return g(s, mn, ra) / at * dmtdy0(s) / (t - x) * np.exp(-y0(z0, x, t) ** 2. \
        / (2. * sigma(s, a) ** 2.)) * np.cosh(y0(z0, x, t)) ** 3.

########################################################################
# Define functions for limits of integration

# ta
def ta(s, mn, ra, tauf):
    return tmid(s, mn, ra) + np.sqrt(tauf ** 2. + (beta(s, mn) * t21(s, mn, ra) \
    / 2.) ** 2.)

# x1
def x1(s, t, mn, ra, tauf):
    return (t - beta(s, mn) ** 2. * t1(s, mn, ra) - np.sqrt(beta(s, mn) ** 2. \
        * ((t - t1(s, mn, ra)) ** 2. - tauf ** 2.) + tauf ** 2.)) \
        / (1. - beta(s, mn) ** 2.)

# x2
def x2(s, t, mn, ra, tauf):
    return (t - beta(s, mn) ** 2. * t2(s, mn, ra) - np.sqrt(beta(s, mn) ** 2. \
        * ((t - t2(s, mn, ra)) ** 2. - tauf ** 2.) + tauf ** 2.)) \
        / (1. - beta(s, mn) ** 2.)

########################################################################
# Definition of epsilon(t)
def epsilon(s, t, mn, ra, at, a, tauf):
    if t < t1(s, mn, ra) + tauf:
        return np.array([0., 0.])
    elif t >= t1(s, mn, ra) + tauf and t < ta(s, mn, ra, tauf):
        return np.sum(np.array(
            [
            dblquad(integrand, \
                t1(s, mn, ra), \
                x1(s, t, mn, ra, tauf), \
                lambda x: -beta(s, mn) * (x - t1(s, mn, ra)), \
                lambda x: beta(s, mn) * (x - t1(s, mn, ra)), \
                args = (s, t, mn, ra, at, a)), \
            dblquad(integrand, \
                x1(s, t, mn, ra, tauf), \
                t - tauf, \
                lambda x: -np.sqrt((t - x) ** 2. - tauf ** 2.), \
                lambda x: np.sqrt((t - x) ** 2. - tauf ** 2.),\
                args = (s, t, mn, ra, at, a))
            ]
        ).transpose(), axis = 1)
    elif t >= ta(s, mn, ra, tauf) and t < t2(s, mn, ra) + tauf:
        return np.sum(np.array(
            [
            dblquad(integrand, \
                t1(s, mn, ra), \
                tmid(s, mn, ra), \
                lambda x: -beta(s, mn) * (x - t1(s, mn, ra)), \
                lambda x: beta(s, mn) * (x - t1(s, mn, ra)), \
                args = (s, t, mn, ra, at, a)), \
            dblquad(integrand, \
                tmid(s, mn, ra), \
                x2(s, t, mn, ra, tauf), \
                lambda x: -beta(s, mn) * (t2(s, mn, ra) - x), \
                lambda x: beta(s, mn) * (t2(s, mn, ra) - x), \
                args = (s, t, mn, ra, at, a)), \
            dblquad(integrand, \
                x2(s, t, mn, ra, tauf), \
                t - tauf, \
                lambda x: -np.sqrt((t - x) ** 2. - tauf ** 2.), \
                lambda x: np.sqrt((t - x) ** 2. - tauf ** 2.),\
                args = (s, t, mn, ra, at, a))
            ]
        ).transpose(), axis = 1)
    else:
        return np.sum(np.array(
        [
        dblquad(integrand, \
            t1(s, mn, ra), \
            tmid(s, mn, ra), \
            lambda x: -beta(s, mn) * (x - t1(s, mn, ra)), \
            lambda x: beta(s, mn) * (x - t1(s, mn, ra)), \
            args = (s, t, mn, ra, at, a)), \
        dblquad(integrand, \
            tmid(s, mn, ra), \
            t2(s, mn, ra), \
            lambda x: -beta(s, mn) * (t2(s, mn, ra) - x), \
            lambda x: beta(s, mn) * (t2(s, mn, ra) - x), \
            args = (s, t, mn, ra, at, a))
        ]
        ).transpose(), axis = 1)

########################################################################

def energyDensity(a, sqrtsnn, tauf, ntimes):
    # Define parameters

    # Nucleon mass
    mn = 0.94

    # Nucleon radius
    r0 = 1.12

    # Nucleus radius
    ra = r0 * a ** (1. / 3.)

    # Nucleus transverse area
    at = np.pi * ra ** 2.

    # Calculate epsilon(t)

    # Set time array for calculating densities
    timesMax = 3 * t2(sqrtsnn, mn, ra) + tauf
    if(timesMax < 1):
        timesMax = 1
    times = np.linspace(0, timesMax, ntimes)

    # Declare empty arrays for densities and integration errors
    densities = np.zeros(ntimes)
    # errors = np.zeros(ntimes)

    # Populate densities and errors arrays
    for i in range(ntimes):
        x = epsilon(sqrtsnn, times[i], mn, ra, at, a, tauf)
        densities[i] = x[0]
        # errors[i] = x[1]
        # print(x)

    # Calculate interpolation function of epsilon vs time

    # Time array to evaluate the interperolation and derivative
    tint = np.linspace(ta(sqrtsnn, mn, ra, tauf), t2(sqrtsnn, mn, ra) + tauf, ntimes)

    # Quartic interpolation spline using epsilon(t)
    f = InterpolatedUnivariateSpline(times, densities, k = 4)

    # Roots method only applies to cubic splines! So we take derivative of f
    critpoints = f.derivative().roots()

    # Also include end points in array of critical points
    critpoints = np.append(critpoints, (tint[0], tint[-1]))

    # Values of critical points
    critvals = f(critpoints)

    # Maximum value of critical points array
    maxindx = np.argmax(critvals)
    tmax = critpoints[maxindx]
    emax = critvals[maxindx]

    # Write to file

    # output = np.array([times, densities, errors]).transpose()
    output = np.array([times, densities]).transpose()

    # myheader = 'tmax = ' + str(tmax) + ', emax = ' + str(emax) + '\n \
    #     time (fm/c), epsilon (GeV/fm^3), total integration error'
    myHeader = 'time (fm/c), energy density (GeV/fm^3)'

    # datdir = 'static/'
    #
    # outfile = datdir + 'e-vs-t-' + str(sqrtsnn) + '-GeV-central-A=' \
    #     + str(a) + '-tauF=' + str(tauf) + '.dat'

    outfile = '/home/toddmmendenhall/mysite/energy_density/results/results.dat'
    # outfile = str(os.getcwd())

    # np.savetxt(outfile, output, delimiter = ',', header = myheader)
    np.savetxt(outfile, output, delimiter = ',', fmt='%.4f', header = myHeader)

    # Make plot and save to file
    fig, ax = plt.subplots()
    # plt.plot(times, densities, '.', tint, f(tint), '-')
    plt.plot(times, densities, marker = '.')

    plt.axhline(emax, c = 'k', ls = ':')
    plt.axvline(tmax, c = 'k', ls = ':')

    # plt.xlim(0,6 * t2(sqrtsnn, mn, ra) + tauf)
    plt.xlim(0, timesMax)
    plt.ylim(bottom=0)

    plt.xlabel('t (fm/c)')
    plt.ylabel('$\mathrm{\epsilon}$(t) (GeV/fm$^3$)')
    plt.title('$\sqrt{\mathrm{s_{NN}}}$ = ' + str(sqrtsnn) + ' GeV, A = ' \
        + str(int(a)) + ', $\\tau_F$ = ' + str(tauf) + ' fm/c')

    leglist = [
        'Semi-analytical result', \
        # 'Interpolation', \
        '$\mathrm{\epsilon}^{max}$ = ' + str(np.round(emax, decimals = 2)) \
            + ' GeV/fm$^3$', \
        't$_{max}$ = ' + str(np.round(tmax, decimals = 2)) + ' fm/c']

    plt.legend(leglist, frameon = False)

    # figdir = 'figures/'
    #
    # figfile = figdir + 'e-vs-t-' + str(sqrtsnn) + '-GeV-central-A=' \
    #     + str(a) + '-tauF=' + str(tauf) + '.pdf'

    figfile = '/home/toddmmendenhall/mysite/energy_density/results/results.pdf'

    plt.tight_layout()
    plt.savefig(figfile)


    return
