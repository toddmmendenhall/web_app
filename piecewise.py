import numpy as np
from scipy.integrate import dblquad
from density_calc import t1, ta, x1, beta, tmid, x2, t2

def piecewise_solution(integrand, s, t, ra, at, a, tauf) -> np.ndarray:
    startTime = t1(s, ra)
    midTime = tmid(s, ra)
    finalTime = t2(s, ra)
    relativisticVelocity = beta(s)

    if t <= startTime + tauf:
        return np.array([0, 0])

    elif t > startTime + tauf and t < ta(s, ra, tauf):
        
        return np.sum(np.array([dblquad(integrand,
                                        startTime,
                                        x1(s, ra, t, tauf),
                                        lambda x: -relativisticVelocity * (x - startTime),
                                        lambda x: relativisticVelocity * (x - startTime),
                                        args = (s, t, ra, at, a)),
                                dblquad(integrand,
                                        x1(s, ra, t, tauf),
                                        t - tauf,
                                        lambda x: -np.sqrt((t - x)**2 - tauf**2),
                                        lambda x: np.sqrt((t - x)**2 - tauf**2),
                                        args = (s, t, ra, at, a))]).transpose(), axis = 1)

    elif t >= ta(s, ra, tauf) and t < finalTime + tauf:
        return np.sum(np.array([dblquad(integrand,
                                        startTime,
                                        midTime,
                                        lambda x: -relativisticVelocity * (x - startTime),
                                        lambda x: relativisticVelocity * (x - startTime),
                                        args = (s, t, ra, at, a)),
                                dblquad(integrand,
                                        midTime,
                                        x2(s, ra, t, tauf),
                                        lambda x: -relativisticVelocity * (finalTime - x),
                                        lambda x: relativisticVelocity * (finalTime - x),
                                        args = (s, t, ra, at, a)),
                                dblquad(integrand,
                                        x2(s, ra, t, tauf),
                                        t - tauf,
                                        lambda x: -np.sqrt((t - x)**2 - tauf**2),
                                        lambda x: np.sqrt((t - x)**2 - tauf**2),
                                        args = (s, t, ra, at, a))]).transpose(), axis = 1)

    else:
        return np.sum(np.array([dblquad(integrand,
                                        startTime,
                                        midTime,
                                        lambda x: -relativisticVelocity * (x - startTime),
                                        lambda x: relativisticVelocity * (x - startTime),
                                        args = (s, t, ra, at, a)),
                                dblquad(integrand,
                                        midTime,
                                        finalTime,
                                        lambda x: -relativisticVelocity * (finalTime - x),
                                        lambda x: relativisticVelocity * (finalTime - x),
                                        args = (s, t, ra, at, a))]).transpose(), axis = 1)