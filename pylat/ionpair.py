"""
Created on Mon Aug  3 10:49:36 2015

@author: mhumbert

PyLAT: Python LAMMPS Analysis Tools
Copyright (C) 2018  Michael Humbert, Yong Zhang and Ed Maginn

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

import copy
import math
import sys
import warnings

import numpy as np
from numba import njit, prange
from scipy.optimize import curve_fit


class ionpair:
    def runionpair(
        self,
        comx,
        comy,
        comz,
        Lx,
        Ly,
        Lz,
        moltypel,
        moltype,
        tsjump,
        dt,
        output,
        ver,
        skipframes,
    ):
        """
        Calculates the ion pair lifetime for all combinations of molecule types
        in the system. An ion pair for types A and B is defined as the closest
        molecule of type B around a molecule of type A

        The integral is fit using multiexponentials from one to five
        exponentials to obtain a good fit without overfitting
        """

        output["Ion_Pair_Lifetime"] = {}
        output["Ion_Pair_Lifetime"]["Units"] = "picoseconds"
        output["Ion_Pair_Lifetime"]["Explanation"] = (
            "The Ion Pair Lifetime correlation function is fit to a single exponential, a double exponential up to 5 exponentials. The results shown are the result of these successive fittings"
        )
        (closest, begin, end, C) = self.init(len(comx[0]), moltypel, len(comx), moltype)
        for step in range(0, len(comx)):
            r = self.calcdistance(comx[step], comy[step], comz[step], Lx, Ly, Lz)
            closest = self.findclosest(r, closest, begin, end, step)
            if ver:
                sys.stdout.write("\rIPL distance calculation {:.2f}% complete".format((step + 1) * 100.0 / len(comx)))

        if ver:
            sys.stdout.write("\n")

        correlation = self.correlation(closest, moltype, moltypel, ver, skipframes)
        if ver:
            print("correlation complete")
        time = []
        for i in range(0, len(correlation)):
            time.append(float(i * tsjump * dt / 1000))
        begin = int(1000 / dt / tsjump)
        end = len(time)
        for i in range(0, len(moltypel)):
            for j in range(0, len(moltypel)):
                if i != j:
                    y = []
                    end = 0
                    for k in range(0, len(correlation)):
                        y.append(float(correlation[k][i][j]))
                        if correlation[k][i][j] <= 0.04 and end == 0:
                            end = k
                    if end == 0:
                        end = len(y)
                    (IPL, r2) = self.curvefit(y, time, begin, end)
                    output["Ion_Pair_Lifetime"]["{0} around {1}".format(moltypel[j], moltypel[i])] = IPL
                    output["Ion_Pair_Lifetime"]["{0} around {1} r2".format(moltypel[j], moltypel[i])] = r2
                    output["Ion_Pair_Lifetime"]["{0} around {1} correlation".format(moltypel[j], moltypel[i])] = (
                        copy.deepcopy(y)
                    )
        output["Ion_Pair_Lifetime"]["Correlation_Time"] = copy.deepcopy(time)

    def init(self, nummol, moltypel, numtimesteps, moltype):
        # initializes arrays for the calculations
        closest = np.zeros((numtimesteps, nummol, len(moltypel)))
        C = np.zeros((len(moltypel), len(moltypel)))
        begin = [0]
        end = []
        for i in range(1, len(moltype)):
            if moltype[i] != moltype[i - 1]:
                end.append(i)
                begin.append(i)

        end.append(len(moltype))
        return (closest, begin, end, C)

    def calcdistance(self, comx, comy, comz, Lx, Ly, Lz):
        # Runs a fortran script calculating the distance between all molecules
        r = calcdistances(len(comx), comx, comy, comz, Lx, Ly, Lz)
        return r

    def findclosest(self, r, closest, begin, end, timestep):
        # Search molecules to find the closest molecules at each timestep
        closest = findclosests(r, closest, begin, end, timestep)
        return closest

    def correlation(self, closest, moltype, moltypel, ver, skipframes):
        # Runs a fortran script perfroming the correlation function
        correlation = ipcorr(
            closest,
            skipframes,
            len(closest),
            len(closest[0]),
            len(closest[0][0]),
            int((len(closest) - skipframes) / 2),
            moltype,
        )
        return correlation

    def curvefit(self, correlation, time, begin, end):
        # Fit the exponential functions to the correlation function to estimate the ion pair lifetime
        funlist = [
            exponential1,
            exponential2,
            exponential3,
            exponential4,
            exponential5,
        ]
        IPL = []
        r2 = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for fun in funlist:
                popt, pcov = curve_fit(fun, time[begin:end], correlation[begin:end], maxfev=100000000)
                fit = []
                for i in time:
                    fit.append(fun(i, *popt))
                yave = np.average(correlation[begin:end])
                SStot = 0
                SSres = 0
                for i in range(begin, end):
                    SStot += (correlation[i] - yave) ** 2
                    SSres += (correlation[i] - fit[i]) ** 2
                r2.append(1 - SSres / SStot)
                IPL.append(0)
                for i in range(0, int(len(popt) / 2)):
                    IPL[-1] += popt[i] * popt[i + int(len(popt) / 2)]
        return (IPL, r2)


@njit
def exponential1(x, A1, B1):
    return A1 * np.exp(-x / B1)


@njit
def exponential2(x, A1, A2, B1, B2):
    return A1 * np.exp(-x / B1) + A2 * np.exp(-x / B2)


@njit
def exponential3(x, A1, A2, A3, B1, B2, B3):
    return A1 * np.exp(-x / B1) + A2 * np.exp(-x / B2) + A3 * np.exp(-x / B3)


@njit
def exponential4(x, A1, A2, A3, A4, B1, B2, B3, B4):
    return A1 * np.exp(-x / B1) + A2 * np.exp(-x / B2) + A3 * np.exp(-x / B3) + A4 * np.exp(-x / B4)


@njit
def exponential5(x, A1, A2, A3, A4, A5, B1, B2, B3, B4, B5):
    return (
        A1 * np.exp(-x / B1) + A2 * np.exp(-x / B2) + A3 * np.exp(-x / B3) + A4 * np.exp(-x / B4) + A5 * np.exp(-x / B5)
    )


@njit
def exponential6(x, A1, A2, A3, A4, A5, A6, B1, B2, B3, B4, B5, B6):
    return (
        A1 * np.exp(-x / B1)
        + A2 * np.exp(-x / B2)
        + A3 * np.exp(-x / B3)
        + A4 * np.exp(-x / B4)
        + A5 * np.exp(-x / B5)
        + A6 * np.exp(-x / B6)
    )


@njit
def calcdistances(nummol, comx, comy, comz, Lx, Ly, Lz):
    r = np.zeros((nummol, nummol))
    for i in range(nummol - 1):
        for j in range(i + 1, nummol):
            dx = comx[i] - comx[j]
            dy = comy[i] - comy[j]
            dz = comz[i] - comz[j]
            dx -= Lx * round(dx / Lx)
            dy -= Ly * round(dy / Ly)
            dz -= Lz * round(dz / Lz)
            distance = math.sqrt(dx**2 + dy**2 + dz**2)
            r[i, j] = r[j, i] = distance
    return r


@njit
def findclosests(r, closest, begin, end, timestep):
    for i in range(len(r)):
        for j in range(len(begin)):
            distance = 10000
            for k in range(begin[j], end[j]):
                if r[i][k] < distance:
                    distance = r[i][k]
                    closest[timestep][i][j] = k
    return closest


@njit(parallel=True)
def ipcorr(closest, skip, L1, L2, L3, CL, moltype):
    correlation = np.zeros((CL + 1, L3, L3))
    for j in prange(L3):
        for i in range(L2):
            for k in range(skip, skip + CL):
                for m in range(k, k + CL + 1):
                    if closest[k, i, j] == closest[m, i, j]:
                        correlation[m - k, moltype[i], j] += 1
    for i in range(L3):
        for j in range(L3):
            norm = correlation[0, i, j]
            correlation[:, i, j] /= norm
    return correlation
