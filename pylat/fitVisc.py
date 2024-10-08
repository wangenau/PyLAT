"""Created on Wed May 18 16:05:34 2016

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

import numpy as np
from numba import njit
from scipy import optimize


class fitVisc:
    def fitvisc(self, time, visc, stddev, plot, popt2, i):
        # Makre sure our time scale starts at zero for the fit
        time = time - 2 * time[0] + time[1]

        foundcutoff = False
        foundstart = False
        start = 1
        while not foundstart and start < len(visc):
            if time[start] > 2000:
                foundstart = True
            else:
                start += 1
        cut = 1
        while not foundcutoff and cut < len(visc):
            if stddev[cut] > 0.4 * visc[cut]:
                foundcutoff = True
            else:
                cut += 1

        if np.any(stddev < 1e-15):
            stddev = None
        else:
            stddev = stddev[start:cut]
        popt2, pcov2 = optimize.curve_fit(
            doubexp,
            time[start:cut],
            visc[start:cut],
            maxfev=1000000,
            p0=popt2,
            sigma=stddev,
            bounds=(0, [np.inf, 1, np.inf, np.inf]),
        )

        fit = []
        fit1 = []
        fit2 = []
        for t in time:
            fit.append(doubexp(t, *popt2))
            fit1.append(doubexp1(t, *popt2))
            fit2.append(doubexp2(t, *popt2))
        Value = popt2[0] * popt2[1] * popt2[2] + popt2[0] * (1 - popt2[1]) * popt2[3]

        if plot:
            timep = time / 1000000
            from matplotlib import pyplot as plt
            from matplotlib import rcParams

            rcParams.update({"font.size": 14})
            print("Viscosity estimate is {}".format(Value))
            print("A={}, alpha={}, tau1={}, tau2={}".format(popt2[0], popt2[1], popt2[2], popt2[3]))
            print("Time cutoff is {}".format(time[cut - 1]))
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            plt.plot(timep[: len(visc)], visc, label="Viscosity")
            plt.plot(timep[: len(fit)], fit, label="Double Exponential fit")
            plt.plot(timep[: len(fit1)], fit1, label=r"Contribution of $\tau_1$")
            plt.plot(timep[: len(fit2)], fit2, label=r"Contribution of $\tau_2$")
            plt.axvline(timep[cut - 1])
            plt.ylabel("Viscosity (mPa*s)")
            plt.xlabel("Time (ns)")
            plt.legend()
            if isinstance(plot, str):
                plt.savefig("{}/viscosity_{}.png".format(plot, i + 1))
            else:
                plt.show()
            plt.close()
        return Value


@njit
def doubexp(x, A, alpha, tau1, tau2):
    return A * alpha * tau1 * (1 - np.exp(-x / tau1)) + A * (1 - alpha) * tau2 * (1 - np.exp(-x / tau2))


@njit
def doubexp1(x, A, alpha, tau1, tau2):
    return A * alpha * tau1 * (1 - np.exp(-x / tau1))


@njit
def doubexp2(x, A, alpha, tau1, tau2):
    return A * (1 - alpha) * tau2 * (1 - np.exp(-x / tau2))


@njit
def singleexp(x, A, tau):
    return A * (1 - np.exp(-x / tau))
