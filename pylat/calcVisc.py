"""Created on Fri Dec 11 09:16:20 2015

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

import random
import sys

import numpy as np

from .fitVisc import fitVisc
from .viscio import LammpsLog


class calcVisc:
    def calcvisc(
        self,
        numtrj,
        numskip,
        dirbase,
        logname,
        output,
        ver,
        numsamples,
        numboot,
        plot,
        popt2,
    ):
        """Calculates average and standard deviation of the integral of the
        pressure tensor autocorrelation function over numtrj lammps trajectories

        """
        folders = None
        if isinstance(numtrj, (list, tuple)):
            folders = numtrj
            numtrj = len(folders)

        output["Viscosity"] = {}
        output["Viscosity"]["Units"] = "cP"
        if dirbase is None:
            dirbase = "./"
        if folders is None:
            filename = dirbase + "1/" + logname
        else:
            filename = dirbase + str(folders[0]) + "/" + logname
        Log = LammpsLog.from_file(filename)
        (Time, visco) = Log.viscosity(numskip)
        trjlen = len(Time)
        viscosity = np.zeros((numtrj, trjlen))
        for i in range(len(visco)):
            viscosity[0][i] += visco[i]
        if ver >= 1:
            sys.stdout.write("Viscosity Trajectory 1 of {} complete".format(numtrj))

        for i in range(2, numtrj + 1):
            if folders is None:
                filename = dirbase + str(i) + "/" + logname
            else:
                filename = dirbase + str(folders[i - 1]) + "/" + logname
            Log = LammpsLog.from_file(filename)
            (Time, visco) = Log.viscosity(numskip)
            if len(visco) < trjlen:
                trjlen = len(visco)
            for j in range(trjlen):
                viscosity[i - 1][j] += visco[j]
            if ver >= 1:
                sys.stdout.write("\rViscosity Trajectory {} of {} complete".format(i, numtrj))
        if ver >= 1:
            sys.stdout.write("\n")

        # Begin Bootstrapping for error estimate
        Values = []
        fv = fitVisc()
        # random.seed(123456789)
        for i in range(numboot):
            Values.append(self.Bootstrap(numsamples, trjlen, numtrj, viscosity, Time, fv, plot, popt2, i))
            if ver > 1:
                sys.stdout.write("\rViscosity Bootstrap {} of {} complete".format(i + 1, numboot))
        if ver > 1:
            sys.stdout.write("\n")

        (ave, stddev, Values) = self.getAverage(Values, numsamples, trjlen, numtrj, viscosity, Time, fv)

        output["Viscosity"]["Average Value"] = ave
        output["Viscosity"]["Standard Deviation"] = stddev
        return output

    def getAverage(self, Values, numsamples, trjlen, numtrj, viscosity, Time, fv):
        # calculate average and standard deviation of Values array
        # Was originally implemented to perform a z-test on the values to determine outliers
        ave = np.average(Values)
        stddev = np.std(Values)
        return (ave, stddev, Values)

    def Bootstrap(self, numsamples, trjlen, numtrj, viscosity, Time, fv, plot, popt2, i):
        # Perform calculate the viscosity of one bootstrapping sample
        Bootlist = np.zeros((numsamples, trjlen))
        for j in range(numsamples):
            rint = random.randint(0, numtrj - 1)
            for k in range(trjlen):
                Bootlist[j][k] = viscosity[rint][k]
        average = np.zeros(trjlen)
        stddev = np.zeros(trjlen)
        for j in range(trjlen):
            average[j] = np.average(Bootlist.transpose()[j])
            stddev[j] = np.std(Bootlist.transpose()[j])
        return fv.fitvisc(Time, average, stddev, plot, popt2, i)
