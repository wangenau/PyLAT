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
        output["Viscosity"]["units"] = "cP"
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
        viscosity[0][: len(visco)] += visco[: len(visco)]
        if ver >= 1:
            sys.stdout.write(f"Viscosity Trajectory 1 of {numtrj} complete")

        for i in range(2, numtrj + 1):
            if folders is None:
                filename = dirbase + str(i) + "/" + logname
            else:
                filename = dirbase + str(folders[i - 1]) + "/" + logname
            Log = LammpsLog.from_file(filename)
            (Time, visco) = Log.viscosity(numskip)
            if len(visco) < trjlen:
                trjlen = len(visco)
            viscosity[i - 1][:trjlen] += visco[:trjlen]
            if ver >= 1:
                sys.stdout.write(f"\rViscosity Trajectory {i} of {numtrj} complete")
        if ver >= 1:
            sys.stdout.write("\n")

        fv = fitVisc()
        if numsamples == 0 or numboot == 0:
            average = np.average(viscosity, axis=0)
            stddev = np.std(viscosity, axis=0)
            ave, stddev = fv.fitvisc(Time, average, stddev, plot, popt2, "avg", ver, average=True)
        else:
            # Begin Bootstrapping for error estimate
            Values = []
            random.seed(123456789)
            for i in range(numboot):
                Values.append(self.Bootstrap(numsamples, trjlen, numtrj, viscosity, Time, fv, plot, popt2, i, ver))
                if ver >= 1:
                    sys.stdout.write(f"\rViscosity Bootstrap {i + 1} of {numboot} complete")
            if ver >= 1:
                sys.stdout.write("\n")
            (ave, stddev, Values) = self.getAverage(Values)

        output["Viscosity"]["Average Value"] = ave
        output["Viscosity"]["Standard Deviation"] = stddev
        return output

    def getAverage(self, Values):
        # calculate average and standard deviation of Values array
        # Was originally implemented to perform a z-test on the values to determine outliers
        ave = np.average(Values)
        stddev = np.std(Values)
        return (ave, stddev, Values)

    def Bootstrap(self, numsamples, trjlen, numtrj, viscosity, Time, fv, plot, popt2, i, ver):
        # Perform calculate the viscosity of one bootstrapping sample
        Bootlist = np.zeros((numsamples, trjlen))
        for j in range(numsamples):
            rint = random.randint(0, numtrj - 1)
            Bootlist[j, :] = viscosity[rint, :]
        average = np.average(Bootlist, axis=0)
        stddev = np.std(Bootlist, axis=0)
        return fv.fitvisc(Time, average, stddev, plot, popt2, i, ver)[0]
