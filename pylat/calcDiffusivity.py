"""Created on Wed May 27 16:05:33 2015

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
import warnings

import numpy as np
from scipy import stats


class calcdiffusivity:
    def calcdiffusivity(self, output, moltypel, dt, tol, numsamples=0, numboot=0):
        """This function fits the mean square displacement to calculate the
        diffusivity for all molecule types in the system
        """
        output["Diffusivity"] = {}
        output["Diffusivity"]["units"] = "m^2/s"
        time = output["MSD"]["time"]
        for i in range(len(moltypel)):
            if isinstance(output["MSD"][moltypel[i]], list):
                MSD = output["MSD"][moltypel[i]]
                lnMSD = np.log(MSD[1:])
                lntime = np.log(time[1:])
                firststep = self.findlinearregion(lnMSD, lntime, dt, tol)
                diffusivity = self.getdiffusivity(time, MSD, firststep)
                output["Diffusivity"][moltypel[i]] = diffusivity
            else:
                if numsamples == 0 or numboot == 0:
                    MSD = np.average(output["MSD"][moltypel[i]], axis=0)
                    lnMSD = np.log(MSD[1:])
                    lntime = np.log(time[1:])
                    firststep = self.findlinearregion(lnMSD, lntime, dt, tol)
                    diffusivity = self.getdiffusivity(time, MSD, firststep)
                    output["Diffusivity"][moltypel[i]] = diffusivity
                else:
                    Values = []
                    random.seed(123456789)
                    for j in range(numboot):
                        Values.append(self.Bootstrap(numsamples, output["MSD"][moltypel[i]], time, dt, tol))
                        sys.stdout.write(f"\rDiffusivity Bootstrap {j + 1} of {numboot} for {moltypel[i]} complete")
                    sys.stdout.write("\n")
                    ave, stddev = self.getAverage(Values)
                    output["Diffusivity"][f"{moltypel[i]}"] = ave
                    output["Diffusivity"][f"{moltypel[i]} Deviation"] = stddev

    def findlinearregion(self, lnMSD, lntime, dt, tol):
        # Uses the slope of the log-log plot to find linear regoin of MSD
        timestepskip = np.ceil(500 / dt)
        maxtime = len(lnMSD)
        numskip = 1
        while True:
            if numskip * timestepskip + 1 > maxtime:
                return maxtime - 1 - (numskip - 1) * timestepskip
            else:
                t1 = int(maxtime - 1 - (numskip - 1) * timestepskip)
                t2 = int(maxtime - 1 - numskip * timestepskip)
                slope = (lnMSD[t1] - lnMSD[t2]) / (lntime[t1] - lntime[t2])
                if abs(slope - 1.0) < tol:
                    numskip += 1
                else:
                    return t1

    def getdiffusivity(self, Time, MSD, firststep):
        # Fits the linear region of the MSD to obtain the diffusivity
        calctime = []
        calcMSD = []
        for i in range(int(firststep), len(Time)):
            calctime.append(Time[i])
            calcMSD.append(MSD[i])
        if len(calctime) == 1:
            diffusivity = "runtime not long enough"
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                line = stats.linregress(calctime, calcMSD)
            slope = line[0]
            diffusivity = slope / 600000
        return diffusivity

    def writeLogLog(self, lnMSD, lntime, moltype):
        outfile = open(f"LogLog{moltype}.dat", "w")
        for i in range(len(lnMSD)):
            outfile.write(f"{lntime[i]}\t{lnMSD[i]}\n")
        outfile.close()

    def getAverage(self, Values):
        ave = np.average(Values)
        stddev = np.std(Values)
        return ave, stddev

    def Bootstrap(self, numsamples, MSD, time, dt, tol):
        Bootlist = np.zeros((numsamples, MSD.shape[-1]))
        for j in range(numsamples):
            rint = random.randint(0, len(MSD) - 1)
            Bootlist[j, :] = MSD[rint, :]
        MSD = np.average(Bootlist, axis=0)
        lnMSD = np.log(MSD[1:])
        lntime = np.log(time[1:])
        firststep = self.findlinearregion(lnMSD, lntime, dt, tol)
        return self.getdiffusivity(time, MSD, firststep)
