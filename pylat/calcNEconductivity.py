"""Created on Thu May  7 15:06:47 2015

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

import math


class calcNEconductivity:
    def calcNEconductivity(self, output, molcharge, Lx, Ly, Lz, nummoltype, moltypel, T, nu=None):
        """This function uses the Nernst-Einstein equation to estimate the ionic
        conductivity of the system from the diffusivities

        """

        output["Conductivity"] = {}
        V = Lx * Ly * Lz * 10**-30
        e = 1.60217657e-19
        k = 1.3806488e-23
        if nu is not None:
            xi = 2.837298
            assert abs(Lx - Ly) < 1e-6
            assert abs(Lx - Lz) < 1e-6
            corr = k * T * xi / (math.pi * 6 * nu * 1e-3 * Lx * 1e-10)
            output["Diffusivity Corrected"] = {}
            output["Diffusivity Corrected"]["Yeh_Hummer_Correction"] = corr
            output["Conductivity Corrected"] = {}

        NEcond = 0
        NEcond_corr = 0
        for i in range(len(moltypel)):
            q = float(molcharge[moltypel[i]])
            if q != 0:
                try:
                    D = float(output["Diffusivity"][moltypel[i]])
                    if nu is not None:
                        D_corr = D + corr
                        output["Diffusivity Corrected"][moltypel[i]] = D_corr
                except ValueError:
                    output["Nernst Einstein Conductivity in S/m"] = "runtime not long enough"
                    return output
                N = int(nummoltype[i])
                NEcond += N * q**2 * D
                output["Conductivity"][f"Nernst_Einstein_{moltypel[i]}"] = N * q**2 * D * e**2 / k / T / V
                if nu is not None:
                    NEcond_corr += N * q**2 * D_corr
                    output["Conductivity Corrected"][f"Nernst_Einstein_{moltypel[i]}"] = (
                        N * q**2 * D_corr * e**2 / k / T / V
                    )
        NEcond *= e**2 / k / T / V
        output["Conductivity"]["Nernst_Einstein"] = NEcond
        if nu is not None:
            NEcond_corr *= e**2 / k / T / V
            output["Conductivity Corrected"]["Nernst_Einstein"] = NEcond_corr

        return output
