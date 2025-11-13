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

import numpy as np


class calcNEconductivity:
    def calcNEconductivity(self, output, molcharge, Lx, Ly, Lz, nummoltype, moltypel, T):
        """This function uses the Nernst-Einstein equation to estimate the ionic
        conductivity of the system from the diffusivities

        """

        output["Conductivity"] = {}
        output["Conductivity"]["units"] = "S/m"
        V = Lx * Ly * Lz * 10**-30
        e = 1.60217657e-19
        k = 1.3806488e-23
        # If viscosity is given, calculate the Yeh Hummer correction and its standard deviation
        if "Viscosity" in output:
            xi = 2.837298
            assert abs(Lx - Ly) < 1e-6
            assert abs(Lx - Lz) < 1e-6
            corr = k * T * xi / (np.pi * 6 * output["Viscosity"]["Average Value"] * 1e-3 * Lx * 1e-10)
            corr_err = np.abs(
                output["Viscosity"]["Standard Deviation"]
                * k
                * T
                * xi
                / (np.pi * 6 * -(output["Viscosity"]["Average Value"] ** 2) * 1e-3 * Lx * 1e-10)
            )
            output["Diffusivity Corrected"] = {}
            output["Diffusivity Corrected"]["units"] = "m^2/s"
            output["Diffusivity Corrected"]["Yeh_Hummer_Correction"] = corr
            output["Diffusivity Corrected"]["Yeh_Hummer_Correction Deviation"] = corr_err
            output["Conductivity Corrected"] = {}
            output["Conductivity Corrected"]["units"] = "S/m"

        NEcond = 0
        NEcond_err = 0
        NEcond_corr = 0
        NEcond_corr_err = 0
        for i in range(len(moltypel)):
            D_err = 0
            q = float(molcharge[moltypel[i]])
            if q != 0:
                try:
                    D = float(output["Diffusivity"][moltypel[i]])
                    if "Viscosity" in output:
                        D_corr = D + corr
                        output["Diffusivity Corrected"][moltypel[i]] = D_corr
                except ValueError:
                    output["Nernst Einstein Conductivity in S/m"] = "runtime not long enough"
                    return output
                try:
                    D_err = float(output["Diffusivity"][f"{moltypel[i]} Deviation"])
                    # Error propagation for the corrected diffusivity
                    if "Viscosity" in output:
                        D_corr_err = np.sqrt(D_err**2 + corr_err**2)
                        output["Diffusivity Corrected"][f"{moltypel[i]} Deviation"] = D_corr_err
                except KeyError:
                    pass

                N = int(nummoltype[i])
                NEcond += N * q**2 * D
                output["Conductivity"][f"Nernst_Einstein_{moltypel[i]}"] = N * q**2 * D * e**2 / k / T / V
                if "Viscosity" in output:
                    NEcond_corr += N * q**2 * D_corr
                    output["Conductivity Corrected"][f"Nernst_Einstein_{moltypel[i]}"] = (
                        N * q**2 * D_corr * e**2 / k / T / V
                    )
                if D_err != 0:
                    NEerr = N * q**2 * e**2 / k / T / V  # One part of the error propagation
                    output["Conductivity"][f"Nernst_Einstein_{moltypel[i]} Deviation"] = np.sqrt(NEerr**2 * D_err**2)
                    NEcond_err += NEerr**2 * D_err**2
                    if "Viscosity" in output:
                        NEcond_corr_err += NEerr**2 * D_corr_err**2
                        output["Conductivity Corrected"][f"Nernst_Einstein_{moltypel[i]} Deviation"] = np.sqrt(
                            NEerr**2 * D_corr_err**2
                        )
        NEcond *= e**2 / k / T / V
        output["Conductivity"]["Nernst_Einstein"] = NEcond
        output["Conductivity"]["Nernst_Einstein Deviation"] = np.sqrt(NEcond_err)
        if "Viscosity" in output:
            NEcond_corr *= e**2 / k / T / V
            output["Conductivity Corrected"]["Nernst_Einstein"] = NEcond_corr
            output["Conductivity Corrected"]["Nernst_Einstein Deviation"] = np.sqrt(NEcond_corr_err)

        return output
