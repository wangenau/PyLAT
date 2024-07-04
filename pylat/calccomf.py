import numpy as np
from numba import njit


@njit
def calccom(n, nummol, x, y, z, mol, amass, molmass, Lx, Ly, Lz, Lx2, Ly2, Lz2):
    comxt = np.zeros(nummol, dtype=np.float32)
    comyt = np.zeros(nummol, dtype=np.float32)
    comzt = np.zeros(nummol, dtype=np.float32)
    xt = np.zeros(nummol, dtype=np.float32)
    yt = np.zeros(nummol, dtype=np.float32)
    zt = np.zeros(nummol, dtype=np.float32)
    ux = np.zeros(n, dtype=np.float32)
    uy = np.zeros(n, dtype=np.float32)
    uz = np.zeros(n, dtype=np.float32)
    for i in range(n):
        mol_i = int(mol[i] - 1)
        if xt[mol_i] == 0.0:
            xt[mol_i] = x[i]
            yt[mol_i] = y[i]
            zt[mol_i] = z[i]
            ux[i] = x[i]
            uy[i] = y[i]
            uz[i] = z[i]
        else:
            if (x[i] - xt[mol_i]) > Lx2:
                ux[i] = x[i] - Lx
            elif (xt[mol_i] - x[i]) > Lx2:
                ux[i] = x[i] + Lx
            else:
                ux[i] = x[i]
            if (y[i] - yt[mol_i]) > Ly2:
                uy[i] = y[i] - Ly
            elif (yt[mol_i] - y[i]) > Ly2:
                uy[i] = y[i] + Ly
            else:
                uy[i] = y[i]
            if (z[i] - zt[mol_i]) > Lz2:
                uz[i] = z[i] - Lz
            elif (zt[mol_i] - z[i]) > Lz2:
                uz[i] = z[i] + Lz
            else:
                uz[i] = z[i]
        comxt[mol_i] += ux[i] * amass[i]
        comyt[mol_i] += uy[i] * amass[i]
        comzt[mol_i] += uz[i] * amass[i]
    for i in range(nummol):
        comxt[i] /= molmass[i]
        comyt[i] /= molmass[i]
        comzt[i] /= molmass[i]
    return comxt, comyt, comzt
