import math
from numba import njit
import numpy as np


@njit
def calcdistances(nummol, comx, comy, comz, Lx, Ly, Lz):
    r = np.zeros((nummol, nummol), dtype=np.float32)
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
