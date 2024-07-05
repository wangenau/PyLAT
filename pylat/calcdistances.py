import math

import numpy as np
from numba import njit


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
