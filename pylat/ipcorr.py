import numpy as np
from numba import njit


@njit
def ipcorr(closest, skip, L1, L2, L3, CL, moltype):
    correlation = np.zeros((CL + 1, L3, L3), dtype=np.float32)
    for i in range(L2):
        for j in range(L3):
            for k in range(skip, skip + CL):
                for m in range(k, k + CL + 1):
                    if closest[k, i, j] == closest[m, i, j]:
                        correlation[m - k, moltype[i], j] += 1
    for i in range(L3):
        for j in range(L3):
            norm = correlation[0, i, j]
            for k in range(CL + 1):
                correlation[k, i, j] /= norm
    return correlation
