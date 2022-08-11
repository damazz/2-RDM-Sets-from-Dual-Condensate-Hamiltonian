import numpy as np
import numpy.linalg as LA
from itertools import combinations
from .prune import prune
from .rdms import lazyrdmcal,lazygmatcal
import sparse
import numba

@numba.jit(nopython=True, nogil=True, cache = True)
def comb1(n, k):
    if n < k:
        return 0
    elif n == k:
        return 1
    c = n - k
    pa = np.int64(1)
    pb = np.int64(1)
    if k > c:
        for i in range(k + 1, n + 1):
            pa *= i
        for i in range(1, c + 1):
            pb *= i
    else:
        for i in range(c + 1, n + 1):
            pa *= i
        for i in range(1, k + 1):
            pb *= i
    return pa // pb

@numba.jit(nopython=True, nogil=True, parallel=True)
def rdmred(v, c, N):
    out = np.zeros((v.shape[0], 2 * N, 2 * N, 2 * N, 2 * N), dtype=np.float64)
    for i in numba.prange(v.shape[0]):
        for j in range(c.shape[1]):
            out[i, c[2, j], c[3, j], c[4, j], c[5, j]] += v[i, c[0, j]] * v[i, c[1, j]]
    return out

class common:
    def __init__(self, N):
        self.N = N
        self.ind = np.int64(prune(self.N))

    def rdms(self, vmin):
        if vmin.shape[1]>self.ind.shape[0]:
            vmin = vmin[:,self.ind]
        d2ground = lazyrdmcal(self.N, self.states, vmin, self.ind)
        g2ground = lazygmatcal(self.N, self.states, vmin, self.ind)
        return d2ground, g2ground

    def lambdas(self, vmin):
        d2ground, g2ground = self.rdms(vmin)
        d2eigs = LA.eigvalsh(d2ground)[:, -1] /2
        g2eigs = LA.eigvalsh(g2ground)[:, -2] 
        return d2eigs, g2eigs
