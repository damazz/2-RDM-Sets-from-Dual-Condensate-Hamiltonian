import numpy as np
import numpy.linalg as LA
from itertools import combinations
from .prune import prune
from .rdms import g2gen,d2gen
from .rdms2 import lazyrdmcal,lazygmatcal
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
    #    return np.int64(pa/pb)
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
#        if N == 12:
#            self.d2matind = np.load("load/12p_d2indices.npz")["arr_0"]
#            self.g2matind = np.load("load/12p_d2indices.npz")["arr_0"]
#        elif N == 10:
#            self.d2matind = np.load("load/10p_d2indices.npz")["arr_0"]
#            self.g2matind = np.load("load/10p_d2indices.npz")["arr_0"]
#        elif N == 8:
#            self.d2matind = np.load("load/8p_d2indices.npz")["arr_0"]
#            self.g2matind = np.load("load/8p_d2indices.npz")["arr_0"]
#        else:
#            self.d2matind = d2gen(self.N, self.ind)
#            self.g2matind = g2gen(self.N, self.ind)

    def rdms(self, vmin):
        if vmin.shape[1]>self.ind.shape[0]:
            vmin = vmin[:,self.ind]
#        d2ground = rdmred(vmin, self.d2matind.T, self.N)
        d2ground = lazyrdmcal(self.N, self.states, vmin, self.ind)
        g2ground = lazygmatcal(self.N, self.states, vmin, self.ind)
#        g2ground = rdmred(vmin, self.g2matind.T, self.N)
#        d2ground = np.reshape(
#            d2ground, (np.shape(vmin)[0], (2 * self.N) ** 2, (2 * self.N) ** 2)
#        )
#        g2ground = np.swapaxes(g2ground, 1, 2).reshape(
#            np.shape(g2ground)[0], (2 * self.N) ** 2, (2 * self.N) ** 2
#        )
        return d2ground, g2ground

    def lambdas(self, vmin):
        d2ground, g2ground = self.rdms(vmin)
        d2eigs = LA.eigvalsh(d2ground)[:, -1] /2
        g2eigs = LA.eigvalsh(g2ground)[:, -2] 
#        from rdms2 import lazyrdmcal,lazygmatcal
#        d2ground = lazyrdmcal(self.N, self.states, vmin, self.ind)
#        g2ground = lazygmatcal(self.N, self.states, vmin, self.ind)
#        d2eigs1 = LA.eigvalsh(d2ground)[:,-1]/2
#        g2eigs1 = LA.eigvalsh(g2ground)[:,-2]
#        print(np.argwhere(np.round(d2eigs-d2eigs1,4)))
#        print(np.argwhere(np.round(g2eigs-g2eigs1,4)))

#        print(g2eigs[np.argwhere(np.round(g2eigs-g2eigs1,3))])
 #       print(g2eigs1[np.argwhere(np.round(g2eigs-g2eigs1,3))])

#        cat
        return d2eigs, g2eigs
