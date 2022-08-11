from .parent import common
import numpy as np
import numpy.linalg as LA
from itertools import combinations
from .h import Hinit
import numba
import time


@numba.jit(nopython=True, nogil=True, parallel=True, cache = True)
def expred(v, mat):
    out = np.zeros(v.shape[0])
    for i in numba.prange(v.shape[0]):
        for j in range(mat.shape[0]):
            for k in range(mat.shape[0]):
                out[i] += v[i, j] * v[i, k] * mat[j, k]
    return out

@numba.jit(nopython=True, nogil=True, parallel=True, cache = True)
def reshuffle(lamc,lamd,gamc,gamd,gmc,gmd,ind):
    lamm = np.zeros((ind.shape[0], ind.shape[0]), dtype = np.float64)
    gamm = np.zeros((ind.shape[0], ind.shape[0]), dtype = np.float64)
    gm = np.zeros((ind.shape[0], ind.shape[0]), dtype = np.float64)
    for i in numba.prange(lamd.shape[0]):
        jl = np.argwhere(lamc[0,i]==ind)[0,0]
        kl = np.argwhere(lamc[1,i]==ind)[0,0]
        lamm[jl,kl] = lamd[i]
    for i in numba.prange(gamd.shape[0]):
        jg = np.argwhere(gamc[0,i]==ind)[0,0]
        kg = np.argwhere(gamc[1,i]==ind)[0,0]
        gamm[jg,kg] = gamd[i]
    for i in numba.prange(gmd.shape[0]):
        jgm= np.argwhere(gmc[0,i]==ind)[0,0]
        kgm= np.argwhere(gmc[1,i]==ind)[0,0] gm[jgm,kgm] = gmd[i]
    return lamm, gamm, gm

@numba.jit(nopython=True, nogil=True, parallel = True, cache = True)
def hammake(path, lamm, gamm, gm):
    out = np.zeros((path.shape[0],lamm.shape[0],lamm.shape[0]), dtype = np.float64)
    for i in numba.prange(path.shape[0]):
        for j in range(lamm.shape[0]):
            for k in range(lamm.shape[0]):
                out[i,j,k] += path[i,0]*lamm[j,k]
                out[i,j,k] += path[i,1]*gamm[j,k]
                out[i,j,k] += path[i,2]*gm[j,k]
    return out


class densed(common):
    def __init__(self, N):
        super().__init__(N)
        lammt, gammt, gmt , self.states= Hinit(self.N, self.ind)
        self.lamm, self.gamm, self.gm = reshuffle(lammt.coords,lammt.data,gammt.coords,gammt.data,gmt.coords,gmt.data, self.ind)

    def ham(self, path):
        return hammake(path,self.lamm, self.gamm, self.gm)

    def expects(self, vmin):
        lamexp = expred(vmin, self.lamm)
        gamexp = expred(vmin, self.gamm)
        gmexp = expred(vmin, self.gm)
        return lamexp, gamexp, gmexp



if __name__ == "__main__":
    N = 4  # Declaring the number of particles
    p = 100  # Declaring the number of random RDMs to be sampled

    run = densed(N)  # initializing run
    rand = np.random.uniform(
        -1, 1, (p, 3)
    )  # choosing random Hamiltonian Configurations

    hammat = ham(rand,run.lamm,run.gamm,run.gm)

    vmin = LA.eigh(hammat)[1][:, :, 0]  # choosing the lowest energy eigenvalue

    lamD, LamG = run.lambdas(vmin)  # Getting the Lambda values from the D and G 2-RDMs
    lamexp, gamexp, gexp = run.expects(
        vmin
    )  # Getting expectation values for each component of Hamiltonian

    """ #plot of expectation values 
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    #Plotting Lambda, Gamma, and G expectation Value with Lambda_D as the color mapping
    ax.scatter(lamexp, gamexp, gmexp, c = lamD, cmap = 'jet')

    norm = mpl.colors.Normalize(vmin = np.min(lamD),vmax = np.max(lamD))
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="jet"), ax=ax, shrink=.5)
    cbar.set_label(r'$\lambda_D$', rotation = 0, fontsize = 20)

    ax.set_xlabel(r'$\langle \Lambda \rangle$',fontsize = 20)
    ax.set_ylabel(r'$\langle W \rangle$', fontsize = 20)
    ax.set_zlabel(r'$\langle G \rangle$', fontsize = 20)
    plt.show()
    """
