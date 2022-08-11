from parent import common
import numpy as np
from h import Hinit
import sparse
import scipy.sparse.linalg as LA2
import numba


@numba.jit(nopython=True, nogil=True, parallel=True, cache = True)
def expred(v, mat):
    out = np.zeros(v.shape[0])
    c = mat.coords
    d = mat.data
    for i in numba.prange(v.shape[0]):
        for j in range(d.shape[0]):
            out[i] += v[i, c[0, j]] * v[i, c[1, j]] * d[j]
    return out


class sparsed(common):
    def __init__(self, N):
        super().__init__(N)
        self.lammsp, self.gammsp, self.gmsp, self.states = Hinit(self.N, self.ind)

    def vmake(self, path):
        vmin = np.zeros((np.shape(path)[0], self.gammsp.shape[0]))
        vmin[-1, :] = np.random.rand(self.gammsp.shape[0])
        emin = np.zeros(np.shape(path)[0])
        for i in range(np.shape(path)[0]):
            hammat = (
                path[i, 0] * self.lammsp
                + path[i, 1] * self.gammsp
                + path[i, 2] * self.gmsp
            )
            e1, v1 = LA2.eigsh(hammat, k=1, which="SA", v0=vmin[-1, :])
            vmin[i, :] = v1.reshape(-1)
            emin[i] = e1
        return vmin, emin

    def expects(self, vmin):
        lamexp = expred(vmin, self.lammsp)
        gamexp = expred(vmin, self.gammsp)
        gmexp = expred(vmin, self.gmsp)
        return lamexp, gamexp, gmexp


if __name__ == "__main__":
    N = 6  # Declaring particle number
    p = 50  # The number of sampled random Hamiltonian configurations
    rand = np.random.uniform(
        -1, 1, (p, 3)
    )  # generating random Hamiltonian configurations

    run = sparsed(N)  # Initializing setup
    vmin, emin = run.vmake(rand)  # Calculating minimal eigenvector and its energy
    lamD, lamG = run.lambdas(
        vmin
    )  # Calculating largest eigenvalues of D and G(mod) 2-RDMs
    lamexp, gamexp, gmexp = run.expects(
        vmin
    )  # Calculating expectation values of each Hamiltonian term

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
