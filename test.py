import numpy as np
import numpy.linalg as LA
import time
from densed import dense
N = 2
p = 5
rand = np.random.uniform(-10, 10, (p, 3))  # generating random Hamiltonian configurations
t = time.time()
run = dense(N)  # Initializing setup
print(time.time()-t)

t = time.time()
hammat = run.ham(rand)  # generating the Hamiltonian
print(time.time()-t)

t = time.time()
vmin = LA.eigh(hammat)[1][:, :, 0]  # choosing the lowest energy eigenvalue
print('diag',time.time()-t)

from rdms2 import lazyrdmcal, lazygmatcal
t = time.time()
dmat, gmat = run.rdms(vmin)
print('matmake',time.time()-t)
t = time.time()
dmat1 = lazyrdmcal(N, run.states, vmin, run.ind)
gmat1 = lazygmatcal(N, run.states, vmin, run.ind)
#eig1 = LA.eigvalsh(gmat)
#eig2 = LA.eigvalsh(gmat1)
#print(eig1[0,:])
#print(eig2[0,:])

catk

t = time.time()
lamD, lamG = run.lambdas(vmin)  # Calculating largest eigenvalues of D and G(mod) 2-RDMs
print('matmake',time.time()-t)

t = time.time()
lamexp, gamexp, gmexp = run.expects(
    vmin
)  # Calculating expectation values of each Hamiltonian term
print(time.time()-t)

'''
from sparsed import sparsed
N = 10  # Declaring particle number
p = 50  # The number of sampled random Hamiltonian configuration
rand = np.random.uniform(-1, 1, (p, 4))  # generating random Hamiltonian configurations
rand[:, 0] = 0  # Setting epsilon contribution to zero (see paper for reasoning)

t = time.time()
run = sparsed(N)  # Initializing setup
print(time.time()-t)

t = time.time()
vmin, emin = run.vmake(rand)  # Calculating minimal eigenvector and its energy
print(time.time()-t)

t = time.time()
lamD, lamG = run.lambdas(vmin)  # Calculating largest eigenvalues of D and G(mod) 2-RDMs
print(time.time()-t)

t = time.time()
lamexp, gamexp, gmexp = run.expects(
    vmin
)  # Calculating expectation values of each Hamiltonian term
print(time.time()-t)

'''
