# Dense Example
from densed import densed
import numpy as np
import numpy.linalg as LA
import time
t = time.time()
N = 4  # Declaring the number of particle
p = 1000  # Declaring the number of random RDMs to be sampled

run = densed(N)  # initializing run
rand = np.random.uniform(-1, 1, (p, 3))  # choosing random Hamiltonian Configurations

hammat = run.ham(rand)  # generating the Hamiltonian
vmin = LA.eigh(hammat)[1][:, :, 0]  # choosing the lowest energy eigenvalue

lamD, lamG = run.lambdas(vmin)  # Getting the Lambda values from the D and G 2-RDMs
lamexp, gamexp, gmexp = run.expects(
    vmin
)  # Getting expectation values for each component of Hamiltonian

#""" #plot of expectation values 
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
#"""
print(time.time()-t)

# Sparse Example
from sparsed import sparsed
t = time.time()
N = 8  # Declaring particle number
p = 100  # The number of sampled random Hamiltonian configurations
rand = np.random.uniform(-1, 1, (p, 3))  # generating random Hamiltonian configurations

run = sparsed(N)  # Initializing setup
vmin, emin = run.vmake(rand)  # Calculating minimal eigenvector and its energy
lamD, lamG = run.lambdas(vmin)  # Calculating largest eigenvalues of D and G(mod) 2-RDMs
lamexp, gamexp, gmexp = run.expects(
    vmin
)  # Calculating expectation values of each Hamiltonian term
print(time.time()-t)
#""" #plot of expectation values 
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
#"""
