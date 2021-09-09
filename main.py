import numpy as np
import matplotlib.pyplot as plt
import time

from numpy.core.function_base import linspace
from numpy.linalg import eigvals


import DFT_1d
 



# numeric paramters
FLOAT_PRECISION = 1e-05
Ngrid = 200
xmin = -5
xmax = 5
(xvec, dX) = np.linspace(xmin, xmax, Ngrid,retstep=True)
print("dX = {0}".format(dX))

# phisical constants
omega = 2*np.pi
m = 1
hbar = 1

# DFT hyper parameters
NumberofElectrons = 3


def get_kinetic_mat(Ngrid, xmin, xmax):
    xvec = np.linspace(xmin, xmax, Ngrid)
    dx = dX
    dia = -2*np.ones(Ngrid)
    offdia = np.ones(Ngrid-1)
    d2grid = np.mat(np.diag(dia, 0) + np.diag(offdia, -1) + \
                    np.diag(offdia, 1))/dx**2
    # avoid strange things at the edge of the grid
    #d2grid[0, :] = 0
    #d2grid[Ngrid-1, :] = 0
    d2grid[-1,-1] = d2grid[0,0]
    Ekin = -hbar**2/(2*m)*d2grid
    return np.asarray(Ekin)


def get_potential_mat(Ngrid, xmin, xmax):
    xvec = np.linspace(xmin, xmax, Ngrid)
    Vx = xvec**2  # simple X^2 potential
    Vx = np.diag(Vx, 0)
    return np.asarray(Vx)


def check_normalized_state(state):
    s = 0
    for k in state:
        s = s + k**2
    s = float(s)
    assert (abs(s - 1) <
            FLOAT_PRECISION), "probability of state should sum to 1! prob={0} instead".format(s)


def check_density(density, num_electrons):
    """ verifies that the density sums to number of electrons

    Args:
        density ([1d matrix(vector)]): [description]
        num_electrons ([integer]): [description]
    """
    s = float(np.sum(density))
    assert (abs(s - num_electrons) < FLOAT_PRECISION), \
        "density should sum to {0} ! prob={1} instead".format(num_electrons, s)


#ToDo: change occupation type to spin orbitals
def calc_density(EigenVecs, num_electrons):
    """ returns density vector by using the Eigen vectors. 
    also calculates occupation.

    Args:
        EigenVecs ([type]): [description]
        num_electrons ([type]): [description]
    """    
    test_state = EigenVecs[:, 2]
    check_normalized_state(test_state)
    density = 0
    assert num_electrons > 1
        
    for i in range(0, int(num_electrons/2)):
        print("i = ",i)
        density += np.power(np.abs(EigenVecs[:, i]), 2)
        density += np.power(np.abs(EigenVecs[:, i*2 + 1]), 2)
    if num_electrons%2 == 1:
        density += np.power(np.abs(EigenVecs[:, int(num_electrons/2) + 1]), 2)
    #for i in range(0, num_electrons):
    #    density = density + np.power(np.abs(EigenVecs[:, i]), 2)
    check_density(density, num_electrons)
    return density

#ToDo: check if dX is needed in the energy calc here??
def calculate_exchange(density):
    x_energy_const = -(3/4)*np.power((3/np.pi),1/3)
    x_potential_const = -1 * np.power((3/np.pi),1/3)
    x_energy = x_energy_const * np.sum(np.power(density,4/3)) * dX
    x_potential = x_potential_const * np.power(density,1/3)
    x_potential = np.mat(np.diag(x_potential, 0))
    return x_energy, np.asarray(x_potential)


def calculate_coloumb(density, eps=1e-1):
    start_time = time.time()
    c_energy = 0
    c_potential = np.zeros(xvec.shape)
    for t1 in range(density.size):
        x1 = xvec[t1]
        d1 = density[t1]
        for t2 in range(density.size):
            x2 = xvec[t2]
            d2 = density[t2]
            c_energy = c_energy + 0.5*dX**2*(d1*d2)/np.sqrt(np.power(x1-x2,2) + eps)
            c_potential[t1] = c_potential[t1] + dX*d1/np.sqrt(np.power(x1-x2,2) + eps)

    c_potential = np.mat(np.diag(c_potential, 0))
    end_time = time.time()
    print("--- took %s seconds ---" % (time.time() - start_time))
    return float(c_energy), np.asarray(c_potential)


def diagonalize_hamiltonian(H):
    """ solves Kohn Sham Equations (hamiltonian diagonalization)
        returns sorted EigenVals,EigenVecs
    Args:
        H ([ndarray (2d)]): hamiltonian to diagonalize
    """
    print("Hamiltonian Diagonaliztion...")
    epsilon_n, psi_gn = np.linalg.eigh(H)
    sortinds = np.argsort(epsilon_n)
    sorted_EigVecs = psi_gn[:, sortinds]
    sorted_EigVals = epsilon_n[sortinds]
    return sorted_EigVals, sorted_EigVecs

def run_scf(Ngrid, xmin, xmax, initial_density, max_iterations, convergence_threshold=0.001):
    xvec = np.linspace(xmin, xmax, Ngrid)
    prev_density = np.zeros(xvec.shape) # just to initiate the scf loop
    iteration = 0
    new_density = initial_density
    print(np.sum(np.abs(prev_density - new_density)))
    while(np.sum(np.abs(prev_density - new_density)) > convergence_threshold) and (iteration < max_iterations):
        print("running SCF loop iteration {0}".format(iteration))
        # change the density for the next iteration
        prev_density = new_density
        density = new_density
        # calculate the effective potential using the density
        Vex = get_potential_mat(Ngrid, xmin, xmax)
        x_energy, x_potential = calculate_exchange(density)
        c_energy, c_potential = calculate_coloumb(density)

        Veff = Vex + x_potential + c_potential
        
        # construct the Hamiltonian
        T = get_kinetic_mat(Ngrid, xmin, xmax)
        H = T + Veff
        
        # solve Kohn-Sham equations - find eigenvalues and eigenvectors for the hamiltonian
        EigenVals, EigenVecs = diagonalize_hamiltonian(H)
        print("visualizing new eigenfunctions:")
        for i in range(5):
            plt.plot(xvec,EigenVecs[:,i],label=EigenVals[i])
            plt.legend(loc=1)
        plt.show()
        # calculate new density
        new_density = calc_density(EigenVecs, NumberofElectrons)
        #print(prev_density)
        print("@@@@@@calculated new density@@@@@\n")
        #print(new_density)
        print(" diff from prev density is:")
        print(np.sum(np.abs(prev_density - new_density)))
        print("ground state energy is {0} AU".format(EigenVals[0]))
        iteration += 1
        


# step 1 - Initiate a grid
# xvec = np.linspace(xmin, xmax, Ngrid)
# print("Initiating a grid with dimensions:Ngrid = {0}, xmin = {1},xmax = {2}"\
#     .format(Ngrid, xmin, xmax))

# # step 2 - calc kinetic energy
# print("Calculating kinetic energy matrix")
# T = get_kinetic_mat(Ngrid, xmin, xmax)
# print(T)
# # step ? - calc total Hamiltonian
# print("Calculating external potential matrix")
# V = get_potential_mat(Ngrid, xmin, xmax)
# H =  T + V

# step 3 - diagonalize Hamiltonian
# print("Hamiltonian Diagonaliztion...")
# epsilon_n, psi_gn = np.linalg.eigh(H)
# sortinds = np.argsort(epsilon_n)
# sorted_EigVecs = psi_gn[:, sortinds]
# sorted_EigVals = epsilon_n[sortinds]

# step 4 - Normalize all eigenstates to sum(psi^2) = 1 ???

# step ? - visualize an example eigenfunction
# print("simple Visualization")
# n = 0
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# ax1.plot(xvec, np.real(sorted_EigVecs[:, n]))
# ax1.set(title='Real Eigenfunction %d' % (n), xlabel='x')
# ax2.plot(xvec, np.power(np.abs(sorted_EigVecs[:, n]), 2))
# ax2.set(title='AbsQuadrat Eigenfunction {0} eigenval={1}'.format(
#     n, sorted_EigVals[n]), xlabel='x')
# fig.tight_layout()
# plt.show()
# plt.figure()
# for i in range(5):
#     plt.plot(xvec,sorted_EigVecs[:,i],label=sorted_EigVals[i])
#     plt.legend(loc=1)

# fig = plt.figure()
# plt.plot(xvec, np.diag(c_potential))
# fig.suptitle('Coloumb Potential')
# plt.xlabel('x')
# plt.ylabel('potential')
#plt.show()

if __name__ == "__main__":
    print("\n\n#####################################################################################")
    print("Starting DFT Calc!")
    initial_density = NumberofElectrons* np.ones(xvec.shape)/(Ngrid)
    check_density(initial_density,NumberofElectrons)
    run_scf(Ngrid, xmin, xmax, initial_density, max_iterations=9, convergence_threshold=0.0000001)