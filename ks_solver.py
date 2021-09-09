
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from numpy.linalg import eigvals


from coloumb_ee_potential import calculate_coloumb
from external_potential import get_external_harmonic_potential, get_radial_potential_term
from xc_potential import calculate_exchange
from kinetic_energy import get_kinetic_mat

FLOAT_PRECISION = 1e-05



class KS_Solver:
    """ Kohn sham solver. Contains the self consistent loop logic """
    def __init__(self, grid, numelectrons, maxiterations, \
            initial_density,convergence_threshold=0.001) -> None:
        """[summary]

        Args:
            grid ([Radial_Grid or Cartesian_1d]): [description]
            numelectrons ([int]): number of electrons in the molecule.
            maxiteraions ([int]): [description]
            convergence_threshold (float, optional): [description]. Defaults to 0.001.
        """
        self.grid_type = grid.grid_type
        self.grid = grid # currently assuming only 1d grids
        self.maxiterations = maxiterations
        self.convergence_threshold = convergence_threshold
        self.initial_density = initial_density
        self.numelectrons = numelectrons

    def _check_normalized_state(self, state):
        s = 0
        for k in state:
            s = s + k**2
        s = float(s)
        assert (abs(s - 1) <
                FLOAT_PRECISION), "probability of state should sum to 1! prob={0} instead".format(s)
    
    def _check_density(self,density, num_electrons):
        """ verifies that the density sums to number of electrons

        Args:
            density ([1d matrix(vector)]): [description]
            num_electrons ([integer]): [description]
        """
        s = float(np.sum(density))
        assert (abs(s - num_electrons) < FLOAT_PRECISION), \
            "density should sum to {0} ! prob={1} instead".format(num_electrons, s)

    
    #ToDo: change occupation type to spin orbitals
    def _calc_density(self, EigenVecs, num_electrons):
        """ returns density vector by using the Eigen vectors. 
        also calculates occupation.

        Args:
            EigenVecs ([type]): [description]
            num_electrons ([type]): [description]
        """    
        test_state = EigenVecs[:, 2]
        self._check_normalized_state(test_state)
        density = 0
        assert num_electrons > 1
            
        for i in range(0, int(num_electrons/2)):
            density += np.power(np.abs(EigenVecs[:, i]), 2)
            density += np.power(np.abs(EigenVecs[:, i*2 + 1]), 2)
        if num_electrons%2 == 1:
            density += np.power(np.abs(EigenVecs[:, int(num_electrons/2) + 1]), 2)
        #for i in range(0, num_electrons):
        #    density = density + np.power(np.abs(EigenVecs[:, i]), 2)
        self._check_density(density, num_electrons)
        return density

    def _diagonalize_hamiltonian(self,H):
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

    def runSCF(self):
        """ performs an self consistent loop. 
            the loop is implemented as while loop with 2 conditions: 
                1) density convergence 
                2) maxiteraions of the loop
        """
        prev_density = np.zeros(self.grid.gridvec.shape) # just to initiate the scf loop
        iteration = 0
        new_density = self.initial_density
        #print(np.sum(np.abs(prev_density - new_density)))
        
        while(np.sum(np.abs(prev_density - new_density)) > self.convergence_threshold) \
            and (iteration < self.maxiterations):
            
            print("running SCF loop iteration {0}".format(iteration))
            print(" grid type - ", self.grid_type)
            # change the density for the next iteration
            prev_density = new_density
            density = new_density

            # calculate the effective potential using the density
            if self.grid_type == "cartesian_1d":
                Vext = get_external_harmonic_potential(self.grid.gridvec)
            if self.grid_type == "radial_grid":
                Vext = get_radial_potential_term(self.grid.gridvec)

            x_energy, x_potential = calculate_exchange(density, self.grid.grid_dr)
            c_energy, c_potential = calculate_coloumb(density, self.grid.gridvec,self.grid.grid_dr)

            Veff = Vext + x_potential + c_potential
            
            # construct the Hamiltonian
            T = get_kinetic_mat(self.grid.gridvec,self.grid.grid_dr)
            H = T + Veff
            
            # solve Kohn-Sham equations - find eigenvalues and eigenvectors for the hamiltonian
            EigenVals, EigenVecs = self._diagonalize_hamiltonian(H)
            print("visualizing new eigenfunctions:")
            for i in range(3):
                plt.plot(self.grid.gridvec,EigenVecs[:,i],label=EigenVals[i])
                plt.legend(loc=1)
            plt.show()
            # calculate new density
            new_density = self._calc_density(EigenVecs, self.numelectrons)
            #print(prev_density)
            print("@@@@@@calculated new density@@@@@\n")
            #print(new_density)
            print(" diff from prev density is:")
            print(np.sum(np.abs(prev_density - new_density)))
            print("ground state energy is {0} AU".format(EigenVals[0]))
            iteration += 1