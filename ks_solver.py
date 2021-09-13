import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from numpy.linalg import eigvals
from scipy import constants as const


from hartree_potential import calculate_hartree_pot
from external_potential import get_external_harmonic_potential, get_radial_potential_term
from xc_potential import calculate_exchange
from kinetic_energy import get_kinetic_mat

FLOAT_PRECISION = 1e-05

# phisical constants
omega = 2*np.pi
m_e = const.m_e 
hbar = const.hbar
e = const.e
eps_0 = const.epsilon_0


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
        self.occupation_list = self._create_occupation_list()
        
    def _create_occupation_list(self):
        occupation_list = []            
        for i in range(0, int(self.numelectrons/2)):
            occupation_list.append(2)
        if self.numelectrons%2 == 1:
            occupation_list.append(1)
        return occupation_list

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

        for i in range (0, len(self.occupation_list)):
            #print("orbital number - {0} adding occupation: {1}".format(i, self.occupation_list[i]))
            density += self.occupation_list[i] * np.power(np.abs(EigenVecs[:, i]), 2)


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

    def plot_density(self, r, density, eigenvalues):
        plt.xlabel('x ($\\mathrm{\AA}$)')
        plt.ylabel('probability density ($\\mathrm{\AA}^{-1}$)')
        plt.title('Atom density for l=0, n=1') 
        energy = 'E = {: >5.2f} eV'.format(eigenvalues[0] / e)
        plt.plot(r * 1e+10, density, color='blue',  label=energy)
        plt.legend()
        plt.show()
        return

    def get_ground_state_energy(self, EigenVals, E_Ha, x_energy):
        E0 = 0
        E_ks = 0
        E_integrated_xc = 0
        E_xc = x_energy

        for i in range (0, len(self.occupation_list)):
            E_ks += EigenVals[i] * self.occupation_list[i]
        
        E0 = E_ks - E_Ha - E_integrated_xc + E_xc
        return E0


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
        print(" grid type - ", self.grid_type)

        while(np.sum(np.abs(prev_density - new_density)) > self.convergence_threshold) \
            and (iteration < self.maxiterations):
            
            print("running SCF loop iteration {0}".format(iteration))
            # change the density for the next iteration
            prev_density = new_density
            density = new_density

            # calculate the effective potential using the density
            if self.grid_type == "cartesian_1d":
                Vext = get_external_harmonic_potential(self.grid.gridvec)
            if self.grid_type == "radial_grid":
                Vext = get_radial_potential_term(self.grid.gridvec,self.numelectrons)

            x_energy, x_potential = calculate_exchange(density, self.grid.grid_dr)
            Ha_energy, Ha_potential = calculate_hartree_pot(density, self.grid.gridvec,self.grid.grid_dr)

            #Veff = Vext + x_potential + Ha_potential
            Veff = Vext + Ha_potential
            # construct the Hamiltonian
            T = get_kinetic_mat(self.grid.gridvec, self.grid.grid_dr)
            H = - T + Veff
            
            # solve Kohn-Sham equations - find eigenvalues and eigenvectors for the hamiltonian
            EigenVals, EigenVecs = self._diagonalize_hamiltonian(H)
            
            # ks orbitals visualization
            for i in range(3):
                plt.plot(self.grid.gridvec,EigenVecs[:,i],label=EigenVals[i]/e)
                plt.legend(loc=1)
            plt.show()
            # calculate new density
            new_density = self._calc_density(EigenVecs, self.numelectrons)
            #print(prev_density)
            print("@@@@@@calculated new density@@@@@\n")
            #print(new_density)
            print(" diff from prev density is:", np.sum(np.abs(prev_density - new_density)))
            E0 = self.get_ground_state_energy(EigenVals, Ha_energy,0) /e
            print("ground state energy is {0} ev".format(E0))
            iteration += 1
        
        print("finished SCF loop")
        self.plot_density(self.grid.gridvec, new_density, EigenVals)
