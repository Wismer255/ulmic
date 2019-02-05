import numpy as np
import matplotlib.pyplot as plt
from ulmic.atomic_units import  AtomicUnits
from ulmic.external.hdf5interface import Hdf5Interface
from matplotlib import animation
from scipy.integrate import cumtrapz
import h5py
import os
# import clog

MAX_ITERATIONS_NEIGHBORS = 10
MAX_POTENTIAL_DEVIATION = 1e-12
au = AtomicUnits()


class LocalPotential1D:

    def __init__(self,):
        self.set_spatial_parameters()
        self.set_potential_parameters()
        self.set_wave_function_parameters()
        self.set_medium_parameters()

    def set_spatial_parameters(self,
                               lattice_constant=5.0,
                               points_per_cell=80,
                               cell_repetitions=1):
        #Independent variables
        self.lattice_constant = lattice_constant
        self.points_per_cell = points_per_cell
        self.cell_repetitions = cell_repetitions

        #Dependent variables
        self.total_spatial_points = points_per_cell*cell_repetitions
        self.total_potential_length = lattice_constant*cell_repetitions
        self.spatial_axis = np.arange(self.total_spatial_points)*lattice_constant/points_per_cell
        self.spatial_step = self.spatial_axis[1] - self.spatial_axis[0]
        self.normalized_core_positions = (0.5 + np.arange(self.cell_repetitions))
        self.reset_convergence()

    def set_potential_parameters(self,
                                 lattice_potential_width=0.2,
                                 lattice_potential_depth=1.0,
                                 lattice_potential_type='sech'):
        self.lattice_potential_width = lattice_potential_width
        self.lattice_potential_depth = lattice_potential_depth
        self.lattice_potential_type = lattice_potential_type
        self.reset_convergence()

    def set_wave_function_parameters(self,
                                     positive_wavenumbers=20,
                                     ):
        self.positive_wavenumbers = positive_wavenumbers
        self.total_wavenumbers = 2*positive_wavenumbers+1

    def set_medium_parameters(self,
                              bands=10,
                              k_points=10,
                              valence_bands=1,
                              spin_factor=2
                              ):
        self.bands = bands
        self.k_points = k_points
        self.valence_bands = valence_bands
        self.spin_factor = spin_factor

        self.k_axis = np.linspace(0.0, 2*np.pi/self.total_potential_length, k_points + 1)[:-1]

    def reset_convergence(self):
        self.neighbors_for_convergence = 1
        self.neighbors_converged = False

    def lattice_potential_function(self,x):
        if self.lattice_potential_type == 'sech':
            return -self.lattice_potential_depth/np.cosh(x/self.lattice_potential_width)

    def create_potential(self,check_for_convergence=True):
        if check_for_convergence and not self.neighbors_converged:
            self.converge_number_of_neighboring_cells()
        core_positions = self.normalized_core_positions * self.lattice_constant
        relative_positions = self.spatial_axis[None,:]-core_positions[:,None]
        potential = np.sum(self.lattice_potential_function(relative_positions),axis=0)
        for i in range(1,self.neighbors_for_convergence):
            potential += np.sum(self.lattice_potential_function(relative_positions+i*self.lattice_constant),axis=0)
            potential += np.sum(self.lattice_potential_function(relative_positions-i*self.lattice_constant),axis=0)
        return potential

    def converge_number_of_neighboring_cells(self,):
        ''' Determine how many neighboring cells are needed to achieve
            numerical convergence '''
        deviation = 1.0
        potential_fine = self.create_potential(check_for_convergence=False)
        while deviation > MAX_POTENTIAL_DEVIATION:
            potential_coarse = np.copy(potential_fine)
            self.neighbors_for_convergence += 1
            potential_fine = self.create_potential(check_for_convergence=False)
            deviation = sum(abs(potential_fine-potential_coarse))
            if self.neighbors_for_convergence > MAX_ITERATIONS_NEIGHBORS:
                error = "MAX_ITERATIONS_NEIGHBORS reached: {:d}".format(self.neighbors_for_convergence)
                raise RuntimeError(error)
        self.neighbors_converged = True


    def get_real_space_hamiltonian(self,q=0.0):
        ''' Real space Hamiltonian of periodic part
            H_q = -(hbar^2/2m)(nabla - 1j*q)^2 + V(x) '''
        nx = self.total_spatial_points
        dx = self.spatial_step
        potential = self.create_potential()
        kinetic_operator = np.zeros((nx,nx), dtype=np.complex)
        potential_operator = np.zeros((nx,nx), dtype=np.complex)
        for i in range(nx):
            # kinetic_operator[i-1,i] = -0.5/(dx**2) -1j*q/(2*dx)
            # kinetic_operator[i,i-1] = -0.5/(dx**2) +1j*q/(2*dx)
            # kinetic_operator[i,i]   =  1.0/(dx**2) +0.5*q**2
            kinetic_operator[i-2,i] = 0.5*(1.0/12.0)/(dx**2)
            kinetic_operator[i,i-2] = 0.5*(1.0/12.0)/(dx**2)
            kinetic_operator[i-1,i] = -0.5*(4.0/3.0)/(dx**2) +1j*q/(2*dx)
            kinetic_operator[i,i-1] = -0.5*(4.0/3.0)/(dx**2) -1j*q/(2*dx)
            kinetic_operator[i,i]   =  0.5*(5.0/2.0)/(dx**2) +0.5*q**2
            potential_operator[i,i] = potential[i]
        return kinetic_operator + potential_operator

    def get_plane_wave(self,index):
        '''Plane waves in real space '''
        argument = 1j*2.0*np.pi*index*self.spatial_axis/self.total_potential_length
        return np.exp(argument)/np.sqrt(self.total_potential_length)

    def get_plane_wave_hamiltonian(self,q=0.0):
        ''' Hamiltonian H_q in basis of plane waves
            H_ijq =  <j| -(hbar^2/2m)(k-q)^2 + V(x) |i>'''
        '''TODO: Check indices'''
        kn = self.positive_wavenumbers
        knt = self.total_wavenumbers
        hamilton_kinetic = np.zeros((knt,knt), dtype=np.complex)
        hamilton_potential = np.zeros((knt,knt), dtype=np.complex)
        hamilton_periodic = np.zeros((knt,knt), dtype=np.complex)
        potential = self.create_potential()
        for i in range(knt):
            wave_vector_i = 2.0*np.pi*(i-kn)/self.total_potential_length
            hamilton_kinetic[i,i] = 0.5*wave_vector_i**2
            plane_wave_i = self.get_plane_wave(i-kn) #np.exp(1j*wavevector_i*self.spatial_axis)/np.sqrt(self.total_potential_length)
            hamilton_periodic[i,i] = 0.5*q**2 - q*wave_vector_i
            for j in range(i,knt):
                # wave_vector_j = 2.0*np.pi*(j-self.kn)/self.total_potential_length
                plane_wave_j = np.conj(self.get_plane_wave(j-kn)) #np.exp(-1j*wave_vector_j*self.spatial_axis)/np.sqrt(self.total_potential_length)
                hamilton_potential[i,j] = self.total_potential_length*np.mean(plane_wave_j*potential*plane_wave_i)
                hamilton_potential[j,i] = hamilton_potential[i,j].conj()
        return hamilton_kinetic + hamilton_potential + hamilton_periodic

    def get_eigenstates(self,q=0.0):
        hamilton_matrix = self.get_plane_wave_hamiltonian(q=q)
        energy,coefficients = np.linalg.eigh(hamilton_matrix)
        return energy,coefficients

    def get_eigenvalues(self,q=0.0):
        hamilton_matrix = self.get_plane_wave_hamiltonian(q=q)
        energy = np.linalg.eigvalsh(hamilton_matrix)
        return energy

    def get_real_space_states(self,q=0.0,band_min_index=0,band_max_index=None):
        ''' Get a range of wave functions for wave vector q '''
        kn = self.positive_wavenumbers
        if band_max_index is None:
            band_max_index = band_min_index+1
        energy,coefficients = self.get_eigenstates(q=q)
        argument = 1j*2*np.pi*(np.arange(-kn,kn+1)[None,:,None]
                               *self.spatial_axis[:,None,None]) / self.total_potential_length
        wave_function = np.sum(np.exp(argument)*coefficients[None,:,band_min_index:band_max_index],axis=1)
        return np.sqrt(self.cell_repetitions)*wave_function/np.sqrt(np.sum(abs(wave_function)**2))

    def plot_potential(self,ax=None):
        if ax is None:
            fig,ax = plt.subplots()
        ax.plot(self.spatial_axis,au.eV*self.create_potential())
        ax.set_ylabel('Potential (eV)')

    def plot_band_structure(self,ax=None):
        if ax == None:
            fig,ax = plt.subplots()
        ax.plot(self.k_axis,[au.eV*self.get_eigenvalues(q) for q in self.k_axis])
        ax.set_ylabel('Energy (eV)')

    def plot_wave_function(self,band_index=0,wave_vector=0.0,ax=None):
        if ax == None:
            fig,ax = plt.subplots()
        wave_function = self.get_real_space_states(q=wave_vector, band_min_index=band_index)
        ax.plot(self.spatial_axis, abs(wave_function),linewidth=2)
        ax.plot(self.spatial_axis, np.real(wave_function),'--')
        ax.plot(self.spatial_axis, np.imag(wave_function),':')

    def check_real_space_eigenstate(self,q=0.0, band=0):
        ''' Check accuracy of solution by applying the Hamiltonian
            to the wave function in real space '''
        energy = self.get_eigenvalues(q=q)[band]
        wave_function = self.get_real_space_states(q=q,band_min_index=band)
        hamiltonian = self.get_real_space_hamiltonian(q=q)
        deviation = sum(abs(wave_function - np.dot(hamiltonian,wave_function)/energy))
        return deviation

    def plot_check_real_space_eigenstate(self,q=0.0, band=0):
        ''' Check accuracy of solution by applying the Hamiltonian
            to the wave function in real space '''
        energy = self.get_eigenvalues(q=q)[band]
        wave_function = self.get_real_space_states(q=q,band_min_index=band)
        hamiltonian = self.get_real_space_hamiltonian(q=q)
        deviation = sum(abs(wave_function - np.dot(hamiltonian,wave_function)/energy))
        plt.figure()
        plt.plot(abs(wave_function),'k')
        plt.plot(abs(np.dot(hamiltonian,wave_function))/abs(energy),'r')

        plt.plot(np.real(wave_function),'k--')
        plt.plot(np.real(np.dot(hamiltonian,wave_function))/(energy),'r--')

        plt.plot(np.imag(wave_function),'k:')
        plt.plot(np.imag(np.dot(hamiltonian,wave_function))/(energy),'r:')

        plt.figure()
        plt.plot(abs(wave_function)-abs(np.dot(hamiltonian,wave_function))/abs(energy))
        plt.show()
        return deviation

    def get_momentum_and_overlap_matrices(self,):
        nk = self.k_points
        nb = self.bands
        nx = self.total_spatial_points
        dx = self.spatial_step

        def get_momentum_operator():
            ''' Three point-stencil '''
            derivative_operator = np.zeros((nx, nx), dtype=np.complex)
            for i in range(nx):
                derivative_operator[i - 1, i] = -1 / (2 * dx)
                derivative_operator[i, i - 1] =  1 / (2 * dx)
            return -1j*derivative_operator
        derivative_operator = get_momentum_operator()

        def get_all_wave_functions():
            wave_functions = np.zeros((nk,nx,nb), dtype=np.complex)
            for k in range(nk):
                wave_functions[k] = self.get_real_space_states(self.k_axis[k],0,nb)

            for k in range(nk):
                for i in range(nb):
                    wave_functions[k,:,i] = wave_functions[k,:,i]/np.sqrt(np.dot(wave_functions[k,:,i].conj(),wave_functions[k,:,i]))
            return wave_functions
        wave_functions = get_all_wave_functions()

        def get_momentum_matrix_elements():
            momentum = np.zeros((nk, nb, nb), dtype=np.complex)
            for k in range(nk):
                for i in range(nb):
                    momentum[k,i,i] += self.k_axis[k]*np.dot(wave_functions[k,:,i].conj(),wave_functions[k,:,i])
                    for j in range(nb):
                        momentum[k,i,j] += np.dot(wave_functions[k,:,i].conj(),np.dot(derivative_operator,wave_functions[k,:,j]))
            return momentum

        def get_overlap_matrix_elements():
            overlap = np.zeros((nk, nb, nb), dtype=np.complex)
            for k in range(nk):
                states1 = wave_functions[k]
                if k < nk-1:
                    states2 = wave_functions[k+1]
                elif k == nk-1:
                    #states2 = wave_functions[0]*(np.exp(-1j*self.spatial_axis*2*np.pi/self.total_potential_length)[:,None])
                    states2 = wave_functions[0]*(np.exp(1j * self.spatial_axis * 2 * np.pi / self.total_potential_length)[:, None])
                for i in range(nb):
                    for j in range(nb):
                        overlap[k,i,j] = np.dot(states1[:,i].conj(),states2[:,j])
            return overlap
        return get_momentum_matrix_elements(), get_overlap_matrix_elements()

    def save_as_medium_to_hdf5(self,output_file):
        ''' Generate data needed for a complete Medium HDF5 file '''
        nk,nb = self.k_points, self.bands
        energy = np.array([self.get_eigenvalues(q)[:nb] for q in self.k_axis])
        momentum = np.zeros((nk,nb,nb,3), dtype=np.complex)
        overlap = np.zeros((nk,3,2,nb,nb), dtype=np.complex)
        momentum1d,overlap1d = self.get_momentum_and_overlap_matrices()

        momentum[:,:,:,0] = np.copy(momentum1d)
        overlap[:,0,0,:,:] = np.copy(overlap1d)
        overlap[:,0,-1,:,:] = np.array([overlap1d[q-1,:,:].conj().T for q in range(nk)])
        overlap[:,1,0,:,:] = np.array([np.eye(nb) for _ in range(nk)])
        overlap[:,1,1,:,:] = np.array([np.eye(nb) for _ in range(nk)])
        overlap[:,2,0,:,:] = np.array([np.eye(nb) for _ in range(nk)])
        overlap[:,2,1,:,:] = np.array([np.eye(nb) for _ in range(nk)])

        valence_bands = self.valence_bands
        size = np.array([nk,1,1])
        spin_factor = self.spin_factor

        lv1 = np.array([self.total_potential_length,0,0])
        lv2 = np.array([0,1,0])
        lv3 = np.array([0,0,1])
        lattice = np.array([lv1,lv2,lv3]).T
        klist1d = np.zeros((self.k_points,3))
        klist1d[:,0] = np.linspace(0.0,1.0,nk+1)[:-1]

        hdf5 = Hdf5Interface(energy=energy, klist1d=klist1d, lattice_vectors=lattice,
                             momentum=momentum,overlap=overlap,valence_bands=valence_bands,
                             spin_factor=spin_factor, size=size)
        hdf5.save(output_file,'w')


if __name__ == "__main__":
    medium = LocalPotential1D()
    medium.plot_potential()
    plt.show()
    medium.plot_wave_function()
    plt.show()
    medium.save_as_medium_to_hdf5('test1d.hdf5')
    print(medium.check_real_space_eigenstate(0,0))
    print(medium.check_real_space_eigenstate(0,1))
    print(medium.check_real_space_eigenstate(0.2,0))
    medium.plot_check_real_space_eigenstate()
