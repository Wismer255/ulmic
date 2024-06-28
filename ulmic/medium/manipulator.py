import numpy as np
import logging
from ulmic.external import nearest_neighbor_table
from ulmic.medium.loadhdf5 import LoadHdf5
import gc
# try:
#     from mpi4py import MPI
#     MPI4PY_INSTALLED = True
# except:
#     MPI4PY_INSTALLED = False

TOLERANCE_DEGENERACY = 1e-3


class MediumManipulator(LoadHdf5):

    def __init__(self, input_file, **kwargs):
        """ Class for manipulating properties of the
            medium.
            [Should only change for numerical or
            physical reasons]
            input_file:     input hdf5 file
        """

        fix_hermiticity = kwargs.get('fix_hermiticity', True)
        fix_overlap = kwargs.get('fix_overlap', True)
        write_summary_to_log = kwargs.get('log',True)
        super(MediumManipulator, self).__init__(input_file,
                                             **kwargs)

        self.get_direct_band_gap()
        self.get_indirect_band_gap()

        if hasattr(self,'momentum'):
            self.check_hermiticity()
            if fix_hermiticity:
                self.fix_hermiticity()

        if hasattr(self,'overlap') and fix_overlap:
            self.fix_overlap()

        if write_summary_to_log:
            logging.info('{} loaded as Medium.'.format(self.input_file))

    def check_hermiticity(self,):
        """ Calculate how much the numerically obtained momentum matrix
            deviate from a hermitian matrix. """
        max_deviation = 0.0
        for i in range(self.nk_local):
            for j in range(3):
                deviation = np.max(np.abs(self.momentum[i,:,:,j]-self.momentum[i,:,:,j].conj().T))
                if deviation > max_deviation:
                    max_deviation = deviation
        logging.info('Hermiticity (max. deviation): {:e}'.format(max_deviation))

    def fix_hermiticity(self,):
        """ Ensure hermiticity of momentum matrix through addition. """
        for i in range(self.nk_local):
            for j in range(3):
                self.momentum[i,:,:,j] = 0.5*(self.momentum[i,:,:,j] + self.momentum[i,:,:,j].conj().T)

    def fix_overlap(self,):
        """ Perform singular value decomposition and normalize
            all singular values. """
        nk,ndim,nn,_,_ = self.overlap.shape
        for i in range(self.nk_local):
            for j in range(ndim):
                for k in range(nn):
                    U,S,Vh = np.linalg.svd(self.overlap[i,j,k,:,:])
                    self.overlap[i,j,k,:,:] = np.dot(U,Vh)

    def set_min_band_index(self, nb_min):
        """ Discard bands below nb_min """
        self.nb -= nb_min
        self.nv -= nb_min
        if hasattr(self,'energy'):
            self.energy = self.energy[:, nb_min:]
        if hasattr(self, 'momentum'):
            self.momentum = self.momentum[:,nb_min:,nb_min:,:]
        if hasattr(self,'overlap'):
            self.overlap = self.overlap[:,:,:,nb_min:,nb_min:]
        # free the memory
        gc.collect()

    def set_max_band_index(self,nb_max):
        """ Discard bands above nb_max (including nb_max) """
        self.nb = min(nb_max,self.nb)
        if hasattr(self,'energy'):
            self.energy = self.energy[:,:nb_max]
        if hasattr(self, 'momentum'):
            self.momentum = self.momentum[:,:nb_max,:nb_max,:]
        if hasattr(self, 'overlap'):
            self.overlap = self.overlap[:,:,:,:nb_max,:nb_max]
        # free the memory
        gc.collect()

    def get_direct_band_gap(self, ):
        """ Calculate direct band gap. """
        self.direct_gap = np.min(self.energy[:, self.nv] - self.energy[:, self.nv - 1])
        return self.direct_gap

    def get_indirect_band_gap(self, ):
        """ Calculate indirect band gap. """
        self.indirect_gap = np.min(self.energy[:, self.nv]) - np.max(self.energy[:, self.nv - 1])
        return self.indirect_gap

    def calculate_expanded_neighbour_table(self,nn=2):
        """ Update number of neighbors in nn_table. """
        self.neighbour_table = nearest_neighbor_table(self.klist3d,nn)

    def calculate_second_neighbour_overlaps(self,directions='both'):
        """ Calculate overlap for next-nearest neighbors of all
            'inner' k-points in all directions.
            Currently only supports nn = 2."""
        nn = 2
        old_overlaps = np.copy(self.overlap)
        self.overlap = np.zeros((self.nk,3,4,self.nb,self.nb), dtype=complex)
        self.overlap[:,:,0,:,:] = np.copy(old_overlaps[:,:,0,:,:])
        self.overlap[:,:,-1,:,:] = np.copy(old_overlaps[:,:,-1,:,:])
        for i in range(self.nk):
            i_global = self.k_slice_buffer[i]
            if i_global in self.k_inner:
                for alpha in range(3):
                    if directions == 'forward' or directions == 'both':
                        k_nn = self.neighbour_table[i_global, alpha, 0]
                        local_k_nn = self.global_to_local[str(k_nn)]
                        self.overlap[i,alpha,1,:,:] = np.dot(self.overlap[i,alpha,0,:,:],self.overlap[local_k_nn,alpha,0,:,:])
                    if directions == 'backward' or directions == 'both':
                        k_nn = self.neighbour_table[i_global, alpha, -1]
                        local_k_nn = self.global_to_local[str(k_nn)]
                        self.overlap[i,alpha,-2,:,:] = np.dot(self.overlap[local_k_nn,alpha,-1,:,:],self.overlap[i,alpha,-1,:,:])

    def set_k_point_reduction(self,nk_factorization,reduction='isotropic'):
        """ Reduce number of k-points in the Brillouin zone in structured ways.
            nk_factorization:   int or list of three integers
            reduction:          'isotropic' or 'X_line'
            """
        #TODO: Currently not compatible with MPI or any other partitioning of the BZ.

        if type(nk_factorization) == type(int()):
            nk_factorization = [nk_factorization]*3
        total_factor = nk_factorization[0]*nk_factorization[1]*nk_factorization[2]
        size = self.size

        if self.nk%(total_factor) == 0:
            i0 = self.klist3d[0,0,0]
            indices = np.rint(np.array(size)[np.newaxis,:] * (self.klist1d -
                self.klist1d[i0][np.newaxis,:])).astype(np.intp)
            # boolean_vector = np.isclose(indices % nk_factorization, 0.0)
            boolean_vector = (indices % nk_factorization == 0)
            if reduction=='isotropic':
                boolean_vector = np.array([np.all(q) for q in boolean_vector])
            if reduction=='X_line':
                for i in range(self.nk):
                    if indices[i,0] !=  indices[i,1] or indices[i,1] != indices[i,2]:
                        boolean_vector[i] = False
                boolean_vector = np.array([np.all(q) for q in boolean_vector])

            self.size = size/np.array(nk_factorization)
            self.klist1d = self.klist1d[boolean_vector]
            self.energy = self.energy[boolean_vector]
            self.momentum = self.momentum[boolean_vector]
            self.nk = np.sum(boolean_vector)
            self.nk_vol = self.nk
            self.nk_eval = self.nk
            self.nk_buffer = 0
            self.nk_local = self.nk

            # rest klist3d
            self.klist3d = np.zeros(self.size, dtype=np.intp)
            for i in range(self.nk):
                idx,idy,idz = indices[i,:]
                self.klist3d[idx,idy,idz] = i

            if hasattr(self, 'overlap'):
                nn = self.neighbour_table.shape[2]//2
                self.neighbour_table = nearest_neighbor_table(self.klist3d,nn)
                del self.overlap
                logging.warning('The method set_k_point_reduction was invoked,'
                                'but it does not support overlaps.'
                                'The overlap matrix is therefore removed to'
                                'avoid wrong results.')
        else:
            logging.warning("nk = ",self.nk)
            logging.warning("nk_factor = ",nk_factorization)
            logging.error('nk and nk_factor are not commensurable!')

    def get_effective_number_of_electrons(self,krange=None):
        """Check sum rules by calculating the effective number of electrons.
            Does not include second derivative of bands as this will be zero
            for a sufficiently fine k-mesh. """
        Neffx = 0.0
        Neffy = 0.0
        Neffz = 0.0
        if krange is None:
            krange = range(self.nk)

        for k in krange:
            for b1 in range(self.nv):
                for b2 in range(self.nv,self.nb):
                    Neffx += 2*(np.abs(self.momentum[k,b1,b2,0])**2)/((self.energy[k,b2]-self.energy[k,b1]))
                    Neffy += 2*(np.abs(self.momentum[k,b1,b2,1])**2)/((self.energy[k,b2]-self.energy[k,b1]))
                    Neffz += 2*(np.abs(self.momentum[k,b1,b2,2])**2)/((self.energy[k,b2]-self.energy[k,b1]))
        return np.array([Neffx,Neffy,Neffz])/self.nk

    def calculate_berry_curvature_for_band(self,band_index):
        berry_curvature = np.zeros((self.nk,3,3,), dtype=complex)
        #for i in range(self.nk):
        for k in range(self.nb):
            if k != band_index:
                #if np.abs(self.medium.energy[i,k]-self.medium.energy[i,band_index]) > TOLERANCE_DEGENERACY:
                for alpha in range(3):
                    for beta in range(3):
                        mask = np.abs(self.energy[:,k]-self.energy[:,band_index]) > TOLERANCE_DEGENERACY
                        berry_curvature[:,alpha,beta] += mask*(self.momentum[:,k,band_index,alpha]*
                                                          self.momentum[:,band_index,k,beta])/(
                                                           ((self.energy[:,k]-
                                                             self.energy[:,band_index])**2)
                                                       )
        berry_curvature = -2*np.imag(berry_curvature)
        return berry_curvature

    def calculate_diagonal_momentum(self,):
        """ Calculate the derivative of the energy bands. """
        if self.nk_eval == self.nk_vol:
            energy_derivative_cartesian = np.zeros(tuple(self.size)+(self.nb,3,))
            cart2red = np.linalg.inv(self.reciprocal_vectors).T
            for i in range(self.nb):
                dE_d1 = 0.5*(self.size[0])*(
                            +np.roll(self.energy[self.klist3d[:,:,:],i],-1,0)[:,:,:]
                            -np.roll(self.energy[self.klist3d[:,:,:],i],1,0)[:,:,:])

                dE_d2 = 0.5*(self.size[1])*(
                            +np.roll(self.energy[self.klist3d[:,:,:],i],-1,1)[:,:,:]
                            -np.roll(self.energy[self.klist3d[:,:,:],i],1,1)[:,:,:])

                dE_d3 = 0.5*(self.size[2])*(
                            +np.roll(self.energy[self.klist3d[:,:,:],i],-1,2)[:,:,:]
                            -np.roll(self.energy[self.klist3d[:,:,:],i],1,2)[:,:,:])

                dE_dx = dE_d1*cart2red[0,0] + dE_d2*cart2red[0,1] + dE_d3*cart2red[0,2]
                dE_dy = dE_d1*cart2red[1,0] + dE_d2*cart2red[1,1] + dE_d3*cart2red[1,2]
                dE_dz = dE_d1*cart2red[2,0] + dE_d2*cart2red[2,1] + dE_d3*cart2red[2,2]

                energy_derivative_cartesian[:,:,:,i,0] = np.copy(dE_dx)
                energy_derivative_cartesian[:,:,:,i,1] = np.copy(dE_dy)
                energy_derivative_cartesian[:,:,:,i,2] = np.copy(dE_dz)

            energy_derivative_cartesian1d = np.zeros((self.nk,self.nb,3))
            for i1 in range(self.size[0]):
                for i2 in range(self.size[1]):
                        for i3 in range(self.size[2]):
                            energy_derivative_cartesian1d[self.klist3d[i1,i2,i3]] = energy_derivative_cartesian[i1,i2,i3,:,:].real

            return energy_derivative_cartesian1d
        else:
            logging.warning('Cannot calculate derivative of energy bands'
                            'for incomplete Brilluoin zone.')

    def calculate_inverse_mass(self,):
        """ Calculate the derivative of the diagonal momentum matrix elements. """
        if self.nk_eval == self.nk_vol and hasattr(self,'momentum'):
            inverse_mass_cartesian1d = np.zeros((self.nk,self.nb,3,))
            inverse_mass_cartesian3d = np.zeros(tuple(self.size)+(self.nb,3,))
            cart2red = np.linalg.inv(self.reciprocal_vectors).T
            for i in range(self.nb):
                dp_d1 = 0.5*(self.size[0])*(
                            +np.roll(self.momentum[self.klist3d[:,:,:],i,i,0],-1,0)
                            -np.roll(self.momentum[self.klist3d[:,:,:],i,i,0],1,0))

                dp_d2 = 0.5*(self.size[1])*(
                            +np.roll(self.momentum[self.klist3d[:,:,:],i,i,1],-1,1)
                            -np.roll(self.momentum[self.klist3d[:,:,:],i,i,1],1,1))

                dp_d3 = 0.5*(self.size[2])*(
                            +np.roll(self.momentum[self.klist3d[:,:,:],i,i,2],-1,2)
                            -np.roll(self.momentum[self.klist3d[:,:,:],i,i,2],1,2))

                dp_dx = dp_d1*cart2red[0,0] + dp_d2*cart2red[0,1] + dp_d3*cart2red[0,2]
                dp_dy = dp_d1*cart2red[1,0] + dp_d2*cart2red[1,1] + dp_d3*cart2red[1,2]
                dp_dz = dp_d1*cart2red[2,0] + dp_d2*cart2red[2,1] + dp_d3*cart2red[2,2]

                inverse_mass_cartesian3d[:,:,:,i,0] = np.copy(dp_dx.real)
                inverse_mass_cartesian3d[:,:,:,i,1] = np.copy(dp_dy.real)
                inverse_mass_cartesian3d[:,:,:,i,2] = np.copy(dp_dz.real)

            for i1 in range(self.size[0]):
                for i2 in range(self.size[1]):
                        for i3 in range(self.size[2]):
                            inverse_mass_cartesian1d[self.klist3d[i1,i2,i3],:,:] = inverse_mass_cartesian3d[i1,i2,i3,:,:]

            return inverse_mass_cartesian1d, inverse_mass_cartesian3d
        else:
            logging.warning('Cannot calculate derivative of diagonal'
                            'momentum matrix elements'
                            'for incomplete Brilluoin zone.')

    def apply_scissors_correction(self, scissors_correction):
        """ Apply or re-apply scissors correction."""
        if not hasattr(self,'scissors_correction'):
            self.scissors_correction = 0.0
        self.energy[:, self.nv:] -= self.scissors_correction
        self.energy[:, self.nv:] += scissors_correction
        self.scissors_correction = scissors_correction
