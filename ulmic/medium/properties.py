import numpy as np
import logging
from ulmic.medium.manipulator import MediumManipulator

try:
    from mpi4py import MPI
    MPI4PY_INSTALLED = True
except:
    MPI4PY_INSTALLED = False

TOLERANCE_DEGENERACY = 1e-3


class MediumProperties(MediumManipulator):

    def __init__(self, input_file, **kwargs):
        """ Class for manipulating properties of the
            medium.
            [Should only change for numerical or
            physical reasons]
            input_file:     input hdf5 file
        """

        super(MediumProperties, self).__init__(input_file, **kwargs)

    def get_direct_band_gap(self, ):
        """ Calculate direct band gap. """
        self.direct_gap = np.min(self.energy[:, self.nv] - self.energy[:, self.nv - 1])
        return self.direct_gap

    def get_indirect_band_gap(self, ):
        """ Calculate indirect band gap. """
        self.indirect_gap = np.min(self.energy[:, self.nv]) - np.max(self.energy[:, self.nv - 1])
        return self.indirect_gap

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
        berry_curvature = np.zeros((self.nk,3,3,), dtype=np.complex)
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

    def calculate_inverse_mass(self, gap_threshold=1e-3, \
        finite_difference_at_Gamma_point=True):
        """ Compute the inverse-mass tensor at each crystal momentum in each band.

        CURRENTLY, THIS FUNCTION WORKS ONLY IN THREE SPATIAL DIMENSIONS

        Parameters
        ----------
        gap_threshold: (float) terms where the denominator is smaller than this
            value will be evaluated using the finite-difference method from the
            available diagonal momentum-matrix elements;
        finite_difference_at_Gamma_point: (bool) if True and the Gamma point (k=0)
            belongs to the k-grid, then the effective masses at the Gamma point
            will be evaluated with "numerical_inverse_mass", that is, by the
            numerical differentiation of band energies obtained by a numerical
            version of the k-dot-p method.

        Returns
        -------
        inverse_mass: (nk, nb, ndims, ndims)
        """
        ndims = self.momentum.shape[-1] # the number of dimensions
        inverse_mass = np.zeros([self.nk, self.nb, ndims, ndims])
        numerator   = np.empty(self.nk, dtype=np.complex)
        denominator = np.empty(self.nk, dtype=np.complex)
        FD_inverse_mass = self.finite_difference_inverse_mass()
        for n in range(self.nb):
            denominator = self.energy[:, n].reshape((self.nk, 1)) - self.energy
            denominator_is_small = (np.abs(denominator) <= gap_threshold)
            denominator[denominator_is_small] = 1e+10
            for n1 in range(self.nb):
                if n == n1:
                    continue
                for alpha in range(ndims):
                    for beta in range(alpha + 1):
                        p_beta  = self.momentum[...,beta]
                        p_alpha = self.momentum[...,alpha]
                        numerator = 2 * np.real(self.momentum[:, n, n1, alpha] * \
                            self.momentum[:, n1, n, beta])
                        numerator[denominator_is_small[:, n1]] = 0.0
                        inverse_mass[:, n, alpha, beta] += numerator / denominator[:, n1]
            for alpha in range(ndims):
                inverse_mass[:, n, alpha, alpha] += 1.0
            # replace the "unreliable" data with that from FD_inverse_mass
            denominator_is_small = np.any(denominator_is_small, axis=1) # (nk,)
            for alpha in range(ndims):
                for beta in range(alpha + 1):
                    inverse_mass[denominator_is_small, n, alpha, beta] = \
                        FD_inverse_mass[denominator_is_small, n, alpha, beta]
            # if necessary, give the Gamma point a special treatment
            if finite_difference_at_Gamma_point:
                # check if the Gamma point belongs to the grid
                epsilon = np.finfo(np.float).eps # a negligibly small real number
                k_magnitudes = np.sqrt(np.sum(self.klist1d**2, axis=1))
                i0 = np.argmin(k_magnitudes)
                if k_magnitudes[i0] <= 10 * epsilon:
                    # differentiate band energies numerically
                    inverse_mass[i0, ...] = self.numerical_inverse_mass(i0)
            # initialize the other half of the inverse mass tensor
            for alpha in range(ndims):
                for beta in range(alpha + 1, ndims):
                    inverse_mass[:, n, alpha, beta] = inverse_mass[:, n, beta, alpha]
        return np.real(inverse_mass)


    def numerical_inverse_mass(self, ik, dk=1e-5):
        """ Compute the inverse-mass tensor (in each band) at the given crystal momentum.

        CURRENTLY, THIS FUNCTION WORKS ONLY IN THREE SPATIAL DIMENSIONS

        Parameters
        ----------
        ik: (int) the index of the crystal momentum in self.klist1d;
        dk: (float) the step size for differentiating with respect to the
            crystal momentum;

        Returns
        -------
        inverse_mass: (nb, ndims, ndims)
        """
        ndims = self.momentum.shape[-1]  # number of dimensions
        inverse_mass = np.empty((self.nb, ndims, ndims))
        H0 = np.diag(self.energy[ik, :]+0.5*dk**2).astype(np.complex)
        Ek = np.zeros((self.nb, 3, 3, 3))
        pk = self.momentum[ik, ...]
        Ek[:, 1, 1, 1] = self.energy[ik, :]
        Ek[:, 0, 1, 1] = np.sort(np.linalg.eigvals(H0 - dk * pk[:, :, 0]).real)
        Ek[:, 2, 1, 1] = np.sort(np.linalg.eigvals(H0 + dk * pk[:, :, 0]).real)
        Ek[:, 1, 0, 1] = np.sort(np.linalg.eigvals(H0 - dk * pk[:, :, 1]).real)
        Ek[:, 1, 2, 1] = np.sort(np.linalg.eigvals(H0 + dk * pk[:, :, 1]).real)
        Ek[:, 1, 1, 0] = np.sort(np.linalg.eigvals(H0 - dk * pk[:, :, 2]).real)
        Ek[:, 1, 1, 2] = np.sort(np.linalg.eigvals(H0 + dk * pk[:, :, 2]).real)
        Ek[:, 0, 0, 1] = np.sort(np.linalg.eigvals(H0 - dk * pk[:, :, 0] - dk * pk[:, :, 1]).real)
        Ek[:, 2, 2, 1] = np.sort(np.linalg.eigvals(H0 + dk * pk[:, :, 0] + dk * pk[:, :, 1]).real)
        Ek[:, 0, 1, 0] = np.sort(np.linalg.eigvals(H0 - dk * pk[:, :, 0] - dk * pk[:, :, 2]).real)
        Ek[:, 2, 1, 2] = np.sort(np.linalg.eigvals(H0 + dk * pk[:, :, 0] + dk * pk[:, :, 2]).real)
        Ek[:, 1, 0, 0] = np.sort(np.linalg.eigvals(H0 - dk * pk[:, :, 1] - dk * pk[:, :, 2]).real)
        Ek[:, 1, 2, 2] = np.sort(np.linalg.eigvals(H0 + dk * pk[:, :, 1] + dk * pk[:, :, 2]).real)
        Ek[:, 0, 2, 1] = np.sort(np.linalg.eigvals(H0 - dk * pk[:, :, 0] + dk * pk[:, :, 1]).real)
        Ek[:, 2, 0, 1] = np.sort(np.linalg.eigvals(H0 + dk * pk[:, :, 0] - dk * pk[:, :, 1]).real)
        Ek[:, 0, 1, 2] = np.sort(np.linalg.eigvals(H0 - dk * pk[:, :, 0] + dk * pk[:, :, 2]).real)
        Ek[:, 2, 1, 0] = np.sort(np.linalg.eigvals(H0 + dk * pk[:, :, 0] - dk * pk[:, :, 2]).real)
        Ek[:, 1, 0, 2] = np.sort(np.linalg.eigvals(H0 - dk * pk[:, :, 1] + dk * pk[:, :, 2]).real)
        Ek[:, 1, 2, 0] = np.sort(np.linalg.eigvals(H0 + dk * pk[:, :, 1] - dk * pk[:, :, 2]).real)
        # evaluate the diagonal elements of the inverse mass tensor
        inverse_mass[:, 0, 0] = Ek[:, 2, 1, 1] - 2 * Ek[:, 1, 1, 1] + Ek[:, 0, 1, 1]
        inverse_mass[:, 1, 1] = Ek[:, 1, 2, 1] - 2 * Ek[:, 1, 1, 1] + Ek[:, 1, 0, 1]
        inverse_mass[:, 2, 2] = Ek[:, 1, 1, 2] - 2 * Ek[:, 1, 1, 1] + Ek[:, 1, 1, 0]
        # evaluate the off-diagonal elements of the inverse mass tensor
        inverse_mass[:, 0, 1] = 0.25 * (Ek[:, 2, 2, 1] - Ek[:, 2, 0, 1] - \
            Ek[:, 0, 2, 1] + Ek[:, 0, 0, 1])
        inverse_mass[:, 1, 0] = inverse_mass[:, 0, 1]
        inverse_mass[:, 0, 2] = 0.25 * (Ek[:, 2, 1, 2] - Ek[:, 2, 1, 0] - \
            Ek[:, 0, 1, 2] + Ek[:, 0, 1, 0])
        inverse_mass[:, 2, 0] = inverse_mass[:, 0, 2]
        inverse_mass[:, 1, 2] = 0.25 * (Ek[:, 1, 2, 2] - Ek[:, 1, 2, 0] - \
            Ek[:, 1, 0, 2] + Ek[:, 1, 0, 0])
        inverse_mass[:, 2, 1] = inverse_mass[:, 1, 2]
        # finalize
        inverse_mass /= dk**2
        return inverse_mass


    def finite_difference_inverse_mass(self,):
        """ Calculate the derivative of the diagonal momentum matrix elements.

        This function is deprecated--calculate_inverse_mass should be used instead.

        Returns
        -------
        inverse_mass: (nk, nb, 3, 3)
        """
        if self.nk_eval == self.nk_vol and hasattr(self,'momentum'):
            inverse_mass_cartesian1d = np.zeros((self.nk,self.nb,3,3))
            inverse_mass_cartesian3d = np.zeros(tuple(self.size)+(self.nb,3,3))
            cart2red = np.linalg.inv(self.reciprocal_vectors).T
            for i in range(self.nb):
                dp_d1 = 0.5*(self.size[0])*(
                            +np.roll(self.momentum[self.klist3d[:,:,:],i,i,:],-1,0)
                            -np.roll(self.momentum[self.klist3d[:,:,:],i,i,:],1,0))

                dp_d2 = 0.5*(self.size[1])*(
                            +np.roll(self.momentum[self.klist3d[:,:,:],i,i,:],-1,1)
                            -np.roll(self.momentum[self.klist3d[:,:,:],i,i,:],1,1))

                dp_d3 = 0.5*(self.size[2])*(
                            +np.roll(self.momentum[self.klist3d[:,:,:],i,i,:],-1,2)
                            -np.roll(self.momentum[self.klist3d[:,:,:],i,i,:],1,2))

                dp_dx = dp_d1*cart2red[0,0] + dp_d2*cart2red[0,1] + dp_d3*cart2red[0,2]
                dp_dy = dp_d1*cart2red[1,0] + dp_d2*cart2red[1,1] + dp_d3*cart2red[1,2]
                dp_dz = dp_d1*cart2red[2,0] + dp_d2*cart2red[2,1] + dp_d3*cart2red[2,2]

                inverse_mass_cartesian3d[...,i,:,0] = np.copy(dp_dx.real)
                inverse_mass_cartesian3d[...,i,:,1] = np.copy(dp_dy.real)
                inverse_mass_cartesian3d[...,i,:,2] = np.copy(dp_dz.real)

            for i1 in range(self.size[0]):
                for i2 in range(self.size[1]):
                        for i3 in range(self.size[2]):
                            inverse_mass_cartesian1d[self.klist3d[i1,i2,i3],...] = \
                                inverse_mass_cartesian3d[i1,i2,i3,...]
            # enforce the symmetry
            inverse_mass_cartesian1d = 0.5 * (inverse_mass_cartesian1d + \
                inverse_mass_cartesian1d.swapaxes(2, 3))
            return inverse_mass_cartesian1d, inverse_mass_cartesian3d
        else:
            logging.warning('Cannot calculate derivative of diagonal'
                            'momentum matrix elements'
                            'for incomplete Brillouin zone.')


    def get_diagonal_momentum(self,):
        momentum_diagonal = np.zeros(self.energy.shape+(3,))
        for i in range(self.nk_eval):
            for j in range(3):
                momentum_diagonal[i,:,j] = np.diag(self.momentum[i,:,:,j]).real
        return momentum_diagonal
