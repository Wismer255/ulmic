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
                    Neffx += 2*(abs(self.momentum[k,b1,b2,0])**2)/((self.energy[k,b2]-self.energy[k,b1]))
                    Neffy += 2*(abs(self.momentum[k,b1,b2,1])**2)/((self.energy[k,b2]-self.energy[k,b1]))
                    Neffz += 2*(abs(self.momentum[k,b1,b2,2])**2)/((self.energy[k,b2]-self.energy[k,b1]))
        return np.array([Neffx,Neffy,Neffz])/self.nk

    def calculate_berry_curvature_for_band(self,band_index):
        berry_curvature = np.zeros((self.nk,3,3,),complex)
        #for i in range(self.nk):
        for k in range(self.nb):
            if k != band_index:
                #if abs(self.medium.energy[i,k]-self.medium.energy[i,band_index]) > TOLERANCE_DEGENERACY:
                for alpha in range(3):
                    for beta in range(3):
                        mask = abs(self.energy[:,k]-self.energy[:,band_index]) > TOLERANCE_DEGENERACY
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


    def get_diagonal_momentum(self,):
        momentum_diagonal = np.zeros(self.energy.shape+(3,))
        for i in range(self.nk_eval):
            for j in range(3):
                momentum_diagonal[i,:,j] = np.diag(self.momentum[i,:,:,j]).real
        return momentum_diagonal
