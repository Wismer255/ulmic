import numpy as np
from numba import jit, njit
import sys

from ulmic.inputs import flags
for arg in sys.argv:
    if arg in flags:
        flags[arg] = True

#@jit('void(complex128[:,:,:],int64,int64,float64[:,:],complex128[:,:,:,:],float64,int64,int64,float64,float64[:,:,:],float64[:,:])',nopython=True,nogil=True,cache=not flags['--no-cache'])
@njit(cache=not flags['--no-cache'])
def evaluate_current_jit_all(rho,index,energy3d,momentum3d,time,nk_range,nk_vol,volume,
                                    result_jk,result_j):
    """ Evaluate the expectation value of the velocity
        for every k in the
        interaction picture using wave functions. """

    normalisation = (1.0/(nk_vol*volume))
    evaluated_current = np.zeros(3,np.float64)
    for k in nk_range:
        v1 = np.exp(1j*time*energy3d[k,:])
        v2 = np.exp(-1j*time*energy3d[k,:])
        mask_int = np.outer(v1,v2)
        result_jk[index,k,0] = -( np.trace(np.dot(np.conj(rho[k]).T,np.dot(mask_int*momentum3d[k,:,:,0],rho[k])))).real*normalisation
        result_jk[index,k,1] = -( np.trace(np.dot(np.conj(rho[k]).T,np.dot(mask_int*momentum3d[k,:,:,1],rho[k])))).real*normalisation
        result_jk[index,k,2] = -( np.trace(np.dot(np.conj(rho[k]).T,np.dot(mask_int*momentum3d[k,:,:,2],rho[k])))).real*normalisation
        evaluated_current += result_jk[index,k,:]
    result_j[index, :] = evaluated_current

@njit(cache=not flags['--no-cache'])
def evaluate_current_jit(rho,index,energy3d,momentum3d,time,nk_range,nk_vol,volume,
                                    result_j):
    """ Evaluate the expectation value of the velocity in the
        interaction picture using wave functions. """
    result_jk = np.zeros(3,dtype=np.float64)
    normalisation = (1.0/(nk_vol*volume))
    nk,nb,nv = rho.shape
    evaluated_current = np.zeros(3, np.float64)
    for k in nk_range:
        v1 = np.exp( 1j*time*energy3d[k,:])
        v2 = np.exp(-1j*time*energy3d[k,:])
        mask_int = np.outer(v1,v2)
        for i in range(nv):
            norm = np.dot(np.conj(rho[k,:,i]),rho[k,:,i]).real
            result_jk[0] = -( np.dot(np.conj(rho[k,:,i]),np.dot(mask_int*momentum3d[k,:,:,0],rho[k,:,i]))).real*normalisation/norm
            result_jk[1] = -( np.dot(np.conj(rho[k,:,i]),np.dot(mask_int*momentum3d[k,:,:,1],rho[k,:,i]))).real*normalisation/norm
            result_jk[2] = -( np.dot(np.conj(rho[k,:,i]),np.dot(mask_int*momentum3d[k,:,:,2],rho[k,:,i]))).real*normalisation/norm
            evaluated_current += result_jk
    result_j[index,:] = evaluated_current


#@jit('void(complex128[:,:,:],int64,int64,complex128[:,:],complex128[:,:,:,:],float64,int64,int64,float64,float64[:,:,:],float64[:,:])',nopython=True,nogil=True,cache=not flags['--no-cache'])
@njit(cache=not flags['--no-cache'])
def evaluate_acceleration_jit(rho,index,energy,momentum3d,time,nk_range,nk_vol,volume,
                                    result_j):
    """ Evaluate the expectation value of the acceleration in the
        interaction picture using wave functions. """
    energy3d = energy.astype(np.complex128)
    result_jk = np.zeros(3,dtype=np.float64)
    normalisation = (1.0/(nk_vol*volume))
    evaluated_current = np.zeros(3, np.float64)
    for k in nk_range:
        v1 = np.exp(1j*time*energy3d[k,:])
        v2 = np.exp(-1j*time*energy3d[k,:])
        mask_int = np.outer(v1,v2)
        result_jk[0] = -( np.trace(np.dot(np.conj(rho[k]).T,np.dot(mask_int*1j*(np.dot(momentum3d[k,:,:,0],np.diag(energy3d[k,:]))-np.dot(np.diag(energy3d[k,:]),momentum3d[k,:,:,0])  ),rho[k])))).real*normalisation
        result_jk[1] = -( np.trace(np.dot(np.conj(rho[k]).T,np.dot(mask_int*1j*(np.dot(momentum3d[k,:,:,1],np.diag(energy3d[k,:]))-np.dot(np.diag(energy3d[k,:]),momentum3d[k,:,:,1])  ),rho[k])))).real*normalisation
        result_jk[2] = -( np.trace(np.dot(np.conj(rho[k]).T,np.dot(mask_int*1j*(np.dot(momentum3d[k,:,:,2],np.diag(energy3d[k,:]))-np.dot(np.diag(energy3d[k,:]),momentum3d[k,:,:,2])  ),rho[k])))).real*normalisation
        evaluated_current += result_jk
    result_j[index,:] = evaluated_current

#@jit('void(complex128[:,:,:],int64,int64,int64,int64,float64[:,:],complex128[:,:,:,:],float64,int64,int64,float64,float64[:],float64[:],float64[:,:])',nopython=True,nogil=True,cache=not flags['--no-cache'])
@njit(cache=not flags['--no-cache'])
def evaluate_electrons_jit(rho,nb,nv,index,energy3d,momentum3d,time,nk_range,nk_vol,volume,
                                    result_q,result_n,result_pop):
    """ Evaluate the total number of electrons, number of electrons in
        the conduction band states, and number of electrons in
        each conduction band using wave functions. """
    normalisation = (1.0/(nk_vol*volume))
    electron_number = 0.0
    excitation_number = 0.0
    conduction_band_population = np.zeros(nb-nv,np.float64)
    for k in nk_range:
        for j in range(nv):
            for i in range(nb):
                electron_number     += (abs(rho[k,i,j])**2)*normalisation
            for i in range(nv,nb):
                conduction_band_population[i-nv] += (abs(rho[k,i,j])**2)*normalisation
                excitation_number   += (abs(rho[k,i,j])**2)*normalisation
    result_q[index] = electron_number
    result_n[index] = excitation_number
    result_pop[index] = conduction_band_population

#@jit('void(complex128[:,:,:],int64,int64,int64,int64,float64[:,:],complex128[:,:,:,:],float64,int64,int64,float64,float64[:])',nopython=True,nogil=True,cache=not flags['--no-cache'])
@njit(cache=not flags['--no-cache'])
def evaluate_energy_jit(rho,nb,nv,index,energy3d,momentum3d,time,nk_range,nk_vol,volume,
                                    result_e):
    """ Evaluate the expectation value of the field-free
        Hamiltonian using wave functions. """
    normalisation = (1.0/(nk_vol*volume))
    absorbed_energy = 0
    for k in nk_range:
        for j in range(nv):
            norm = 1.0
            for i in range(nb):
                absorbed_energy += ((abs(rho[k,i,j])**2)*energy3d[k,i]/norm)*normalisation
    result_e[index] = absorbed_energy

#@jit('float64[:,:](complex128[:,:,:],float64,complex128[:,:,:,:,:],int64[:,:,:],complex128[:,:],int64,float64[:,:],int64[:],float64)',nopython=True,nogil=True,cache=not flags['--no-cache'])
@njit(cache=not flags['--no-cache'])
def evaluate_covariant_current_jit(rho,time,S,table,energy3d,nk,lattice_vectors,nk_periodicity,volume):
    current = np.zeros((nk,3))
    for k in range(nk):
        rho0 = rho[k,:,:]
        v1 = np.exp(1j*energy3d[k,:]*time)
        for alpha in range(3):
            k1 = table[k,alpha,0]
            k2 = table[k,alpha,-1]
            rho1 = rho[k1,:,:]
            rho2 = rho[k2,:,:]

            v2_1 = np.exp(-1j*energy3d[k1,:]*time)
            mask1 = np.outer(v1,v2_1)
            v2_2 = np.exp(-1j*energy3d[k2,:]*time)
            mask2 = np.outer(v1,v2_2)

            S1 = np.dot(np.conj(rho0).T,np.dot(mask1*S[k,alpha,0],rho1))
            Sinv1 = np.linalg.pinv(S1)
            cov1 = np.dot(rho1,Sinv1)

            S2 = np.dot(np.conj(rho0).T,np.dot(mask2*S[k,alpha,-1],rho2))
            Sinv2 = np.linalg.pinv(S2)
            cov2 = np.dot(rho2,Sinv2)

            current[k,alpha] = 2*(np.trace(np.dot(np.conj(rho0).T, np.dot(np.diag(energy3d[k,:]), np.dot(mask1*S[k,alpha,0],cov1)-np.dot(mask2*S[k,alpha,-1],cov2)) ))).real
    for k in range(nk):
        current[k,:] = np.dot(lattice_vectors,current[k,:])
    current[:,0] /= nk_periodicity[1]*nk_periodicity[2]
    current[:,1] /= nk_periodicity[0]*nk_periodicity[2]
    current[:,2] /= nk_periodicity[0]*nk_periodicity[1]
    return current/(4*np.pi*volume)


#@jit('UniTuple(float64[:,:],2)(complex128[:,:,:],float64,complex128[:,:,:,:],complex128[:,:,:,:,:],int64[:,:,:],complex128[:,:],int64,float64[:,:],int64[:],int64[:,:,:],int64,float64,int64,int64)',nopython=True,nogil=True,cache=not flags['--no-cache'])
@njit(cache=not flags['--no-cache'])
def evaluate_current_and_neff_using_berry(rho,time,momentum,S,table,energy,nk_range,lattice_vectors,periodicity,klist3d,nk_vol,volume,nv,neighbor_order=1):
    term_current = np.zeros((len(nk_range),3),np.float64)
    term_momentum = np.zeros((len(nk_range),3),np.float64)

    _,_,n3 = rho.shape
    if n3 == nv:
        PureState = True
    else:
       PureState = False

    for k0 in nk_range:
        for alpha in range(3):
            k_nn = table[k0,alpha,neighbor_order-1]
            v1 = np.exp(1j*energy[k0,:]*time)
            v2 = np.exp(-1j*energy[k_nn,:]*time)
            mask = np.outer(v1,v2)
            #Differential part:
            derivative_H = (
                          np.dot(np.conj(rho[k0,:,:]).T,np.dot(mask*S[k0,alpha,neighbor_order-1], np.dot(np.diag(energy[k_nn,:]),rho[k_nn,:,:])))
                         -np.dot(np.dot(np.conj(rho[k0,:,:]).T,np.diag(energy[k0,:])),np.dot(mask*S[k0,alpha,neighbor_order-1], rho[k_nn,:,:]))
                            )
            #Inverse part:
            overlap_product = (
                        np.dot(np.conj(rho[k0,:,:]).T,np.dot(mask*S[k0,alpha,neighbor_order-1],rho[k_nn,:,:])))

            derivative_p = np.linalg.det(
                          np.dot(np.conj(rho[k0,:,:]).T,np.dot(mask*S[k0,alpha,neighbor_order-1], np.dot(momentum[k_nn,:,:,alpha],rho[k_nn,:,:])))
                         -np.dot(np.dot(np.conj(rho[k0,:,:]).T,momentum[k0,:,:,alpha]),np.dot(mask*S[k0,alpha,neighbor_order-1], rho[k_nn,:,:]))
                            )

            if PureState:
                 term_current[k0,alpha] = ( np.trace( np.dot( np.dot(np.conj(rho[k0,:,:]).T,rho[k0,:,:]), np.dot(np.linalg.inv(overlap_product),derivative_H))) ).real
            else:
                VV,SS,UU = np.linalg.svd(overlap_product)
                normalized_SS = np.ones(SS.shape,np.complex128)
                for q in range(len(SS)):
                    if SS[q] > 1e-20:
                        normalized_SS[q] = np.sqrt(1/SS[q])
                normalized_pseudo_inverse = np.dot(np.conj(UU).T,np.dot(np.diag(normalized_SS),np.conj(VV).T))
                term_current[k0,alpha] = ( np.trace(np.dot(normalized_pseudo_inverse,derivative_H)) ).real
    return term_current,term_momentum


def evaluate_current_using_geometric_phase(rho,time,momentum,overlap,forward_neighbour_table,energy,nk_range_eval,lattice_vectors,size,klist3d,nk_vol,volume,nv,neighbor_order=1,order=1):
    # Evaluate distributed covariant current to second order
    current_mixed_j1,current_mixed_neff1 = evaluate_current_and_neff_using_berry(rho,time,momentum,
                                                                                 overlap,
                                                                                 forward_neighbour_table,
                                                                                 energy.astype(np.complex128),nk_range_eval,
                                                                                 lattice_vectors,
                                                                                 size,klist3d,
                                                                                 nk_vol,volume,nv,1)
    if order == 1:
        mixed_j5 = (1/(2*np.pi*volume))*np.dot(lattice_vectors,
                                                                    np.sum(current_mixed_j1,axis=0)/np.array(
                                                                        [size[1]*size[2],
                                                                         size[0]*size[2],
                                                                         size[0]*size[1]]))
        return mixed_j5

    if order > 1:
        current_mixed_j2,current_mixed_neff2 = evaluate_current_and_neff_using_berry(rho,time,momentum,
                                                                                     overlap,
                                                                                     forward_neighbour_table,
                                                                                     energy.astype(np.complex128),nk_range_eval,
                                                                                     lattice_vectors,
                                                                                     size,klist3d,
                                                                                     nk_vol,volume,nv,2)

        mixed_j5 = (1/(2*np.pi*volume))*np.dot(lattice_vectors,np.sum(
        (4.0/3.0)*current_mixed_j1-(1.0/6.0)*current_mixed_j2,axis=0)/np.array(
        [size[1]*size[2],size[0]*size[2],size[0]*size[1]]))
        return mixed_j5

#========================= Density matrix ======================================
#@jit('void(complex128[:,:,:],int64,int64,float64[:,:],complex128[:,:,:,:],float64,int64,int64,float64,float64[:,:,:],float64[:,:])',nopython=True,nogil=True,cache=not flags['--no-cache'])
@njit(cache=not flags['--no-cache'])
def evaluate_lvn_current_jit_all(rho,index,energy3d,momentum3d,time,nk_range,nk_vol,volume,
                                    result_jk,result_j):
    normalisation = (1.0/(nk_vol*volume))
    current = np.zeros(3,np.float64)
    for k in nk_range:
        v1 = np.exp(1j*time*energy3d[k,:])
        v2 = np.exp(-1j*time*energy3d[k,:])
        mask_int = np.outer(v1,v2)
        result_jk[index,k,0] = -( np.trace(np.dot(mask_int*momentum3d[k,:,:,0],rho[k]))).real*normalisation
        result_jk[index,k,1] = -( np.trace(np.dot(mask_int*momentum3d[k,:,:,1],rho[k]))).real*normalisation
        result_jk[index,k,2] = -( np.trace(np.dot(mask_int*momentum3d[k,:,:,2],rho[k]))).real*normalisation
        current += result_jk[index,k,:]
    result_j[index, :] = current

@njit(cache=not flags['--no-cache'])
def evaluate_lvn_current_jit(rho,index,energy3d,momentum3d,time,nk_range,nk_vol,volume,
                                    result_j):
    result_jk = np.zeros(3,np.float64)
    normalisation = (1.0/(nk_vol*volume))
    current = np.zeros(3,np.float64)
    for k in nk_range:
        v1 = np.exp(1j*time*energy3d[k,:])
        v2 = np.exp(-1j*time*energy3d[k,:])
        mask_int = np.outer(v1,v2)
        result_jk[0] = -( np.trace(np.dot(mask_int*momentum3d[k,:,:,0],rho[k]))).real*normalisation
        result_jk[1] = -( np.trace(np.dot(mask_int*momentum3d[k,:,:,1],rho[k]))).real*normalisation
        result_jk[2] = -( np.trace(np.dot(mask_int*momentum3d[k,:,:,2],rho[k]))).real*normalisation
        current += result_jk
    result_j[index, :] = current


#@jit('void(complex128[:,:,:],int64,int64,complex128[:,:],complex128[:,:,:,:],float64,int64,int64,float64,float64[:,:,:],float64[:,:])',nopython=True,nogil=True,cache=not flags['--no-cache'])
@njit(cache=not flags['--no-cache'])
def evaluate_lvn_acceleration_jit(rho,index,energy,momentum3d,time,nk_range,nk_vol,volume,
                                    result_j):
    energy3d = energy.astype(np.complex128)
    result_jk = np.zeros(3,dtype=np.float64)
    normalisation = (1.0/(nk_vol*volume))
    current = np.zeros(3,np.float64)
    for k in nk_range:
        v1 = np.exp(1j*time*energy3d[k,:])
        v2 = np.exp(-1j*time*energy3d[k,:])
        mask_int = np.outer(v1,v2)
        #norm = np.trace(np.dot(np.conj(rho[k]).T,rho[k])).real/4.0
        result_jk[0] = -( np.trace(np.dot(mask_int*1j*(np.dot(momentum3d[k,:,:,0],np.diag(energy3d[k,:]))-np.dot(np.diag(energy3d[k,:]),momentum3d[k,:,:,0])  ),rho[k]))).real*normalisation#/norm
        result_jk[1] = -( np.trace(np.dot(mask_int*1j*(np.dot(momentum3d[k,:,:,1],np.diag(energy3d[k,:]))-np.dot(np.diag(energy3d[k,:]),momentum3d[k,:,:,1])  ),rho[k]))).real*normalisation#/norm
        result_jk[2] = -( np.trace(np.dot(mask_int*1j*(np.dot(momentum3d[k,:,:,2],np.diag(energy3d[k,:]))-np.dot(np.diag(energy3d[k,:]),momentum3d[k,:,:,2])  ),rho[k]))).real*normalisation#/norm
        current += result_jk
    result_j[index, :] = current

#@jit('void(complex128[:,:,:],int64,int64,int64,int64,float64[:,:],complex128[:,:,:,:],float64,int64,int64,float64,float64[:],float64[:],float64[:,:])',nopython=True,nogil=True,cache=not flags['--no-cache'])
@njit(cache=not flags['--no-cache'])
def evaluate_lvn_electrons_jit(rho,nb,nv,index,energy3d,momentum3d,time,nk_range,nk_vol,volume,
                                    result_q,result_n,result_pop):
    normalisation = (1.0/(nk_vol*volume))
    electron_number = 0.0
    excitation_number = 0.0
    conduction_band_populations = np.zeros(nb-nv,np.float64)
    for k in nk_range:
        for i in range(nb):
            electron_number += (rho[k,i,i].real)*normalisation
        for i in range(nv,nb):
            conduction_band_populations += (rho[k,i,i].real)*normalisation
            excitation_number       += (rho[k,i,i].real)*normalisation
    result_q[index] = electron_number
    result_n[index] = excitation_number
    result_pop[index,:] = conduction_band_populations

#@jit('void(complex128[:,:,:],int64,int64,int64,int64,float64[:,:],complex128[:,:,:,:],float64,int64,int64,float64,float64[:])',nopython=True,nogil=True,cache=not flags['--no-cache'])
@njit(cache=not flags['--no-cache'])
def evaluate_lvn_energy_jit(rho,nb,nv,index,energy3d,momentum3d,time,nk_range,nk_vol,volume,
                                    result_e):
    normalisation = (1.0/(nk_vol*volume))
    energy = 0.0
    for k in nk_range:
        for i in range(nb):
            energy += (rho[k,i,i].real)*energy3d[k,i]*normalisation
    result_e[index] = energy

def evaluate_adiabatic_corrections(time,medium,pulses):
    NAeff = np.zeros((len(time),3))
    for i in range(len(time)):
        A = pulses.eval_potential_fast(time[i])
        energy = medium.energy
        momentum = medium.momentum
        nk_vol = medium.nk_vol
        volume = medium.volume
        nv = medium.nv
        nk = medium.nk
        rho = np.zeros(1,1,1,complex)
        NAeff[i,:] = jit_get_correction(time[i],rho,A,energy,momentum,nk_vol,volume,nv,nk)

#@jit('float64[:](float64,complex128[:,:,:],float64[:],float64[:,:],complex128[:,:,:,:],int64,float64,int64,int64)',nopython=True,nogil=True,cache=not flags['--no-cache'])
@njit(cache=not flags['--no-cache'])
def jit_get_correction(time,rho,A,energy3d,momentum,nk_vol,volume,nv,nk_range):
    rho00 = np.zeros((rho[0].shape),np.complex128)
    for i in range(nv):
        rho00[i,i] = 1.0

    Neff = np.zeros(3,np.float64)
    for k in nk_range:
        v1 = np.exp(1j*time*energy3d[k,:])
        v2 = np.exp(-1j*time*energy3d[k,:])
        mask = np.outer(v1,v2)

        Hk_dk = (1.0+0.0*1j)*np.diag(energy3d[k]+ 0.5*np.dot(A,A))
        for i in range(3):
            Hk_dk += A[i]*momentum[k,:,:,i]
        eigval, eigvec = np.linalg.eigh(Hk_dk)
        U,Uh = eigvec,np.conj(eigvec).T

        # evaluate adiabatic currents
        rho_ad = np.dot(U,rho00)
        p_A_x = np.dot(U,np.dot(momentum[k,:,:,0],Uh))
        p_A_y = np.dot(U,np.dot(momentum[k,:,:,1],Uh))
        p_A_z = np.dot(U,np.dot(momentum[k,:,:,2],Uh))
        Neff[0] += np.trace(np.dot(np.conj(rho_ad).T, np.dot(p_A_x - momentum[k,:,:,0],rho_ad))).real
        Neff[1] += np.trace(np.dot(np.conj(rho_ad).T, np.dot(p_A_y - momentum[k,:,:,1],rho_ad))).real
        Neff[2] += np.trace(np.dot(np.conj(rho_ad).T, np.dot(p_A_z - momentum[k,:,:,2],rho_ad))).real
    return Neff / (nk_vol*volume)



#@jit('complex128[:,:](complex128[:,:,:],float64,complex128[:,:,:,:,:],int64[:,:,:],float64[:,:],int64,float64[:,:],int64[:],int64[:,:,:],int64,float64,int64,int64)',nopython=True,nogil=True,cache=not flags['--no-cache'])
@njit(cache=not flags['--no-cache'])
def evaluate_angles_for_polarisation(rho,time,S,table,energy,nk_range,lattice_vectors,periodicity,klist3d,nk_vol,volume,nv,neighbor_order=1):
    _,n,m = rho.shape
    phases = np.zeros((len(nk_range),3),np.float64)
    products = np.zeros((len(nk_range),3),np.complex128)
    if n == m:
        for k0 in nk_range:
            for alpha in range(3):
                k_nn = table[k0,alpha,neighbor_order-1]
                v1 = np.exp(1j*energy[k0,:]*time)
                v2 = np.exp(-1j*energy[k_nn,:]*time)
                mask = np.outer(v1,v2)
                eigenvalues = np.linalg.eigvals(np.dot(np.conj(rho[k0,:,:]).T,np.dot(mask*S[k0,alpha,neighbor_order-1],rho[k_nn,:,:])))
                products[k0,alpha] = np.prod(eigenvalues**np.abs(eigenvalues))
    else:
        for k0 in nk_range:
            for alpha in range(3):
                k_nn = table[k0,alpha,neighbor_order-1]
                v1 = np.exp(1j*energy[k0,:]*time)
                v2 = np.exp(-1j*energy[k_nn,:]*time)
                mask = np.outer(v1,v2)
                product = np.linalg.det(np.dot(np.conj(rho[k0,:,:]).T,np.dot(mask*S[k0,alpha,neighbor_order-1],rho[k_nn,:,:])))
                products[k0,alpha] = product
    return products
