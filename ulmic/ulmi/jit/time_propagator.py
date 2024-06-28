import numpy as np
from numpy.linalg import eigh
from numba import jit, njit, prange
from ulmic.inputs import flags, options
from scipy.linalg.blas import zhemm


import sys

## from ulmic.inputs import flags
## for arg in sys.argv:
##     if arg in flags:
##         flags[arg] = True

relative_error_min = 1e-30


#@jit('np.complex128[:,:](np.complex128[:,:],np.complex128[:,:])',nopython=True,nogil=True,cache=not flags['--no-cache'])
@njit
def lindblad_term(rho,operator):
     operator2 = np.dot(operator,operator)
     result = np.dot(operator,np.dot(rho,operator)) - 0.5*(np.dot(rho,operator2) + np.dot(operator2,rho))
     return result

#==================== Dormand-Prince45, velocity gauge, Stochastic TDSE ================================
@njit
def jit_step_vg_wavefunctions_k_dp45_stdse(k,t,dt,energy3d,momentum3d,rho,As,gamma):
    nb,nv = rho.shape
    n_cs = 7
    n,m = rho.shape
    drho = np.zeros((n_cs,n,m), dtype=np.complex128)
    cs = np.array([0.0,0.2,0.3,0.8,8.0/9.0,1.0,1.0])
    bs = np.zeros((n_cs,n_cs))
    bs[1,0] = 0.2
    bs[2,:2] = np.array([3.0/40.0, 9.0/40.0])
    bs[3,:3] = np.array([44.0/45.0, -56.0/15.0, 32.0/9.0])
    bs[4,:4] = np.array([19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0])
    bs[5,:5] = np.array([9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0])
    bs[6,:6] = np.array([35.0/384.0, 0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0])

    sol4 = np.array([35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0])
    sol5 = np.array([5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0])

    for i in range(n_cs):

        tmp_rho = 1.0*rho
        for j in range(i):
            tmp_rho += dt*bs[i,j]*drho[j]

        v1 = np.exp(1j*(t+cs[i]*dt)*energy3d[k,:])
        mask = np.outer(v1,np.conj(v1))
        Hamiltonian = (momentum3d[k,:,:,0]*As[i,0]
                       +momentum3d[k,:,:,1]*As[i,1]
                       +momentum3d[k,:,:,2]*As[i,2])*mask
        drho[i] = -1j*np.dot(Hamiltonian,tmp_rho)

        Z = np.diag(energy3d[k,:]).astype(np.complex128) + Hamiltonian
        non_hermitian = np.dot(Z, Z)
        drho[i] += -0.5*gamma*np.dot(non_hermitian, tmp_rho)


    out5 = np.zeros((n,m), dtype=np.complex128)
    error = np.zeros((n,m), dtype=np.complex128)
    for i in range(n_cs):
       out5 += dt*sol5[i]*drho[i]
       # out5 += dt*sol5[i]*drho[i]
       error += dt*(sol5[i]-sol4[i])*drho[i]

    absolute_error = np.sqrt(np.trace(np.dot(np.conj(error).T,error)).real)
    denominator = max(relative_error_min,np.sqrt(np.trace(np.dot(np.conj(out5).T,out5)).real))
    relative_error = absolute_error/denominator

    rho_out = np.zeros(rho.shape, dtype=np.complex128)
    jump_occured = False
    min_norm = 1.0
    for i in range(nv):
        norm = np.dot(np.conj(rho[:,i]+out5[:,i]),rho[:,i]+out5[:,i]).real
        jump = np.random.rand() > norm
        if min_norm > norm:
           min_norm = norm

        if jump:
            v1 = np.exp(1j*t*energy3d[k,:])
            mask = np.outer(v1,np.conj(v1))
            Hamiltonian = (momentum3d[k,:,:,0]*As[0,0]
                           +momentum3d[k,:,:,1]*As[0,1]
                           +momentum3d[k,:,:,2]*As[0,2])*mask
            rho_out[:,i] = np.dot(np.diag(energy3d[k,:]).astype(np.complex128)+Hamiltonian, rho[:,i])
            jump_occured = True
        else:
            rho_out[:,i] = rho[:,i] + out5[:,i]
        rho_out[:,i] /= np.sqrt(np.dot(np.conj(rho_out[:,i]),rho_out[:,i]))

    return rho_out, absolute_error, relative_error, jump_occured


#==================== Dormand-Prince45, velocity gauge, TDSE ================================
#@jit('Tuple((np.complex128[:,:],float64,float64))(int64,float64,float64,float64[:,:],np.complex128[:,:,:,:],np.complex128[:,:],float64[:,:])',nopython=True,nogil=True,cache=not flags['--no-cache'])
@njit
def jit_step_vg_wavefunctions_k_dp45(k,t,dt,energy3d,momentum3d,rho,As):

    n_cs = 7
    n,m = rho.shape
    drho = np.zeros((n_cs,n,m), dtype=np.complex128)
    cs = np.array([0.0,0.2,0.3,0.8,8.0/9.0,1.0,1.0])
    bs = np.zeros((n_cs,n_cs))
    bs[1,0] = 0.2
    bs[2,:2] = np.array([3.0/40.0, 9.0/40.0])
    bs[3,:3] = np.array([44.0/45.0, -56.0/15.0, 32.0/9.0])
    bs[4,:4] = np.array([19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0])
    bs[5,:5] = np.array([9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0])
    bs[6,:6] = np.array([35.0/384.0, 0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0])

    sol4 = np.array([35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0])
    sol5 = np.array([5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0])

    for i in range(n_cs):

        tmp_rho = 1.0*rho
        for j in range(i):
            tmp_rho += dt*bs[i,j]*drho[j]

        v1 = np.exp(1j*(t+cs[i]*dt)*energy3d[k,:])
        mask = np.outer(v1,np.conj(v1))
        drho[i] = -1j*np.dot((momentum3d[k,:,:,0]*As[i,0]
                             +momentum3d[k,:,:,1]*As[i,1]
                             +momentum3d[k,:,:,2]*As[i,2])*mask,tmp_rho)


    out5 = np.zeros((n,m), dtype=np.complex128)
    error = np.zeros((n,m), dtype=np.complex128)
    for i in range(n_cs):
       out5 += dt*sol5[i]*drho[i]
       # out5 += dt*sol5[i]*drho[i]
       error += dt*(sol5[i]-sol4[i])*drho[i]

    absolute_error = np.sqrt(np.trace(np.dot(np.conj(error).T,error)).real)
    denominator = max(relative_error_min,np.sqrt(np.trace(np.dot(np.conj(out5).T,out5)).real))
    relative_error = absolute_error/denominator
    return rho+out5, absolute_error, relative_error


#@jit('Tuple((np.complex128[:,:,:],float64,float64))(int64,float64,float64,float64[:,:],np.complex128[:,:,:,:],np.complex128[:,:,:],float64[:,:])',nopython=True,nogil=True,cache=not flags['--no-cache'])
@njit(parallel=True)
def jit_step_vg_wavefunctions_dp45(nk,t,dt,energy3d,momentum3d,rho,As):
    rho_out5 = np.zeros(rho.shape, dtype=np.complex128)
    absolute_error = np.zeros(nk)
    relative_error = np.zeros(nk)
    for k in prange(nk):
        rho_out5[k,:,:], ae, re = \
            jit_step_vg_wavefunctions_k_dp45(k,t,dt,energy3d,momentum3d,rho[k],As)
        absolute_error[k] = ae
        relative_error[k] = re

    return rho_out5, np.max(absolute_error), np.max(relative_error)

@njit(parallel=True)
def jit_step_vg_wavefunctions_dp45_stdse(nk,t,dt,energy3d,momentum3d,rho,As,gamma):
    rho_out5 = np.zeros(rho.shape, dtype=np.complex128)
    absolute_error = np.zeros(nk)
    relative_error = np.zeros(nk)
    ## jump_occured_all = False
    for k in prange(nk):
        rho_out5[k,:,:], ae, re, jump_occured = \
            jit_step_vg_wavefunctions_k_dp45_stdse(k,t,dt,energy3d,momentum3d,rho[k],As,gamma)
        absolute_error[k] = ae
        relative_error[k] = re
        ## if jump_occured:
        ##     jump_occured_all = True
    return rho_out5, np.max(absolute_error), np.max(relative_error) #,jump_occured_all


#==================== Dormand-Prince45, velocity gauge, LvN ================================
#@jit('Tuple((np.complex128[:,:],float64,float64))(int64,float64,float64,float64[:,:],np.complex128[:,:,:,:],np.complex128[:,:],float64[:,:],float64)',nopython=True,nogil=True,cache=not flags['--no-cache'])
@njit
def jit_step_vg_lvn_k_dp45(k,t,dt,energy3d,momentum3d,rho,As,gamma):

    n_cs = 7
    n,m = rho.shape
    drho = np.zeros((n_cs,n,m), dtype=np.complex128)
    cs = np.array([0.0,0.2,0.3,0.8,8.0/9.0,1.0,1.0])
    bs = np.zeros((n_cs,n_cs))
    bs[1,0] = 0.2
    bs[2,:2] = np.array([3.0/40.0, 9.0/40.0])
    bs[3,:3] = np.array([44.0/45.0, -56.0/15.0, 32.0/9.0])
    bs[4,:4] = np.array([19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0])
    bs[5,:5] = np.array([9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0])
    bs[6,:6] = np.array([35.0/384.0, 0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0])

    sol4 = np.array([35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0])
    sol5 = np.array([5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0])

    for i in range(n_cs):
        v1 = np.exp(1j*(t+cs[i]*dt)*energy3d[k,:])
        mask = np.outer(v1,np.conj(v1))

        tmp_rho = 1.0*rho
        for j in range(i):
            tmp_rho += dt*bs[i,j]*drho[j]

        Hamiltonian = (momentum3d[k,:,:,0]*As[i,0]
                      +momentum3d[k,:,:,1]*As[i,1]
                      +momentum3d[k,:,:,2]*As[i,2])*mask

        drho[i] = -1j*np.dot(Hamiltonian,tmp_rho)
        drho[i] += np.conj(drho[i].T)
        drho[i] += gamma*lindblad_term(tmp_rho,
            np.diag(energy3d[k,:]).astype(np.complex128) + Hamiltonian)

    out5 = np.zeros((n,m), dtype=np.complex128)
    error = np.zeros((n,m), dtype=np.complex128)
    for i in range(n_cs):
       out5 += dt*sol5[i]*drho[i]
       error += dt*(sol5[i]-sol4[i])*drho[i]

    absolute_error = np.sqrt(np.trace(np.dot(np.conj(error).T,error)).real)
    denominator = max(relative_error_min,np.sqrt(np.trace(np.dot(np.conj(out5).T,out5)).real))
    relative_error = absolute_error/denominator
    return rho+out5, absolute_error, relative_error


#@jit('Tuple((np.complex128[:,:,:],float64,float64))(int64,float64,float64,float64[:,:],np.complex128[:,:,:,:],np.complex128[:,:,:],float64[:,:],float64)',nopython=True,nogil=True,cache=not flags['--no-cache'])
@njit(parallel=True)
def jit_step_vg_lvn_dp45(nk,t,dt,energy3d,momentum3d,rho,As,gamma):
    rho_out5 = np.zeros(rho.shape, dtype=np.complex128)
    absolute_error = np.zeros(nk)
    relative_error = np.zeros(nk)
    for k in prange(nk):
        rho_out5[k,:,:], ae, re = \
            jit_step_vg_lvn_k_dp45(k,t,dt,energy3d,momentum3d,rho[k],As,gamma)
        # ae = np.max(np.abs(rho_out5[k,:,:]-np.conj(rho_out5[k,:,:].T)))
        absolute_error[k] = ae
        relative_error[k] = re
    return rho_out5, np.max(absolute_error), np.max(relative_error)


@njit(parallel=True)
def step_vg_lvn(nk,t,energy3d,momentum3d,rho,As):
    drho = np.zeros(rho.shape, dtype=np.complex128)
    for k in prange(nk):
        v1 = np.exp(1j*(t)*energy3d[k,:])
        mask = np.outer(v1,np.conj(v1))
        drho[k] = -1j*np.dot((momentum3d[k,:,:,0]*As[0]
                             +momentum3d[k,:,:,1]*As[1]
                             +momentum3d[k,:,:,2]*As[2])*mask,rho[k])
        drho[k] += np.conj(drho[k].T)
    return drho


#==================== Dormand-Prince45, velocity gauge, LvN ================================
#@jit('Tuple((np.complex128[:,:],float64,float64))(int64,float64,float64,float64[:,:],np.complex128[:,:,:,:],np.complex128[:,:],float64[:,:],float64)',nopython=True,nogil=True,cache=not flags['--no-cache'])
@njit
def jit_step_vg_lvn_k_dp45_FAKE_DECOHERENCE(k,t,dt,energy3d,momentum3d,rho,As,gamma, average_energy_difference):

    n_cs = 7
    n,m = rho.shape
    drho = np.zeros((n_cs,n,m), dtype=np.complex128)
    cs = np.array([0.0,0.2,0.3,0.8,8.0/9.0,1.0,1.0])
    bs = np.zeros((n_cs,n_cs))
    bs[1,0] = 0.2
    bs[2,:2] = np.array([3.0/40.0, 9.0/40.0])
    bs[3,:3] = np.array([44.0/45.0, -56.0/15.0, 32.0/9.0])
    bs[4,:4] = np.array([19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0])
    bs[5,:5] = np.array([9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0])
    bs[6,:6] = np.array([35.0/384.0, 0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0])

    sol4 = np.array([35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0])
    sol5 = np.array([5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0])


    decoherence_exponents = np.zeros((n,n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            if i != j:
                decoherence_exponents[i,j]  = 0.5*gamma*(average_energy_difference)**2

    for i in range(n_cs):
        v1 = np.exp(1j*(t+cs[i]*dt)*energy3d[k,:])
        mask = np.outer(v1,np.conj(v1))

        tmp_rho = 1.0*rho
        for j in range(i):
            tmp_rho += dt*bs[i,j]*drho[j]

        Hamiltonian = (momentum3d[k,:,:,0]*As[i,0]
                      +momentum3d[k,:,:,1]*As[i,1]
                      +momentum3d[k,:,:,2]*As[i,2])*mask

        drho[i] = -1j*np.dot(Hamiltonian,tmp_rho*np.exp(-cs[i]*dt*decoherence_exponents))
        drho[i] += np.conj(drho[i].T)
        drho[i] = drho[i] * np.exp(cs[i] * dt * decoherence_exponents)
        #lindblad = gamma*lindblad_term(tmp_rho,(1.0+0.0*1j)*np.diag(energy3d[k,:])+0*Hamiltonian)
        #drho[i] += 0.5*(lindblad + np.conj(lindblad.T))

    out5 = np.zeros((n,m), dtype=np.complex128)
    error = np.zeros((n,m), dtype=np.complex128)
    for i in range(n_cs):
       out5 += dt*sol5[i]*drho[i]
       error += dt*(sol5[i]-sol4[i])*drho[i]

    absolute_error = np.sqrt(np.trace(np.dot(np.conj(error).T,error)).real)
    denominator = max(relative_error_min,np.sqrt(np.trace(np.dot(np.conj(out5).T,out5)).real))
    relative_error = absolute_error/denominator
    return rho+out5, absolute_error, relative_error


#@jit('Tuple((np.complex128[:,:,:],float64,float64))(int64,float64,float64,float64[:,:],np.complex128[:,:,:,:],np.complex128[:,:,:],float64[:,:],float64)',nopython=True,nogil=True,cache=not flags['--no-cache'])
@njit(parallel=True)
def jit_step_vg_lvn_dp45_FAKE_DECOHERENCE(nk,t,dt,energy3d,momentum3d,rho,As,gamma, average_energy_difference):
    rho_out5 = np.zeros(rho.shape, dtype=np.complex128)
    absolute_error = np.zeros(nk)
    relative_error = np.zeros(nk)
    for k in prange(nk):
        rho_out5[k,:,:], ae, re = \
            jit_step_vg_lvn_k_dp45_FAKE_DECOHERENCE(k,t,dt,energy3d,momentum3d,rho[k],As,gamma, average_energy_difference)
        absolute_error[k] = ae
        relative_error[k] = re
    return rho_out5, np.max(absolute_error), np.max(relative_error)


#==================== RK4, velocity gauge, TDSE ================================
#@jit('np.complex128[:,:](int64,float64,float64,float64[:,:],np.complex128[:,:,:,:],np.complex128[:,:,:],float64[:],float64[:],float64[:])',nopython=True,nogil=True,cache=not flags['--no-cache'])
@njit
def jit_step_vg_wavefunctions_k(k,t,dt,energy3d,momentum3d,rho,A1,A2,A4):
    v1 = np.exp(1j*t*energy3d[k,:])
    v2 = np.exp(-1j*t*energy3d[k,:])
    mask = np.outer(v1,v2)

    v1_half = np.exp(1j*(0.5*dt)*energy3d[k,:])
    v2_half = np.exp(-1j*(0.5*dt)*energy3d[k,:])
    mask_int_half = np.outer(v1_half,v2_half)

    H1  = mask*(A1[0]*momentum3d[k,:,:,0]+A1[1]*momentum3d[k,:,:,1]+A1[2]*momentum3d[k,:,:,2])
    H2  = mask*mask_int_half*(A2[0]*momentum3d[k,:,:,0]+A2[1]*momentum3d[k,:,:,1]+A2[2]*momentum3d[k,:,:,2])
    H4  = mask*mask_int_half*mask_int_half*(A4[0]*momentum3d[k,:,:,0]+A4[1]*momentum3d[k,:,:,1]+A4[2]*momentum3d[k,:,:,2])

    k1 = -1j*np.dot(H1,rho[k,:,:])
    k2 = -1j*np.dot(H2,rho[k,:,:]+0.5*dt*k1)
    k3 = -1j*np.dot(H2,rho[k,:,:]+0.5*dt*k2)
    k4 = -1j*np.dot(H4,rho[k,:,:]+dt*k3)
    return  rho[k,:,:] + (dt/6)*(k1+2*k2+2*k3+k4)


#@jit('np.complex128[:,:,:](int64,float64,float64,float64[:,:],np.complex128[:,:,:,:],np.complex128[:,:,:],float64[:],float64[:],float64[:])',nopython=True,nogil=True,cache=not flags['--no-cache'])
@njit(parallel=True)
def jit_step_vg_wavefunctions(nk,t,dt,energy3d,momentum3d,rho,A1,A2,A4):
    rho_out = np.zeros(rho.shape, dtype=np.complex128)
    for k in prange(nk):
        rho_out[k,:,:] = jit_step_vg_wavefunctions_k(k,t,dt,energy3d,momentum3d,rho,A1,A2,A4)
    return rho_out


## #@jit('np.complex128[:,:,:](int64[:],float64,float64,float64[:,:],np.complex128[:,:,:,:],np.complex128[:,:,:],float64[:],float64[:],float64[:])',nopython=True,nogil=True,cache=not flags['--no-cache'])
## @njit
## def jit_step_vg_wavefunctions_chunk(k_chunk,t,dt,energy3d,momentum3d,rho,A1,A2,A4):
##     rho_out = np.zeros(rho.shape, dtype=np.complex128)
##     for k in k_chunk:
##         rho_out[k,:,:] = jit_step_vg_wavefunctions_k(k,t,dt,energy3d,momentum3d,rho,A1,A2,A4)
##     return rho_out

#==================== RK4, velocity gauge, Liouville-von Neumann ===============
#@jit('np.complex128[:,:](int64,float64,float64,float64[:,:],np.complex128[:,:,:,:],np.complex128[:,:,:],float64[:],float64[:],float64[:],float64)',nopython=True,nogil=True,cache=not flags['--no-cache'])
@njit
def jit_step_vg_lvn_k(k,t,dt,energy3d,momentum3d,rho,A1,A2,A4,gamma):
    v1 = np.exp(1j*t*energy3d[k,:])
    v2 = np.exp(-1j*t*energy3d[k,:])
    mask = np.outer(v1,v2)

    v1_half = np.exp(1j*(0.5*dt)*energy3d[k,:])
    v2_half = np.exp(-1j*(0.5*dt)*energy3d[k,:])
    mask_int_half = np.outer(v1_half,v2_half)

    H1  = mask*(A1[0]*momentum3d[k,:,:,0]+A1[1]*momentum3d[k,:,:,1]+A1[2]*momentum3d[k,:,:,2])
    H2  = mask*mask_int_half*(A2[0]*momentum3d[k,:,:,0]+A2[1]*momentum3d[k,:,:,1]+A2[2]*momentum3d[k,:,:,2])
    H4  = mask*mask_int_half*mask_int_half*(A4[0]*momentum3d[k,:,:,0]+A4[1]*momentum3d[k,:,:,1]+A4[2]*momentum3d[k,:,:,2])

    k1 = -1j*np.dot(H1,rho[k,:,:])
    k1 += np.conj(k1).T
    k1 += gamma*lindblad_term(rho[k,:,:],(1.0+0.0*1j)*np.diag(energy3d[k,:]) + H1)
    k2 = -1j*np.dot(H2,rho[k,:,:]+0.5*dt*k1)
    k2 += np.conj(k2).T
    k2 += gamma*lindblad_term(rho[k,:,:]+0.5*dt*k1,(1.0+0.0*1j)*np.diag(energy3d[k,:]) + H2)
    k3 = -1j*np.dot(H2,rho[k,:,:]+0.5*dt*k2)
    k3 += np.conj(k3).T
    k3 += gamma*lindblad_term(rho[k,:,:]+0.5*dt*k2,(1.0+0.0*1j)*np.diag(energy3d[k,:]) + H2)
    k4 = -1j*np.dot(H4,rho[k,:,:]+dt*k3)
    k4 += np.conj(k4).T
    k4 +=  gamma*lindblad_term(rho[k,:,:]+dt*k3,(1.0+0.0*1j)*np.diag(energy3d[k,:]) + H4)
    return  rho[k,:,:] + (dt/6)*(k1+2*k2+2*k3+k4)


#@jit('np.complex128[:,:,:](int64,float64,float64,float64[:,:],np.complex128[:,:,:,:],np.complex128[:,:,:],float64[:],float64[:],float64[:],float64)',nopython=True,nogil=True,cache=not flags['--no-cache'])
@njit(parallel=True)
def jit_step_vg_lvn(nk,t,dt,energy3d,momentum3d,rho,A1,A2,A4,gamma):
    rho_out = np.zeros(rho.shape, dtype=np.complex128)
    for k in prange(nk):
        rho_out[k,:,:] = jit_step_vg_lvn_k(k,t,dt,energy3d,momentum3d,rho,A1,A2,A4,gamma)
    return rho_out


## #@jit('np.complex128[:,:,:](int64[:],float64,float64,float64[:,:],np.complex128[:,:,:,:],np.complex128[:,:,:],float64[:],float64[:],float64[:],float64)',nopython=True,nogil=True,cache=not flags['--no-cache'])
## @njit
## def jit_step_vg_lvn_chunk(k_chunk,t,dt,energy3d,momentum3d,rho,A1,A2,A4,gamma):
##     rho_out = np.zeros(rho.shape, dtype=np.complex128)
##     for k in k_chunk:
##         rho_out[k,:,:] = jit_step_vg_lvn_k(k,t,dt,energy3d,momentum3d,rho,A1,A2,A4,gamma)
##     return rho_out


#==================== Length gauge, TDSE ================================
@njit
def step_lg(k,nk_periodicity,nv,rho,S,table,energy3d,Et_crystal,time,directions):
    rho_out = np.zeros(rho[k,:,:].shape, dtype=np.complex128)
    rho0 = np.copy(rho[k,:,:])
    rho0_norm = np.copy(rho[k,:,:])
    for i in range(nv):
        rho0_norm[:,i] = rho0[:,i]/np.sqrt(np.sum(np.abs(rho0[:,i])**2))

    v1 = np.exp(1j*energy3d[k,:]*time)
    for alpha in directions:
        k1 = table[k,alpha,0]
        k2 = table[k,alpha,-1]

        rho1 = rho[k1,:,:]
        rho2 = rho[k2,:,:]

        v2_1 = np.exp(-1j*energy3d[k1,:]*time)
        mask1 = np.outer(v1,v2_1)
        v2_2 = np.exp(-1j*energy3d[k2,:]*time)
        mask2 = np.outer(v1,v2_2)

        S1 = np.dot(np.conj(rho0_norm).T,np.dot(mask1*S[k,alpha,0],rho1))
        Sinv1 = np.linalg.inv(S1)
        cov1 = np.dot(rho1,Sinv1)

        S2 = np.dot(np.conj(rho0_norm).T,np.dot(mask2*S[k,alpha,-1],rho2))
        Sinv2 = np.linalg.inv(S2)
        cov2 = np.dot(rho2,Sinv2)

        weight = np.sqrt(np.dot(np.conj(rho0.T),rho0))
        integrand = Et_crystal[alpha]*np.dot(np.dot(mask1*S[k,alpha,0],cov1) - np.dot(mask2*S[k,alpha,-1],cov2),weight)
        rho_out += ((1j*nk_periodicity[alpha])/(4.0*np.pi))*integrand
    return rho_out


@njit(parallel=True)
def nonperturbative_jit_solver_lvn(nk,nk_periodicity,nv,rho,S,table,energy3d,Et_crystal,time,directions):
    rho_out = np.zeros(rho.shape, dtype=np.complex128)
    rho_norm = np.zeros(rho.shape, dtype=np.complex128)

    for k in prange(nk):
        Uk,Sk,Vhk = np.linalg.svd(rho[k,:,:])
        rho_norm[k] = np.dot(Uk,np.dot(np.diag(np.sqrt(Sk.astype(np.complex128))),Vhk))

    for k in prange(nk):
        rho0 = rho_norm[k]
        v1 = np.exp(1j*energy3d[k,:]*time)
        for alpha in directions:
            k1 = table[k,alpha,0]
            k2 = table[k,alpha,-1]

            rho1 = rho_norm[k1,:,:]
            rho2 = rho_norm[k2,:,:]

            v2_1 = np.exp(-1j*energy3d[k1,:]*time)
            mask1 = np.outer(v1,v2_1)
            v2_2 = np.exp(-1j*energy3d[k2,:]*time)
            mask2 = np.outer(v1,v2_2)

            S1 = (mask1*S[k,alpha,0])
            S2 = (mask2*S[k,alpha,-1])

            Srho1_normS = np.dot(np.dot(S1,rho1),np.conj((S1).T))
            Srho2_normS = np.dot(np.dot(S2,rho2),np.conj((S2).T))

            integrand = Et_crystal[alpha]*( np.dot(Srho1_normS,rho0) - np.dot(Srho2_normS,rho0))

            rho_out[k,:,:] += ((1j*nk_periodicity[alpha])/(4.0*np.pi))*integrand
    return rho_out


#@jit('np.complex128[:,:,:](int64,int64[:],int64,np.complex128[:,:,:],np.complex128[:,:,:,:,:], int64[:,:,:], float64[:,:], float64[:],float64)',nopython=True,cache=True)#,parallel=True)#,nogil=True,cache=not flags['--no-cache'])
@njit(parallel=True)
def nonperturbative_jit_solver(nk,nk_periodicity,nv,rho,S,table,energy3d,Et_crystal,time,directions):
    rho_out = np.zeros(rho.shape, dtype=np.complex128)
    for k in prange(nk):
        tmp = step_lg(k,nk_periodicity,nv,rho,S,table,energy3d,Et_crystal,time,directions)
        rho_out[k,:,:] = tmp
    return rho_out


#=================== Dormand-Prince45, length gauge, TDSE ================================
#@jit('Tuple((np.complex128[:,:,:],float64,float64))(int64,float64,float64,float64[:,:],np.complex128[:,:,:,:,:],int64[:,:,:],int64[:],np.complex128[:,:,:],float64[:,:])',nopython=True,nogil=True,cache=not flags['--no-cache'])
@njit(parallel=True)
def jit_step_lg_wavefunctions_dp45(nk,t,dt,energy3d,overlap,neighbour_table,size,rho,Es_cartesian,directions,lattice_vectors):
    n_cs = 7
    Es = np.zeros(Es_cartesian.shape, dtype=np.float64)
    for i in range(n_cs):
        Es[i,:] = np.dot(lattice_vectors.T,Es_cartesian[i,:])

    _,n,m = rho.shape
    nv = m
    drho = np.zeros((n_cs,nk,n,m), dtype=np.complex128)
    cs = np.array([0.0,0.2,0.3,0.8,8.0/9.0,1.0,1.0])
    bs = np.zeros((n_cs,n_cs))
    bs[1,0] = 0.2
    bs[2,:2] = np.array([3.0/40.0, 9.0/40.0])
    bs[3,:3] = np.array([44.0/45.0, -56.0/15.0, 32.0/9.0])
    bs[4,:4] = np.array([19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0])
    bs[5,:5] = np.array([9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0])
    bs[6,:6] = np.array([35.0/384.0, 0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0])

    sol4 = np.array([35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0])
    sol5 = np.array([5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0])

    for i in range(n_cs):
        tmp_rho = 1.0*rho
        for j in range(i):
            tmp_rho += dt*bs[i,j]*drho[j]

        drho[i] = -1j*nonperturbative_jit_solver(nk,size,nv,tmp_rho,overlap,neighbour_table,
                                                 energy3d,Es[i],t+cs[i]*dt,directions)

    out5 = np.zeros((nk,n,m), dtype=np.complex128)
    error = np.zeros((nk,n,m), dtype=np.complex128)
    for i in range(n_cs):
       out5 += dt*sol5[i]*drho[i]
       error += dt*(sol5[i]-sol4[i])*drho[i]

    absolute_error = np.zeros(nk)
    relative_error = np.zeros(nk)
    for i in prange(nk):
        absolute_error[i] = np.sqrt(np.trace(np.dot(np.conj(error[i]).T,error[i])).real)
        denominator = max(relative_error_min, np.sqrt(np.trace(np.dot(np.conj(out5[i]).T,out5[i])).real))
        relative_error[i] = absolute_error[i] / denominator

    return rho+out5, np.max(absolute_error), np.max(relative_error)


#@jit('Tuple((np.complex128[:,:,:],float64,float64))(int64,float64,float64,float64[:,:],np.complex128[:,:,:,:,:],int64[:,:,:],int64[:],np.complex128[:,:,:],float64[:,:],float64)',nopython=True,nogil=True,cache=not flags['--no-cache'])
@njit(parallel=True)
def jit_step_lg_lvn_dp45(nk,t,dt,energy3d,overlap,neighbour_table,size,rho,Es_cartesian,gamma,directions,lattice_vectors):
    n_cs = 7
    Es = np.zeros(Es_cartesian.shape, dtype=np.float64)
    for i in range(n_cs):
        Es[i,:] = np.dot(lattice_vectors.T,Es_cartesian[i,:])
    _,n,m = rho.shape
    nv = m
    drho = np.zeros((n_cs,nk,n,m), dtype=np.complex128)
    cs = np.array([0.0,0.2,0.3,0.8,8.0/9.0,1.0,1.0])
    bs = np.zeros((n_cs,n_cs))
    bs[1,0] = 0.2
    bs[2,:2] = np.array([3.0/40.0, 9.0/40.0])
    bs[3,:3] = np.array([44.0/45.0, -56.0/15.0, 32.0/9.0])
    bs[4,:4] = np.array([19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0])
    bs[5,:5] = np.array([9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0])
    bs[6,:6] = np.array([35.0/384.0, 0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0])

    sol4 = np.array([35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0])
    sol5 = np.array([5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0])

    decoherence_exponents = np.zeros((nk,n,n), dtype=np.complex128)
    for k in prange(nk):
        for i in range(n):
            for j in range(n):
                decoherence_exponents[k,i,j] = 0.5*gamma*(energy3d[k,i]-energy3d[k,j])**2

    for i in range(n_cs):
        tmp_rho = 1.0*rho
        for j in range(i):
            tmp_rho += dt*bs[i,j]*drho[j]

        tmp_rho = tmp_rho*np.exp(-cs[i]*dt*decoherence_exponents)
        drho[i] = -1j*nonperturbative_jit_solver_lvn(nk,size,nv,tmp_rho,overlap,neighbour_table,energy3d,Es[i],t+cs[i]*dt,directions)
        for k in prange(nk):
            drho[i,k,:,:] += np.conj(drho[i,k,:,:].T)
        drho[i] = drho[i]*np.exp(cs[i]*dt*decoherence_exponents)

    out5 = np.zeros((nk,n,m), dtype=np.complex128)
    error = np.zeros((nk,n,m), dtype=np.complex128)
    for i in range(n_cs):
       out5 += dt*sol5[i]*drho[i]
       error += dt*(sol5[i]-sol4[i])*drho[i]

    absolute_error = np.zeros(nk)
    relative_error = np.zeros(nk)
    for i in prange(nk):
        absolute_error[i] = np.sqrt(np.trace(np.dot(np.conj(error[i]).T,error[i])).real)
        denominator = max(relative_error_min, np.sqrt(np.trace(np.dot(np.conj(out5[i]).T,out5[i])).real))
        relative_error[i] = absolute_error[i] / denominator

    output = np.exp(-dt*decoherence_exponents)*(rho+out5)
    return output, np.max(absolute_error), np.max(relative_error)


#@jit('Tuple((np.complex128[:,:,:],float64,float64))(int64,float64,float64,float64[:,:],np.complex128[:,:,:,:,:],int64[:,:,:],int64[:],np.complex128[:,:,:],float64[:,:],float64)',nopython=True,nogil=True,cache=not flags['--no-cache'])
@njit(parallel=True)
def jit_step_lg_lvn_dp45_constant_dephasing(nk,t,dt,energy3d,overlap,neighbour_table,size,rho,Es_cartesian,gamma,
                                            directions,lattice_vectors,average_energy_difference):
    n_cs = 7
    Es = np.zeros(Es_cartesian.shape, dtype=np.float64)
    for i in range(n_cs):
        Es[i,:] = np.dot(lattice_vectors.T,Es_cartesian[i,:])
    _,n,m = rho.shape
    nv = m
    drho = np.zeros((n_cs,nk,n,m), dtype=np.complex128)
    cs = np.array([0.0,0.2,0.3,0.8,8.0/9.0,1.0,1.0])
    bs = np.zeros((n_cs,n_cs))
    bs[1,0] = 0.2
    bs[2,:2] = np.array([3.0/40.0, 9.0/40.0])
    bs[3,:3] = np.array([44.0/45.0, -56.0/15.0, 32.0/9.0])
    bs[4,:4] = np.array([19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0])
    bs[5,:5] = np.array([9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0])
    bs[6,:6] = np.array([35.0/384.0, 0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0])

    sol4 = np.array([35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0])
    sol5 = np.array([5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0])

    decoherence_exponents = np.zeros((nk,n,n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            if i != j:
                for k in prange(nk):
                    decoherence_exponents[k,i,j]  = 0.5*gamma*(average_energy_difference)**2

    for i in range(n_cs):
        tmp_rho = 1.0*rho
        for j in range(i):
            tmp_rho += dt*bs[i,j]*drho[j]

        tmp_rho = tmp_rho*np.exp(-cs[i]*dt*decoherence_exponents)
        drho[i] = -1j*nonperturbative_jit_solver_lvn(nk,size,nv,tmp_rho,overlap,neighbour_table,energy3d,Es[i],t+cs[i]*dt,directions)
        for k in prange(nk):
            drho[i,k,:,:] += np.conj(drho[i,k,:,:].T)
        drho[i] = drho[i]*np.exp(cs[i]*dt*decoherence_exponents)

    out5 = np.zeros((nk,n,m), dtype=np.complex128)
    error = np.zeros((nk,n,m), dtype=np.complex128)
    for i in range(n_cs):
       out5 += dt*sol5[i]*drho[i]
       error += dt*(sol5[i]-sol4[i])*drho[i]

    absolute_error = np.zeros(nk)
    relative_error = np.zeros(nk)
    for i in prange(nk):
        absolute_error[i] = np.sqrt(np.trace(np.dot(np.conj(error[i]).T,error[i])).real)
        denominator = max(relative_error_min,np.sqrt(np.trace(np.dot(np.conj(out5[i]).T,out5[i])).real))
        relative_error[i] = absolute_error / denominator

    output = np.exp(-dt*decoherence_exponents)*(rho+out5)
    return output, np.max(absolute_error), np.max(relative_error)
