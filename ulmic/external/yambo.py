""" Python script for processing Yambo data. """
import os
import h5py
import numpy as np
from yambopy import YamboSaveDB, YamboLatticeDB, YamboDipolesDB
from ulmic import AtomicUnits

au = AtomicUnits()
TOLERANCE_ZERO = 1e-5


def read_yambo(directory,size=None,out='out.hdf5',save='SAVE'):
    """
    directory:  string
                Parent folder of the SAVE directory
    size:       list, array or None
                If None, attempts to guess the correct size
    out:        string
                Name of file to which output is written
    """

    save_directory = os.path.join(directory,save,'')

    yambo_save = YamboSaveDB(save_directory)
    yambo_lattice = YamboLatticeDB(save_directory)
    yambo_dipoles = YamboDipolesDB(yambo_lattice,save_directory,dip_type='P')

    nn = 1
    nk = yambo_save.nkpoints
    spin_factor = int(yambo_save.spin_degen)
    nv = int(np.rint(yambo_save.electrons/spin_factor))
    energy = yambo_save.eigenvalues/au.eV
    klist = yambo_save.kpts_car
    lattice = yambo_save.lat
    reciprocal_lattice = 2*np.pi*yambo_save.rlat
    spin_factor = yambo_save.spin_degen
    momentum = np.rollaxis(yambo_dipoles.dipoles,1,4).astype(np.complex128)

    klist1d = np.zeros((nk, 3))
    for i in range(nk):
        klist1d[i] = np.dot(lattice,klist[i])

    non_zeros = np.abs(klist1d) > TOLERANCE_ZERO
    if size is None:
        size = np.array([int(np.rint(np.max(1/klist1d[:,q][non_zeros[:,q]])))
                         for q in range(3)])

    klist3d = np.zeros((size), dtype=np.intp)
    nn_table = np.zeros((nk, 3, 2*nn), dtype=np.intp)
    shift_vector = size * klist1d[0, :]
    shift_vector -= np.floor(shift_vector)
    for i in range(nk):
        indices = np.rint(size*klist1d[i, :] - shift_vector)
        idx, idy, idz = [int(q) for q in indices]
        klist3d[idx, idy, idz] = i

    for ix in range(size[0]):
        for iy in range(size[1]):
            for iz in range(size[2]):
                i = klist3d[ix, iy, iz]
                for j in range(-nn, nn):
                    if j < 0:
                        nn_table[i, 0, j] = klist3d[(ix+j)%size[0], iy, iz]
                        nn_table[i, 1, j] = klist3d[ix, (iy+j)%size[1], iz]
                        nn_table[i, 2, j] = klist3d[ix, iy, (iz+j)%size[2]]
                    if j >= 0:
                        nn_table[i, 0, j] = klist3d[(ix+j+1)%size[0], iy, iz]
                        nn_table[i, 1, j] = klist3d[ix, (iy+j+1)%size[1], iz]
                        nn_table[i, 2, j] = klist3d[ix, iy, (iz+j+1)%size[2]]

    hdf5 = h5py.File(out, 'w')
    hdf5.create_dataset("energy", data=energy)
    hdf5.create_dataset("klist1d", data=klist1d)
    hdf5.create_dataset("klist3d", data=klist3d)
    hdf5.create_dataset("lattice_vectors", data=lattice.T)
    hdf5.create_dataset("reciprocal_vectors",data=reciprocal_lattice.T)
    hdf5.create_dataset("momentum", data=momentum)
    # hdf5.create_dataset("neighbour_table", data=nn_table)
    hdf5.create_dataset("valence_bands", data=nv)
    hdf5.create_dataset("size", data=size)
    hdf5.create_dataset("spin_factor", data=spin_factor)


