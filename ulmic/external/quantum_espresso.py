""" Python script for processing Quantum Espresso data. """
import numpy as np
import h5py
from ulmic.external import nearest_neighbor_table
from ulmic.external.wannier90 import read_mmn
from ulmic.atomic_units import AtomicUnits

au = AtomicUnits()

def read_quantum_espresso(qe_file,size,output_file='out.hdf5',spin_factor=2,
                          overlap_input=None,momentum_input=None,):
    """
       qe_file:         quantum espresso output file
       output_file:     output
       spin_factor:     spin degeneracy
       overlap_input:   mmn file
       momentum_input:  momentum matrix file
    """
    nn = 1
    with open(qe_file,'r') as qe_out:
        line = next(qe_out)

        while not line.startswith('     lattice parameter (alat)  ='):

            line = next(qe_out)
        lattice_constant = float(line.split()[-2])

        while not line.startswith('     number of electrons       ='):
            line = next(qe_out)
        number_electrons = int(float(line.strip().split('=')[1]))
        number_valence_bands = number_electrons/spin_factor

        while not line.startswith('     number of Kohn-Sham states='):
            line = next(qe_out)
        number_bands = int(line.split('=')[1])

        while not line.startswith('     crystal axes: (cart. coord. in units of alat)'):
            line = next(qe_out)
        lv1 = lattice_constant*np.array(map(float,next(qe_out).replace('a(1) = (','').split()[0:3]))
        lv2 = lattice_constant*np.array(map(float,next(qe_out).replace('a(2) = (','').split()[0:3]))
        lv3 = lattice_constant*np.array(map(float,next(qe_out).replace('a(3) = (','').split()[0:3]))

        while not line.startswith('     number of k points='):
            line = next(qe_out)
        number_kpoints = int(line.split('=')[1])

        klist1d = np.zeros((number_kpoints,3))
        while not line.startswith('                       cryst. coord.'):
            line = next(qe_out)
        for i in range(number_kpoints):
            line = next(qe_out)
            line = line.split(')')[1].split('(')[1].split()
            klist1d[i] = np.array(map(float,line))
        line = next(qe_out)


        energy = np.zeros((number_kpoints,number_bands))
        while not line.startswith('     End of band structure calculation'):
            line = next(qe_out)
        for i in range(number_kpoints):
            line = next(qe_out)
            line = next(qe_out)
            line = next(qe_out)
            energies = []
            while len(energies) < number_bands:
                energies += next(qe_out).split()
            energy[i] = map(float,energies)

    nk = number_kpoints
    nv = number_valence_bands
    klist3d = np.zeros(tuple(size), dtype=np.intp)
    shift_vector = size * klist1d[0, :]
    shift_vector -= np.floor(shift_vector)
    for i in range(nk):
        indices = np.rint(size*klist1d[i, :] - shift_vector)
        idx, idy, idz = [int(q) for q in indices]
        klist3d[idx, idy, idz] = i

    nn_table = nearest_neighbor_table(klist3d,nn)

    vol = (np.dot(lv1,np.cross(lv2,lv3)))
    rv1,rv2,rv3 = (2*np.pi*np.cross(lv2,lv3)/vol,
                   2*np.pi*np.cross(lv3,lv1)/vol,
                   2*np.pi*np.cross(lv1,lv2)/vol)

    lattice = np.array([lv1,lv2,lv3]).T
    reciprocal_lattice = np.array([rv1,rv2,rv3]).T

    if not overlap_input is None:
        overlap = read_mmn(overlap_input,size,klist1d,klist3d,transpose=False)

    if not momentum_input is None:
        momentum = np.load(momentum_input)
        momentum = np.rollaxis(momentum,1,4)

    hdf5 = h5py.File(output_file, 'w')
    hdf5.create_dataset("energy", data=energy/au.eV)
    hdf5.create_dataset("klist1d", data=klist1d)
    hdf5.create_dataset("klist3d", data=klist3d)
    hdf5.create_dataset("lattice_vectors", data=lattice)
    hdf5.create_dataset("reciprocal_vectors",data=reciprocal_lattice)
    hdf5.create_dataset("neighbour_table", data=nn_table)
    hdf5.create_dataset("valence_bands", data=nv)
    hdf5.create_dataset("size", data=size)
    hdf5.create_dataset("spin_factor", data=spin_factor)

    if not overlap_input is None:
        hdf5.create_dataset("overlap", data=overlap)

    if not momentum_input is None:
        hdf5.create_dataset("momentum", data=momentum)

