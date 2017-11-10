"""Convert data from quantum espresso to hdf5. """
import numpy as np
import h5py
from ulmic.external.wannier90 import read_mmn
from ulmic.atomic_units import AtomicUnits

au = AtomicUnits()

def read_quantum_espresso(qe_file,size,case='case',spin_factor=2,overlap_input=None,momentum_input=None,):
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
    nb = number_bands

    klist3d = np.zeros(tuple(size),int)
    for i in range(nk):
        idx, idy, idz = [int(q) for q in np.rint(size*klist1d[i, :])]
        klist3d[idx, idy, idz] = i

    nn_table = np.zeros((nk,3,2*nn),int)
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

    vol = (np.dot(lv1,np.cross(lv2,lv3)))
    rv1,rv2,rv3 = 2*np.pi*np.cross(lv2,lv3)/vol,2*np.pi*np.cross(lv3,lv1)/vol,2*np.pi*np.cross(lv1,lv2)/vol

    lattice = np.array([lv1,lv2,lv3])
    reciprocal_lattice = np.array([rv1,rv2,rv3])
 
    if not overlap_input is None:
        overlap = read_mmn(overlap_input,size,klist1d,klist3d,transpose=False)        

    if not momentum_input is None:
        momentum = np.load(momentum_input)
        momentum = np.rollaxis(momentum,1,4) 

    hdf5 = h5py.File(case+'_quantum_espresso.hdf5', 'w')
    dset_energy = hdf5.create_dataset("energy", data=energy/au.eV)
    dset_klist1d = hdf5.create_dataset("klist1d", data=klist1d)
    dset_klist3d = hdf5.create_dataset("klist3d", data=klist3d)
    dset_lattice = hdf5.create_dataset("lattice", data=lattice)
    dset_reciprocal = hdf5.create_dataset("reciprocal_lattice",data=reciprocal_lattice)
    dset_table = hdf5.create_dataset("neighbour_table", data=nn_table)
    dset_valence = hdf5.create_dataset("valence", data=nv)
    dset_size = hdf5.create_dataset("size", data=size)
    dset_spin = hdf5.create_dataset("spin_factor", data=spin_factor)

    if not overlap_input is None:
        dset_overlap = hdf5.create_dataset("overlap", data=overlap)

    if not momentum_input is None:
        dset_momentum = hdf5.create_dataset("momentum", data=momentum)

