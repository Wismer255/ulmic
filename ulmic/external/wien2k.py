
""" Python script for processing Wien2k data. """
from __future__ import print_function
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from ulmic.external.wannier90 import read_mmn


def read_wien2k(input_case, input_overlap=None,
                spin_factor=2, out='out.hdf5',
                nb_max=None,fixed_position=True):
    """
        Read data from a Wien2k calculation and it save as hdf5.
        input_case:     case name (e.g. path/case/case)
        input_overlap:  name of .mmn file used by Wannier90
        spin_factor:    number of electrons per band
        out:            name of output file
        nb_max:         maximum number of bands to be read
        fixed_position: assume values are not separated by white space
    """

    nn = 1
    input_scf = input_case + '.scf'
    input_outputkgen = input_case + '.outputkgen'
    input_energy = input_case + '.energy'
    input_momentum = input_case + '.mommat2'

    with open(input_scf,'r') as scf:
        line = next(scf)
        while not line.startswith(':NOE'):
            line = next(scf)
        number_electrons = int(float(line.split()[-1]))
    nv = number_electrons//spin_factor

    with open(input_outputkgen,'r') as outputkgen:
        line = next(outputkgen)
        lv1 = map(float,next(outputkgen).split()[-3:])
        lv2 = map(float,next(outputkgen).split()[-3:])
        lv3 = map(float,next(outputkgen).split()[-3:])
        vol = (np.dot(lv1,np.cross(lv2,lv3)))

        rv1,rv2,rv3 = (2*np.pi*np.cross(lv2,lv3)/vol,
                       2*np.pi*np.cross(lv3,lv1)/vol,
                       2*np.pi*np.cross(lv1,lv2)/vol
                      )

        while not line.startswith('  NO. OF MESH POINTS IN THE BRILLOUIN ZONE ='):
            line = next(outputkgen)
        nk = int(line.split()[-1])
        klist1d = np.zeros((nk,3))

        while not line.startswith('  DIVISION OF RECIPROCAL LATTICE VECTORS (INTERVALS)='):
            line = next(outputkgen)
        size = np.array(map(int,line.split()[-3:]))
        klist3d = np.zeros(tuple(size),int)

        while not line.startswith('  internal and cartesian k-vectors:'):
            line = next(outputkgen)

        for i in range(nk):
            line = next(outputkgen)
            klist1d[i,:] = np.array(map(float,line.split()[1:4]))

    with open(input_energy) as energy_file:
        skip_lines_energy = 5
        [next(energy_file) for _ in range(skip_lines_energy)]
        line = next(energy_file)
        while len(line.split()) == 2:
            nb = int(line.split()[0])-1
            line = next(energy_file)

    if nb_max is not None:
        nb = min(nb,nb_max)

    energy = np.zeros((nk,nb))
    with open(input_energy) as energy_file:
        skip_lines_energy = 4
        [next(energy_file) for _ in range(skip_lines_energy)]

        cnt_k = -1
        for line0 in energy_file:
            line = line0.split()
            if len(line) >= 6:
                cnt_k += 1
                klist1d[cnt_k,0] = float(line0[0:19])
                klist1d[cnt_k,1] = float(line0[19:38])
                klist1d[cnt_k,2] = float(line0[38:57])
            elif len(line) == 2:
                band_index,energy_value = int(line[0])-1,float(line[1])
                if band_index < nb:
                    energy[cnt_k,band_index] = energy_value
            else:
                print("Warning, line skipped!")
                print(line)
    energy = 0.5*(np.copy(energy) - np.max(energy[:,nv-1]))

    lattice = np.array([lv1,lv2,lv3]).T
    reciprocal_lattice = np.array([rv1,rv2,rv3]).T
    nn_table = np.zeros((nk, 3, 2*nn), int)

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
    hdf5.create_dataset("lattice_vectors", data=lattice)
    hdf5.create_dataset("reciprocal_vectors",data=reciprocal_lattice)
    hdf5.create_dataset("neighbour_table", data=nn_table)
    hdf5.create_dataset("valence_bands", data=nv)
    hdf5.create_dataset("size", data=size)
    hdf5.create_dataset("spin_factor", data=spin_factor)

    if not input_overlap is None:
        overlap = read_mmn(input_overlap,size,klist1d,klist3d,transpose=False)
        hdf5.create_dataset("overlap", data=overlap)

    try:
        momentum = np.zeros((nk,nb,nb,3),complex)
        skip_lines = 2
        with open(input_momentum) as momentum_file:
            [next(momentum_file) for _ in range(skip_lines)]
            cnt_k = -1
            for line0 in momentum_file:
                line = line0.split()
                if line0.startswith("   KP:"):
                    cnt_k += 1
                elif len(line) > 2:
                    if fixed_position:
                        band1 = int(line0[4:7])-1
                        band2 = int(line0[8:11])-1
                        rex = float(line0[11:24])
                        imx = float(line0[24:37])
                        rey = float(line0[37:50])
                        imy = float(line0[50:63])
                        rez = float(line0[63:76])
                        imz = float(line0[76:89])
                    else:
                        band1 = int(line[0])-1
                        band2 = int(line[1])-1
                        rex = float(line[2])
                        imx = float(line[3])
                        rey = float(line[4])
                        imy = float(line[5])
                        rez = float(line[6])
                        imz = float(line[7])

                    if band1 < nb and band2 < nb:
                        momentum[cnt_k,band1,band2,0] = rex+1j*imx
                        momentum[cnt_k,band1,band2,1] = rey+1j*imy
                        momentum[cnt_k,band1,band2,2] = rez+1j*imz

                        momentum[cnt_k,band2,band1,0] = rex-1j*imx
                        momentum[cnt_k,band2,band1,1] = rey-1j*imy
                        momentum[cnt_k,band2,band1,2] = rez-1j*imz
                elif not line:
                    pass
                else:
                    print("Warning, line skipped!")
                    print(line0)
        dset_momentum = hdf5.create_dataset("momentum", data=momentum)
    except:
        dset_momentum = hdf5.create_dataset("momentum", (nk,nb,nb,3), dtype=np.complex128)
        momentum = np.zeros((nb,nb,3),complex)
        with open(input_momentum) as momentum_file:
            skip_lines = 2
            [next(momentum_file) for _ in range(skip_lines)]
            cnt_k = -1
            for line0 in momentum_file:
                line = line0.split()
                if line0.startswith("   KP:"):
                    cnt_k += 1
                elif len(line) > 2:
                    if fixed_position:
                        band1 = int(line0[4:7])-1
                        band2 = int(line0[8:11])-1
                        rex = float(line0[11:24])
                        imx = float(line0[24:37])
                        rey = float(line0[37:50])
                        imy = float(line0[50:63])
                        rez = float(line0[63:76])
                        imz = float(line0[76:89])
                    else:
                        band1 = int(line[0])-1
                        band2 = int(line[1])-1
                        rex = float(line[2])
                        imx = float(line[3])
                        rey = float(line[4])
                        imy = float(line[5])
                        rez = float(line[6])
                        imz = float(line[7])

                    if band1 < nb and band2 < nb:
                        momentum[band1,band2,0] = rex+1j*imx
                        momentum[band1,band2,1] = rey+1j*imy
                        momentum[band1,band2,2] = rez+1j*imz

                        momentum[band2,band1,0] = rex-1j*imx
                        momentum[band2,band1,1] = rey-1j*imy
                        momentum[band2,band1,2] = rez-1j*imz
                        dset_momentum[cnt_k,:,:,:] = np.copy(momentum)
                elif not line:
                    pass
                else:
                    print("Warning, line skipped!")
                    print(line0)


class ConvergenceCheckerWien2k:

    def __init__(self,case,path):
        """ Create oject for loading files from a Wien2k calculation. """
        self.case = case
        self.path = path
        self.file = os.path.join(path,case)

    def check_convergence_plane_waves(self,):
        """ Check convergence of data in case.output1. """
        file_output1 = self.file+'.output1'

        with open(file_output1) as output1:
            for line in output1:
                if line.startswith('     K= '):
                    matrix_line = next(output1)
                    matrix_size = int(matrix_line.split()[2])
                    next(output1)
                    energies = ''
                    energy_line = next(output1)
                    while 'EIGENVALUES BELOW THE ENERGY' not in energy_line:
                        energies += energy_line
                        energy_line = next(output1)

                if line.startswith('   RECIPROCAL LATTICE VECTORS'):
                    [next(output1) for _ in range(1)]
                    band_indices = [int(q)-1 for q in
                                    next(output1).replace('.ENERGY','').split()]
                    coeff_indices = np.zeros((matrix_size,3))
                    coeff_values = np.zeros((matrix_size,len(band_indices)))
                    for i in range(matrix_size):
                        line_indices = next(output1)
                        line_values = next(output1)
                        coeff_indices[i,:] = [int(q) for q in line_indices.split()]
                        coeff_values[i,:] = [float(q) for q in
                                             line_values.replace(
                                                 '*********','0.0').split(
                                                     )[0:len(band_indices)]
                                            ]

                        plt.figure()
                        plt.semilogy([np.sqrt(np.dot(q,q)) for q in coeff_indices],abs(coeff_values),'.')
                        plt.show()

    def check_convergence_radial(self,):
        """ Check convergence of data in case.radwf. """

        file_radwf = self.file+'.radwf'
        r_discretization = 781

        with open(file_radwf,'r') as radwf:
            next(radwf)
            for line in radwf:
                if len(line.split()) == 1:
                    k_index = int(line)
                    counter = 0
                else:
                    if counter == 0:
                        data = np.zeros((r_discretization,len(line.split())))
                    data[counter,:] = [float(q) for q in line.split()]
                    counter += 1
                    if counter == r_discretization:
                        plt.plot(data)
                        plt.title(str(k_index))
                        plt.show()

    def check_convergence_inner_core(self,):
        """ Check convergence of data in case.almblm. """
        file_almblm = self.file+'.almblm'

        with open(file_almblm) as almblm:
            for _ in almblm:
                pass

