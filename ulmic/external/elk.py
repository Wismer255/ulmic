import os
import numpy as np
import h5py


def create_elk_hdf5(path_to_elk_OUT, output_file_path):
    """
    path_to_elk_OUT:    string
                        Path to the folder containing following elk output files:
                        'INFO.OUT',
                        'LATTICE.OUT',
                        'KPOINTS.OUT',
                        'EIGVAL.OUT'
                        'PMAT.OUT'.
    output_file_path:   string
                        path where where hdf5 file will be created
    return:        HDF5 file:>> "material_name_exc_*x*x*_nb_*.hdf5"  which contains:

                        "nk":  number of kpoints
                        "nb":  number of bands
             "valence_bands":  number of valence bands
                      "size":  kpoints grid size (nk_x, nk_y, nk_z)
               "spin_factor":  number of electrons per energy state
                   "klist1d":  kpoints (nk,3)
                   "klist3d":  indices of klist1d (nk_x, nk_y, nk_z)
           "lattice_vectors":  (3,3)
                               lattice vector--> a_{n} = lattice_vectors[n-1,:],
                                                                where n = 0,1,2
        "reciprocal_vectors":  (3,3)
                                reciprocal vector--> b_{n} = reciprocal_vectors[n-1,:],
                                                                 where n = 0,1,2
                    "energy":   energies of states (nk,nb)
                  "momentum":   momentum matrix elements (nk,nb,nb,3)
    """

    input_info     = os.path.join(path_to_elk_OUT,'INFO.OUT'   )
    input_lattice  = os.path.join(path_to_elk_OUT,'LATTICE.OUT')
    input_kpoints  = os.path.join(path_to_elk_OUT,'KPOINTS.OUT')
    input_energy   = os.path.join(path_to_elk_OUT,'EIGVAL.OUT' )
    input_momentum = os.path.join(path_to_elk_OUT,'PMAT.OUT'   )
    ################################################################################
    #          Reading lattice vectors and reciprocal lattice vectors
    ################################################################################

    with open(input_lattice, 'r') as f:
        lattice_file = f.readlines()


    lattice_vectors = np.empty([3,3])
    reciprocal_vectors = np.empty([3,3])
    avec_ind = [x+5 for x in range(len(lattice_file)) if 'vector a1' in lattice_file[x]][0]
    bvec_ind = [x+5 for x in range(len(lattice_file)) if 'vector b1' in lattice_file[x]][0]

    for i in range(3):
        for j in range(3):
            lattice_vectors[j,i]    = np.float(lattice_file[avec_ind+i].split()[j])
            reciprocal_vectors[j,i] = np.float(lattice_file[bvec_ind+i].split()[j])
    '''
    cell_vol_ind = [x for x in range(len(lattice_file)) if 'Unit cell volume' in lattice_file[x]][0]
    unit_cell_volume = np.float(lattice_file[cell_vol_ind].split()[-1])
    BZ_vol_ind = [x for x in range(len(lattice_file)) if 'Brillouin zone volume' in lattice_file[x]][0]
    Brillouin_zone_volume = np.float(lattice_file[BZ_vol_ind].split()[-1])
    '''
    ################################################################################
    #              Reading kpoints grid size and grid shift from INFO.OUT
    ################################################################################
    with open(input_info, 'r') as f:
        info_file = f.readlines()


    kgrid_index = [np.int(x) for x in range(len(info_file)) if 'k-point grid' in info_file[x]][0]
    size = np.asarray([np.int(x) for x in info_file[kgrid_index].split()[-3:]])
    kgrid_offset_index = [np.int(x) for x in range(len(info_file)) if 'k-point offset' in info_file[x]][0]
    vkloff = np.asarray([np.float(x) for x in info_file[kgrid_offset_index].split()[-3:]])

    ################################################################################
    #                   Reading kpoints, Energies from EIGVAL.OUT
    ################################################################################


    with open(input_energy, 'r') as f:
        eigenvalue_file = f.readlines()

    with open(input_kpoints,'r') as f:
        kpoints_file = f.readlines()


    nk = np.int(eigenvalue_file[0].split()[0])
    nb = np.int(eigenvalue_file[1].split()[0])


    for i in range(nb):
        if int(np.round(np.float(eigenvalue_file[5+i].split()[2]))) != 2 :
            break
    nv = i

    spin_index = [int(x) for x in range(len(info_file)) if 'Total valence charge' in info_file[x]][0]
    spin_factor =np.float([x for x in info_file[spin_index].split()][-1])/ nv

    klist1d = np.empty([nk,3])
    klist3d = np.empty(size)
    energy = np.empty([nk,nb])

    for ik in range(nk):
        klist1d[ik,0] = np.float64(kpoints_file[1+ik].split()[1])
        klist1d[ik,1] = np.float64(kpoints_file[1+ik].split()[2])
        klist1d[ik,2] = np.float64(kpoints_file[1+ik].split()[3])
        for ie in range(nb):
            energy[ik,ie] =  np.float64(eigenvalue_file[5+ie + (nb+4)*ik ].split()[1])

    klist3d = np.zeros(size,dtype=np.int)

    for i in range(nk):
        idx,idy,idz = [int(q) for q in np.rint(klist3d.shape * klist1d[i,:] - vkloff)]
        klist3d[idx,idy,idz] = i


    ################################################################################
    ### for hdf5 file name
    name_index = [x for x in range(len(info_file)) if 'name' in info_file[x]][0]
    material_name = info_file[name_index].split()[-1]
    exc_index = [x for x in range(len(info_file)) if 'Exchange-correlation' in info_file[x]][0]
    exc = [int(x) for x in (info_file[exc_index].split()[-3:])]

    if exc[0] is 100:
        exc = 'TrB'
    elif exc[0] == 2 or exc[0]== 4:
        exc = 'LDA'
    else:
        exc = ''

    Name = material_name+'_'+exc+\
        '_'+str(int(size[1]))+'x'+str(int(size[1]))+'x'+str(int(size[2]))+\
        '_nb_'+str(int(nb))+'.hdf5'

    ################################################################################
    #                  Reading matrix elements for "PMAT.OUT"
    ################################################################################
    k_bytes    = 8 # number of bytes requbired to store a single component of 3D k_vector
    pmat_bytes = 8 # number of bytes required to store a real/imaginary part of a single momentum matrix element
    n_bytes    = 4 # number of bytes required to store total number of bands

    momentum = np.empty([nk,nb,nb,3],dtype = complex)

    for i in range(nk):
        buffer =3*k_bytes+n_bytes + i*(3*k_bytes+n_bytes+ pmat_bytes*3*nb*nb*2)
        #print(buffer)
        with open('../PMAT.OUT','rb') as f:
            f.seek(buffer)
            data = np.fromfile(f, dtype = 'float64')

        temp = np.reshape(data[0:2*3*nb*nb],(2,nb,nb,3), order='F')
        momentum[i,...] = temp[0,...]+ 1j*temp[1,...]

    with h5py.File(os.path.join(output_file_path,Name), "w") as f:
        f.create_dataset("nk",data = nk)
        f.create_dataset("nb",data = nb)
        f.create_dataset("valence_bands",data = nv)
        f.create_dataset("size",data = size)
        f.create_dataset("spin_factor",data = spin_factor)
        f.create_dataset("klist1d",data = klist1d)
        f.create_dataset("klist3d",data = klist3d)
        f.create_dataset("lattice_vectors",data = lattice_vectors)
        f.create_dataset("reciprocal_vectors",data = reciprocal_vectors)
        f.create_dataset("energy",data = energy)
        f.create_dataset("momentum",data = momentum)

    print("HDF5 file is created : "+output_file_path )
