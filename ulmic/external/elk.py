import os
import numpy as np
import h5py


def create_elk_hdf5(path_to_elk_OUT, output_file_path):
    input_info     = os.path.join(path_to_elk_OUT,'INFO.OUT'   )
    input_lattice  = os.path.join(path_to_elk_OUT,'LATTICE.OUT')
    input_kpoints  = os.path.join(path_to_elk_OUT,'KPOINTS.OUT')
    input_energy   = os.path.join(path_to_elk_OUT,'EIGVAL.OUT' )
    input_momentum = os.path.join(path_to_elk_OUT,'PMAT.OUT'   )


    ############################################################################
    #          Reading lattice vectors and reciprocal lattice vectors
    ############################################################################

    with open(input_lattice, 'r') as f:
        lattice_file = f.readlines()


    lattice_vectors = np.empty([3,3])
    reciprocal_vectors = np.empty([3,3])
    avec_ind = [x+5 for x in range(len(lattice_file)) if 'vector a1' in lattice_file[x]][0]
    bvec_ind = [x+5 for x in range(len(lattice_file)) if 'vector b1' in lattice_file[x]][0]

    for i in range(3):
        for j in range(3):
            lattice_vectors[i,j]    = np.float(lattice_file[avec_ind+i].split()[j])
            reciprocal_vectors[i,j] = np.float(lattice_file[bvec_ind+i].split()[j])


    ############################################################################
    #              Reading kpoints grid size and grid shift from INFO.OUT
    ############################################################################
    with open(input_info, 'r') as f:
        info_file = f.readlines()


    kgrid_index = [np.int(x) for x in range(len(info_file)) if 'k-point grid' in info_file[x]][0]
    size = np.asarray([np.int(x) for x in info_file[kgrid_index].split()[-3:]])
    kgrid_offset_index = [np.int(x) for x in range(len(info_file)) if 'k-point offset' in info_file[x]][0]
    vkloff = np.asarray([np.float(x) for x in info_file[kgrid_offset_index].split()[-3:]])


    ############################################################################
    #                   Reading kpoints, Energies from EIGVAL.OUT
    ############################################################################


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


    ############################################################################


    ### for hdf5 file name

    name_index = [x for x in range(len(info_file)) if 'name' in info_file[x]][:]
    materialname = ["" for x in range(len(name_index))]
    for i in range(len(name_index)):
        material_name = info_file[name_index[i]-2].split()[-1]
        materialname[i] = material_name[1:-1]
    material_name = ''.join(materialname)

    band_gap_index = [x for x in range(len(info_file)) if 'direct band gap' in info_file[x]][-1]
    Eg =np.round(np.float64(info_file[band_gap_index].split()[-1])*27.21,2)

    exc_index = [x for x in range(len(info_file)) if 'Exchange-correlation' in info_file[x]][0]
    exc = [int(x) for x in (info_file[exc_index].split()[-3:])]

    if exc[0] is 100:
        exc = 'TrB'
    elif exc[0] == 2 or exc[0]== 4:
        exc = 'LDA'
    else:
        exc = ''

    Name = material_name+'_'+exc+'_'+str(Eg)+'eV'+\
        '_'+str(int(size[1]))+'x'+str(int(size[1]))+'x'+str(int(size[2]))+\
        '_nb_'+str(int(nb))+'.hdf5'

    if nk == np.prod(size):
        Name = material_name+\
        "_{0}x{1}x{2}_TrB_{3:4.2f}eV_nb_{4}.hdf5".format(size[0],size[1],size[2],Eg,nb)
    else:
        Name = material_name+\
        "_kpts_{0}_TrB_{1:4.2f}eV_nb_{2}.hdf5".format(nk,Eg,nb)




    ############################################################################
    #                  Reading matrix elements for "PMAT.OUT"
    ############################################################################
    """

    ELK stores momentum matrix elements in the following binary format:

    Lets assume, we have "nk" number of kpoints and "nb" number of energy bands.

    To store a single band number it needs 4bytes.
    To store a single kpoint component (let say "kx") it needs 8bytes.
    To store a complete k-vector it needs 8x3 = 24bytes
    To store a single component of momentum matrix it needs 16bytes (8 for the real part
    and 8 for the imaginary part)

    Total size of the binary file PMAT.OUT is = nk*(4+24+nb*nb*3*2) bytes

    A single line in "PMAT.OUT" file represents a byte.
    First 28bytes(3*8+4) or lines of PMAT.OUT stores the k-vectors and band number information then
    next nb*nb*3*2 bytes contains momentum matrix elements for the k-vectors defined in first 28bytes.

    | kx[0]    }
    | ky[0]      }
    | kz[0]        }
    | nb             }>| First (4+24+nb*nb*3*2) bytes or lines contains inforamtion about all
    | PMAT[0,...]    }>| the momentum matrix elements belongs to the k-point number zero
    |  .           }
    |  .         }
    |  .       }


    | kx[1]   }
    | ky[1]     }
    | kz[1]       }
    | nb            }>| Next, (4+24+nb*nb*3*2) bytes or lines contains inforamtion about all the momentum matrix
    | PMATs         }>| elements belongs to the k-point number one
    |  .          }
    |  .        }
    |  .      }

    and so...

    """

    k_bytes    = 8 # number of bytes required to store a single component of 3D k_vector
    pmat_bytes = 8 # number of bytes required to store a real/imaginary part of a single momentum matrix element
    n_bytes    = 4 # number of bytes required to store total number of bands

    offset = 3*k_bytes+n_bytes # first 28 bytes are used to store 3 components of k_vector(3*k_bytes) and band index ()
    pmat_total_bytes = pmat_bytes*3*nb*nb*2
    momentum = np.empty([nk,nb,nb,3],dtype = complex)

    with open(input_momentum,'rb') as f:
        for i in range(nk):
            f.seek(offset + i*(offset+pmat_total_bytes))
            data = np.fromfile(f, dtype = 'float64',count = pmat_total_bytes)
            temp = np.reshape(data[0:int(pmat_total_bytes/pmat_bytes)],(2,nb,nb,3), order='F')
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
