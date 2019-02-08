""" Python script for processing GPAW data. """
import h5py
import numpy as np
from ulmic import AtomicUnits
from ulmic.external import nearest_neighbor_table
from gpaw import restart
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.response.pair import PairDensity
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.wannier90 import get_overlap


def read_gpaw(input_gpw,nb_max=None,spin_factor=2,
              output_file='out.hdf5',calculate_overlap=True,
              calculate_momentum=True):
    """ Read .gpw file and create a hdf5 as output.
        Partially adapted from gpaw.wannier90.write_overlaps.

        input_gpw:      location of .gpw input file
        nb_max:         maximum band
        spin_factor:    electrons per band (only default value supported!)
    """

    nn = 1

    au = AtomicUnits()
    atoms, calc = restart(input_gpw, txt=None)
    # Generate full Brillouin zone if necessary
    if calc.parameters['symmetry'] != 'off':
        calc.set(symmetry='off')
        calc.get_potential_energy()
        calc.write(input_gpw.replace('.gpw', '_FBZ.gpw'), mode='all')
    calc.get_potential_energy()

    #Optional parameters:
    nb = calc.parameters['convergence']['bands']
    if nb_max is not None:
        nb = min(nb,nb_max)

    nk_size = calc.parameters['kpts']['size']
    nk = calc.wfs.kd.nbzkpts
    kpts = calc.get_bz_k_points()
    wfs = calc.wfs
    nvalence = calc.setups.nvalence//spin_factor
    lattice = atoms.get_cell()/au.AA
    reciprocal_lattice = (2*np.pi*au.AA)*atoms.get_reciprocal_cell()

    bands = range(nb)
    direction = np.eye(3, dtype=np.intp)
    klist1d = np.zeros((nk, 3))
    klist3d = np.zeros((nk_size), dtype=np.intp)
    energy = np.zeros((nk, nb))

    for i in range(nk):
        klist1d[i, :] = calc.wfs.kd.bzk_kc[i]

    for i in range(nk):
        ix, iy, iz = [int(q) for q in np.rint(nk_size*kpts[i, :])]
        klist3d[ix, iy, iz] = i

    nn_table = nearest_neighbor_table(klist3d,nn)

    energy_fermi = calc.get_fermi_level()
    for i in range(nk):
        energy[i, :] = (calc.get_eigenvalues(kpt=i, spin=0) - energy_fermi)[:nb]/au.eV

    if calculate_overlap:
        overlap = np.zeros((nk, 3, 2*nn, nb, nb), dtype=np.complex)
        icell_cv = (2 * np.pi) * np.linalg.inv(calc.wfs.gd.cell_cv).T
        r_grid = calc.wfs.gd.get_grid_point_coordinates()
        n_grid = np.prod(np.shape(r_grid)[1:]) * (False + 1)

        u_kng = []
        for i in range(nk):
            u_kng.append(np.array([wfs.get_wave_function_array(n, i, 0) for n in bands]))

        d0_aii = []
        for i in calc.wfs.kpt_u[0].p_ani.keys():
            d0_ii = calc.wfs.setups[i].d0_ii
            d0_aii.append(d0_ii)

        p_kani = []
        for i in range(nk):
            p_kani.append(calc.wfs.kpt_u[0 * nk + i].p_ani)

        for ik1 in range(nk):
            u1_ng = u_kng[ik1]
            for i in range(3):
                for j in range(-nn, nn):
                    ik2 = nn_table[ik1, i, j]
                    if j < 0:
                        g_vector = (kpts[ik1] + 1.0*j*direction[i, :]/nk_size)-kpts[ik2]
                    if j >= 0:
                        g_vector = (kpts[ik1] + 1.0*(j+1)*direction[i, :]/nk_size)-kpts[ik2]
                    bg_c= kpts[ik2] - kpts[ik1] + g_vector
                    bg_v = np.dot(bg_c, icell_cv)
                    u2_ng = u_kng[ik2] * np.exp(-1j * np.inner(r_grid.T, bg_v).T)
                    overlap[ik1, i, j, :, :] = get_overlap(calc,
                                                           bands,
                                                           np.reshape(u1_ng, (len(u1_ng), n_grid)),
                                                           np.reshape(u2_ng, (len(u2_ng), n_grid)),
                                                           p_kani[ik1],
                                                           p_kani[ik2],
                                                           d0_aii,
                                                           bg_v)[:nb, :nb]

    if calculate_momentum:
        pair = PairDensity(calc=calc)
        momentum = np.zeros((nk, nb, nb, 3), dtype=np.complex)
        delta_q_vector = [0.0, 0.0, 0.0]
        delta_q_descriptor = KPointDescriptor([delta_q_vector])
        plane_wave_descriptor = PWDescriptor(pair.ecut, calc.wfs.gd, complex,
                                             delta_q_descriptor)
        for i in range(nk):
            kpoint_pair = pair.get_kpoint_pair(plane_wave_descriptor,
                                               s=0, K=i, n1=0, n2=nb, m1=0, m2=nb)
            kpoint_momentum = pair.get_pair_momentum(plane_wave_descriptor,
                                                     kpoint_pair,
                                                     np.arange(0, nb),
                                                     np.arange(0, nb))
            momentum[i, :, :, :] = kpoint_momentum[..., 0]

    hdf5 = h5py.File(output_file, 'w')
    hdf5.create_dataset("energy", data=energy)
    hdf5.create_dataset("klist1d", data=klist1d)
    hdf5.create_dataset("klist3d", data=klist3d)
    hdf5.create_dataset("lattice_vectors", data=lattice.T)
    hdf5.create_dataset("reciprocal_vectors",data=reciprocal_lattice.T)
    hdf5.create_dataset("valence_bands", data=nvalence)
    hdf5.create_dataset("size", data=nk_size)
    hdf5.create_dataset("spin_factor", data=spin_factor)
    if calculate_momentum:
        hdf5.create_dataset("momentum", data=momentum)
    if calculate_overlap:
        hdf5.create_dataset("overlap", data=overlap)
        hdf5.create_dataset("neighbour_table", data=nn_table)
