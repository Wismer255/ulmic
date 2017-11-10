from gpaw import restart
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.response.pair import PairDensity
from gpaw.wavefunctions.pw import PWDescriptor
from ase.units import Bohr


def read_gpaw(input_gpw,nb=None):

    spin_factor = 2
    nn = 1

    au = AtomicUnits()
    atoms, calc = restart(input_gpw, txt=None)
    if calc.parameters['symmetry'] != 'off':
        calc.set(symmetry='off')
        calc.get_potential_energy()
        calc.write(input_gpw.replace('.gpw', '_FBZ.gpw'), mode='all')
    calc.get_potential_energy()

    #Optional parameters:
    case = input_gpw.replace('.gpw', '')
    nb = calc.parameters['convergence']['bands']
    if 'nb' in options:
        nb = int(options['nb'])

    nk_size = calc.parameters['kpts']['size']
    nk = calc.wfs.kd.nbzkpts
    kpts = calc.get_bz_k_points()
    wfs = calc.wfs
    nvalence = calc.setups.nvalence//spin_factor
    lattice = atoms.get_cell()/au.A
    reciprocal_lattice = (2*np.pi*au.A)*atoms.get_reciprocal_cell()

    bands = range(nb)
    direction = np.eye(3, dtype=int)

    klist1d = np.zeros((nk, 3))
    klist3d = np.zeros((nk_size), int)
    S = np.zeros((nk, 3, 2*nn, nb, nb), complex)
    energy = np.zeros((nk, nb))
    nn_table = np.zeros((nk, 3, 2*nn), int)

    np.save(output_directory+case+".gpaw.size.npy", nk_size)
    np.save(output_directory+case+".gpaw.valence.npy", nvalence)
    np.save(output_directory+case+".gpaw.lattice.npy", lattice)
    np.save(output_directory+case+".gpaw.reciprocal.npy", reciprocal_lattice)

    for i in range(nk):
        klist1d[i, :] = calc.wfs.kd.bzk_kc[i]
    np.save(output_directory+case+".gpaw.klist1d.npy", klist1d)

    for i in range(nk):
        idx, idy, idz = [int(q) for q in np.rint(nk_size*kpts[i, :])]
        klist3d[idx, idy, idz] = i
    np.save(output_directory+case+".gpaw.klist3d.npy", klist3d)

    for ix in range(nk_size[0]):
        for iy in range(nk_size[1]):
            for iz in range(nk_size[2]):
                i = klist3d[ix, iy, iz]
                for j in range(-nn, nn):
                    if j < 0:
                        nn_table[i, 0, j] = klist3d[(ix+j)%nk_size[0], iy, iz]
                        nn_table[i, 1, j] = klist3d[ix, (iy+j)%nk_size[1], iz]
                        nn_table[i, 2, j] = klist3d[ix, iy, (iz+j)%nk_size[2]]
                    if j >= 0:
                        nn_table[i, 0, j] = klist3d[(ix+j+1)%nk_size[0], iy, iz]
                        nn_table[i, 1, j] = klist3d[ix, (iy+j+1)%nk_size[1], iz]
                        nn_table[i, 2, j] = klist3d[ix, iy, (iz+j+1)%nk_size[2]]
    np.save(output_directory+case+".gpaw.nn_table.npy", nn_table)

    ef = calc.get_fermi_level()
    for ik in range(nk):
        energy[ik, :] = (calc.get_eigenvalues(kpt=ik, spin=0) - ef)[:nb]/au.eV
    np.save(output_directory+case+".gpaw.energy.npy", energy)


    if not flags['--no-overlap']:
        icell_cv = (2 * np.pi) * np.linalg.inv(calc.wfs.gd.cell_cv).T
        r_g = calc.wfs.gd.get_grid_point_coordinates()
        Ng = np.prod(np.shape(r_g)[1:]) * (False + 1)

        u_knG = []
        for i in range(nk):
            u_knG.append(np.array([wfs.get_wave_function_array(n, i, 0) for n in bands]))

        dO_aii = []
        for ia in calc.wfs.kpt_u[0].P_ani.keys():
            dO_ii = calc.wfs.setups[ia].dO_ii
            dO_aii.append(dO_ii)

        P_kani = []
        for ik in range(nk):
            P_kani.append(calc.wfs.kpt_u[0 * nk + ik].P_ani)

        def get_overlap(calc, bands, u1_nG, u2_nG, P1_ani, P2_ani, dO_aii, bG_v):
            M_nn = np.dot(u1_nG.conj(), u2_nG.T) * calc.wfs.gd.dv
            r_av = calc.atoms.positions / Bohr
            for ia in range(len(P1_ani.keys())):
                P1_ni = P1_ani[ia][bands]
                P2_ni = P2_ani[ia][bands]
                phase = np.exp(-1.0j * np.dot(bG_v, r_av[ia]))
                dO_ii = dO_aii[ia]
                M_nn += P1_ni.conj().dot(dO_ii).dot(P2_ni.T) * phase
            return M_nn

        for ik1 in range(nk):
            u1_nG = u_knG[ik1]
            for i in range(3):
                for j in range(-nn, nn):
                    ik2 = nn_table[ik1, i, j]
                    if j < 0:
                        G = (kpts[ik1] + 1.0*j*direction[i, :]/nk_size)-kpts[ik2]
                    if j >= 0:
                        G = (kpts[ik1] + 1.0*(j+1)*direction[i, :]/nk_size)-kpts[ik2]
                    bG_c = kpts[ik2] - kpts[ik1] + G
                    bG_v = np.dot(bG_c, icell_cv)
                    u2_nG = u_knG[ik2] * np.exp(-1j * np.inner(r_g.T, bG_v).T)
                    S[ik1, i, j, :, :] = get_overlap(  calc, 
                                                       bands, 
                                                       np.reshape(u1_nG, (len(u1_nG), Ng)), 
                                                       np.reshape(u2_nG, (len(u2_nG), Ng)), 
                                                       P_kani[ik1], 
                                                       P_kani[ik2], 
                                                       dO_aii, 
                                                       bG_v)[:nb, :nb]
            np.save(output_directory+case+".gpaw.S.npy", S)

    pair = PairDensity(calc=calc)
    momentum = np.zeros((nk, nb, nb, 3), complex)
    for i in range(nk):
        q_c = [0.0, 0.0, 0.0]
        qd = KPointDescriptor([q_c])
        pd = PWDescriptor(pair.ecut, calc.wfs.gd, complex, qd)
        kptpair = pair.get_kpoint_pair(pd, s=0, K=i, n1=0, n2=nb, m1=0, m2=nb)
        ol = np.allclose(q_c, 0.0)
        n_nmvG = pair.get_pair_momentum(pd, kptpair, np.arange(0, nb), np.arange(0, nb))
        momentum[i, :, :, :] = n_nmvG[..., 0][:nb, :nb, :]
    np.save(output_directory+case+".gpaw.momentum.npy", momentum)

    hdf5 = h5py.File(output_directory.rstrip('/')+'.hdf5', 'w')
    dset_energy = hdf5.create_dataset("energy", data=energy)
    dset_klist1d = hdf5.create_dataset("klist1d", data=klist1d)
    dset_klist3d = hdf5.create_dataset("klist3d", data=klist3d)
    dset_lattice = hdf5.create_dataset("lattice", data=lattice)
    dset_reciprocal = hdf5.create_dataset("reciprocal_lattice",data=reciprocal_lattice)
    dset_table = hdf5.create_dataset("neighbour_table", data=nn_table)
    dset_valence = hdf5.create_dataset("valence", data=nvalence)
    dset_size = hdf5.create_dataset("size", data=nk_size)
    dset_spin = hdf5.create_dataset("spin_factor", data=spin_factor)
    dset_momentum = hdf5.create_dataset("momentum", data=momentum)
    if not flags['--no-overlap']:
        dset_overlap = hdf5.create_dataset("overlap", data=S)