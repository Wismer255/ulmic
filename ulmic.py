#! /usr/bin/env python
""" Utility program for ULMIC. Converts .gpw files to files
needed for dynamical calculations. Plots summaries of data sets. """


import os
import sys
import h5py
import numpy as np



if __name__ == "__main__":

    if '-h' in sys.argv or 'help' in sys.argv:
        print('Usage:')
        print('ulmic init')
        print('ulmic convert <input_format> <input_file> <output_directory> <optional_key=value>')
        print('ulmic analyse <input_data> <output_directory> <optional_key=value>')
        quit()

    if 'init' in sys.argv:
        variables = ['ULMIC_HOME','ULMIC_DATA','ULMIC_TEST','ULMIC_LOG']
        for variable in variables:
            print("The variable {} is current set to:".format(variable))
            try:
                print("  " + os.environ[variable])
            except:
                print("")
            try:
                new_path = str(input("New path for {}:".format(variable)))
            except SyntaxError:
                new_path = None

            print(new_path)
           # if new_path is not None:
           #     print(new_path)
           #     os.environ[variable] = new_path
        quit()

    if 'run_tests' in sys.argv:
    # #    from ulmic import testsuite
    # #    quit()
    #     import unittest
    #     from ulmic import testsuite
    #     # loader = unittest.TestLoader()
    #     # suite  = unittest.TestSuite()

    #     # # add tests to the test suite
    #     # suite.addTests(loader.loadTestsFromModule(post))
    #     # suite.addTests(loader.loadTestsFromModule(scenario))
    #     # suite.addTests(loader.loadTestsFromModule(thing))

    #     # # initialize a runner, pass it your suite and run it
    #     # runner = unittest.TextTestRunner(verbosity=3)
    #     # result = runner.run(suite)
    #     unittest.main()
    #     quit()
        import unittest
        # import your test modules
        # import ulmic.tests.general
        #import scenario
        #import thing

        # initialize the test suite
        loader = unittest.TestLoader()
        suite  = unittest.TestSuite()

        # add tests to the test suite
        # suite.addTests(loader.loadTestsFromModule(ulmic.tests.general))
        #suite.addTests(loader.loadTestsFromModule(scenario))
        #suite.addTests(loader.loadTestsFromModule(thing))

        # initialize a runner, pass it your suite and run it
        if 'profile' in sys.argv:
            import cProfile
            cProfile.run('unittest.TextTestRunner().run(suite)')
        else:
           runner = unittest.TextTestRunner(verbosity=3)
           result = runner.run(suite)



    if 'convert' in sys.argv:
        try:
            input_format = sys.argv[sys.argv.index('convert')+1]
        except:
            raise('No input format specified!')
        try:
            input_file = sys.argv[sys.argv.index('convert')+2]
        except:
            raise('No input file specified!')
        try:
            output_directory = sys.argv[sys.argv.index('convert')+3]
        except:
            raise('No output directory specified!')
        try:
            optional_arguments = sys.argv[sys.argv.index('convert')+4:]
            options = {}
            for arg in optional_arguments:
                key, val = arg.split('==')
                options[key] = val
            if bool(options):
                print('Optional arguments:')
                print(options)
        except:
            pass
    elif 'analyse' in sys.argv:
        try:
            input_data = sys.argv[sys.argv.index('analyse')+1]
        except:
            raise('No input data specified!')
        try:
            output_directory = sys.argv[sys.argv.index('analyse')+2]
        except:
            raise('No output directory specified!')
        try:
            optional_arguments = sys.argv[sys.argv.index('analyse')+3:]
            options = {}
            for arg in optional_arguments:
                key, val = arg.split('=')
                options[key] = val
            if bool(options):
                print('Optional arguments:')
                print(options)
        except:
            pass
    elif 'directional_analysis' in sys.argv:
            input_data = sys.argv[sys.argv.index('directional_analysis')+1]
    elif 'test_decoherence' in sys.argv:
            input_data = sys.argv[sys.argv.index('test_decoherence')+1]
    else:
        print('No command chosen. Use -h to list available commands.')
        quit()



    from ulmic import AtomicUnits
    au = AtomicUnits()


    flags = {'--no-overlap':False,}
    for arg in sys.argv:
        if arg in flags:
            flags[arg] = True


    if 'convert' in sys.argv and input_format == 'gpaw':
        from gpaw import restart
        from gpaw.kpt_descriptor import KPointDescriptor
        from gpaw.response.pair import PairDensity
        from gpaw.wavefunctions.pw import PWDescriptor
        from ase.units import Bohr

        au = AtomicUnits()
        output_directory += os.sep
        try:
            os.mkdir(output_directory)
        except:
            pass

        atoms, calc = restart(input_file, txt=None)
        if calc.parameters['symmetry'] != 'off':
            calc.set(symmetry='off')
            calc.get_potential_energy()
            calc.write(input_file.replace('.gpw', '_FBZ.gpw'), mode='all')
        calc.get_potential_energy()

        #Optional parameters:
        case = input_file.replace('.gpw', '')
        nb = calc.parameters['convergence']['bands']
        spin_factor = 2
        if 'nb' in options:
            nb = int(options['nb'])

        nk_size = calc.parameters['kpts']['size']
        nkpts = calc.wfs.kd.nbzkpts
        kpts = calc.get_bz_k_points()
        wfs = calc.wfs
        nvalence = calc.setups.nvalence//spin_factor
        lattice = atoms.get_cell()/au.A
        reciprocal_lattice = (2*np.pi*au.A)*atoms.get_reciprocal_cell()

        nk = len(kpts)
        nn = 1
        icell_cv = (2 * np.pi) * np.linalg.inv(calc.wfs.gd.cell_cv).T
        r_g = calc.wfs.gd.get_grid_point_coordinates()
        Ng = np.prod(np.shape(r_g)[1:]) * (False + 1)
        bands = range(nb)
        direction = np.eye(3, dtype=np.intp)

        klist1d = np.zeros((nkpts, 3))
        klist3d = np.zeros((nk_size), int)
        S = np.zeros((nk, 3, 2*nn, nb, nb), complex)
        energy = np.zeros((nk, nb))
        nn_table = np.zeros((nk, 3, 2*nn), int)

        np.save(output_directory+case+".gpaw.size.npy", nk_size)
        np.save(output_directory+case+".gpaw.valence.npy", nvalence)
        np.save(output_directory+case+".gpaw.lattice.npy", lattice)
        np.save(output_directory+case+".gpaw.reciprocal.npy", reciprocal_lattice)

        for i in range(nkpts):
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


        np.array([wfs.get_wave_function_array(0, 0, 0)])
        if not flags['--no-overlap']:
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
        momentum = np.zeros((nkpts, nb, nb, 3), complex)
        for i in range(nkpts):
            #k = b1*calc.wfs.kd.bzk_kc[i][0] + b2*calc.wfs.kd.bzk_kc[i][1] +b3*calc.wfs.kd.bzk_kc[i][2]
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
        dset_lattice = hdf5.create_dataset("lattice_vectors", data=lattice.T)
        dset_reciprocal = hdf5.create_dataset("reciprocal_vectors",data=reciprocal_lattice.T)
        dset_momentum = hdf5.create_dataset("momentum", data=momentum)
        if not flags['--no-overlap']:
            dset_overlap = hdf5.create_dataset("overlap", data=S)
        dset_table = hdf5.create_dataset("neighbour_table", data=nn_table)
        dset_valence = hdf5.create_dataset("valence_bands", data=nvalence)
        dset_size = hdf5.create_dataset("size", data=nk_size)
        dset_spin = hdf5.create_dataset("spin_factor", data=spin_factor)


    if 'analyse' in sys.argv:
        from ulmic import Medium
        from ulmic.medium.dielectric_function import DielectricResponse
        # from ulmic.medium.extrapolation import Extrapolation
        import matplotlib.pyplot as plt

        if input_data.endswith('hdf5'):
            medium = Medium(input_data,'hdf5')


        #medium.set_max_band_index(200)
        print(medium.nb)

        plt.matshow(abs(medium.momentum[medium.k_gamma,:,:,0]))
        plt.colorbar()

        plt.matshow(np.log(abs(medium.momentum[medium.k_gamma,:,:,0])))
        plt.colorbar()
    #    plt.matshow(np.real(medium.momentum[0,:,:,0]))
    #    plt.matshow(np.imag(medium.momentum[0,:,:,0]))
        plt.show()

        #print(medium.klist1d)

        for i in range(medium.nb):
            for j in range(i+1,medium.nb):
                if abs(medium.energy[0,i]-medium.energy[0,j]) < 1e-5:
                    print(abs(medium.energy[0,i]-medium.energy[0,j]),abs(medium.momentum[0,i,j,0]))


        # from ulmic.medium.sum_rules import MediumConvergence

        # conv = MediumConvergence(medium)
        #conv.test_sum_rule_plane_z()
        #conv.test_1_and_2_derivative()
        #conv.test_gamma_extrapolation()

       # medium.set_max_band_index(160)
       # medium.set_min_band_index(0)
       # medium.set_k_point_reduction([8,8,4])

    #    direction = np.array([[1,1,1],[0,1,-1],[2,-1,-1]]).astype(float)
    #    print([direction[q,:] for q in range(3)])

        #print(medium.momentum)
        #dielectric_response = DielectricResponse(medium)
        #fig,ax = dielectric_response.plot_dielectric_response_3(variable_decoherence=True)
        #fig,ax = dielectric_response.plot_dielectric_response_3()#direction=direction)
        #plt.show()
        print(medium.klist1d)
        medium.set_max_band_index(200)
        # extrapolation = Extrapolation(medium)
        # extrapolation.extrapolate_fcc_band_plot(range(medium.nk))

        # plt.figure()
        # for i in [300, 250, 200, 150, 100, 50]:
        #     medium.set_max_band_index(i)
        #     extrapolation.extrapolate_fcc_band_plot([0],newfigure=False,show=False)
        # plt.show()

        #fig,ax = extrapolation.check_validity()

        #Energy, oscillator, berry
        # extrapolation.plot_bands()

        # extrapolation.plot_extrapolation_along_axis()
        # plt.show()

        # gauge_dependent_response = GaugeDependentReponse(medium)
        # gauge_dependent_response.get_delta_response()
        # extrapolation.plot_bands()
        #    extrapolation.plot_extrapolation_along_axis()

        # with PdfPages(output_directory+'.pdf') as pdf:
        #     # fig,ax = dielectric_response.plot_dielectric_response()
        #     fig,ax = extrapolation.check_validity()
        #     pdf.savefig()
        #     plt.close()

        direct_band_gap = au.eV*np.min(medium.energy[:, medium.nv]-medium.energy[:, medium.nv-1])
        indirect_band_gap = au.eV*(np.min(medium.energy[:, medium.nv])-np.max(medium.energy[:, medium.nv-1]))

        output_text = '''Material summary:
    Direct band gap: {:.3f} eV
    Indirect band gap: {:.3f} eV
    Number of bands: {:d}
    Number of valence bands: {:d}
        '''.format(direct_band_gap, indirect_band_gap, medium.nb, medium.nv)

        print(output_text)

        plt.show()




    if 'directional_analysis' in sys.argv:
        from ulmic import Medium
        # from ulmic.medium.dielectric_function import DielectricResponse
        # from ulmic.medium.extrapolation import Extrapolation
        import matplotlib.pyplot as plt

        if input_data.endswith('hdf5'):
            medium = Medium(input_data,'hdf5')

        #from ulmic.medium.sum_rules import MediumConvergence

        #conv = MediumConvergence(medium)
        #conv.test_sum_rule_plane_z()
        #conv.test_1_and_2_derivative()
        #conv.test_gamma_extrapolation()

       # medium.set_max_band_index(160)
       # medium.set_min_band_index(0)
        medium.set_k_point_reduction(medium.size)

    #    direction = np.array([[1,1,1],[0,1,-1],[2,-1,-1]]).astype(float)
    #    print([direction[q,:] for q in range(3)])

        # print(medium.momentum)
        # dielectric_response = DielectricResponse(medium)
        # fig,ax = dielectric_response.plot_dielectric_response_3(variable_decoherence=True)
        # direction = np.array([[1,1,1],[0,1,-1],[2,-1,-1]]).astype(float)
        # fig,ax = dielectric_response.plot_dielectric_response_3(direction=direction,yaxis='log',omega=np.linspace(0.0,20.0/au.eV,4000))

        # direction = np.array([[1,0,0],[0,1,0],[0,0,1]]).astype(float)
        # fig,ax = dielectric_response.plot_dielectric_response_3(direction=direction,yaxis='log',omega=np.linspace(0.0,20.0/au.eV,4000))
        # plt.show()
        # print(medium.klist1d)
        # medium.set_max_band_index(200)
        # extrapolation = Extrapolation(medium)
        # extrapolation.extrapolate_fcc_band_plot(range(medium.nk))

        # plt.figure()
        # for i in [300, 250, 200, 150, 100, 50]:
        #     medium.set_max_band_index(i)
        #     extrapolation.extrapolate_fcc_band_plot([0],newfigure=False,show=False)
        # plt.show()

        # #fig,ax = extrapolation.check_validity()

        # #Energy, oscillator, berry
        # extrapolation.plot_bands()

        # extrapolation.plot_extrapolation_along_axis()
        # plt.show()

        # # gauge_dependent_response = GaugeDependentReponse(medium)
        # gauge_dependent_response.get_delta_response()
        # extrapolation.plot_bands()
        #    extrapolation.plot_extrapolation_along_axis()

        # with PdfPages(output_directory+'.pdf') as pdf:
        #     # fig,ax = dielectric_response.plot_dielectric_response()
        #     fig,ax = extrapolation.check_validity()
        #     pdf.savefig()
        #     plt.close()

        direct_band_gap = au.eV*np.min(medium.energy[:, medium.nv]-medium.energy[:, medium.nv-1])
        indirect_band_gap = au.eV*(np.min(medium.energy[:, medium.nv])-np.max(medium.energy[:, medium.nv-1]))

        output_text = '''Material summary:
    Direct band gap: {:.3f} eV
    Indirect band gap: {:.3f} eV
    Number of bands: {:d}
    Number of valence bands: {:d}
        '''.format(direct_band_gap, indirect_band_gap, medium.nb, medium.nv)

        print(output_text)

        plt.show()




    if 'test_decoherence' in sys.argv:
        from ulmic import Medium
        import matplotlib.pyplot as plt
        from ulmic import UltrafastLightMatterInteraction
        from ulmic import Pulses

        if input_data.endswith('hdf5'):
            medium = Medium(input_data,'hdf5')

        medium.set_max_band_index(20)
        medium.set_k_point_reduction(medium.size)
        print(medium.klist1d)

        nm_Eph = 2*np.pi*137.035999/(1/0.052917721)
        v1 = np.array([1,0,0])

        pulses = Pulses(['pulse_A'])
        pulses.variables[0]['E0'] = 1.0/au.VA
        pulses.variables[0]['omega'] = nm_Eph/700.0
        pulses.variables[0]['FWHM'] = 4.0/au.fs
        pulses.variables[0]['polarisation_vector'] = v1

        time = np.linspace(-60,60,40000)/au.fs

        # First test
        # ulmi_tdse = UltrafastLightMatterInteraction(medium,pulses,time,compression=1,step_size='fixed',animation=True,n_threads=0)
        # ulmi_tdse.run_stepwise('vg','tdse')
        # ulmi_lvn = UltrafastLightMatterInteraction(medium,pulses,time,compression=1,step_size='fixed',animation=True,n_threads=0)
        # ulmi_lvn.run_stepwise('vg','lvn')

        # ulmi_lvn1 = UltrafastLightMatterInteraction(medium,pulses,time,compression=1,step_size='fixed',animation=True,n_threads=0)
        # ulmi_lvn1.gamma = 0.001/(medium.get_direct_band_gap()**2)
        # ulmi_lvn1.run_stepwise('vg','lvn')
        # ulmi_lvn2 = UltrafastLightMatterInteraction(medium,pulses,time,compression=1,step_size='fixed',animation=True,n_threads=0)
        # ulmi_lvn2.gamma = 0.01/(medium.get_direct_band_gap()**2)
        # ulmi_lvn2.run_stepwise('vg','lvn')
        # plt.figure()
        # plt.plot(ulmi_tdse.result_t,ulmi_tdse.result_j2[:,0])
        # plt.plot(ulmi_lvn.result_t,ulmi_lvn.result_j2[:,0],'--')
        # plt.plot(ulmi_lvn1.result_t,ulmi_lvn1.result_j2[:,0])
        # plt.plot(ulmi_lvn2.result_t,ulmi_lvn2.result_j2[:,0])
        # plt.figure()
        # plt.semilogy(abs(np.fft.fft(ulmi_tdse.result_j2[:,0])))
        # plt.semilogy(abs(np.fft.fft(ulmi_lvn.result_j2[:,0])))
        # plt.semilogy(abs(np.fft.fft(ulmi_lvn1.result_j2[:,0])))
        # plt.semilogy(abs(np.fft.fft(ulmi_lvn2.result_j2[:,0])))
        # plt.show()

        # Second test

        fig1,ax1 = plt.subplots()
        fig2,ax2 = plt.subplots()
        fields = [1.0,1.5,2.0]
        for i in range(len(fields)):
            pulses.variables[0]['E0'] = fields[i]/au.VA
            ulmi_lvn1 = UltrafastLightMatterInteraction(medium,pulses,time,compression=1,step_size='fixed',animation=True,n_threads=0)
            ulmi_lvn1.gamma = 0.01/(medium.get_direct_band_gap()**2)
            ulmi_lvn1.run_stepwise('vg','lvn')

            ax1.plot(ulmi_lvn1.result_t*au.fs,ulmi_lvn1.result_j2[:,0]/fields[i])
            # plt.plot(ulmi_lvn2.result_t,ulmi_lvn2.result_j2[:,0])
            # plt.plot(ulmi_lvn3.result_t,ulmi_lvn3.result_j2[:,0])
            freq = 2*np.pi*au.eV*np.fft.fftfreq(len(ulmi_lvn1.result_t),np.max(np.diff(ulmi_lvn1.result_t)))
            ax2.semilogy(freq,abs(np.fft.fft(ulmi_lvn1.result_j2[:,0]))**2 / fields[i] )
            # plt.semilogy(abs(np.fft.fft(ulmi_lvn2.result_j2[:,0])))
            # plt.semilogy(abs(np.fft.fft(ulmi_lvn3.result_j2[:,0])))
        plt.show()
