import pytest
import numpy as np
from ulmic.environment import UlmicEnvironment
import matplotlib.pyplot as plt
from ulmic.atomic_units import AtomicUnits as au
from ulmic.tests.simulation_batch import SimulationBatch
from ulmic.logs import Logs

file_medium = 'convergencetest2.hdf5'
#args = (file_medium, 'probe_UV', np.linspace(-400.0,400.0,20))
#MAX_BAND = 3
log = Logs('convergencetests')


class TestsConvergence():
    """ Create medium for 1d potential and calculate optical response
        in both gauges with dephasing. """

    # @pytest.mark.first
    # def test_create_1d_potential_fast(self,):
    #     """ Test construction of a 1D potential data file. """
    #     from ulmic.external.localpotential import LocalPotential1D
    #     medium = LocalPotential1D()
    #     medium.set_spatial_parameters(points_per_cell=800)
    #     medium.set_medium_parameters(bands=12,
    #                                  k_points=400,
    #                                  valence_bands=1,
    #                                  spin_factor=2)
    #     medium.save_as_medium_to_hdf5(file_medium)
    #     print(medium.check_real_space_eigenstate(0,0))
    #     print(medium.check_real_space_eigenstate(0,1))
    #     print(medium.check_real_space_eigenstate(0.2,0))
    #


#class TestsPublication():
    def test_dynamical_calculation_vg_1d_potential_weak(self,):
        import numpy as np
        from ulmic import Medium, Pulses, AtomicUnits
        from ulmic.ulmi.ulmi import UltrafastLightMatterInteraction

        au = AtomicUnits()
        medium = Medium(file_medium)
        medium.calculate_second_neighbour_overlaps()
        medium.set_max_band_index(6)
        pulses = Pulses(['pulse_A'])
        pulses.variables[0]['E0'] = 0.5/au.VA
        pulses.variables[0]['FWHM'] = 2.0/au.fs
        pulses.variables[0]['omega'] = 1.6/au.eV
        time = np.linspace(-20.0/au.fs,120.0/au.fs,8000)
        ulmi = UltrafastLightMatterInteraction(medium,pulses,time)
        options = {'time_step': 'auto'}
        ulmi.set_flags_and_options(options)
        ulmi.set_parameters(gauge='vg', equation='tdse')
        result = ulmi.run()
        result.save_data_to_file('convergencetests_weak_vg_tdse.hdf5')
        alpha = result.get_susceptibility_scalar()
        beta = result.compare_polarisations()
        gamma = result.calc_susceptibility_time_domain()
        log.log('alpha {0}'.format(alpha))
        log.log('beta {0}'.format(beta))
        log.log('gamma {0}'.format(gamma))

    def test_dynamical_calculation_lg_1d_potential_weak(self,):
        import numpy as np
        from ulmic import Medium, Pulses, UltrafastLightMatterInteraction,AtomicUnits

        au = AtomicUnits()
        medium = Medium(file_medium)
        medium.calculate_second_neighbour_overlaps()
        medium.set_max_band_index(2)
        pulses = Pulses(['pulse_A'])
        pulses.variables[0]['E0'] = 0.5/au.VA
        pulses.variables[0]['FWHM'] = 2.0/au.fs
        pulses.variables[0]['omega'] = 1.6/au.eV
        time = np.linspace(-20.0/au.fs,120.0/au.fs,8000)

        ulmi = UltrafastLightMatterInteraction(medium,pulses,time)#,options={'time_step_min':1e-3,'relative_error_tolerance':1e-7})
        options = {'time_step': 'auto'}
        ulmi.set_flags_and_options(options)
        ulmi.set_parameters(gauge='lg', equation='tdse')

        result = ulmi.run()
        result.save_data_to_file('convergencetests_weak_lg_tdse2.hdf5')
        alpha = result.get_susceptibility_scalar()
        beta = result.compare_polarisations()
        gamma = result.calc_susceptibility_time_domain()
        log.log('alpha {0}'.format(alpha))
        log.log('beta {0}'.format(beta))
        log.log('gamma {0}'.format(gamma))
#
#     def test_dynamical_calculation_vg_1d_potential_mid(self,):
#         #Running: 200,10,(-10,60,4000),1e-3,1e-8,0.1  (a)   ->
#         #Running: 200,8,(-20,90,25000),1e-3,1e-8,0.1  (aa)  ->
#         import numpy as np
#         from ulmic import Medium, Pulses, UltrafastLightMatterInteraction,AtomicUnits
#
#         file_name = os.path.join(test_dir,'data_1d_convergence_100k.hdf5')
#         au = AtomicUnits()
#         medium = Medium(file_name,'hdf5')
#         medium.calculate_second_neighbour_overlaps()
#         medium.set_max_band_index(8)
#         pulses = Pulses(['pulse_A'])
#         pulses.variables[0]['E0'] = 0.5/au.VA
#         pulses.variables[0]['FWHM'] = 2.0/au.fs
#         pulses.variables[0]['omega'] = 1.6/au.eV
#         time = np.linspace(-20.0/au.fs,120.0/au.fs,8000)
#         ulmi = UltrafastLightMatterInteraction(medium,pulses,time,step_size='auto',options={'time_step_min':1e-3,relative_error_tolerance':1e-8})
#         ulmi.gamma = 0.1
#         result = ulmi.run_stepwise(gauge='vg',equation='lvn')
#         file_result = os.path.join(test_dir,'result_1d_vg_dephasing_convergence_pub5_100k_1e-20.hdf5')
#         result.save_data_to_file(file_result)
#
#
#     def test_dynamical_calculation_lg_1d_potential_mid(self,):
#         # 200, 4, 0.5, 4000, 0.1, 1e-2, 1e-6        -> Breaks at 6.31 a.u.
#
#         import numpy as np
#         from ulmic import Medium, Pulses, UltrafastLightMatterInteraction,AtomicUnits
#
#         file_name = os.path.join(test_dir,'data_1d_convergence_800k.hdf5')
#         au = AtomicUnits()
#         medium = Medium(file_name,'hdf5')
#         medium.calculate_second_neighbour_overlaps()
#         medium.set_max_band_index(3)#(6)
#         pulses = Pulses(['pulse_A'])
#         pulses.variables[0]['E0'] = 0.5/au.VA
#         pulses.variables[0]['FWHM'] = 2.0/au.fs
#         pulses.variables[0]['omega'] = 1.6/au.eV
#         time = np.linspace(-20.0/au.fs,120.0/au.fs,8000)
#         ulmi = UltrafastLightMatterInteraction(medium,pulses,time,step_size='auto',options={'time_step_min':1e-3,'relative_error_tolerance':1e-8})
#         ulmi.gamma = 0.1
#         result = ulmi.run_stepwise(gauge='lg',equation='lvn')
#         file_result = os.path.join(test_dir,'result_1d_lg_dephasing_convergence_pub5_800k_3bands.hdf5')
#         result.save_data_to_file(file_result)
#
#     # 1D with dephasing:
#     def test_dynamical_calculation_vg_1d_potential_fast(self,):
#         import numpy as np
#         from ulmic import Medium, Pulses, UltrafastLightMatterInteraction,AtomicUnits
#
#         file_name = os.path.join(test_dir,'data_1d_convergence_200k.hdf5')
#         au = AtomicUnits()
#         medium = Medium(file_name,'hdf5')
#         medium.calculate_second_neighbour_overlaps()
#         medium.set_max_band_index(8)
#         pulses = Pulses(['pulse_A'])
#         pulses.variables[0]['E0'] = 1.5/au.VA
#         pulses.variables[0]['FWHM'] = 2.0/au.fs
#         pulses.variables[0]['omega'] = 1.6/au.eV
#         time = np.linspace(-20.0/au.fs,120.0/au.fs,8000)
#         ulmi = UltrafastLightMatterInteraction(medium,pulses,time,step_size='auto',options={'time_step_min':1e-3,'relative_error_tolerance':1e-8})
#         ulmi.gamma = 0.1
#         result = ulmi.run_stepwise(gauge='vg',equation='lvn')
#         file_result = os.path.join(test_dir,'result_1d_vg_dephasing_convergence_pub1.hdf5')
#         result.save_data_to_file(file_result)
#
#
#     def test_dynamical_calculation_lg_1d_potential_fast(self,):
#         #        100 nk, 2 bands, 1.5 V/A, (1e-1,1e-5), 0.01 works...
#         #Testing 200 nk, 3 bands, 1.5 V/A, (1e-3,1e-7), 0  -> stops at -0.5 fs (-20 au)
#         #Testing 100 nk, 3 bands, 1.5 V/A, (1e-3,1e-7), 0  -> stops at -0.5 fs (-1.28 au)
#         #Testing 100 nk, 3 bands, 1.5 V/A, (1e-3,1e-7), 0  -> works perfectly (1934 seconds)
#         #Testing 200 nk, 3 bands, 1.5 V/A, (1e-3,1e-7), 0  -> works perfectly (6000 seconds)
#         #Testing 100 nk, 3 bands, 1.5 V/A, (1e-3,1e-7), 0.1  -> works perfectly (3800 seconds)
#
#         #Testing 200 nk, 3 bands, 1.5 V/A, (1e-3,1e-7), 0.1  -> works perfectly (5800 seconds)
#         #Testing 200 nk, 4 bands, 1.5 V/A, (1e-3,1e-7), 0.1  -> breaks at -10.5
#         import numpy as np
#         from ulmic import Medium, Pulses, UltrafastLightMatterInteraction,AtomicUnits
#
#         file_name = os.path.join(test_dir,'data_1d_convergence_800k.hdf5')
#         au = AtomicUnits()
#         medium = Medium(file_name,'hdf5')
#         medium.calculate_second_neighbour_overlaps()
#         medium.set_max_band_index(4)#(6)
#         pulses = Pulses(['pulse_A'])
#         pulses.variables[0]['E0'] = 1.5/au.VA#1.0/au.VA
#         pulses.variables[0]['FWHM'] = 2.0/au.fs
#         pulses.variables[0]['omega'] = 1.6/au.eV
# #        time = np.linspace(-20.0/au.fs,60.0/au.fs,4000)
#         time = np.linspace(-20.0/au.fs,120.0/au.fs,8000)
# #        ulmi = UltrafastLightMatterInteraction(medium,pulses,time,step_size='auto',options={'time_step_min':1e-3,'relative_error_tolerance':1e-8})
#         ulmi = UltrafastLightMatterInteraction(medium,pulses,time,step_size='auto',options={'time_step_min':1e-3,'relative_error_tolerance':1e-8})
#         ulmi.gamma = 0.1
#         result = ulmi.run_stepwise(gauge='lg',equation='lvn')
#         file_result = os.path.join(test_dir,'result_1d_lg_dephasing_convergence_pub1_800k_4bands.hdf5')
#         result.save_data_to_file(file_result)
#
#
#
# class TestsPublicationMid():
#     def test_dynamical_calculation_vg_1d_potential_weak(self,):
#         import numpy as np
#         from ulmic import Medium, Pulses, UltrafastLightMatterInteraction,AtomicUnits
#
#         file_name = os.path.join(test_dir,'data_1d_convergence_200k.hdf5')
#         au = AtomicUnits()
#         medium = Medium(file_name,'hdf5')
#         medium.calculate_second_neighbour_overlaps()
#         medium.set_max_band_index(8)
#         pulses = Pulses(['pulse_A'])
#         pulses.variables[0]['E0'] = 0.5/au.VA
#         pulses.variables[0]['FWHM'] = 2.0/au.fs
#         pulses.variables[0]['omega'] = 1.6/au.eV
#         time = np.linspace(-20.0/au.fs,60.0/au.fs,4000)
#         ulmi = UltrafastLightMatterInteraction(medium,pulses,time,step_size='auto',options={'time_step_min':1e-3,'relative_error_tolerance':1e-8})
#         ulmi.gamma = 0.#1
#         result = ulmi.run_stepwise(gauge='vg',equation='lvn')
#         file_result = os.path.join(test_dir,'result_1d_vg_dephasing_convergence_pub9.hdf5')
#         result.save_data_to_file(file_result)
#
#
#     def test_dynamical_calculation_lg_1d_potential_weak(self,):
#         import numpy as np
#         from ulmic import Medium, Pulses, UltrafastLightMatterInteraction,AtomicUnits
#
#         file_name = os.path.join(test_dir,'data_1d_convergence_400k.hdf5')
#         au = AtomicUnits()
#         medium = Medium(file_name,'hdf5')
#         medium.calculate_second_neighbour_overlaps()
#         medium.set_max_band_index(3)
#         pulses = Pulses(['pulse_A'])
#         pulses.variables[0]['E0'] = 0.5/au.VA
#         pulses.variables[0]['FWHM'] = 2.0/au.fs
#         pulses.variables[0]['omega'] = 1.6/au.eV
#         time = np.linspace(-10.0/au.fs,60.0/au.fs,4000)
#         ulmi = UltrafastLightMatterInteraction(medium,pulses,time,step_size='auto',options={'time_step_min':1e-3,'relative_error_tolerance':1e-9})
#         ulmi.gamma = 0#.1
#         result = ulmi.run_stepwise(gauge='lg',equation='lvn')
#         file_result = os.path.join(test_dir,'result_1d_lg_dephasing_convergence_pub9.hdf5')
#         result.save_data_to_file(file_result)
#
#
# class TestsPublicationWeak():
#     def test_dynamical_calculation_vg_1d_potential_weak(self,):
#         import numpy as np
#         from ulmic import Medium, Pulses, UltrafastLightMatterInteraction,AtomicUnits
#
#         file_name = os.path.join(test_dir,'data_1d_convergence_200k.hdf5')
#         au = AtomicUnits()
#         medium = Medium(file_name,'hdf5')
#         medium.calculate_second_neighbour_overlaps()
#         medium.set_max_band_index(8)
#         pulses = Pulses(['pulse_A'])
#         pulses.variables[0]['E0'] = 0.1/au.VA
#         pulses.variables[0]['FWHM'] = 2.0/au.fs
#         pulses.variables[0]['omega'] = 1.6/au.eV
#         time = np.linspace(-10.0/au.fs,60.0/au.fs,4000)
#         ulmi = UltrafastLightMatterInteraction(medium,pulses,time,step_size='auto',options={'time_step_min':1e-2,'relative_error_tolerance':1e-6})
#         ulmi.gamma = 0.1
#         result = ulmi.run_stepwise(gauge='vg',equation='lvn')
#         file_result = os.path.join(test_dir,'result_1d_vg_dephasing_convergence_pub2.hdf5')
#         result.save_data_to_file(file_result)
#
#
#     def test_dynamical_calculation_lg_1d_potential_weak(self,):
#         import numpy as np
#         from ulmic import Medium, Pulses, UltrafastLightMatterInteraction,AtomicUnits
#
#         file_name = os.path.join(test_dir,'data_1d_convergence_200k.hdf5')
#         au = AtomicUnits()
#         medium = Medium(file_name,'hdf5')
#         medium.calculate_second_neighbour_overlaps()
#         medium.set_max_band_index(2)
#         pulses = Pulses(['pulse_A'])
#         pulses.variables[0]['E0'] = 0.1/au.VA
#         pulses.variables[0]['FWHM'] = 2.0/au.fs
#         pulses.variables[0]['omega'] = 1.6/au.eV
#         time = np.linspace(-10.0/au.fs,60.0/au.fs,4000)
#         ulmi = UltrafastLightMatterInteraction(medium,pulses,time,step_size='auto',options={'time_step_min':5e-3,'relative_error_tolerance':1e-8})
#         ulmi.gamma = 0.1
#         result = ulmi.run_stepwise(gauge='lg',equation='lvn')
#         file_result = os.path.join(test_dir,'result_1d_lg_dephasing_convergence_pub2.hdf5')
#         result.save_data_to_file(file_result)
