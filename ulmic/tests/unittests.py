import pytest
import numpy as np
from ulmic.tests.simulation_batch import SimulationBatch

file_medium = 'unittest.hdf5'
args = (file_medium, 'probe_UV', np.linspace(0.0001,0.0002,10))


class TestImports():

    @pytest.mark.first
    def test_local_potential(self):
        from ulmic.external.localpotential import LocalPotential1D
        medium = LocalPotential1D()
        medium.plot_potential()
        medium.plot_wave_function()
        medium.save_as_medium_to_hdf5(file_medium)
        print(medium.check_real_space_eigenstate(0,0))
        print(medium.check_real_space_eigenstate(0,1))
        print(medium.check_real_space_eigenstate(0.2,0))

    def test_pulses(self):
        from ulmic.pulses import Pulses
        pulses = Pulses(['probe_UV'])
        pulses.eval_field_fast(0.0)
        pulses.eval_potential_fast(0.0)

    def test_medium(self):
        from ulmic.medium import Medium
        medium = Medium(file_medium)

    def test_ulmi_zero_field(self):
        batch = SimulationBatch(*args)
        batch.pulses.variables[0]['E0'] = 0.0
        batch.run_all()

    def test_ulmi_constant_field(self):
        batch = SimulationBatch(*args)
        batch.pulses.variables[0]['E0'] = 0.00001
        batch.pulses.variables[0]['envelope'] = 'slope'
        batch.run_all()

    def test_ulmi_constant_potential(self):
        batch = SimulationBatch(*args)
        batch.pulses.variables[0]['E0'] = 0.00001
        batch.pulses.variables[0]['envelope'] = 'constant'
        batch.run_all()

    def test_comparison_constant_field(self):
        batch = SimulationBatch(*args)
        batch.pulses.variables[0]['E0'] = 0.00001
        batch.pulses.variables[0]['envelope'] = 'slope'
        dict1 = {'gauge':'vg', 'equation':'tdse'}
        dict2 = {'gauge':'vg', 'equation':'lvn'}
        batch.compare_pairs(dict1,dict2)
        batch.run_all()

        dict1 = {'gauge':'lg', 'equation':'tdse'}
        dict2 = {'gauge':'lg', 'equation':'lvn'}
        batch.compare_pairs(dict1,dict2)
        batch.run_all()

        dict1 = {'gauge':'vg', 'equation':'tdse'}
        dict2 = {'gauge':'lg', 'equation':'tdse'}
        batch.compare_pairs(dict1,dict2)
        batch.run_all()
