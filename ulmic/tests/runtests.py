import pytest
import numpy as np
from ulmic.environment import UlmicEnvironment
import matplotlib.pyplot as plt
from ulmic.atomic_units import AtomicUnits as au
from ulmic.tests.simulation_batch import SimulationBatch

file_medium = 'runtest.hdf5'
args = (file_medium, 'probe_UV', np.linspace(-400.0,400.0,800))
MAX_BAND = 3

class TestImports():

    @pytest.mark.first
    def test_local_potential(self):
        from ulmic.external.localpotential import LocalPotential1D
        medium = LocalPotential1D()
        medium.set_spatial_parameters(points_per_cell=200)
        medium.set_medium_parameters(bands=10,
                                     k_points=100,
                                     valence_bands=1,
                                     spin_factor=2)
        medium.save_as_medium_to_hdf5(file_medium)
        print(medium.check_real_space_eigenstate(0,0))
        print(medium.check_real_space_eigenstate(0,1))
        print(medium.check_real_space_eigenstate(0.2,0))

    def test_medium(self):
        from ulmic.medium import Medium
        medium = Medium(file_medium)
        medium.calculate_diagonal_momentum()
        rtol = 5e-2
        ptol = 5e-2
        np.testing.assert_allclose(medium.calculate_diagonal_momentum()[:,:MAX_BAND,:],
                                   medium.get_diagonal_momentum()[:,:MAX_BAND,:],
                                   rtol=rtol,
                                   atol=ptol*np.max(medium.calculate_diagonal_momentum()[:,:MAX_BAND,:]))

    def test_ulmi_susceptibility(self):
        batch = SimulationBatch(*args)
        batch.pulses.variables[0]['E0'] = 0.00001
        results = batch.get_results()
        for result in results:
            alpha = result.get_susceptibility_scalar()
            beta = result.compare_polarisations()
            gamma = result.calc_susceptibility_time_domain()

