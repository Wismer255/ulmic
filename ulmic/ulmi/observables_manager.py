import numpy as np
from ulmic.ulmi.jit.evaluate_observables import jit_get_correction, evaluate_current_and_neff_using_berry, evaluate_angles_for_polarisation
from ulmic.ulmi.jit.evaluate_observables import evaluate_current_jit,evaluate_electrons_jit,evaluate_energy_jit, evaluate_acceleration_jit
from ulmic.ulmi.jit.evaluate_observables import evaluate_lvn_current_jit, evaluate_lvn_electrons_jit,evaluate_lvn_energy_jit, evaluate_lvn_acceleration_jit


class ObservablesManager(object):

    def __init__(self,medium,pulses):

        self.medium = medium
        self.pulses = pulses
        self.electron_dependent_observables = ['current', 'current_2nd', 'polarisation', 'polarisation_2nd',
                                               'acceleration_2nd', 'geometric_polarisation_1st', 'geometric_polarisation_2nd',
                                               'geometric_current',
                                                'electron_number', 'absorbed_energy',
                                               'conduction_band_populations', 'excitation_number',
                                               'primitive_electron_number', 'effective_electron_number']

    def set_managers(self,
                     solver_manager=None,
                     state_manager=None,
                     ):
        self.solver = solver_manager
        self.state = state_manager

    def init(self):
        self.initialize_arrays()
        self.forward_neighbour_table = self.get_forward_neighbour_table()
        self.eval_ground_state_observables()
        self.eval_initial_state_observables()


    def initialize_arrays(self, ):
        self.current = np.zeros((self.solver.nt_out, 3))
        self.current_2nd = np.zeros((self.solver.nt_out, 3))
        self.polarisation = np.zeros((self.solver.nt_out, 3))
        self.polarisation_2nd = np.zeros((self.solver.nt_out, 3))
        self.acceleration_2nd = np.zeros((self.solver.nt_out, 3))
        self.geometric_polarisation_1st = np.zeros((self.solver.nt_out, 3))
        self.geometric_polarisation_2nd = np.zeros((self.solver.nt_out, 3))
        self.geometric_current = np.zeros((self.solver.nt_out, 3))
        self.geometric_current_2nd = np.zeros((self.solver.nt_out, 3))
        self.excitation_number = np.zeros(self.solver.nt_out)
        self.electron_number = np.zeros(self.solver.nt_out)
        self.absorbed_energy = np.zeros(self.solver.nt_out)
        self.conduction_band_populations = np.zeros((self.solver.nt_out, self.medium.nb - self.medium.nv))

        self.electric_field = np.zeros((self.solver.nt_out, 3))
        self.vector_potential = np.zeros((self.solver.nt_out, 3))

        self.primitive_electron_number = np.zeros((self.solver.nt_out, 3))
        self.effective_electron_number = np.zeros((self.solver.nt_out, 3))


    def evaluate_observables(self, time_now, index):
        ''' Evaluate physical observables '''

        self.vector_potential[index] = self.pulses.eval_potential_fast(time_now)
        self.electric_field[index] = self.pulses.eval_field_fast(time_now)

        if self.state.equation in ['tdse', 'stdse']:
            self.eval_operator_quantities_wf()

        elif self.state.equation in ['lvn', 'me']:
            self.eval_operator_quantities_dm()

        if not self.solver.flags['--no-overlap']:
            if not self.solver.flags['--no-covariant']:
                self.geometric_current[index, :] = self.get_covariant_current()

            if not self.solver.flags['--no-berry']:
                self.get_geometric_phase()

        self.primitive_electron_number[index] = self.electron_number[index]*self.pulses.eval_potential_fast(time_now)
        if self.solver.flags['--no-vg-correction']:
          self.effective_electron_number[index] = self.primitive_electron_number[index]
        else:
          self.effective_electron_number[index] = self.get_effective_electron_number()

    def get_effective_electron_number(self):
        if not self.solver.flags['--no-vg-correction']:
            return jit_get_correction(self.solver.time_progression, self.state.state,
                                      self.pulses.eval_potential_fast(self.solver.time_progression),
                                      self.medium.energy, self.medium.momentum, self.medium.nk_vol,
                                      self.medium.volume, self.medium.nv, self.solver.nk_range_eval)

    def eval_operator_quantities_wf(self):
        time_now = self.solver.time_progression
        index = self.solver.index_progression
        evaluate_current_jit(self.state.state, index, self.medium.energy, self.medium.momentum, time_now,
                             self.solver.nk_range_eval, self.medium.nk_vol,
                             self.medium.volume, self.current)

        evaluate_acceleration_jit(self.state.state, index, self.medium.energy.astype(np.complex),
                                  self.medium.momentum, time_now, self.solver.nk_range_eval,
                                  self.medium.nk_vol, self.medium.volume, self.acceleration_2nd)

        evaluate_electrons_jit(self.state.state, self.medium.nb, self.medium.nv, index, self.medium.energy,
                               self.medium.momentum, time_now, self.solver.nk_range_eval, self.medium.nk_vol,
                               self.medium.volume, self.electron_number, self.excitation_number,
                               self.conduction_band_populations)

        evaluate_energy_jit(self.state.state, self.medium.nb, self.medium.nv, index, self.medium.energy,
                            self.medium.momentum, time_now, self.solver.nk_range_eval, self.medium.nk_vol,
                            self.medium.volume, self.absorbed_energy)

    def eval_operator_quantities_dm(self):
        time_now = self.solver.time_progression
        index = self.solver.index_progression
        evaluate_lvn_current_jit(self.state.state, index, self.medium.energy, self.medium.momentum, time_now,
                                 self.solver.nk_range_eval, self.medium.nk_vol, self.medium.volume, self.current)

        evaluate_lvn_electrons_jit(self.state.state, self.medium.nb, self.medium.nv, index, self.medium.energy,
                                   self.medium.momentum, time_now, self.solver.nk_range_eval,
                                   self.medium.nk_vol, self.medium.volume, self.electron_number,
                                   self.excitation_number, self.conduction_band_populations)

        evaluate_lvn_energy_jit(self.state.state, self.medium.nb, self.medium.nv, index, self.medium.energy,
                                self.medium.momentum, time_now, self.solver.nk_range_eval, self.medium.nk_vol,
                                self.medium.volume, self.absorbed_energy)
        evaluate_lvn_acceleration_jit(self.state.state, index, self.medium.energy.astype(np.complex),
                                      self.medium.momentum, time_now, self.solver.nk_range_eval,
                                      self.medium.nk_vol, self.medium.volume, self.acceleration_2nd)


    def get_covariant_current(self):
        time_now = self.solver.time_progression
        current_mixed_j1, current_mixed_neff1 = evaluate_current_and_neff_using_berry(self.state.state, time_now,
                                                                                      self.medium.momentum,
                                                                                      self.medium.overlap,
                                                                                      self.forward_neighbour_table,
                                                                                      self.medium.energy.astype(
                                                                                          np.complex128),
                                                                                      self.solver.nk_range_eval,
                                                                                      self.medium.lattice_vectors,
                                                                                      self.medium.size,
                                                                                      self.medium.klist3d,
                                                                                      self.medium.nk_vol,
                                                                                      self.medium.volume,
                                                                                      self.medium.nv, 1)
        current_mixed_j2, current_mixed_neff2 = evaluate_current_and_neff_using_berry(self.state.state, time_now,
                                                                                      self.medium.momentum,
                                                                                      self.medium.overlap,
                                                                                      self.forward_neighbour_table,
                                                                                      self.medium.energy.astype(
                                                                                          np.complex128),
                                                                                      self.solver.nk_range_eval,
                                                                                      self.medium.lattice_vectors,
                                                                                      self.medium.size,
                                                                                      self.medium.klist3d,
                                                                                      self.medium.nk_vol,
                                                                                      self.medium.volume,
                                                                                      self.medium.nv, 2)
        return (1 / (2 * np.pi * self.medium.volume)) * np.dot(self.medium.lattice_vectors,
                                    np.sum((4.0 / 3.0) * current_mixed_j1 - (1.0 / 6.0) * current_mixed_j2, axis=0)
                                    / np.array([ self.medium.size[1] *self.medium.size[2],
                                                 self.medium.size[0] *self.medium.size[2],
                                                 self.medium.size[0] *self.medium.size[1]]))

    def get_geometric_phase(self):
        time_now = self.solver.time_progression
        index = self.solver.index_progression
        current_products_k = evaluate_angles_for_polarisation(self.state.state, time_now, self.medium.overlap,
                                                              self.medium.neighbour_table, self.medium.energy,
                                                              self.solver.nk_range_eval,
                                                              self.medium.lattice_vectors,
                                                              self.medium.size, self.medium.klist3d,
                                                              self.medium.nk_vol, self.medium.volume,
                                                              self.medium.nv, 1)
        if index == 0:
            current_angles_k = np.log(current_products_k).imag
        else:
            current_angles_k = np.log((current_products_k) / self._previous_products_k).imag + self._previous_angles_k
        self.geometric_polarisation_1st[index, :] = (1 / (2 * np.pi * self.medium.volume)) * np.dot(self.medium.lattice_vectors,
                                                                                          np.sum(current_angles_k,
                                                                                                 axis=0) / np.array([
                                                                                              self.medium.size[
                                                                                                  1] *
                                                                                              self.medium.size[
                                                                                                  2],
                                                                                              self.medium.size[
                                                                                                  0] *
                                                                                              self.medium.size[
                                                                                                  2],
                                                                                              self.medium.size[
                                                                                                  0] *
                                                                                              self.medium.size[
                                                                                                  1]]))

        current_products_k2 = evaluate_angles_for_polarisation(self.state.state, time_now, self.medium.overlap,
                                                               self.medium.neighbour_table, self.medium.energy,
                                                               self.solver.nk_range_eval, self.medium.lattice_vectors,
                                                               self.medium.size, self.medium.klist3d,
                                                               self.medium.nk_vol, self.medium.volume, self.medium.nv, 2)
        if index == 0:
            current_angles_k2 = np.log(current_products_k2).imag
        if index > 0:
            current_angles_k2 = np.log((current_products_k2) / self._previous_products_k2).imag + self._previous_angles_k2
        self.geometric_polarisation_2nd[index, :] = (1 / (2 * np.pi * self.medium.volume)) * np.dot(self.medium.lattice_vectors,
                                                                                          np.sum(current_angles_k2,
                                                                                                 axis=0) / np.array([
                                                                                              self.medium.size[
                                                                                                  1] *
                                                                                              self.medium.size[
                                                                                                  2],
                                                                                              self.medium.size[
                                                                                                  0] *
                                                                                              self.medium.size[
                                                                                                  2],
                                                                                              self.medium.size[
                                                                                                  0] *
                                                                                              self.medium.size[
                                                                                                  1]]))
        self._previous_products_k = np.copy(current_products_k)
        self._previous_products_k2 = np.copy(current_products_k2)
        self._previous_angles_k = np.copy(current_angles_k)
        self._previous_angles_k2 = np.copy(current_angles_k2)




    def eval_ground_state_observables(self):
        """ Evaluate off-set for ground state;
            modifies both state and observables objects
            TODO: Refactor to not modify state object """
        self.state.state = self.state.get_initial_state('ground_state')
        self.evaluate_observables(self.solver.time_out[0],0)
        self.ground_state_observables = {}
        for observable in self.electron_dependent_observables:
            self.ground_state_observables[observable] = np.copy(getattr(self, observable)[0])

    def eval_initial_state_observables(self):
        """ Evaluate off-set for initial state;
            modifies both state and observables objects
            TODO: Refactor to not modify state object """
        self.state.state = self.state.get_initial_state(self.state.initial_state)
        self.evaluate_observables(self.solver.time_out[0],0)
        self.initial_state_observables = {}
        for observable in self.electron_dependent_observables:
            self.initial_state_observables[observable] = np.copy(getattr(self, observable)[0])


    def get_forward_neighbour_table(self,nn=2):
        medium = self.medium
        forward_neighbour_table = np.zeros((medium.nk_eval,3,nn), dtype=np.intp)
        assert(medium.nk_eval == len(medium.unique_points_no_buffer))
        for i in range(medium.nk_eval):
            for alpha in range(3):
                for j in range(nn):
                    i_global = medium.unique_points_no_buffer[i]
                    i_nn = medium.global_to_local[str( medium.neighbour_table[i_global,alpha,j] )]
                    forward_neighbour_table[i,alpha,j] = i_nn
                    assert(medium.neighbour_table[i_global,alpha,j] in medium.local_to_global)
        return forward_neighbour_table
