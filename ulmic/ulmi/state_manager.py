import warnings
import h5py
import numpy as np
from ulmic.ulmi.jit.time_propagator import jit_step_vg_lvn_dp45, jit_step_vg_wavefunctions_dp45
from ulmic.ulmi.jit.time_propagator import jit_step_lg_wavefunctions_dp45, jit_step_lg_lvn_dp45
from ulmic.ulmi.jit.time_propagator import jit_step_vg_wavefunctions_dp45_stdse, jit_step_lg_lvn_dp45_constant_dephasing
from ulmic.ulmi.jit.time_propagator import jit_step_vg_lvn_dp45_FAKE_DECOHERENCE
from ulmic.ulmi.extra.initial_state import InitialState
from ulmic.atomic_units import AtomicUnits

au = AtomicUnits()


class StateManager(InitialState):

    def __init__(self, medium, pulses):
        self.medium = medium
        self.pulses = pulses

        self.equation = 'tdse'
        self.gauge = 'vg'
        self.gamma = 0.0
        self.nv = medium.nv
        self.initial_state = 'ground_state'

    def get_time_independent_decoherence_operator(self):
        decoherence_exponents = np.zeros((self.medium.nk_local,self.medium.nb,self.medium.nb),dtype=np.complex128)
        for i in range(self.medium.nb):
            for j in range(self.medium.nb):
                for k in range(self.medium.nk_eval):
                    decoherence_exponents[k,i,j] = 0.5*self.gamma*(self.medium.energy[k,i]-self.medium.energy[k,j])**2

        self.average_energy_difference = 0.5*(np.min(self.medium.energy[:,self.nv] - self.medium.energy[:, self.nv-1])
                                            + np.max(self.medium.energy[:,self.nv] - self.medium.energy[:, self.nv-1]))
        return decoherence_exponents

    def set_managers(self,solver_manager=None):
        self.solver = solver_manager

    def init(self):
        self.result_absolute_error = np.zeros(self.solver.nt_out)
        self.result_relative_error = np.zeros(self.solver.nt_out)
        self.decoherence_exponents = self.get_time_independent_decoherence_operator()
        self.explicit_solver_scheme = 'dp45'
        if self.solver.flags['--constant-time-step']:
            # self.explicit_solver_scheme = 'rk4'
            self.force_propagation = True
        else:
            # self.explicit_solver_scheme = 'dp45'
            self.force_propagation = False

        if self.equation in ['tdse','stdse']:
            self.state_object = 'wave_functions'
        elif self.equation in ['lvn', 'me']:
            self.state_object = 'density_matrix'
        self.set_initial_state()

    def set_initial_state(self,):
        self.state = self.get_initial_state(self.initial_state)

    def get_initial_state(self,initial_state):
        if isinstance(self.initial_state, str):
            if self.initial_state.endswith('.hdf5'):
                final_state = h5py.File(self.initial_state)['final_state'][()]
                state = np.copy(final_state)
            else:
                state = self.get_initial_state_by_name()

        if isinstance(self.initial_state, np.ndarray):
            if self.state.shape == self.initial_state.shape:
                state = np.copy(self.initial_state)
            else:
                raise (ValueError,"Shape of input is inconsistent.")
        return state

    def propagate_solution(self,):
        """ Propagate over a time interval of self.solver.default_dt """
        self.time_now = self.solver.time_progression
        self.dt_now = self.solver.default_dt / (2 ** self.solver.counter)
        self.cumulative_dt = 0.0

        while True:
            self.solver.total_number_of_steps += 1
            if self.field_is_zero():
                tmp_state, tmp_abs_error, tmp_rel_error = self.field_free_propagation()

            elif self.gauge == 'lg':
                if self.equation == 'tdse':
                    tmp_state, tmp_abs_error, tmp_rel_error = self.solve_tdse_lg()
                elif self.equation == 'lvn':
                    tmp_state, tmp_abs_error, tmp_rel_error = self.solve_lvn_lg()

            elif self.gauge == 'vg':
                if self.equation == 'tdse':
                    tmp_state, tmp_abs_error, tmp_rel_error = self.solve_tdse_vg()
                elif self.equation == 'stdse':
                    tmp_state, tmp_abs_error, tmp_rel_error = self.solve_stdse_vg()
                elif self.equation == 'lvn':
                    tmp_state, tmp_abs_error, tmp_rel_error = self.solve_lvn_vg()

            if self.check_solution(tmp_state, tmp_abs_error, tmp_rel_error):
                if self.solver.default_dt - self.cumulative_dt < 0.75 * self.dt_now:
                    # since self.solver.default_dt is always an integer
                    # multiple of self.dt_now, the above condition means that
                    # the final substep has been successfully accomplished
                    break



    def field_is_zero(self,):
        if self.equation == 'stdse':
            return False
        electric_field = self.pulses.eval_field_fast(self.solver.time_progression)
        vector_potential = self.pulses.eval_potential_fast(self.solver.time_progression)
        abs_electric_field = np.sqrt(np.dot(electric_field, electric_field))
        abs_vector_potential = np.sqrt(np.dot(vector_potential, vector_potential))
        return (abs_electric_field < self.solver.options['tolerance_zero_field']
                and abs_vector_potential < self.solver.options['tolerance_zero_potential'])

    def field_free_propagation(self):
        dt_now = self.solver.default_dt
        if self.equation == 'tdse':
            tmp_state, tmp_abs_error, tmp_rel_error = self.state, 0.0, 0.0

        elif self.equation == 'stdse':
            raise ValueError('Field free propagation is disabled for sTDSE')

        elif self.equation in ['lvn', 'me']:
            tmp_state, tmp_abs_error, tmp_rel_error = self.state*np.exp(-dt_now * self.decoherence_exponents), 0.0, 0.0

        return tmp_state, tmp_abs_error, tmp_rel_error

    def get_electric_fields(self):
        time_now = self.solver.time_progression
        dt_now = self.dt_now
        cumulative_dt = self.cumulative_dt
        coeff = self.get_explicit_solver_coefficients()
        n_coeff = len(coeff)

        electric_fields = np.zeros((n_coeff, 3))
        for i in range(n_coeff):
            electric_fields[i, :] = self.pulses.eval_field_fast(time_now + cumulative_dt + coeff[i]*dt_now)
        return electric_fields

    def get_vector_potential(self):
        time_now = self.solver.time_progression
        dt_now = self.dt_now
        cumulative_dt = self.cumulative_dt
        coeff = self.get_explicit_solver_coefficients()
        n_coeff = len(coeff)

        vector_potential = np.zeros((n_coeff, 3))
        for i in range(n_coeff):
            vector_potential[i, :] = self.pulses.eval_potential_fast(time_now + cumulative_dt + coeff[i]*dt_now)
        return vector_potential


    def get_explicit_solver_coefficients(self):
        if self.explicit_solver_scheme == 'dp45':
            return [0.0, 0.2, 0.3, 0.8, 8.0/9.0, 1.0, 1.0]
        elif self.explicit_solver_scheme == 'rk4':
            return [0.0, 0.5, 1.0]
        else:
            raise ValueError('Explicit solver scheme not recognized!')


    def eval_courant_number(self):
        ''' Evaluate Courant number to determine maximum allowed time step '''
        omega_typical = 1.0 / au.eV
        reciprocal_dt_Courant = np.max(
            abs(self.medium.size * np.dot(self.medium.lattice_vectors, self.pulses.eval_field_fast(self.time_now))))
        reciprocal_dt_Courant_int = np.max(abs(self.medium.size * np.dot(self.medium.lattice_vectors,
                                                                         omega_typical * self.pulses.eval_potential_fast(
                                                                             self.time_now))))
        reciprocal_dt_Courant = max(reciprocal_dt_Courant, reciprocal_dt_Courant_int)
        if reciprocal_dt_Courant > 1e-8:
            minimum_counter_exponent = max(self.counter, np.ceil(np.log2(self.default_dt * reciprocal_dt_Courant)))
        return minimum_counter_exponent

    def solve_tdse_lg(self):
        electric_fields = self.get_electric_fields()
        return jit_step_lg_wavefunctions_dp45(self.medium.nk_local,
                                              self.time_now + self.cumulative_dt, self.dt_now,
                                              self.medium.energy, self.medium.overlap,
                                              self.medium.neighbour_table,
                                              self.medium.size, self.state,
                                              electric_fields, self.solver.directions,
                                              self.medium.lattice_vectors)

    def solve_lvn_lg(self):
        electric_fields = self.get_electric_fields()
        if self.solver.flags['--energy-independent-decoherence']:
            return jit_step_lg_lvn_dp45_constant_dephasing(self.medium.nk_local,
                                                           self.time_now + self.cumulative_dt,
                                                           self.dt_now, self.medium.energy,
                                                           self.medium.overlap,
                                                           self.medium.neighbour_table,
                                                           self.medium.size,
                                                           self.state, electric_fields,
                                                           self.gamma, self.solver.directions,
                                                           self.medium.lattice_vectors,
                                                           self.average_energy_difference)
        else:
            return jit_step_lg_lvn_dp45(self.medium.nk_local,
                                        self.time_now + self.cumulative_dt,
                                        self.dt_now, self.medium.energy,
                                        self.medium.overlap,
                                        self.medium.neighbour_table,
                                        self.medium.size,
                                        self.state, electric_fields,
                                        self.gamma, self.solver.directions,
                                        self.medium.lattice_vectors)

    def solve_tdse_vg(self):
        vector_potentials = self.get_vector_potential()
        if self.solver.flags['--energy-independent-decoherence']:
            jit_step_vg_lvn_dp45_FAKE_DECOHERENCE(self.medium.nk_local,
                                                  self.time_now + self.cumulative_dt,
                                                  self.dt_now,
                                                  self.medium.energy, self.medium.momentum,
                                                  self.state, vector_potentials,
                                                  self.gamma,
                                                  self.average_energy_difference)
        else:
            return jit_step_vg_wavefunctions_dp45(self.medium.nk_local,
                                                  self.time_now + self.cumulative_dt, self.dt_now,
                                                  self.medium.energy, self.medium.momentum, self.state,
                                                  vector_potentials)

    def solve_stdse_vg(self):
        vector_potentials = self.get_vector_potential()
        return jit_step_vg_wavefunctions_dp45_stdse(
                                                    self.medium.nk_local, self.time_now + self.cumulative_dt,
                                                    self.dt_now, self.medium.energy, self.medium.momentum,
                                                    self.state, vector_potentials, self.gamma)

    def solve_lvn_vg(self):
        vector_potentials = self.get_vector_potential()
        if self.solver.flags['--energy-independent-decoherence']:
            return jit_step_vg_lvn_dp45_FAKE_DECOHERENCE(self.medium.nk_local,
                                                         self.time_now + self.cumulative_dt, self.dt_now,
                                                         self.medium.energy, self.medium.momentum,
                                                         self.state, vector_potentials, self.gamma,
                                                         self.average_energy_difference)
        else:
            return jit_step_vg_lvn_dp45(self.medium.nk_local, self.time_now + self.cumulative_dt, self.dt_now,
                                        self.medium.energy,
                                        self.medium.momentum, self.state, vector_potentials, self.gamma)

    def check_solution(self, tmp_state, tmp_abs_error, tmp_rel_error):
        ''' Check if returned solution passes all criteria and update
            state attribute if it does '''
        index = self.solver.index_progression
        if tmp_state is None:
            raise ValueError('Combination of gauge=%s and equation=%s is invalid!' % (self.gauge, self.equation))

        is_accepted = (self.force_propagation
                        or tmp_rel_error < self.solver.options['tolerance_relative_error']
                        or tmp_abs_error < self.solver.options['tolerance_absolute_error'])

        if is_accepted:
            self.state = tmp_state
            self.cumulative_dt += self.dt_now
            if tmp_abs_error > self.result_absolute_error[index]:
                self.result_absolute_error[index] = tmp_abs_error
            if tmp_rel_error > self.result_relative_error[index]:
                self.result_relative_error[index] = tmp_rel_error
        else:
            self.solver.counter += 1
            self.dt_now = self.solver.default_dt / (2 ** self.solver.counter)
            if self.dt_now < self.solver.options['time_step_min']:
                if self.solver.flags['--dt-break'] and self.solver.flags['--dump-state']:
                    raise (ValueError('Time step is below dt_tolerance!'))
                self.force_propagation = True
                warnings.warn('Time step is below dt_tolerance! Forcing advancement')
        return is_accepted
