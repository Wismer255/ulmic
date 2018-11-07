import numpy as np
from scipy.integrate import cumtrapz
from ulmic.result import Result
import matplotlib.pyplot as plt

class ResultManager(object):

    def __init__(self, medium):
        self.medium = medium

    def set_managers(self, observables_manager=None, solver_manager=None):
        self.observables = observables_manager
        self.solver = solver_manager

    def post_process_observables(self,):
        self.ground_state_observables = {}
        self.initial_state_observables = {}

        self.scale_and_transfer(self.observables, self, self.medium.spin_factor)
        self.scale_and_transfer_dict(self.observables.ground_state_observables,
                                self.ground_state_observables, self.medium.spin_factor)
        self.scale_and_transfer_dict(self.observables.initial_state_observables,
                                self.initial_state_observables, self.medium.spin_factor)

    def get_results(self):
        # Currents and negative velocities
        current = self.adjust_offset('current')
        geometric_current = self.adjust_offset('geometric_current')
        if self.observables.state.gauge == 'vg':
            kinematic_current_primitive = current - self.primitive_electron_number
            kinematic_current_effective = current - self.effective_electron_number
            geometric_current_primitive = geometric_current - self.primitive_electron_number
            geometric_current_effective = geometric_current - self.effective_electron_number

        elif self.observables.state.gauge == 'lg':
            kinematic_current_primitive = current
            kinematic_current_effective = current
            geometric_current_primitive = geometric_current
            geometric_current_effective = geometric_current

        #Polarisations
        polarisation = self.adjust_offset('polarisation')
        geometric_polarisation = ( (4.0/3.0)*self.adjust_offset('geometric_polarisation_1st')
                                  -(1.0/6.0)*self.adjust_offset('geometric_polarisation_2nd'))
        integrated_current_primitive = cumtrapz((kinematic_current_primitive).T, self.solver.time_out, initial=0.0).T
        integrated_current_effective = cumtrapz((kinematic_current_effective).T, self.solver.time_out, initial=0.0).T
        integrated_geometric_current = cumtrapz((geometric_polarisation - self.primitive_electron_number).T, self.solver.time_out, initial=0.0).T

        #Absorbed energies
        absorbed_energy = self.adjust_offset('absorbed_energy')
        energy_lg = np.array([np.dot(self.observables.vector_potential[q], self.polarisation[q])
                              for q in range(self.solver.nt_out)])
        # self.energy_vg = np.array([np.dot(self.result_E[q],self.result_jkinematic2[q]) for q in range(self.nt_out)])
        energy_vg = np.array([np.dot(self.observables.vector_potential[q], self.current[q])
                              for q in range(self.solver.nt_out)])
        energy_A2 = self.medium.spin_factor * self.medium.nv * (1.0 / (self.medium.volume)) * \
                    np.array([0.5 * np.dot(self.observables.vector_potential[q], self.observables.vector_potential[q])
                    for q in range(self.solver.nt_out)])

        classical_work = cumtrapz([np.dot(self.observables.electric_field[q], kinematic_current_primitive[q])
                                   for q in range(self.solver.nt_out)], self.solver.time_out, initial=0)

        polarisation = integrated_current_effective
        result = Result(time_output=self.solver.time_out,
                        equation=self.observables.state.equation,
                        gauge=self.observables.state.gauge,
                        final_state=self.observables.state.state,
                        polarisation=polarisation,
                        canonical_momentum=-current,
                        kinematic_momentum=-kinematic_current_primitive,
                        integrated_current_primitive=integrated_current_primitive,
                        geometric_current=geometric_current,
                        geometric_current_primitive=geometric_current_primitive,
                        geometric_current_effective=geometric_current_effective,
                        integrated_geometric_current=geometric_current_primitive,
                        current=kinematic_current_effective,
                        energy_h0=absorbed_energy,
                        energy_lg=energy_lg,
                        energy_vg=energy_vg,
                        energy_A2=energy_A2,
                        classical_work=classical_work,
                        neff_simplest=self.primitive_electron_number,
                        neff=self.effective_electron_number,
                        geometric_polarisation=geometric_polarisation,
                        absolute_error=self.observables.state.result_absolute_error,
                        relative_error=self.observables.state.result_relative_error,
                        electric_field=self.observables.electric_field,
                        vector_potential = self.observables.vector_potential,
                        )
        return result

    def scale_and_transfer(self,origin, destination, scaling_factor):
        for observable in self.observables.electron_dependent_observables:
            value = getattr(origin, observable)
            value *= scaling_factor
            setattr(destination, observable, value)
            delattr(origin, observable)

    def scale_and_transfer_dict(self, origin, destination, scaling_factor):
        for observable in self.observables.electron_dependent_observables:
            value = origin[observable]
            value *= scaling_factor
            destination[observable] = value
            del origin[observable]

    def adjust_offset(self,observable):
        return getattr(self, observable) - self.ground_state_observables[observable]
