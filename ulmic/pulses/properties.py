import numpy as np


class PulseProperties:


    def eval_field_fast(self,time):
        field = np.zeros(3)
        for pulse in self.pulse_list:
            field += pulse.eval_field_fast(time)
        return field

    def eval_potential_fast(self, time):
        potential = np.zeros(3)
        for pulse in self.pulse_list:
            potential += pulse.eval_potential_fast(time)
        return potential

    def eval_field(self, *args, **kwargs):
        field = np.zeros(3)
        for pulse in self.pulse_list:
            field += pulse.eval_field(*args, **kwargs)
        return field

    def eval_potential(self, *args, **kwargs):
        potential = np.zeros(3)
        for pulse in self.pulse_list:
            potential += pulse.eval_potential(*args, **kwargs)
        return potential

    def get_parameters(self):
        return self.variables

