import os
import numpy as np
from ulmic.atomic_units import AtomicUnits
from ulmic.pulses.properties import PulseProperties
from ulmic.pulses.interpolated import InterpolatedPulse
from ulmic.pulses.analytical import AnalyticalPulse

au = AtomicUnits()
pi = np.pi
eV_au = 1/27.211385
fs_au = 1/0.02418884326505
nm_au = 1/0.052917721
VA_au = 1/51.4220652
nm_Eph = 2*pi*137.035999/(1/0.052917721)
c_SI = 299792458

class Pulses(PulseProperties):

    def __init__(self, pulses, data_type=None, E_max=None, polarisation_vector=None, *args, **kwargs):
        """ Class for evaluating electric fields.

            pulses:         list containing strings (names of supported pulses or input files)
                            or dicts (pulse containing variables or arguments when reading input files).
            data_type:      'A' or 'E'. Short-cut for setting all input files to be
                            read as vector potential or electric field.
            file_E_max:     float. Short-cut for setting all input files to have
                            designated peak field strength.
            polarisation:   list or numpy array. Force all pulses to have designated polarisation vector.
        """

        analytical_pulse_names = ['lin_IR', 'circ_IR', 'probe_UV', 'pulse_A', 'pulse_B']
        self.pulse_list = []

        for pulse in pulses:
            if type(pulse) == type(str()):
                if pulse in analytical_pulse_names:
                    self.pulse_list.append(AnalyticalPulse(pulse))
                elif os.path.isfile(pulse):
                    interpolated_pulse_kwargs = {}
                    if data_type is not None:
                        interpolated_pulse_kwargs['data_type'] = data_type
                    if E_max is not None:
                        interpolated_pulse_kwargs['E_max'] = E_max
                    if polarisation_vector is not None:
                        interpolated_pulse_kwargs['polarisation_vector'] = polarisation_vector
                    self.pulse_list.append(InterpolatedPulse(pulse,**interpolated_pulse_kwargs))
                else:
                    raise(ValueError,'String {} not recognized as a valid pulse name or file is not found.'.format(pulse))
            elif type(pulse) == type(dict()):
                if all (key in pulse for key in ['E0','omega','envelope','FWHM','polarisation_vector','delay','cep']):
                    self.pulse_list.append(AnalyticalPulse(pulse))
                elif all (key in pulse for key in ['file','data_type','E0','polarisation_vector']):
                    self.pulse_list.append(InterpolatedPulse(**pulse))
            else:
                raise(KeyError,'Pulse argument not understood. Must be list of strings or list of dictionaries.')
        self.variables = [pulse.variables for pulse in self.pulse_list]
