import numpy as np
from ulmic import AtomicUnits as au
from ulmic.medium import Medium
from ulmic.pulses import Pulses
from ulmic.ulmi.ulmi import UltrafastLightMatterInteraction

class SimulationBatch():

    flags = []
    options = {}
    #flags.append('--constant-time-step')
    rtol = 5e-2
    ptol = 1e-2
    atol = 1e-13
    log = None

    def __init__(self, file_medium, pulse_type, time):
        self.medium = Medium(file_medium)
        self.pulses = Pulses([pulse_type])
        self.time = time
        self.pair_list = [('vg','tdse')]

    def run_all(self,):
        pair_list = self.get_pair_list()
        ulmi = UltrafastLightMatterInteraction(self.medium, self.pulses, self.time)
        ulmi.set_flags_and_options(self.flags)
        ulmi.set_flags_and_options(self.options)
        for gauge,equation in pair_list:
            ulmi.set_parameters(gauge = gauge, equation = equation)
            result = ulmi.run()
            alpha = result.get_susceptibility_scalar()
            beta = result.compare_polarisations()
            gamma = result.calc_susceptibility_time_domain()
            if self.log is not None:
                self.log('alpha = {}'.format(alpha))
                self.log('beta = {}'.format(beta))
                self.log('gamma = {}'.format(gamma))

    def compare_pairs(self, dict1, dict2):
        ulmi = UltrafastLightMatterInteraction(self.medium, self.pulses, self.time)
        ulmi.set_flags_and_options(self.flags)

        ulmi.set_parameters(**dict1)
        result1 = ulmi.run()

        ulmi.set_parameters(**dict2)
        result2 = ulmi.run()

        observables_to_test = ['polarisation']
        for observable in observables_to_test:
            observable1 = getattr(result1,observable)
            observable2 = getattr(result2,observable)
            np.testing.assert_allclose(observable1,observable2,rtol=self.rtol,atol=max(self.atol,
                                                                                       np.max(abs(observable1))*self.ptol))

    def check_susceptibility(self,):
        ulmi = UltrafastLightMatterInteraction(self.medium, self.pulses, self.time)
        ulmi.set_flags_and_options(self.flags)

        pairs = [('vg','tdse')]
        results = []
        self.medium.set_max_band_index(7)
        for pair in pairs:
            ulmi.set_parameters(gauge = pair[0], equation = pair[1])
            result = ulmi.run()
            results.append(result)
            alpha = result.get_susceptibility_scalar()
            beta = result.compare_polarisations()
            gamma = result.calc_susceptibility_time_domain()


    def get_results(self,):
        ulmi = UltrafastLightMatterInteraction(self.medium, self.pulses, self.time)
        ulmi.set_flags_and_options(self.flags)
        ulmi.set_flags_and_options(self.options)
        pairs = [('vg','tdse')]
        results = []
        self.medium.set_max_band_index(7)
        for pair in pairs:
            ulmi.set_parameters(gauge = pair[0], equation = pair[1])
            result = ulmi.run()
            results.append(result)
        return results

    def get_pair_list(self,gauges=['vg','lg'], equations=['tdse','lvn','stdse']):
        pair_list = []
        for gauge in gauges:
            for equation in equations:
                if equation == 'stdse':
                    if gauge == 'lg':
                        continue
                pair_list.append((gauge,equation))
        return pair_list

    def set_pair_list(self,gauges=['vg','lg'], equations=['tdse','lvn','stdse']):
        return self.get_pair_list(gauges=gauges, equations=equations)

