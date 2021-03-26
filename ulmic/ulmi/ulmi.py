import datetime
from ulmic.ulmi.observables_manager import ObservablesManager
from ulmic.ulmi.solver_manager import SolverManager
from ulmic.ulmi.parallel_manager import ParallelManager
from ulmic.ulmi.result_manager import ResultManager
from ulmic.ulmi.state_manager import StateManager
from ulmic.logs import Logs
import numpy as np
import sys

log = Logs('ulmi')


class UltrafastLightMatterInteraction:

    def __init__(self,medium,pulses,time,*args,**kwargs):

        self.medium = medium
        self.pulses = pulses
        self.time = time

        self.original_time = time
        self.nt = len(time)
        self.gamma = 0.0

        # Set default values:
        self.state_object = 'wave_functions'
        self.energy_independent_dephasing = False

        # Define managers
        self.solver_manager = SolverManager(time,pulses)
        self.parallel_manager = ParallelManager(medium)
        self.state_manager = StateManager(medium,pulses)
        self.observables_manager = ObservablesManager(medium,pulses)
        self.result_manager = ResultManager(medium)

        # Set manager dependencies
        self.solver_manager.set_managers(parallel_manager=self.parallel_manager)
        self.observables_manager.set_managers(solver_manager=self.solver_manager,
                                              state_manager=self.state_manager)
        self.state_manager.set_managers(solver_manager=self.solver_manager)
        self.result_manager.set_managers(observables_manager=self.observables_manager,
                                         solver_manager=self.solver_manager)

    def set_flags_and_options(self,*args, **kwargs):
        """ Pass flags (strings) or a list of flags (list of strings)
            Pass a dictionary with options or pass options as keywords """
        for arg in args:
            if isinstance(arg,list) or isinstance(arg,str):
                self.solver_manager.set_flags(arg)
            elif isinstance(arg, dict):
                self.solver_manager.set_options(arg)
            else:
                raise ValueError('Argument {} not recognized'.format(arg))
        self.solver_manager.set_options(**kwargs)

    def run(self,):
        # Initialize managers
        self.solver_manager.init()
        self.state_manager.init()
        self.observables_manager.init()

        if self.solver_manager.flags['--print-timestep']:
            datetime_initial = datetime.datetime.now()
            log.log('')
            log.log('Start calculation at {0}.... '.format(datetime_initial.isoformat()))

        while True:
            # because of poor code design, I have to adjust default_dt here
            i = self.solver_manager.index_progression
            self.solver_manager.default_dt = self.solver_manager.time_out[i+1] - \
                self.solver_manager.time_out[i]
            # propagate to the next point of time where observables need to be evaluated
            try:
                self.state_manager.propagate_solution()
            except np.linalg.LinAlgError as err:
                print(err)
            if self.solver_manager.flags['--print-timestep']:
                log.log('Counter={0}, Time={1:+.3f} abs_error = {2:+.2e}, rel_error={3:+.2e}'.format(self.solver_manager.division_counter,
                                                                                     self.solver_manager.time_progression,
                                                                                     self.state_manager.result_absolute_error.max(),
                                                                                     self.state_manager.result_relative_error.max()))
            # update the current time
            self.solver_manager.time_progression += self.default_dt
            self.solver_manager.index_progression += 1            
            # evaluate observables
            self.observables_manager.evaluate_observables(self.solver_manager.time_progression,
                                                          self.solver_manager.index_progression)
            # see if it's time to stop
            if self.index_progression >= self.nt_out - 1:
                break

        if self.solver_manager.flags['--print-timestep']:
            datetime_final = datetime.datetime.now()
            log.log('...')
            log.log('Calculation finished.')
            log.log('Number of steps: {0}'.format(self.solver_manager.total_number_of_steps))
            log.log('Duration:        {0}'.format(str(datetime_final-datetime_initial)))

        self.result_manager.post_process_observables()
        self.results = self.result_manager.get_results()
        return self.results

    def set_parameters(self,**kwargs):
        solver_parameters = []
        parallel_parameters = []
        state_parameters = ['gauge', 'equation', 'initial_state']
        for kwarg in kwargs:
            if kwarg in state_parameters:
                setattr(self.state_manager, kwarg, kwargs[kwarg])
            else:
                raise ValueError('Parameter {} not recognized!'.format(kwarg))
        # check the parameters
        if self.state_manager.gauge != "vg":
            for pulse in self.pulses:
                if pulse.variables['envelope'] == 'delta_spike':
                    print("ERROR: the delta_spike envelope is allowed only in the velocity gauge", file=sys.stderr)
