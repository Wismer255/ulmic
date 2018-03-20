from ulmic.ulmi.observables_manager import ObservablesManager
from ulmic.ulmi.solver_manager import SolverManager
from ulmic.ulmi.parallel_manager import ParallelManager
from ulmic.ulmi.result_manager import ResultManager
from ulmic.ulmi.state_manager import StateManager


class UltrafastLightMatterInteraction(object):

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
                raise ValueError('Argument %s not recgnized')
        self.solver_manager.set_options(**kwargs)

    def run(self,):
        # Initialize managers
        self.solver_manager.init()
        self.state_manager.init()
        self.observables_manager.init()

        while self.solver_manager.running():
            print(self.solver_manager.time_progression)
            self.observables_manager.evaluate_observables(self.solver_manager.time_progression,
                                                          self.solver_manager.index_progression)
            self.state_manager.propagate_solution()


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
                raise ValueError('Parameter %s not recognized!' %kwarg)
