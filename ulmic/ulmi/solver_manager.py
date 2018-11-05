import numpy as np
import time as time_module
from ulmic.inputs import flags as default_flags
from ulmic.inputs import options as default_options


class SolverManager:

    def __init__(self,time_in,pulses):

        self.time_in = time_in
        self.time_out = time_in
        self.counter = 0
        self.total_number_of_steps = 0
        self.nt_out = len(self.time_out)
        self.default_dt = np.mean(np.diff(time_in))

        self.flags = dict(default_flags)
        self.options = dict(default_options)
        self.pulses = pulses

    def set_managers(self, parallel_manager=None):
        self.parallel = parallel_manager

    def init(self):
        self.nk_range_eval = self.parallel.get_eval_k_mesh()
        self.time_progression = self.time_out[0]
        self.index_progression = 0
        self.set_field_directions()
        self.total_number_of_steps = 0

    def set_flags(self,*args):
        """ Pass flags (strings) or a list of flags (list of strings)"""
        for arg in args:
            if isinstance(arg,str):
                self._set_flag(arg)
            elif isinstance(arg,list) or isinstance(arg,tuple):
                self.set_flags(*tuple(arg))
            else:
                raise (ValueError('Flag %s not recognized '%arg))
        return

    def _set_flag(self,arg):
        if arg in self.flags:
            self.flags[arg] = True
        else:
            raise (ValueError('Flag %s not recognized' %arg) )

    def set_options(self,*args,**kwargs):
        """ Pass a dictionary with options or pass options as keywords """
        for arg in args:
            if isinstance(arg,dict):
                self.set_options(**arg)
            else:
                raise(ValueError,'Argument type not recognized')

        for kwarg in kwargs:
            if kwarg in self.options:
                self.options[kwarg] = kwargs[kwarg]
            else:
                raise(ValueError, 'unknown option: {}'.format(kwarg))

    def load_default_parameters(self,flags,options):

        if flags is not None:
            if type(flags) == type({}):
                for key in flags:
                    if key in default_flags:
                        default_flags[key] = flags[key]
                    else:
                        print('ulmi: Unrecognized flag')
            if type(flags) == type([]):
                for key in flags:
                    if key in default_flags:
                        default_flags[key] = True
                    else:
                        print('ulmi: Unrecognized flag')
        self.flags = default_flags

        if not options is None:
            for key in options:
                if key in default_options:
                    default_options[key] = options[key]
                else:
                    print('ulmi: Unrecognized option')
        self.options = default_options

        self.time_step_min = self.options['time_step_min']
        self.relative_error_tolerance = self.options['relative_error_tolerance']
        self.timestamp = int(time_module.time())
        self.tolerance_zero_field = self.options['tolerance_zero_field']
        self.tolerance_absolute_error = self.options['tolerance_absolute_error']


    def set_field_directions(self):
        directions = np.zeros(3,bool)
        for pulse in self.pulses.get_parameters():
            directions += pulse['polarisation_vector'].astype(bool)
        self.directions = np.where(directions)[0].astype(int)

    def track(self,time_now):
        if self.flags['--print-timestep']:
            print(time_now)

    def track_finish(self):
        print('Total number of steps: %d' %self.solver_manager.total_number_of_steps)

    def running(self):
        if self.index_progression < self.nt_out-1:
            self.time_progression += self.default_dt
            self.index_progression += 1
            return True

