#Default values for flags, optional arguments and sample data files
#Flags are invoked as --flag and options as option==value when
#executing scripts from the terminal.
import sys
import pprint

flags = {
    '--verbose': False,
    '--help': False,
    '--no-vg-correction': False,
    '--no-berry': False,
    '--no-overlap': False,
    '--no-covariant': False,
    '--print-timestep': True,
    '--no-cache': False,
    '--dt-break': False,
    '--dump-state': False,
    '--energy-independent-decoherence': False,
}
    # '--use-mpi':False,

options = {
    'time_step': 'auto',
    'time_step_min': 1e-8,
    'tolerance_zero_field': 1e-16,
    'tolerance_zero_potential': 1e-16,
    'tolerance_relative_error': 1e-6,
    'tolerance_absolute_error': 1e-20,
}

for arg in sys.argv:
    if arg.startswith('--'):
        if arg in flags:
            flags[arg] = True
        else:
            print('ulmic: Unrecognized flag {}'.format(arg))

for arg in sys.argv:
    if '==' in arg:
        key,val = arg.split('==')
        if key in options:
            options[key] = list(map(type(options[key]),[val]))[0]
        else:
            print('ulmic: Unrecognized parameter {}'.format(arg))

if flags['--verbose']:
    print(flags)

if flags['--help']:
    print("Use '--flag' for flags and key==value for parameters.")
    print("")
    print("=== Flags available ===")
    pprint.pprint(flags,width=1)
    print("=== Options available ===")
    pprint.pprint(options,width=1)
    quit()


sample_data_files = {'quantum_espresso':'quantum_espresso_mos2.out',
                     'gpaw':'',
                     'wien2k':'',
                     'yambo':''}
