import numpy as np
from scipy.interpolate import interpolate
import matplotlib.pyplot as plt
from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

ZERO = 1e-16

class VisualizeObject:

    def __init__(self,result):
        self.result = result

    def current(self,show_now=True):
        fig,ax = plt.subplots()
        ax.plot(self.result.time_output,self.result.current)
        if show_now:
            plt.show()


class Result:
    """ Class for containing all output data and simple data manipulation. """
    """ First column always refer to time. """

    def __init__(self,result_file='',**kwargs):

        if result_file.endswith('.hdf5'):
            self.load_data_from_file(result_file,format='hdf5')
        else:
            for kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    def add_array1d(self,*args):
        n = len(self.time_output)
        for arg in args:
            setattr(self, arg, np.zeros(n))

    def add_array3d(self,*args):
        n = len(self.time_output)
        for arg in args:
            setattr(self, arg, np.zeros((n,3)))

    def add_array1d_k(self,nk,*args):
        n = len(self.time_output)
        for arg in args:
            setattr(self, arg, np.zeros((n,nk)))

    def add_array3d_k(self,nk,*args):
        n = len(self.time_output)
        for arg in args:
            setattr(self, arg, np.zeros((n,nk,3)))


    def get_freq(self,nt_crop=0):
        std = np.diff(self.time_output).std()
        if std < 1e-12:
            nt = len(self.time_output)
            dt = self.time_output[1]-self.time_output[0]
            return 2*np.pi*np.fft.fftfreq(nt-nt_crop,dt)
        else:
            print('std = %e' %(std))
            raise(ValueError)

    def interpolate_values_to_uniform_grid(self,t_min=None):
        time_old = np.copy(self.time_output)
        if t_min is None:
            t_min = np.min(np.diff(time_old))
        self.time_output = np.arange(time_old[0],time_old[-1],t_min)

        for attr, value in self.__dict__.iteritems():
            if attr is not 'time_output' and type(attr) is type(self.time_output):
                if value.shape[0] == len(time_old):
                    f = interpolate.interp1d(time_old, value,axis=0)
                    setattr(self, attr, f(self.time_output))


    def store_fields(self,pulses):
        self.electric_field = np.zeros((len(self.time_output),3))
        self.vector_potential = np.zeros((len(self.time_output),3))
        for i in range(len(self.time_output)):
            time = self.time_output[i]
            self.electric_field[i,:] = pulses.eval_field_fast(time)
            self.vector_potential[i,:] = pulses.eval_potential_fast(time)


    def save(self,file_output,format='hdf5'):
        """ Save data in object as an hdf5 file. """
        self.save_data_to_file(file_output,format)


    def load(self,file_output,format='hdf5'):
        """ Load data from an hdf5 file. """
        self.load_data_from_file(file_output,format)


    def save_data_to_file(self,file_output,format='hdf5'):
        """ Keep function for backwards compatibility. """
        if format is 'hdf5':
            import h5py
            if comm_size > 1:
                file_output = file_output.replace('.hdf5','.p%d.hdf5' %(comm_rank))

            hdf5 = h5py.File(file_output, 'w')
            for attr, value in self.__dict__.iteritems():
                dset = hdf5.create_dataset(attr, data=value)


    def load_data_from_file(self,file_output,format='hdf5'):
        """ Keep function for backwards compatibility. """
        if format is 'hdf5':
            import h5py
            hdf5 = h5py.File(file_output, 'r')
            for dset in hdf5:
                setattr(self, dset, hdf5[dset][()])


    def calc_susceptibility_time_domain(self):

        if np.dot(self.polarisation.flatten(), self.polarisation.flatten()) > 1e-14:
            alpha = (np.dot(self.polarisation.flatten(),self.electric_field.flatten())
                    /np.dot(self.polarisation.flatten(),self.polarisation.flatten()))
        else:
            alpha = 0.0
        return alpha



    def get_proportionality_constant(self, A, B):
        array_A = getattr(self, A).flatten()
        array_B = getattr(self, B).flatten()

        if np.dot(array_A, array_A) > ZERO:
            scalar = (np.dot(array_B,array_A)/np.dot(array_A,array_A))
        else:
            scalar = 0.0
        return scalar

    def calc_deviation(self, A, B, scalar=1.0):
        array_A = getattr(self, A).flatten()
        array_B = getattr(self, B).flatten()
        denominator = np.sqrt(np.dot(array_A,array_A)*np.dot(array_B,array_B))
        if denominator < ZERO:
            return 0.0
        else:
            return np.sum((scalar*array_A-array_B)**2)/denominator

    def compare_arrays(self, A, B):
        scalar = self.get_proportionality_constant(A, B)
        deviation = self.calc_deviation(A, B, scalar)
        return (scalar, deviation)

    def compare_polarisations(self,):
        return self.compare_arrays('polarisation','geometric_polarisation')

    def get_susceptibility_scalar(self,):
        return self.compare_arrays('polarisation','electric_field')

    def plot(self,*args):
        time = getattr(self,'time_output')
        fig,ax = plt.subplots()
        for arg in args:
            if isinstance(arg,str):
                value = getattr(self, arg)
            if isinstance(arg,np.ndarray):
                value = arg

            ax.plot(time, value)
        plt.show()

    def get_scaled(self, A, B):
        scalar = self.get_proportionality_constant(A, B)
        return getattr(self, A)*scalar

