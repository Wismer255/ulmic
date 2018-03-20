import h5py
import numpy as np
import datetime

class Hdf5Interface:

    def __init__(self,**kwargs):

        self.variables = {'energy':None,
                          'momentum':None,
                          'overlap':None,
                          'nk':None,
                          'nb':None,
                          'lattice_vectors':None,
                          'reciprocal_vectors':None,
                          'valence_bands':None,
                          'size':None,
                          'klist1d':None,
                          'klist3d':None,
                          'neighbour_table':None,
                          'spin_factor':2,
                          'crystal_structure':None,
                          'datetime':str(datetime.datetime.now()),
                          'description':''
                         }

        for kwarg in kwargs:
            if kwarg in self.variables:
                self.variables[kwarg] = kwargs[kwarg]
            else:
                raise('Keyword %s not recognized' %kwarg)

    def check_variables(self,):

        if self.variables['nk'] is None or self.variables['nb'] is None:
            self.variables['nk'],self.variables['nb'] = self.variables['energy'].shape

        if self.variables['reciprocal_vectors'] is None:
            self.variables['reciprocal_vectors'] = Hdf5Interface.reciprocal_vectors(self.variables['lattice_vectors'])

        if self.variables['klist3d'] is None:
            self.variables['klist3d'] = Hdf5Interface.klist3d_table(self.variables['klist1d'], self.variables['size'])

        if self.variables['neighbour_table'] is None:
            self.variables['neighbour_table'] = Hdf5Interface.nearest_neighbour_table(self.variables['klist3d'], 1)

        if self.variables['crystal_structure'] is None:
            del self.variables['crystal_structure']

        for key in self.variables:
            value = self.variables[key]
            if isinstance(value, np.ndarray):
                if np.any(np.isnan(value)):
                    raise ValueError('NaN value in '+key)

            if value is None:
                if key in ['momentum', 'overlap']:
                    print('Warning: %s is missing' %s)
                    del self.variables[key]
                else:
                    raise ValueError('%s is None' %key)

    def save(self, output_file=None, open_file_as='w', check_variables=True, readable_by_octave=False):

        if check_variables:
            self.check_variables()

        if readable_by_octave:
            for key in self.variables:
                value = self.variables[key]
                if isinstance(value, str):
                    del self.variables[key]

        hdf5 = h5py.File(output_file, open_file_as)
        for key in self.variables:
            value = self.variables[key]
            hdf5.create_dataset(key, data=value)

    @staticmethod
    def reciprocal_vectors(lattice_vectors):
        reciprocal_lattice = 2*np.pi*np.linalg.inv(lattice_vectors).T
        return reciprocal_lattice

    @staticmethod
    def klist3d_table(klist1d,size):
        klist3d = np.zeros(size,int)
        for i in range(len(klist1d)):
            i1, i2, i3 = [int(q) for q in np.rint(size*klist1d[i, :])]
            klist3d[i1, i2, i3] = i
        return klist3d

    @staticmethod
    def nearest_neighbour_table(klist3d, nn):
        """ Generate table of nn nearest neighbors. """
        nk = len(klist3d.flatten())
        nn_table = np.zeros((nk, 3, 2 * nn), int)
        size = klist3d.shape
        for ix in range(size[0]):
            for iy in range(size[1]):
                for iz in range(size[2]):
                    i = klist3d[ix, iy, iz]
                    for j in range(-nn, nn):
                        if j < 0:
                            nn_table[i, 0, j] = klist3d[(ix + j) % size[0], iy, iz]
                            nn_table[i, 1, j] = klist3d[ix, (iy + j) % size[1], iz]
                            nn_table[i, 2, j] = klist3d[ix, iy, (iz + j) % size[2]]
                        if j >= 0:
                            nn_table[i, 0, j] = klist3d[(ix + j + 1) % size[0], iy, iz]
                            nn_table[i, 1, j] = klist3d[ix, (iy + j + 1) % size[1], iz]
                            nn_table[i, 2, j] = klist3d[ix, iy, (iz + j + 1) % size[2]]
        return nn_table
