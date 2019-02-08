import os
import numpy as np
import h5py
from ulmic.atomic_units import AtomicUnits
import logging
from ulmic.environment import UlmicEnvironment as ue
try:
    from mpi4py import MPI
    MPI4PY_INSTALLED = True
except:
    MPI4PY_INSTALLED = False


class LoadHdf5:

    def __init__(self, input_file,
                 k_points=None, buffer_width=(0,0,0),
                 read_now=True, band_max=None,
                 read_momentum=True, read_overlap=False,
                 logger=None,
                 **kwargs):
        """
            Class responsible for loading data from hdf5-files.
            Causes the specified k-points to be loaded.
            [Should only change if hdf5-format changes]
            input_file:     hdf5 input File
            k_points:       None for all k-points, int for single k-point,
                            ndarray for list for set of k-points,
                            (size,rank) tuple for MPI with automatic partioning,
                            (n,m,l,rank) tuple for MPI with custom partitioning,
            buffer_width:       three-dimensional tuple with number of buffer points
                            in each direction. Positive number for buffer in one
                            direction, or negative number for buffer in both
                            directions.
        """

        """ HDF5 Data format:
            nk:         number of k-points
            nb:         number of bands
            nv:         number of valence band electrons
            size:       number of k-points along each reciprocal axis (Monkhorst-Pack w/ Gamma point)
            spin_factor:number of electrons per band
            klist1d:    (nk,3) array of reduced coordinates of the k-points
            kllist3d:   (nk1,nk2,nk3) array of k-point indices with (0,0,0) being the Gamma point
            lattice_vectors:    (3,3) column vectors of the unit cell in Cartesian coordinates
            reciprocal_vectors: (3,3) column vectors
            energy:     (nk,nb)
            momentum:   (nk,nb,nb,3)
            overlap:    (nk,3,2,nb,nb)
            neighbour_table: (nk,3,2)

        """

        self.logger = logger or logging.getLogger(__name__)
        self.input_file = input_file
        self.buffer_width = buffer_width
        self.au = AtomicUnits()

        if os.path.isfile(input_file):
            hdf5_data = h5py.File(input_file, 'r')
        elif os.path.isfile(os.path.join(ue.get_data_dir(),input_file)):
            hdf5_data = h5py.File(os.path.join(ue.get_data_dir(),input_file), 'r')
        else:
            raise RuntimeError('File {} not found'.format(input_file))
        self.klist1d = hdf5_data['klist1d'][()]
        self.klist3d = hdf5_data['klist3d'][()]
        self.lattice_vectors = hdf5_data['lattice_vectors'][()]
        self.reciprocal_vectors = hdf5_data['reciprocal_vectors'][()]
        self.size = hdf5_data['size'][()]
        self.spin_factor = hdf5_data['spin_factor'][()]
        self.nv = hdf5_data['valence_bands'][()]
        if read_overlap:
            self.neighbour_table = hdf5_data['neighbour_table'][()]
        self.volume = np.abs(np.dot(self.lattice_vectors[:,0],
                                 np.cross(self.lattice_vectors[:,1],self.lattice_vectors[:,2])))
        self.check_input_data()
        self.nk1d = len(self.klist1d)
        self.nk = len(self.klist1d)
        self.nk_vol,self.nb = hdf5_data['energy'].shape
        self.unique_points_no_buffer = list(set(list(self.klist3d.flatten())))

        self.k_inner = self.generate_k_slice(k_points)
        self.k_buffer = self.generate_k_buffer(self.k_inner, buffer_width)
        self.k_slice_buffer = np.concatenate((self.k_inner, self.k_buffer))
        self.local_to_global = np.copy(self.k_slice_buffer)
        self.global_to_local = {}
        for i in range(len(self.k_slice_buffer)):
            self.global_to_local[str(self.k_slice_buffer[i])] = i

        self.nk_eval = len(self.k_inner)
        self.nk_buffer = len(self.k_buffer)
        self.nk_local = len(self.k_slice_buffer)

        if band_max is None:
            band_max = self.nb
        else:
            band_max = min(self.nb, band_max)
        self.nb = band_max
        self.band_slice = np.arange(self.nb, dtype=np.intp)

        if read_now:
            self.read(read_momentum,read_overlap)

        self.k_gamma = None
        for i in range(self.nk):
            if np.allclose(self.klist1d[i],0.0):
                self.k_gamma = i
        logging.info('{} successfully loaded.'.format(self.input_file))

    def check_input_data(self):
        np.testing.assert_allclose(np.dot(self.reciprocal_vectors.T,self.lattice_vectors),
                                          2*np.pi*np.eye(3),atol=1e-4,rtol=1e-4)
        assert((self.size == self.klist3d.shape).all())
        assert(len(self.klist1d) == np.prod(self.size))
        assert(self.klist1d.shape[1] == 3)
        i0 = self.klist3d[0,0,0]
        for i in range(len(self.klist1d)):
            n1,n2,n3 = np.rint(self.size * (self.klist1d[i] - self.klist1d[i0])).astype(np.intp)
            assert(i == self.klist3d[n1,n2,n3])

    def read(self, read_momentum=True, read_overlap=True):
        self.load_energy(self.k_slice_buffer, self.band_slice)
        if read_momentum:
            self.load_momentum(self.k_slice_buffer, self.band_slice)
        if read_overlap:
            self.load_overlap(self.k_slice_buffer, self.band_slice)

    def generate_k_slice(self, k_points):
        if k_points is None:
            k_slice = np.arange(self.nk1d)

        elif type(k_points) == type(int()):
            k_slice = np.array([k_points])

        elif type(k_points) == type(list()):
            k_slice = np.array(k_points)

        elif type(k_points) == type(tuple()):
            if len(k_points) == 2:
                pass

            elif len(k_points) == 4:
                partitions_x, partitions_y, partitions_z = k_points[:3]
                assert(self.klist3d.shape[0] % partitions_x == 0)
                assert(self.klist3d.shape[1] % partitions_y == 0)
                assert(self.klist3d.shape[2] % partitions_z == 0)
                size_x = self.klist3d.shape[0]//partitions_x
                size_y = self.klist3d.shape[1]//partitions_y
                size_z = self.klist3d.shape[2]//partitions_z
                self.size_mpi = (size_x,size_y,size_z)
                rank = k_points[3]
                ix = rank%partitions_x
                iy = (rank//partitions_x)%partitions_y
                iz = rank//(partitions_x*partitions_y)
                k_slice = self.klist3d[ix*size_x:(ix+1)*size_x,
                                       iy*size_y:(iy+1)*size_y,
                                       iz*size_z:(iz+1)*size_z].flatten()
        return k_slice.astype(np.intp)

    def generate_k_buffer(self, k_slice, k_buffer_tuple):
        ''' Generate buffer of k-points around the edge if
            the the k-grid is partitioned. '''
        k_buffer_points = []
        buffer_lists = []
        for i in range(3):
            if k_buffer_tuple[i] > 0:
                buffer_directional = np.arange(1,k_buffer_tuple[i]+1)
            elif k_buffer_tuple[i] < 0:
                buffer_directional = np.concatenate((
                                            np.arange(k_buffer_tuple[i],0),
                                            np.arange(1,np.abs(k_buffer_tuple[i])+1)))
            else:
                buffer_directional = []
            buffer_lists.append(buffer_directional)

        for i in k_slice:
            for j in range(3):
                for k in buffer_lists[j]:
                    coordinates = self.klist1d_int[i]
                    coordinates[j] += k
                    coordinates[j] = coordinates[j]%self.size[j]
                    k_point = self.klist3d[coordinates[0],
                                           coordinates[1],
                                           coordinates[2]]
                    if k_point not in k_slice:
                        k_buffer_points.append(k_point)
        return np.array(k_buffer_points).astype(np.intp)

    def load_energy(self, slice_k, slice_band):
        hdf5_data = h5py.File(self.input_file, 'r')
        self.energy = np.zeros((len(slice_k),len(slice_band)))
        for i in range(len(slice_k)):
            self.energy[i] = hdf5_data['energy'][slice_k[i]][slice_band]

    def load_momentum(self, slice_k, slice_band):
        hdf5_data = h5py.File(self.input_file, 'r')
        if 'momentum' in hdf5_data:
            self.momentum = np.zeros((len(slice_k),len(slice_band),len(slice_band),3),np.complex128)
            slice_band = np.array(slice_band) # just to be sure
            for i in range(len(slice_k)):
                P = hdf5_data['momentum'][slice_k[i]].astype(np.complex128)
                self.momentum[i] = P[slice_band[:,None], slice_band, :]
        else:
            logging.warning('Momentum matrix not present in {}. '
                            'Set read_momentum=False to disable '
                            'this warning'.format(self.input_file))

    def load_overlap(self, slice_k, slice_band):
        hdf5_data = h5py.File(self.input_file, 'r')
        if 'overlap' in hdf5_data:
            self.overlap = np.zeros((len(slice_k),3,2,len(slice_band),len(slice_band)),np.complex128)
            slice_band = np.array(slice_band) # just to be sure
            for i in range(len(slice_k)):
                O = hdf5_data['overlap'][slice_k[i]]
                self.overlap[i] = O[:, :, slice_band[:,None], slice_band]
        else:
            logging.warning('Overlap matrix not present in {}. '
                            'Set read_overlap=False to disable '
                            'this warning'.format(self.input_file))
