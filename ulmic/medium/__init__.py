from ulmic.medium.properties import MediumProperties

class Medium(MediumProperties):

    def __init__(self, input_file, **kwargs):
        """ Medium class used for interacting with
            the other modules in the package.
            [Should only change due to changes in
            MPI-logic, user-friendliness or when
            adding more equilibrium properties]
            input_file:     input hdf5 file

            Relevant keyword arguments:
            k_points:       None for all k-points, int for single k-point,
                            ndarray for list for set of k-points,
                            (size,rank) tuple for MPI with automatic partioning (TODO),
                            (n,m,l,rank) tuple for MPI with custom partitioning,
            buffer_width:   three-dimensional tuple with number of buffer points
                            in each direction. Positive number for buffer in one
                            direction, or negative number for buffer in both
                            directions.
            band_max:       int or None (default:None)
            read_now:       bool (default: True)
            read_momentum:  bool (default: True)
            read_overlap:   bool (default: True)

            fix_hermiticity:bool (default: True)
            fix_overlap:    bool (default: True)
            log:            bool (default: True)

        """
        super(Medium, self).__init__(input_file, **kwargs)
