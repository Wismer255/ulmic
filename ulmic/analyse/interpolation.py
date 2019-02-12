import numpy as np
from scipy import interpolate
from ulmic.medium import Medium

class ReciprocalSpaceInterpolator:
    """ This class interpolates periodic functions of crystal momentum. """

    def __init__(self, medium, data=None):
        """
        Parameters
        ----------
        medium : an object of the class ulmic.medium.Medium;
        data : an array where the first dimension has a length of len(klist1d);
            the array specifies a k-dependent quantities that will be interpolated.
        """
        self.klist1d = medium.klist1d
        self.klist3d = medium.klist3d
        self.lattice_vectors = medium.lattice_vectors
        # self.reciprocal_vectors = medium.reciprocal_vectors
        self.reset(data)

    def reset(self, data):
        """ Set a quantity to be interpolated.

        Parameters
        ----------
        data : an array where the first dimension has a length of len(klist1d);
            the array specifies a k-dependent quantities that will be interpolated.
        """
        self.extra_dimensions = tuple()
        if data is None:
            self.data = np.zeros_like(self.klist3d)
            self.Fdata = np.zeros_like(self.klist3d)
            self.data_is_real = True
            self.data_is_nonnegative = True
            self.max_data = 0
        else:
            self.data = data[self.klist3d[...], ...]
            self.Fdata = np.fft.ifftn(self.data, axes = (0,1,2))
            if len(data.shape) > 1:
                self.extra_dimensions = data.shape[1:]
            self.data_is_real = np.all(np.isreal(data))
            self.data_is_nonnegative = None
            if self.data_is_real:
                self.data_is_nonnegative = np.all(data.real >= 0)
            self.max_data = np.max(np.abs(data), axis=0)

    def interpolate_Cartesian3D(self, crystal_momenta, Fourier_weight=0.9):
        """ Return interpolated values at the given crystal momenta.

        This function returns interpolated values at crystal momenta
        specified by their 3D Cartesian coordinates.

        Parameters
        ----------
        crystal_momenta : (N_k, 3) the Cartesian coordinates of N_k crystal momenta.
        Fourier_weight : (float) a number between 0 and 1; if it is zero, then
            the linear interpolation is performed; if Fourier_weight==1, the Fourier
            interpolation is performed; it the value is in between, the function
            returns a weighted sum of the two methods.

        Returns
        -------
        An array of interpolated values; the first dimension of this 
        array has a length of N, while the other dimensions, if present,
        match those of the array passed to the "reset" method.
        """
        N_k = len(crystal_momenta)
        N1 = self.klist3d.shape[0]
        N2 = self.klist3d.shape[1]
        N3 = self.klist3d.shape[2]
        xi1 = np.sum(crystal_momenta * self.lattice_vectors[:, 0].reshape((1, 3)), axis=-1)
        xi2 = np.sum(crystal_momenta * self.lattice_vectors[:, 1].reshape((1, 3)), axis=-1)
        xi3 = np.sum(crystal_momenta * self.lattice_vectors[:, 2].reshape((1, 3)), axis=-1)
        i1_array = xi1 * N1 / (2 * np.pi)
        i2_array = xi2 * N2 / (2 * np.pi)
        i3_array = xi3 * N3 / (2 * np.pi)
        assert(Fourier_weight >= 0.0)
        assert(Fourier_weight <= 1.0)
        if Fourier_weight > 0.0:
            # Fourier interpolation
            result_Fourier = np.zeros((N_k,) + self.extra_dimensions, dtype=self.Fdata.dtype)
            dims = np.ones(len(result_Fourier.shape), dtype=np.intp)
            dims[0] = N_k
            for m1 in range(N1):
                mm1 = m1 if m1 < np.ceil(N1 / 2) else m1 - N1
                exp_arg1 = mm1 * i1_array / N1
                for m2 in range(N2):
                    mm2 = m2 if m2 < np.ceil(N2 / 2) else m2 - N2
                    exp_arg2 = mm2 * i2_array / N2
                    for m3 in range(N3):
                        mm3 = m3 if m3 < np.ceil(N3 / 2) else m3 - N3
                        exp_arg3 = mm3 * i3_array / N3
                        factor = np.exp(-2j * np.pi * (exp_arg1 + exp_arg2 + exp_arg3))
                        result_Fourier += factor.reshape(dims) * \
                            self.Fdata[m1, m2, m3, ...].reshape((1,) + self.extra_dimensions)
            if self.data_is_real:
                result_Fourier = result_Fourier.real
                if self.data_is_nonnegative:
                    result_Fourier[result_Fourier < 0] = 0
            result = result_Fourier
        if Fourier_weight < 1.0:
            # linear interpolation
            if len(self.extra_dimensions) == 0:
                result_linear = np.zeros((N_k, 1), dtype=self.data.dtype)
                N = 1
            else:
                N = np.prod(np.array(self.extra_dimensions))
                result_linear = np.zeros((N_k, N), dtype=self.data.dtype)
            X = (np.arange(N1+1), np.arange(N2+1), np.arange(N3+1))
            Y = np.zeros((N1+1, N2+1, N3+1, N))
            Y[:N1, :N2, :N3, :] = self.data.reshape((N1, N2, N3, N), order='C')
            # ensure periodicity
            Y[N1, :, :, :] = Y[0, :, :, :]
            Y[:, N2, :, :] = Y[:, 0, :, :]
            Y[:, :, N3, :] = Y[:, :, 0, :]
            # perform interpolation
            i1_array -= N1 * np.floor(i1_array / (N1 + 0.)) # python2-safe
            i2_array -= N2 * np.floor(i2_array / (N2 + 0.)) # python2-safe
            i3_array -= N3 * np.floor(i3_array / (N3 + 0.)) # python2-safe
            X_out = np.column_stack((i1_array, i2_array, i3_array))
            for n in range(N):
                result_linear[:, n] = interpolate.interpn(X, Y[:, :, :, n], X_out)
            result_linear = result_linear.reshape((N_k,) + self.extra_dimensions, order='C')
            if self.data_is_real:
                result_linear = result_linear.real
                if self.data_is_nonnegative:
                    result_linear[result_linear < 0] = 0
            result = result_linear
        if Fourier_weight > 0.0 and Fourier_weight < 1.0:
            Y = np.abs(result_linear)
            # use the linear interpolation if Y is below this level:
            lower_threshold = (1.0 - Fourier_weight)**2 * self.max_data
            # use the Fourier interpolation if Y is above this level:
            upper_threshold = (1.0 - Fourier_weight) * self.max_data
            assert(np.all(lower_threshold < upper_threshold))
            # combine the interpolation methods at each data point
            dims = np.array(Y.shape)
            dims[0] = 1
            lower_threshold = np.asarray(lower_threshold).reshape(dims)
            upper_threshold = np.asarray(upper_threshold).reshape(dims)
            weights = np.sin(np.pi/2.0 * (Y - lower_threshold) / \
                (upper_threshold - lower_threshold))**2
            weights[Y <= lower_threshold] = 0.0
            weights[Y >= upper_threshold] = 1.0
            result = weights * result_Fourier + (1 - weights) * result_linear
        return result.astype(self.data.dtype)


    def interpolate_reduced2D(self, xi1, xi2, projection_axis=2, Fourier_weight=0.9):
        """ Return interpolated projected data.

        This function projects the 3D data along one of the primitive
        reciprocal-space vectors and then interpolates the 2D data.
        If projection_axis=2, the function returns
        \\[ y(\\xi_1, \\xi_2) = \\int_{-1/2}^{1/2} d\\xi_3
            \mbox{data}\\left(\\mathbf{k}=\\sum_{i=1}^3 \\xi_i \\mathbf{b}_i \\right),
        \\]
        where $\\mathbf{b}_i$ are the primitive vectors of the reciprocal lattice.
        The coordinates of the crystal momentum in the basis of the $\\mathbf{b}_i$
        vectors can be expressed via the primitive vectors of the real-space lattice as
        $\\xi_i = (\\mathbf{a}_i \\mathbf{k}) / (2 \\pi)$.

        Parameters
        ----------
        xi1 : reduced coordinates along the first reciprocal-space dimension that
            remains after projection; the values in this array are supposed to vary
            between -0.5 and 0.5 (the data is assumed to be periodic with respect to xi1).
        xi2 : reduced coordinates along the second reciprocal-space dimension that
            remains after projection; the values in this array are supposed to vary
            between -0.5 and 0.5 (the data is assumed to be periodic with respect to xi2).
        projection_axis: (0, 1, or 2) the reciprocal-space dimension to project along;
            in the following, the coordinates in the two remaining dimensions are
            denoted by xi1 and xi2.
        Fourier_weight : a floating-point number between 0 and 1; if it is zero, then
            the linear interpolation is performed; if Fourier_weight==1, the Fourier
            interpolation is performed; it the value is in between, the function
            returns a weighted sum of the two methods.

        Returns
        -------
        An array of interpolated values on a regular grid; the first two dimensions of
        this array correspond to xi1 and xi2, respectively.
        """
        N_xi1 = len(xi1)
        N_xi2 = len(xi2)
        # check the parameters
        assert(projection_axis in {0, 1, 2})
        assert(Fourier_weight >= 0.0)
        assert(Fourier_weight <= 1.0)
        # project along the selected axis        
        data = np.mean(self.data, axis=projection_axis)
        N1 = data.shape[0]
        N2 = data.shape[1]
        # evaluate "floating-point indices" that relate xi1 and xi2 to the k-grid
        xi_origin = np.delete(self.klist1d[self.klist3d[0,0,0]], projection_axis)
        xi1_adapted = xi1 - xi_origin[0]
        xi1_adapted -= np.rint(xi1_adapted)
        xi1_adapted[xi1_adapted < 0] += 1
        i1_array = xi1_adapted * N1
        xi2_adapted = xi2 - xi_origin[1]
        xi2_adapted -= np.rint(xi2_adapted)
        xi2_adapted[xi2_adapted < 0] += 1
        i2_array = xi2_adapted * N2
        # interpolate
        if Fourier_weight > 0.0:
            # Fourier interpolation
            Fdata = np.fft.ifftn(data, axes = (0,1))
            result_Fourier = np.zeros((N_xi1, N_xi2) + self.extra_dimensions, \
                dtype=Fdata.dtype)
            dims = np.ones(len(result_Fourier.shape), dtype=np.intp)
            dims[0] = N_xi1
            dims[1] = N_xi2
            for m1 in range(N1):
                mm1 = m1 if m1 < np.ceil(N1 / 2) else m1 - N1
                exp_arg1 = np.reshape(mm1 * i1_array / N1, (N_xi1, 1))
                for m2 in range(N2):
                    mm2 = m2 if m2 < np.ceil(N2 / 2) else m2 - N2
                    exp_arg2 = np.reshape(mm2 * i2_array / N2, (1, N_xi2))
                    factor = np.exp(-2j * np.pi * (exp_arg1 + exp_arg2))
                    result_Fourier += factor.reshape(dims) * \
                        Fdata[m1, m2, ...].reshape((1,1) + self.extra_dimensions)
            if self.data_is_real:
                result_Fourier = result_Fourier.real
                if self.data_is_nonnegative:
                    result_Fourier[result_Fourier < 0] = 0
            result = result_Fourier
        if Fourier_weight < 1.0:
            # linear interpolation
            if len(self.extra_dimensions) == 0:
                result_linear = np.zeros((N_xi1, N_xi2, 1), dtype=data.dtype)
                N = 1
            else:
                N = np.prod(np.array(self.extra_dimensions))
                result_linear = np.zeros((N_xi1, N_xi2, N), dtype=data.dtype)
            X1 = np.arange(N1+1)
            X2 = np.arange(N2+1)
            Y = np.zeros((N1+1, N2+1, N))
            Y[:N1, :N2, :] = data.reshape((N1, N2, N), order='C')
            # ensure periodicity
            Y[N1, :, :] = Y[0, :, :]
            Y[:, N2, :] = Y[:, 0, :]
            # perform interpolation
            XX1, XX2 = np.meshgrid(i1_array, i2_array, indexing='ij')
            XX1 = XX1.flatten()
            XX2 = XX2.flatten()
            X_out = np.column_stack((XX1, XX2))
            for n in range(N):
                Y_interpolated = interpolate.interpn((X1, X2), Y[:, :, n], X_out)
                result_linear[:, :, n] = Y_interpolated.reshape((N_xi1, N_xi2))
            result_linear = result_linear.reshape((N_xi1, N_xi2) + self.extra_dimensions, order='C')
            if self.data_is_real:
                result_linear = result_linear.real
                if self.data_is_nonnegative:
                    result_linear[result_linear < 0] = 0
            result = result_linear
        if Fourier_weight > 0.0 and Fourier_weight < 1.0:
            Y = np.abs(result_linear)
            # use the linear interpolation if Y is below this level:
            lower_threshold = (1.0 - Fourier_weight)**2 * self.max_data
            # use the Fourier interpolation if Y is above this level:
            upper_threshold = (1.0 - Fourier_weight) * self.max_data
            assert(np.all(lower_threshold < upper_threshold))
            # combine the interpolation methods at each data point
            dims = np.array(Y.shape)
            dims[:2] = 1
            lower_threshold = np.asarray(lower_threshold).reshape(dims)
            upper_threshold = np.asarray(upper_threshold).reshape(dims)
            weights = np.sin(np.pi/2.0 * (Y - lower_threshold) / \
                (upper_threshold - lower_threshold))**2
            weights[Y <= lower_threshold] = 0.0
            weights[Y >= upper_threshold] = 1.0
            result = weights * result_Fourier + (1 - weights) * result_linear
        return result.astype(self.data.dtype)
