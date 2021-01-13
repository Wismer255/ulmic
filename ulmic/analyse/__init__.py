import numpy as np
from ulmic.atomic_units import AtomicUnits
from ulmic.ulmi.jit.linear_response import jit_linear_interband_response
from ulmic.external.hdf5interface import Hdf5Interface
# from ulmic.result import Result
# from ulmic.medium import Medium

au = AtomicUnits()


class FinalStateAnalyzer:
    """ A class for analyzing the electronic state at the end of a simulation.

    CURRENTLY, THIS CLASS WORKS ONLY FOR TDSE SIMULATIONS. """

    def __init__(self, medium, result, copy_wavefunctions=False):
        self.medium = medium
        self.neighbour_table = None
        if copy_wavefunctions:
            self.psi = result.final_state.copy()
        else:
            self.psi = result.final_state
        self.rho_diagonal = np.sum(np.real(self.psi * self.psi.conj()), axis=-1) # (nk, nb)


    def delta_response(self, t0, A0, t_array,
            ignore_coherences=False, subtract_initial_state=False):
        """ Return the electric current density induced by A(t) = A_0 delta(t-t0).

        The function uses an analytical formula for evaluating the linear response.
        It's main purpose is to investiage the importance of interband coherences
        in the final state for the linear response of that state. The response
        returned by this function is not sufficient for evaluating the response to
        an arbitrary probe pulse. THE FUNCTION CURRENTLY WORKS ONLY FOR TDSE RESULTS.

        Parameters
        ----------
        t0 : scalar
            The moment when the delta-spike arrives (usually after the final time of the simulation).
        A0 : (3)
            The vector specifying the direction and the amplitude of the delta-spike.
        t_array : (nt)
            The times at which the response must be evaluated.
        ignore_coherences : bool
            If True, the function will neglect contributions from off-diagonal elements
            of the density matrix describing the final state.
        subtract_initial_state : bool
            If True, the function return the difference between the response of the final
            state and that of the ground state.

        Returns
        -------
        J_response : (nt, 3)
            The array will contain zeros for all times t < t0; for t >= t0, the array will
            contain the "non-instantaneous" part of the response to the delta spike of the
            vector potential.
        """
        nt = len(t_array)
        nb = self.medium.nb
        J_response = np.zeros([nt, 3])
        p_matrix = self.medium.momentum # (nk, nb, nb, 3)
        for n in range(nb):
            # if np.max(np.abs(self.psi[:, n, :])) < np.finfo(np.float).eps:
            #     continue
            if ignore_coherences:
                m_range = [n]
            else:
                m_range = list(range(nb))
            for m in m_range:
                rho_nm = np.sum(np.conj(self.psi[:, m, :]) * self.psi[:, n, :], axis=-1) # (nk,)
                if subtract_initial_state and n==m and n < self.medium.nv:
                    rho_nm -= 1.0
                # if np.max(rho_nm) < np.finfo(np.float).eps:
                #    continue
                omega_nm = self.medium.energy[:, n] - self.medium.energy[:, m]
                # rho_nm *= np.exp((-1j*omega_nm - decoherence_rate) * t0)
                rho_nm *= np.exp(-1j*omega_nm * t0)
                A0p_product = np.sum(p_matrix[:, :, n, :] * A0, axis=-1) # (nk, nb)
                omega_n1m = self.medium.energy - \
                    np.tile(self.medium.energy[:, m][:, np.newaxis], (1, nb)) # (nk, nb)
                for it in range(len(t_array)):
                    if t_array[it] < t0:
                        continue
                    # Z = A0p_product * np.exp((-1j*omega_n1m - decoherence_rate) * (t_array[it] - t0))
                    Z = A0p_product * np.exp(-1j*omega_n1m * (t_array[it] - t0))
                    Z = np.sum(p_matrix[:, m, :, :] * Z[:, :, np.newaxis], axis=1) # (nk, 3)
                    J_response[it, :] += np.imag(np.sum(rho_nm[:, np.newaxis] * Z, axis=0))
        J_response *= - 2 * self.medium.spin_factor / (self.medium.volume * self.medium.nk)
        return J_response

    def concentration_of_charge_carriers(self):
        """ Return the concentration of charge carriers.

        The function returns a tuple with two numbers: the first one is the
        concentration of electrons, and the second one is the concentration
        of holes. Ideally, the two numbers must be the same, but they generally
        differ due to discretization errors.
        """
        nk = self.medium.nk
        nv = self.medium.nv
        n_e = np.sum(self.rho_diagonal[:, nv:])
        n_h = np.sum(1 - self.rho_diagonal[:, :nv])
        normalization_factor = self.medium.spin_factor / (self.medium.volume * nk)
        return (normalization_factor*n_e, normalization_factor*n_h)

    def residual_intraband_current(self):
        """ Return the residual electric-current density neglecting coherences. """
        J_residual = np.zeros((3,)) # [J_x, J_y, J_z]
        p_matrix = self.medium.momentum # (nk, nb, nb, 3)
        nk = self.medium.nk
        nv = self.medium.nv
        nb = self.medium.nb
        normalization_factor = self.medium.spin_factor / (self.medium.volume * nk)
        for ib in range(nv, nb):
            rho = self.rho_diagonal[:, ib].reshape((nk, 1))
            J_residual += np.sum(np.real(p_matrix[:, ib, ib, :]) * rho, axis=0)
        for ib in range(0, nv):
            rho = 1 - self.rho_diagonal[:, ib].reshape((nk, 1))
            J_residual -= np.sum(np.real(p_matrix[:, ib, ib, :]) * rho, axis=0)
        J_residual *= - normalization_factor
        return J_residual

    def average_inverse_mass(self, gap_threshold=1e-4, sum_over_bands=True,
        sum_over_k=True, initial_band=None, inverse_mass=None):
        """ Return the tensor of the averaged inverse mass.

        This function assumes an infinitesimally small amplitude of the vector
        potential of the probe pulse, so that the reciprocal-space excursion
        is infinitesimally small.

        Parameters
        ----------
        gap_threshold: (float) terms where the denominator is smaller than
            this value will be dropped (see Medium.calculate_inverse_mass);
        sum_over_bands: (bool) if True, the function returns the average over
            all the bands;
        sum_over_k: (bool) if True, the function returns the average over the
            first Brillouin zone;
        initial_band: (int or None) if None, then contributions from all the
            initial states will be added together; if "initial_band" is an
            integer, then only contributions from that initial state will count.
        inverse_mass : (int or None) if None, then the function will call
            medium.calculate_inverse_mass(gap_threshold) to evaluate it;
            otherwise, this function will use the provided band curvatures.

        Returns
        -------
          If sum_over_bands and sum_over_k are true, the function returns
        a 3x3 matrix of the effective inverse-mass tensor. If there are no
        excitations, the matrix will be filled with zeros. Otherwise, the
        inverse-mass tensor returned by Medium.calculate_inverse_mass will be
        averaged the using the excitation probabilities as weights.
          If sum_over_bands==False and sum_over_k==True, the function
        returns an array of shape (nb, 3, 3); summing over the first dimension
        will produce the same result as calling the function with
        sum_over_bands==True and sum_over_k==True. This can be useful,
        e.g., to separate the contributions from holes and conduction-band
        electrons.
          If sum_over_bands==True and sum_over_k==False, the function
        returns an array of shape (nk, 3, 3). Summing over the first dimension
        will produce the averaged effective mass.
        If sum_over_bands==False and sum_over_k==False, the function
        returns an array of shape (nk, nb, 3, 3).
        """
        ndims = self.medium.momentum.shape[-1] # the number of dimensions
        nk = self.medium.nk
        nb = self.medium.nb
        nv = self.medium.nv
        result_shape = [ndims, ndims]
        if not sum_over_bands:
            result_shape = [nb] + result_shape
        if not sum_over_k:
            result_shape = [nk] + result_shape
        # check if there are any excitations
        n_e = self.concentration_of_charge_carriers()[0]
        ## if n_e < nk*nv*np.finfo(float).eps:
        ##     return np.zeros(result_shape)
        # calculate the curvatures of the band energies
        if inverse_mass is None:
            inverse_mass = self.medium.calculate_inverse_mass(gap_threshold)
        # do the averaging
        numerator = np.empty((nk, nb, ndims, ndims))
        denominator = np.sum(self.rho_diagonal[:, nv:]).reshape((1, 1))
        if initial_band is None:
            rho_diagonal = self.rho_diagonal[:, :, np.newaxis, np.newaxis]
            numerator[:, nv:, :, :] = rho_diagonal[:, nv:, :, :] * inverse_mass[:, nv:, :, :]
            numerator[:, :nv, :, :] = (rho_diagonal[:, :nv, :, :] - 1) * inverse_mass[:, :nv, :, :]
        else:
            if initial_band >= nv:
                return np.zeros(result_shape)
            psi = self.psi[:, :, initial_band]
            rho_diagonal = np.real(psi * psi.conj())
            rho_diagonal = rho_diagonal[:, :, np.newaxis, np.newaxis]
            i0 = initial_band
            numerator[:, :i0, :, :] = rho_diagonal[:, :i0:, :, :] * inverse_mass[:, :i0:, :, :]
            numerator[:, i0, :, :] = (rho_diagonal[:, i0, :, :] - 1) * inverse_mass[:, i0, :, :]
            numerator[:, i0+1:, :, :] = rho_diagonal[:, i0+1:, :, :] * inverse_mass[:, i0+1:, :, :]
        if sum_over_bands:
            numerator = np.sum(numerator, axis=1)
        if sum_over_k:
            numerator = np.sum(numerator, axis=0)
        return numerator / denominator

    def cycle_averaged_inverse_mass(self, E0_vector, omega0,
            sum_over_bands=True, sum_over_k=True, CAIM=None):
        '''Compute the cycle-averaged inverse mass.

        Parameters
        ----------
        E0_vector: (3) The Cartesian components of a vector specifying the
            amplitude of a probe electric field.
        omega0: (scalar) The central frequency of a probe electric field.
        sum_over_bands: if True, the function returns the average over
            all the bands;
        sum_over_k: if True, the function returns the average over the
            first Brillouin zone;
        CAIM : if None, then the function will call
            medium.cycle_averaged_inverse_mass(E0_vector, omega0) to evaluate it;
            otherwise, this function will use the provided array.


        Returns
        -------
          If sum_over_bands and sum_over_k are true, the function returns
        a vector of three elements. In the weak-field limit, this vector
        approaches the product of the conventional inverse-mass tensor
        and a unit vector pointing along "E0_vector" (see
        Medium.cycle_averaged_inverse_mass for more details). If there are no
        excitations, all the elements of the vector will be zero. Otherwise,
        averaging is performed using the excitation probabilities as weights.
          If sum_over_bands==False and sum_over_k==True, the function
        returns an array of shape (nb, 3); summing over the first dimension
        will produce the same result as calling the function with
        sum_over_bands==True and sum_over_k==True. This can be useful,
        e.g., to separate the contributions from holes and conduction-band
        electrons.
          If sum_over_bands==True and sum_over_k==False, the function
        returns an array of shape (nk, 3). Summing over the first dimension
        will produce the averaged effective mass.
        If sum_over_bands==False and sum_over_k==False, the function
        returns an array of shape (nk, nb, 3).
        '''
        ndims = self.medium.momentum.shape[-1] # the number of dimensions
        nk = self.medium.nk
        nb = self.medium.nb
        nv = self.medium.nv
        result_shape = [ndims]
        if not sum_over_bands:
            result_shape = [nb] + result_shape
        if not sum_over_k:
            result_shape = [nk] + result_shape
        # check if there are any excitations
        n_e = self.concentration_of_charge_carriers()[0]
        # if (n_e < nk*nv*np.finfo(float).eps):
        #     return np.zeros(result_shape)
        # calculate the k-resolved cycle-averaged inverse masses
        if CAIM is None:
            CAIM = self.medium.cycle_averaged_inverse_mass(E0_vector, omega0)
        # do the averaging
        rho_diagonal = self.rho_diagonal[:, :, np.newaxis]
        numerator = np.empty((nk, nb, ndims))
        numerator[:, nv:, :] = rho_diagonal[:, nv:, :] * CAIM[:, nv:, :]
        numerator[:, :nv, :] = (rho_diagonal[:, :nv, :] - 1) * CAIM[:, :nv, :]
        denominator = np.sum(rho_diagonal[:, nv:], axis=(0,1))
        if sum_over_bands:
            numerator = np.sum(numerator, axis=1)
        if sum_over_k:
            numerator = np.sum(numerator, axis=0)
        return numerator / denominator

    def Drude_response(self, no_Drude_response_from_full_bands=False,
            gap_threshold=1e-4, inverse_mass=None):
        """ Return the intraband electric-current response to vector potential.

        The function evaluates the part of the linear response that is due to
        the intraband motion. It returns a tensor that relates the induced current
        density to the vector potential of a probe pulse. Let this tensor be
        sigma_{alpha,beta}. Then
            J_{alpha}(omega) = sum_{beta} sigma_{alpha,beta} * A_{beta}(omega).
        THE FUNCTION CURRENTLY WORKS ONLY FOR TDSE RESULTS.

        Parameters
        ----------
        no_Drude_response_from_full_bands : (bool)
            If True, the term responsible for the intraband motion will be dropped for
            uniformly populated bands.
        gap_threshold: (float)
            terms involving transitions between energy states separated by a gap
            smaller this value will be ignored.
        inverse_mass : (nk, nb, ndims, ndims) if None, then the function will call
            medium.calculate_inverse_mass(gap_threshold) to evaluate it;
            otherwise, this function will use the provided band curvatures.

        Returns
        -------
        sigma: an array of shape (3, 3)
        """
        epsilon = np.finfo(np.float).eps # a negligibly small real number
        if inverse_mass is None:
            inverse_mass = self.medium.calculate_inverse_mass(gap_threshold)
        rho_diagonal = self.rho_diagonal.copy()
        rho_diagonal[:, :self.medium.nv] -= 1.0 # use the electron-hole representation
        normalization_factor = self.medium.spin_factor / (self.medium.volume * self.medium.nk)
        # evaluate the contributions from the intraband motion (Drude response)
        Y = inverse_mass * rho_diagonal[:, :, np.newaxis, np.newaxis] # (nk, nb, 3, 3)
        if no_Drude_response_from_full_bands:
            # check if any of the bands is uniformly populated
            Drude_mask = (rho_diagonal.ptp(axis=0) > epsilon)
            # use only those bands that are not uniformly populated
            if np.count_nonzero(Drude_mask) > 0:
                sigma = -np.sum(Y[:, Drude_mask, :, :], axis=(0, 1))
            else:
                sigma = np.zeros((3, 3))
        else:
            sigma = -np.sum(Y, axis=(0, 1))
        sigma *= normalization_factor
        return sigma


    def linear_response(self, omega_array, decoherence_rate=2.418884e-04,
            subtract_initial_state=False, no_Drude_response_from_full_bands=False,
            gap_threshold=1e-4, inverse_mass=None, adjust_decoherence=False):
        """ Return the linear electric-current response to vector potential.

        The function uses an analytical formula for evaluating the linear response of an
        excited state. It returns a tensor that relates the induced electric-current
        density to the vector potential of a probe pulse. Let this tensor be
        sigma_{alpha,beta}(omega). Then
            J_{alpha}(omega) = sum_{beta} sigma_{alpha,beta}(omega) * A_{beta}(omega).
        THE FUNCTION CURRENTLY WORKS ONLY FOR TDSE RESULTS.

        Parameters
        ----------
        omega_array : an array of shape (nw,)
            An array of circular frequencies, at which the linear response must be evaluated.
            It is highly recommended to obtain this array from the function called
            "dominant_transition_frequencies".
        decoherence_rate : scalar
            The decay rate of the polarization induced by interband transitions.
        subtract_initial_state : bool
            If True, the function returns the difference between the response of the final
            state and that of the initial (ground) state.
        no_Drude_response_from_full_bands : bool
            If True, the term responsible for the intraband motion will be dropped for
            uniformly populated bands.
        gap_threshold: scalar
            terms involving transitions between energy states separated by a gap
            smaller this value will be ignored.
        inverse_mass : if None, then the function will call
            medium.calculate_inverse_mass(gap_threshold) to evaluate it;
            otherwise, this function will use the provided band curvatures.
        adjust_decoherence : if True, the decoherence rate is adjusted for response
            at low frequencies to ensure that dephasing broadens absorption lines by
            no more than a fraction of the energy spacing between the initial and
            final states.

        Returns
        -------
        sigma : an array of shape (nw, 3, 3)
            The last two dimensions are Cartesian indices.
        """
        ## nk = self.medium.nk
        ## nb = self.medium.nb
        nw = len(omega_array)
        sigma = np.zeros((nw, 3, 3), dtype=np.complex)
        if inverse_mass is None:
            inverse_mass = self.medium.calculate_inverse_mass(gap_threshold)
        if subtract_initial_state:
            rho_diagonal = self.rho_diagonal.copy()
            rho_diagonal[:, :self.medium.nv] -= 1.0
        else:
            rho_diagonal = self.rho_diagonal
        p_matrix = self.medium.momentum # (nk, nb, nb, 3)
        for iw in range(nw):
            sigma[iw,:,:] = jit_linear_interband_response(omega_array[iw],
                decoherence_rate, gap_threshold, inverse_mass, adjust_decoherence,
                rho_diagonal, p_matrix, self.medium.spin_factor, self.medium.volume,
                self.medium.energy)
        ## normalization_factor = self.medium.spin_factor / (self.medium.volume * nk)
        ## # evaluate the contributions from momentum matrix elements
        ## for n in range(nb):
        ##     omega_mn = self.medium.energy - self.medium.energy[:, n, np.newaxis] # (nk, nb)
        ##     if adjust_decoherence:
        ##         # make sure that dephasing broadens absorption lines by not more than
        ##         # a fraction of the energy spacing between the initial and final states
        ##         gamma = decoherence_rate * np.ones((nk, nb))
        ##         max_broadening_fraction = 0.1
        ##         mask = (max_broadening_fraction * np.abs(omega_mn) < decoherence_rate)
        ##         gamma[mask] = max_broadening_fraction * np.abs(omega_mn[mask])
        ##         gamma = gamma[np.newaxis, :, :]
        ##     else:
        ##         gamma = decoherence_rate
        ##     # evaluate denominators
        ##     d0 = omega_array[:, np.newaxis, np.newaxis] + 1j * gamma
        ##     d1 = omega_mn[np.newaxis, :, :] - \
        ##         omega_array[:, np.newaxis, np.newaxis] - 1j * gamma
        ##     d2 = omega_mn[np.newaxis, :, :] + \
        ##         omega_array[:, np.newaxis, np.newaxis] + 1j * gamma
        ##     mask1 = (np.abs(omega_mn) < gap_threshold)
        ##     mask2 = np.tile(mask1, (nw,1,1)) # (nw, nk, nb)
        ##     # iterate over the Cartesian indices
        ##     for ialpha in range(3):
        ##         for ibeta in range(3):
        ##             pp = p_matrix[:, n, :, ialpha] * p_matrix[:, :, n, ibeta] # (nk, nb)
        ##             pp_times_rho = pp * rho_diagonal[:, n, np.newaxis]
        ##             numerator = 2j * np.imag(pp_times_rho) / d0 + \
        ##               pp_times_rho / d1 + pp_times_rho.conj() / d2
        ##             numerator[mask2] = 0.0
        ##             denominator = omega_mn**2
        ##             denominator[mask1] = 1.0
        ##             denominator = np.tile(denominator, (nw,1,1))
        ##             sigma[:, ialpha, ibeta] += np.sum(numerator / denominator, axis=(1,2))
        ## sigma *= normalization_factor
        ## sigma = sigma * omega_array.reshape((nw, 1, 1))**2
        # evaluate the contributions from the intraband motion (Drude response)
        sigma_Drude = self.Drude_response(no_Drude_response_from_full_bands, gap_threshold, inverse_mass)
        return sigma + sigma_Drude[np.newaxis, :, :]


    def dominant_transition_frequencies(self, omega_min, omega_max, threshold=0.9,
        Cartesian_index=None, decoherence_rate=0.0004):
        """ Return an array of circular frequencies that correspond to interband transitions.

        In typical 3D simulations, the number of crystal momenta is rather limited. Evaluating
        the linear absorption, one frequently observes gaps at frequencies that happen to
        be remote from any of the interband transitions supported the chosen k-grid. These
        gaps are numerical artifacts, and even though a large dephasing rate alleviates the
        problem, a prohibitively large rate is often required to produce a good-looking plot
        of absorption. At the same time, the functions "linear_response" and
        "linear_susceptibility" return accurate results at frequencies that correspond
        to dipole-allowed transitions between states that have different occupations.
        This function returns a list of such "reliable" frequencies.

        Parameters
        ----------
        omega_min, omega_max : (scalars) an interval of circular frequencies to
            search for interband transitions; both frequencies must be positive;
        threshold : (scalar) the value of this parameter should not exceed 1; the
            larger it is, the more aggressively the function eliminates transitions
            that it doesn't "trust";
        Cartesian_index : (integer scalar or None) if None, then the absorption is
            estimated from the sum over all the Cartesian components; otherwise,
            a single component is used, and the value of this parameter must be an
            integer number between 0 (x-axis) and 2 (z-axis);
        decoherence_rate : (scalar) this assumed decoherence rate is used to roughly estimate
            absorption, and it also controls how closely spaced the selected frequencies
            may be (if the function returns too many frequencies, increase the value of this
            parameters); this rate does not have to be the same as that used to evaluate
            the linear response, and it must be larger than zero.

        Returns
        -------
        omega_array : (nw,) a list of circular frequencies that correspond to most
            prominent interband transitions; each of these frequencies will be between
            omega_min and omega_max.
        """
        nk = self.medium.nk
        nb = self.medium.nb
        nv = self.medium.nv
        # check the parameters
        assert omega_min < omega_max
        assert omega_min > 0
        # if necessary, cacluate the table of nearest neighbours
        if self.neighbour_table is None:
            self.neighbour_table = Hdf5Interface.nearest_neighbour_table(self.medium.klist3d, 1)
        # find all transition frequencies from lower to upper states
        omega_mn = np.zeros((nk, nb, nb))
        for n in range(nb-1): # cycle over initial bands
            omega_mn[:, n+1:, n] = self.medium.energy[:, n+1:] - self.medium.energy[:, n, np.newaxis]
        # find all the transitions within the given range of frequencies and sort them
        # (use a slightly broader frequency range to handle decoherence)
        index_tuple = np.nonzero(np.logical_and(omega_mn > 0,
            np.logical_and(omega_mn >= omega_min - 2*decoherence_rate,
            omega_mn <= omega_max + 2*decoherence_rate)))
        k_indices, m_indices, n_indices = index_tuple
        omega_mn = omega_mn[index_tuple]
        order = np.argsort(omega_mn)
        omega_mn = omega_mn[order]
        k_indices = k_indices[order]
        m_indices = m_indices[order]
        n_indices = n_indices[order]
        del index_tuple, order
        # for each of the identified transitions, calculate
        # $ |f_m(k) - f_n(k)| / \omega_{m n}^2 \alpha |p_{mn}^\alpha|^2$
        # which is the omega-independent part of the expression for Im[\chi_{\alpha \alpha}];
        # if necessary, sum over the Cartesian components ($\alpha$)
        occupation_differences = self.rho_diagonal[(k_indices, m_indices)] - \
            self.rho_diagonal[(k_indices, n_indices)] # f_m(k) - f_n(k)
        weights = np.abs(occupation_differences) / omega_mn**2
        if Cartesian_index is None:
            p_weights = np.zeros(len(omega_mn))
            for j in range(3): # loop over Cartesian indices
                Z = self.medium.momentum[(k_indices, m_indices, n_indices,
                    np.full(len(omega_mn), j, dtype=np.int))]
                p_weights += np.real(Z * Z.conj())
            weights *= p_weights
        else:
            Z = self.medium.momentum[(k_indices, m_indices, n_indices,
                    np.full(len(omega_mn), Cartesian_index, dtype=np.int))]
            weights *= np.real(Z * Z.conj())
        # mark those transitions that are expected to give significant contributions
        Y = weights * omega_mn**2
        selection = (Y >= 1e-3 * np.max(Y))
        ## # BEGIN DEBUGGING
        ## import sys
        ## flat_klist3d = self.medium.klist3d.flatten()
        ## f = open("weights.dat", 'w')
        ## for i in range(len(weights)):
        ##     if selection[i]:
        ##         indices = np.nonzero(flat_klist3d == k_indices[i])[0]
        ##         if len(indices) == 0:
        ##             print("ERROR: len(indices) == 0")
        ##             sys.exit(1)
        ##         if len(indices) > 1:
        ##             print("ERROR: len(indices) > 1")
        ##             sys.exit(1)
        ##         ik1, ik2, ik3 = np.unravel_index(indices[0], self.medium.klist3d.shape)
        ##         f.write("{:8.5f} {:9.3e} {:3d} {:3d} {:2d} {:2d} {:2d}\n".format(
        ##             omega_mn[i]*27.21, weights[i], m_indices[i], n_indices[i], ik1, ik2, ik3))
        ## f.close()
        ## # END DEBUGGING
        # roughly estimate the absorption (amplification) at those transition frequencies
        # that lie in the [omega_min, omega_max] range (omitting constant prefactors)
        selection[omega_mn < omega_min] = False
        selection[omega_mn > omega_max] = False
        abs_Im_chi = np.zeros_like(weights)
        for transition_index in range(len(omega_mn)):
            if selection[transition_index]:
                omega = omega_mn[transition_index]
                # identify transitions that may contribute
                i1 = np.searchsorted(omega_mn, omega - 2*decoherence_rate)
                i2 = np.searchsorted(omega_mn, omega + 2*decoherence_rate)
                # add the contributions
                abs_Im_chi[transition_index] = np.sum(weights[i1:i2] /
                        ((omega - omega_mn[i1:i2])**2 + decoherence_rate**2))
        ## # BEGIN DEBUGGING
        ## print(len(weights))
        ## flat_klist3d = self.medium.klist3d.flatten()
        ## f = open("abs_Im_chi_before.dat", 'w')
        ## for i in range(len(weights)):
        ##     if selection[i]:
        ##         indices = np.nonzero(flat_klist3d == k_indices[i])[0]
        ##         if len(indices) == 0:
        ##             print("ERROR: len(indices) == 0")
        ##             sys.exit(1)
        ##         if len(indices) > 1:
        ##             print("ERROR: len(indices) > 1")
        ##             sys.exit(1)
        ##         ik1, ik2, ik3 = np.unravel_index(indices[0], self.medium.klist3d.shape)
        ##         f.write("{:8.5f} {:9.3e} {:3d} {:3d} {:2d} {:2d} {:2d}\n".format(
        ##             omega_mn[i]*27.21, abs_Im_chi[i], m_indices[i], n_indices[i], ik1, ik2, ik3))
        ## f.close()
        ## # END DEBUGGING
        # "weed out" transition frequencies that are too unimportant or too unreliable
        # step 1: get rid of very close frequencies
        while True:
            selected_indices = np.nonzero(selection)[0]
            selected_frequencies = omega_mn[selection]
            d_omega = selected_frequencies[1:] - selected_frequencies[:-1]
            indices = np.nonzero(np.abs(d_omega) < decoherence_rate)[0]
            if len(indices) == 0:
                break
            for i in indices:
                j1 = selected_indices[i]
                j2 = selected_indices[i+1]
                if selection[j1] and selection[j2]:
                    if abs_Im_chi[j1] > abs_Im_chi[j2]:
                        selection[j2] = False
                    else:
                        selection[j1] = False
        # step 2: consider transitions at neighboring crystal momenta,
        # starting from the most prominent transitions
        for transition_index in np.argsort(abs_Im_chi)[::-1]:
            if not selection[transition_index]:
                continue
            omega0 = omega_mn[transition_index]
            ik_central = k_indices[transition_index]
            for ik in self.neighbour_table[ik_central, :, :].flatten():
                # of all the transitions at this crystal momentum,
                # find the one closest in frequency to omega
                indices = np.nonzero(k_indices == ik)[0]
                if len(indices) == 0: continue
                j = indices[np.argmin(np.abs(omega0 - omega_mn[indices]))]
                if selection[j]:
                    # if the difference between the transition frequencies is not too large,
                    # examine transitions in the frequency range between omega0 and omega_mn[j]
                    if np.abs(omega0 - omega_mn[j]) < 0.01 / au.eV:
                        min_tolerable_abs_Im_chi = threshold * min(abs_Im_chi[transition_index],
                            abs_Im_chi[j])
                        j_min = 1 + min(transition_index, j)
                        j_max = max(transition_index, j)
                        indices = np.nonzero(abs_Im_chi[j_min:j_max] < min_tolerable_abs_Im_chi)[0]
                        if len(indices) > 0:
                            selection[indices+j_min] = False
        
        ## # BEGIN DEBUGGING
        ## flat_klist3d = self.medium.klist3d.flatten()
        ## f = open("abs_Im_chi_after.dat", 'w')
        ## for i in range(len(weights)):
        ##     if selection[i]:
        ##         indices = np.nonzero(flat_klist3d == k_indices[i])[0]
        ##         if len(indices) == 0:
        ##             print("ERROR: len(indices) == 0")
        ##             sys.exit(1)
        ##         if len(indices) > 1:
        ##             print("ERROR: len(indices) > 1")
        ##             sys.exit(1)
        ##         ik1, ik2, ik3 = np.unravel_index(indices[0], self.medium.klist3d.shape)
        ##         f.write("{:8.5f} {:9.3e} {:3d} {:3d} {:2d} {:2d} {:2d}\n".format(
        ##             omega_mn[i]*27.21, abs_Im_chi[i], m_indices[i], n_indices[i], ik1, ik2, ik3))
        ## f.close()
        ## import sys
        ## sys.exit(0)
        ## # END DEBUGGING
        return omega_mn[selection]


    def Drude_susceptibility(self, omega_array, no_Drude_response_from_full_bands=False,
            gap_threshold=1e-4, inverse_mass=None):
        """ Return the tensor of the Drude susceptibility.

        The function evaluates the part of the linear response that is due to
        the intraband motion. THE FUNCTION CURRENTLY WORKS ONLY FOR TDSE RESULTS.

        Parameters
        ----------
        omega_array : an array of shape (nw,)
            An array of circular frequencies, at which the linear response must be evaluated.
            None of the frequencies may be equal to zero, but negative frequencies are allowed.
        no_Drude_response_from_full_bands : bool
            If True, the term responsible for the intraband motion will be dropped for
            uniformly populated bands.
        gap_threshold: scalar
            terms where the denominator is smaller than this value will be dropped.
        inverse_mass : if None, then the function will call
            medium.calculate_inverse_mass(gap_threshold) to evaluate it;
            otherwise, this function will use the provided band curvatures.

        Returns
        -------
        chi: an array of shape (nw, 3, 3)
        """
        nw = len(omega_array)
        omega_squared = omega_array**2
        sigma = self.Drude_response(no_Drude_response_from_full_bands,
            gap_threshold, inverse_mass)
        chi = sigma[np.newaxis, :] / omega_squared[:, np.newaxis, np.newaxis]
        return chi

    def linear_susceptibility(self, omega_array, decoherence_rate=2.418884e-04,
            subtract_initial_state=False, no_Drude_response_from_full_bands=False,
            gap_threshold=1e-4, inverse_mass=None, adjust_decoherence=False):
        """ Return the tensor of the linear susceptibility.

        The function uses an analytical formula for evaluating the linear response of an excited state.
        THE FUNCTION CURRENTLY WORKS ONLY FOR TDSE RESULTS.

        Parameters
        ----------
        omega_array : an array of shape (nw,)
            An array of circular frequencies, at which the linear response must be evaluated.
            None of the frequencies may be equal to zero, but negative frequencies are allowed.
            It is highly recommended to obtain this array from the function called
            "dominant_transition_frequencies".
        subtract_initial_state : bool
            If True, the function returns the difference between the response of the final
            state and that of the ground state.
        decoherence_rate : scalar
            The decay rate of the polarization induced by interband transitions.
        no_Drude_response_from_full_bands : bool
            If True, the term responsible for the intraband motion will be dropped for
            uniformly populated bands.
        gap_threshold: scalar
            terms involving transitions between energy states separated by a gap
            smaller this value will be ignored.
        inverse_mass : if None, then the function will call
            medium.calculate_inverse_mass(gap_threshold) to evaluate it;
            otherwise, this function will use the provided band curvatures.
        adjust_decoherence : if True, the decoherence rate is adjusted for response
            at low frequencies to ensure that dephasing broadens absorption lines by
            no more than a fraction of the energy spacing between the initial and
            final states.

        Returns
        -------
        chi : an array of shape (nw, 3, 3)
            The tensor of the linear susceptibility (dimensions 1 and 2 are Cartesian indices).
        """
        nw = len(omega_array)
        omega_squared = omega_array**2
        sigma = self.linear_response(omega_array, decoherence_rate,
            subtract_initial_state, no_Drude_response_from_full_bands,
            gap_threshold, inverse_mass, adjust_decoherence)
        chi = sigma / omega_squared[:, np.newaxis, np.newaxis]
        return chi

    def deposited_energy(self):
        """ Return the net energy of electronic excitations (per unit volume). """
        nv = self.medium.nv
        nk = self.medium.nk
        electron_energy = np.sum(self.rho_diagonal[:, nv:] * self.medium.energy[:, nv:])
        hole_energy = - np.sum((1 - self.rho_diagonal[:, :nv]) * self.medium.energy[:, :nv])
        return (electron_energy + hole_energy) * self.medium.spin_factor / (self.medium.volume * nk)

    def energy_dependent_probabilities(self, energies, averaging_window_FWHM):
        """ Calculate the probability of finding a charge carrier with a given energy

        For energies that correspond to conduction bands, the function returns the
        probabilities of fiding an electron with a certain energy. For valence bands,
        the function returns the probabilities of finding a hole.
        
        Parameters
        ----------
        energies : an array of energies, for which the DOS needs to be evaluated;
        averaging_window_FWHM: the full width at half maximum of the Gaussian window
                            that is used for averaging;
        
        Returns
        -------
        result : an array that has the same shape as "energies";
        """
        result = np.zeros(energies.size)
        exp_factor = 4 * np.log(2.0) / averaging_window_FWHM**2
        # find the energies limiting the band gap
        E1 = np.max(self.medium.energy[:, self.medium.nv - 1])
        E2 = np.min(self.medium.energy[:, self.medium.nv])
        # evaluate the probabilities
        for i in range(energies.size):
            energy = energies[i]
            Y = exp_factor * (self.medium.energy - energy)**2
            weights = np.zeros_like(Y)
            s = (Y < 30)
            weights[s] = np.exp(-Y[s])
            if energy <= E1: # holes
                weights[self.medium.energy > E1] = 0
                Y = 1 - self.rho_diagonal
                Y[Y < 0] = 0
                result[i] = np.sum(weights * Y) / np.sum(weights)
            elif energy >= E2:
                weights[self.medium.energy < E2] = 0
                result[i] = np.sum(weights * self.rho_diagonal) / np.sum(weights)
        return result.reshape(energies.shape)

    ## def analyze_excitations(self, bin_size, maximal_transition_energy):
    ##     """ Analyze how excitation probabilities depend on transition frequencies.
    ## 
    ##     Parameters
    ##     ----------
    ##     bin_size: (float) the size of a single energy bin; it must be positive;
    ##     maximal_transition_energy: (float).
    ## 
    ##     Returns: (energy_bins, bin_occupations)
    ##     -------
    ##     energy_bins: (nbins,) An array specifying the energy bins.
    ##     bin_occupations: an array of shape (nbins,) that contains the average excitation
    ##         probabilities within each energy bin. The i-th element of this array contains
    ##         the average number of valence electrons excited to conduction bands where the
    ##         transition energy lies within the interval
    ##             energy_bins[i-1] < hbar*omega_{nm} <= energy_bins[i].
    ##     """
    ##     nv = self.medium.nv
    ##     nk = self.medium.nk
    ##     # initialize the energy bins
    ##     E_min = np.min(self.medium.energy[:, nv] - self.medium.energy[:, nv-1])
    ##     E_max = np.max(self.medium.energy[:, -1] - self.medium.energy[:, 0])
    ##     nbins = 1 + int(np.ceil((maximal_transition_energy - E_min) / bin_size))
    ##     if maximal_transition_energy < E_max:
    ##         maximal_transition_energy = E_min + bin_size * (nbins - 1)
    ##         if maximal_transition_energy < E_max:
    ##             nbins += 1
    ##     energy_bins = np.empty(nbins)
    ##     if maximal_transition_energy < E_max:
    ##         energy_bins[:-1] = E_min + bin_size * np.arange(nbins-1)
    ##         energy_bins[-1] = E_max
    ##     else:
    ##         energy_bins[:] = E_min + bin_size * np.arange(nbins)
    ##     # fill the bins
    ##     bin_occupations = np.zeros(nbins)
    ##     counts = np.zeros(nbins, dtype=np.intp)
    ##     for initial_band in range(nv):
    ##         psi = self.psi[:, :, initial_band]
    ##         excitation_probabilities = np.real(psi * psi.conj()) # (nk, nb)
    ##         # sort the transitions by frequency
    ##         X = self.medium.energy[:, nv:] - self.medium.energy[:, initial_band, np.newaxis]
    ##         X = X.flatten()
    ##         Y = excitation_probabilities[:, nv:].flatten()
    ##         order = np.argsort(X)
    ##         X = X[order]
    ##         Y = Y[order]
    ##         # put the data into the bins
    ##         indices = np.digitize(X, energy_bins[:-1], right=True)
    ##         for i in range(len(Y)): # THERE MUST BE A SMARTER WAY TO DO THIS!
    ##             bin_occupations[indices[i]] += Y[i]
    ##             counts[indices[i]] += 1
    ##     # average the occupation numbers
    ##     c = (counts != 0)
    ##     bin_occupations[c] = bin_occupations[c] / counts[c]
    ##     # diagnostics
    ##     print("average counts per bin:", counts.mean())
    ##     # return the results
    ##     return (energy_bins, bin_occupations)
