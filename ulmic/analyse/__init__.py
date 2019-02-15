import numpy as np
from ulmic.atomic_units import AtomicUnits
from ulmic.ulmi.jit.linear_response import jit_linear_interband_response
# from ulmic.result import Result
# from ulmic.medium import Medium

au = AtomicUnits()


class FinalStateAnalyzer:
    """ A class for analyzing the electronic state at the end of a simulation.

    CURRENTLY, THIS CLASS WORKS ONLY FOR TDSE SIMULATIONS. """

    def __init__(self, medium, result, copy_wavefunctions=False):
        self.medium = medium
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
        """ Return the intraband electric-current response to vector potential.

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
        sigma_Drude = self.Drude_response(no_Drude_response_from_full_bands, gap_threshold)
        return sigma + sigma_Drude[np.newaxis, :, :]

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

    def analyze_excitations(self, bin_size, maximal_transition_energy):
        """ Analyze how excitation probabilities depend on transition frequencies.

        Parameters
        ----------
        bin_size: (float) the size of a single energy bin; it must be positive;
        maximal_transition_energy: (float).

        Returns: (energy_bins, bin_occupations)
        -------
        energy_bins: (nbins,) An array specifying the energy bins.
        bin_occupations: an array of shape (nbins,) that contains the average excitation
            probabilities within each energy bin. The i-th element of this array contains
            the average number of valence electrons excited to conduction bands where the
            transition energy lies within the interval
                energy_bins[i-1] < hbar*omega_{nm} <= energy_bins[i].
        """
        nv = self.medium.nv
        nk = self.medium.nk
        # initialize the energy bins
        E_min = np.min(self.medium.energy[:, nv] - self.medium.energy[:, nv-1])
        E_max = np.max(self.medium.energy[:, -1] - self.medium.energy[:, 0])
        nbins = 1 + int(np.ceil((maximal_transition_energy - E_min) / bin_size))
        if maximal_transition_energy < E_max:
            maximal_transition_energy = E_min + bin_size * (nbins - 1)
            if maximal_transition_energy < E_max:
                nbins += 1
        energy_bins = np.empty(nbins)
        if maximal_transition_energy < E_max:
            energy_bins[:-1] = E_min + bin_size * np.arange(nbins-1)
            energy_bins[-1] = E_max
        else:
            energy_bins[:] = E_min + bin_size * np.arange(nbins)
        # fill the bins
        bin_occupations = np.zeros(nbins)
        counts = np.zeros(nbins, dtype=np.intp)
        for initial_band in range(nv):
            psi = self.psi[:, :, initial_band]
            excitation_probabilities = np.real(psi * psi.conj()) # (nk, nb)
            # sort the transitions by frequency
            X = self.medium.energy[:, nv:] - self.medium.energy[:, initial_band, np.newaxis]
            X = X.flatten()
            Y = excitation_probabilities[:, nv:].flatten()
            order = np.argsort(X)
            X = X[order]
            Y = Y[order]
            # put the data into the bins
            indices = np.digitize(X, energy_bins[:-1], right=True)
            for i in range(len(Y)): # THERE MUST BE A SMARTER WAY TO DO THIS!
                bin_occupations[indices[i]] += Y[i]
                counts[indices[i]] += 1
        # average the occupation numbers
        c = (counts != 0)
        bin_occupations[c] = bin_occupations[c] / counts[c]
        # diagnostics
        print("average counts per bin:", counts.mean())
        # return the results
        return (energy_bins, bin_occupations)