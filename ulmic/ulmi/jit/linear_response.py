import numpy as np
from numba import jit, njit, prange
import sys

@njit(parallel=True)
def jit_linear_interband_response(omega, decoherence_rate,
            gap_threshold, inverse_mass, adjust_decoherence,
            rho_diagonal, p_matrix, spin_factor, cell_volume, band_energies):
        """ Return the intraband electric-current response to vector potential.

        The function uses an analytical formula for evaluating the linear response of an
        excited state. It returns a tensor that relates the induced electric-current
        density to the vector potential of a probe pulse. Let this tensor be
        sigma_{alpha,beta}(omega). Then
            J_{alpha}(omega) = sum_{beta} sigma_{alpha,beta}(omega) * A_{beta}(omega).
        THE FUNCTION CURRENTLY WORKS ONLY FOR TDSE RESULTS.

        Parameters
        ----------
        omega : (float) a circular frequency of probe light;
            An array of circular frequencies, at which the linear response must be evaluated.
        decoherence_rate : scalar
            The decay rate of the polarization induced by interband transitions.
        gap_threshold: scalar
            terms involving transitions between energy states separated by a gap
            smaller this value will be ignored.
        inverse_mass : band curvatures provided by "calculate_inverse_mass" in
            medium.properties;
        adjust_decoherence : if True, the decoherence rate is adjusted for response
            at low frequencies to ensure that dephasing broadens absorption lines by
            not more than a fraction of the energy spacing between the initial and
            final states.

        Returns
        -------
        sigma : a complex array of shape (3, 3) that contains the interband contribution
            to the susceptibility multiplied with omega**2.
        """
        nk = rho_diagonal.shape[0]
        nb = rho_diagonal.shape[1]
        sigma = np.zeros((nk, 3, 3), dtype=np.complex128)
        max_broadening_fraction = 0.1
        # evaluate the contributions from momentum matrix elements
        for ik in prange(nk):
            for n in range(nb):
                for m in range(nb):
                    if m == n:
                        continue
                    omega_mn = band_energies[ik, m] - band_energies[ik, n]
                    abs_omega_mn = np.abs(omega_mn)
                    if abs_omega_mn < gap_threshold:
                        continue
                    if adjust_decoherence:
                        gamma = min(decoherence_rate, max_broadening_fraction * abs_omega_mn)
                    else:
                        gamma = decoherence_rate
                    d0 = omega + 1j * gamma
                    d1 = omega_mn - omega - 1j * gamma
                    d2 = omega_mn + omega + 1j * gamma
                    for ialpha in range(3):
                        for ibeta in range(3):
                            pp = p_matrix[ik, n, m, ialpha] * p_matrix[ik, m, n, ibeta]
                            conj_pp = np.conj(pp)
                            sigma[ik, ialpha, ibeta] += rho_diagonal[ik, n] / omega_mn**2 * \
                                (2j * pp.imag / d0 + pp / d1 + conj_pp / d2)
        result = omega**2 * spin_factor / (cell_volume * nk) * np.sum(sigma, axis=0)
        return result
