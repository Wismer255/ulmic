import os
import numpy as np
from scipy.special import factorial
from ulmic.atomic_units import AtomicUnits
from ulmic.pulses.interpolated import InterpPulses

au = AtomicUnits()
pi = np.pi
eV_au = 1/27.211385
fs_au = 1/0.02418884326505
nm_au = 1/0.052917721
VA_au = 1/51.4220652
nm_Eph = 2*pi*137.035999/(1/0.052917721)


class AnalyticalPulse:

    def __init__(self, pulse_type):
        r"""Specify pulse parameters.

            pulse_type:     either a string or a dictionary with the following keys
                            {'omega':0.05,'E0':0.01,'FWHM':5.0*fs_au,'delay':0.0,'cep':0.0,
                            'envelope':'gauss','polarisation_vector':np.array([1, 0, 0])}
                            All values are given in atomic units. cep is given in units of \pi.
        """
        if type(pulse_type) == type(str()):
            self.variables = self.set_pulse(pulse_type)
        elif type(pulse_type) == type(dict()):
            self.variables = pulse_type
        else:
            raise ValueError("Wrong pulse type input! Input must be either string or dict.")

    def set_pulse(self,arg):

        if arg == 'lin_IR':
            return {'omega':nm_Eph/750.0,'E0':1.0*VA_au,'FWHM':5.0*fs_au,'delay':0.0,'cep':0.0,'envelope':'gauss','polarisation_vector':np.array([1, 0, 0])}
        elif arg == 'circ_IR':
            return {'omega':nm_Eph/750.0,'E0':1.0*VA_au,'FWHM':5.0*fs_au,'delay':0.0,'cep':0.0,'envelope':'gauss','polarisation_vector':np.array([1, 1j, 0])}
        elif arg == 'probe_UV':
            return {'omega':nm_Eph/250.0,'E0':0.001*VA_au,'FWHM':2.5*fs_au,'delay':0.0,'cep':0.0,'envelope':'gauss','polarisation_vector':np.array([1, 0, 0])}
        elif arg == 'pulse_A':
            return {'omega':nm_Eph/700.0,'E0': 1.0*VA_au,'FWHM':4*fs_au,'delay':0.0,'cep':0.0,'envelope':'gauss','polarisation_vector':np.array([1, 0, 0])}
        elif arg == 'pulse_B':
            return {'omega':nm_Eph/2100.0,'E0': 0.1*VA_au,'FWHM':15*fs_au,'delay':0.0,'cep':0.0,'envelope':'gauss','polarisation_vector':np.array([0, 1, 0])}
        elif isinstance(arg,tuple):
            if os.path.isfile(arg[0]):
                return InterpPulses(arg[0],arg[1])

    def eval(self,t,derivative=0):
        #Vector potential
        if type(t)==float or type(t)==int or type(t) == np.float64:
            nt = 1
            t =[t]
        else:
            nt = len(np.atleast_1d(t))
        vector_potential = np.zeros((nt,3))
        for pulse in self.variables:
            vector_potential += self.get_vector_potential(pulse,t,derivative)
        return vector_potential

    def eval_potential_fast(self,t):
        vector_potential = np.zeros(3)
        if self.variables['envelope'] == 'gauss':
            vector_potential += self.get_gaussian_vector_potential(self.variables,t)
        elif self.variables['envelope'] == 'HCP':
            vector_potential += self.get_HCP_vector_potential(self.variables,t)
        elif self.variables['envelope'] == '1cycle':
            vector_potential += self.get_single_cycle_vector_potential(self.variables,t)
        elif self.variables['envelope'] == 'slope':
            vector_potential += -self.variables['E0']*t*self.variables['polarisation_vector'].real
        elif self.variables['envelope'] == 'cos4':
            vector_potential += self.get_cos4_vector_potential(self.variables,t)
        elif self.variables['envelope'] == 'cos8':
            vector_potential += self.get_cos8_vector_potential(self.variables,t)
        elif self.variables['envelope'] == 'delta_spike':
            if t >= self.variables['delay']:
                vector_potential += self.variables['A0'] * self.variables['polarisation_vector'].real
        else:
            raise ValueError('Pulse parameter {} for envelope is not valid'.format(self.variables['envelope']))
        return vector_potential

    def eval_field_fast(self,t):
        electric_field = np.zeros(3)
        if self.variables['envelope'] == 'gauss':
            electric_field += self.get_gaussian_electric_field(self.variables,t)
        elif self.variables['envelope'] == 'HCP':
            electric_field += self.get_HCP_electric_field(self.variables,t)
        elif self.variables['envelope'] == '1cycle':
            electric_field += self.get_single_cycle_electric_field(self.variables,t)
        elif self.variables['envelope'] == 'slope':
            electric_field += self.variables['E0']*self.variables['polarisation_vector'].real
        elif self.variables['envelope'] == 'cos4':
            electric_field += self.get_cos4_electric_field(self.variables,t)
        elif self.variables['envelope'] == 'cos8':
            electric_field += self.get_cos8_electric_field(self.variables,t)
        elif self.variables['envelope'] == 'delta_spike':
            if t == self.variables['delay'] and self.variables['A0'] != 0:
                electric_field += np.finfo(np.float).max * self.variables['polarisation_vector'].real
        else:
            raise ValueError('Pulse parameter {} for envelope s not valid'.format(self.variables['envelope']))
        return electric_field

    def get_vector_potential(self,pulse,time,derivative=0):
        E0 =  pulse['E0']
        omega = pulse['omega']
        envelope_type = pulse['envelope']
        FWHM = pulse['FWHM']
        polarisation_vector = pulse['polarisation_vector']
        delay = pulse['delay']
        cep = pulse['cep']

        if envelope_type == 'gauss':
            M = derivative+1 # Number of terms
            vector_potential = np.zeros((len(time),3))
            for i in range(len(time)):
                env = (E0/omega)*np.exp(-2*np.log(2)*(((time[i]-delay)/FWHM)**2))
                #nfac = factorial(M-1)/(factorial(np.arange(M))*factorial(np.arange(M)[::-1]))
                #print(nfac)
                phases = ((-1j*omega)**(np.arange(M)[::-1]))*1j*np.exp(-1j*(cep + omega*(time[i]-delay)))
                hermite_sum = np.polynomial.hermite.hermval(np.sqrt(2*np.log(2))*(time[i]-delay)/FWHM,
                                            self.nfac[derivative]*phases*(-np.sqrt(2*np.log(2))/FWHM)**np.arange(M))
                vector_potential[i] = -env*np.real(np.outer(hermite_sum,polarisation_vector))

            return vector_potential
        else:
            raise ValueError('Pulse parameter {} for envelope s not valid'.format(self.variables['envelope']))

    def set_factorials(self,n=3):
        self.nfac = []
        for i in range(n):
            M = i+1
            self.nfac.append(factorial(M-1)/(factorial(np.arange(M))*factorial(np.arange(M)[::-1])))

    def get_gaussian_vector_potential(self,pulse,time):
        E0 =  pulse['E0']
        omega = pulse['omega']
        envelope_type = pulse['envelope']
        FWHM = pulse['FWHM']
        polarisation_vector = pulse['polarisation_vector']
        delay = pulse['delay']
        cep = pulse['cep']
        env = (E0/omega)*np.exp(-2*np.log(2)*(((time-delay)/FWHM)**2))
        phases = 1j*np.exp(-1j*(cep + omega*(time-delay)))
        return -env*np.real(phases*polarisation_vector)

    def get_gaussian_electric_field(self,pulse,time):
        E0 =  pulse['E0']
        omega = pulse['omega']
        envelope_type = pulse['envelope']
        FWHM = pulse['FWHM']
        polarisation_vector = pulse['polarisation_vector']
        delay = pulse['delay']
        cep = pulse['cep']
        env = (E0/omega)*np.exp(-2*np.log(2)*(((time-delay)/FWHM)**2))
        phases = 1j*np.exp(-1j*(cep + omega*(time-delay)))
        diff_env = (-4*np.log(2)*(time-delay)/FWHM**2)*env
        diff_phases = -1j*omega*phases
        return np.real((env*diff_phases + diff_env*phases) * polarisation_vector)

    def get_HCP_vector_potential(self,pulse,time):
        E0 =  pulse['E0']
        FWHM = pulse['FWHM']
        polarisation_vector = pulse['polarisation_vector'].real
        delay = pulse['delay']
        tau_l = FWHM*np.pi/(4*np.arccos(2**(-0.125)))
        t = time - delay
        if t <= -tau_l:
            A = 0.0
        else:
            t = min(t, tau_l)
            phi = np.pi*t / tau_l
            A = -E0 * (6*np.pi * (t + tau_l) + 8*tau_l * np.sin(phi) +
                tau_l * np.sin(2*phi)) / (16 * np.pi)
        return A * polarisation_vector

    def get_HCP_electric_field(self,pulse,time):
        E0 =  pulse['E0']
        FWHM = pulse['FWHM']
        polarisation_vector = pulse['polarisation_vector'].real
        delay = pulse['delay']
        tau_l = FWHM*np.pi/(4*np.arccos(2**(-0.125)))
        t = time - delay
        H = np.heaviside(tau_l-np.abs(t), 0.0)
        return H * E0 * np.cos(np.pi*t / (2*tau_l))**4 * polarisation_vector

    def get_single_cycle_vector_potential(self,pulse,time):
        E0 =  pulse['E0']
        # FWHM = pulse['FWHM']
        omega = pulse['omega']
        polarisation_vector = pulse['polarisation_vector'].real
        delay = pulse['delay']
        tau_l = np.pi / omega
        t = time - delay
        H = np.heaviside(tau_l-np.abs(t), 0.0)
        return H * E0 / omega * (1 + np.cos(omega*t)) * polarisation_vector

    def get_single_cycle_electric_field(self,pulse,time):
        E0 =  pulse['E0']
        # FWHM = pulse['FWHM']
        omega = pulse['omega']
        polarisation_vector = pulse['polarisation_vector'].real
        delay = pulse['delay']
        tau_l = np.pi / omega
        t = time - delay
        H = np.heaviside(tau_l-np.abs(t), 0.0)
        return H * E0 * np.sin(omega*t) * polarisation_vector

    def get_cos4_vector_potential(self,pulse,time):
        E0 =  pulse['E0']
        omega = pulse['omega']
        # envelope_type = pulse['envelope']
        FWHM = pulse['FWHM']
        polarisation_vector = pulse['polarisation_vector']
        delay = pulse['delay']
        cep = pulse['cep']
        if np.iscomplexobj(polarisation_vector):
            cep = cep + np.angle(polarisation_vector)
            polarisation_vector = np.abs(polarisation_vector)
        tau_l = FWHM*np.pi/(4*np.arccos(2**(-0.125)))
        t = time - delay
        H=np.heaviside(tau_l-np.abs(t),0.5)
        pulse_cos4 = -E0/omega * (np.cos(np.pi*t/(2*tau_l)))**4 * np.sin(omega*t+cep)
        return H*pulse_cos4*polarisation_vector

    def get_cos4_electric_field(self,pulse,time):
        E0 =  pulse['E0']
        omega = pulse['omega']
        # envelope_type = pulse['envelope']
        FWHM = pulse['FWHM']
        polarisation_vector = pulse['polarisation_vector']
        delay = pulse['delay']
        cep = pulse['cep']
        if np.iscomplexobj(polarisation_vector):
            cep = cep + np.angle(polarisation_vector)
            polarisation_vector = np.abs(polarisation_vector)
        tau_l = FWHM*np.pi/(4*np.arccos(2**(-0.125)))
        t = time - delay
        theta = np.pi*t / (2*tau_l)
        cos_term = np.cos(theta)
        sin_term = np.sin(theta)
        H = np.heaviside(tau_l-np.abs(t), 0.5)
        field_cos4 = -E0 * cos_term**3 * \
            (2*np.pi/(omega*tau_l) * sin_term * np.sin(omega*t+cep) -
                cos_term * np.cos(omega*t+cep))
        return H * field_cos4 * polarisation_vector

    def get_cos8_vector_potential(self,pulse,time):
        E0 =  pulse['E0']
        omega = pulse['omega']
        # envelope_type = pulse['envelope']
        FWHM = pulse['FWHM']
        polarisation_vector = pulse['polarisation_vector']
        delay = pulse['delay']
        cep = pulse['cep']
        if np.iscomplexobj(polarisation_vector):
            cep = cep + np.angle(polarisation_vector)
            polarisation_vector = np.abs(polarisation_vector)
        tau_l = FWHM*np.pi/(4*np.arccos(2**(-0.0625)))
        t = time - delay
        H=np.heaviside(tau_l-np.abs(t),0.5)
        pulse_cos8 = -E0/omega * (np.cos(np.pi*t/(2*tau_l)))**8 * np.sin(omega*t+cep)
        return H*pulse_cos8*polarisation_vector

    def get_cos8_electric_field(self,pulse,time):
        E0 =  pulse['E0']
        omega = pulse['omega']
        # envelope_type = pulse['envelope']
        FWHM = pulse['FWHM']
        polarisation_vector = pulse['polarisation_vector']
        delay = pulse['delay']
        cep = pulse['cep']
        if np.iscomplexobj(polarisation_vector):
            cep = cep + np.angle(polarisation_vector)
            polarisation_vector = np.abs(polarisation_vector)
        tau_l = FWHM*np.pi/(4*np.arccos(2**(-0.0625)))
        t = time - delay
        theta = np.pi*t / (2*tau_l)
        H = np.heaviside(tau_l-np.abs(t), 0.5)
        field_cos8 = E0 * np.cos(theta)**8 * \
            (np.cos(omega*t+cep) -
             4*np.pi/(omega*tau_l) * np.tan(theta) * np.sin(omega*t+cep))
        return H * field_cos8 * polarisation_vector
