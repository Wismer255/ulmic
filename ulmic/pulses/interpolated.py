import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import UnivariateSpline
from scipy.signal import hilbert
from ulmic.atomic_units import AtomicUnits

au = AtomicUnits()


class InterpolatedPulse:

    def __init__(self,pulse_name, data_type='E', E_max=1.0, polarisation_vector=np.array([1,0,0]),
                 use_frequency_filter=False, Nfilter=28):
        """ Read electric field or vector potential from file. """

        self.E_max = np.copy(E_max)
        self.polarisation_vector = np.copy(polarisation_vector)
        self.variables = [{'envelope':'interp'}]

        data = np.loadtxt(pulse_name)
        t_exp = data[:,0]
        if data_type == 'E':
            E_exp = data[:,1]
            A_exp = -cumtrapz(E_exp,t_exp,initial=0)
        elif data_type == 'A':
            A_exp = data[:,1]

        self.t_exp = t_exp
        self.A_exp = A_exp
        Nv = len(A_exp)
        M = 5
        T = t_exp[-1]-t_exp[0]
        t_new = np.linspace(-0.5*(M-1)*T + t_exp[0],0.5*(M-1)*T + t_exp[-1],M*len(t_exp))

        if use_frequency_filter:
            # Extend time axis
            A_new = np.zeros(M*Nv)
            A_new[Nv*int((M-1)/2):Nv*int((M+1)/2)] = A_exp

            # Apply frequency-domain filter
            A_new_fft = np.fft.fft(A_new)
            A_new_fft[Nfilter*M:-Nfilter*M] = np.zeros(len(A_new_fft[Nfilter*M:-Nfilter*M]))

            # Apply time-domain filter
            A_new = np.fft.ifft(A_new_fft).real*np.exp(-(t_new/(T))**6)
        else:
            t_new = t_exp
            A_new = A_exp

        # Construct interpolation function
        self.spl = UnivariateSpline(t_new,A_new,s=0,ext='zeros')
        self.spl_deriv = UnivariateSpline(t_new,A_new,s=0,ext='zeros').derivative(n=1)

        # Normalize with respect to electric field envelope
        electric_field = -self.spl_deriv(t_new)
        envelope = abs(hilbert(electric_field))
        norm = np.max((envelope))

        self.spl = UnivariateSpline(t_new,(self.E_max/norm)*A_new,s=0,ext='zeros')
        self.spl_deriv = UnivariateSpline(t_new,(self.E_max/norm)*A_new,s=0,ext='zeros').derivative(n=1)

        if abs(A_new[0]) > 1e-10 or abs(A_new[-1]) > 1e-10:
            print('Warning: A[0]={:e} and A[-1]={:e}'.format(A_new[0], A_new[-1]))

    def eval_potential_fast(self,t):
        return  self.polarisation_vector * self.spl(t)

    def eval_field_fast(self,t):
        return - self.polarisation_vector * self.spl_deriv(t)


class InterpPulses(InterpolatedPulse):

    def __init__(self,*args,**kwargs):
        """ Obsolete class. Use InterpolatedPulse instead. """
        super(InterpPulses,self).__init__(self,*args,**kwargs)

