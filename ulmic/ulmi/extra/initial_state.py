import numpy as np

class InitialState:

    def get_initial_state_by_name(self):

        nv = self.medium.nv
        if self.initial_state == 'ground_state':
            if self.state_object =='wave_functions':
                self.state = np.zeros((self.medium.nk_local,self.medium.nb,self.medium.nv), dtype=np.complex)
            elif self.state_object =='density_matrix':
                self.state = np.zeros((self.medium.nk_local,self.medium.nb,self.medium.nb), dtype=np.complex)
            for i in range(self.medium.nv):
                self.state[:,i,i] = np.ones(self.medium.nk_local)
            return self.state

        elif (self.equation == 'tdse' or self.equation == 'stdse') and self.initial_state == 'skewed':
            for i in range(self.medium.size[0]):
                k0 = self.medium.klist3d[i, 0, 0]
                kk = self.medium.klist1d[k0, 0]
                if kk > 0.0 and kk < 0.5:
                    self.state[k0, nv, nv - 1] = 0.3 * np.sin(2 * np.pi * kk)
                    self.state[k0, nv - 1, nv - 1] = np.sqrt(1 - self.state[k0, nv, nv - 1] ** 2)

        elif (self.equation == 'tdse' or self.equation == 'stdse') and self.initial_state == 'skewed':
            for i in range(self.medium.size[0]):
                k0 = self.medium.klist3d[i, 0, 0]
                kk = self.medium.klist1d[k0, 0]
                if kk > 0.0 and kk < 0.5:
                    self.state[k0, nv, nv - 1] = 0.3 * np.sin(2 * np.pi * kk)
                    self.state[k0, nv - 1, nv - 1] = np.sqrt(1 - self.state[k0, nv, nv - 1] ** 2)

        elif self.equation == 'lvn' and self.initial_state == 'skewed':
            for i in range(self.medium.size[0]):
                k0 = self.medium.klist3d[i, 0, 0]
                kk = self.medium.klist1d[k0, 0]
                if kk > 0.0 and kk < 0.5:
                    self.state[k0, nv, nv] = (0.3 * np.sin(2 * np.pi * kk)) ** 2
                    self.state[k0, nv - 1, nv - 1] = 1 - self.state[k0, nv, nv]
                    self.state[k0, nv, nv - 1] = np.sqrt(self.state[k0, nv, nv] * self.state[k0, nv - 1, nv - 1])
                    self.state[k0, nv - 1, nv] = np.sqrt(self.state[k0, nv, nv] * self.state[k0, nv - 1, nv - 1])

        elif (self.equation == 'tdse' or self.equation == 'stdse') and self.initial_state == 'extremely_skewed':
            for i in range(self.medium.size[0]):
                k0 = self.medium.klist3d[i, 0, 0]
                kk = self.medium.klist1d[k0, 0]
                if kk > 0.0 and kk < 0.5:
                    self.state[k0, nv, nv - 1] = 1.0
                    self.state[k0, nv - 1, nv - 1] = 0.0

        elif self.equation == 'lvn' and self.initial_state == 'thermal_skewed':
            for i in range(self.medium.size[0]):
                k0 = self.medium.klist3d[i, 0, 0]
                kk = self.medium.klist1d[k0, 0]
                if kk > 0.0 and kk < 0.5:
                    self.state[k0, nv, nv] = (0.3 * np.sin(2 * np.pi * kk)) ** 2
                    self.state[k0, nv - 1, nv - 1] = 1.0 - self.state[k0, nv, nv]

        elif self.equation == 'lvn' and self.initial_state == 'partially_thermal_skewed':
            for i in range(self.medium.size[0]):
                k0 = self.medium.klist3d[i, 0, 0]
                kk = self.medium.klist1d[k0, 0]
                if kk > 0.0 and kk < 0.5:
                    self.state[k0, nv, nv] = (0.3 * np.sin(2 * np.pi * kk)) ** 2
                    self.state[k0, nv - 1, nv - 1] = 1.0 - self.state[k0, nv, nv]
                    self.state[k0, nv - 1, nv] = 0.25 * np.sqrt(self.state[k0, nv - 1, nv - 1] * self.state[k0, nv, nv])
                    self.state[k0, nv, nv - 1] = self.state[k0, nv - 1, nv].conj()

        elif self.equation == 'lvn' and self.initial_state == 'partially_filled':
            for i in range(self.medium.size[0]):
                k0 = self.medium.klist3d[i, 0, 0]
                kk = self.medium.klist1d[k0, 0]
                if kk > 0.0 and kk < 0.5:
                    self.state[k0, nv - 1, nv - 1] = 1.0 - (0.3 * np.sin(2 * np.pi * kk)) ** 2

        elif self.equation == 'lvn' and self.initial_state == 'gaussian':
            for i in range(self.medium.size[0]):
                k0 = self.medium.klist3d[i, 0, 0]
                kk = self.medium.klist1d[k0, 0]
                if kk > 0.5:
                    kk -= 1.0
                self.state[k0, nv - 1, nv - 1] = 0.3 * np.exp(-6 * np.sin(kk * np.pi) ** 2)

        elif (self.equation == 'tdse' or self.equation == 'stdse') and self.initial_state == 'gaussian':
            for i in range(self.medium.size[0]):
                k0 = self.medium.klist3d[i, 0, 0]
                kk = self.medium.klist1d[k0, 0]
                if kk > 0.5:
                    kk -= 1.0
                self.state[k0, nv - 1, nv - 1] = 0.3 * np.exp(-6 * np.cos(kk * np.pi) ** 2)

        elif self.equation == 'lvn' and self.initial_state == 'gaussians':
            for i in range(self.medium.size[0]):
                k0 = self.medium.klist3d[i, 0, 0]
                kk = self.medium.klist1d[k0, 0]
                if kk > 0.5:
                    kk -= 1.0
                self.state[k0, nv - 1, nv - 1] = 0.3 * np.exp(-4 * np.sin(np.pi * kk) ** 2)
                self.state[k0, nv, nv] = 0.1 * np.exp(-6 * np.sin(
                    np.pi * kk) ** 2)  # + 0.2*np.exp(-4*np.sin(np.pi*kk)**2)*abs(self.medium.momentum[k0,0,1,0])
                # self.state[k0,nv-1,nv-1] = 0.0
                # self.state[k0,nv-1,nv-1] += 0.2*np.exp(-4*np.sin(np.pi*kk)**2)*abs(self.medium.momentum[k0,0,1,0])
                # self.state[k0,nv,nv]     += 0.2*np.exp(-4*np.sin(np.pi*kk)**2)*abs(self.medium.momentum[k0,0,1,0])
                # self.state[k0,nv-1,nv] =    0.2*np.exp(-4*np.sin(np.pi*kk)**2)*self.medium.momentum[k0,0,1,0]
                # self.state[k0,nv,nv-1] =    0.2*np.exp(-4*np.sin(np.pi*kk)**2)*self.medium.momentum[k0,1,0,0]

        elif (self.equation == 'tdse' or self.equation == 'stdse') and self.initial_state == 'gaussians':
            for i in range(self.medium.size[0]):
                k0 = self.medium.klist3d[i, 0, 0]
                kk = self.medium.klist1d[k0, 0]
                if kk > 0.5:
                    kk -= 1.0
                self.state[k0, nv - 2, nv - 2] = np.sqrt(0.3 * np.exp(-4 * np.cos(np.pi * kk) ** 2))
                self.state[k0, nv - 1, nv - 1] = np.sqrt(0.1 * np.exp(-6 * np.cos(np.pi * kk) ** 2))

        elif self.equation == 'lvn' and self.initial_state == 'gaussians_partially_coherent':
            for i in range(self.medium.size[0]):
                k0 = self.medium.klist3d[i, 0, 0]
                kk = self.medium.klist1d[k0, 0]
                if kk > 0.5:
                    kk -= 1.0
                self.state[k0, nv - 1, nv - 1] = 0.3 * np.exp(-4 * np.sin(np.pi * kk) ** 2)
                self.state[k0, nv, nv] = 0.1 * np.exp(-6 * np.sin(np.pi * kk) ** 2)
                self.state[k0, nv - 1, nv] = 0.2 * np.sqrt(self.state[k0, nv - 1, nv - 1] * self.state[k0, nv, nv])
                self.state[k0, nv, nv - 1] = 0.2 * np.sqrt(self.state[k0, nv - 1, nv - 1] * self.state[k0, nv, nv])



        elif self.equation == 'lvn' and self.initial_state == 'gaussians_coherent':
            for i in range(self.medium.size[0]):
                k0 = self.medium.klist3d[i, 0, 0]
                kk = self.medium.klist1d[k0, 0]
                if kk > 0.5:
                    kk -= 1.0
                self.state[k0, nv - 1, nv - 1] = 0.3 * np.exp(-4 * (kk) ** 2)
                self.state[k0, nv, nv] = 0.1 * np.exp(-6 * (kk) ** 2)
                self.state[k0, nv - 1, nv] = np.sqrt(self.state[k0, nv - 1, nv - 1] * self.state[k0, nv, nv])

        elif self.equation == 'lvn' and self.initial_state == 'coherent':
            for i in range(self.medium.size[0]):
                k0 = self.medium.klist3d[i, 0, 0]
                kk = self.medium.klist1d[k0, 0]
                if kk > 0.5:
                    kk -= 1.0
                self.state[k0, nv - 1, nv - 1] = 1 - 0.3 * (np.sin(np.pi * kk) ** 2)
                self.state[k0, nv, nv] = 0.3 * (np.sin(np.pi * kk) ** 2)
                self.state[k0, nv - 1, nv] = np.sqrt(self.state[k0, nv - 1, nv - 1] * self.state[k0, nv, nv]) * np.exp(
                    1j * np.angle(self.medium.momentum[k0, nv - 1, nv, 0]))
                self.state[k0, nv, nv - 1] = np.sqrt(self.state[k0, nv - 1, nv - 1] * self.state[k0, nv, nv]) * np.exp(
                    -1j * np.angle(self.medium.momentum[k0, nv - 1, nv, 0]))

        elif (self.equation == 'tdse' or self.equation == 'stdse') and self.initial_state == 'coherent':
            for i in range(self.medium.size[0]):
                k0 = self.medium.klist3d[i, 0, 0]
                kk = self.medium.klist1d[k0, 0]
                if kk > 0.5:
                    kk -= 1.0
                magnitude = 0.3 * np.exp(-20 * (np.cos(np.pi * kk) ** 2))
                self.state[k0, nv - 1, nv - 1] = np.sqrt(1 - magnitude)
                self.state[k0, nv, nv - 1] = np.sqrt(magnitude) * np.exp(
                    1j * np.angle(self.medium.momentum[k0, nv - 1, nv, 0]))

        elif self.equation == 'lvn' and self.initial_state == 'filled_partially_coherent':
            for i in range(self.medium.size[0]):
                k0 = self.medium.klist3d[i, 0, 0]
                kk = self.medium.klist1d[k0, 0]
                if kk > 0.5:
                    kk -= 1.0
                self.state[k0, nv - 1, nv - 1] = 0.5
                self.state[k0, nv, nv] = 0.5
                self.state[k0, nv - 1, nv] = (0.1 + 0.1 * (np.sin(np.pi * kk) ** 2)) * np.exp(
                    1j * np.angle(self.medium.momentum[k0, nv - 1, nv, 0]))
                self.state[k0, nv, nv - 1] = (0.1 + 0.1 * (np.sin(np.pi * kk) ** 2)) * np.exp(
                    -1j * np.angle(self.medium.momentum[k0, nv - 1, nv, 0]))

        return self.state
