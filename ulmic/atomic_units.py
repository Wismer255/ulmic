

class AtomicUnits:
    eV = 27.211385
    fs = 0.02418884326505
    nm = 0.052917721
    A = 0.52917721  # Angstrom
    AA = 0.52917721  # Angstrom
    VA = 51.4220652  # V/Angstrom
    mA = (1.602e-4 / fs) * 1e3  # miliampere  (1D)
    uA = (1.602e-4 / fs) * 1e6  # microampere (1D)
    eV2au = 1 / eV
    fs2au = 1 / fs
    nm2au = 1 / nm
    VA2au = 1 / VA
    c = 137.035999

    def __init__(self,):
        self.eV = 27.211385
        self.fs = 0.02418884326505
        self.nm = 0.052917721
        self.A = 0.52917721                 #Angstrom
        self.AA = 0.52917721                #Angstrom
        self.VA = 51.4220652                #V/Angstrom

        self.mA = (1.602e-4/self.fs)*1e3    #miliampere  (1D)
        self.uA = (1.602e-4/self.fs)*1e6    #microampere (1D)

        self.eV2au = 1/self.eV
        self.fs2au = 1/self.fs
        self.nm2au = 1/self.nm
        self.VA2au = 1/self.VA

        self.c = 137.035999
