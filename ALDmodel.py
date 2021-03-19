"""
ALD model

Basic ALD model considering the interaction of two precursors with
a growing surface.

All units are in SI unless specifically mentioned:

  - M: molecular mass, atomic mass units
  - T: temperature, Kelvin
  - u: flow velocity, m/s
  - p: precursor pressure, Pa
  - L: radios of reactor height, m
  - sitearea: m^2
  - tp: time to purge: s
  - ald dose and purge times: s

"""

import numpy as np

kb = 1.38e-23
amu = 1.660e-27


def vth(M, T):
    return np.sqrt(8*kb*T/(np.pi*amu*M))



class ALDGrowth:
    """Implement a 0D ALD process

    Implement a simple 0D model of an ALD process, providing the evolution
    of coverage and thickness as a function of ALD cycles.

    """

    def __init__(self, T, sitearea, chem1, chem2, noise=0.0):
        """
        Parameters
        ----------
        T : float
            temperature in K
        sitearea : float
            area of a surface site, in m^2
        chem1, chem2 : tuples
            tuples containing the chemical paramters of
            the two precursors Each is a five element tuple:

                (p, M, beta, tp, dm)

            where p is the precursor pressure in Pa, M
            is the molecular mass in atomic mass units,
            beta is the sticking probability, tp is
            a characteristic time for precursor evacuation in s,
            and dm is the mass change
            for that particular half-cycle (ng/cm2).
        """

        p1, M1, beta1, tp1, dm1 = chem1
        p2, M2, beta2, tp2, dm2 = chem2

        v1 = vth(M1, T)
        v2 = vth(M2, T)

        self.a01 = sitearea*beta1*0.25*v1*p1/(kb*T)
        self.a02 = sitearea*beta2*0.25*v2*p2/(kb*T)

        dt = 10**(-int(np.log10(max(self.a01, self.a02))+1))

        self.tp1 = tp1
        self.tp2 = tp2
        self.dm1 = dm1
        self.dm2 = dm2

        self.a1 = 0
        self.a2 = 0
        self.c1 = 0

        self.noise = noise

        self.dt = dt
        self.dp1 = np.exp(-self.dt/self.tp1)
        self.dp2 = np.exp(-self.dt/self.tp2)
        self.glist = []

    def next_step(self, c1, a1, a2, dt):
        return (c1+a1*dt)/(1 + (a1+a2)*dt)

    def cycle(self, t1, t2, t3, t4, N=10):

        a1corr = self.a01*t1/(t1+self.tp1)
        a2corr = self.a02*t3/(t3+self.tp2)

        nd1 = int(t1/self.dt)
        nd2 = int(t3/self.dt)
        np1 = int(t2/self.dt)
        np2 = int(t4/self.dt)
        nc = nd1 + np1 + nd2 + np2
        gold = 0

        a1 = self.a1
        a2 = self.a2
        c1 = self.c1

        growthrates = []

        for nc in range(N):
            a1 = a1corr
            J1 = 0
            J2 = 0
            for i in range(nd1):
                a2 = self.dp2*a2
                c1 = self.next_step(c1, a1, a2, self.dt)
                J1 += a1*(1-c1)
                J2 += a2*c1

            for i in range(np1):
                a2 = self.dp2*a2
                a1 = self.dp1*a1
                c1 = self.next_step(c1, a1, a2, self.dt)
                J1 += a1*(1-c1)
                J2 += a2*c1

            a2 = a2corr
            for i in range(nd2):
                a1 = self.dp1*a1
                c1 = self.next_step(c1, a1, a2, self.dt)
                J1 += a1*(1-c1)
                J2 += a2*c1

            for i in range(np2):
                a2 = self.dp2*a2
                a1 = self.dp1*a1
                c1 = self.next_step(c1, a1, a2, self.dt)
                J1 += a1*(1-c1)
                J2 += a2*c1

            gr = self.dt*(self.dm1*J1+self.dm2*J2)
            growthrates.append(gr+self.noise*np.random.normal())
        self.c1 = c1
        self.glist.extend(growthrates)

        return growthrates


if __name__ == '__main__':

    chem1 = (10, 100, 1e-2, 0.1, 40)
    chem2 = (100, 18, 1e-2, 0.1, -10)

    ald = ALDGrowth(473, 10e-20, chem1, chem2)
    gr = ald.cycle(1,5,1,5,10)
    print(gr)
