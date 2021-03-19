from ALDmodel import ALDGrowth

import numpy as np


class ALDOpt:
    """Physics-inpired algorithm for ALD process optimization

    Parameters
    ----------

    ep0 : float
        Relative tolerance for growth per cycle variations

    Nop : int
        Maximum number of consecutive dose/purge optimizations

    Nav : int
        Number of cycles used to extract average growth per cycle

    Ndrop : int
        Number of cycles dropped to compute growth statistics

    Nmax : int
        Maximum number of dose or purge minimization steps

    tmin : float
        Minimum dose or purge time in seconds

    dtmax : float
        Maximum dose or purge increment in seconds

    nthres : float
        Maximum value of the normalized standard deviation in growth
        per cycle allowed to compute statistics

    verbose : bool
        If true, prints out information after each substep of the algorithm

    """

    def __init__(self, ep0=0.02, Nop=10, Nav=10, Ndrop=2, Nmax=50,
        tmin=0.1, dtmax=3, dto=0.2, dtm=0.1, nthres=0.2, verbose=False):

            self.ep0 = ep0
            self.Nop = Nop
            self.Nav = Nav
            self.Ndrop = Ndrop
            self.Nmax = Nmax
            self.tmin = tmin
            self.dtmax = dtmax
            self.dto = dto
            self.dtm = dtm
            self.nthres = nthres
            self.verbose = verbose


    def optimize(self, ald, tc):

        t1, t2, t3, t4 = tc
        g = ald.cycle(t1, t2, t3, t4, self.Nav)
        gr, std = self.growth_stats(g)
        self.history = [[t1, t2, t3, t4, gr, std]]
        epd = 1
        epp = 1
        grold = gr
        ntc = tc
        while epd > self.ep0 or epp > self.ep0:

            ntc, gr = self.optimize_purges(ald, ntc, gr)
            if self.verbose:
                print("Purge:", ntc, gr)
            epp = abs(grold-gr)/grold
            ntc, gr2 = self.optimize_doses(ald, ntc, gr)
            if self.verbose:
                print("Dose:", ntc, gr2)
            epd = abs(gr2-gr)/gr
            grold = gr2

        ntc, gr = self.minimize_doses(ald, ntc, gr)
        if self.verbose:
            print("Minimizing dose:", ntc, gr)
        ntc, gr = self.minimize_purges(ald, ntc, gr)
        if self.verbose:
            print("Minimizing purges:", ntc, gr)
        return ntc, gr, np.array(self.history)

    def growth_stats(self, gr):
        return  np.mean(gr[self.Ndrop:]), np.std(gr[self.Ndrop:])


    def optimize_purges(self, ald, tc, gr):
        """
        Subset of the optimization algorithm in charge of purge time optimization
        """

        t1, t2, t3, t4 = tc
        dt2 = self.dto
        dt4 = self.dto

        while True:
            dt = min(t2*dt2, self.dtmax)
            ttest = t2 + dt
            g = ald.cycle(t1, ttest, t3, t4, self.Nav)
            gr2, sd2 = self.growth_stats(g)
            ep2 = (gr2-gr)/gr
            ep = max(3*sd2/gr2, self.ep0)
            self.history.append([t1, ttest, t3, t4, gr2, sd2])
            if ep > self.nthres or abs(ep2) < ep:
                break
            else:
                t2 = ttest
                gr = gr2

        while True:
            dt = min(t4*dt4, self.dtmax)
            ttest = t4 + dt
            g = ald.cycle(t1, t2, t3, ttest, self.Nav)
            gr4, sd4 = self.growth_stats(g)
            self.history.append([t1, t2, t3, ttest, gr4, sd4])
            ep4 = abs(gr4-gr)/gr
            ep = max(3*sd4/gr4,self.ep0)
            if ep > self.nthres or abs(ep4) < ep:
                break
            else:
                t4 = ttest
                gr = gr4

        return (t1, t2, t3, t4), gr


    def optimize_doses(self, ald, tc, gr):
        """Subset  of the  algorithm in charge of dose optimization
        """

        t1, t2, t3, t4 = tc
        dt1 = self.dto
        dt3 = self.dto
        iop = 0
        while iop < self.Nop:
            iop += 1
            dt = min(t1*dt1, self.dtmax)
            ttest = t1+dt
            g = ald.cycle(ttest, t2, t3, t4, self.Nav)
            gr1, sd1 = self.growth_stats(g)
            ep = max(self.ep0, 3*sd1/gr1)
            self.history.append([ttest, t2, t3, t4, gr1, sd1])
            if gr1 < 3*sd1:
                continue
                t1 = ttest
                gr = gr1
            else:
                ep1 = (gr1-gr)/gr
                if abs(ep1) < ep:
                    break
                else:
                    t1 = ttest
                    gr = gr1

        gr = gr1
        iop = 0

        while iop < self.Nop:
            dt = min(t3*dt3, self.dtmax)
            ttest = t3+dt
            g = ald.cycle(t1, t2, ttest, t4, self.Nav)
            gr3, sd3 = self.growth_stats(g)
            ep = max(self.ep0, 3*sd3/gr3)
            self.history.append([t1, t2, ttest, t4, gr3, sd3])
            if gr3 < 3*sd3:
                t3 = ttest
                gr = gr3
                continue
            else:
                ep3 = (gr3-gr)/gr
                if abs(ep3) < ep:
                    break
                else:
                    t3 = ttest
                    gr = gr3

        gr = gr3

        return (t1, t2, t3, t4), gr


    def minimize_doses(self, ald, tc, gr):
        t1, t2, t3, t4 = tc
        dt1 = self.dtm
        dt3 = self.dtm
        ep1 = 0
        ep3 = 0
        iop = 0
        t1 = 1.2*t1
        g = ald.cycle(t1, t2, t3, t4, self.Nav)
        gr0, sd0 = self.growth_stats(g)
        self.history.append([t1, t2, t3, t4, gr0, sd0])

        while iop < self.Nop:
            iop += 1
            dt = max(t1*dt1/(1+dt1), self.tmin)
            ttest = t1-dt
            if ttest < self.tmin:
                break
            g = ald.cycle(ttest, t2, t3, t4, self.Nav)
            gr1, sd1 = self.growth_stats(g)
            self.history.append([ttest, t2, t3, t4, gr1, sd1])
            ep1 = abs(gr1-gr0)/gr0
            ep = max(self.ep0,3*sd1/gr1)
            if ep1 < ep:
                t1 = ttest
            else:
                break

        iop = 0
        t3 = 1.2*t3
        g = ald.cycle(t1, t2, t3, t4, self.Nav)
        gr0, sd0 = self.growth_stats(g)
        gr = gr0
        self.history.append([t1, t2, t3, t4, gr0, sd0])

        while iop < self.Nop:
            iop += 1
            dt = max(t3*dt3/(1+dt3),self.tmin)
            ttest = t3-dt
            if ttest < self.tmin:
                break
            g = ald.cycle(t1, t2, ttest, t4, self.Nav)
            gr3, sd3 = self.growth_stats(g)
            self.history.append([t1, t2, ttest, t4, gr3, sd3])
            ep3 = abs(gr3-gr0)/gr0
            ep = max(self.ep0,3*sd3/gr3)
            if ep3 < ep:
                t3 = ttest
                gr = gr3
            else:
                break

        return (t1, t2, t3, t4), gr

    def minimize_purges(self, ald, tc, gr):
        t1, t2, t3, t4 = tc
        dt2 = self.dtm
        dt4 = self.dtm
        ep2 = 0
        ep4 = 0

        iop = 0
        gr0 = gr
        while iop < self.Nop:
            iop += 1
            dt = max(dt2*t2/(1+dt2),self.tmin)
            ttest = t2 - dt
            if  ttest < self.tmin:
                break
            g = ald.cycle(t1, ttest, t3, t4, self.Nav)
            gr2, sd2 = self.growth_stats(g)
            self.history.append([t1, ttest, t3, t4, gr2, sd2])
            ep2 = abs(gr2-gr0)/gr0
            ep = max(3*sd2/gr2,self.ep0)
            if ep2 < ep:
                t2 = ttest
                gr = gr2
            else:
                break

        gr0 = gr
        iop = 0
        while iop < self.Nop:
            iop += 1
            dt = max(dt4*t4/(1+dt4),self.tmin)
            ttest = t4-dt
            if ttest < self.tmin:
                break
            g = ald.cycle(t1, t2, t3, ttest, self.Nav)
            gr4, sd4 = self.growth_stats(g)
            self.history.append([t1, t2, t3, ttest, gr4, sd4])
            ep4 = abs(gr4-gr0)/gr0
            ep = max(3*sd4/gr4,self.ep0)
            if ep4 < ep:
                t4 = ttest
                gr = gr4
            else:
                break

        return (t1, t2, t3, t4), gr




if __name__ == "__main__":

    tma200 = ((26.66, 72, 1e-3, 0.1, 1.0), (26.66, 18, 1e-4, 0.1, 0.0), 22.5e-20)
    ttip200 = ((0.665, 284, 1e-4,0.1, 1.0),  (26.66, 18, 1e-4, 0.1, 0.0), 117e-20)
    WF6 = ((6.665, 297, 0.2, 0.1, 1.0), (26.66, 62, 0.05, 0.1, 0), 3.6e-20)
    tma100 = ((26.66,72,1e-4,3, 1.0), (26.66, 18, 1e-5, 10, 0), 25.1e-20)



    chem1, chem2,s0 = tma200
    ald = ALDGrowth(473, s0, chem1, chem2, noise=0.01)

    tc = (1, 2, 1, 2)

#    timings, gr, cycles = optimize_ald(ald, tc, ep0=0.02, Ndose=4, Nav=10)
    opt = ALDOpt()

    timings, gr, cycles = opt.optimize(ald, tc)

    print(timings)
    print(gr)


    import matplotlib.pyplot as pt

    pt.plot(cycles[:,0], 'o', linestyle="-", label="dose 1")
    pt.plot(cycles[:,1], 'o', linestyle="-", label="purge 1")
    pt.plot(cycles[:,2], 'o', linestyle="-", label="dose 2")
    pt.plot(cycles[:,3], 'o', linestyle="-", label="purge 2")
    pt.plot(cycles[:,4], 'o', linestyle="-", label="growth")

    pt.xlabel("Optimization sequence")
    pt.ylabel("ALD times (s), Growth per cycle")
    pt.legend()
    pt.savefig("TMA200.png", dpi=300)
    pt.show()
