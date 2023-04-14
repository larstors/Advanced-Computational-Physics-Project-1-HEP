import random
from numpy import pi, sqrt, tan, arctan, log, cos, sin

from utils.vector import Vec4
from utils.particle import Particle, check_event

# QCD group constants
NC = 3.
TR = 1./2.
CA = NC
CF = (NC*NC-1.)/(2.*NC)

class Kernel:
    """
    Abstract base class for calculating a given 1->2 splitting.
    """

    def __init__(self,ptcl_nums):
        self.ptcl_nums = ptcl_nums

class Pqq (Kernel):
    """
    Calculator for a q->qg splitting.
    """

    def Value(self,z,y):
        return CF*(2./(1.-z*(1.-y))-(1.+z))

    def Estimate(self,z):
        return CF*2./(1.-z)

    def Integral(self,zm,zp):
        return CF*2.*log((1.-zm)/(1.-zp))

    def GenerateZ(self,zm,zp):
        return 1.+(zp-1.)*pow((1.-zm)/(1.-zp),random.random())

class Pgg (Kernel):
    """
    Calculator for a g->gg splitting.
    """

    def Value(self,z,y):
        return CA/2.*(2./(1.-z*(1.-y))-2.+z*(1.-z))

    def Estimate(self,z):
        return CA/(1.-z)

    def Integral(self,zm,zp):
        return CA*log((1.-zm)/(1.-zp))

    def GenerateZ(self,zm,zp):
        return 1.+(zp-1.)*pow((1.-zm)/(1.-zp),random.random())

class Pgq (Kernel):
    """
    Calculator for a g->qq splitting.
    """

    def Value(self,z,y):
        return TR/2.*(1.-2.*z*(1.-z))

    def Estimate(self,z):
        return TR/2.

    def Integral(self,zm,zp):
        return TR/2.*(zp-zm)

    def GenerateZ(self,zm,zp):
        return zm+(zp-zm)*random.random()

class Shower:
    """
    A simple shower cascade simulator.
    """

    def __init__(self,alphas,t0=1.0):
        """Initializes the shower and its splitting kernels, given a AlphaS
        strong coupling instance `alphas` and a lower cut-off scale `t0`."""
        self.t0 = t0
        self.alphas = alphas
        self.alphas_max = alphas(self.t0)
        # set up q->qg splitting kernels
        self.kernels = [ Pqq([fl,fl,21]) for fl in [-5,-4,-3,-2,-1,1,2,3,4,5] ]
        # set up g->qq splitting kernels
        self.kernels += [ Pgq([21,fl,-fl]) for fl in [1,2,3,4,5] ]
        # set up g->gg splitting kernels
        self.kernels += [ Pgg([21,21,21]) ]

    def make_kinematics(self,z,y,phi,pijt,pkt):
        """Calculate the two momenta of the daughters after a splitting, and
        the one momentum of the `spectator`, taking the recoil of the
        splitting.

        Returns the momenta in a list in that order.
        """
        Q = pijt+pkt
        rkt = sqrt(Q.invariant_mass_squared()*y*z*(1.-z))
        kt1 = pijt.cross_product(pkt)
        if kt1.length_3d() < 1.e-6: kt1 = pijt.cross_product(Vec4(0.,1.,0.,0.))
        kt1 *= rkt*cos(phi)/kt1.length_3d()
        kt2cms = Q.boost(pijt).cross_product(kt1)
        kt2cms *= rkt*sin(phi)/kt2cms.length_3d()
        kt2 = Q.boost_back(kt2cms)
        pi = z*pijt + (1.-z)*y*pkt + kt1 + kt2
        pj = (1.-z)*pijt + z*y*pkt - kt1 - kt2
        pk = (1.-y)*pkt
        return [pi,pj,pk]

    def make_colors(self,ptcl_nums,colij,colk):
        """Assign new colors after a splitting, using the leading color
        approximation.

        Returns a list of two color pairs, one for each of the daughters of the
        splitting.
        """
        self.current_color_index += 1
        if ptcl_nums[0] != 21:
            if ptcl_nums[0] > 0:
                return [ [self.current_color_index,0], [colij[0],self.current_color_index] ]
            else:
                return [ [0,self.current_color_index], [self.current_color_index,colij[1]] ]
        else:
            if ptcl_nums[1] == 21:
                if colij[0] == colk[1]:
                    if colij[1] == colk[0] and random.random()>0.5:
                        return [ [colij[0],self.current_color_index], [self.current_color_index,colij[1]] ]
                    return [ [self.current_color_index,colij[1]], [colij[0],self.current_color_index] ]
                else:
                    return [ [colij[0],self.current_color_index], [self.current_color_index,colij[1]] ]
            else:
                if ptcl_nums[1] > 0:
                    return [ [colij[0],0], [0,colij[1]] ]
                else:
                    return [ [0,colij[1]], [colij[0],0] ]

    def generate_next_emission(self,event):
        """Generate the next emission starting from the current scale `self.t`,
        using the Sudakov veto algorithm. The passed event (= list of Particle instances)
        is modified in-place, if a splitting occurs."""
        while self.t > self.t0:
            t = self.t0
            for split in event[2:]:
                for spect in event[2:]:
                    if spect == split: continue
                    if not split.is_color_connected(spect): continue
                    for sf in self.kernels:
                        if sf.ptcl_nums[0] != split.pid: continue
                        m2 = (split.mom+spect.mom).invariant_mass_squared()
                        if m2 < 4.*self.t0: continue
                        zp = .5*(1.+sqrt(1.-4.*self.t0/m2))
                        g = self.alphas_max/(2.*pi)*sf.Integral(1.-zp,zp)
                        tt = self.t*pow(random.random(),1./g)
                        if tt > t:
                            t = tt
                            s = [ split, spect, sf, m2, zp ]
            self.t = t
            if t > self.t0:
                z = s[2].GenerateZ(1.-s[4],s[4])
                y = t/s[3]/z/(1.-z)
                if y < 1.:
                    f = (1.-y)*self.alphas(t)*s[2].Value(z,y)
                    g = self.alphas_max*s[2].Estimate(z)
                    if f/g > random.random():
                        phi = 2.*pi*random.random()
                        moms = self.make_kinematics(z,y,phi,s[0].mom,s[1].mom)
                        cols = self.make_colors(s[2].ptcl_nums,s[0].color,s[1].color)
                        event.append(Particle(s[2].ptcl_nums[2],moms[1],cols[1]))
                        s[0].set(s[2].ptcl_nums[1],moms[0],cols[0])
                        s[1].mom = moms[2]
                        return
    
    def run(self,event,t):
        """Runs the shower on a given event (= list of Particle instances),
        starting from the scale `t`. The event is modified in-place to take
        into account the occurring emissions (if any).

        It is assumed that the first two particles in the event are the
        incoming particles.  Since this is a final-state shower only, they are
        ignored (but assumed to be present in the list).
        """
        self.current_color_index = 1
        # generate emissions as long as we are above the cut-off scale `t0`
        self.t = t
        while self.t > self.t0:
            self.generate_next_emission(event)
