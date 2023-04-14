from numpy import pi, log

# QCD group constants
NC = 3.
TR = 1./2.
CA = NC
CF = (NC*NC-1.)/(2.*NC)

class AlphaS:
    """
    Calculator for the strong coupling \alpha_s, implementing the running at
    the 0th and the 1st (=default) perturbative order.
    The starting point of the evolution is assumed to be \alpha_s(\m_Z**2).

    Examples usage, printing the strong coupling value at a scale of
    100 GeV**2:

      alphas = AlphaS(91.8**2, 0.118)
      print(alphas(100))

    """

    def __init__(self,squared_m_Z,alphas_at_squared_m_Z,perturbative_order=1,squared_m_b=4.75**2,squared_m_c=1.3**2):
        self.order = perturbative_order
        self.mc2 = squared_m_c
        self.mb2 = squared_m_b
        self.mz2 = squared_m_Z
        self.asmz = alphas_at_squared_m_Z
        self.asmb = self(self.mb2)
        self.asmc = self(self.mc2)

    def beta0(self,n_light_flavours):
        return 11./6.*CA-2./3.*TR*n_light_flavours

    def beta1(self,n_light_flavours):
        return 17./6.*CA*CA-(5./3.*CA+CF)*TR*n_light_flavours

    def as0(self,t):
        if t >= self.mb2:
            tref = self.mz2
            asref = self.asmz
            b0 = self.beta0(5)/(2.*pi)
        elif t >= self.mc2:
            tref = self.mb2
            asref = self.asmb
            b0 = self.beta0(4)/(2.*pi)
        else:
            tref = self.mc2
            asref = self.asmc
            b0 = self.beta0(3)/(2.*pi)
        return 1./(1./asref+b0*log(t/tref))

    def as1(self,t):
        if t >= self.mb2:
            tref = self.mz2
            asref = self.asmz
            b0 = self.beta0(5)/(2.*pi)
            b1 = self.beta1(5)/pow(2.*pi,2)
        elif t >= self.mc2:
            tref = self.mb2
            asref = self.asmb
            b0 = self.beta0(4)/(2.*pi)
            b1 = self.beta1(4)/pow(2.*pi,2)
        else:
            tref = self.mc2
            asref = self.asmc
            b0 = self.beta0(3)/(2.*pi)
            b1 = self.beta1(3)/pow(2.*pi,2)
        w = 1.+b0*asref*log(t/tref)
        return asref/w*(1.-b1/b0*asref*log(w)/w)

    def __call__(self,t):
        if self.order == 0: return self.as0(t)
        return self.as1(t)
