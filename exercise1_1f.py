import numpy as np
import matplotlib.pyplot as plt
from numba import njit, jit
import scipy as sci
import vegas

# ################### CONSTANTS #################
alpha = 1.0 / 129.0     # QED coupling
alpha_s = 0.118         # QCD coupling at the Z mass scale
M_Z = 91.2              # Z boson mass
Gamma_Z = 2.5           # Z boson decay width
sin2theta_w = 0.223     # square sine of the Weinberg angle
Qe = -1                 # electric charge of the electron
Quc = 2.0 / 3.0         # electric charge of (light) up-type quarks
Qdsb = -1.0 / 3.0       # electric charge of down-type quarks
T3uc = 1.0 / 2.0        # weak isospin of (light) up-type quarks
T3edsb = -1.0 / 2.0     # weak isospin of down-type quarks and electron
N_q = 5                 # number of light quark flavours
N_C = 3                 # number of QCD colours
f_conv = 3.89379656e8   # phyical unit conversion factor

# Beam energy fixed
s = M_Z ** 2            # Beam energy

# prefactor
kappa = 1.0 / (4.0 * sin2theta_w * (1.0 - sin2theta_w))

# integration domain
phimin = 0
phimax = 2.0 * np.pi
cosTmax = 1
cosTmin = -1
smin = (M_Z - 3 * Gamma_Z) ** 2
smax = (M_Z + 3 * Gamma_Z) ** 2

# sampling domain for importance sampling
rhomin = np.arctan((smin - M_Z ** 2) / (M_Z * Gamma_Z))
rhomax = np.arctan((smax - M_Z ** 2) / (M_Z * Gamma_Z))

#  ############################## setting seed for RNG ######################################
np.random.seed(0)


def Chi_1(s: float):
    """Photon-Z intereference

    Args:
        s (float): beam energy

    Returns:
        float: Photon-Z intereference
    """
    return kappa * s * (s - M_Z ** 2) / ((s - M_Z ** 2) ** 2 + Gamma_Z ** 2 * M_Z ** 2)


def Chi_2(s: float):
    """Resonance from Z amplitude

    Args:
        s (float): beam energy

    Returns:
        float: resonance of Z amplitude
    """
    return kappa ** 2 * s ** 2 / ((s - M_Z ** 2) ** 2 + Gamma_Z ** 2 * M_Z ** 2)


def M_qqbar(s: float, cosTheta: float, phi: float, q: int):
    """ Matrix element of a 2->2 scattering e-e+ -> q qbar

    Args:
        s (float): Beam energy
        cosTheta (float): cosine of scattering angle
        phi (float): _description_
        q (int): quark flavour
    """
    # constants
    Q = 0
    A = 0
    Ae = T3edsb
    Vq = 0
    Ve = Ae - 2.0 * Qe * sin2theta_w

    # up or charm quark
    if q == 2 or q == 4:
        Q = Quc
        A = T3uc
    else:
        Q = Qdsb
        A = T3edsb

    Vq = A - 2.0 * Q * sin2theta_w

    M2 = (4 * np.pi * alpha) ** 2 * N_C * ((1 + cosTheta ** 2) * (Qe ** 2 * Q ** 2 + 2 * Qe * Q * Ve * Vq * Chi_1(s) + (Ae ** 2 +
                                                                                                                        Ve ** 2) * (A ** 2 + Vq ** 2) * Chi_2(s)) + cosTheta * (4 * Qe * Q * Ae * A * Chi_1(s) + 8 * Ae * Ve * A * Vq * Chi_2(s)))

    return M2

def BreitWigner(s: float):
    """Breit-Wigner function

    Args:
        s (float): beam energy

    Returns:
        float: Breit-Wigner function given beam energy s
    """
    return 1.0 / ((s - M_Z ** 2) ** 2 + M_Z ** 2 * Gamma_Z ** 2)


def Integrator_sum(N: int, free_s=False):
    """Monte-Carlo integrator for a 2->2 differential cross-section with fixed q

    Args:
        N (int): number of integration points
        free_s (bool, optional): True if s to be drawn from distribution. Defaults to False.

    Returns:
        list: mean and variance of integration
    """
    # integrator that sums up each contribution individually

    # constants
    beam_energy = s

    M = []

    # sample
    for n in range(N):
        # is s fixed?
        if free_s:
            beam_energy = np.random.uniform(low=rhomin, high=rhomax)

        cosT = np.random.uniform(low=cosTmin, high=cosTmax)
        phi = np.random.uniform(low=phimin, high=phimax)
        m = 0
        # loop over all quark flavour contributions
        for q in range(1, 1 + N_q):

            # calculating cross-section for MC point
            m += M_qqbar(beam_energy, cosT, phi, q)

        m *= f_conv * 1.0 / (8.0 * np.pi * 4.0 * np.pi *
                             2 * s)

        M.append(m)

    M = np.array(M)
    m_mean = 0
    dev = 0
    if free_s:
        # have to divide
        m_mean = 1.0 / (N) * np.sum(M)
        dev = np.sqrt((1.0 / N * np.sum(M ** 2) - m_mean ** 2) / N)
        m_mean *= (smax - smin)
        dev *= (smax - smin)
    else:
        m_mean = 1.0 / (N) * np.sum(M)
        dev = np.sqrt((1.0 / N * np.sum(M ** 2) - m_mean ** 2) / N)

    m_mean *= (cosTmax - cosTmin) * (phimax - phimin)
    dev *= (cosTmax - cosTmin) * (phimax - phimin)

    return m_mean, dev


def Integrator_pick(N: int, free_s=False):
    """Monte-Carlo integrator for a 2->2 differential cross-section with random quark flavour

    Args:
        N (int): number of integration points
        free_s (bool, optional): True if s to be drawn from distribution. Defaults to False.

    Returns:
        list: mean and variance of integration
    """
    # integrator with random flavour

    # constants
    beam_energy = s

    M = []

    # sample for each quark flavour
    for n in range(N):
        q = np.random.randint(low=1, high=6)
        # is s fixed?
        if free_s:
            beam_energy = np.random.uniform(low=rhomin, high=rhomax)
            beam_energy = M_Z * Gamma_Z * np.tan(beam_energy) + M_Z ** 2
            #print(beam_energy, smin, smax)

        cosT = np.random.uniform(low=cosTmin, high=cosTmax)
        phi = np.random.uniform(low=phimin, high=phimax)

        # calculating cross-section for MC point
        m = N_q * f_conv * 1.0 / (8.0 * np.pi * 4.0 * np.pi * 2.0 * s) * \
            M_qqbar(beam_energy, cosT, phi, q) / BreitWigner(beam_energy) * (rhomax - rhomin) / (M_Z * Gamma_Z) / (smax - smin)

        M.append(m)

    M = np.array(M)
    m_mean = 0
    dev = 0
    if free_s:
        # have to divide
        m_mean = 1.0 / (N) * np.sum(M)
        dev = np.sqrt((1.0 / N * np.sum(M ** 2) - m_mean ** 2) / N)
        m_mean *= (smax - smin)
        dev *= (smax - smin)
    else:
        m_mean = 1.0 / (N) * np.sum(M)
        dev = np.sqrt((1.0 / N * np.sum(M ** 2) - m_mean ** 2) / N)

    m_mean *= (cosTmax - cosTmin) * (phimax - phimin)
    dev *= (cosTmax - cosTmin) * (phimax - phimin)

    return m_mean, dev


print("RNG Test:       ", np.random.rand(5),
      "\nSupposed to be   [0.5488135  0.71518937 0.60276338 0.54488318 0.4236548 ]")


N = np.array([10, 100, 1000, 10000])

I = []

for n in N:
    I.append(Integrator_pick(n)[1])


fig = plt.figure()

plt.plot(N, I, "r-x", label="MC integrator")
plt.plot(N, 13000*np.sqrt(1/N), "k--", label=r"$\sim N^{-1/2}$")
plt.yscale("log")
plt.xscale("log")
plt.xlabel(r"$N$")
plt.ylabel(r"$\mathrm{var}(\sigma)$")
plt.grid()
plt.legend()
plt.show()
#plt.savefig("ex1_1_1.pdf", dpi=500, bbox_inches="tight")


N = 10000
I = Integrator_pick(N, True)

print("Using N=%d we get a result of sigma=%.2f +- %.2f" % (N, I[0], I[1]))
