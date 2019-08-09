import numpy as np
import random

# Global Definition of Particle Masses and lifetimes
mPion = 140.
mB = 5300.
mD = 2000.
D_lifetime = 1.0e-12
B_lifetime = 1.5e-12  # check??

class Particle:
    '''Class to handle true particles for our toy simulations'''

    def __init__(self, m, phi, theta, magp, vtx=np.zeros(3)):
        '''
        use MeV as default units
        phi in range -pi to pi
        theta in range 0 to pi
        '''
        self.m = m
        self.phi = phi
        self.theta = theta
        self.magp = magp  # the three-momentum magnitude
        self.origin = vtx # defaults to (0,0,0)
        self.properLifetime = None # default to B lifetime in seconds

        self.unitDirection = self.calculateUnitDirection()
        self.px = magp * self.unitDirection[0]
        self.py = magp * self.unitDirection[1]
        self.pz = magp * self.unitDirection[2]
        self.betaMod = (magp / m) / np.sqrt(1 + (magp / m) ** 2)
        self.beta = self.betaMod * self.unitDirection
        self.gamma = 1 / np.sqrt(1 - (np.linalg.norm(self.beta)) ** 2)
        self.relE = np.sqrt(self.m ** 2 + self.magp ** 2)
        self.fourMom = np.array(([self.relE, self.px, self.py, self.pz]))  # contravariant form

    def setOrigin(self, origin):
        #print(" caution, modifying the particle origin")
        self.origin = origin

    def setProperLifetime(self, properLifetime):
        self.properLifetime = properLifetime

    def calculateUnitDirection(self):
        dirvec = np.array(([np.sin(self.theta) * np.cos(self.phi),
                            np.sin(self.theta) * np.sin(self.phi),
                            np.cos(self.theta)]))
        unitDir = dirvec / np.linalg.norm(dirvec)
        return unitDir

    def propagate(self):
        labLifetime = self.gamma * np.random.exponential(self.properLifetime)
        displacement = self.beta * 3e8 * labLifetime  # 3-dimensional position in metres
        return (self.origin + displacement)

    def propagate_and_decay(self, decay_mode):
        decay_vertex = self.propagate()
        decay_tool = DecayTool(self, decay_mode, decay_vertex)
        return decay_tool.propagate_and_decay()

    def propagateAndDecay(self, decay):
        decayVtx = self.propagate()
        if decay == "2pions":
            mD1 = 140.
            mD2 = 140.
            p1, p2 = random2DecayLabFrame(self, mD1, mD2)
            p1.setOrigin(decayVtx)
            p2.setOrigin(decayVtx)
            return p1, p2

        elif decay == "Dpion":
            mpion = 140.
            mD = 2000.
            p1, p2 = random2DecayLabFrame(self, mD, mpion)
            p1.setOrigin(decayVtx)
            p2.setOrigin(decayVtx)
            p1.setProperLifetime(D_lifetime) # D meson lifetime
            return p1, p2

        elif decay == "3pions":
            mpion = 140.
            p1, p2, p3 = random3DecayLabFrame(self, mpion, mpion, mpion)
            p1.setOrigin(decayVtx)
            p2.setOrigin(decayVtx)
            p3.setOrigin(decayVtx)
            return p1,p2,p3

        elif decay == "Dpionpion":
            mpion = 140.
            mD=2000.
            p1, p2, p3 = random3DecayLabFrame(self, mD, mpion, mpion)
            p1.setOrigin(decayVtx)
            p2.setOrigin(decayVtx)
            p3.setOrigin(decayVtx)
            p1.setProperLifetime(D_lifetime) # D meson lifetime
            return p1,p2,p3

        elif decay == "4pions":
            mpion=140.
            p1, p2, p3, p4 = random4DecayLabFrame(self, mpion, mpion, mpion, mpion)
            p1.setOrigin(decayVtx)
            p2.setOrigin(decayVtx)
            p3.setOrigin(decayVtx)
            p4.setOrigin(decayVtx)
            return p1, p2, p3, p4

        elif decay == "Dpionpionpion":
            mpion = 140.
            mD = 2000.
            p1, p2, p3, p4 = random4DecayLabFrame(self, mD, mpion, mpion, mpion)
            p1.setOrigin(decayVtx)
            p2.setOrigin(decayVtx)
            p3.setOrigin(decayVtx)
            p4.setOrigin(decayVtx)
            p1.setProperLifetime(D_lifetime) # D meson lifetime
            return p1, p2, p3, p4

        else:
            print("ERROR: Not a Valid Decay Mode")
            return None


########################################################################################################################
########################################################################################################################

class DecayTool:

    #decay_modes = { "2pi" : self.2pi_decay, "3pi" : self.3pi_decay, "4pi" : self.4pi_decay,
    #                   "D+pi" : self.D_pi_decay, "D+2pi" : self.D_2pi_decay,
    #                   "D+3pi" : self.D_3pi_decay }

    def __init__(self, parent, decay_mode, decay_vertex):
        self.parent = parent
        self.mode = decay_mode
        self.decay_vertex = decay_vertex

    def propagate_and_decay(self):
        """
        Here we select the appropriate function from the decay mode
        """
        if self.mode == "2pions":
            p1, p2 = random2DecayLabFrame(self.parent, mPion, mPion)  # maybe more readable if i specify the last two are daughters
            p1.setOrigin(self.decay_vertex)
            p2.setOrigin(self.decay_vertex)
            return p1, p2

        elif self.mode == "Dpion":

            p1, p2 = random2DecayLabFrame(self.parent, mD, mPion)
            p1.setOrigin(self.decay_vertex)
            p2.setOrigin(self.decay_vertex)
            p1.setProperLifetime(D_lifetime) # D meson lifetime
            return p1, p2

        elif self.mode == "3pions":
            p1, p2, p3 = random3DecayLabFrame(self.parent, mPion, mPion, mPion)
            p1.setOrigin(self.decay_vertex)
            p2.setOrigin(self.decay_vertex)
            p3.setOrigin(self.decay_vertex)
            return p1,p2,p3

        elif self.mode == "Dpionpion":
            p1, p2, p3 = random3DecayLabFrame(self.parent, mD, mPion, mPion)
            p1.setOrigin(self.decay_vertex)
            p2.setOrigin(self.decay_vertex)
            p3.setOrigin(self.decay_vertex)
            p1.setProperLifetime(D_lifetime) # D meson lifetime
            return p1,p2,p3

        elif self.mode == "4pions":
            p1, p2, p3, p4 = random4DecayLabFrame(self.parent, mPion, mPion, mPion, mPion)
            p1.setOrigin(self.decay_vertex)
            p2.setOrigin(self.decay_vertex)
            p3.setOrigin(self.decay_vertex)
            p4.setOrigin(self.decay_vertex)
            return p1, p2, p3, p4

        elif self.mode == "Dpionpionpion":
            p1, p2, p3, p4 = random4DecayLabFrame(self.parent, mD, mPion, mPion, mPion)
            p1.setOrigin(self.decay_vertex)
            p2.setOrigin(self.decay_vertex)
            p3.setOrigin(self.decay_vertex)
            p4.setOrigin(self.decay_vertex)
            p1.setProperLifetime(D_lifetime) # D meson lifetime
            return p1, p2, p3, p4

        else:
            print("ERROR: Not a Valid Decay Mode")
            return None

########################################################################################################################
########################################################################################################################

### Decay Functions:

def random2DecayLabFrame(parent, massDaughter1, massDaughter2):
    '''
    uses particle_utils to create daughters in ZMF then converts back to lab frame
    randomly chooses phi and theta, momentum constrained by conservation law
    '''

    parent_mass = parent.m  # why pass the whole parent?
    parent_boost = parent.beta

    magP, arbitraryPhi, arbitraryTheta = random2DecayPThetaPhi(parent_mass, massDaughter1, massDaughter2)

    pd1 = calculateFourMomentum(massDaughter1, magP, arbitraryPhi, arbitraryTheta)
    pd2 = calculateFourMomentum(massDaughter2, magP, np.pi + arbitraryPhi, np.pi - arbitraryTheta)

    labFramePD1 = lorentzBoost(pd1, -parent_boost)
    labFramePD2 = lorentzBoost(pd2, -parent_boost)

    daughter1 = createParticleFromFourMomentum(labFramePD1)
    daughter2 = createParticleFromFourMomentum(labFramePD2)

    d1speed = np.linalg.norm(daughter1.beta)
    d2speed = np.linalg.norm(daughter2.beta)
    if d1speed > 1.:
        print("ERROR: non-physical particle 1 travelling faster than speed of light")
        print(d1speed)
        print(str(labFramePD1))

    if d2speed > 1.:
        print("ERROR: non-physical particle 2 travelling faster than speed of light")
        print(d2speed)
        print(str(labFramePD2))

    return daughter1, daughter2

def random3DecayLabFrame(parent, m1, m2, m3):
    '''
    Simulate a decay vertex that creates 3 tracks. Extra free parameter is needed, the momentum of particle 1 magp1.
    Instead choose as free parameter the mass m_23, this is essentially equivalent. We introduce one un-physical
    constraint however, that m_23 is on-shell such that m2+m3<m_23<M-m1.
    Then the problem reduces to two two-body decays. Only need additional four angles to solve problem.
    Method: 2-decay of parent to m1 and m_23, 2-decay of m_23 to m2 and m3.
    '''
    m_23 = random.uniform(m2+m3,parent.m-m1)

    magp1, phi1, theta1 = random2DecayPThetaPhi(parent.m, m1, m_23)  # first two-body decay

    pd1 = calculateFourMomentum(m1, magp1, phi1, theta1)  # first 4mom calculated in zmf of 3-decay

    beta_23 = calculateBeta(magp1, m_23, np.pi + phi1, np.pi - theta1)

    pd2, pd3 = random2DecayAndTransformBack(beta_23, m_23, m2, m3)  # second and third 4mom in zmf of 3-decay

    labFramePD1 = lorentzBoost(pd1, -parent.beta)
    labFramePD2 = lorentzBoost(pd2, -parent.beta)
    labFramePD3 = lorentzBoost(pd3, -parent.beta)

    daughter1 = createParticleFromFourMomentum(labFramePD1)
    daughter2 = createParticleFromFourMomentum(labFramePD2)
    daughter3 = createParticleFromFourMomentum(labFramePD3)

    if ((np.linalg.norm(daughter1.beta) > 1.) or (np.linalg.norm(daughter2.beta) > 1.) or (
            np.linalg.norm(daughter3.beta) > 1.)):
        print("ERROR: non-physical particle travelling faster than speed of light")

    return daughter1, daughter2, daughter3


def random4DecayLabFrame(parent, m1, m2, m3, m4):
    '''
    An extension of 3decay. But now instead of needing just an extra parameter m_23.
    We need two: m_12 and m_34 (in addition to 6 angles: phi/theta12, phi/theta1, phi/theta3)
    Process: 2-decay parent to m_12, m_34. Two decay those intermediates to final particles.
    '''
    if (np.random.random() < 0.5):  # kinematic constraint. PROBLEM: m_12 is usually larger as it is selected 1st
        m_12 = random.uniform(m1 + m2, parent.m - m3 - m4)
        m_34 = random.uniform(m3 + m4, parent.m - m_12)

    else:  # sort of fix
        m_34 = random.uniform(m3 + m4, parent.m - m1 - m2)
        m_12 = random.uniform(m1 + m2, parent.m - m_34)

    magp_12, phi_12, theta_12 = random2DecayPThetaPhi(parent.m, m_12, m_34)  # first two-body decay
    phi_34 = np.pi + phi_12
    theta_34 = np.pi - theta_12

    beta_12 = calculateBeta(magp_12, m_12, phi_12, theta_12)
    beta_34 = calculateBeta(magp_12, m_34, phi_34, theta_34)

    pd1, pd2 = random2DecayAndTransformBack(beta_12, m_12, m1, m2)  # second and third 4mom in zmf of 3-decay
    pd3, pd4 = random2DecayAndTransformBack(beta_34, m_34, m3, m4)  # second and third 4mom in zmf of 3-decay

    labFramePD1 = lorentzBoost(pd1, -parent.beta)
    labFramePD2 = lorentzBoost(pd2, -parent.beta)
    labFramePD3 = lorentzBoost(pd3, -parent.beta)
    labFramePD4 = lorentzBoost(pd4, -parent.beta)

    daughter1 = createParticleFromFourMomentum(labFramePD1)
    daughter2 = createParticleFromFourMomentum(labFramePD2)
    daughter3 = createParticleFromFourMomentum(labFramePD3)
    daughter4 = createParticleFromFourMomentum(labFramePD4)

    if ((np.linalg.norm(daughter1.beta) > 1.) or (np.linalg.norm(daughter2.beta) > 1.)
            or (np.linalg.norm(daughter3.beta) > 1.) or (np.linalg.norm(daughter4.beta) > 1.)):
        print("ERROR: non-physical particle travelling faster than speed of light")

    return daughter1, daughter2, daughter3, daughter4


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

### This is the location of useful functions for the particle package:
### especially functions needed for the calculation of different decays

def random2DecayPThetaPhi(mParent, massDaughter1, massDaughter2):
    '''
    Simulate decay of parent into two daughter particles in ZMF of parent
    Randomly generates phi, theta of first daughter, second follows by symmetry
    0<phi<2pi and 0<theta<pi. masses all in MeV
    Returns daughter particles in lab frame
    '''
    if (massDaughter1 + massDaughter2 > mParent):
        print("Decay is impossible")
        print("mParent was "+str(mParent))
        print("This is less than mD1+mD2="+str(massDaughter1+massDaughter2))
        return None

    magP = np.sqrt(
        ((mParent ** 2 - massDaughter1 ** 2 - massDaughter2 ** 2) ** 2 - (2 * massDaughter1 * massDaughter2) ** 2)) / (
                       2 * mParent)
    arbitraryPhi = random.uniform(-np.pi, np.pi) #watch out for convention
    arbitraryTheta = random.uniform(0, np.pi)

    return magP, arbitraryPhi, arbitraryTheta


def random2DecayAndTransformBack(beta, mass_12, massDaughter1, massDaughter2):
    '''
    for use in CHAINED 2-decays, such as the m_23 decay of a three-decay process
    function calls boost beta, in order to transform back from zmf of 12 to local frame
    '''
    magP, arbitraryPhi, arbitraryTheta = random2DecayPThetaPhi(mass_12, massDaughter1, massDaughter2)  # the decay

    pd1 = calculateFourMomentum(massDaughter1, magP, arbitraryPhi, arbitraryTheta)
    pd2 = calculateFourMomentum(massDaughter2, magP, np.pi + arbitraryPhi, np.pi - arbitraryTheta)

    localFramePD1 = lorentzBoost(pd1, -beta)  # check correct signs
    localFramePD2 = lorentzBoost(pd2, -beta)

    return localFramePD1, localFramePD2


def calculateFourMomentum(mass, magp, phi, theta):
    unitDir=calculateUnitDirection(phi, theta)
    relE=np.sqrt(mass**2 + magp**2)
    fourMom = np.array(([relE,magp*unitDir[0],magp*unitDir[1],magp*unitDir[2]])) #contravariant form
    return fourMom

def calculateUnitDirection(phi, theta):
    dirvec=np.array(([np.sin(theta)*np.cos(phi),
                                      np.sin(theta)*np.sin(phi),
                                      np.cos(theta)]))
    unitDir=dirvec/np.linalg.norm(dirvec)
    return unitDir

def calculateBeta(magp, m, phi, theta):
    unitDir = calculateUnitDirection(phi, theta)
    betaMod = (magp / m) / np.sqrt(1 + (magp / m) ** 2)
    beta = betaMod * unitDir
    return beta

def cartesian2polar(px, py, pz):
    '''
    calculate magp,phi,theta from cartesian 3-mom
    can be used for generic transforms too (x,y,z)->(r,phi,theta)
    '''
    magP = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    theta = np.arctan2(np.sqrt(px ** 2 + py ** 2), pz)  # note definition of theta angle 0 to pi, measured from z axis
    phi = np.arctan2(py, px) # what could pos
    return magP, phi, theta


def createParticleFromFourMomentum(fourMomentum):
    p0 = fourMomentum[0]
    mass = np.sqrt(
        p0 ** 2 - fourMomentum[1] ** 2 - fourMomentum[2] ** 2 - fourMomentum[3] ** 2)  # could call hard code mass?
    # print("calculated mass is " + str(mass))
    # maybe i should call mass instead of calculating it
    magp, phi, theta = cartesian2polar(fourMomentum[1], fourMomentum[2], fourMomentum[3])

    return Particle(mass, phi, theta, magp)


def lorentzBoost(fourMomentum, beta):
    '''
    for particle with fourMomentum in frame S
    boost to frame S' travelling at velocity beta, origins coincide at t=t'=0
    e.g. if particle had velocity beta in S, S' is its Zero Momentum Frame
    if S is ZMF frame of particle travelling at beta in lab frame,
    transform back to lab frame using lorentzBoost(fourMomentum, -beta)
    WARNING, we do introduce rounding errors of order 1e-10
    '''
    lorentzMatrix = np.zeros((4, 4))  # for use on contravariant vectors (t,x,y,z)
    gamma = 1 / np.sqrt(1 - (np.linalg.norm(beta) ** 2))
    lorentzMatrix[0, 0] = gamma
    for i in [1, 2, 3]:
        lorentzMatrix[0, i] = lorentzMatrix[i, 0] = -gamma * beta[i - 1]
        for j in range(i, 4):
            if (i == j):
                lorentzMatrix[i, j] = lorentzMatrix[j, i] = (gamma - 1) * (
                            beta[i - 1] * beta[j - 1] / (np.linalg.norm(beta)) ** 2) + 1
            else:
                lorentzMatrix[i, j] = lorentzMatrix[j, i] = (gamma - 1) * (
                            beta[i - 1] * beta[j - 1] / (np.linalg.norm(beta)) ** 2)
    # lorentzMatrix doesnt exactly have det = 1, but no way around it
    return np.matmul(lorentzMatrix, fourMomentum)