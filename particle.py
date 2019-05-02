import numpy as np
import random

class Particle:
    '''Class to handle true particles for our toy simulations'''

    def __init__(self, m, phi, theta, magp, vtx=np.zeros(3)):
        '''use MeV as default units'''
        self.m = m
        self.phi = phi
        self.theta = theta
        self.magp = magp  # the three-momentum magnitude
        self.origin = vtx # defaults to (0,0,0)
        self.properLifetime = 1.5e-12 # default to B lifetime in seconds

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

    def propagateAndDecay(self, decay):
        decayVtx = self.propagate()
        if decay == "2pions":
            mD1 = 140.
            mD2 = 140.
            p1, p2 = random2DecayLabFrame(self, mD1, mD2)
            p1.setOrigin(decayVtx)
            p2.setOrigin(decayVtx)
            return p1, p2
        elif decay == "pionD":
            mpion = 140.
            mD = 2000.
            p1, p2 = random2DecayLabFrame(self, mpion, mD)
            p1.setOrigin(decayVtx)
            p2.setOrigin(decayVtx)
            return p1, p2
        else:
            print("ERROR: Not a Valid Decay Mode")
            return None, None


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

### Decay Functions:

def random2DecayLabFrame(parent, massDaughter1, massDaughter2):
    '''
    uses particle_utils to create daughters in ZMF then converts back to lab frame
    randomly chooses phi and theta, momentum constrained by conservation law
    '''
    magP, arbitraryPhi, arbitraryTheta = random2DecayPThetaPhi(parent.m, massDaughter1, massDaughter2)

    pd1 = calculateFourMomentum(massDaughter1, magP, arbitraryPhi, arbitraryTheta)
    pd2 = calculateFourMomentum(massDaughter2, magP, np.pi + arbitraryPhi, np.pi - arbitraryTheta)

    labFramePD1 = lorentzBoost(pd1, -parent.beta)
    labFramePD2 = lorentzBoost(pd2, -parent.beta)

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
        return None

    magP = np.sqrt(
        ((mParent ** 2 - massDaughter1 ** 2 - massDaughter2 ** 2) ** 2 - (2 * massDaughter1 * massDaughter2) ** 2)) / (
                       2 * mParent)
    arbitraryPhi = random.uniform(0, 2 * np.pi)
    arbitraryTheta = random.uniform(0, np.pi)

    return magP, arbitraryPhi, arbitraryTheta


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
    print("calculated mass is " + str(mass))
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