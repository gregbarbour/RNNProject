import numpy as np
import pandas as pd
import particle
import track
import random


class Jet:

    def __init__(self, energy, flavour):
        self.energy = energy
        self.flavour = flavour  # i.e. b-jet is 5, c-jet is 4, l-jet is 1
        self.particles = None
        self.tracks = []
        self.primaryVtx = np.array(([0, 0, 0]))
        self.secondaryVtx = np.array(([None, None, None]))
        self.tertiaryVtx = np.array(([None, None, None]))
        # self.trksAtSecondary ??

    def setSecondaryVtx(self, secondaryVtx):
        self.secondaryVtx = secondaryVtx
        # print( "Candidate jet true secondary vtx at: "+str(secondaryVtx))

    def setTertiaryVtx(self, tertiaryVtx):
        self.tertiaryVtx = tertiaryVtx
        # print( "Candidate jet true tertiary vtx at: "+str(tertiaryVtx))

    def addTrack(self, trackParams):
        """Takes track represented by parameters vector [d0,z0,phi,theta,qOverP,x,y,z]"""
        self.tracks.append(trackParams)
        # print("Track added to Jet Candidate")

    def data(self):
        data = self.jetLabels()
        data.append(self.tracks)
        return [data]  # trackcolumns=["d0","z0","phi","theta","qOverP","x","y","z"])

    def dataAsPD(self):
        return pd.DataFrame(self.data(), columns=["jet_energy", "jet_flavour",
                                                  "secVtx_x", "secVtx_y", "secVtx_z",
                                                  "terVtx_x", "terVtx_y", "terVtx_z",
                                                  "tracks"])

    def jetLabels(self):
        jet_labs = [self.energy, self.flavour]
        jet_labs.extend(self.secondaryVtx)
        jet_labs.extend(self.tertiaryVtx)
        return jet_labs  # pd.DataFrame( jet_labs, columns = ["jet_energy", "jet_flavour","secVtx_x","secVtx_y","secVtx_z",terVtx_x","terVtx_y","terVtx_z"])


##########################################################################################################
# Methods for generating jets with primary vtx tracks

def addTwoPrimaryTracks(energy, phi, theta):
    """
    Adds two light tracks (pions) with directions summing to phi, theta
    This is a hack to maintain jet direction phi theta (e.g. B-meson direction)

    Create particle mass M travelling in phi,theta with total energy E
    then random2decaylabframe this to give the two primary tracks,
    Choosing M: Lower lim 2mpion or random2decay wont work,
    Upper Lim: Constrain the tracks by their dR=0.4 (phi and theta), ballpark:
    constraining M<E/4 means dANGLE<0.167 or thereabouts, so dR<0.33
    """
    mPion = 140.
    M = np.random.uniform(2 * mPion, energy / 4)  # this is the virtual mass M that two decays
    magp = np.sqrt(energy ** 2 - M ** 2)

    betaM = particle.calculateBeta(magp, M, phi, theta)

    primPionP1, primPionP2 = particle.random2DecayAndTransformBack(betaM, M, mPion, mPion)

    primaryPion1 = particle.createParticleFromFourMomentum(primPionP1)
    primaryPion2 = particle.createParticleFromFourMomentum(primPionP2)

    return primaryPion1, primaryPion2


def generateBJetPrimary(jet_energy):
    '''
    Need to generate a b-hadron and some other primary tracks/particles
    Whilst ensuring they all share a common jet direction
    '''
    mPion = 140.
    mB=5300.
    phi = np.random.uniform(-np.pi,np.pi)
    theta = np.random.uniform(0.,np.pi)
    B_energy = .8 * jet_energy  # an approximation of the 80% hadronization fraction of Bs
    B_momentum = np.sqrt(B_energy**2 - mB**2)
    candB = particle.Particle(mB, phi, theta, B_momentum)
    candB.setProperLifetime(1.5e-12)

    # this leaves 20% of the jet energy to assign to some light charged particles
    # done using two-decay methodology with virtual mass M
    # how many light particles is determined by a randomly assigning fractions of remainder energy
    energy_remainder = jet_energy - B_energy
    primary_particles = []
    while True:
        energy_frac_1 = np.random.uniform(8 * mPion, energy_remainder)
        if (energy_frac_1 < 16 * mPion):
            energy_frac_1 = energy_remainder
        primaryPion1, primaryPion2 = addTwoPrimaryTracks(energy_frac_1, phi, theta)
        energy_remainder -= energy_frac_1

        primary_particles.append(primaryPion1)
        primary_particles.append(primaryPion2)

        if (energy_remainder < 8 * mPion):
            #print("Error in energy measurement: " + str(energy_remainder / jet_energy))
            break

    return candB, primary_particles


def generateCJetPrimary(jet_energy):
    '''
    Need to generate a c-hadron and some other primary tracks/particles
    Whilst ensuring they all share a common jet direction
    '''
    mD = 2000.
    mPion = 140.
    phi = np.pi / 4
    theta = np.pi / 4
    D_energy = .5 * jet_energy  # an approximation of the 50% hadronization fraction of c
    D_momentum = np.sqrt(D_energy ** 2 - mD ** 2)
    candD = particle.Particle(mD, phi, theta, D_momentum)
    candD.setProperLifetime(1.0e-12)

    # Assign remainin energy to primary vertex tracks
    # randomly assigning fractions of remainder energy
    energy_remainder = 0.5 * jet_energy
    primary_particles = []

    # I have an idea to prevent the problem of trk distribution being only even
    if np.random.random()>0.5:
        # randomly create a pion travelling along the jet axis? dunno...
        random_trk_E = 0.125*jet_energy
        random_trk_mom = np.sqrt(random_trk_E**2 - mPion**2)
        jet_axis_pion = particle.Particle(mPion, phi, theta, random_trk_mom)
        primary_particles.append(jet_axis_pion)
        energy_remainder-=random_trk_E

    while True:
        energy_frac_1 = np.random.uniform(8 * mPion, energy_remainder)
        if (energy_frac_1 < 16 * mPion):
            energy_frac_1 = energy_remainder
        primaryPion1, primaryPion2 = addTwoPrimaryTracks(energy_frac_1, phi, theta)
        energy_remainder -= energy_frac_1

        primary_particles.append(primaryPion1)
        primary_particles.append(primaryPion2)

        if (energy_remainder < 8 * mPion):
            #print("Error in energy measurement: " + str(energy_remainder / jet_energy))
            break

    return candD, primary_particles


def generateLightJetPrimary(jet_energy):
    '''
    Need to generate a c-hadron and some other primary tracks/particles
    Whilst ensuring they all share a common jet direction
    '''
    mPion = 140.
    phi = np.pi / 4
    theta = np.pi / 4

    energy_remainder = 0.8 * jet_energy  # make an assumption that not all jet energy contained in tracks! Reduces n of trks in l-jets
    primary_particles = []

    # I have an idea to prevent the problem of trk distribution being only even
    if np.random.random() > 0.5:
        # randomly create a pion travelling along the jet axis? dunno...
        random_trk_E = np.random.uniform(0.4 * jet_energy, 0.6 * jet_energy)
        random_trk_mom = np.sqrt(random_trk_E ** 2 - mPion ** 2)
        jet_axis_pion = particle.Particle(mPion, phi, theta, random_trk_mom)
        primary_particles.append(jet_axis_pion)
        energy_remainder -= random_trk_E

    # Assign all energy to primary vertex tracks
    # randomly assigning fractions of remainder energy
    while True:

        energy_frac_1 = np.random.uniform(8 * mPion, energy_remainder)  # 0.5*jet_energy)

        if (energy_frac_1 < 16 * mPion):  # keep dR<0.4
            energy_frac_1 = energy_remainder

        primaryPion1, primaryPion2 = addTwoPrimaryTracks(energy_frac_1, phi, theta)
        energy_remainder -= energy_frac_1

        primary_particles.append(primaryPion1)
        primary_particles.append(primaryPion2)

        if (energy_remainder <= 8 * mPion):
            #print("Error in energy measurement: " + str(energy_remainder / jet_energy))
            break

    return primary_particles
