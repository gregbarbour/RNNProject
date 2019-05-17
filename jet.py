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
        self.secondaryVtx = None
        self.tertiaryVtx = None
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


def generateBJet(jet_energy):
    '''
    Need to generate a b-hadron and some other primary tracks/particles
    Whilst ensuring they all share a common jet direction
    '''
    mPion = 140.
    phi = np.pi / 4
    theta = np.pi / 4
    B_momentum = .8 * jet_energy  # an approximation of the 80% hadronization fraction of Bs
    candB = particle.Particle(5300., phi, theta, B_momentum)
    candB.setProperLifetime(1.5e-12)

    # this leaves 20% of the jet energy to assign to some light charged particles
    # done using two-decay methodology with virtual mass M
    # how many light particles is determined by a randomly assigning fractions of remainder energy
    energy_remainder = 0.2 * jet_energy
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
            print("Error in energy measurement: " + str(energy_remainder / jet_energy))
            break

    return candB, primary_particles