'''
Jet Generator Class: Used to initialize both the jet container and all primary particles
for a jet. Taking only two input parameters, the flavour and energy of the jet.
'''

import numpy as np
import jet
import particle


class JetGenerator:

    def __init__(self, energy, flavour):
        self.energy = energy
        self.flavour = flavour

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

        betaM = particle.calculateBeta(magp, M, phi, theta) #!!! this is bad practice

        primPionP1, primPionP2 = particle.random2DecayAndTransformBack(betaM, M, mPion, mPion)

        primaryPion1 = particle.createParticleFromFourMomentum(primPionP1)
        primaryPion2 = particle.createParticleFromFourMomentum(primPionP2)

        return primaryPion1, primaryPion2
