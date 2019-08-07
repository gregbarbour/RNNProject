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


    def create_jet_container_and_primaries(self):
        '''
        create appropriate jet container and primary vertex particles
        i.e. b or c jets have additional meson
        '''
        if self.flavour == 'light':
            primary_particles = self.create_light_jet_primaries()
            jet_container = jet.Jet(self.energy, 1) # light is 1
            return jet_container, primary_particles

        if self.flavour == 'c':
            D_meson, primary_particles = self.create_c_jet_primaries()
            jet_container = jet.Jet(self.energy, 4)
            return jet_container, D_meson, primary_particles

        if self.flavour =='b':
            B_meson, primary_particles = self.create_b_jet_primaries()
            jet_container = jet.Jet(self.energy, 5)
            return jet_container, B_meson, primary_particles


    def create_light_jet_primaries(self):
        '''
        Need to generate a c-hadron and some other primary tracks/particles
        Whilst ensuring they all share a common jet direction
        '''
        jet_energy = self.energy

        phi = np.pi / 4
        theta = np.pi / 4

        energy_remainder = 0.8 * jet_energy  # make an assumption that not all jet energy contained in tracks! Reduces n of trks in l-jets
        primary_particles = []

        # I have an idea to prevent the problem of trk distribution being only even
        if np.random.random() > 0.5:
            # randomly create a pion travelling along the jet axis? dunno...
            random_trk_E = np.random.uniform(0.4 * jet_energy, 0.6 * jet_energy)
            random_trk_mom = np.sqrt(random_trk_E ** 2 - particle.mPion ** 2)
            jet_axis_pion = particle.Particle(particle.mPion, phi, theta, random_trk_mom)
            primary_particles.append(jet_axis_pion)
            energy_remainder -= random_trk_E

        # Assign all energy to primary vertex tracks
        # randomly assigning fractions of remainder energy
        while True:

            energy_frac_1 = np.random.uniform(8 * particle.mPion, energy_remainder)  # 0.5*jet_energy)

            if (energy_frac_1 < 16 * particle.mPion):  # keep dR<0.4
                energy_frac_1 = energy_remainder

            primaryPion1, primaryPion2 = self.add_two_primary_tracks(energy_frac_1, phi, theta)
            energy_remainder -= energy_frac_1

            primary_particles.append(primaryPion1)
            primary_particles.append(primaryPion2)

            if (energy_remainder <= 8 * particle.mPion):
                # print("Error in energy measurement: " + str(energy_remainder / jet_energy))
                break

        return primary_particles


    def create_c_jet_primaries(self):
        '''
        Need to generate a c-hadron and some other primary tracks/particles
        Whilst ensuring they all share a common jet direction
        '''
        jet_energy = self.energy

        phi = np.pi / 4
        theta = np.pi / 4
        D_energy = .5 * jet_energy  # an approximation of the 50% hadronization fraction of c
        D_momentum = np.sqrt(D_energy ** 2 - particle.mD ** 2)
        candD = particle.Particle(particle.mD, phi, theta, D_momentum)
        candD.setProperLifetime(1.0e-12)

        # Assign remainin energy to primary vertex tracks
        # randomly assigning fractions of remainder energy
        energy_remainder = 0.5 * jet_energy
        primary_particles = []

        # I have an idea to prevent the problem of trk distribution being only even
        if np.random.random() > 0.5:
            # randomly create a pion travelling along the jet axis? dunno...
            random_trk_E = 0.125 * jet_energy
            random_trk_mom = np.sqrt(random_trk_E ** 2 - particle.mPion ** 2)
            jet_axis_pion = particle.Particle(particle.mPion, phi, theta, random_trk_mom)
            primary_particles.append(jet_axis_pion)
            energy_remainder -= random_trk_E

        while True:
            energy_frac_1 = np.random.uniform(8 * particle.mPion, energy_remainder)
            if (energy_frac_1 < 16 * particle.mPion):
                energy_frac_1 = energy_remainder
            primaryPion1, primaryPion2 = self.add_two_primary_tracks(energy_frac_1, phi, theta)
            energy_remainder -= energy_frac_1

            primary_particles.append(primaryPion1)
            primary_particles.append(primaryPion2)

            if (energy_remainder < 8 * particle.mPion):
                # print("Error in energy measurement: " + str(energy_remainder / jet_energy))
                break

        return candD, primary_particles


    def create_b_jet_primaries(self):
        '''
        Need to generate a b-hadron and some other primary tracks/particles
        Whilst ensuring they all share a common jet direction
        '''
        jet_energy = self.energy

        phi = np.pi / 4  # np.random.uniform(-np.pi,np.pi)
        theta = np.pi / 4  # np.random.uniform(0.,np.pi)
        B_energy = .8 * jet_energy  # an approximation of the 80% hadronization fraction of Bs
        B_momentum = np.sqrt(B_energy ** 2 - particle.mB ** 2)
        candB = particle.Particle(particle.mB, phi, theta, B_momentum)
        candB.setProperLifetime(1.5e-12)

        # this leaves 20% of the jet energy to assign to some light charged particles
        # done using two-decay methodology with virtual mass M
        # how many light particles is determined by a randomly assigning fractions of remainder energy
        energy_remainder = jet_energy - B_energy
        primary_particles = []
        while True:
            energy_frac_1 = np.random.uniform(8 * particle.mPion, energy_remainder)
            if (energy_frac_1 < 16 * particle.mPion):
                energy_frac_1 = energy_remainder
            primaryPion1, primaryPion2 = self.add_two_primary_tracks(energy_frac_1, phi, theta)
            energy_remainder -= energy_frac_1

            primary_particles.append(primaryPion1)
            primary_particles.append(primaryPion2)

            if (energy_remainder < 8 * particle.mPion):
                # print("Error in energy measurement: " + str(energy_remainder / jet_energy))
                break

        return candB, primary_particles


    def add_two_primary_tracks(self, energy, phi, theta):
        """
        Adds two light tracks (pions) with directions summing to phi, theta
        This is a hack to maintain jet direction phi theta (e.g. B-meson direction)

        Create particle mass M travelling in phi,theta with total energy E
        then random2decaylabframe this to give the two primary tracks,
        Choosing M: Lower lim 2mpion or random2decay wont work,
        Upper Lim: Constrain the tracks by their dR=0.4 (phi and theta), ballpark:
        constraining M<E/4 means dANGLE<0.167 or thereabouts, so dR<0.33
        """
        M = np.random.uniform(2 * particle.mPion, energy / 4)  # this is the virtual mass M that two decays
        magp = np.sqrt(energy ** 2 - M ** 2)

        betaM = particle.calculateBeta(magp, M, phi, theta)  # !!! this is bad practice?

        primPionP1, primPionP2 = particle.random2DecayAndTransformBack(betaM, M, particle.mPion, particle.mPion)

        primaryPion1 = particle.createParticleFromFourMomentum(primPionP1)
        primaryPion2 = particle.createParticleFromFourMomentum(primPionP2)

        return primaryPion1, primaryPion2

