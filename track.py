# The class to contain tracks for use in our first version of an RNN
# These are created from particle objects from simulated jets

import particle
import numpy as np
import random

class Track:

    def __init__(self, particle, charge, noise=True):
        """
        Initialize from particle object. Adding gaussian errors to parameters.
        Errors taken from: https://www.mpp.mpg.de/~sct/welcomeaux/papers/MOlivo_DPG06.pdf
        https://cds.cern.ch/record/1746744/files/ATL-PHYS-SLIDE-2014-503.pdf
        """
        if noise:
            # print("noising")
            if np.all(particle.origin==0.):
                #print("primary track created, assigning correct err")
                # err=1e-5 # vertexing error of aprrox 3e-5 for primary
                # self.position = self.addGaussianError(particle.origin, err*np.ones(3)) if noise else particle.origin
                self.position = particle.origin
                # position at which params are defined
            else:
                # So I actually add the error to the position measurement (1% if not primary)
                # Instead of adding the error to the impact parameter
                self.position = particle.origin
            self.phi = self.addGaussianError(particle.phi, 1e-4) #-5 is min errs
            self.theta = self.addThetaGaussianError(particle.theta, 1e-4) #-5 is min errs
            self.qOverP = charge * self.addGaussianError(1./particle.magp, abs((1./particle.magp) / 100))  # defaults to 0.1% error

            true_d0, true_z0, perigee = self.calculateImpactParams(particle.unitDirection, particle.origin)  # use true values to calculate IPs
            if true_d0 == 0.: # should instead be something like isPrimaryTrack...
                self.d0 = abs(self.addGaussianError(true_d0, 1e-4))
                self.z0 = self.addGaussianError(true_z0, 1e-4)
                true_polar_angle= np.pi/4 #arbitrary choice
            else:
                self.d0 = abs(self.addGaussianError(true_d0, 1e-4))
                self.z0 = self.addGaussianError(true_z0, 1e-4)
                true_polar_angle = np.sign(perigee[1])*np.arccos(perigee[0] / true_d0)
            z_perigee = self.z0
            x_perigee = self.d0 * np.cos(true_polar_angle) #note noise is only added to d0, not to the azimuthal angular position of the perigee!! Maybe should use same error as phi
            y_perigee = self.d0 * np.sin(true_polar_angle)

            self.perigee = np.array([x_perigee, y_perigee, z_perigee])
        else:
            print("No noise")
            self.position = particle.origin
            self.phi = particle.phi  # -5 is min errs
            self.theta = particle.theta  # -5 is min errs
            self.qOverP = charge * (1 / particle.magp)  # defaults to 0.1% error

            self.d0, self.z0, self.perigee = self.calculateImpactParams(particle.unitDirection,
                                                          particle.origin)  # use true values to calculate IPs

        self.covariance = np.identity(5)  # unused for now, set covariance matrix to identity

    def calculateImpactParams(self, true_unitDir, true_origin):  #
        """
        If we choose to do this, simplify problem by assuming tracks just straight lines
        Calculate second positon on track, use to find perigeePoint assuming straight line
        Then calculate z0 and d0 from perigee and origin: 0,0,0
        """
        perigeePoint = POCA(true_origin, true_unitDir)
        d0 = np.sqrt(perigeePoint[0] ** 2 + perigeePoint[1] ** 2) # what if prim vtx not 0,0,0??
        z0 = perigeePoint[2]
        self.perigee = perigeePoint
        return d0, z0, perigeePoint

    def calculateUnitDirection(self):
        dirvec = np.array(([np.sin(self.theta) * np.cos(self.phi),
                            np.sin(self.theta) * np.sin(self.phi),
                            np.cos(self.theta)]))
        unitDir = dirvec / np.linalg.norm(dirvec)
        return unitDir

    def addGaussianError(self, parameter, std=None):
        """ Method to add gaussian error to given parameters, e.g. change magP by a small gaussian err"""
        if std is None:
            # automatically deduce a std dev
            std = abs(parameter / 1000)  #0.1% is min err # essentially a 1% error

        err = np.random.normal(0, std)
        modified_parameter = parameter + err
        return modified_parameter

    def addThetaGaussianError(self, theta, std=None):
        """
        Method to add gaussian error to ANGLe parameters, e.g. change theta by a small gaussian err
        Theta needs special consideration as its domain is 0<theta<pi, so adding a small err at
        theta ~ 0 or pi is dangerous
        """
        if std == None:
            # automatically deduce a std dev
            std = theta / 100  # essentially a 1% error

        err = np.random.normal(0, std)
        modified_theta = theta + err
        if modified_theta > np.pi:
            print("Gaussian error pushed theta over pi, setting equal to pi")
            modified_theta = np.pi
        elif modified_theta < 0.:
            print("Gaussian error pushed theta below 0, setting equalt to 0")
            modified_theta = 0.
        return modified_theta

    # def addErrorsToParameters(self):
    #    self.phi = addGaussianError(self.phi)
    #    self.theta = addGaussianError(self.theta)
    #    self.qOverP = addGaussianError(self.qOverP)
    #    self.z0 = addGaussianError(self.z0) #??? do we do this or add errors to position,phi,theta and then calculate d0,z0
    #   self.d0 = addGaussianError(self.d0) #??? would need to recalculate direction in that case
    #    self.position#???

    def printParameters(self):
        """returns some vector representation for the rnn and storage, come back to this later"""
        parametersAndPosition = [self.d0, self.z0, self.phi, self.theta, self.qOverP,
                                 self.position[0], self.position[1], self.position[2], self.perigee[0], self.perigee[1], self.perigee[2]]
        covariance = self.covariance  # as a vector perhaps?
        # print(parametersAndPosition)
        return parametersAndPosition


def POCA(r, m):
    '''Gives position of closest approach to origin'''
    m = m / np.linalg.norm(m)
    POCA = r - (np.dot(m,r))*m
    return POCA