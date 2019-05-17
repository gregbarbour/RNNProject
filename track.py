# The class to contain tracks for use in our first version of an RNN
# These are created from particle objects from simulated jets

import particle
import numpy as np
import random
import mathutils


class Track:

    def __init__(self, particle, charge):
        """
        Initialize from particle object. Adding gaussian errors to parameters.
        Errors taken from: https://www.mpp.mpg.de/~sct/welcomeaux/papers/MOlivo_DPG06.pdf
        https://cds.cern.ch/record/1746744/files/ATL-PHYS-SLIDE-2014-503.pdf
        """
        if np.all(particle.origin==0.):
            print("yee")
            err=1e-5 # vertexing error of aprrox 3e-5
            self.position = self.addGaussianError(particle.origin, err*np.ones(3))
            # position at which params are defined
        else:
            self.position = self.addGaussianError(particle.origin)
        self.phi = self.addGaussianError(particle.phi, 1e-3)
        self.theta = self.addThetaGaussianError(particle.theta, 1e-3) #actually much worse at vals close to 0 or pi, but use approx to keep close to phi
        self.qOverP = charge / self.addGaussianError(particle.magp) #use std of 1%
        self.d0, self.z0 = self.calculateImpactParams()  # use newly smudged values to calculate IPs
        self.covariance = np.identity(5)  # placeholder for now, set covariance matrix to 5d identity

    def calculateImpactParams(self):  #
        """
        If we choose to do this, simplify problem by assuming tracks just straight lines
        Calculate second positon on track, use to find perigeePoint assuming straight line
        Then calculate z0 and d0 from perigee and origin: 0,0,0
        """
        direction = self.calculateUnitDirection()
        position1 = self.position
        position2 = position1 + direction
        perigeePoint, t = mathutils.geometry.intersect_point_line([0, 0, 0], position1, position2)
        d0 = np.sqrt(perigeePoint[0] ** 2 + perigeePoint[1] ** 2)
        z0 = perigeePoint[2]
        return d0, z0

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
            std = abs(parameter / 100)  # essentially a 1% error, absolute value taken

        err = np.random.normal(0, std)
        parameter += err
        return parameter

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
        theta += err
        if theta > np.pi:
            print("Gaussian error pushed theta over pi, setting equal to pi")
            theta = np.pi
        elif theta < 0.:
            print("Gaussian error pushed theta below 0, setting equalt to 0")
            theta = 0.
        return theta

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
                                 self.position[0], self.position[1], self.position[2]]
        covariance = self.covariance  # as a vector perhaps?

        return parametersAndPosition