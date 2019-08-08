import numpy as np
import pandas as pd
import particle
import jet
import jet_generator
import track
import random

# This script is run to generate a number of toy jets of three flavours and save them using pickle
n_jets = 30

# the following used to add a gaussian err to the jet kinematic vars
def addJetVarsGaussianError(parameter, std=None):
    """ Method to add gaussian error to given parameters, e.g. change 1/magP by a small gaussian err"""
    if std is None:
        # automatically deduce a std dev
        std = abs(parameter / 100)  # essentially a 1% error, absolute value taken

    err = np.random.normal(0, std)
    modified_parameter = parameter + err
    return modified_parameter

def addThetaGaussianError(theta, std=None):
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
        print("jetGaussian error pushed theta over pi, setting equal to pi")
        modified_theta = np.pi
    elif modified_theta < 0.:
        print("jetGaussian error pushed theta below 0, setting equalt to 0")
        modified_theta = 0.
    return modified_theta


# generate light jets
# ljets_df = pd.DataFrame()
print("generating light jets")

ljets_list = n_jets*[None]

for i in range(n_jets):
    jet_energy = random.uniform(1e4, 1e5) # energy not pT is uniform 10GeV to 100GeV
    ljet_creator = jet_generator.JetGenerator(jet_energy, 'light')

    # print("Jet energy is " + str(jet_energy))
    ljet, primary_particles = ljet_creator.create_jet_container_and_primaries()

    # GREG!! Need to extract the phi and theta of the jet in the instance when it is not pi/4!!!
    jet_phi = addJetVarsGaussianError(np.pi/4 , 1e-5) # what value should i give for errs?
    jet_theta = addThetaGaussianError(np.pi/4, 1e-5)
    jet_p = jet_energy # but this assumes light jet mass is 0!!! more about the direction though
    jet_oneOverP =  addJetVarsGaussianError(1/jet_p) # 1% err

    jet_kinematics_as_track = [0.,0.,jet_phi,jet_theta,jet_oneOverP, 0., 0., 0.]
    ljet.addTrack(jet_kinematics_as_track)

    # create tracks form all visible particles and add to the jet
    allparticles = primary_particles

    # When track crated from particle, errors are added
    for p in allparticles:
        tp = track.Track(p, 1)  # Come back to the issue of charge later, for now assume ntrl, qOverP->oneOverP
        ljet.addTrack(tp.printParameters())
        # print("[d0,z0,phi,theta,qOverP,x,y,z]: "+str(tp.printRepresentation()))

    # ljets_df = ljets_df.append(ljet.dataAsPD(), ignore_index=True)
    ljets_list[i] = ljet.data()

ljets_df = pd.DataFrame(ljets_list, columns=["jet_energy", "jet_flavour", "nSecTracks","nTerTracks",
                                                  "secVtx_x", "secVtx_y", "secVtx_z",
                                                  "terVtx_x", "terVtx_y", "terVtx_z",
                                                  "tracks"])


# generate b jets
print("generating b jets")
# bjets_df = pd.DataFrame()
bjets_list = n_jets*[None]
for i in range(n_jets):
    jet_energy = random.uniform(1e4, 1e5)
    bjet_creator = jet_generator.JetGenerator(jet_energy, 'b')

    # print("Jet energy is " + str(jet_energy))
    bjet, B_meson, primary_particles = bjet_creator.create_jet_container_and_primaries()

    # select b decay mode and corresponding number of child particles
    bDecayMode = np.random.choice(["Dpipipi", "Dpipi", "Dpi"])  # To do: add Dneutral,Dneutralpi etc.
    if bDecayMode == "Dpipipi":
        D_meson, pion1, pion2, pion3 = B_meson.propagateAndDecay("Dpionpionpion")
        pions = [pion1, pion2, pion3]
        nSecTracks = 3
    elif bDecayMode == "Dpipi":
        D_meson, pion1, pion2 = B_meson.propagateAndDecay("Dpionpion")
        pions = [pion1, pion2]
        nSecTracks =2
    elif bDecayMode == "Dpi":
        D_meson, pion1 = B_meson.propagateAndDecay("Dpion")
        pions = [pion1]
        nSecTracks =1
    else:
        print("error: no b deacay mode selected")
        break

    bjet.setSecondaryVtx(secondaryVtx=D_meson.origin)
    bjet.setNSecTracks(nSecTracks)

    cDecayMode = np.random.choice(["4pi", "3pi", "2pi"])  # To Do:add Dne,Dnepi etc.

    if cDecayMode == "4pi":
        pion4, pion5, pion6, pion7 = D_meson.propagateAndDecay("4pions")
        bjet.setTertiaryVtx(pion4.origin)
        pions.extend([pion4, pion5, pion6, pion7])
        nTerTracks = 4

    elif cDecayMode == "3pi":
        pion4, pion5, pion6 = D_meson.propagateAndDecay("3pions")
        bjet.setTertiaryVtx(pion4.origin)
        pions.extend([pion4, pion5, pion6])
        nTerTracks = 3

    elif cDecayMode == "2pi":
        pion4, pion5 = D_meson.propagate_and_decay("2pions")
        bjet.setTertiaryVtx(pion4.origin)
        pions.extend([pion4, pion5])
        nTerTracks = 2

    else:
        print("error: no decay mode selected")
        break

    bjet.setNTerTracks(nTerTracks)

    # A simple check to ensure four-mom conservation
    # Not needed but just there to ensure consistency
    sumfourMom = [0., 0., 0., 0.]
    for p in pions:
        sumfourMom += p.fourMom

    #print(np.allclose(sumfourMom, candB.fourMom))
    if not (np.allclose(sumfourMom, B_meson.fourMom)):
        print("ERROR")
        print(sumfourMom)
        print(B_meson.fourMom)
        break

    # create tracks form all visible particles and add to the jet
    # allparticles = primary_particles + pions
    # Reverse the ordering, place primarys at the back
    allparticles = pions + primary_particles

    # Add the track kinematics, e.g. phi, theta, pT, as a "dummy" first track to pass to the RNN

    jet_phi = addJetVarsGaussianError(B_meson.phi , 1e-5) # what value should i give for errs?
    jet_theta = addThetaGaussianError(B_meson.theta, 1e-5)
    jet_p = np.sqrt(B_meson.relE**2 - particle.mB**2) # but this assumes jet mass is B mass!!! more about the direction though
    jet_oneOverP =  addJetVarsGaussianError(1/jet_p) # 1% err

    jet_kinematics_as_track = [0.,0.,jet_phi,jet_theta,jet_oneOverP, 0., 0., 0.]
    bjet.addTrack(jet_kinematics_as_track)

    for p in allparticles:
        tp = track.Track(p, 1)  # Come back to the issue of charge later, for now assume ntrl, qOverP->oneOverP
        bjet.addTrack(tp.printParameters())
        # print("[d0,z0,phi,theta,qOverP,x,y,z]: "+str(tp.printRepresentation()))

    # bjets_df = bjets_df.append(bjet.dataAsPD(), ignore_index=True)
    # GREG!!, the append function above is veeeery slow at large df
    # instead nook out the shape of the df first!!

    bjets_list[i] = bjet.data()


bjets_df = pd.DataFrame(bjets_list, columns=["jet_energy", "jet_flavour", "nSecTracks","nTerTracks",
                                                  "secVtx_x", "secVtx_y", "secVtx_z",
                                                  "terVtx_x", "terVtx_y", "terVtx_z",
                                                  "tracks"])


# print(bjets_df)
# print("and array")
# print(df_from_list)
# print(bjets_df==df_from_list)
# generate c-jets
print("generating c jets")

#cjets_df = pd.DataFrame()

cjets_list = n_jets*[None]

for i in range(n_jets):
    jet_energy = random.uniform(1e4, 1e5)
    cjet_creator = jet_generator.JetGenerator(jet_energy, 'c')

    # print("Jet energy is " + str(jet_energy))
    cjet, D_meson, primary_particles = cjet_creator.create_jet_container_and_primaries()

    cDecayMode = np.random.choice(["4pi", "3pi", "2pi"])  # add Dne,Dnepi etc.

    if cDecayMode == "4pi":
        pion1, pion2, pion3, pion4 = D_meson.propagateAndDecay("4pions")
        pions = [pion1, pion2, pion3, pion4]

    elif cDecayMode == "3pi":
        pion1, pion2, pion3 = D_meson.propagateAndDecay("3pions")
        pions = [pion1, pion2, pion3]

    elif cDecayMode == "2pi":
        pion1, pion2 = D_meson.propagateAndDecay("2pions")
        pions = [pion1, pion2]

    else:
        print("error: no c decay mode selected")
        break

    cjet.setSecondaryVtx(pion1.origin)

    # A simple check to ensure four-mom conservation
    sumfourMom = [0., 0., 0., 0.]
    for p in pions:
        sumfourMom += p.fourMom

    # print(np.allclose(sumfourMom, candD.fourMom))
    if not (np.allclose(sumfourMom, D_meson.fourMom)):
        print("ERROR")
        print(sumfourMom)
        print(D_meson.fourMom)
        break

    # create tracks form all visible particles and add to the jet
    allparticles = primary_particles + pions

    jet_phi = addJetVarsGaussianError(D_meson.phi , 1e-5) # what value should i give for errs?
    jet_theta = addThetaGaussianError(D_meson.theta, 1e-5)
    jet_p = np.sqrt(D_meson.relE**2 - particle.mD**2) # but this assumes jet mass is B mass!!! more about the direction though
    jet_oneOverP =  addJetVarsGaussianError(1/jet_p) # 1% err

    jet_kinematics_as_track = [0.,0.,jet_phi,jet_theta,jet_oneOverP, 0., 0., 0.]
    cjet.addTrack(jet_kinematics_as_track)

    for p in allparticles:
        tp = track.Track(p, 1)  # Come back to the issue of charge later, for now assume ntrl, qOverP->oneOverP
        cjet.addTrack(tp.printParameters())
        # print("[d0,z0,phi,theta,qOverP,x,y,z]: "+str(tp.printRepresentation()))

    # cjets_df = cjets_df.append(cjet.dataAsPD(), ignore_index=True)
    cjets_list[i] = cjet.data()

cjets_df = pd.DataFrame(cjets_list, columns=["jet_energy", "jet_flavour", "nSecTracks","nTerTracks",
                                                  "secVtx_x", "secVtx_y", "secVtx_z",
                                                  "terVtx_x", "terVtx_y", "terVtx_z",
                                                  "tracks"])



# save all results using pickle

bjets_df.to_pickle("./bjets_test.pkl")
cjets_df.to_pickle("./cjets_test.pkl")
ljets_df.to_pickle("./ljets_test.pkl")