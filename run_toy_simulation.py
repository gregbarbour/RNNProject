import numpy as np
import pandas as pd
import particle
import jet
import track
import random

# This script is run to generate a number of toy jets of three flavours and save them using pickle
n_jets = 100

# generate light jets
ljets_df = pd.DataFrame()
print("generating light jets")


for i in range(n_jets):
    jet_energy = random.uniform(1e4, 1e5) # energy not pT is uniform 10GeV to 100GeV
    ljet = jet.Jet(jet_energy, 1)

    #print("Jet energy is " + str(jet_energy))
    primary_particles = jet.generateLightJetPrimary(jet_energy)

    # create tracks form all visible particles and add to the jet
    allparticles = primary_particles

    # When track crated from particle, errors are added
    for p in allparticles:
        tp = track.Track(p, 1)  # Come back to the issue of charge later, for now assume ntrl, qOverP->oneOverP
        ljet.addTrack(tp.printParameters())
        # print("[d0,z0,phi,theta,qOverP,x,y,z]: "+str(tp.printRepresentation()))

    ljets_df = ljets_df.append(ljet.dataAsPD(), ignore_index=True)


# generate b jets
print("generating b jets")
bjets_df = pd.DataFrame()
for i in range(n_jets):
    jet_energy = random.uniform(1e4, 1e5)
    bjet = jet.Jet(jet_energy, 5)

    #print("Jet energy is " + str(jet_energy))
    candB, primary_particles = jet.generateBJetPrimary(jet_energy)

    # select b decay mode and corresponding number of child particles
    bDecayMode = np.random.choice(["Dpipipi", "Dpipi", "Dpi"])  # To do: add Dneutral,Dneutralpi etc.
    if bDecayMode == "Dpipipi":
        candD, pion1, pion2, pion3 = candB.propagateAndDecay("Dpionpionpion")
        pions = [pion1, pion2, pion3]
    elif bDecayMode == "Dpipi":
        candD, pion1, pion2 = candB.propagateAndDecay("Dpionpion")
        pions = [pion1, pion2]
    elif bDecayMode == "Dpi":
        candD, pion1 = candB.propagateAndDecay("Dpion")
        pions = [pion1]

    bjet.setSecondaryVtx(secondaryVtx=candD.origin)

    cDecayMode = np.random.choice(["4pi", "3pi", "2pi"])  # To Do:add Dne,Dnepi etc.

    if cDecayMode == "4pi":
        pion4, pion5, pion6, pion7 = candD.propagateAndDecay("4pions")
        bjet.setTertiaryVtx(pion4.origin)
        pions.extend([pion4, pion5, pion6, pion7])

    if cDecayMode == "3pi":
        pion4, pion5, pion6 = candD.propagateAndDecay("3pions")
        bjet.setTertiaryVtx(pion4.origin)
        pions.extend([pion4, pion5, pion6])

    if cDecayMode == "2pi":
        pion4, pion5 = candD.propagateAndDecay("2pions")
        bjet.setTertiaryVtx(pion4.origin)
        pions.extend([pion4, pion5])

    # A simple check to ensure four-mom conservation
    # Not needed but just there to ensure consistency
    sumfourMom = [0., 0., 0., 0.]
    for p in pions:
        sumfourMom += p.fourMom

    #print(np.allclose(sumfourMom, candB.fourMom))
    if not (np.allclose(sumfourMom, candB.fourMom)):
        print("ERROR")
        print(sumfourMom)
        print(candB.fourMom)
        break

    # create tracks form all visible particles and add to the jet
    allparticles = primary_particles + pions

    for p in allparticles:
        tp = track.Track(p, 1)  # Come back to the issue of charge later, for now assume ntrl, qOverP->oneOverP
        bjet.addTrack(tp.printParameters())
        # print("[d0,z0,phi,theta,qOverP,x,y,z]: "+str(tp.printRepresentation()))

    bjets_df = bjets_df.append(bjet.dataAsPD(), ignore_index=True)


# generate c-jets
print("generating c jets")

cjets_df = pd.DataFrame()
for i in range(n_jets):
    jet_energy = random.uniform(1e4, 1e5)
    cjet = jet.Jet(jet_energy, 4)

    #print("Jet energy is " + str(jet_energy))
    candD, primary_particles = jet.generateCJetPrimary(jet_energy)

    cDecayMode = np.random.choice(["4pi", "3pi", "2pi"])  # add Dne,Dnepi etc.

    if cDecayMode == "4pi":
        pion1, pion2, pion3, pion4 = candD.propagateAndDecay("4pions")
        pions = [pion1, pion2, pion3, pion4]

    if cDecayMode == "3pi":
        pion1, pion2, pion3 = candD.propagateAndDecay("3pions")
        pions = [pion1, pion2, pion3]

    if cDecayMode == "2pi":
        pion1, pion2 = candD.propagateAndDecay("2pions")
        pions = [pion1, pion2]

    cjet.setSecondaryVtx(pion1.origin)

    # A simple check to ensure four-mom conservation
    sumfourMom = [0., 0., 0., 0.]
    for p in pions:
        sumfourMom += p.fourMom

    #print(np.allclose(sumfourMom, candD.fourMom))
    if not (np.allclose(sumfourMom, candD.fourMom)):
        print("ERROR")
        print(sumfourMom)
        print(candD.fourMom)
        break

    # create tracks form all visible particles and add to the jet
    allparticles = primary_particles + pions

    for p in allparticles:
        tp = track.Track(p, 1)  # Come back to the issue of charge later, for now assume ntrl, qOverP->oneOverP
        cjet.addTrack(tp.printParameters())
        # print("[d0,z0,phi,theta,qOverP,x,y,z]: "+str(tp.printRepresentation()))

    cjets_df = cjets_df.append(cjet.dataAsPD(), ignore_index=True)



# save all results using pickle

bjets_df.to_pickle("./bjetsMINERRs.pkl")
cjets_df.to_pickle("./cjetsMINERRs.pkl")
ljets_df.to_pickle("./ljetsMINERRs.pkl")