#!/usr/bin/env python3
import math

import numpy as np
import pandas as pd
from pathlib import Path

from .constants import EV, H, K_B, ME, QE
import pynonthermal.collion

DATADIR = Path(__file__).absolute().parent / 'data'

experiment_use_Latom_in_spencerfano = False


def electronlossfunction(energy_ev, nne_cgs):
    # free-electron plasma loss rate (as in Kozma & Fransson 1992)
    # - dE / dX [eV / cm]
    # returns a positive number

    # return math.log(energy_ev) / energy_ev
    nne = nne_cgs
    energy = energy_ev * EV  # convert eV to erg

    # omegap = math.sqrt(4 * math.pi * nne_cgs * pow(QE, 2) / ME)
    omegap = 5.64e4 * math.sqrt(nne_cgs)
    zetae = H * omegap / 2 / math.pi

    if energy_ev > 14:
        assert 2 * energy > zetae
        lossfunc = nne * 2 * math.pi * QE ** 4 / energy * math.log(2 * energy / zetae)
    else:
        v = math.sqrt(2 * energy / ME)  # velocity in m/s
        eulergamma = 0.577215664901532
        lossfunc = nne * 2 * math.pi * QE ** 4 / energy * math.log(ME * pow(v, 3) / (eulergamma * pow(QE, 2) * omegap))

    # lossfunc is now [erg / cm]
    return lossfunc / EV  # return as [eV / cm]


def get_nntot(ions, ionpopdict):
    # total number density of all nuclei [cm^-3]
    nntot = 0.
    for Z, ionstage in ions:
        nntot += ionpopdict[(Z, ionstage)]
    return nntot


def get_Zbar(ions, ionpopdict):
    # number density-weighted average atomic number
    # i.e. protons per nucleus
    Zbar = 0.
    nntot = get_nntot(ions, ionpopdict)
    for Z, ionstage in ions:
        ionnumberdensity = ionpopdict[(Z, ionstage)]
        Zbar += Z * ionnumberdensity / nntot

    return Zbar


def get_Zboundbar(ions, ionpopdict):
    # number density-weighted average number of bound electrons per nucleus
    Zboundbar = 0.
    nntot = get_nntot(ions, ionpopdict)
    for Z, ionstage in ions:
        ionnumberdensity = ionpopdict[(Z, ionstage)]
        Zboundbar += (Z - ionstage + 1) * ionnumberdensity / nntot

    return Zboundbar


def get_nnetot(ions, ionpopdict):
    # total number density of electrons (free + bound) [cm-^3]
    # return get_Zbar(ions, ionpopdict) * get_nntot(ions, ionpopdict)
    nnetot = 0.
    for Z, ionstage in ions:
        nnetot += Z * ionpopdict[(Z, ionstage)]

    return nnetot


def get_nne(ions, ionpopdict):
    # number density of free electrons [cm-^3]
    nne = 0.
    for Z, ionstage in ions:
        charge = ionstage - 1
        assert(charge >= 0)
        nne += charge * ionpopdict[(Z, ionstage)]

    return nne


def get_lte_pops(adata, ions, ionpopdict, temperature):
    poplist = []

    for _, ion in adata.iterrows():
        ionid = (ion.Z, ion.ion_stage)
        if ionid in ions:
            Z = ion.Z
            ionstage = ion.ion_stage
            ionnumberdensity = ionpopdict[(Z, ionstage)]

            ltepartfunc = ion.levels.eval('g * exp(-energy_ev / @K_B / @temperature)').sum()

            for levelindex, level in ion.levels.iterrows():
                ion_popfrac = 1. / ltepartfunc * level.g * math.exp(-level.energy_ev / K_B / temperature)
                levelnumberdensity = ionnumberdensity * ion_popfrac

                poprow = (Z, ionstage, levelindex, levelnumberdensity, levelnumberdensity, ion_popfrac)
                poplist.append(poprow)

    dfpop = pd.DataFrame(poplist, columns=['Z', 'ion_stage', 'level', 'n_LTE', 'n_NLTE', 'ion_popfrac'])
    return dfpop


def get_d_etaexcitation_by_d_en_vec(engrid, yvec, ions, dftransitions, deposition_density_ev):
    npts = len(engrid)
    part_integrand = np.zeros(npts)

    for Z, ion_stage in ions:
        if not (Z, ion_stage) in dftransitions:
            continue
        for _, row in dftransitions[(Z, ion_stage)].iterrows():
            levelnumberdensity = row.lower_pop
            # levelnumberdensity = ionnumberdensity
            epsilon_trans_ev = row.epsilon_trans_ev
            if epsilon_trans_ev >= engrid[0]:
                xsvec = pynonthermal.excitation.get_xs_excitation_vector(engrid, row)
                part_integrand += (levelnumberdensity * epsilon_trans_ev * xsvec / deposition_density_ev)

    return yvec * part_integrand


def get_d_etaion_by_d_en_vec(engrid, yvec, ions, ionpopdict, dfcollion, deposition_density_ev):
    npts = len(engrid)
    part_integrand = np.zeros(npts)

    for Z, ionstage in ions:
        ionnumberdensity = ionpopdict[(Z, ionstage)]
        dfcollion_thision = dfcollion.query('Z == @Z and ionstage == @ionstage', inplace=False)
        # print(dfcollion_thision)

        for index, shell in dfcollion_thision.iterrows():
            # J = pynonthermal.collion.get_J(shell.Z, shell.ionstage, shell.ionpot_ev)
            xsvec = pynonthermal.collion.get_arxs_array_shell(engrid, shell)

            part_integrand += (ionnumberdensity * shell.ionpot_ev * xsvec / deposition_density_ev)

    return yvec * part_integrand


def get_nne_nt(engrid, yvec):
    # oneovervelocity = np.sqrt(9.10938e-31 / 2 / engrid / 1.60218e-19) / 100  # in s/cm
    # enovervelocity = engrid * oneovervelocity
    # en_tot = np.dot(yvec, enovervelocity) * (engrid[1] - engrid[0])
    nne_nt = 0.
    deltaen = (engrid[1] - engrid[0])
    for i, en in enumerate(engrid):
        # oneovervelocity = np.sqrt(9.10938e-31 / 2 / en / 1.60218e-19) / 100.
        velocity = np.sqrt(2 * en * 1.60218e-19 / 9.10938e-31) * 100.  # cm/s
        nne_nt += yvec[i] / velocity * deltaen

    return nne_nt


def get_energyindex_lteq(en_ev, engrid):
    # find energy bin lower boundary is less than or equal to search value
    # assert en_ev >= engrid[0]
    deltaen = engrid[1] - engrid[0]
    # assert en_ev < (engrid[-1] + deltaen)

    index = math.floor((en_ev - engrid[0]) / deltaen)

    if index < 0:
        return 0
    elif (index > len(engrid) - 1):
        return len(engrid) - 1
    else:
        return index


def get_energyindex_gteq(en_ev, engrid):
    # find energy bin lower boundary is greater than or equal to search value
    deltaen = engrid[1] - engrid[0]

    index = math.ceil((en_ev - engrid[0]) / deltaen)

    if index < 0:
        return 0
    elif (index > len(engrid) - 1):
        return len(engrid) - 1
    else:
        return index
