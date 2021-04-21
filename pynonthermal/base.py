#!/usr/bin/env python3
import math
import numba

from pathlib import Path

from .constants import EV, H, ME, QE

DATADIR = Path(__file__).absolute().parent / 'data'

experiment_use_Latom_in_spencerfano = False


def electronlossfunction(energy_ev, n_e_cgs):
    # free-electron plasma loss rate (as in Kozma & Fransson 1992)
    # - dE / dX [eV / cm]
    # returns a positive number

    # return math.log(energy_ev) / energy_ev
    n_e = n_e_cgs
    energy = energy_ev * EV  # convert eV to erg

    # omegap = math.sqrt(4 * math.pi * n_e_cgs * pow(QE, 2) / ME)
    omegap = 5.64e4 * math.sqrt(n_e_cgs)
    zetae = H * omegap / 2 / math.pi

    if energy_ev > 14:
        assert 2 * energy > zetae
        lossfunc = n_e * 2 * math.pi * QE ** 4 / energy * math.log(2 * energy / zetae)
    else:
        v = math.sqrt(2 * energy / ME)  # velocity in m/s
        eulergamma = 0.577215664901532
        lossfunc = n_e * 2 * math.pi * QE ** 4 / energy * math.log(ME * pow(v, 3) / (eulergamma * pow(QE, 2) * omegap))

    # lossfunc is now [erg / cm]
    return lossfunc / EV  # return as [eV / cm]


def get_n_tot(ions, ionpopdict):
    # total number density of all nuclei [cm^-3]
    n_tot = 0.
    for Z, ionstage in ions:
        n_tot += ionpopdict[(Z, ionstage)]
    return n_tot


def get_Zbar(ions, ionpopdict):
    # number density-weighted average atomic number
    # i.e. protons per nucleus
    Zbar = 0.
    n_tot = get_n_tot(ions, ionpopdict)
    for Z, ionstage in ions:
        n_ion = ionpopdict[(Z, ionstage)]
        Zbar += Z * n_ion / n_tot

    return Zbar


def get_Zboundbar(ions, ionpopdict):
    # number density-weighted average number of bound electrons per nucleus
    Zboundbar = 0.
    n_tot = get_n_tot(ions, ionpopdict)
    for Z, ionstage in ions:
        n_ion = ionpopdict[(Z, ionstage)]
        Zboundbar += (Z - ionstage + 1) * n_ion / n_tot

    return Zboundbar


@numba.njit()
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


@numba.njit()
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
