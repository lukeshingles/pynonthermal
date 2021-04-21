import math
import pandas as pd
import numpy as np
from pathlib import Path
from math import atan

import pynonthermal


def read_colliondata(collionfilename='collion.txt'):
    with open(Path(pynonthermal.DATADIR, collionfilename), 'r') as collionfile:
        expectedrowcount = int(collionfile.readline().strip())   # can ignore this line
        dfcollion = pd.read_csv(
            collionfile, delim_whitespace=True, header=None, names=[
                'Z', 'nelec', 'n', 'l', 'ionpot_ev', 'A', 'B', 'C', 'D'])

    dfcollion.eval('ionstage = Z - nelec + 1', inplace=True)

    return dfcollion


def Psecondary(e_p, ionpot_ev, J, e_s=-1, epsilon=-1):
    # probability distribution function for secondaries energy e_s [eV] (or equivalently the energy loss of
    # the primary electron epsilon [eV]) given a primary energy e_p [eV] for an impact ionisation event

    assert e_s >= 0 or epsilon >= 0
    # if e_p < I:
    #     return 0.
    #
    if e_s < 0:
        e_s = epsilon - ionpot_ev
    if epsilon < 0:
        epsilon = e_s + ionpot_ev

    #
    # if epsilon < I:
    #     return 0.
    # if e_s < 0:
    #     return 0.
    # if e_s > e_p - I:
    #     return 0.
    # if e_s > e_p:
    #     return 0.

    # test case: constant, always below ionisation
    # Psecondary_e_s_max = 1. / J / 2.
    # return 1. / Psecondary_e_s_max if (e_s < Psecondary_e_s_max) else 0.

    return 1. / J / atan((e_p - ionpot_ev) / 2. / J) / (1 + ((e_s / J) ** 2))


def get_J(Z, ionstage, ionpot_ev):
    # returns an energy in eV
    # values from Opal et al. 1971 as applied by Kozma & Fransson 1992
    if (ionstage == 1):
        if (Z == 2):  # He I
            return 15.8
        elif (Z == 10):  # Ne I
            return 24.2
        elif (Z == 18):  # Ar I
            return 10.0

    return 0.6 * ionpot_ev


def ar_xs(energy_ev, ionpot_ev, A, B, C, D):
    u = energy_ev / ionpot_ev
    if u <= 1:
        return 0

    return 1e-14 * (
        A * (1 - 1 / u) + B * pow((1 - 1 / u), 2) + C * math.log(u) + D * math.log(u) / u) / (u * pow(ionpot_ev, 2))


def get_arxs_array_shell(arr_enev, shell):
    ar_xs_array = np.array(
        [ar_xs(energy_ev, shell.ionpot_ev, shell.A, shell.B, shell.C, shell.D) for energy_ev in arr_enev])

    return ar_xs_array


def get_arxs_array_ion(arr_enev, dfcollion, Z, ionstage):
    ar_xs_array = np.zeros(len(arr_enev))
    dfcollion_thision = dfcollion.query('Z == @Z and ionstage == @ionstage')
    for index, shell in dfcollion_thision.iterrows():
        ar_xs_array += get_arxs_array_shell(arr_enev, shell)

    return ar_xs_array
