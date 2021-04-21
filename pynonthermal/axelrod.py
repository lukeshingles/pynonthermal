# functions related to Axelrod 1980 non-thermal treatment

import math
import numpy as np
from pathlib import Path

import pynonthermal

from pynonthermal.constants import CLIGHT, EV, H, ME, QE


def read_binding_energies(modelpath=None):
    collionfilename = Path(pynonthermal.DATADIR, 'binding_energies.txt')

    with open(collionfilename, "r") as f:
        nt_shells, n_z_binding = [int(x) for x in f.readline().split()]
        electron_binding = np.zeros((n_z_binding, nt_shells))

        for i in range(n_z_binding):
            electron_binding[i] = np.array([float(x) for x in f.readline().split()]) * EV

    return electron_binding


def get_electronoccupancy(atomic_number, ion_stage, nt_shells):
    # adapted from ARTIS code
    q = np.zeros(nt_shells)

    ioncharge = ion_stage - 1
    nbound = atomic_number - ioncharge  # number of bound electrons

    for electron_loop in range(nbound):
        if q[0] < 2:  # K 1s
            q[0] += 1
        elif(q[1] < 2):  # L1 2s
            q[1] += 1
        elif(q[2] < 2):  # L2 2p[1/2]
            q[2] += 1
        elif(q[3] < 4):  # L3 2p[3/2]
            q[3] += 1
        elif(q[4] < 2):  # M1 3s
            q[4] += 1
        elif(q[5] < 2):  # M2 3p[1/2]
            q[5] += 1
        elif(q[6] < 4):  # M3 3p[3/2]
            q[6] += 1
        elif ioncharge == 0:
            if q[9] < 2:  # N1 4s
                q[9] += 1
            elif q[7] < 4:  # M4 3d[3/2]
                q[7] += 1
            elif q[8] < 6:  # M5 3d[5/2]
                q[8] += 1
            else:
                print("Going beyond the 4s shell in NT calculation. Abort!\n")
        elif ioncharge == 1:
            if q[9] < 1:  # N1 4s
                q[9] += 1
            elif q[7] < 4:  # M4 3d[3/2]
                q[7] += 1
            elif q[8] < 6:  # M5 3d[5/2]
                q[8] += 1
            else:
                print("Going beyond the 4s shell in NT calculation. Abort!\n")
        elif(ioncharge > 1):
            if q[7] < 4:  # M4 3d[3/2]
                q[7] += 1
            elif q[8] < 6:  # M5 3d[5/2]
                q[8] += 1
            else:
                print("Going beyond the 4s shell in NT calculation. Abort!\n")
    return q


def get_mean_binding_energy(atomic_number, ion_stage, electron_binding, ionpot_ev):
    # LJS: this came from ARTIS and I'm not sure what this actually is - inverse binding energy? electrons per erg?
    n_z_binding, nt_shells = electron_binding.shape
    q = get_electronoccupancy(atomic_number, ion_stage, nt_shells)

    total = 0.0
    for electron_loop in range(nt_shells):
        electronsinshell = q[electron_loop]
        if ((electronsinshell) > 0):
            use2 = electron_binding[atomic_number - 1][electron_loop]
            use3 = ionpot_ev * EV
        if (use2 <= 0):
            use2 = electron_binding[atomic_number - 1][electron_loop - 1]
            # to get total += electronsinshell/electron_binding[get_element(element)-1][electron_loop-1];
            # set use3 = 0.
            if (electron_loop != 8):
                # For some reason in the Lotz data, this is no energy for the M5 shell before Ni. So if the complaint
                # is for 8 (corresponding to that shell) then just use the M4 value
                print("WARNING: I'm trying to use a binding energy when I have no data. "
                      f"element {atomic_number} ionstage {ion_stage}\n")
                assert(electron_loop == 8)
                # print("Z = %d, ion_stage = %d\n", get_element(element), get_ionstage(element, ion));
        if (use2 < use3):
            total += electronsinshell / use3
        else:
            total += electronsinshell / use2
        # print("total total)

    return total


def get_mean_binding_energy_alt(atomic_number, ion_stage, electron_binding, ionpot_ev):
    # LJS: this should be mean binding energy [erg] per electron
    n_z_binding, nt_shells = electron_binding.shape
    q = get_electronoccupancy(atomic_number, ion_stage, nt_shells)

    total = 0.0
    ecount = 0
    for electron_loop in range(nt_shells):
        electronsinshell = q[electron_loop]
        ecount += electronsinshell
        if ((electronsinshell) > 0):
            use2 = electron_binding[atomic_number - 1][electron_loop]
            use3 = ionpot_ev * EV
        if (use2 <= 0):
            use2 = electron_binding[atomic_number - 1][electron_loop - 1]
            # to get total += electronsinshell/electron_binding[get_element(element)-1][electron_loop-1];
            # set use3 = 0.
            if (electron_loop != 8):
                # For some reason in the Lotz data, this is no energy for the M5 shell before Ni. So if the complaint
                # is for 8 (corresponding to that shell) then just use the M4 value
                print("WARNING: I'm trying to use a binding energy when I have no data. "
                      f"element {atomic_number} ionstage {ion_stage}\n")
                assert(electron_loop == 8)
                # print("Z = %d, ion_stage = %d\n", get_element(element), get_ionstage(element, ion));
        if (use2 < use3):
            total += electronsinshell * use3
        else:
            total += electronsinshell * use2

    assert ecount == (atomic_number - ion_stage + 1)

    return total / ecount


def get_lotz_xs_ionisation(atomic_number, ion_stage, electron_binding, ionpot_ev, en_ev):
    # Axelrod 1980 Eq 3.38

    en_erg = en_ev * EV
    gamma = en_erg / (ME * CLIGHT ** 2) + 1
    beta = math.sqrt(1. - 1. / (gamma ** 2))
    # beta = 0.99
    # print(f'{gamma=} {beta=}')

    n_z_binding, nt_shells = electron_binding.shape
    q = get_electronoccupancy(atomic_number, ion_stage, nt_shells)

    part_sigma = 0.0
    for electron_loop in range(nt_shells):
        electronsinshell = q[electron_loop]
        if ((electronsinshell) > 0):
            use2 = electron_binding[atomic_number - 1][electron_loop]
            use3 = ionpot_ev * EV
        if (use2 <= 0):
            use2 = electron_binding[atomic_number - 1][electron_loop - 1]
            # to get total += electronsinshell/electron_binding[get_element(element)-1][electron_loop-1];
            # set use3 = 0.
            if (electron_loop != 8):
                # For some reason in the Lotz data, this is no energy for the M5 shell before Ni. So if the complaint
                # is for 8 (corresponding to that shell) then just use the M4 value
                print("WARNING: I'm trying to use a binding energy when I have no data. "
                      f"element {atomic_number} ionstage {ion_stage}\n")
                assert(electron_loop == 8)
                # print("Z = %d, ion_stage = %d\n", get_element(element), get_ionstage(element, ion));

        if (use2 < use3):
            p = use3
        else:
            p = use2

        if 0.5 * beta ** 2 * ME * CLIGHT ** 2 > p:
            part_sigma += electronsinshell / p * (
                (math.log(beta ** 2 * ME * CLIGHT ** 2 / 2. / p) - math.log10(1 - beta ** 2) - beta ** 2))

    Aconst = 1.33e-14 * EV * EV
    # me is electron mass
    sigma = 2 * Aconst / (beta ** 2) / ME / (CLIGHT ** 2) * part_sigma
    assert(sigma >= 0)
    return sigma


def get_Latom_axelrod(Zboundbar, en_ev):
    # Axelrod 1980 Eq 3.21
    # Latom is 1/N * dE/dX where E is in erg
    # should be units of erg cm^2

    en_erg = en_ev * EV

    # relativistic
    gamma = en_erg / (ME * CLIGHT ** 2) + 1
    beta = math.sqrt(1. - 1. / (gamma ** 2))
    vel = beta * CLIGHT  # in cm/s

    # classical
    # vel = math.sqrt(2. * en_erg / ME)
    # beta = vel / CLIGHT

    # I = ionpot_ev * EV
    I = 280 * EV  # assumed in Axelrod thesis

    if 2 * ME * vel ** 2 < I:
        return 0.

    # if beta > 1.:
    #     print(vel, beta)
    #     beta = 0.9999

    return 4 * math.pi * QE ** 4 / (ME * vel ** 2) * Zboundbar * (
        math.log(2 * ME * vel ** 2 / I) + math.log(1. / (1. - beta ** 2)) - beta ** 2)


def get_Lelec_axelrod(en_ev, n_e, n_e_tot, n_tot):
    # - 1/N * dE / dX [erg cm^2]
    # returns a positive number

    # Axelrod Eq 3.36 (classical low energy limit)

    # return 1.95e-13 * math.log(3.2e4 * en_ev) / en_ev

    # Axelrod 1980 Eq 3.24

    HBAR = H / 2. / math.pi
    en_erg = en_ev * EV
    gamma = en_erg / (ME * CLIGHT ** 2) + 1
    beta = math.sqrt(1. - 1. / (gamma ** 2))
    vel = beta * CLIGHT  # in cm/s
    omegap = 5.64e4 * math.sqrt(n_e)  # in per second
    return 4 * math.pi * QE ** 4 / (ME * vel ** 2) * n_e / n_tot * (
        math.log(2 * ME * vel ** 2 / (HBAR * omegap)) + 0.5 * math.log(1. / (1. - beta ** 2)) - 0.5 * beta ** 2)


def electronlossfunction_axelrod(en_ev, n_e, n_e_tot):
    # - dE / dX [erg / cm]
    # returns a positive number

    return get_Lelec_axelrod(en_ev, n_e=n_e, n_e_tot=n_e_tot, n_tot=1)
