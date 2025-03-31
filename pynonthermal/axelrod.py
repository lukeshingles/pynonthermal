# functions related to Axelrod 1980 non-thermal treatment

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt

import pynonthermal
from pynonthermal.constants import CLIGHT
from pynonthermal.constants import EV
from pynonthermal.constants import H
from pynonthermal.constants import ME
from pynonthermal.constants import QE


@lru_cache
def get_binding_energies() -> npt.NDArray[np.float64]:
    collionfilepath = Path(pynonthermal.DATADIR, "binding_energies_lotz_tab1and2.txt")

    with collionfilepath.open() as f:
        line = f.readline()
        while line.startswith("#"):
            line = f.readline()
        nt_shells, num_elements = (int(x) for x in line.split())
        electron_binding = np.zeros((num_elements, nt_shells))

        for i in range(num_elements):
            line = f.readline()
            while line.startswith("#"):
                line = f.readline()
            linesplit = line.split()
            assert len(linesplit) == nt_shells + 1
            assert int(linesplit[0]) == i + 1
            electron_binding[i] = np.array([float(x) for x in linesplit[1:]]) * EV

    return electron_binding


@lru_cache
def get_shell_configs() -> npt.NDArray[np.int64]:
    shellfilepath = Path(pynonthermal.DATADIR, "electron_shell_occupancy.txt")

    with shellfilepath.open() as f:
        line = f.readline()
        while line.startswith("#"):
            line = f.readline()
        nt_shells, num_elements = (int(x) for x in line.split())
        shells_q = np.zeros((num_elements, nt_shells), dtype=int)

        for i in range(num_elements):
            line = f.readline()
            while line.startswith("#"):
                line = f.readline()
            linesplit = line.split()
            assert len(linesplit) == nt_shells + 1
            assert int(linesplit[0]) == i + 1
            shells_q[i, :] = np.array([int(x) for x in linesplit[1:]])
            assert sum(shells_q[i]) == i + 1

    return shells_q


def get_shell_occupancies(
    atomic_number: int, ion_stage: int, electron_binding: npt.NDArray[np.float64], all_shells_q: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    nbound = atomic_number - ion_stage + 1
    element_shells_q_neutral = all_shells_q[atomic_number - 1]
    shellcount = min(len(element_shells_q_neutral), len(electron_binding[atomic_number - 1]))
    element_shells_q = np.zeros_like(element_shells_q_neutral)

    electron_count = 0
    for shellindex in range(shellcount):
        electronsinshell_neutral = element_shells_q_neutral[shellindex]

        electronsinshell = 0
        if (electron_count + electronsinshell_neutral) <= nbound:
            electronsinshell = electronsinshell_neutral
        else:
            electronsinshell = nbound - electron_count
        assert electronsinshell <= electronsinshell_neutral
        element_shells_q[shellindex] = electronsinshell
        electron_count += electronsinshell
        assert electron_count <= nbound

    assert sum(element_shells_q) == nbound

    return element_shells_q


def get_sum_q_over_binding_energy(atomic_number: int, ion_stage: int, ionpot_ev: float) -> float:
    # LJS: translated from artis nonthermal.cc
    electron_binding = get_binding_energies()
    all_shells_q = get_shell_configs()
    q = get_shell_occupancies(atomic_number, ion_stage, electron_binding, all_shells_q)

    total = 0.0
    for electron_loop in range(q.size):
        electronsinshell = q[electron_loop]
        if (electronsinshell) > 0:
            enbinding = electron_binding[atomic_number - 1][electron_loop]
            ionpot = ionpot_ev * EV
            if enbinding <= 0:
                enbinding = electron_binding[atomic_number - 1][electron_loop - 1]
                assert enbinding > 0

            total += electronsinshell / max(enbinding, ionpot)

    return total


def get_workfn_ev(atomic_number: int, ion_stage: int, ionpot_ev: float, Zbar: float) -> float:
    binding = get_sum_q_over_binding_energy(atomic_number, ion_stage, ionpot_ev)
    Aconst = 1.33e-14 * EV * EV
    oneoverW = Aconst * binding / Zbar / (2 * math.pi * pow(QE, 4))

    return (1 / oneoverW) / EV


def get_lotz_xs_ionisation(shell: dict[str, int | float], en_ev: float) -> float:
    # Axelrod 1980 Eq 3.38

    en_erg = en_ev * EV
    # gamma = en_erg / (ME * CLIGHT**2) + 1
    # beta = math.sqrt(1.0 - 1.0 / (gamma**2))

    beta = math.sqrt(2 * en_erg / ME) / CLIGHT
    betasq = beta**2
    # beta = 0.99
    # print(f'{gamma=} {beta=}')
    atomic_number = int(shell["Z"])
    ion_stage = int(shell["ion_stage"])
    ionpot_ev = shell["ionpot_ev"]
    shellindex = -int(shell["l"])

    electron_binding = get_binding_energies()
    all_shells_q = get_shell_configs()
    electronsinshell = get_shell_occupancies(atomic_number, ion_stage, electron_binding, all_shells_q)[shellindex]

    p = ionpot_ev * EV

    # if 0.5 * betasq * ME * CLIGHT**2 > p:
    if en_erg > p:
        part_sigma_shell = (
            electronsinshell / p * (math.log(betasq * ME * CLIGHT**2 / 2.0 / p) - math.log10(1 - betasq) - betasq)
        )

        if part_sigma_shell > 0:
            Aconst = 1.33e-14 * EV * EV
            # me is electron mass
            return 2 * Aconst / betasq / ME / (CLIGHT**2) * part_sigma_shell

    return 0.0


def get_Latom_axelrod(Zboundbar: float, en_ev: float) -> float:
    # Axelrod 1980 Eq 3.21
    # Latom is 1/N * dE/dX where E is in erg
    # should be units of erg cm^2

    en_erg = en_ev * EV

    # relativistic
    gamma = en_erg / (ME * CLIGHT**2) + 1
    beta = math.sqrt(1.0 - 1.0 / (gamma**2))
    vel = beta * CLIGHT  # in cm/s

    # classical
    # vel = math.sqrt(2. * en_erg / ME)
    # beta = vel / CLIGHT

    # I = ionpot_ev * EV
    I = 280 * EV  # assumed in Axelrod thesis

    if 2 * ME * vel**2 < I:
        return 0.0

    # if beta > 1.:
    #     print(vel, beta)
    #     beta = 0.9999

    return (
        4
        * math.pi
        * QE**4
        / (ME * vel**2)
        * Zboundbar
        * (math.log(2 * ME * vel**2 / I) + math.log(1.0 / (1.0 - beta**2)) - beta**2)
    )


def get_Lelec_axelrod(en_ev: float, n_e: float, n_e_tot: float, n_tot: float) -> float:
    # - 1/N * dE / dX [erg cm^2]
    # returns a positive number

    # Axelrod Eq 3.36 (classical low energy limit)

    # return 1.95e-13 * math.log(3.2e4 * en_ev) / en_ev

    # Axelrod 1980 Eq 3.24

    HBAR = H / 2.0 / math.pi
    en_erg = en_ev * EV
    gamma = en_erg / (ME * CLIGHT**2) + 1
    beta = math.sqrt(1.0 - 1.0 / (gamma**2))
    vel = beta * CLIGHT  # in cm/s
    omegap = 5.64e4 * math.sqrt(n_e)  # in per second
    return (
        4
        * math.pi
        * QE**4
        / (ME * vel**2)
        * n_e
        / n_tot
        * (math.log(2 * ME * vel**2 / (HBAR * omegap)) + 0.5 * math.log(1.0 / (1.0 - beta**2)) - 0.5 * beta**2)
    )


def electronlossfunction_axelrod(en_ev: float, n_e: float, n_e_tot: float) -> float:
    # - dE / dX [erg / cm]
    # returns a positive number

    return get_Lelec_axelrod(en_ev, n_e=n_e, n_e_tot=n_e_tot, n_tot=1)
