# functions related to Axelrod 1980 non-thermal treatment

import math
from functools import cache
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt

import pynonthermal
from pynonthermal.base import get_betasq
from pynonthermal.constants import CLIGHT
from pynonthermal.constants import EV
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

    # handed straight to callers from the cache, so a write must raise at the mutation site
    electron_binding.flags.writeable = False

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

    # handed straight to callers from the cache, so a write must raise at the mutation site
    shells_q.flags.writeable = False

    return shells_q


@cache
def get_shell_occupancies(atomic_number: int, ion_stage: int) -> npt.NDArray[np.int64]:
    # electrons in each shell of one ion, taken from the neutral configuration with the outermost
    # electrons removed. Cached, like the two tables it reads, because every ionisation cross section
    # evaluation needs it and it depends only on the ion.
    electron_binding = get_binding_energies()
    all_shells_q = get_shell_configs()
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

    # the cached array is handed straight to callers, so make a write raise at the mutation site
    # rather than silently changing the occupancies every later cross section evaluation sees
    element_shells_q.flags.writeable = False

    return element_shells_q


def get_sum_q_over_binding_energy(atomic_number: int, ion_stage: int, ionpot_ev: float) -> float:
    # LJS: translated from artis nonthermal.cc
    electron_binding = get_binding_energies()
    q = get_shell_occupancies(atomic_number, ion_stage)

    total = 0.0
    for electron_loop in range(q.size):
        electronsinshell = q[electron_loop]
        if (electronsinshell) > 0:
            enbinding = electron_binding[atomic_number - 1][electron_loop]
            ionpot = ionpot_ev * EV
            if enbinding <= 0:
                # fall back to the next shell in, which a negative index would silently
                # turn into the outermost shell instead
                if electron_loop == 0:
                    msg = f"No binding energy for the innermost shell of Z={atomic_number}"
                    raise ValueError(msg)
                enbinding = electron_binding[atomic_number - 1][electron_loop - 1]
                assert enbinding > 0

            total += electronsinshell / max(enbinding, ionpot)

    return total


def get_workfn_ev(atomic_number: int, ion_stage: int, ionpot_ev: float, Zbar: float) -> float:
    # the Axelrod 1980 high-energy-limit approximation to the work per ion pair W, an estimate
    # of the effective ionisation potential without solving the Spencer-Fano equation:
    # 1/W = sigma / L with both taken in their high-energy limits and losses to the free
    # electrons neglected, which reduces to the shell-occupancy-over-binding-energy sum
    binding = get_sum_q_over_binding_energy(atomic_number, ion_stage, ionpot_ev)
    Aconst = 1.33e-14 * EV * EV
    oneoverW = Aconst * binding / Zbar / (2 * math.pi * pow(QE, 4))

    return (1 / oneoverW) / EV


def get_lotz_xs_ionisation_vec(
    shell: dict[str, int | float], arr_en_ev: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    # Axelrod 1980 Eq 3.38 evaluated at an array of energies [eV]

    arr_en_erg = arr_en_ev * EV

    # relativistic, to match the relativistic correction terms in the Axelrod equation below. The
    # classical form reaches one at 255 keV, above which every cross section was silently set to zero.
    betasq = get_betasq(arr_en_ev)

    atomic_number = int(shell["Z"])
    ion_stage = int(shell["ion_stage"])
    ionpot_ev = shell["ionpot_ev"]
    shellindex = -int(shell["l"])

    electronsinshell = get_shell_occupancies(atomic_number, ion_stage)[shellindex]

    p = ionpot_ev * EV
    Aconst = 1.33e-14 * EV * EV

    # WARNING: The Axelrod equation uses both ln() and log10(), but the log10() term is likely a typo and has been
    # corrected to ln(). Fortunately, at our typical 16 keV value of EMAX, 511 keV electrons are only mildly
    # relativistic and the log(1 - beta^2) term is small anyway.
    valid = arr_en_erg > p
    with np.errstate(divide="ignore", invalid="ignore"):
        part_sigma_shell = (
            electronsinshell / p * (np.log(betasq * ME * CLIGHT**2 / 2.0 / p) - np.log(1 - betasq) - betasq)
        )
        xs = 2 * Aconst / betasq / ME / CLIGHT**2 * part_sigma_shell

    return np.where(valid & (part_sigma_shell > 0), xs, 0.0)
