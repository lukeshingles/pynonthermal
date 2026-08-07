#!/usr/bin/env python3
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pynonthermal.constants import CLIGHT
from pynonthermal.constants import EV
from pynonthermal.constants import H
from pynonthermal.constants import ME
from pynonthermal.constants import QE

DATADIR = Path(__file__).absolute().parent / "data"


def get_betasq(en_ev: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64]:
    """Get (v/c)^2 for an electron of kinetic energy en_ev [eV], relativistically.

    The classical 2 * en_ev * EV / ME reaches one at 255 keV and is already 5 per cent high at 16 keV,
    which is within the range of emax_ev this package is used over.
    """
    gamma = np.asarray(en_ev, dtype=np.float64) * EV / (ME * CLIGHT**2) + 1.0

    return 1.0 - 1.0 / gamma**2


def electronlossfunction(energy_ev: float, n_e_cgs: float) -> float:
    # free-electron plasma loss rate (as in Kozma & Fransson 1992)
    # - dE / dX [eV / cm]
    # returns a positive number

    # return math.log(energy_ev) / energy_ev
    if n_e_cgs <= 0.0:
        # the plasma frequency would be zero, making the Coulomb logarithm infinite
        msg = f"the free-electron loss function requires a positive free electron density but n_e is {n_e_cgs}"
        raise ValueError(msg)

    n_e = n_e_cgs
    energy = energy_ev * EV  # convert eV to erg

    omegap = math.sqrt(4 * math.pi * n_e_cgs * QE**2 / ME)
    zetae = H * omegap / 2 / math.pi

    # Kozma & Fransson (1992) give separate Coulomb logarithms above and below 14 eV and the two do not
    # meet there: the branch below is lower by ln(hbar * v / (exp(eulergamma) * QE^2)), which is a step of
    # about 2-3 per cent in the loss rate for the free electron densities of interest. The step is part of
    # the published prescription (and of the ARTIS implementation this follows), so it is left in place
    # rather than smoothed with a formula that is nobody's.
    if energy_ev > 14:
        if 2 * energy <= zetae:
            # the Coulomb logarithm would be zero or negative, giving a non-positive loss rate
            msg = (
                f"the free-electron loss function is not valid at {energy_ev} eV for n_e = {n_e_cgs} cm^-3:"
                f" the plasma energy hbar*omega_p is {zetae / EV:.3g} eV, which is not below 2E."
            )
            raise ValueError(msg)
        lossfunc = n_e * 2 * math.pi * QE**4 / energy * math.log(2 * energy / zetae)
    else:
        v = math.sqrt(2 * energy / ME)  # velocity in cm/s (energy is in erg and ME in g, so cgs)
        # Kozma & Fransson (1992) eq. 2 describes the gamma in this Coulomb logarithm as "Euler's
        # constant (Schunk & Hays 1971)", but Schunk & Hays (1971, p. 114) define it by "ln gamma is
        # Euler's constant", i.e. gamma = exp(0.5772) = 1.781 rather than 0.5772 itself.
        exp_eulergamma = math.exp(np.euler_gamma)
        lossfunc = (
            n_e * 2 * math.pi * QE**4 / energy * math.log(ME * pow(v, 3) / (exp_eulergamma * pow(QE, 2) * omegap))
        )

    # lossfunc is now [erg / cm]
    return lossfunc / EV  # return as [eV / cm]


def get_n_tot(ions: Sequence[tuple[int, int]], ionpopdict: dict[tuple[int, int], float]) -> float:
    # total number density of all nuclei [cm^-3]
    n_tot = 0.0
    for Z, ion_stage in ions:
        n_tot += ionpopdict[(Z, ion_stage)]
    return n_tot


def get_Zbar(ions: Sequence[tuple[int, int]], ionpopdict: dict[tuple[int, int], float]) -> float:
    # number density-weighted average atomic number
    # i.e. protons per nucleus
    Zbar = 0.0
    n_tot = get_n_tot(ions, ionpopdict)
    for Z, ion_stage in ions:
        n_ion = ionpopdict[(Z, ion_stage)]
        Zbar += Z * n_ion / n_tot

    return Zbar


def get_energyindex_lteq(en_ev: float, engrid: npt.NDArray[np.float64], deltaen: float | None = None) -> int:
    # find energy bin lower boundary is less than or equal to search value.
    # deltaen is the grid spacing, which callers that already hold it should pass in: re-deriving it
    # from the array costs more than the index arithmetic itself when called in a loop.
    if deltaen is None:
        deltaen = float(engrid[1]) - float(engrid[0])

    index = math.floor((en_ev - float(engrid[0])) / deltaen)

    return 0 if index < 0 else min(index, len(engrid) - 1)


def get_energyindex_gteq(en_ev: float, engrid: npt.NDArray[np.float64], deltaen: float | None = None) -> int:
    # find energy bin lower boundary is greater than or equal to search value
    if deltaen is None:
        deltaen = float(engrid[1]) - float(engrid[0])

    index = math.ceil((en_ev - float(engrid[0])) / deltaen)

    return 0 if index < 0 else min(index, len(engrid) - 1)
