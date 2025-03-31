import math
import typing as t

import numpy as np
import numpy.typing as npt

import pynonthermal
from pynonthermal.constants import CLIGHT
from pynonthermal.constants import EV
from pynonthermal.constants import H
from pynonthermal.constants import H_ionpot
from pynonthermal.constants import ME
from pynonthermal.constants import QE

use_collstrengths = False


def get_xs_excitation(en_ev: float, row: dict[str, t.Any]) -> float:
    """Get the excitation cross section in cm^2 at energy en_ev [eV]."""
    A_naught_squared = 2.800285203e-17  # Bohr radius squared in cm^2

    coll_str = row["collstr"]
    epsilon_trans_ev = row["epsilon_trans_ev"]
    assert isinstance(epsilon_trans_ev, float)
    epsilon_trans = epsilon_trans_ev * EV

    if en_ev < epsilon_trans_ev:
        return 0.0

    if coll_str >= 0 and use_collstrengths:
        # collision strength is available, so use it
        # Li et al. 2012 equation 11
        constantfactor = pow(H_ionpot, 2) / row["lower_g"] * coll_str * math.pi * A_naught_squared

        return constantfactor * (en_ev * EV) ** -2

    if not row["forbidden"]:
        nu_trans = epsilon_trans / H
        g = row["upper_g"] / row["lower_g"]
        fij: float = g * ME * pow(CLIGHT, 3) / (8 * pow(QE * nu_trans * math.pi, 2)) * row["A"]
        # permitted E1 electric dipole transitions

        g_bar = 0.2

        A = 0.28
        B = 0.15

        prefactor = 45.585750051
        # Eq 4 of Mewe 1972, possibly from Seaton 1962?
        constantfactor = prefactor * A_naught_squared * pow(H_ionpot / epsilon_trans, 2) * fij

        U = en_ev / epsilon_trans_ev
        g_bar = A * np.log(U) + B

        return constantfactor * g_bar / U

    return 0.0


def get_xs_excitation_vector(engrid: npt.NDArray[np.float64], row: dict[str, t.Any]) -> npt.NDArray[np.float64]:
    """Get an array containing the excitation cross section in cm^2 at every energy in the array engrid (eV)."""
    A_naught_squared = 2.800285203e-17  # Bohr radius squared in cm^2
    npts = len(engrid)
    xs_excitation_vec = np.empty(npts)

    coll_str = row["collstr"]
    epsilon_trans_ev = row["epsilon_trans_ev"]
    assert isinstance(epsilon_trans_ev, float)
    epsilon_trans = epsilon_trans_ev * EV

    startindex = pynonthermal.get_energyindex_gteq(en_ev=epsilon_trans_ev, engrid=engrid)
    xs_excitation_vec[:startindex] = 0.0

    if coll_str >= 0 and use_collstrengths:
        # collision strength is available, so use it
        # Li et al. 2012 equation 11
        constantfactor = pow(H_ionpot, 2) / row["lower_g"] * coll_str * math.pi * A_naught_squared

        xs_excitation_vec[startindex:] = constantfactor * (engrid[startindex:] * EV) ** -2

    elif not row["forbidden"]:
        nu_trans = epsilon_trans / H
        g = row["upper_g"] / row["lower_g"]
        fij = g * ME * pow(CLIGHT, 3) / (8 * pow(QE * nu_trans * math.pi, 2)) * row["A"]
        # permitted E1 electric dipole transitions

        g_bar = 0.2

        A = 0.28
        B = 0.15

        prefactor = 45.585750051
        # Eq 4 of Mewe 1972, possibly from Seaton 1962?
        constantfactor = prefactor * A_naught_squared * pow(H_ionpot / epsilon_trans, 2) * fij

        U = engrid[startindex:] / epsilon_trans_ev
        g_bar = A * np.log(U) + B

        xs_excitation_vec[startindex:] = constantfactor * g_bar / U
    else:
        xs_excitation_vec[startindex:] = 0.0

    return xs_excitation_vec
