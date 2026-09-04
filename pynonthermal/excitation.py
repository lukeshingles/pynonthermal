import math
import typing as t
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

import pynonthermal
from pynonthermal.constants import CLIGHT
from pynonthermal.constants import EV
from pynonthermal.constants import H
from pynonthermal.constants import H_ionpot
from pynonthermal.constants import ME
from pynonthermal.constants import QE


@dataclass(frozen=True, slots=True, eq=False)
class ExcitationTransition:
    """One bound-bound excitation that the solver holds, as SpencerFanoSolver.add_excitation() records it.

    The cross section is an array on the solver energy grid, because every reader needs it only
    there: the matrix fill, the excitation fraction of Kozma & Fransson 1992 equation 9, and the
    single interpolated point that calculate_N_e() takes at energy_ev + epsilon_trans_ev.
    """

    levelnumberdensity: float
    """The population density of the lower level [cm^-3]."""

    xs_vec: npt.NDArray[np.float64]
    """The cross section [cm^2] at every energy of SpencerFanoSolver.engrid."""

    epsilon_trans_ev: float
    """The transition energy [eV]."""


def get_xs_excitation_vector(
    engrid: npt.NDArray[np.float64], row: dict[str, t.Any], use_collstrengths: bool = True
) -> npt.NDArray[np.float64]:
    """Get an array containing the excitation cross section in cm^2 at every energy in the array engrid (eV).

    This is the sigma of the excitation term in the degradation equation (Kozma & Fransson 1992
    equation 7): computed from the transition's collision strength via Li et al. 2012 equation 11
    when one is available, and otherwise from the oscillator strength of a permitted E1 transition
    via the van Regemorter 1962 approximation, with the g-bar factor g_bar = 0.15 + 0.28 ln(U)
    built from the A and D ln(U) terms of the fitting formula in equation 5 of Mewe 1972
    (described as its "first two terms" by Shingles et al. 2020, section 2.5).
    """
    A_naught_squared = 2.800285203e-17  # Bohr radius squared in cm^2
    npts = len(engrid)
    xs_excitation_vec = np.empty(npts)

    coll_str = row["collstr"]
    epsilon_trans_ev = row["epsilon_trans_ev"]
    assert isinstance(epsilon_trans_ev, float)
    epsilon_trans = epsilon_trans_ev * EV

    if epsilon_trans_ev > engrid[-1]:
        # no electron on the grid has enough energy to drive this transition. Without this check,
        # get_energyindex_gteq clamps to the last grid point and leaves a spurious cross section
        # there, which for the E1 branch below comes out negative (g_bar < 0 for U < 0.585).
        return np.zeros(npts)

    startindex = pynonthermal.get_energyindex_gteq(en_ev=epsilon_trans_ev, engrid=engrid)
    xs_excitation_vec[:startindex] = 0.0

    if coll_str >= 0 and use_collstrengths:
        # collision strength is available, so use it
        # Li et al. 2012 equation 11: sigma = pi * a_0^2 * (H_ionpot / E) * coll_str / lower_g,
        # with k_i^2 = E / H_ionpot in units of the inverse Bohr radius squared
        constantfactor = H_ionpot / row["lower_g"] * coll_str * math.pi * A_naught_squared

        xs_excitation_vec[startindex:] = constantfactor / (engrid[startindex:] * EV)

    elif not row["forbidden"]:
        nu_trans = epsilon_trans / H
        g = row["upper_g"] / row["lower_g"]
        fij = g * ME * pow(CLIGHT, 3) / (8 * pow(QE * nu_trans * math.pi, 2)) * row["A"]
        # permitted E1 electric dipole transitions

        # Mewe (1972) equation 5 fits g(U) = A + B/U + C/U^2 + D*ln(U); keep the A and D ln(U)
        # terms, with the D = sqrt(3)/(2 pi) that Mewe recommends for all optically allowed
        # transitions rounded to 0.28 (Shingles et al. 2020, section 2.5, where this pair is
        # described as the formula's "first two terms")
        mewe_A = 0.15
        mewe_D = 0.28

        prefactor = 45.585750051
        # van Regemorter (1962) approximation with the g_bar below from Mewe (1972)
        constantfactor = prefactor * A_naught_squared * pow(H_ionpot / epsilon_trans, 2) * fij

        U = engrid[startindex:] / epsilon_trans_ev
        # g_bar = 0.2
        g_bar = mewe_D * np.log(U) + mewe_A

        xs_excitation_vec[startindex:] = constantfactor * g_bar / U
    else:
        xs_excitation_vec[startindex:] = 0.0

    return xs_excitation_vec
