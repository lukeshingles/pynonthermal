from __future__ import annotations

import dataclasses
import math
import typing as t
import warnings
from collections.abc import Mapping
from pathlib import Path

import artistools as at
import matplotlib.axes as mplax
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl

import pynonthermal
from pynonthermal.axelrod import get_workfn_ev
from pynonthermal.base import electronlossfunction
from pynonthermal.base import get_betasq
from pynonthermal.base import get_xs_on_grid
from pynonthermal.base import get_Zbar
from pynonthermal.collion import IonisationChannel
from pynonthermal.constants import CLIGHT
from pynonthermal.constants import K_B
from pynonthermal.excitation import ExcitationTransition
from pynonthermal.ionbalance import get_ion_fractions
from pynonthermal.ionbalance import get_saha_factor
from pynonthermal.ionbalance import solve_charge_neutral_n_e

if t.TYPE_CHECKING:
    from collections.abc import Sequence

# The weight of the new value when the ionisation balance mixes the ratio coefficients in log space.
# The fixed-point map ln(c_new) = F(ln(c)) has a slope between -1/2 (the balanced element gives the
# electrons and heating takes most of the energy, so Gamma is proportional to 1 / n_e and n_e to
# sqrt(c)) and 0 (fixed ions give the electrons, or ionisation takes most of the energy). A weight of
# 2/3 gives a contraction ratio of at most 1/3 over that range. The unmixed iteration alternates with
# a ratio of up to 1/2.
BALANCE_MIXING_WEIGHT: float = 2.0 / 3.0

# The maximum number of iterations of the ionisation balance in solve(). With the contraction ratio
# above, a case that needs more than a few tens of iterations has a problem that more iterations
# would not fix, so solve() raises a RuntimeError instead.
BALANCE_MAXITER: int = 100

# The top stage of a balanced element is a sink: its ionisation is an energy loss in the matrix, but
# the ions it makes have no stage to go to. In a longer chain those ions would sit in the next stage,
# and their fraction of the element is about the ionisation rate out of the top stage divided by the
# total ionisation rate of the element (which recombination balances). Warn when that ratio exceeds
# this value, because the chain then needs a higher stage.
BALANCE_TOP_STAGE_LEAK_WARN_FRACTION: float = 0.01


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class _ExcitationTemplate:
    # one bound-bound excitation of an ion with a population from the ionisation balance. The lower
    # level population is popfrac times the ion population, so the matrix band scales with the ion.
    popfrac: float
    xs_vec: npt.NDArray[np.float64]
    epsilon_trans_ev: float


@dataclasses.dataclass(slots=True, eq=False)
class _BalancedElement:
    # one element whose ion populations come from the ionisation balance in solve()
    Z: int
    n_elem: float
    ion_stages: tuple[int, ...]
    # recombination rate coefficients [cm^3 s^-1] keyed by the recombining (upper) ion stage, or None
    # for the Saha mode
    recomb_ratecoeffs: dict[int, float] | None
    # the Saha ratio coefficients n_{i+1} n_e / n_i [cm^-3] of each pair of adjacent stages, or None
    # for the recombination mode
    saha_factors: tuple[float, ...] | None
    # key is the ion stage, value is {transitionkey: template}
    excitation_templates: dict[int, dict[t.Any, _ExcitationTemplate]] = dataclasses.field(default_factory=dict)
    # key is the ion stage, value is {band width k: (vec, fracvec)} for a unit ion population, the
    # pre-summed matrix bands of the excitation templates (see SpencerFanoSolver._add_excitation_band)
    excitation_unit_bands: dict[int, dict[int, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]] = (
        dataclasses.field(default_factory=dict)
    )
    # the non-thermal ionisation rate coefficient per unit deposition rate density [cm^3 eV^-1] of each
    # stage below the top, from the last solve. The next solve starts from these values.
    ratecoeffs_per_deposition: dict[int, float] | None = None


# Nodes for the sub-grids that resolve the parts of the Kozma & Fransson equation 11 integrals that the
# solver's own energy grid cannot. Both integrands vary on the scale of J = 0.6 * ionpot_ev, so the node
# count is set from the domain width in units of J, with a floor for the narrow domains that
# calculate_frac_heating produces and a cap for the wide ones a direct calculate_N_e call can ask for.
NPTS_SUBGRID_MIN: int = 65
NPTS_SUBGRID_MAX: int = 1025
NPTS_SUBGRID_PER_J: int = 32

# how many whole grid cells above the lower limit of equation 11's secondary-electron integral (the
# one from 2E + I) get the sub-grid treatment. The integrand is steepest at that limit and smooth a
# few J above it.
NCELLS_SECOND_INTEGRAL_SUBGRID: int = 2

# Number of nodes for the integral over E in [0, E_0] of Kozma & Fransson equation 8. This one is not
# free: every node costs a full calculate_N_e() over all ions and shells, so it dominates the cost of
# the analysis. It is integrated by Simpson's rule, which is fourth order here and reaches an accuracy
# at 9 nodes that the trapezoid rule needs several hundred for. The node count is a property of this
# integral alone: the old ceil(E_0 / deltaen) * 5 tied it to the main grid, which gave 5 nodes when
# emin_ev was below the grid spacing and several hundred when it was well above.
NPTS_SUB_E0_INTEGRAL: int = 9


def solve_upper_triangular(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    diag_add: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Solve (a + diag(diag_add)) x = b for upper-triangular a by back-substitution.

    diag_add lets a purely diagonal term (like the free-electron loss function) be applied
    without copying a into a second npts x npts matrix just to modify its diagonal.
    """
    # scipy.linalg.solve_triangular rejected these by default (check_finite and the LAPACK
    # singularity flag); without the checks, bad inputs would propagate nan/inf into the
    # solution silently instead of raising
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        msg = "matrix and right-hand side must be finite (no nan or inf entries)"
        raise ValueError(msg)
    if diag_add is not None and not np.isfinite(diag_add).all():
        msg = "diagonal addition must be finite (no nan or inf entries)"
        raise ValueError(msg)
    diag = np.diagonal(a)
    if diag_add is not None:
        diag = diag + diag_add
    if np.any(diag == 0.0):
        msg = "matrix is singular: zero on the diagonal"
        raise np.linalg.LinAlgError(msg)
    n = b.shape[0]
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        # move the known terms of row i's equation to the right-hand side and divide by the
        # diagonal: a[i, i+1:] @ x[i+1:] is a dot product (@ is numpy matrix multiplication,
        # which for two 1-D vectors is their inner product) of the row's off-diagonal
        # coefficients with the already-solved higher-index part of x
        x[i] = (b[i] - a[i, i + 1 :] @ x[i + 1 :]) / diag[i]
    return x


def integrate_simpson_uniform(y: npt.NDArray[np.float64], x: npt.NDArray[np.float64]) -> float:
    """Composite Simpson's rule on a uniformly-spaced grid with an odd number of points."""
    npts = x.shape[0]
    assert npts >= 3
    assert npts % 2 == 1
    weights = np.full(npts, 2.0, dtype=np.float64)
    weights[1::2] = 4.0
    weights[0] = weights[-1] = 1.0
    h = (x[-1] - x[0]) / (npts - 1)
    return float(h / 3.0 * (weights @ y))


class SpencerFanoSolver:
    """Solve the Spencer-Fano equation for non-thermal heating, ionisation, and excitation.

    The Spencer-Fano equation describes the energy distribution y(E) of non-thermal electrons
    in a plasma as they degrade from a high-energy source (such as radioactive-decay products)
    by heating the free thermal electrons and by ionising and exciting ions. It is the form of
    the Boltzmann equation first written down for electron slowing-down by Spencer & Fano
    (1954, Phys. Rev., 93, 1172). This solver follows the supernova application of Kozma &
    Fransson (1992, ApJ 390, 602; KF92 below) as implemented in ARTIS (Shingles et al. 2020):
    the integral form of the degradation equation (KF92 equation 7; also equation 2 of Li,
    Dessart & Hillier 2012) is discretised on a uniform energy grid as an upper-triangular
    matrix equation for y at each grid energy.

    Each part of KF92 equation 7 maps onto the code as follows:

    - the thermal-electron loss term y(E) L(E): electronlossfunction() in pynonthermal.base
      (KF92 equations 1 and 2), applied along the matrix diagonal in solve()
    - the excitation term, for each transition the level population times the integral of
      y(E') sigma(E') dE' over E' in [E, E + epsilon_trans]: add_excitation() and
      add_ion_ltepopexcitation(), with cross sections from pynonthermal.excitation
      (Li et al. 2012 equation 11 from a collision strength, or the van Regemorter 1962
      approximation with the Mewe 1972 g-bar factor)
    - the ionisation term, integrals of y(E') sigma_ic(E') P(E', epsilon - I) with the channel's
      total ionisation cross section sigma_ic (pynonthermal.collion) and the secondary-electron
      energy distribution P of KF92 equation 4: add_ionisation() and add_ionisation_channel(),
      via _add_ionisation_channel_to_matrix()
    - the right-hand side, the integral of the source function from E to E_max: rhsvec,
      built in __init__ for a source spread over the top of the energy grid

    After solve(), the KF92 deposition fractions and rate coefficients are available:
    the heating fraction (KF92 equation 8) from get_frac_heating(), the excitation channels
    (KF92 equation 9) from get_frac_excitation_tot() and get_excitation_ratecoeff(), and the
    ionisation fractions (KF92 equation 10) through the effective ionisation potential
    (KF92 equation 12, summed over shells in analyse_ntspectrum()) with rate coefficients
    from get_ionisation_ratecoeff() (KF92 equation 13). calculate_N_e() evaluates N(E) of
    KF92 equation 11, the rate at which electrons appear below the solved grid.

    Every cross section is adjustable. add_excitation() and add_ionisation_channel() each take one
    custom cross section, as an array of cross sections at every energy of the solver energy grid.

    If heating_only_approximation is True, the solver removes the excitation and ionisation
    loss terms from the matrix and keeps only the heating loss. The solver still stores the
    excitation transitions and the ionisation cross sections, and it calculates the channel
    fractions and rate coefficients from the heating-only solution. The channel fractions
    then do not sum to one.

    The ion populations of an element can also come from an ionisation balance instead of
    from the caller. add_element_ionbalance() takes recombination rate coefficients, and
    solve() then iterates the non-thermal ionisation rates against recombination until the
    populations converge. add_element_saha() takes a temperature and uses the Saha equation.
    In both cases solve() finds the charge-neutral free electron density, and the converged
    populations are in ionpopdict after solve().
    """

    _solved: bool
    _analysed: bool
    _frac_heating: float | None
    _frac_ionisation_tot: float
    _frac_excitation_tot: float
    _frac_ionisation_ion: dict[tuple[int, int], float]
    _frac_excitation_ion: dict[tuple[int, int], float]
    _eff_ionpot: dict[tuple[int, int], float]
    _nt_ionisation_ratecoeff: dict[tuple[int, int], float]
    _ionisation_channels: dict[tuple[int, int], list[IonisationChannel]]
    depositionratedensity_ev: float
    ionpopdict: dict[tuple[int, int], float]
    excitationlists: dict[tuple[int, int], dict[t.Any, ExcitationTransition]]
    verbose: bool
    heating_only_approximation: bool
    _n_e: float | None
    _n_e_override: float | None
    engrid: npt.NDArray[np.float64]
    deltaen: float
    dfcollion: pl.DataFrame
    rhsvec: npt.NDArray[np.float64]
    E_init_ev: float
    sfmatrix: npt.NDArray[np.float64]
    adata_polars: pl.DataFrame | None
    yvec: npt.NDArray[np.float64]
    _balanced_elements: dict[int, _BalancedElement]
    balance_iterations: int
    temperature: float | None

    def __init__(
        self,
        emin_ev: float = 1.0,
        emax_ev: float = 3000.0,
        npts: int = 4096,
        verbose: bool = False,
        use_ar1985: bool = False,
        heating_only_approximation: bool = False,
    ) -> None:
        """Make a solver with a uniform linear energy grid and the given options.

        If heating_only_approximation is True, the solver removes the excitation and
        ionisation loss terms from the matrix and keeps only the heating loss.
        """
        if npts < 2:
            msg = f"npts must be at least 2 to define an energy grid spacing but is {npts}"
            raise ValueError(msg)
        if emin_ev <= 0.0:
            # the loss function and the cross section formulae all diverge at zero energy
            msg = f"emin_ev must be greater than zero but is {emin_ev}"
            raise ValueError(msg)
        if emax_ev <= emin_ev:
            msg = f"emax_ev ({emax_ev}) must be greater than emin_ev ({emin_ev})"
            raise ValueError(msg)

        self._solved = False
        self._n_e = None
        self._n_e_override = None
        self.reset_solution_analysis()

        # key is (Z, ion_stage), value is the list of that ion's ionisation channels
        self._ionisation_channels = {}

        self.ionpopdict = {}  # key is (Z, ion_stage) value is number density

        # key is (Z, ion_stage) value is {transitionkey: ExcitationTransition}
        self.excitationlists = {}

        # key is Z, value is the element whose ion populations solve() finds
        self._balanced_elements = {}
        self.balance_iterations = 0

        # the one temperature [K] of the solver, for the LTE level populations and the Saha equation.
        # The first call that gives a temperature sets it (see _resolve_temperature()).
        self.temperature = None

        self.verbose = verbose
        self.heating_only_approximation = heating_only_approximation
        self.engrid = np.linspace(emin_ev, emax_ev, num=npts, endpoint=True, dtype=float)
        # handed to the cross section functions of the caller, so a write must raise at the
        # mutation site. An in-place write would otherwise leave the grid and deltaen inconsistent.
        self.engrid.flags.writeable = False
        self.deltaen = self.engrid[1] - self.engrid[0]

        self.dfcollion = pynonthermal.collion.read_colliondata(
            collionfilename=("collion-AR1985.txt" if use_ar1985 else "collion.txt")
        )

        sourcevec = np.zeros(self.engrid.shape)
        # spread the source over the top 1/30th of the energy range
        # (0.1 keV for a 3 keV Emax) to match Kozma & Fransson 1992
        source_spread_pts = math.ceil(npts / 30.0)
        if source_spread_pts < 1:
            msg = "source_spread_pts must be at least 1"
            raise ValueError(msg)

        sourcevec[npts - source_spread_pts :] = 1.0 / (self.deltaen * source_spread_pts)

        source_emin = self.engrid[np.flatnonzero(sourcevec)[0]]
        source_emax = self.engrid[np.flatnonzero(sourcevec)[-1]]

        # integral of the source from each energy to the top of the grid: the right-hand side
        # of the integral form of the degradation equation (Kozma & Fransson 1992 equation 7)
        self.rhsvec = np.cumsum((sourcevec * self.deltaen)[::-1])[::-1]

        # E_init_ev is the deposition rate density that we assume when solving the SF equation.
        # The solution will be scaled to the true deposition rate later
        self.E_init_ev = np.dot(self.engrid, sourcevec) * self.deltaen

        self.adata_polars = None

        if self.verbose:
            print(
                f"\nSetting up Spencer-Fano equation with {npts} energy points from"
                f" {self.engrid[0]} to {self.engrid[-1]} eV..."
            )
            print(
                f"  source is a box function from {source_emin:.2f} to"
                f" {source_emax:.2f} eV with E_init {self.E_init_ev:7.2f} [eV/s/cm3]"
            )

        self.sfmatrix = np.zeros((npts, npts))

    def __enter__(self) -> t.Self:
        """Enter the context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit the context manager."""

    def get_energyindex_lteq(self, en_ev: float) -> int:
        return pynonthermal.get_energyindex_lteq(en_ev, engrid=self.engrid)

    def get_energyindex_gteq(self, en_ev: float) -> int:
        return pynonthermal.get_energyindex_gteq(en_ev, engrid=self.engrid)

    def electronlossfunction(self, en_ev: float) -> float:
        return electronlossfunction(en_ev, self.get_n_e())

    def _require_solved(self) -> None:
        if not self._solved:
            msg = "The Spencer-Fano equation must be solved first. Call solve()."
            raise RuntimeError(msg)

    def _require_not_solved(self, action: str) -> None:
        if self._solved:
            msg = f"Can't {action} after solving the Spencer-Fano equation"
            raise RuntimeError(msg)

    def _get_all_ions(self) -> list[tuple[int, int]]:
        # every ion with an ionisation or an excitation channel: ions with a registered population
        # first (in population registration order), then ions that only have manually-added
        # excitation channels (in the order their first excitation was added)
        return list(dict.fromkeys([*self.ionpopdict, *self.excitationlists]))

    def _resolve_temperature(self, temperature: float | None) -> float:
        # the solver has one temperature. A call that gives one must agree with the value that an
        # earlier call gave, and a call that gives None takes the value that is set. The caller stores
        # the result in self.temperature when every other check of the call has passed, so that a
        # rejected call leaves the solver unchanged.
        if temperature is None:
            if self.temperature is None:
                msg = (
                    "temperature is required, because no earlier call on this solver gave one. Give the"
                    " temperature in K here, or first in add_element_saha() or add_ion_ltepopexcitation()."
                )
                raise ValueError(msg)
            return self.temperature

        if not 0.0 < temperature < math.inf:
            msg = f"temperature must be greater than zero and finite but is {temperature}"
            raise ValueError(msg)
        if self.temperature is not None and not math.isclose(self.temperature, temperature, rel_tol=1e-9):
            msg = (
                f"the solver has one temperature, and an earlier call set it to {self.temperature} K, so"
                f" {temperature} K cannot be used. Leave temperature as None to use the value that is set."
            )
            raise ValueError(msg)

        return float(temperature)

    def _check_not_balanced(self, Z: int) -> None:
        # the populations of a balanced element come from solve(), so a caller cannot give one
        if Z in self._balanced_elements:
            msg = (
                f"The ion populations of Z={Z} come from the ionisation balance, so a population cannot be"
                " given for its ions. To add LTE excitations of a balanced ion, call"
                " add_ion_ltepopexcitation() with n_ion=None."
            )
            raise ValueError(msg)

    def _register_ion_population(self, Z: int, ion_stage: int, n_ion: float) -> None:
        # an ion's number density must agree between its ionisation and excitation calls
        self._check_not_balanced(Z)
        if Z < 1:
            msg = f"Z must be at least 1 but is {Z}"
            raise ValueError(msg)
        # ion_stage is one more than the charge, so a value below one gives a negative charge and
        # a negative free electron density
        if ion_stage < 1:
            msg = f"ion_stage must be at least 1 (neutral) but is {ion_stage}"
            raise ValueError(msg)
        # the chained comparison also rejects nan, for which every comparison is False, and inf,
        # which would make the free electron density infinite without ever raising
        if not 0.0 <= n_ion < math.inf:
            msg = f"n_ion must be non-negative and finite but is {n_ion}"
            raise ValueError(msg)

        n_ion_existing = self.ionpopdict.get((Z, ion_stage))
        if n_ion_existing is not None and not math.isclose(n_ion_existing, n_ion, rel_tol=1e-6):
            msg = f"Can't add Z={Z} ion_stage {ion_stage} twice with different populations"
            raise ValueError(msg)

        self.ionpopdict[(Z, ion_stage)] = n_ion
        # the free electron density derived from the ion populations is no longer current
        self._n_e = None

    def _check_ionpot_above_emin(self, Z: int, ion_stage: int, ionpots_ev: list[float]) -> None:
        # Kozma & Fransson 1992 assume that every threshold lies above the low-energy cutoff E_0, so that
        # all energy reaching E_0 is thermalised by the free electrons. A channel below E_0 breaks that
        # assumption and can't be accounted for consistently: leaving it out of the matrix drops a real
        # energy sink, and including it credits ionisation below E_0 that the heating term already claims.
        ionpots_below_emin = sorted(ionpot_ev for ionpot_ev in ionpots_ev if ionpot_ev < self.engrid[0])
        if ionpots_below_emin:
            msg = (
                f"Z={Z} ion_stage {ion_stage} has {len(ionpots_below_emin)} ionisation channel(s) with a"
                f" ionisation potential below emin_ev={self.engrid[0]} eV (lowest {ionpots_below_emin[0]} eV)."
                f" The energy fractions would not sum to one, so set emin_ev <= {ionpots_below_emin[0]} eV."
            )
            raise ValueError(msg)

    def _check_ionisation_channel_keys(self, Z: int, ion_stage: int, keys: list[t.Any]) -> None:
        # every channel of an ion needs its own key. This runs before any channel is stored, so a
        # rejected call leaves the solver unchanged instead of half-adding the ion.
        seen = {channel.key for channel in self._ionisation_channels.get((Z, ion_stage), [])}
        for key in keys:
            if key in seen:
                msg = f"Ionisation channel {key} already added for Z={Z} ion_stage={ion_stage}"
                raise ValueError(msg)
            seen.add(key)

    def _store_ionisation_channel(self, Z: int, ion_stage: int, channel: IonisationChannel) -> None:
        # record one channel for an ion. _check_ionisation_channel_keys() runs first.
        self._ionisation_channels.setdefault((Z, ion_stage), []).append(channel)

    def add_excitation(
        self,
        Z: int,
        ion_stage: int,
        levelnumberdensity: float,
        xs_vec: npt.NDArray[np.float64],
        epsilon_trans_ev: float,
        transitionkey: t.Any | None = None,
    ) -> None:
        """Add a bound-bound non-thermal collisional excitation to the solver.

        The transition contributes its part of the excitation term of the degradation equation
        (Kozma & Fransson 1992 equation 7) to the matrix: the level population times the
        integral of y(E') sigma(E') dE' over E' in [E, E + epsilon_trans] for each energy E.

        levelnumberdensity:
            the level population density in cm^-3
        xs_vec:
            an array of cross sections in cm^2 at every energy of the SpencerFanoSolver.engrid
            array [eV]. The solver keeps a read-only copy, so a later write to your own array
            cannot change it.
        epsilon_trans_ev:
            the transition energy in eV
        transitionkey:
            any key to uniquely identify the transition so that the rate coefficient can be retrieved later
        """
        vec, k, frac = self._store_excitation(Z, ion_stage, levelnumberdensity, xs_vec, epsilon_trans_ev, transitionkey)
        self._add_excitation_band(vec, vec * frac, k)

    def _store_excitation(
        self,
        Z: int,
        ion_stage: int,
        levelnumberdensity: float,
        xs_vec: npt.NDArray[np.float64],
        epsilon_trans_ev: float,
        transitionkey: t.Any | None,
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        # validate and record one transition, and describe its matrix band: the values vec[j]
        # (zeroed below the threshold index, where no grid electron can drive the transition),
        # the band width k in whole bins, and the weight frac of the partial final bin
        self._require_not_solved("add excitation")
        self._check_not_balanced(Z)
        # get_xs_on_grid() returns a read-only copy, so a later write by the caller cannot change
        # what the solver holds
        xs_vec = get_xs_on_grid(xs_vec, self.engrid, "xs_vec")

        self._check_epsilon_trans(epsilon_trans_ev)
        # >= rather than < 0, so that NaN, for which every comparison is False, is rejected too
        if not levelnumberdensity >= 0.0:
            msg = f"levelnumberdensity must be non-negative but is {levelnumberdensity}"
            raise ValueError(msg)
        if (Z, ion_stage) not in self.excitationlists:
            self.excitationlists[(Z, ion_stage)] = {}

        if transitionkey is None:
            transitionkey = len(self.excitationlists[(Z, ion_stage)])  # simple number index

        if transitionkey in self.excitationlists[(Z, ion_stage)]:
            msg = f"Transition {transitionkey} already added for Z={Z} ion_stage={ion_stage}"
            raise ValueError(msg)
        self.excitationlists[(Z, ion_stage)][transitionkey] = ExcitationTransition(
            levelnumberdensity=levelnumberdensity,
            xs_vec=xs_vec,
            epsilon_trans_ev=epsilon_trans_ev,
        )
        return self._excitation_band_vectors(levelnumberdensity, xs_vec, epsilon_trans_ev)

    def _check_epsilon_trans(self, epsilon_trans_ev: float) -> None:
        # a non-positive transition energy would put matrix entries below the diagonal, where the
        # triangular solve silently discards them
        if epsilon_trans_ev <= 0.0:
            msg = f"epsilon_trans_ev must be greater than zero but is {epsilon_trans_ev}"
            raise ValueError(msg)
        if epsilon_trans_ev > self.engrid[-1]:
            msg = (
                f"epsilon_trans_ev ({epsilon_trans_ev} eV) is above the top of the energy grid"
                f" ({self.engrid[-1]} eV), so no electron the solver represents can drive the transition"
            )
            raise ValueError(msg)

    def _excitation_band_vectors(
        self, levelnumberdensity: float, xs_vec: npt.NDArray[np.float64], epsilon_trans_ev: float
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        # describe the matrix band of one transition: the values vec[j] (zeroed below the threshold
        # index, where no grid electron can drive the transition), the band width k in whole bins,
        # and the weight frac of the partial final bin. vec is linear in levelnumberdensity.
        vec = levelnumberdensity * self.deltaen * xs_vec  # cross section times level density times bin width
        vec[: self.get_energyindex_lteq(en_ev=epsilon_trans_ev)] = 0.0
        k = int(epsilon_trans_ev / self.deltaen)
        frac = epsilon_trans_ev / self.deltaen - k
        return vec, k, frac

    @staticmethod
    def _sum_excitation_bands(
        bands: dict[int, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
        vec: npt.NDArray[np.float64],
        k: int,
        frac: float,
    ) -> None:
        # writing a band sweeps a strip of the whole npts x npts matrix and dominates the cost of
        # adding many transitions one at a time. Transitions with the same band width k share
        # identical matrix geometry and their band values add linearly, so pre-sum the vectors of
        # each distinct k and write each k once.
        if k not in bands:
            bands[k] = (np.zeros(len(vec)), np.zeros(len(vec)))
        bandvec, bandfracvec = bands[k]
        bandvec += vec
        bandfracvec += vec * frac

    def _add_excitation_band(self, vec: npt.NDArray[np.float64], fracvec: npt.NDArray[np.float64], k: int) -> None:
        # add one excitation band to the matrix. Row i's integral runs over
        # [E_i, E_i + epsilon_trans_ev]: k full-width bins taking vec[j] at columns
        # j in [i, i + k), plus a partial final bin taking fracvec[i + k], truncated where the
        # window leaves the top of the grid (every remaining bin then counts at full width).
        # The uniform grid makes the band geometry the same for every row, so the whole
        # contribution is one windowed addition plus one matrix diagonal, with a row loop only
        # for the clipped rows. The band values add linearly, so vec and fracvec may also hold
        # the pre-summed contributions of many transitions that share the same k.
        if self.heating_only_approximation:
            # keep the stored transitions but leave the excitation loss out of the matrix
            return
        npts = len(self.engrid)
        bandstop = max(npts - k, 0)  # rows from here on have their window clipped at the top of the grid
        # sfmatrix is C-contiguous (np.zeros), so element (i, i + d) sits at flat index
        # i * (npts + 1) + d; copy=False makes reshape raise rather than silently write to a copy
        flat = self.sfmatrix.reshape(-1, copy=False)
        if 0 < k < npts:
            # rows of this view are the band segments sfmatrix[i, i:i+k] for i in [0, bandstop)
            band = flat[: bandstop * (npts + 1)].reshape(bandstop, npts + 1)[:, :k]
            band += np.lib.stride_tricks.sliding_window_view(vec, k)[:bandstop]
        fracdiag = flat[k :: npts + 1][:bandstop]  # elements sfmatrix[i, i + k]
        fracdiag += fracvec[k : bandstop + k]
        for i in range(bandstop, npts):
            self.sfmatrix[i, i:] += vec[i:]

    def add_ion_ltepopexcitation(
        self,
        Z: int,
        ion_stage: int,
        n_ion: float | None = None,
        temperature: float | None = None,
        adata_polars: pl.DataFrame | None = None,
        use_collstrengths: bool = True,
        maxnlevelslower: int | None = 5,
        maxnlevelsupper: int | None = 250,
    ) -> None:
        """Add bound-bound excitations of one ion, with LTE level populations at the solver temperature.

        Each added transition is keyed by (lower level index, upper level index), the key to pass to
        get_excitation_ratecoeff() after solving.

        If the ion belongs to an element added with add_element_ionbalance() or add_element_saha(),
        n_ion must be None. The level populations then follow the ion population that solve()
        finds. For every other ion, n_ion is required. add_element_ltepopexcitation() calls this
        method for every stage of a balanced element that has level data.

        Transitions whose energy lies outside the solver's energy grid are dropped: above emax_ev no
        electron the solver represents can drive them, and below emin_ev Kozma & Fransson 1992 take every
        electron to have thermalised already, so their energy is accounted for as heating instead. Unlike
        the equivalent case in add_ionisation(), this cannot be raised as an error, because real ions have
        fine-structure transitions far below any usable emin_ev. Pass verbose=True to the solver to see how
        many transitions were dropped.

        n_ion:
            the ion number density in cm^-3, or None for an ion of a balanced element
        temperature:
            the temperature in K for the LTE Boltzmann level populations. The solver has one
            temperature: the first call that gives one (this method, add_element_ltepopexcitation(),
            or add_element_saha()) sets it, and later calls can give None to use it. A different
            value raises a ValueError.
        adata_polars:
            a levels/transitions table to use instead of the internal database (the CMFGEN-derived
            ARTIS atomic data), in the format returned by artistools.atomic.get_levels() with
            get_transitions=True: one row per ion with Z, ion_stage, and nested "levels" and
            "transitions" frames. Once given, it is kept for later calls on this solver.
        use_collstrengths:
            compute cross sections from tabulated collision strengths where available (Li et al.
            2012 equation 11). Permitted transitions without one (or all permitted transitions,
            when False) instead use the oscillator strength via the van Regemorter approximation;
            forbidden transitions outside the collision-strength path get a zero cross section
        maxnlevelslower, maxnlevelsupper:
            include only transitions whose lower level index is below maxnlevelslower and whose
            upper level index is below maxnlevelsupper; None disables that cutoff
        """
        self._require_not_solved("add excitation")
        temperature = self._resolve_temperature(temperature)
        if n_ion is None:
            element = self._balanced_elements.get(Z)
            if element is None or ion_stage not in element.ion_stages:
                msg = (
                    f"n_ion is required for Z={Z} ion_stage {ion_stage}. It can be None only for an ion of an"
                    " element added with add_element_ionbalance() or add_element_saha()."
                )
                raise ValueError(msg)
            templates = self._build_ltepop_excitation_templates(
                Z, ion_stage, temperature, adata_polars, use_collstrengths, maxnlevelslower, maxnlevelsupper
            )
            self._add_balanced_excitation_templates(element, ion_stage, templates)
            self.temperature = temperature
            return

        # the population check runs before the atomic data is read, so a bad n_ion fails fast
        self._check_not_balanced(Z)
        if not 0.0 <= n_ion < math.inf:
            msg = f"n_ion must be non-negative and finite but is {n_ion}"
            raise ValueError(msg)

        templates = self._build_ltepop_excitation_templates(
            Z, ion_stage, temperature, adata_polars, use_collstrengths, maxnlevelslower, maxnlevelsupper
        )

        # register the population so that this ion counts towards n_e and n_ion_tot even when
        # add_ionisation() was never called for it
        self._register_ion_population(Z, ion_stage, n_ion)

        bands: dict[int, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}
        try:
            for transitionkey, template in templates:
                vec, k, frac = self._store_excitation(
                    Z,
                    ion_stage,
                    n_ion * template.popfrac,
                    template.xs_vec,
                    template.epsilon_trans_ev,
                    transitionkey=transitionkey,
                )
                if not self.heating_only_approximation:
                    # the matrix takes no excitation band in the heating-only approximation
                    self._sum_excitation_bands(bands, vec, k, frac)
        finally:
            # _store_excitation validates before it records, so if a transition raises (for
            # example a duplicate key in custom atomic data), exactly the transitions already
            # recorded in excitationlists have accumulated bands. Writing them on the way out
            # keeps the matrix consistent with the bookkeeping for a caller that catches the
            # error, matching the old behaviour of adding each transition atomically.
            for k, (bandvec, bandfracvec) in sorted(bands.items()):
                self._add_excitation_band(bandvec, bandfracvec, k)
        self.temperature = temperature

    def add_element_ltepopexcitation(
        self,
        Z: int,
        temperature: float | None = None,
        adata_polars: pl.DataFrame | None = None,
        use_collstrengths: bool = True,
        maxnlevelslower: int | None = 5,
        maxnlevelsupper: int | None = 250,
    ) -> None:
        """Add LTE bound-bound excitations to every stage of a balanced element that has level data.

        This calls add_ion_ltepopexcitation() with n_ion=None for each stage of the chain of an
        element added with add_element_ionbalance() or add_element_saha(). Stages without level data
        (in the internal database or adata_polars) get no excitations; a ValueError reports an
        element with no such stage. Call add_ion_ltepopexcitation() instead to choose the stages or
        to give each stage its own options. The parameters are those of add_ion_ltepopexcitation().
        """
        element = self._balanced_elements.get(Z)
        if element is None:
            msg = f"Z={Z} is not a balanced element. Call add_element_ionbalance() or add_element_saha() first."
            raise ValueError(msg)
        temperature = self._resolve_temperature(temperature)

        stages_with_levels = [
            ion_stage
            for ion_stage in element.ion_stages
            if ion_stage <= Z and self._get_ion_levels(Z, ion_stage, adata_polars) is not None
        ]
        if not stages_with_levels:
            msg = (
                f"No excitation data for any ion stage {element.ion_stages[0]}-{element.ion_stages[-1]} of Z={Z}."
                " Supply a custom level/transition table via adata_polars."
            )
            raise ValueError(msg)

        for ion_stage in stages_with_levels:
            self.add_ion_ltepopexcitation(
                Z,
                ion_stage,
                n_ion=None,
                temperature=temperature,
                adata_polars=adata_polars,
                use_collstrengths=use_collstrengths,
                maxnlevelslower=maxnlevelslower,
                maxnlevelsupper=maxnlevelsupper,
            )

    def _get_adata_polars(self, adata_polars: pl.DataFrame | None) -> pl.DataFrame:
        # the levels/transitions table: the one given now, else the one kept from an earlier call,
        # else the internal database
        if adata_polars is not None:
            self.adata_polars = adata_polars

        if self.adata_polars is None:
            # use ARTIS atomic data read by the artistools package to get the levels
            self.adata_polars = at.atomic.get_levels(
                Path(pynonthermal.DATADIR, "artis_files"),
                get_transitions=True,
                derived_transitions_columns=["epsilon_trans_ev", "lambda_angstroms", "lower_g", "upper_g"],
            )

        return self.adata_polars

    def _get_ion_levels(self, Z: int, ion_stage: int, adata_polars: pl.DataFrame | None) -> pl.DataFrame | None:
        # the row of one ion in the levels/transitions table, or None if the table has no data for it
        ion = self._get_adata_polars(adata_polars).filter(pl.col("Z") == Z).filter(pl.col("ion_stage") == ion_stage)
        return None if ion.is_empty() else ion

    def _build_ltepop_excitation_templates(
        self,
        Z: int,
        ion_stage: int,
        temperature: float,
        adata_polars: pl.DataFrame | None,
        use_collstrengths: bool,
        maxnlevelslower: int | None,
        maxnlevelsupper: int | None,
    ) -> list[tuple[t.Any, _ExcitationTemplate]]:
        # the part of add_ion_ltepopexcitation() that does not depend on the ion population: for each
        # transition on the energy grid, the key (lower, upper), the LTE population fraction of the
        # lower level, the cross section on the grid, and the transition energy
        ion = self._get_ion_levels(Z, ion_stage, adata_polars)
        if ion is None:
            msg = (
                f"No excitation data for Z={Z} ion_stage {ion_stage} in internal database."
                " Supply a custom level/transition table via adata_polars, or add cross"
                " sections directly with add_excitation()."
            )
            raise ValueError(msg)

        dfpops_thision = ion["levels"].item()

        ltepartfunc = at.transitions.get_lte_partfunc(dfpops_thision, temperature)
        dfpops_thision = (
            dfpops_thision.rename({"levelindex": "level"}).with_columns(
                ion_popfrac=pl.col("g") * (-pl.col("energy_ev") / K_B / temperature).exp() / ltepartfunc
            )
        ).select(["level", "ion_popfrac"])

        lzdftransitions = ion["transitions"].item().filter((pl.col("collstr") >= 0).or_(pl.col("forbidden") == 0))

        # default maxnlevelslower/maxnlevelsupper of 5/250 match the ARTIS defaults
        if maxnlevelslower is not None:
            lzdftransitions = lzdftransitions.filter(pl.col("lower") < maxnlevelslower)
        if maxnlevelsupper is not None:
            lzdftransitions = lzdftransitions.filter(pl.col("upper") < maxnlevelsupper)

        # transitions outside the grid are dropped (see the docstring), but say how many, so that a grid
        # that happens to exclude most of an ion's excitation channels doesn't go unnoticed
        dftransitions_allenergies = lzdftransitions.collect()
        dftransitions = dftransitions_allenergies.filter(
            pl.col("epsilon_trans_ev").is_between(self.engrid[0], self.engrid[-1])
        )

        if self.verbose and len(dftransitions) < len(dftransitions_allenergies):
            arr_epsilon_trans_ev = dftransitions_allenergies["epsilon_trans_ev"]
            n_below_grid = int((arr_epsilon_trans_ev < self.engrid[0]).sum())
            print(
                f"  dropped {n_below_grid} transition(s) below emin_ev={self.engrid[0]} eV and"
                f" {len(dftransitions_allenergies) - len(dftransitions) - n_below_grid} above"
                f" emax_ev={self.engrid[-1]} eV for Z={Z} ion_stage {ion_stage} ({len(dftransitions)} kept)"
            )

        if dftransitions.is_empty():
            return []

        dftransitions = dftransitions.join(
            dfpops_thision.select(pl.col("level").alias("lower"), pl.col("ion_popfrac").alias("lower_popfrac")),
            on="lower",
            how="left",
        )

        if self.verbose:
            print(
                f"  including Z={Z} ion_stage"
                f" {ion_stage} ({at.get_ionstring(Z, ion_stage)}) excitation with T"
                f" {temperature} K (ntransitions {len(dftransitions)},"
                f" maxnlevelslower {maxnlevelslower}, maxnlevelsupper"
                f" {maxnlevelsupper})"
            )

        templates = []
        for transition in dftransitions.iter_rows(named=True):
            xs_vec = pynonthermal.excitation.get_xs_excitation_vector(
                self.engrid, transition, use_collstrengths=use_collstrengths
            )
            # the balanced path hands this array to ExcitationTransition, so it must not change later
            xs_vec.flags.writeable = False
            templates.append(
                (
                    (transition["lower"], transition["upper"]),
                    _ExcitationTemplate(
                        popfrac=float(transition["lower_popfrac"]),
                        xs_vec=xs_vec,
                        epsilon_trans_ev=float(transition["epsilon_trans_ev"]),
                    ),
                )
            )

        return templates

    def _add_balanced_excitation_templates(
        self, element: _BalancedElement, ion_stage: int, templates: list[tuple[t.Any, _ExcitationTemplate]]
    ) -> None:
        # record the excitation templates of one balanced ion, and add their bands to the matrix at
        # the ion's current (provisional) population. _set_balanced_populations() rescales them later.
        Z = element.Z
        stored = element.excitation_templates.setdefault(ion_stage, {})
        # every check runs before anything is recorded, so a rejected call leaves the solver unchanged
        seen = set(stored)
        for transitionkey, template in templates:
            self._check_epsilon_trans(template.epsilon_trans_ev)
            if transitionkey in seen:
                msg = f"Transition {transitionkey} already added for Z={Z} ion_stage={ion_stage}"
                raise ValueError(msg)
            seen.add(transitionkey)

        n_ion = self.ionpopdict[(Z, ion_stage)]
        unit_bands = element.excitation_unit_bands.setdefault(ion_stage, {})
        new_bands: dict[int, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}
        transitions = self.excitationlists.setdefault((Z, ion_stage), {})
        for transitionkey, template in templates:
            stored[transitionkey] = template
            transitions[transitionkey] = ExcitationTransition(
                levelnumberdensity=n_ion * template.popfrac,
                xs_vec=template.xs_vec,
                epsilon_trans_ev=template.epsilon_trans_ev,
            )
            vec, k, frac = self._excitation_band_vectors(template.popfrac, template.xs_vec, template.epsilon_trans_ev)
            self._sum_excitation_bands(new_bands, vec, k, frac)

        for k, (bandvec, bandfracvec) in sorted(new_bands.items()):
            if k in unit_bands:
                unit_bands[k][0][:] += bandvec
                unit_bands[k][1][:] += bandfracvec
            else:
                unit_bands[k] = (bandvec, bandfracvec)
            self._add_excitation_band(n_ion * bandvec, n_ion * bandfracvec, k)

    def _add_ionisation_channel_to_matrix(self, n_ion: float, channel: IonisationChannel) -> None:
        # add one channel's contribution to the ionisation term of the degradation equation
        # (Kozma & Fransson 1992 equation 7): integrals of y(E') sigma_ic(E') P(E', epsilon - I),
        # using their equation 5 factorisation of the differential cross section into the channel's
        # total cross section sigma_ic (IonisationChannel.xs_grid) times the secondary-electron energy
        # distribution P of KF92 equation 4, whose integrals over epsilon are taken analytically
        # via the arctan antiderivative below. That analytic step fixes the shape of P, so a channel
        # sets only its width J and not the shape.
        self._require_not_solved("add ionisation")
        if self.heating_only_approximation:
            # leave the ionisation loss out of the matrix. add_ionisation keeps the ion
            # registration and the channel data that the analysis reads.
            return
        deltaen = self.deltaen
        ionpot_ev = channel.ionpot_ev
        J = channel.J_ev
        npts = len(self.engrid)

        ar_xs_array = channel.xs_grid

        xsstartindex = 0 if ionpot_ev <= self.engrid[0] else self.get_energyindex_gteq(en_ev=ionpot_ev)

        # J * atan[(epsilon - ionpot_ev) / J] is the indefinite integral of
        # 1/(1 + (epsilon - ionpot_ev)^2/ J^2) d_epsilon
        # in Kozma & Fransson 1992 equation 4

        # the cross section is zero wherever atan(...) is zero (energy at or below the ionisation
        # potential), so those prefactors are set to zero rather than dividing by zero
        atan_epsilon = np.arctan((self.engrid - ionpot_ev) / 2.0 / J)
        with np.errstate(divide="ignore", invalid="ignore"):
            prefactors = np.divide(
                n_ion * ar_xs_array * deltaen, atan_epsilon, out=np.zeros(npts), where=atan_epsilon != 0.0
            )

        # Luke Shingles: the use of min and max on the epsilon limits keeps energies
        # from becoming unphysical. This insight came from reading the
        # CMFGEN Fortran source code (Li, Dessart, Hillier 2012, doi:10.1111/j.1365-2966.2012.21198.x)
        # I had neglected this, so the limits of integration were incorrect. The fix didn't massively affect
        # ionisation rates or spectra, but it was a source of error that led to energy fractions not adding up to 100%.

        epsilon_uppers = np.minimum((self.engrid + ionpot_ev) / 2, self.engrid)
        int_eps_uppers = np.arctan((epsilon_uppers - ionpot_ev) / J)

        # for the resulting arrays, use index j - i corresponding to energy endash - en, so each
        # entry depends only on the column j and the offset j - i
        epsilon_lowers1 = np.maximum(self.engrid - self.engrid[0], ionpot_ev)
        int_eps_lowers1 = np.arctan((epsilon_lowers1 - ionpot_ev) / J)

        # each integral is non-empty (epsilon_lower <= epsilon_upper) on a contiguous column
        # range located below without evaluating per-row masks, so the slice writes touch only
        # the non-zero part of each row. For the first integral: every written column has
        # engrid[j] >= ionpot_ev, so the range is non-empty while epsilon_lowers1 sits on its
        # ionpot_ev plateau, and past the plateau it rises with j at the full grid spacing
        # against half that for epsilon_uppers, giving at most one non-empty-to-empty switch
        # per row; raising en only moves that switch right, so a forward-only pointer tracks it
        # with the same comparisons the mask would make. For the second integral the lower
        # limit is constant along the row, so the range start is a searchsorted into the
        # non-decreasing epsilon_uppers.
        cut1 = 0  # end of the first integral's non-empty column range, clamped to jstart per row
        top_en_plus_deltaen = self.engrid[-1] + deltaen
        for i, en in enumerate(self.engrid):
            # endash ranges from en to SF_EMAX, but skip over the zero-cross section points
            jstart = max(i, xsstartindex)

            # first integral of the KF92 equation 7 ionisation term: primaries at endash carried
            # across en by the energy loss epsilon of an ionisation event.
            # at each endash (columns j >= jstart), the integral in epsilon ranges from
            # epsilon_lower = max(endash - en, ionpot_ev)
            # epsilon_upper = min((endash + ionpot_ev) / 2, endash)
            cut1 = max(cut1, jstart)
            while cut1 < npts and epsilon_lowers1[cut1 - i] <= epsilon_uppers[cut1]:
                cut1 += 1
            self.sfmatrix[i, jstart:cut1] += prefactors[jstart:cut1] * (
                int_eps_uppers[jstart:cut1] - int_eps_lowers1[jstart - i : cut1 - i]
            )

            if 2 * en + ionpot_ev < top_en_plus_deltaen:
                secondintegralstartindex = self.get_energyindex_lteq(float(2 * en + ionpot_ev))

                # second integral of the KF92 equation 7 ionisation term, subtracted: an
                # ionisation by a primary above 2 * en + ionpot_ev leaves the secondary with
                # more energy than en, adding to the electrons that cross en like an extra
                # source above it.
                # endash ranges from 2 * en + ionpot_ev to SF_EMAX
                # at each endash, the integral in epsilon ranges from
                # epsilon_lower = en + ionpot_ev
                # epsilon_upper = min((endash + ionpot_ev) / 2, endash)
                epsilon_lower2 = float(en + ionpot_ev)
                int_eps_lower2 = math.atan((epsilon_lower2 - ionpot_ev) / J)
                jstart2 = max(secondintegralstartindex, int(np.searchsorted(epsilon_uppers, epsilon_lower2)))
                self.sfmatrix[i, jstart2:] -= prefactors[jstart2:] * (int_eps_uppers[jstart2:] - int_eps_lower2)

    def add_ionisation(self, Z: int, ion_stage: int, n_ion: float) -> None:
        """Add collisional ionisation of one ion, contributing every shell with cross-section data.

        The shells enter the ionisation term of the degradation equation (Kozma & Fransson 1992
        equation 7). Each ion may be added only once through this method; an ion with n_ion of
        exactly zero is skipped without being registered. A ValueError is raised if any shell's
        ionisation potential lies below the energy grid's emin_ev (the message gives the emin_ev
        to use instead). To add a channel that the built-in table does not hold, or to replace the
        built-in shells of an ion, use add_ionisation_channel() instead.

        n_ion:
            the ion number density in cm^-3
        """
        self._require_not_solved("add ionisation")
        self._check_not_balanced(Z)
        if not 0.0 <= n_ion < math.inf:
            msg = f"n_ion must be non-negative and finite but is {n_ion}"
            raise ValueError(msg)
        if n_ion == 0.0:
            return

        channels = pynonthermal.collion.get_ion_ionisation_channels(self.dfcollion, Z, ion_stage, self.engrid)

        self._check_ionpot_above_emin(Z, ion_stage, [channel.ionpot_ev for channel in channels])
        self._check_ionisation_channel_keys(Z, ion_stage, [channel.key for channel in channels])

        if self.verbose:
            print(
                f"  including Z={Z} ion_stage"
                f" {ion_stage} ({at.get_ionstring(Z, ion_stage)}) ionisation with n_ion"
                f" {n_ion:.1e} [/cm3]"
            )
        self._register_ion_population(Z, ion_stage, n_ion)

        for channel in channels:
            self._store_ionisation_channel(Z, ion_stage, channel)
            self._add_ionisation_channel_to_matrix(n_ion, channel)

    def add_ionisation_channel(
        self,
        Z: int,
        ion_stage: int,
        n_ion: float,
        ionpot_ev: float,
        xs_vec: npt.NDArray[np.float64],
        channelkey: t.Any | None = None,
    ) -> None:
        """Add one collisional ionisation channel of an ion, with a custom cross section.

        The channel enters the ionisation term of the degradation equation (Kozma & Fransson 1992
        equation 7) in the same way as a shell of the built-in table. Call this method as many
        times as the ion has channels. To keep the built-in shells as well, also call
        add_ionisation() for the ion; to replace them, do not call add_ionisation() for it.

        The solver keeps the Lorentzian secondary-electron distribution of Kozma & Fransson 1992
        equation 4, whose width comes from pynonthermal.collion.get_J(Z, ion_stage, ionpot_ev).
        The matrix fill integrates that distribution analytically, so the shape is not adjustable.

        n_ion:
            the ion number density in cm^-3. It must agree with the value that any other call
            for this ion gives.
        ionpot_ev:
            the ionisation potential of the channel in eV. It must be between emin_ev and
            emax_ev. The cross section must be zero at and below it.
        xs_vec:
            an array of cross sections in cm^2 at every energy of the SpencerFanoSolver.engrid
            array [eV]. The solver keeps a read-only copy, so a later write to your own array
            cannot change it. calculate_N_e() needs the cross section between the grid points
            just above the ionisation potential, and interpolates the array there.
        channelkey:
            any key to identify the channel in the ion. The default is the number of channels
            that the ion already has.
        """
        self._require_not_solved("add ionisation")
        self._check_not_balanced(Z)
        if not 0.0 <= n_ion < math.inf:
            msg = f"n_ion must be non-negative and finite but is {n_ion}"
            raise ValueError(msg)

        if ionpot_ev > self.engrid[-1]:
            # the matrix fill would write nothing and the channel would be inert. The equivalent
            # excitation check is in _store_excitation().
            msg = (
                f"ionpot_ev ({ionpot_ev} eV) is above the top of the energy grid"
                f" ({self.engrid[-1]} eV), so no electron the solver represents can ionise this channel"
            )
            raise ValueError(msg)

        if channelkey is None:
            channelkey = len(self._ionisation_channels.get((Z, ion_stage), []))

        # every check runs before anything is recorded, so a rejected call leaves the solver
        # unchanged. The cross section is checked even for a zero population, which adds no channel.
        channel = IonisationChannel.from_xs_grid(
            arr_enev=self.engrid,
            Z=Z,
            ion_stage=ion_stage,
            ionpot_ev=ionpot_ev,
            xs_vec=xs_vec,
            key=channelkey,
        )
        self._check_ionpot_above_emin(Z, ion_stage, [channel.ionpot_ev])
        self._check_ionisation_channel_keys(Z, ion_stage, [channel.key])

        if n_ion == 0.0:
            return

        if self.verbose:
            print(
                f"  including Z={Z} ion_stage {ion_stage} ({at.get_ionstring(Z, ion_stage)})"
                f" ionisation channel {channelkey} (ionpot {ionpot_ev:.2f} eV) with n_ion"
                f" {n_ion:.1e} [/cm3]"
            )

        self._register_ion_population(Z, ion_stage, n_ion)
        self._store_ionisation_channel(Z, ion_stage, channel)
        self._add_ionisation_channel_to_matrix(n_ion, channel)

    def add_element_ionbalance(self, Z: int, n_elem: float, recomb_ratecoeffs: Mapping[int, float]) -> None:
        """Add an element whose ion populations solve() finds from an ionisation/recombination balance.

        For each pair of adjacent ion stages i and i+1, the balance is
        n_i Gamma_i = n_{i+1} n_e alpha_{i+1}, with Gamma_i the non-thermal ionisation rate
        coefficient [s^-1] of stage i from the Spencer-Fano solution and alpha_{i+1} the
        recombination rate coefficient of stage i+1. The solution depends on the populations, so
        solve() iterates until the populations converge, and it finds the free electron density
        from charge neutrality. Thermal collisional ionisation, photoionisation, and charge
        exchange are not included, so the populations depend on depositionratedensity_ev.

        The chain of ion stages runs from one below the lowest key to the highest key. The top
        stage is a sink: its ionisation is an energy loss in the matrix, but the ions it makes have
        no stage to go to. Extend the chain to a stage whose ionisation is negligible, or solve()
        warns. Every stage gets the built-in ionisation channels of add_ionisation(). To add LTE
        excitations of a stage, call add_ion_ltepopexcitation() with n_ion=None.

        Until solve() runs, ionpopdict holds a provisional population of equal fractions for the
        stages, and get_n_e() and get_n_ion_tot() include it.

        n_elem:
            the number density of the element in cm^-3, summed over the stages of the chain
        recomb_ratecoeffs:
            the recombination rate coefficients in cm^3 s^-1, keyed by the ion stage that
            recombines (the upper stage of each pair). The keys must be contiguous and lie between
            2 and Z + 1.
        """
        # a sequence would be iterated as keys, which gives a misleading message about the ion stages
        if not isinstance(recomb_ratecoeffs, Mapping):
            msg = (
                "recomb_ratecoeffs must be a mapping from the recombining ion stage to the rate coefficient"
                f" in cm^3 s^-1, for example {{2: 3e-13, 3: 3e-12}}, but is {type(recomb_ratecoeffs).__name__}"
            )
            raise TypeError(msg)
        if not recomb_ratecoeffs:
            msg = f"Z={Z} needs at least one recombination rate coefficient"
            raise ValueError(msg)
        for ion_stage in recomb_ratecoeffs:
            if not isinstance(ion_stage, int) or isinstance(ion_stage, bool):
                msg = f"the keys of recomb_ratecoeffs must be ion stages (integers) but one is {ion_stage!r}"
                raise TypeError(msg)

        upper_stages = sorted(recomb_ratecoeffs)
        ion_stages = tuple(range(upper_stages[0] - 1, upper_stages[-1] + 1))
        if upper_stages != list(ion_stages[1:]):
            msg = (
                f"the recombination rate coefficient keys of Z={Z} must be contiguous ion stages but are {upper_stages}"
            )
            raise ValueError(msg)
        self._check_new_balanced_element(Z, n_elem, ion_stages)

        for ion_stage, alpha in recomb_ratecoeffs.items():
            # the chained comparison also rejects nan. A zero would divide the balance by zero.
            if not 0.0 < alpha < math.inf:
                msg = (
                    f"the recombination rate coefficient of Z={Z} ion_stage {ion_stage} must be greater than zero"
                    f" and finite but is {alpha}"
                )
                raise ValueError(msg)

        self._add_balanced_element(
            _BalancedElement(
                Z=Z,
                n_elem=n_elem,
                ion_stages=ion_stages,
                recomb_ratecoeffs={int(ion_stage): float(alpha) for ion_stage, alpha in recomb_ratecoeffs.items()},
                saha_factors=None,
            )
        )

    def add_element_saha(
        self,
        Z: int,
        n_elem: float,
        temperature: float | None,
        ion_stages: Sequence[int],
        partfuncs: Mapping[int, float] | None = None,
        adata_polars: pl.DataFrame | None = None,
    ) -> None:
        """Add an element whose ion populations solve() finds from the Saha equation.

        For each pair of adjacent ion stages i and i+1,
        n_{i+1} n_e / n_i = 2 (U_{i+1} / U_i) (2 pi m_e k_B T / h^2)^(3/2) exp(-chi_i / (k_B T)),
        with the ionisation potentials chi_i from the NIST table. solve() finds the free electron
        density from charge neutrality. No recombination rate coefficients are needed.

        Every stage gets the built-in ionisation channels of add_ionisation(), so the ionisation
        of the top stage is an energy loss in the matrix. To add LTE excitations of a stage, call
        add_ion_ltepopexcitation() with n_ion=None.

        Until solve() runs, ionpopdict holds a provisional population of equal fractions for the
        stages, and get_n_e() and get_n_ion_tot() include it.

        n_elem:
            the number density of the element in cm^-3, summed over the stages of the chain
        temperature:
            the temperature in K. The solver has one temperature: the first call that gives one sets
            it, and later calls can give None to use it. A different value raises a ValueError.
        ion_stages:
            at least two contiguous ion stages between 1 and Z + 1
        partfuncs:
            partition functions keyed by ion stage. A stage without an entry gets the LTE
            partition function at the temperature from the level data (the internal database or
            adata_polars), or 1 for the bare nucleus. A ValueError names a stage that has neither.
        adata_polars:
            a levels table in the format of add_ion_ltepopexcitation(). Once given, it is kept
            for later calls on this solver.
        """
        stages = tuple(int(ion_stage) for ion_stage in ion_stages)
        self._check_new_balanced_element(Z, n_elem, stages)
        temperature = self._resolve_temperature(temperature)
        if partfuncs is not None:
            # a partition function for a stage outside the chain is most likely a mistake in the keys
            stages_outside_chain = sorted(ion_stage for ion_stage in partfuncs if ion_stage not in stages)
            if stages_outside_chain:
                msg = (
                    f"partfuncs has ion stages {stages_outside_chain} that are not in the chain {list(stages)} of Z={Z}"
                )
                raise ValueError(msg)

        partfunc_of_stage: dict[int, float] = {}
        for ion_stage in stages:
            if partfuncs is not None and ion_stage in partfuncs:
                partfunc = float(partfuncs[ion_stage])
            elif ion_stage == Z + 1:
                partfunc = 1.0
            else:
                ion = self._get_ion_levels(Z, ion_stage, adata_polars)
                if ion is None:
                    msg = (
                        f"No level data for Z={Z} ion_stage {ion_stage} to calculate a partition function."
                        " Give it in partfuncs or supply a level table via adata_polars."
                    )
                    raise ValueError(msg)
                partfunc = at.transitions.get_lte_partfunc(ion["levels"].item(), temperature)
            if not 0.0 < partfunc < math.inf:
                msg = (
                    f"the partition function of Z={Z} ion_stage {ion_stage} must be greater than zero but is {partfunc}"
                )
                raise ValueError(msg)
            partfunc_of_stage[ion_stage] = partfunc

        ionpots = pynonthermal.collion.get_nist_ionisation_energies_ev()
        saha_factors = []
        for ion_stage in stages[:-1]:
            ionpot_ev = ionpots.get((Z, ion_stage))
            if ionpot_ev is None:
                msg = f"No NIST ionisation energy for Z={Z} ion_stage {ion_stage}"
                raise ValueError(msg)
            saha_factors.append(
                get_saha_factor(temperature, ionpot_ev, partfunc_of_stage[ion_stage], partfunc_of_stage[ion_stage + 1])
            )

        self._add_balanced_element(
            _BalancedElement(
                Z=Z,
                n_elem=n_elem,
                ion_stages=stages,
                recomb_ratecoeffs=None,
                saha_factors=tuple(saha_factors),
            )
        )
        self.temperature = temperature

    def _check_new_balanced_element(self, Z: int, n_elem: float, ion_stages: tuple[int, ...]) -> None:
        # the checks of both add_element methods that need no atomic data
        self._require_not_solved("add element")
        if Z < 1:
            msg = f"Z must be at least 1 but is {Z}"
            raise ValueError(msg)
        # the chained comparison also rejects nan
        if not 0.0 < n_elem < math.inf:
            msg = f"n_elem must be greater than zero and finite but is {n_elem}"
            raise ValueError(msg)
        if len(ion_stages) < 2 or list(ion_stages) != list(range(ion_stages[0], ion_stages[-1] + 1)):
            msg = f"the ion stages of Z={Z} must be at least two contiguous stages but are {list(ion_stages)}"
            raise ValueError(msg)
        if ion_stages[0] < 1 or ion_stages[-1] > Z + 1:
            msg = f"the ion stages of Z={Z} must lie between 1 and {Z + 1} but are {list(ion_stages)}"
            raise ValueError(msg)
        if Z in self._balanced_elements:
            msg = f"Z={Z} was already added as a balanced element"
            raise ValueError(msg)
        # an element is balanced as a whole, so none of its ions can have a given population
        ions_present = {*self.ionpopdict, *self._ionisation_channels, *self.excitationlists}
        if any(Z_present == Z for Z_present, _ in ions_present):
            msg = f"Z={Z} already has ions with given populations or channels, so it cannot become a balanced element"
            raise ValueError(msg)

    def _add_balanced_element(self, element: _BalancedElement) -> None:
        # record a checked element, give its stages a provisional population of equal fractions, and
        # add its built-in ionisation channels to the matrix at that population
        Z = element.Z
        channels_of_stage = {
            ion_stage: (
                # a bare nucleus has no electrons to remove, and no row in the cross-section table
                []
                if ion_stage == Z + 1
                else pynonthermal.collion.get_ion_ionisation_channels(self.dfcollion, Z, ion_stage, self.engrid)
            )
            for ion_stage in element.ion_stages
        }
        for ion_stage, channels in channels_of_stage.items():
            self._check_ionpot_above_emin(Z, ion_stage, [channel.ionpot_ev for channel in channels])

        if self.verbose:
            mode = "Saha" if element.recomb_ratecoeffs is None else "ionisation/recombination balance"
            print(
                f"  including Z={Z} ion_stages {element.ion_stages[0]}-{element.ion_stages[-1]} with n_elem"
                f" {element.n_elem:.1e} [/cm3] from the {mode}"
            )

        self._balanced_elements[Z] = element
        n_ion_guess = element.n_elem / len(element.ion_stages)
        for ion_stage, channels in channels_of_stage.items():
            self.ionpopdict[(Z, ion_stage)] = n_ion_guess
            for channel in channels:
                self._store_ionisation_channel(Z, ion_stage, channel)
                self._add_ionisation_channel_to_matrix(n_ion_guess, channel)
        self._n_e = None

    def _set_balanced_populations(self, element: _BalancedElement, populations: Mapping[int, float]) -> None:
        # move the ions of one balanced element to new populations. Both matrix fills are linear
        # in the population, so adding the change of each population keeps the matrix equal to the
        # fixed contributions plus the current ionpopdict contributions of the balanced ions, with
        # no copy of the matrix. The rounding residue is of the order of the machine precision
        # times the largest population that was applied.
        Z = element.Z
        for ion_stage in element.ion_stages:
            n_new = populations[ion_stage]
            delta = n_new - self.ionpopdict[(Z, ion_stage)]
            self.ionpopdict[(Z, ion_stage)] = n_new
            if delta != 0.0:
                for channel in self._ionisation_channels.get((Z, ion_stage), []):
                    self._add_ionisation_channel_to_matrix(delta, channel)
                for k, (unitvec, unitfracvec) in sorted(element.excitation_unit_bands.get(ion_stage, {}).items()):
                    self._add_excitation_band(delta * unitvec, delta * unitfracvec, k)
            templates = element.excitation_templates.get(ion_stage)
            if templates:
                self.excitationlists[(Z, ion_stage)] = {
                    transitionkey: ExcitationTransition(
                        levelnumberdensity=n_new * template.popfrac,
                        xs_vec=template.xs_vec,
                        epsilon_trans_ev=template.epsilon_trans_ev,
                    )
                    for transitionkey, template in templates.items()
                }
        self._n_e = None

    def calculate_free_electron_density(self) -> float:
        # number density of free electrons [cm^-3]
        n_e = 0.0
        for Z, ion_stage in self.ionpopdict:
            charge = ion_stage - 1
            assert charge >= 0
            n_e += charge * self.ionpopdict[(Z, ion_stage)]
        return n_e

    def get_n_e(self) -> float:
        if self._n_e_override is not None:
            return self._n_e_override

        if self._n_e is None:
            self._n_e = self.calculate_free_electron_density()

        return self._n_e

    def get_n_ion_tot(self) -> float:
        # total number density of all nuclei [cm^-3]
        n_ion_tot = 0.0
        for Z, ion_stage in self.ionpopdict:
            n_ion_tot += self.ionpopdict[(Z, ion_stage)]
        return n_ion_tot

    def solve(
        self,
        depositionratedensity_ev: float,
        override_n_e: float | None = None,
        *,
        balance_tol: float = 1e-4,
    ) -> None:
        """Solve the Spencer-Fano equation for the deposition rate density [eV s^-1 cm^-3].

        override_n_e:
            a free electron density [cm^-3] to use in place of the one from the ion populations.
            With balanced elements, the balance also uses this density.
        balance_tol:
            the relative tolerance of the ratio n_{i+1} n_e / n_i of every pair of adjacent stages
            of an element added with add_element_ionbalance(). The iteration stops when the ratios
            from the solution agree with the ratios that gave the populations to this tolerance. A
            RuntimeError reports a balance that did not converge within BALANCE_MAXITER iterations.
            After solve(), balance_iterations holds the number of iterations that the balance took
            (zero without balanced elements).
        """
        self._solved = False
        self.reset_solution_analysis()

        # every fraction and rate coefficient is divided by the deposition rate density. A zero gave a bare
        # ZeroDivisionError from inside the analysis, and a negative one silently flipped the sign of yvec
        # and of every rate coefficient while leaving the energy fractions summing to one.
        # the chained comparison (not a "<= 0.0" test) also rejects nan, whose comparisons are
        # always False, and inf, which would scale yvec to inf without ever raising
        if not 0.0 < depositionratedensity_ev < math.inf:
            msg = f"depositionratedensity_ev must be greater than zero and finite but is {depositionratedensity_ev}"
            raise ValueError(msg)

        self.depositionratedensity_ev = depositionratedensity_ev
        if override_n_e is not None and not 0.0 < override_n_e < math.inf:
            msg = f"override_n_e must be greater than zero and finite but is {override_n_e}"
            raise ValueError(msg)

        if not 0.0 < balance_tol < 1.0:
            msg = f"balance_tol must be between zero and one but is {balance_tol}"
            raise ValueError(msg)

        # None clears any previously-set override, so that n_e is calculated on demand from ion populations
        self._n_e_override = override_n_e
        self._n_e = None

        if self._balanced_elements:
            self._solve_ion_balance(balance_tol)
        else:
            self.balance_iterations = 0
            self._solve_matrix()

        if self.verbose:
            n_e = self.get_n_e()
            n_ion_tot = self.get_n_ion_tot()
            x_e = n_e / n_ion_tot if n_ion_tot > 0.0 else math.inf
            print(f" n_ion_tot: {n_ion_tot:.2e} [/cm3]        (total ion density)")
            print(f"       n_e: {n_e:.2e} [/cm3]        (free electron density)")
            print(f"       x_e: {x_e:.2e}               (electrons per nucleus)")
            print(f"deposition: {self.depositionratedensity_ev:7.2f}  [eV/s/cm3]")

        self._solved = True

    def _solve_matrix(self) -> None:
        # solve the matrix equation at the current populations and free electron density, and set yvec
        n_e = self.get_n_e()
        if n_e <= 0.0:
            # without free electrons there is no thermal loss channel, so electrons below the lowest
            # ionisation/excitation threshold have nowhere to deposit their energy and those rows of
            # the Spencer-Fano matrix are all zero
            reason = "every added ion is neutral" if self.ionpopdict else "no ions have been added"
            msg = (
                f"the free electron density is zero because {reason}. The Spencer-Fano"
                " equation is singular without a thermal electron loss channel, so add an ionised stage"
                " with add_ionisation() or pass override_n_e to solve()."
            )
            raise ValueError(msg)

        # the free-electron loss term is diagonal-only, so it is passed to the solver as a 1D
        # vector rather than copying the whole npts x npts matrix just to modify its diagonal
        lossvec = np.array([electronlossfunction(en_ev, n_e) for en_ev in self.engrid])

        # each matrix row is the integral form of the degradation equation (Kozma & Fransson
        # 1992 equation 7; equation 2 of Li et al. 2012) at one grid energy, with rhsvec
        # holding the integrated source term. Every process moves electrons to lower energies,
        # so y(E) depends only on y at higher energies (Kozma & Fransson 1992): only matrix
        # columns j >= i are populated and the matrix is upper triangular. K&F invert it with
        # an unspecified "standard matrix technique"; back-substitution from the highest energy
        # downward (the scheme K&F credit to Xu 1989) exploits the triangularity and is much
        # faster than a general LU solve.
        yvec_reference = solve_upper_triangular(self.sfmatrix, self.rhsvec, diag_add=lossvec)
        self.yvec = np.array(yvec_reference * self.depositionratedensity_ev / self.E_init_ev, dtype=np.float64)

    def _solve_ion_balance(self, balance_tol: float) -> None:
        # find the populations of the balanced elements and the free electron density, and solve the
        # matrix equation at them. On return, yvec, ionpopdict, and the matrix agree with each other.
        elements = list(self._balanced_elements.values())
        recomb_elements = [element for element in elements if element.recomb_ratecoeffs is not None]
        deposition = self.depositionratedensity_ev

        # a first solve has no ionisation rates yet, so one solution at the provisional populations
        # gives them. A later solve starts from the rates of the last solution: at fixed populations
        # the rates are proportional to the deposition rate density.
        if any(element.ratecoeffs_per_deposition is None for element in recomb_elements):
            self._solve_matrix()
            for element in recomb_elements:
                if element.ratecoeffs_per_deposition is None:
                    element.ratecoeffs_per_deposition = self._balanced_ratecoeffs_per_deposition(element)

        # the ratio coefficients n_{i+1} n_e / n_i of the recombination-balance elements, keyed by Z
        ratio_coeffs: dict[int, list[float]] = {}
        for element in recomb_elements:
            assert element.recomb_ratecoeffs is not None
            assert element.ratecoeffs_per_deposition is not None
            ratio_coeffs[element.Z] = [
                element.ratecoeffs_per_deposition[ion_stage] * deposition / element.recomb_ratecoeffs[ion_stage + 1]
                for ion_stage in element.ion_stages[:-1]
            ]

        def get_element_ratio_coeffs(element: _BalancedElement) -> Sequence[float]:
            return element.saha_factors if element.saha_factors is not None else ratio_coeffs[element.Z]

        max_residual = math.inf
        for iteration in range(1, BALANCE_MAXITER + 1):
            self.balance_iterations = iteration
            n_e_fixed = sum(
                (ion_stage - 1) * n_ion
                for (Z, ion_stage), n_ion in self.ionpopdict.items()
                if Z not in self._balanced_elements
            )
            n_e = (
                self._n_e_override
                if self._n_e_override is not None
                else solve_charge_neutral_n_e(
                    n_e_fixed,
                    [
                        (element.n_elem, element.ion_stages[0], get_element_ratio_coeffs(element))
                        for element in elements
                    ],
                )
            )
            for element in elements:
                fractions = get_ion_fractions(get_element_ratio_coeffs(element), n_e)
                self._set_balanced_populations(
                    element,
                    {
                        ion_stage: element.n_elem * frac
                        for ion_stage, frac in zip(element.ion_stages, fractions, strict=True)
                    },
                )

            # the loss function takes n_e from ionpopdict (or the override), so the populations and the
            # loss term agree to machine precision whatever the tolerance of the root find
            self._solve_matrix()

            if not recomb_elements:
                # the Saha ratios do not depend on the solution, so one pass is the answer
                break

            # the residual compares the ratios that the new solution gives with the ratios that gave the
            # populations in the matrix. At convergence the populations, yvec, and the matrix agree, and
            # n_i Gamma_i = n_{i+1} n_e alpha_{i+1} holds to the tolerance.
            max_residual = 0.0
            new_ratio_coeffs: dict[int, list[float]] = {}
            for element in recomb_elements:
                assert element.recomb_ratecoeffs is not None
                element.ratecoeffs_per_deposition = self._balanced_ratecoeffs_per_deposition(element)
                new_ratio_coeffs[element.Z] = []
                for index, ion_stage in enumerate(element.ion_stages[:-1]):
                    c_new = (
                        element.ratecoeffs_per_deposition[ion_stage]
                        * deposition
                        / element.recomb_ratecoeffs[ion_stage + 1]
                    )
                    c_old = ratio_coeffs[element.Z][index]
                    new_ratio_coeffs[element.Z].append(c_new)
                    if c_new != c_old:
                        max_residual = max(max_residual, abs(c_new - c_old) / max(c_new, c_old))

            if self.verbose:
                print(f"  ionisation balance iteration {iteration}: n_e {n_e:.4e} [/cm3], residual {max_residual:.2e}")
                for element in elements:
                    fractions = " ".join(
                        f"{ion_stage}: {self.ionpopdict[(element.Z, ion_stage)] / element.n_elem:.4e}"
                        for ion_stage in element.ion_stages
                    )
                    print(f"    Z={element.Z} ion fractions {fractions}")

            if max_residual <= balance_tol:
                break

            # mix in log space. A zero ratio comes from a zero cross section on the grid, so it stays zero.
            for Z, coeffs in ratio_coeffs.items():
                for index, c_new in enumerate(new_ratio_coeffs[Z]):
                    c_old = coeffs[index]
                    coeffs[index] = (
                        0.0
                        if c_new == 0.0 or c_old == 0.0
                        else c_old ** (1.0 - BALANCE_MIXING_WEIGHT) * c_new**BALANCE_MIXING_WEIGHT
                    )
        else:
            msg = (
                f"the ionisation balance did not converge in {BALANCE_MAXITER} iterations: the largest relative"
                f" change of a population ratio is {max_residual:.2e} (balance_tol {balance_tol})"
            )
            raise RuntimeError(msg)

        for element in recomb_elements:
            self._warn_top_stage_leak(element)

    def _balanced_ratecoeffs_per_deposition(self, element: _BalancedElement) -> dict[int, float]:
        # the ionisation rate coefficient per unit deposition rate density of each stage below the top
        return {
            ion_stage: self._calculate_ionisation_ratecoeff(element.Z, ion_stage) / self.depositionratedensity_ev
            for ion_stage in element.ion_stages[:-1]
        }

    def _warn_top_stage_leak(self, element: _BalancedElement) -> None:
        # the top stage of the chain has no stage to ionise into, so its ionisation rate must be small
        # compared with the total ionisation rate of the element (see BALANCE_TOP_STAGE_LEAK_WARN_FRACTION)
        Z = element.Z
        top = element.ion_stages[-1]
        rates = {
            ion_stage: self.ionpopdict[(Z, ion_stage)] * self._calculate_ionisation_ratecoeff(Z, ion_stage)
            for ion_stage in element.ion_stages
        }
        rate_total = sum(rates.values())
        if rates[top] > BALANCE_TOP_STAGE_LEAK_WARN_FRACTION * rate_total:
            warnings.warn(
                f"the ionisation rate out of the top stage {top} of Z={Z} ({rates[top]:.2e} /s/cm3) is not small"
                f" compared with the total ionisation rate of the element ({rate_total:.2e} /s/cm3), so about"
                f" {rates[top] / rate_total:.1%} of the element belongs in a higher stage. Extend the chain with a"
                f" recombination rate coefficient for ion stage {top + 1}.",
                stacklevel=3,
            )

    def calculate_nt_frac_excitation_ion(self, Z: int, ion_stage: int) -> float:
        if (Z, ion_stage) not in self.excitationlists:
            return 0.0

        # integral in Kozma & Fransson equation 9, but summed over all transitions for given ion
        deltaen = self.deltaen
        npts = len(self.engrid)

        xs_excitation_vec_sum_alltrans = np.zeros(npts)

        for trans in self.excitationlists[(Z, ion_stage)].values():
            xs_excitation_vec_sum_alltrans += trans.levelnumberdensity * trans.epsilon_trans_ev * trans.xs_vec

        return np.dot(xs_excitation_vec_sum_alltrans, self.yvec) * deltaen / self.depositionratedensity_ev

    @staticmethod
    def _npts_subgrid(width_ev: float, J: float) -> int:
        # nodes for a sub-grid spanning width_ev, given that the integrand varies on the scale of J.
        # Odd, so that a Simpson-style rule would see whole panels.
        npts = int(np.clip(NPTS_SUBGRID_PER_J * width_ev / J, NPTS_SUBGRID_MIN, NPTS_SUBGRID_MAX))

        return npts + 1 - npts % 2

    @staticmethod
    def _integrate_shell_secondaries(
        arr_e_p: npt.NDArray[np.float64],
        arr_y: npt.NDArray[np.float64],
        arr_xs: npt.NDArray[np.float64],
        e_s: npt.NDArray[np.float64] | float,
        ionpot_ev: float,
        J: float,
    ) -> float:
        # The integrand shared by both ionisation integrals of Kozma & Fransson equation 11 (the same
        # integrals appear in their differential equation 6): y(E') sigma(E') P(e_s, E')
        # over the primary energies arr_e_p. Both use E' as the variable of integration, since the first
        # one's epsilon differs from it only by a constant. Each caller supplies y and the cross section
        # however is cheapest for its own nodes: off the grid they have to be evaluated, but on it the
        # solution and the cached cross sections can be sliced directly.
        arr_psecondary = pynonthermal.collion.Psecondary_vec(e_p=arr_e_p, e_s=e_s, ionpot_ev=ionpot_ev, J=J)

        return float(np.trapezoid(arr_y * arr_xs * arr_psecondary, arr_e_p))

    def calculate_N_e(self, energy_ev: float) -> float:
        # N(E) of Kozma & Fransson 1992 equation 11: the rate at which electrons appear at an
        # energy E below the solved grid. Its three terms are electrons that excited an ion
        # from E + epsilon_trans, primaries degraded by an ionisation energy loss (with the
        # equation's lambda limit computed as enlambda below), and the secondaries of
        # ionisations by primaries above 2E + I. calculate_frac_heating() integrates
        # E * N(E) over [0, E_0] for the energy that thermalises below the grid.
        # not valid for energy > E_0
        if energy_ev == 0.0:
            return 0.0

        N_e = 0.0
        e_min = float(self.engrid[0])
        e_max = float(self.engrid[-1])
        deltaen = float(self.deltaen)
        lastindex = len(self.engrid) - 2

        for Z, ion_stage in self._get_all_ions():
            N_e_ion = 0.0
            n_ion = self.ionpopdict.get((Z, ion_stage), 0.0)

            for trans in self.excitationlists.get((Z, ion_stage), {}).values():
                # Interpolate y and the cross section at energy_ev + epsilon_trans_ev, as the two
                # ionisation integrals below also do, rather than snapping down to the grid point under
                # it. Snapping lands below the transition energy whenever E has not yet carried the sum
                # past it, and get_xs_excitation_vector makes the cross section exactly zero there, so
                # the whole transition silently dropped out: on the helium benchmark at E = 0.25 eV that
                # was 280 of 296 transitions and 43% of N_e. The interpolation is written out rather
                # than left to np.interp because this runs once per transition per quadrature node,
                # where the call overhead of np.interp costs more than the arithmetic; the grid is
                # uniform, so the index and weight are exact.
                en_upper = energy_ev + trans.epsilon_trans_ev
                if e_min <= en_upper <= e_max:
                    pos = (en_upper - e_min) / deltaen
                    i = min(int(pos), lastindex)
                    weight = pos - i
                    xsvec = trans.xs_vec
                    # the level population is absolute, so this term is not scaled by n_ion below
                    N_e += (
                        trans.levelnumberdensity
                        * (self.yvec[i] + weight * (self.yvec[i + 1] - self.yvec[i]))
                        * (xsvec[i] + weight * (xsvec[i + 1] - xsvec[i]))
                    )

            for channel in self._ionisation_channels.get((Z, ion_stage), ()):
                ionpot_ev = channel.ionpot_ev

                enlambda = min(e_max - energy_ev, energy_ev + ionpot_ev)
                J = channel.J_ev

                # Integral over epsilon from ionpot_ev to enlambda. Its width is at most energy_ev, which
                # is at most E_0 = emin_ev, so it is usually narrower than one cell of self.engrid and has
                # to be resolved on a sub-grid of its own. Evaluating it on self.engrid instead gave an
                # error that swung between missing the domain entirely and overcounting it by an order of
                # magnitude, depending on where the grid points happened to fall relative to ionpot_ev.
                if enlambda > ionpot_ev:
                    arr_epsilon = np.linspace(
                        ionpot_ev, enlambda, num=self._npts_subgrid(enlambda - ionpot_ev, J), dtype=np.float64
                    )
                    arr_e_p = energy_ev + arr_epsilon
                    # every node here lies between grid points, so y is interpolated and the cross section
                    # evaluated directly, the latter because it rises steeply from zero just above the
                    # ionisation threshold that starts this integral
                    N_e_ion += self._integrate_shell_secondaries(
                        arr_e_p=arr_e_p,
                        # asarray because np.interp is typed as returning a scalar for a scalar x
                        arr_y=np.asarray(np.interp(arr_e_p, self.engrid, self.yvec, left=0.0, right=0.0)),
                        arr_xs=channel.xs(arr_e_p),
                        e_s=arr_epsilon - ionpot_ev,
                        ionpot_ev=ionpot_ev,
                        J=J,
                    )

                # Integral from 2E + I up to E_max. Away from the lower limit self.engrid resolves this
                # domain, but the Psecondary normalisation 1/atan((E' - I) / 2J) varies on the scale of
                # J = 0.6 * I rather than deltaen, and 2E + I sits just above the threshold for the small
                # E this is called with. A single trapezoid panel from 2E + I to the first grid point
                # above it therefore spans the steepest, most convex part of the integrand and
                # overestimates it one-sidedly: for Al I on a 1-3000 eV grid at npts=600 that one panel
                # made the integral 35% high. Resolve the leading panels on a sub-grid, and let the main
                # grid carry the smooth remainder. An empty domain is skipped rather than integrated
                # backwards.
                en_lower2 = 2 * energy_ev + ionpot_ev
                if en_lower2 < e_max:
                    startindex = self.get_energyindex_gteq(en_ev=en_lower2)
                    # cover whole grid cells with the sub-grid, so that it meets the grid part exactly
                    stopindex = min(startindex + NCELLS_SECOND_INTEGRAL_SUBGRID, len(self.engrid) - 1)
                    en_upper_low = float(self.engrid[stopindex])
                    arr_e_p_low = np.linspace(
                        en_lower2, en_upper_low, num=self._npts_subgrid(en_upper_low - en_lower2, J), dtype=np.float64
                    )
                    N_e_ion += self._integrate_shell_secondaries(
                        arr_e_p=arr_e_p_low,
                        arr_y=np.asarray(np.interp(arr_e_p_low, self.engrid, self.yvec, left=0.0, right=0.0)),
                        arr_xs=channel.xs(arr_e_p_low),
                        e_s=energy_ev,
                        ionpot_ev=ionpot_ev,
                        J=J,
                    )
                    # the rest is smooth on the scale of J, so y comes from the solution and the
                    # cross section from the on-grid array of the channel, with no new evaluation
                    if stopindex < len(self.engrid) - 1:
                        N_e_ion += self._integrate_shell_secondaries(
                            arr_e_p=self.engrid[stopindex:],
                            arr_y=self.yvec[stopindex:],
                            arr_xs=channel.xs_grid[stopindex:],
                            e_s=energy_ev,
                            ionpot_ev=ionpot_ev,
                            J=J,
                        )

            N_e += n_ion * N_e_ion

        # source term not here because it should be zero at the low end anyway

        return N_e

    def calculate_frac_heating(self) -> float:
        # fraction of the deposited energy that heats the free thermal electrons: Kozma &
        # Fransson 1992 equation 8. Its three parts below are the loss-function integral over
        # the solved grid, the boundary term E_0 y(E_0) L(E_0) for the electrons flowing
        # through the bottom of the grid, and the energy of the electrons that first appear
        # below E_0 (N(E) of KF92 equation 11), all divided by the deposition rate density.
        frac_heating = 0.0
        E_0 = self.engrid[0]
        n_e = self.get_n_e()

        deltaen = self.deltaen
        frac_heating += (
            deltaen
            / self.depositionratedensity_ev
            * sum(electronlossfunction(float(en_ev), n_e) * self.yvec[i] for i, en_ev in enumerate(self.engrid))
        )

        frac_heating_E_0_part = E_0 * self.yvec[0] * electronlossfunction(E_0, n_e) / self.depositionratedensity_ev

        frac_heating += frac_heating_E_0_part

        # if self.verbose:
        #     print(f"            frac_heating E_0 * y * l(E_0) part: {frac_heating_E_0_part:.5f}")

        # the heating-only matrix routes all deposited energy through the loss function, so the
        # N_e term below E_0 would count the same energy twice. calculate_N_e() itself still
        # gives the secondary-electron rate of the approximate solution.
        if not self.heating_only_approximation:
            # E * N_e(E) is smooth over [0, E_0], so Simpson's rule earns its extra order here: it
            # reaches 5e-5 of the converged value at 9 nodes where the trapezoid rule needs several
            # hundred, and every node costs a full calculate_N_e(). Summing the nodes, as the code
            # did before, counted half a node too much at each end of the interval.
            arr_en = np.linspace(0.0, E_0, num=NPTS_SUB_E0_INTEGRAL, endpoint=True, dtype=np.float64)
            arr_en_N_e = np.array([en_ev * self.calculate_N_e(en_ev) for en_ev in arr_en], dtype=np.float64)
            integral_e_n_e = integrate_simpson_uniform(arr_en_N_e, arr_en)
            frac_heating_N_e = integral_e_n_e / self.depositionratedensity_ev

            if self.verbose:
                print(f" frac_heating(E<EMIN): {frac_heating_N_e:.5f}")

            frac_heating += frac_heating_N_e

        self._frac_heating = frac_heating
        return frac_heating

    def _reset_channel_fractions(self) -> None:
        # clear the per-ion accumulators, which analyse_ntspectrum() sums into
        self._analysed = False
        self._frac_ionisation_tot = 0.0
        self._frac_excitation_tot = 0.0
        self._frac_ionisation_ion = {}
        self._frac_excitation_ion = {}
        self._nt_ionisation_ratecoeff = {}
        self._eff_ionpot = {}

    def reset_solution_analysis(self) -> None:
        self._reset_channel_fractions()
        self._frac_heating = None

    def analyse_ntspectrum(self) -> None:
        self._require_solved()
        # keep any frac_heating already computed for this solution, since it only depends on yvec
        self._reset_channel_fractions()

        deltaen = self.deltaen

        if self.verbose:
            print(f"    n_e_nt: {self.get_n_e_nt():.2e} [/cm3]")

        n_ion_tot = self.get_n_ion_tot()
        for Z, ion_stage in self._get_all_ions():
            n_ion = self.ionpopdict.get((Z, ion_stage), 0.0)
            X_ion = n_ion / n_ion_tot if n_ion_tot > 0.0 else 0.0
            # an ion added only for excitation has no ionisation channels in the matrix
            channels = self._ionisation_channels.get((Z, ion_stage), [])

            ionpot_valence = min((channel.ionpot_ev for channel in channels), default=None)

            if self.verbose:
                valencestr = (
                    "no ionisation channel" if ionpot_valence is None else f"valence potential {ionpot_valence:.1f} eV"
                )
                print(f"\n====> Z={Z:2d} ion_stage {ion_stage} {at.get_ionstring(Z, ion_stage)} ({valencestr})")

                print(f"               n_ion: {n_ion:.2e} [/cm3]")
                print(f"     n_ion/n_ion_tot: {X_ion:.5f}")

            self._frac_ionisation_ion[(Z, ion_stage)] = 0.0
            eta_over_ionpot_sum = 0.0
            for channel in channels:
                ar_xs_array = channel.xs_grid

                # the channel's part of the ionisation fraction eta_ic of Kozma & Fransson 1992
                # equation 10: n_ion * ionpot * the integral of y(E) sigma_ic(E) dE, divided
                # by the deposition rate density
                frac_ionisation_shell = (
                    n_ion * channel.ionpot_ev * np.dot(self.yvec, ar_xs_array) * deltaen / self.depositionratedensity_ev
                )

                if self.verbose:
                    print(
                        f"frac_ionisation_shell({channel.key}):"
                        f" {frac_ionisation_shell:.4f} (ionpot"
                        f" {channel.ionpot_ev:.2f} eV)"
                    )

                # the heating-only approximation lets the fractions exceed one, so in that mode
                # only a negative or NaN value is invalid (NaN fails every comparison)
                if not frac_ionisation_shell >= 0.0 or (
                    not self.heating_only_approximation and frac_ionisation_shell > 1.0
                ):
                    warnings.warn(
                        f"invalid frac_ionisation_shell of {frac_ionisation_shell} included in the total",
                        stacklevel=2,
                    )

                self._frac_ionisation_ion[(Z, ion_stage)] += frac_ionisation_shell
                eta_over_ionpot_sum += frac_ionisation_shell / channel.ionpot_ev

            self._frac_ionisation_tot += self._frac_ionisation_ion[(Z, ion_stage)]

            # the ion's effective ionisation potential (Kozma & Fransson 1992 equation 12,
            # modified to a sum over the ion's shells): the shell ionisation rates add, and
            # each is inversely proportional to its potential, so the ion's rate follows from
            # X_ion / eff_ionpot = eta_shell_a / ionpot_a + eta_shell_b / ionpot_b + ...
            eff_ionpot = float(X_ion / eta_over_ionpot_sum) if eta_over_ionpot_sum else float("inf")
            self._eff_ionpot[(Z, ion_stage)] = eff_ionpot

            # eff_ionpot_usevalence = (
            #     ionpot_valence * X_ion / self._frac_ionisation_ion[(Z, ion_stage)]
            #     if self._frac_ionisation_ion[(Z, ion_stage)] > 0. else float('inf'))

            if self.verbose:
                print(f"     frac_ionisation: {self._frac_ionisation_ion[(Z, ion_stage)]:.4f}")

            frac_excitation_thision = self.calculate_nt_frac_excitation_ion(Z, ion_stage)

            if not frac_excitation_thision >= 0.0 or (
                not self.heating_only_approximation and frac_excitation_thision > 1.0
            ):
                # keep it in the total, as for frac_ionisation_shell, so that the energy conservation
                # check below still sees the problem instead of a total that silently lost a channel
                warnings.warn(
                    f"invalid frac_excitation_ion of {frac_excitation_thision} included in the total",
                    stacklevel=2,
                )

            self._frac_excitation_ion[(Z, ion_stage)] = frac_excitation_thision
            self._frac_excitation_tot += frac_excitation_thision

            if self.verbose and frac_excitation_thision > 0.0:
                print(f"     frac_excitation: {self._frac_excitation_ion[(Z, ion_stage)]:.4f}")

            # ionisation rate coefficient: Kozma & Fransson 1992 equation 13 with the deposition rate
            # density per ion in place of their gamma-ray energy absorption rate 4 pi J_gamma
            # sigma_gamma, divided by the effective potential. That equals the direct integral
            # of y(E) sigma(E) dE summed over the shells, which stays finite for a zero population.
            self._nt_ionisation_ratecoeff[(Z, ion_stage)] = self._calculate_ionisation_ratecoeff(Z, ion_stage)
            if self.verbose and ionpot_valence is not None:
                workfn_ev = get_workfn_ev(
                    Z,
                    ion_stage,
                    ionpot_ev=ionpot_valence,
                    Zbar=get_Zbar(ions=tuple(self.ionpopdict.keys()), ionpopdict=self.ionpopdict),
                )
                print(f"   workfn eff_ionpot: {eff_ionpot:8.2f} [eV]")
                print(f"       approx workfn: {workfn_ev:8.2f} [eV] (without Spencer-Fano solution)")
                # print(f'  eff_ionpot_usevalence: {eff_ionpot_usevalence:.2f} [eV]')
                print(f"ionisation ratecoeff: {self._nt_ionisation_ratecoeff[(Z, ion_stage)]:.2e} [/s]")

        # n_e_nt = get_n_e_nt(engrid, yvec)
        # print(f'               n_e_nt: {n_e_nt:.2e} /s/cm3')

        if self.verbose:
            print()
            print(f"  frac_excitation_tot: {self._frac_excitation_tot:.4f}")
            print(f"  frac_ionisation_tot: {self._frac_ionisation_tot:.4f}")

        frac_heating = self.get_frac_heating()

        self._analysed = True

        frac_sum = self._frac_excitation_tot + self._frac_ionisation_tot + frac_heating

        if self.verbose:
            print(f"         frac_heating: {frac_heating:.4f}")
            print(f"             frac_sum: {frac_sum:.4f}")

        # every deposited eV must end up in exactly one of the three channels. The tolerance is loose
        # enough to absorb the discretisation error of a coarse energy grid, so a warning here means
        # a channel is being double counted or omitted rather than merely under-resolved. The
        # heating-only approximation makes the sum exceed one, so there the conserved quantity
        # is the heating fraction alone, which must be close to one.
        if self.heating_only_approximation:
            if not math.isclose(frac_heating, 1.0, rel_tol=0.05):
                warnings.warn(
                    f"the heating fraction is {frac_heating:.4f} instead of 1 with the heating-only"
                    f" approximation. Try a finer energy grid (npts is {len(self.engrid)}).",
                    stacklevel=2,
                )
        elif not math.isclose(frac_sum, 1.0, rel_tol=0.05):
            warnings.warn(
                f"the energy fractions sum to {frac_sum:.4f} instead of 1: heating {frac_heating:.4f},"
                f" ionisation {self._frac_ionisation_tot:.4f}, excitation {self._frac_excitation_tot:.4f}."
                f" Try a finer energy grid (npts is {len(self.engrid)}).",
                stacklevel=2,
            )

    def _calculate_ionisation_ratecoeff(self, Z: int, ion_stage: int) -> float:
        # the non-thermal ionisation rate coefficient [s^-1] of one ion from the solved y(E): the
        # integral of y(E) sigma(E) dE summed over the ion's channels. It does not depend on the
        # ion's population, so the ionisation balance can use it for a stage with zero population.
        return float(
            self.deltaen
            * sum(np.dot(self.yvec, channel.xs_grid) for channel in self._ionisation_channels.get((Z, ion_stage), []))
        )

    def get_n_e_nt(self) -> float:
        """Get the number density of non-thermal electrons in cm^-3."""
        self._require_solved()
        arr_velocity = CLIGHT * np.sqrt(get_betasq(self.engrid))  # cm/s

        return float(np.sum(self.yvec / arr_velocity) * self.deltaen)

    def get_frac_heating(self) -> float:
        self._require_solved()
        if self._frac_heating is None:
            return self.calculate_frac_heating()

        return self._frac_heating

    def get_frac_excitation_tot(self) -> float:
        self._require_solved()
        if not self._analysed:
            self.analyse_ntspectrum()

        return self._frac_excitation_tot

    def get_frac_ionisation_tot(self) -> float:
        self._require_solved()
        if not self._analysed:
            self.analyse_ntspectrum()

        return self._frac_ionisation_tot

    def get_frac_ionisation_ion(self, Z: int, ion_stage: int) -> float:
        self._require_solved()
        if not self._analysed:
            self.analyse_ntspectrum()

        return self._frac_ionisation_ion[(Z, ion_stage)]

    def get_eff_ionpot(self, Z: int, ion_stage: int) -> float:
        """Get the ion's effective ionisation potential in eV (Kozma & Fransson 1992 equation 12)."""
        self._require_solved()
        if not self._analysed:
            self.analyse_ntspectrum()

        return self._eff_ionpot[(Z, ion_stage)]

    def get_ionisation_ratecoeff(self, Z: int, ion_stage: int) -> float:
        """Get the non-thermal ionisation rate coefficient in s^-1 for one ion.

        This is Kozma & Fransson 1992 equation 13 with the deposition rate density per ion in
        place of their gamma-ray energy absorption rate, divided by the effective ionisation
        potential. It scales with depositionratedensity_ev.
        """
        self._require_solved()
        if not self._analysed:
            self.analyse_ntspectrum()

        return self._nt_ionisation_ratecoeff[(Z, ion_stage)]

    def get_excitation_ratecoeff(self, Z: int, ion_stage: int, transitionkey: t.Any) -> float:
        """Get the non-thermal excitation rate coefficient in s^-1 for one transition.

        This is the integral of y(E) * sigma(E) dE in Kozma & Fransson equation 9, matching the
        convention of get_ionisation_ratecoeff(). It scales with depositionratedensity_ev.

        transitionkey is the key given to add_excitation(); for transitions added by
        add_ion_ltepopexcitation() it is (lower level index, upper level index).
        """
        self._require_solved()
        trans = self.excitationlists[(Z, ion_stage)][transitionkey]

        return float(np.dot(trans.xs_vec, self.yvec) * self.deltaen)

    def get_frac_sum(self) -> float:
        return self.get_frac_heating() + self.get_frac_excitation_tot() + self.get_frac_ionisation_tot()

    def get_d_etaheating_by_d_en_vec(self) -> list[float]:
        self._require_solved()
        return [
            self.electronlossfunction(self.engrid[i]) * self.yvec[i] / self.depositionratedensity_ev
            for i in range(len(self.engrid))
        ]

    def get_d_etaexcitation_by_d_en_vec(self) -> npt.NDArray[np.float64]:
        self._require_solved()
        part_integrand = np.zeros(len(self.engrid))

        for Z, ion_stage in self.excitationlists:
            for trans in self.excitationlists[(Z, ion_stage)].values():
                part_integrand += (
                    trans.levelnumberdensity * trans.epsilon_trans_ev * trans.xs_vec / self.depositionratedensity_ev
                )

        return self.yvec * part_integrand

    def get_d_etaion_by_d_en_vec(self) -> npt.NDArray[np.float64]:
        self._require_solved()
        part_integrand = np.zeros(len(self.engrid))

        for (Z, ion_stage), channels in self._ionisation_channels.items():
            n_ion = self.ionpopdict[(Z, ion_stage)]

            for channel in channels:
                part_integrand += n_ion * channel.ionpot_ev * channel.xs_grid / self.depositionratedensity_ev

        return self.yvec * part_integrand

    def plot_yspectrum(
        self,
        en_y_on_d_en: bool = False,
        xscalelog: bool = False,
        outputfilename: Path | str | None = None,
        axis: mplax.Axes | None = None,
    ) -> None:
        """Plot the solved degradation spectrum y(E) against electron energy.

        en_y_on_d_en:
            plot log(E y(E)), the spectrum per unit log energy, instead of log y(E)
        xscalelog:
            use a logarithmic energy axis
        outputfilename:
            save the figure to this path; None shows it interactively instead
        axis:
            draw into this existing matplotlib Axes instead of creating (and saving or
            showing) a new figure
        """
        self._require_solved()
        fs = 12
        fig = None
        if axis is None:
            fig, ax = plt.subplots(
                nrows=1,
                ncols=1,
                sharex=True,
                figsize=(5, 4),
                tight_layout={"pad": 0.5, "w_pad": 0.3, "h_pad": 0.3},
            )
        else:
            ax = axis

        if en_y_on_d_en:
            arr_y = np.log10(self.yvec * self.engrid)
            ax.set_ylabel(r"log(E y(E))", fontsize=fs)
        else:
            arr_y = np.log10(self.yvec)
            ax.set_ylabel(r"log y [y (e$^-$ / cm$^2$ / s / eV)]", fontsize=fs)

        ax.plot(self.engrid, arr_y, marker="None", lw=1.5, color="black")
        # axes[0].plot(engrid, np.log10(yvec), marker="None", lw=1.5, color='black')
        # axes[0].set_ylabel(r'log y(E) [s$^{-1}$ cm$^{-2}$ eV$^{-1}$]', fontsize=fs)
        # axes[0].set_ylim(bottom=15.5, top=19.)

        if xscalelog:
            ax.set_xscale("log")
        ax.set_xlim(left=min(1.0, self.engrid[0]))
        ax.set_xlim(right=self.engrid[-1] * 1.0)
        ax.set_xlabel(r"Electron energy [eV]", fontsize=fs)
        if axis is None:
            if outputfilename is not None:
                print(f"Saving '{outputfilename}'")
                assert fig is not None
                fig.savefig(str(outputfilename))
                plt.close()
            else:
                plt.show()

    def plot_channels(
        self, outputfilename: Path | str | None = None, axis: mplax.Axes | None = None, xscalelog: bool = False
    ) -> None:
        """Plot each electron energy's contribution to ionisation, excitation, and heating.

        The curves are E d(eta)/dE for each deposition channel, scaled so the largest peak is one
        (compare Kozma & Fransson 1992 figure 2).

        outputfilename:
            save the figure to this path; None shows it interactively instead
        axis:
            draw into this existing matplotlib Axes instead of creating (and saving or
            showing) a new figure
        xscalelog:
            use a logarithmic energy axis
        """
        self._require_solved()
        fs = 12
        fig = None
        if axis is None:
            fig, ax = plt.subplots(
                nrows=1,
                ncols=1,
                sharex=True,
                figsize=(5, 4),
                tight_layout={"pad": 0.5, "w_pad": 0.3, "h_pad": 0.3},
            )
        else:
            ax = axis

        E_0 = self.engrid[0]

        # E_init_ev = np.dot(engrid, sourcevec) * deltaen
        # d_etasource_by_d_en_vec = engrid * sourcevec / E_init_ev
        # axes[0].plot(engrid[1:], d_etasource_by_d_en_vec[1:], marker="None", lw=1.5, color='blue', label='Source')

        d_etaion_by_d_en_vec = self.get_d_etaion_by_d_en_vec()

        d_etaexc_by_d_en_vec = self.get_d_etaexcitation_by_d_en_vec()

        d_etaheat_by_d_en_vec = self.get_d_etaheating_by_d_en_vec()

        # The plot extends below E_0 to show where the channels stand there. Every ionisation threshold
        # is above E_0 (add_ionisation rejects any that are not), so that curve is genuinely zero;
        # excitation is drawn flat because add_excitation does allow a transition energy below E_0.
        # The heating curve stops at E_0 instead of continuing, because l(E) * y(E) needs y, which the
        # solver only has above E_0. The energy that thermalises below E_0 is in get_frac_heating()
        # (via the integral of E * N_e over [0, E_0]) but has no per-energy curve to draw here.
        engrid_low = np.arange(0.0, E_0, E_0 / 20.0, dtype=float)
        npts_low = len(engrid_low)
        engridfull = np.append(engrid_low, self.engrid)

        # delta_E_y_on_dE = np.zeros(npts)
        # for i in range(len(engrid) - 1):
        #     # delta_E_y_on_dE[i] = ((yvec[i + 1] * engrid[i + 1]) - (yvec[i] * engrid[i]))
        #     #     / (engrid[i + 1] - engrid[i])
        #     delta_E_y_on_dE[i] = yvec[i] * engrid[i]
        # axes[0].plot(engrid, np.log10(delta_E_y_on_dE), marker="None", lw=1.5, color='black', label='')
        # axes[0].set_ylabel(r'log d(E y(E)) / dE', fontsize=fs)

        detaymax = max(
            [
                float(np.max(d_etaion_by_d_en_vec * self.engrid)),
                float(np.max(d_etaexc_by_d_en_vec * self.engrid)),
                float(np.max(d_etaheat_by_d_en_vec * self.engrid)),
            ]
        )
        ax.plot(
            engridfull,
            np.append(np.zeros(npts_low), d_etaion_by_d_en_vec) * engridfull / detaymax,
            marker="None",
            lw=1.5,
            color="C0",
            label="Ionisation",
        )

        # test the curve itself rather than get_frac_excitation_tot(), so that drawing a plot does not
        # trigger analyse_ntspectrum() and its diagnostic warnings as a side effect
        if d_etaexc_by_d_en_vec.any():
            ax.plot(
                engridfull,
                np.append(np.zeros(npts_low), d_etaexc_by_d_en_vec) * engridfull / detaymax,
                marker="None",
                lw=1.5,
                color="C1",
                label="Excitation",
            )

        ax.plot(
            self.engrid,
            (np.array(d_etaheat_by_d_en_vec) * self.engrid) / detaymax,
            marker="None",
            lw=1.5,
            color="C2",
            label="Heating",
        )

        ax.set_ylim(bottom=0, top=1.0)
        ax.legend(loc="best", handlelength=2, frameon=False, numpoints=1, prop={"size": 10})
        ax.set_ylabel(r"E d$\eta$ / dE [eV$^{-1}$]", fontsize=fs)

        #    ax.annotate(modellabel, xy=(0.97, 0.95), xycoords='axes fraction', horizontalalignment='right',
        #                verticalalignment='top', fontsize=fs)
        if xscalelog:
            ax.set_xscale("log")
        # ax.set_yscale('log')
        ax.set_xlim(left=min(1.0, self.engrid[0]))
        ax.set_xlim(right=self.engrid[-1] * 1.0)
        ax.set_xlabel(r"Electron energy [eV]", fontsize=fs)
        if axis is None:
            if outputfilename is not None:
                print(f"Saving '{outputfilename}'")
                assert fig is not None
                fig.savefig(str(outputfilename))
                plt.close()
            else:
                plt.show()

    def plot_spec_channels(self, outputfilename: Path | str | None = None, xscalelog: bool = False) -> None:
        """Plot the degradation spectrum and the deposition channels as two stacked panels.

        outputfilename:
            save the figure to this path; None shows it interactively instead
        xscalelog:
            use a logarithmic energy axis
        """
        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            figsize=(4.5, 5),
            tight_layout={"pad": 0.5, "w_pad": 0.3, "h_pad": 0.3},
        )
        assert isinstance(axes, np.ndarray)

        self.plot_yspectrum(axis=axes[0], en_y_on_d_en=True, xscalelog=xscalelog)

        self.plot_channels(axis=axes[1], xscalelog=xscalelog)

        if outputfilename is not None:
            print(f"Saving '{outputfilename}'")
            fig.savefig(str(outputfilename))
            plt.close()
        else:
            plt.show()
