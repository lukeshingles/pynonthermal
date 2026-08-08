from __future__ import annotations

import math
import typing as t
import warnings
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
from pynonthermal.base import get_Zbar
from pynonthermal.constants import CLIGHT
from pynonthermal.constants import K_B

# Nodes for the sub-grids that resolve the parts of the Kozma & Fransson equation 6 integrals that the
# solver's own energy grid cannot. Both integrands vary on the scale of J = 0.6 * ionpot_ev, so the node
# count is set from the domain width in units of J, with a floor for the narrow domains that
# calculate_frac_heating produces and a cap for the wide ones a direct calculate_N_e call can ask for.
NPTS_SUBGRID_MIN: int = 65
NPTS_SUBGRID_MAX: int = 1025
NPTS_SUBGRID_PER_J: int = 32

# how many whole grid cells above the lower limit of the second equation 6 integral get the sub-grid
# treatment. The integrand is steepest at that limit and smooth a few J above it.
NCELLS_SECOND_INTEGRAL_SUBGRID: int = 2

# Number of nodes for the integral over E in [0, E_0] of Kozma & Fransson equation 8. This one is not
# free: every node costs a full calculate_N_e() over all ions and shells, so it dominates the cost of
# the analysis. It is integrated by Simpson's rule, which is fourth order here and reaches an accuracy
# at 9 nodes that the trapezoid rule needs several hundred for. The node count is a property of this
# integral alone: the old ceil(E_0 / deltaen) * 5 tied it to the main grid, which gave 5 nodes when
# emin_ev was below the grid spacing and several hundred when it was well above.
NPTS_SUB_E0_INTEGRAL: int = 9

SUBSHELLNAMES = [
    "K ",
    "L1",
    "L2",
    "L3",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "N1",
    "N2",
    "N3",
    "N4",
    "N5",
    "N6",
    "N7",
    "O1",
    "O2",
    "O3",
    "O4",
    "O5",
    "O6",
    "O7",
    "P1",
    "P2",
    "P3",
    "P4",
    "Q1",
]


def solve_upper_triangular(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Solve a x = b for upper-triangular a by back-substitution."""
    n = b.shape[0]
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - a[i, i + 1 :] @ x[i + 1 :]) / a[i, i]
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

    The Spencer-Fano equation is a differential equation that describes the energy deposition
    of non-thermal electrons in a plasma. The solution of the Spencer-Fano equation gives the
    energy density of the non-thermal electrons as a function of energy. The energy density
    can be used to calculate the heating rate, ionisation rate, and excitation rate of the plasma.
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
    _ionisation_ions: set[tuple[int, int]]
    _collionrows_ion: dict[tuple[int, int], list[dict[str, t.Any]]]
    _shell_xs: dict[tuple[int, int, int, int, float], npt.NDArray[np.float64]]
    depositionratedensity_ev: float
    ionpopdict: dict[tuple[int, int], float]
    excitationlists: dict[tuple[int, int], dict[t.Any, tuple[float, npt.NDArray[np.float64], float]]]
    verbose: bool
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

    def __init__(
        self,
        emin_ev: float = 1.0,
        emax_ev: float = 3000.0,
        npts: int = 4096,
        verbose: bool = False,
        use_ar1985: bool = False,
    ) -> None:
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

        self._collionrows_ion = {}  # cache of collisional ionisation shell data per (Z, ion_stage)
        self._shell_xs = {}  # cache of ionisation cross section arrays per shell

        self.ionpopdict = {}  # key is (Z, ion_stage) value is number density
        self._ionisation_ions = set()  # ions passed to add_ionisation(), i.e. those with ionisation channels

        # key is (Z, ion_stage) value is {levelkey : (levelnumberdensity, xs_vec, epsilon_trans_ev)}
        self.excitationlists = {}

        self.verbose = verbose
        self.engrid = np.linspace(emin_ev, emax_ev, num=npts, endpoint=True, dtype=float)
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

        # integral of the source from each energy to the top of the grid
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

    def _register_ion_population(self, Z: int, ion_stage: int, n_ion: float) -> None:
        # an ion's number density must agree between its ionisation and excitation calls
        if n_ion < 0.0:
            msg = f"n_ion must be non-negative but is {n_ion}"
            raise ValueError(msg)

        n_ion_existing = self.ionpopdict.get((Z, ion_stage))
        if n_ion_existing is not None and not math.isclose(n_ion_existing, n_ion, rel_tol=1e-6):
            msg = f"Can't add Z={Z} ion_stage {ion_stage} twice with different populations"
            raise ValueError(msg)

        self.ionpopdict[(Z, ion_stage)] = n_ion
        # the free electron density derived from the ion populations is no longer current
        self._n_e = None

    def _get_ion_collion_rows(self, Z: int, ion_stage: int) -> list[dict[str, t.Any]]:
        # collisional ionisation shell data rows for one ion (cached)
        key = (Z, ion_stage)
        if key not in self._collionrows_ion:
            self._collionrows_ion[key] = self.dfcollion.filter(
                (pl.col("Z") == Z) & (pl.col("ion_stage") == ion_stage)
            ).to_dicts()
        return self._collionrows_ion[key]

    def _get_shell_xs(self, shell: dict[str, t.Any]) -> npt.NDArray[np.float64]:
        # ionisation cross sections on the energy grid for one shell (cached)
        key = (int(shell["Z"]), int(shell["ion_stage"]), int(shell["n"]), int(shell["l"]), float(shell["ionpot_ev"]))
        if key not in self._shell_xs:
            self._shell_xs[key] = pynonthermal.collion.get_arxs_array_shell(self.engrid, shell)
        return self._shell_xs[key]

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

        levelnumberdensity:
            the level population density in cm^-3
        xs_vec:
            an array of cross sections in cm^2 defined at every energy in the SpencerFanoSolver.engrid array [eV]
        epsilon_trans_ev:
            the transition energy in eV
        transitionkey:
            any key to uniquely identify the transition so that the rate coefficient can be retrieved later
        """
        self._require_not_solved("add excitation")
        if len(xs_vec) != len(self.engrid):
            msg = f"xs_vec has {len(xs_vec)} points but engrid has {len(self.engrid)}"
            raise ValueError(msg)

        # a non-positive transition energy would put matrix entries below the diagonal, where the
        # triangular solve silently discards them, and a negative population or cross section would
        # produce a negative excitation fraction that no later check looks for
        if epsilon_trans_ev <= 0.0:
            msg = f"epsilon_trans_ev must be greater than zero but is {epsilon_trans_ev}"
            raise ValueError(msg)
        if epsilon_trans_ev > self.engrid[-1]:
            msg = (
                f"epsilon_trans_ev ({epsilon_trans_ev} eV) is above the top of the energy grid"
                f" ({self.engrid[-1]} eV), so no electron the solver represents can drive the transition"
            )
            raise ValueError(msg)
        # >= rather than < 0, so that NaN, for which every comparison is False, is rejected too
        if not levelnumberdensity >= 0.0:
            msg = f"levelnumberdensity must be non-negative but is {levelnumberdensity}"
            raise ValueError(msg)
        if not (xs_vec >= 0.0).all():
            msg = f"xs_vec must be non-negative and finite but its lowest value is {xs_vec.min()}"
            raise ValueError(msg)

        if (Z, ion_stage) not in self.excitationlists:
            self.excitationlists[(Z, ion_stage)] = {}

        if transitionkey is None:
            transitionkey = len(self.excitationlists[(Z, ion_stage)])  # simple number index

        if transitionkey in self.excitationlists[(Z, ion_stage)]:
            msg = f"Transition {transitionkey} already added for Z={Z} ion_stage={ion_stage}"
            raise ValueError(msg)
        self.excitationlists[(Z, ion_stage)][transitionkey] = (
            levelnumberdensity,
            xs_vec,
            epsilon_trans_ev,
        )
        vec_xs_excitation_levelnumberdensity_deltae = levelnumberdensity * self.deltaen * xs_vec
        xsstartindex = self.get_energyindex_lteq(en_ev=epsilon_trans_ev)
        npts = len(self.engrid)

        # the uniform grid makes every row's stop index expressible in one vectorised operation
        stopindices = np.clip(
            np.floor((self.engrid + epsilon_trans_ev - self.engrid[0]) / self.deltaen).astype(np.int64), 0, npts - 1
        )
        # clamp to deltaen for rows where en + epsilon_trans_ev extends beyond the top of the
        # grid (the excitation integral is truncated at the grid boundary)
        delta_en_actuals = np.minimum(self.engrid + epsilon_trans_ev - self.engrid[stopindices], self.deltaen)

        for i in range(npts):
            stopindex = stopindices[i]
            startindex = max(i, xsstartindex)
            self.sfmatrix[i, startindex:stopindex] += vec_xs_excitation_levelnumberdensity_deltae[startindex:stopindex]

            # do the last bit separately because we're not using the full deltaen interval
            self.sfmatrix[i, stopindex] += (
                vec_xs_excitation_levelnumberdensity_deltae[stopindex] * delta_en_actuals[i] / self.deltaen
            )

    def add_ion_ltepopexcitation(
        self,
        Z: int,
        ion_stage: int,
        n_ion: float,
        temperature: float = 3000,
        adata_polars: pl.DataFrame | None = None,
        use_collstrengths: bool = True,
        maxnlevelslower: int | None = 5,
        maxnlevelsupper: int | None = 250,
    ) -> None:
        """Add bound-bound excitations of one ion, with LTE level populations at the given temperature.

        Transitions whose energy lies outside the solver's energy grid are dropped: above emax_ev no
        electron the solver represents can drive them, and below emin_ev Kozma & Fransson 1992 take every
        electron to have thermalised already, so their energy is accounted for as heating instead. Unlike
        the equivalent case in add_ionisation(), this cannot be raised as an error, because real ions have
        fine-structure transitions far below any usable emin_ev. Pass verbose=True to the solver to see how
        many transitions were dropped.
        """
        self._require_not_solved("add excitation")
        if adata_polars is not None:
            self.adata_polars = adata_polars

        if self.adata_polars is None:
            # use ARTIS atomic data read by the artistools package to get the levels
            self.adata_polars = at.atomic.get_levels(
                Path(pynonthermal.DATADIR, "artis_files"),
                get_transitions=True,
                derived_transitions_columns=["epsilon_trans_ev", "lambda_angstroms", "lower_g", "upper_g"],
            )

        assert self.adata_polars is not None

        ion = self.adata_polars.filter(pl.col("Z") == Z).filter(pl.col("ion_stage") == ion_stage)
        if ion.is_empty():
            msg = f"No excitation data for Z={Z} ion_stage {ion_stage} in internal database."
            raise ValueError(msg)

        # register the population so that this ion counts towards n_e and n_ion_tot even when
        # add_ionisation() was never called for it
        self._register_ion_population(Z, ion_stage, n_ion)

        dfpops_thision = ion["levels"].item()

        ltepartfunc = dfpops_thision.select(pl.col("g") * (-pl.col("energy_ev") / K_B / temperature).exp()).sum().item()
        dfpops_thision = (
            dfpops_thision.rename({"levelindex": "level"})
            .with_columns(ion_popfrac=pl.col("g") * (-pl.col("energy_ev") / K_B / temperature).exp() / ltepartfunc)
            .with_columns(n_LTE=n_ion * pl.col("ion_popfrac"))
            .with_columns(n_NLTE=pl.col("n_LTE"))
        ).select(["level", "n_LTE", "n_NLTE", "ion_popfrac"])

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

        if not dftransitions.is_empty():
            dftransitions = dftransitions.join(
                dfpops_thision.select(pl.col("level").alias("lower"), pl.col("n_NLTE").alias("lower_pop")),
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

            for transition in dftransitions.iter_rows(named=True):
                epsilon_trans_ev = transition["epsilon_trans_ev"]
                xs_vec = pynonthermal.excitation.get_xs_excitation_vector(
                    self.engrid, transition, use_collstrengths=use_collstrengths
                )
                self.add_excitation(
                    Z,
                    ion_stage,
                    transition["lower_pop"],
                    xs_vec,
                    epsilon_trans_ev,
                    transitionkey=(transition["lower"], transition["upper"]),
                )

    def _add_ionisation_shell(self, n_ion: float, shell: dict[str, int | float]) -> None:
        # add the terms related to ionisation cross sections for one shell
        self._require_not_solved("add ionisation")
        deltaen = self.deltaen
        ionpot_ev = float(shell["ionpot_ev"])
        J = pynonthermal.collion.get_J(int(shell["Z"]), int(shell["ion_stage"]), ionpot_ev)
        npts = len(self.engrid)

        ar_xs_array = self._get_shell_xs(shell)

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
        # ionisation rates or spectra, but it was a source of error that let to energy fractions not adding up to 100%.

        epsilon_uppers = np.minimum((self.engrid + ionpot_ev) / 2, self.engrid)
        int_eps_uppers = np.arctan((epsilon_uppers - ionpot_ev) / J)

        # for the resulting arrays, use index j - i corresponding to energy endash - en
        epsilon_lowers1 = np.maximum(self.engrid - self.engrid[0], ionpot_ev)
        int_eps_lowers1 = np.arctan((epsilon_lowers1 - ionpot_ev) / J)

        for i, en in enumerate(self.engrid):
            # endash ranges from en to SF_EMAX, but skip over the zero-cross section points
            jstart = max(i, xsstartindex)

            # KF 92 limit
            # at each endash (columns j >= jstart), the integral in epsilon ranges from
            # epsilon_lower = max(endash - en, ionpot_ev)
            # epsilon_upper = min((endash + ionpot_ev) / 2, endash)]
            self.sfmatrix[i, jstart:] += np.where(
                epsilon_lowers1[jstart - i : npts - i] <= epsilon_uppers[jstart:],
                prefactors[jstart:] * (int_eps_uppers[jstart:] - int_eps_lowers1[jstart - i : npts - i]),
                0.0,
            )

            if 2 * en + ionpot_ev < self.engrid[-1] + deltaen:
                secondintegralstartindex = self.get_energyindex_lteq(float(2 * en + ionpot_ev))

                # endash ranges from 2 * en + ionpot_ev to SF_EMAX
                # at each endash, the integral in epsilon ranges from
                # epsilon_lower = en + ionpot_ev
                # epsilon_upper = min((endash + ionpot_ev) / 2, endash)]
                epsilon_lower2 = en + ionpot_ev
                int_eps_lower2 = math.atan((epsilon_lower2 - ionpot_ev) / J)
                self.sfmatrix[i, secondintegralstartindex:] -= np.where(
                    epsilon_lower2 <= epsilon_uppers[secondintegralstartindex:],
                    prefactors[secondintegralstartindex:]
                    * (int_eps_uppers[secondintegralstartindex:] - int_eps_lower2),
                    0.0,
                )

    def add_ionisation(self, Z: int, ion_stage: int, n_ion: float) -> None:
        self._require_not_solved("add ionisation")
        if (Z, ion_stage) in self._ionisation_ions:
            msg = f"Can't add Z={Z} ion_stage {ion_stage} twice"
            raise ValueError(msg)
        if n_ion == 0.0:
            return
        if n_ion < 0.0:
            msg = f"n_ion must be non-negative but is {n_ion}"
            raise ValueError(msg)

        collionrows = self._get_ion_collion_rows(Z, ion_stage)
        if not collionrows:
            msg = f"No ionisation cross-section data for Z={Z} ion_stage {ion_stage}"
            raise ValueError(msg)

        # Kozma & Fransson 1992 assume that every threshold lies above the low-energy cutoff E_0, so that
        # all energy reaching E_0 is thermalised by the free electrons. A shell below E_0 breaks that
        # assumption and can't be accounted for consistently: leaving it out of the matrix drops a real
        # energy sink, and including it credits ionisation below E_0 that the heating term already claims.
        ionpots_below_emin = sorted(
            float(shell["ionpot_ev"]) for shell in collionrows if shell["ionpot_ev"] < self.engrid[0]
        )
        if ionpots_below_emin:
            msg = (
                f"Z={Z} ion_stage {ion_stage} has {len(ionpots_below_emin)} ionisation shell(s) with an"
                f" ionisation potential below emin_ev={self.engrid[0]} eV (lowest {ionpots_below_emin[0]} eV)."
                f" The energy fractions would not sum to one, so set emin_ev <= {ionpots_below_emin[0]} eV."
            )
            raise ValueError(msg)

        if self.verbose:
            print(
                f"  including Z={Z} ion_stage"
                f" {ion_stage} ({at.get_ionstring(Z, ion_stage)}) ionisation with n_ion"
                f" {n_ion:.1e} [/cm3]"
            )
        self._register_ion_population(Z, ion_stage, n_ion)
        self._ionisation_ions.add((Z, ion_stage))

        for shell in collionrows:
            self._add_ionisation_shell(n_ion, shell)

    def calculate_free_electron_density(self) -> float:
        # number density of free electrons [cm-^3]
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

    def solve(self, depositionratedensity_ev: float, override_n_e: float | None = None) -> None:
        self._solved = False
        self.reset_solution_analysis()

        # every fraction and rate coefficient is divided by the deposition rate density. A zero gave a bare
        # ZeroDivisionError from inside the analysis, and a negative one silently flipped the sign of yvec
        # and of every rate coefficient while leaving the energy fractions summing to one.
        if depositionratedensity_ev <= 0.0:
            msg = f"depositionratedensity_ev must be greater than zero but is {depositionratedensity_ev}"
            raise ValueError(msg)

        self.depositionratedensity_ev = depositionratedensity_ev
        if override_n_e is not None and override_n_e <= 0.0:
            msg = f"override_n_e must be greater than zero but is {override_n_e}"
            raise ValueError(msg)

        # None clears any previously-set override, so that n_e is calculated on demand from ion populations
        self._n_e_override = override_n_e
        self._n_e = None

        npts = len(self.engrid)
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

        if self.verbose:
            n_ion_tot = self.get_n_ion_tot()
            x_e = n_e / n_ion_tot
            print(f" n_ion_tot: {n_ion_tot:.2e} [/cm3]        (total ion density)")
            print(f"       n_e: {n_e:.2e} [/cm3]        (free electron density)")
            print(f"       x_e: {x_e:.2e} [/cm3]        (electrons per nucleus)")
            print(f"deposition: {self.depositionratedensity_ev:7.2f}  [eV/s/cm3]")

        sfmatrix_with_electronloss = self.sfmatrix.copy()
        for i in range(npts):
            sfmatrix_with_electronloss[i, i] += electronlossfunction(self.engrid[i], n_e)

        # every process moves electrons to lower energies, so y(E) depends only on y at higher
        # energies (Kozma & Fransson 1992): only matrix columns j >= i are populated and the
        # matrix is upper triangular. K&F invert it with an unspecified "standard matrix
        # technique"; back-substitution from the highest energy downward (the scheme K&F credit
        # to Xu 1989) exploits the triangularity and is much faster than a general LU solve.
        yvec_reference = solve_upper_triangular(sfmatrix_with_electronloss, self.rhsvec)
        self.yvec = np.array(yvec_reference * self.depositionratedensity_ev / self.E_init_ev, dtype=np.float64)
        self._solved = True

    def calculate_nt_frac_excitation_ion(self, Z: int, ion_stage: int) -> float:
        if (Z, ion_stage) not in self.excitationlists:
            return 0.0

        # integral in Kozma & Fransson equation 9, but summed over all transitions for given ion
        deltaen = self.deltaen
        npts = len(self.engrid)

        xs_excitation_vec_sum_alltrans = np.zeros(npts)

        for (
            levelnumberdensity,
            xsvec,
            epsilon_trans_ev,
        ) in self.excitationlists[(Z, ion_stage)].values():
            xs_excitation_vec_sum_alltrans += levelnumberdensity * epsilon_trans_ev * xsvec

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
        # The integrand shared by both integrals of Kozma & Fransson equation 6: y(E') sigma(E') P(e_s, E')
        # over the primary energies arr_e_p. Both use E' as the variable of integration, since the first
        # one's epsilon differs from it only by a constant. Each caller supplies y and the cross section
        # however is cheapest for its own nodes: off the grid they have to be evaluated, but on it the
        # solution and the cached cross sections can be sliced directly.
        arr_psecondary = pynonthermal.collion.Psecondary_vec(e_p=arr_e_p, e_s=e_s, ionpot_ev=ionpot_ev, J=J)

        return float(np.trapezoid(arr_y * arr_xs * arr_psecondary, arr_e_p))

    def calculate_N_e(self, energy_ev: float) -> float:
        # Kozma & Fransson equation 6.
        # Something related to a number of electrons, needed to calculate the heating fraction in equation 3
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

            for levelnumberdensity, xsvec, epsilon_trans_ev in self.excitationlists.get((Z, ion_stage), {}).values():
                # Interpolate y and the cross section at energy_ev + epsilon_trans_ev, as the two
                # ionisation integrals below also do, rather than snapping down to the grid point under
                # it. Snapping lands below the transition energy whenever E has not yet carried the sum
                # past it, and get_xs_excitation_vector makes the cross section exactly zero there, so
                # the whole transition silently dropped out: on the helium benchmark at E = 0.25 eV that
                # was 280 of 296 transitions and 43% of N_e. The interpolation is written out rather
                # than left to np.interp because this runs once per transition per quadrature node,
                # where the call overhead of np.interp costs more than the arithmetic; the grid is
                # uniform, so the index and weight are exact.
                en_upper = energy_ev + epsilon_trans_ev
                if e_min <= en_upper <= e_max:
                    pos = (en_upper - e_min) / deltaen
                    i = min(int(pos), lastindex)
                    weight = pos - i
                    # the level population is absolute, so this term is not scaled by n_ion below
                    N_e += (
                        levelnumberdensity
                        * (self.yvec[i] + weight * (self.yvec[i + 1] - self.yvec[i]))
                        * (xsvec[i] + weight * (xsvec[i + 1] - xsvec[i]))
                    )

            if (Z, ion_stage) not in self._ionisation_ions:
                continue

            for shell in self._get_ion_collion_rows(Z, ion_stage):
                ionpot_ev = float(shell["ionpot_ev"])

                enlambda = min(e_max - energy_ev, energy_ev + ionpot_ev)
                J = pynonthermal.collion.get_J(int(shell["Z"]), int(shell["ion_stage"]), ionpot_ev)

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
                        arr_xs=pynonthermal.collion.get_arxs_array_shell(arr_e_p, shell),
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
                        arr_xs=pynonthermal.collion.get_arxs_array_shell(arr_e_p_low, shell),
                        e_s=energy_ev,
                        ionpot_ev=ionpot_ev,
                        J=J,
                    )
                    # the rest is smooth on the scale of J, so y and the cross section come straight
                    # from the solution and the cache rather than being re-evaluated
                    if stopindex < len(self.engrid) - 1:
                        N_e_ion += self._integrate_shell_secondaries(
                            arr_e_p=self.engrid[stopindex:],
                            arr_y=self.yvec[stopindex:],
                            arr_xs=self._get_shell_xs(shell)[stopindex:],
                            e_s=energy_ev,
                            ionpot_ev=ionpot_ev,
                            J=J,
                        )

            N_e += n_ion * N_e_ion

        # source term not here because it should be zero at the low end anyway

        return N_e

    def calculate_frac_heating(self) -> float:
        # Kozma & Fransson equation 8
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

        # E * N_e(E) is smooth over [0, E_0], so Simpson's rule earns its extra order here: it reaches
        # 5e-5 of the converged value at 9 nodes where the trapezoid rule needs several hundred, and
        # every node costs a full calculate_N_e(). Summing the nodes, as the code did before, counted
        # half a node too much at each end of the interval.
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
            # an ion added only for excitation has no ionisation shells in the matrix
            collionrows = self._get_ion_collion_rows(Z, ion_stage) if (Z, ion_stage) in self._ionisation_ions else []

            ionpot_valence = min((float(shell["ionpot_ev"]) for shell in collionrows), default=None)

            if self.verbose:
                valencestr = (
                    "no ionisation channel" if ionpot_valence is None else f"valence potential {ionpot_valence:.1f} eV"
                )
                print(f"\n====> Z={Z:2d} ion_stage {ion_stage} {at.get_ionstring(Z, ion_stage)} ({valencestr})")

                print(f"               n_ion: {n_ion:.2e} [/cm3]")
                print(f"     n_ion/n_ion_tot: {X_ion:.5f}")

            self._frac_ionisation_ion[(Z, ion_stage)] = 0.0
            eta_over_ionpot_sum = 0.0
            for shell in collionrows:
                ar_xs_array = self._get_shell_xs(shell)

                frac_ionisation_shell = (
                    n_ion
                    * shell["ionpot_ev"]
                    * np.dot(self.yvec, ar_xs_array)
                    * deltaen
                    / self.depositionratedensity_ev
                )

                if self.verbose:
                    if int(shell["n"]) < 0:
                        strsubshell = SUBSHELLNAMES[-int(shell["l"])]
                        shellname = f"Lotz shell {strsubshell}"
                    else:
                        shellname = f"n {int(shell['n']):d} l {int(shell['l']):d}"
                    print(
                        f"frac_ionisation_shell({shellname}):"
                        f" {frac_ionisation_shell:.4f} (ionpot"
                        f" {shell['ionpot_ev']:.2f} eV)"
                    )

                if not 0.0 <= frac_ionisation_shell <= 1.0:
                    warnings.warn(
                        f"invalid frac_ionisation_shell of {frac_ionisation_shell} included in the total",
                        stacklevel=2,
                    )

                self._frac_ionisation_ion[(Z, ion_stage)] += frac_ionisation_shell
                eta_over_ionpot_sum += frac_ionisation_shell / shell["ionpot_ev"]

            self._frac_ionisation_tot += self._frac_ionisation_ion[(Z, ion_stage)]

            eff_ionpot = float(X_ion / eta_over_ionpot_sum) if eta_over_ionpot_sum else float("inf")
            self._eff_ionpot[(Z, ion_stage)] = eff_ionpot

            # eff_ionpot_usevalence = (
            #     ionpot_valence * X_ion / self._frac_ionisation_ion[(Z, ion_stage)]
            #     if self._frac_ionisation_ion[(Z, ion_stage)] > 0. else float('inf'))

            if self.verbose:
                print(f"     frac_ionisation: {self._frac_ionisation_ion[(Z, ion_stage)]:.4f}")

            frac_excitation_thision = self.calculate_nt_frac_excitation_ion(Z, ion_stage)

            if not 0.0 <= frac_excitation_thision <= 1.0:
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

            self._nt_ionisation_ratecoeff[(Z, ion_stage)] = (
                self.depositionratedensity_ev / n_ion_tot / eff_ionpot if n_ion_tot > 0.0 else 0.0
            )
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
        # a channel is being double counted or omitted rather than merely under-resolved.
        if not math.isclose(frac_sum, 1.0, rel_tol=0.05):
            warnings.warn(
                f"the energy fractions sum to {frac_sum:.4f} instead of 1: heating {frac_heating:.4f},"
                f" ionisation {self._frac_ionisation_tot:.4f}, excitation {self._frac_excitation_tot:.4f}."
                f" Try a finer energy grid (npts is {len(self.engrid)}).",
                stacklevel=2,
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
        self._require_solved()
        if not self._analysed:
            self.analyse_ntspectrum()

        return self._eff_ionpot[(Z, ion_stage)]

    def get_ionisation_ratecoeff(self, Z: int, ion_stage: int) -> float:
        """Get the non-thermal ionisation rate coefficient in s^-1 for one ion.

        This scales with depositionratedensity_ev.
        """
        self._require_solved()
        if not self._analysed:
            self.analyse_ntspectrum()

        return self._nt_ionisation_ratecoeff[(Z, ion_stage)]

    def get_excitation_ratecoeff(self, Z: int, ion_stage: int, transitionkey: t.Any) -> float:
        """Get the non-thermal excitation rate coefficient in s^-1 for one transition.

        This is the integral of y(E) * sigma(E) dE in Kozma & Fransson equation 9, matching the
        convention of get_ionisation_ratecoeff(). It scales with depositionratedensity_ev.
        """
        self._require_solved()
        _levelnumberdensity, xsvec, _epsilon_trans_ev = self.excitationlists[(Z, ion_stage)][transitionkey]

        return float(np.dot(xsvec, self.yvec) * self.deltaen)

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
            for (
                levelnumberdensity,
                xsvec,
                epsilon_trans_ev,
            ) in self.excitationlists[(Z, ion_stage)].values():
                part_integrand += levelnumberdensity * epsilon_trans_ev * xsvec / self.depositionratedensity_ev

        return self.yvec * part_integrand

    def get_d_etaion_by_d_en_vec(self) -> npt.NDArray[np.float64]:
        self._require_solved()
        part_integrand = np.zeros(len(self.engrid))

        for Z, ion_stage in self._ionisation_ions:
            n_ion = self.ionpopdict[(Z, ion_stage)]

            for shell in self._get_ion_collion_rows(Z, ion_stage):
                xsvec = self._get_shell_xs(shell)

                part_integrand += n_ion * shell["ionpot_ev"] * xsvec / self.depositionratedensity_ev

        return self.yvec * part_integrand

    def plot_yspectrum(
        self,
        en_y_on_d_en: bool = False,
        xscalelog: bool = False,
        outputfilename: Path | str | None = None,
        axis: mplax.Axes | None = None,
    ) -> None:
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
            ax.set_ylabel(r"log d(E y)/dE", fontsize=fs)
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
        #     # delta_E_y_on_dE[i] = ((yvec[i + 1] * engrid[i + 1]) - (yvec[i] * engrid[i])) / (engrid[i + 1] - engrid[i])
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

    def plot_spec_channels(self, outputfilename: Path | str | None, xscalelog: bool = False) -> None:
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
