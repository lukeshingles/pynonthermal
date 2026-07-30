from __future__ import annotations

import math
import typing as t
import warnings
from math import atan
from pathlib import Path

import artistools as at
import matplotlib.axes as mplax
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
from scipy import linalg

import pynonthermal
from pynonthermal.axelrod import get_workfn_ev
from pynonthermal.base import electronlossfunction
from pynonthermal.base import get_Zbar
from pynonthermal.constants import K_B

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
        emin_ev: float = 1,
        emax_ev: float = 3000,
        npts: int = 4000,
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
        # every ion with an ionisation or an excitation channel, in the order it was added
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

        # transitions outside the grid can't be driven by any electron the solver represents
        lzdftransitions = lzdftransitions.filter(pl.col("epsilon_trans_ev").is_between(self.engrid[0], self.engrid[-1]))
        dftransitions = lzdftransitions.collect()

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
                int_eps_lower2 = atan((epsilon_lower2 - ionpot_ev) / J)
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
            reason = "no ions have been added" if not self.ionpopdict else "every added ion is neutral"
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

        # every process moves electrons to lower energies, so only matrix columns j >= i are
        # populated and the matrix is upper triangular. Solving by back-substitution
        # (as in Kozma & Fransson 1992) is much faster than a general LU decomposition.
        yvec_reference = np.array(
            linalg.solve_triangular(sfmatrix_with_electronloss, self.rhsvec, lower=False),
            dtype=np.float64,
        )
        self.yvec = np.array(yvec_reference * self.depositionratedensity_ev / self.E_init_ev, dtype=np.float64)
        self._solved = True

    def calculate_nt_frac_excitation_ion(self, Z: int, ion_stage: int) -> float:
        if (Z, ion_stage) not in self.excitationlists:
            return 0.0

        # integral in Kozma & Fransson equation 9, but summed over all transitions for given ion
        deltaen = self.engrid[1] - self.engrid[0]
        npts = len(self.engrid)

        xs_excitation_vec_sum_alltrans = np.zeros(npts)

        for (
            levelnumberdensity,
            xsvec,
            epsilon_trans_ev,
        ) in self.excitationlists[(Z, ion_stage)].values():
            xs_excitation_vec_sum_alltrans += levelnumberdensity * epsilon_trans_ev * xsvec

        return np.dot(xs_excitation_vec_sum_alltrans, self.yvec) * deltaen / self.depositionratedensity_ev

    def calculate_N_e(self, energy_ev: float) -> float:
        # Kozma & Fransson equation 6.
        # Something related to a number of electrons, needed to calculate the heating fraction in equation 3
        # not valid for energy > E_0
        if energy_ev == 0.0:
            return 0.0

        N_e = 0.0

        deltaen = self.engrid[1] - self.engrid[0]

        for Z, ion_stage in self._get_all_ions():
            N_e_ion = 0.0
            n_ion = self.ionpopdict.get((Z, ion_stage), 0.0)

            for levelnumberdensity, xsvec, epsilon_trans_ev in self.excitationlists.get((Z, ion_stage), {}).values():
                if energy_ev + epsilon_trans_ev >= self.engrid[0]:
                    i = self.get_energyindex_lteq(en_ev=energy_ev + epsilon_trans_ev)
                    # the level population is absolute, so this term is not scaled by n_ion below
                    N_e += levelnumberdensity * self.yvec[i] * xsvec[i]
                    # enbelow = engrid[i]
                    # enabove = engrid[i + 1]
                    # x = (energy_ev - enbelow) / (enabove - enbelow)
                    # yvecinterp = (1 - x) * yvec[i] + x * yvec[i + 1]
                    # N_e += levelnumberdensity * yvecinterp * get_xs_excitation(energy_ev + epsilon_trans_ev, row)

            if (Z, ion_stage) not in self._ionisation_ions:
                continue

            for shell in self._get_ion_collion_rows(Z, ion_stage):
                ionpot_ev = shell["ionpot_ev"]

                enlambda = min(self.engrid[-1] - energy_ev, energy_ev + ionpot_ev)
                J = pynonthermal.collion.get_J(int(shell["Z"]), int(shell["ion_stage"]), ionpot_ev)

                ar_xs_array = self._get_shell_xs(shell)

                # integral from ionpot to enlambda
                integral1startindex = self.get_energyindex_lteq(en_ev=ionpot_ev)
                integral2stopindex = self.get_energyindex_lteq(en_ev=enlambda)

                for j in range(integral1startindex, integral2stopindex + 1):
                    endash = self.engrid[j]
                    k = self.get_energyindex_lteq(en_ev=energy_ev + endash)
                    N_e_ion += (
                        deltaen
                        * self.yvec[k]
                        * ar_xs_array[k]
                        * pynonthermal.collion.Psecondary(e_p=self.engrid[k], epsilon=endash, ionpot_ev=ionpot_ev, J=J)
                    )

                # integral from 2E + I up to E_max (skipped when the domain is empty, since
                # get_energyindex_lteq would clamp to the last bin and add a spurious term;
                # same guard as in _add_ionisation_shell)
                if 2 * energy_ev + ionpot_ev < self.engrid[-1] + deltaen:
                    integral2startindex = self.get_energyindex_lteq(en_ev=2 * energy_ev + ionpot_ev)
                    N_e_ion += deltaen * sum(
                        self.yvec[j]
                        * ar_xs_array[j]
                        * pynonthermal.collion.Psecondary(
                            e_p=self.engrid[j],
                            epsilon=energy_ev + ionpot_ev,
                            ionpot_ev=ionpot_ev,
                            J=J,
                        )
                        for j in range(integral2startindex, len(self.engrid))
                    )

            N_e += n_ion * N_e_ion

        # source term not here because it should be zero at the low end anyway

        return N_e

    def calculate_frac_heating(self) -> float:
        # Kozma & Fransson equation 8
        frac_heating = 0.0
        E_0 = self.engrid[0]
        n_e = self.get_n_e()

        deltaen = self.engrid[1] - self.engrid[0]
        frac_heating += (
            deltaen
            / self.depositionratedensity_ev
            * sum(electronlossfunction(float(en_ev), n_e) * self.yvec[i] for i, en_ev in enumerate(self.engrid))
        )

        frac_heating_E_0_part = E_0 * self.yvec[0] * electronlossfunction(E_0, n_e) / self.depositionratedensity_ev

        frac_heating += frac_heating_E_0_part

        # if self.verbose:
        #     print(f"            frac_heating E_0 * y * l(E_0) part: {frac_heating_E_0_part:.5f}")

        frac_heating_N_e: float = 0.0
        npts_integral = math.ceil(E_0 / deltaen) * 5
        # if self.verbose:
        #     print(f'N_e npts_integral: {npts_integral}')
        arr_en, deltaen2 = np.linspace(0.0, E_0, num=npts_integral, retstep=True, endpoint=True, dtype=np.float64)
        arr_en_N_e = np.array([en_ev * self.calculate_N_e(en_ev) for en_ev in arr_en], dtype=np.float64)
        frac_heating_N_e += float(1.0 / self.depositionratedensity_ev * arr_en_N_e.sum() * deltaen2)

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

        deltaen = self.engrid[1] - self.engrid[0]

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

                if frac_ionisation_shell > 1:
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

            if frac_excitation_thision > 1:
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
        self._require_solved()
        n_e_nt = 0.0
        for i, en in enumerate(self.engrid):
            # oneovervelocity = np.sqrt(9.10938e-31 / 2 / en / 1.60218e-19) / 100.
            velocity = np.sqrt(2 * en * 1.60218e-19 / 9.10938e-31) * 100.0  # cm/s
            n_e_nt += self.yvec[i] / velocity * self.deltaen

        return n_e_nt

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

        # the plot extends below E_0, where the cross sections are all zero and every remaining
        # electron thermalises, so only the heating channel is non-zero there
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

        if self.get_frac_excitation_tot() > 0.0:
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
