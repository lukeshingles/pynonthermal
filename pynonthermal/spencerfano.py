from __future__ import annotations

import math
import typing as t
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
    _frac_heating: float
    _frac_ionisation_tot: float
    _frac_excitation_tot: float
    _frac_ionisation_ion: dict[tuple[int, int], float]
    _frac_excitation_ion: dict[tuple[int, int], float]
    _nt_ionisation_ratecoeff: dict[tuple[int, int], float]
    ionpopdict: dict[tuple[int, int], float]
    excitationlists: dict[tuple[int, int], dict[t.Any, tuple[float, npt.NDArray[np.float64], float]]]
    verbose: bool
    _n_e: float
    engrid: npt.NDArray[np.float64]
    deltaen: npt.NDArray[np.float64]
    dfcollion: pl.DataFrame
    sourcevec: npt.NDArray[np.float64]
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
    ):
        self._solved = False
        self.ionpopdict = {}  # key is (Z, ion_stage) value is number density

        # key is (Z, ion_stage) value is {levelkey : (levelnumberdensity, xs_vec, epsilon_trans_ev)}
        self.excitationlists = {}

        self.verbose = verbose
        self.engrid = np.linspace(emin_ev, emax_ev, num=npts, endpoint=True, dtype=float)
        self.deltaen = self.engrid[1] - self.engrid[0]

        self.dfcollion = pynonthermal.collion.read_colliondata(
            collionfilename=("collion-AR1985.txt" if use_ar1985 else "collion.txt")
        )

        self.sourcevec = np.zeros(self.engrid.shape)
        # source_spread_pts = math.ceil(npts / 10.)
        source_spread_pts = math.ceil(npts * 0.1)
        for s in range(npts):
            # spread the source over some energy width
            if s < npts - source_spread_pts:
                self.sourcevec[s] = 0.0
            elif s < npts:
                self.sourcevec[s] = 1.0 / (self.deltaen * source_spread_pts)
        # self.sourcevec[-1] = 1.

        source_emin = self.engrid[np.flatnonzero(self.sourcevec)[0]]
        source_emax = self.engrid[np.flatnonzero(self.sourcevec)[-1]]

        # E_init_ev is the deposition rate density that we assume when solving the SF equation.
        # The solution will be scaled to the true deposition rate later
        self.E_init_ev = np.dot(self.engrid, self.sourcevec) * self.deltaen

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
        assert not self._solved
        assert len(xs_vec) == len(self.engrid)

        if (Z, ion_stage) not in self.excitationlists:
            self.excitationlists[(Z, ion_stage)] = {}

        if transitionkey is None:
            transitionkey = len(self.excitationlists[(Z, ion_stage)])  # simple number index

        assert transitionkey not in self.excitationlists[(Z, ion_stage)]
        self.excitationlists[(Z, ion_stage)][transitionkey] = (
            levelnumberdensity,
            xs_vec,
            epsilon_trans_ev,
        )
        vec_xs_excitation_levelnumberdensity_deltae = levelnumberdensity * self.deltaen * xs_vec
        xsstartindex = self.get_energyindex_lteq(en_ev=epsilon_trans_ev)

        for i, en in enumerate(self.engrid):
            stopindex = self.get_energyindex_lteq(en_ev=en + epsilon_trans_ev)

            startindex = max(i, xsstartindex)
            # for j in range(startindex, stopindex):
            self.sfmatrix[i, startindex:stopindex] += vec_xs_excitation_levelnumberdensity_deltae[startindex:stopindex]

            # do the last bit separately because we're not using the full deltaen interval

            delta_en_actual = en + epsilon_trans_ev - self.engrid[stopindex]
            self.sfmatrix[i, stopindex] += (
                vec_xs_excitation_levelnumberdensity_deltae[stopindex] * delta_en_actual / self.deltaen
            )

    def add_ion_ltepopexcitation(
        self, Z: int, ion_stage: int, n_ion: float, temperature: float = 3000, adata_polars: pl.DataFrame | None = None
    ) -> None:
        if adata_polars is not None:
            self.adata_polars = adata_polars

        if self.adata_polars is None:
            # use ARTIS atomic data read by the artistools package to get the levels
            self.adata_polars = at.atomic.get_levels_polars(
                Path(pynonthermal.DATADIR, "artis_files"),
                get_transitions=True,
                derived_transitions_columns=["epsilon_trans_ev", "lambda_angstroms", "lower_g", "upper_g"],
            )

        assert self.adata_polars is not None

        ion = self.adata_polars.filter(pl.col("Z") == Z).filter(pl.col("ion_stage") == ion_stage)
        if ion.is_empty():
            msg = f"ERROR: No excitation data for Z={Z} ion_stage {ion_stage} in internal database."
            raise AssertionError(msg)

        dfpops_thision = ion["levels"].item()

        ltepartfunc = dfpops_thision.select(pl.col("g") * (-pl.col("energy_ev") / K_B / temperature).exp()).sum().item()
        dfpops_thision = (
            dfpops_thision.rename({"levelindex": "level"})
            .with_columns(ion_popfrac=pl.col("g") * (-pl.col("energy_ev") / K_B / temperature).exp() / ltepartfunc)
            .with_columns(n_LTE=n_ion * pl.col("ion_popfrac"))
            .with_columns(n_NLTE=pl.col("n_LTE"))
        ).select(["level", "n_LTE", "n_NLTE", "ion_popfrac"])

        lzdftransitions = ion["transitions"].item().filter((pl.col("collstr") >= 0).or_(pl.col("forbidden") == 0))

        maxnlevelslower = None
        maxnlevelsupper = None

        # find the highest ground multiplet level
        # groundlevelnoj = ion.levels.iloc[0].levelname.split('[')[0]
        # maxnlevelslower = ion.levels[ion.levels.levelname.str.startswith(groundlevelnoj)].index.max()

        # match ARTIS defaults
        maxnlevelslower = 5
        maxnlevelsupper = 250

        if maxnlevelslower is not None:
            lzdftransitions = lzdftransitions.filter(pl.col("lower") < maxnlevelslower)
        if maxnlevelsupper is not None:
            lzdftransitions = lzdftransitions.filter(pl.col("upper") < maxnlevelsupper)

        lzdftransitions = lzdftransitions.filter(pl.col("epsilon_trans_ev") >= self.engrid[0])
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
                if epsilon_trans_ev >= self.engrid[0]:
                    xs_vec = pynonthermal.excitation.get_xs_excitation_vector(self.engrid, transition)
                    self.add_excitation(
                        Z,
                        ion_stage,
                        transition["lower_pop"],
                        xs_vec,
                        epsilon_trans_ev,
                        transitionkey=(transition["lower"], transition["upper"]),
                    )

    def _add_ionisation_shell(self, n_ion: float, shell: dict[str, int | float]) -> None:
        assert not self._solved
        # this code has been optimised and is now an almost unreadable form, but it contains the terms
        # related to ionisation cross sections
        deltaen = self.engrid[1] - self.engrid[0]
        ionpot_ev = shell["ionpot_ev"]
        J = pynonthermal.collion.get_J(int(shell["Z"]), int(shell["ion_stage"]), ionpot_ev)
        npts = len(self.engrid)

        ar_xs_array = pynonthermal.collion.get_arxs_array_shell(self.engrid, shell)

        if ionpot_ev <= self.engrid[0]:
            xsstartindex = 0
        else:
            xsstartindex = next((i for i in range(npts) if ar_xs_array[i] > 0.0), npts + 1)
            xsstartindex = self.get_energyindex_gteq(en_ev=ionpot_ev)

        # J * atan[(epsilon - ionpot_ev) / J] is the indefinite integral of
        # 1/(1 + (epsilon - ionpot_ev)^2/ J^2) d_epsilon
        # in Kozma & Fransson 1992 equation 4

        prefactors = [
            n_ion * ar_xs_array[j] / atan((self.engrid[j] - ionpot_ev) / 2.0 / J) * deltaen for j in range(npts)
        ]

        # Luke Shingles: the use of min and max on the epsilon limits keeps energies
        # from becoming unphysical. This insight came from reading the
        # CMFGEN Fortran source code (Li, Dessart, Hillier 2012, doi:10.1111/j.1365-2966.2012.21198.x)
        # I had neglected this, so the limits of integration were incorrect. The fix didn't massively affect
        # ionisation rates or spectra, but it was a source of error that let to energy fractions not adding up to 100%.

        epsilon_uppers = [min((self.engrid[j] + ionpot_ev) / 2, self.engrid[j]) for j in range(npts)]
        int_eps_uppers = [atan((epsilon_upper - ionpot_ev) / J) for epsilon_upper in epsilon_uppers]

        # for the resulting arrays, use index j - i corresponding to energy endash - en
        epsilon_lowers1 = [max(self.engrid[j] - self.engrid[0], ionpot_ev) for j in range(npts)]
        int_eps_lowers1 = [atan((epsilon_lower - ionpot_ev) / J) for epsilon_lower in epsilon_lowers1]

        for i, en in enumerate(self.engrid):
            # endash ranges from en to SF_EMAX, but skip over the zero-cross section points
            jstart = max(i, xsstartindex)

            # KF 92 limit
            # at each endash, the integral in epsilon ranges from
            # epsilon_lower = max(endash - en, ionpot_ev)
            # epsilon_upper = min((endash + ionpot_ev) / 2, endash)]

            # integral/J of 1/[1 + (epsilon - ionpot_ev) / J] for epsilon = en + ionpot_ev
            for j in range(jstart, npts):
                # j is the matrix column index which corresponds to the piece of the
                # integral at y(E') where E' >= E and E' = engrid[j]

                if epsilon_lowers1[j - i] <= epsilon_uppers[j]:
                    self.sfmatrix[i, j] += prefactors[j] * (int_eps_uppers[j] - int_eps_lowers1[j - i])

            if 2 * en + ionpot_ev < self.engrid[-1] + (self.engrid[1] - self.engrid[0]):
                secondintegralstartindex = self.get_energyindex_lteq(2 * en + ionpot_ev)
            else:
                secondintegralstartindex = npts + 1

            # endash ranges from 2 * en + ionpot_ev to SF_EMAX
            # at each endash, the integral in epsilon ranges from
            # epsilon_lower = en + ionpot_ev
            # epsilon_upper = min((endash + ionpot_ev) / 2, endash)]
            epsilon_lower2 = en + ionpot_ev
            for j in range(secondintegralstartindex, npts):
                if epsilon_lower2 <= epsilon_uppers[j]:
                    int_eps_lower2 = atan((epsilon_lower2 - ionpot_ev) / J)
                    self.sfmatrix[i, j] -= prefactors[j] * (int_eps_uppers[j] - int_eps_lower2)

    def add_ionisation(self, Z: int, ion_stage: int, n_ion: float) -> None:
        assert not self._solved
        assert (Z, ion_stage) not in self.ionpopdict  # can't add same ion twice
        if n_ion == 0.0:
            return

        if self.verbose:
            print(
                f"  including Z={Z} ion_stage"
                f" {ion_stage} ({at.get_ionstring(Z, ion_stage)}) ionisation with n_ion"
                f" {n_ion:.1e} [/cm3]"
            )
        assert n_ion > 0.0
        self.ionpopdict[(Z, ion_stage)] = n_ion
        dfcollion_thision = self.dfcollion.filter(pl.col("Z") == Z).filter(pl.col("ion_stage") == ion_stage)

        for shell in dfcollion_thision.iter_rows(named=True):
            if shell["ionpot_ev"] >= self.engrid[0]:
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
        if not hasattr(self, "_n_e"):
            self._n_e = self.calculate_free_electron_density()

        return self._n_e

    def get_n_ion_tot(self) -> float:
        # total number density of all nuclei [cm^-3]
        n_ion_tot = 0.0
        for Z, ion_stage in self.ionpopdict:
            n_ion_tot += self.ionpopdict[(Z, ion_stage)]
        return n_ion_tot

    def solve(self, depositionratedensity_ev: float, override_n_e: float | None = None) -> None:
        assert not self._solved

        self.depositionratedensity_ev = depositionratedensity_ev
        if override_n_e is not None:
            self._n_e = override_n_e
            # else it will be calculated on demand from ion populations

        npts = len(self.engrid)
        n_e = self.get_n_e()

        if self.verbose:
            n_ion_tot = self.get_n_ion_tot()
            x_e = n_e / n_ion_tot
            print(f" n_ion_tot: {n_ion_tot:.2e} [/cm3]        (total ion density)")
            print(f"       n_e: {n_e:.2e} [/cm3]        (free electron density)")
            print(f"       x_e: {x_e:.2e} [/cm3]        (electrons per nucleus)")
            print(f"deposition: {self.depositionratedensity_ev:7.2f}  [eV/s/cm3]")

        deltaen = self.engrid[1] - self.engrid[0]
        npts = len(self.engrid)

        constvec = np.zeros(npts)
        for i in range(npts):
            for j in range(i, npts):
                constvec[i] += self.sourcevec[j] * deltaen

        for i in range(npts):
            en = self.engrid[i]
            self.sfmatrix[i, i] += electronlossfunction(en, n_e)

        yvec_reference = np.array(
            linalg.lu_solve(linalg.lu_factor(self.sfmatrix, overwrite_a=False), constvec, trans=0), dtype=np.float64
        )
        self.yvec = np.array(yvec_reference * self.depositionratedensity_ev / self.E_init_ev, dtype=np.float64)
        self._solved = True
        del self.sfmatrix  # this can take up a lot of memory

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

        for Z, ion_stage in self.ionpopdict:
            N_e_ion = 0.0
            n_ion = self.ionpopdict[(Z, ion_stage)]

            if self.excitationlists and (Z, ion_stage) in self.excitationlists:
                for levelnumberdensity, xsvec, epsilon_trans_ev in self.excitationlists[(Z, ion_stage)].values():
                    if energy_ev + epsilon_trans_ev >= self.engrid[0]:
                        i = self.get_energyindex_lteq(en_ev=energy_ev + epsilon_trans_ev)
                        N_e_ion += (levelnumberdensity / n_ion) * self.yvec[i] * xsvec[i]
                        # enbelow = engrid[i]
                        # enabove = engrid[i + 1]
                        # x = (energy_ev - enbelow) / (enabove - enbelow)
                        # yvecinterp = (1 - x) * yvec[i] + x * yvec[i + 1]
                        # N_e_ion += (levelnumberdensity / n_ion) * yvecinterp * get_xs_excitation(energy_ev + epsilon_trans_ev, row)

            dfcollion_thision = self.dfcollion.filter(pl.col("Z") == Z).filter(pl.col("ion_stage") == ion_stage)

            for shell in dfcollion_thision.iter_rows(named=True):
                ionpot_ev = shell["ionpot_ev"]

                enlambda = min(self.engrid[-1] - energy_ev, energy_ev + ionpot_ev)
                J = pynonthermal.collion.get_J(int(shell["Z"]), int(shell["ion_stage"]), ionpot_ev)

                ar_xs_array = pynonthermal.collion.get_arxs_array_shell(self.engrid, shell)

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

                # integral from 2E + I up to E_max
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
        self._frac_heating = 0.0
        E_0 = self.engrid[0]
        n_e = self.get_n_e()

        deltaen = self.engrid[1] - self.engrid[0]
        self._frac_heating += (
            deltaen
            / self.depositionratedensity_ev
            * sum(electronlossfunction(en_ev, n_e) * self.yvec[i] for i, en_ev in enumerate(self.engrid))
        )

        frac_heating_E_0_part = E_0 * self.yvec[0] * electronlossfunction(E_0, n_e) / self.depositionratedensity_ev

        self._frac_heating += frac_heating_E_0_part

        # if self.verbose:
        #     print(f"            frac_heating E_0 * y * l(E_0) part: {frac_heating_E_0_part:.5f}")

        frac_heating_N_e: float = 0.0
        npts_integral = math.ceil(E_0 / deltaen) * 5
        # if self.verbose:
        #     print(f'N_e npts_integral: {npts_integral}')
        arr_en, deltaen2 = np.linspace(0.0, E_0, num=npts_integral, retstep=True, endpoint=True, dtype=np.float64)
        arr_en_N_e = np.array([en_ev * self.calculate_N_e(en_ev) for en_ev in arr_en], dtype=np.float64)
        frac_heating_N_e += float(1.0 / self.depositionratedensity_ev * sum(arr_en_N_e) * deltaen2)

        if self.verbose:
            print(f" frac_heating(E<EMIN): {frac_heating_N_e:.5f}")

        self._frac_heating += frac_heating_N_e

        return self._frac_heating

    def analyse_ntspectrum(self) -> None:
        assert self._solved

        deltaen = self.engrid[1] - self.engrid[0]

        self._frac_ionisation_tot = 0.0
        self._frac_excitation_tot = 0.0
        self._frac_ionisation_ion = {}
        self._frac_excitation_ion = {}
        self._nt_ionisation_ratecoeff = {}

        if self.verbose:
            print(f"    n_e_nt: {self.get_n_e_nt():.2e} [/cm3]")

        for (Z, ion_stage), n_ion in self.ionpopdict.items():
            n_ion_tot = self.get_n_ion_tot()
            X_ion = n_ion / n_ion_tot
            dfcollion_thision = self.dfcollion.filter(pl.col("Z") == Z).filter(pl.col("ion_stage") == ion_stage)
            # if dfcollion.empty:
            #     continue
            ionpot_valence = dfcollion_thision["ionpot_ev"].min()
            assert isinstance(ionpot_valence, float)

            if self.verbose:
                print(
                    f"\n====> Z={Z:2d} ion_stage"
                    f" {ion_stage} {at.get_ionstring(Z, ion_stage)} (valence potential"
                    f" {ionpot_valence:.1f} eV)"
                )

                print(f"               n_ion: {n_ion:.2e} [/cm3]")
                print(f"     n_ion/n_ion_tot: {X_ion:.5f}")

            self._frac_ionisation_ion[(Z, ion_stage)] = 0.0
            integralgamma = 0.0
            eta_over_ionpot_sum = 0.0
            for shell in dfcollion_thision.iter_rows(named=True):
                ar_xs_array = pynonthermal.collion.get_arxs_array_shell(self.engrid, shell)

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

                integralgamma += np.dot(self.yvec, ar_xs_array) * deltaen

                if frac_ionisation_shell > 1:
                    print(f"WARNING: Ignoring invalid frac_ionisation_shell of {frac_ionisation_shell}.")
                    # frac_ionisation_shell = 0.0

                self._frac_ionisation_ion[(Z, ion_stage)] += frac_ionisation_shell
                eta_over_ionpot_sum += frac_ionisation_shell / shell["ionpot_ev"]

            self._frac_ionisation_tot += self._frac_ionisation_ion[(Z, ion_stage)]

            eff_ionpot = X_ion / eta_over_ionpot_sum if eta_over_ionpot_sum else float("inf")

            # eff_ionpot_usevalence = (
            #     ionpot_valence * X_ion / self._frac_ionisation_ion[(Z, ion_stage)]
            #     if self._frac_ionisation_ion[(Z, ion_stage)] > 0. else float('inf'))

            if self.verbose:
                print(f"     frac_ionisation: {self._frac_ionisation_ion[(Z, ion_stage)]:.4f}")

            if self.excitationlists:
                if n_ion > 0.0:
                    self._frac_excitation_ion[(Z, ion_stage)] = self.calculate_nt_frac_excitation_ion(Z, ion_stage)
                else:
                    self._frac_excitation_ion[(Z, ion_stage)] = 0.0

                if self._frac_excitation_ion[(Z, ion_stage)] > 1:
                    self._frac_excitation_ion[(Z, ion_stage)] = 0.0
                    print(
                        f"WARNING: Ignoring invalid frac_excitation_ion of {self._frac_excitation_ion[(Z, ion_stage)]}."
                    )

                self._frac_excitation_tot += self._frac_excitation_ion[(Z, ion_stage)]

                if self.verbose:
                    print(f"     frac_excitation: {self._frac_excitation_ion[(Z, ion_stage)]:.4f}")
            else:
                self._frac_excitation_ion[(Z, ion_stage)] = 0.0

            self._nt_ionisation_ratecoeff[(Z, ion_stage)] = self.depositionratedensity_ev / n_ion_tot / eff_ionpot
            if self.verbose:
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

                # complicated eff_ionpot thing should match a simple integral of xs * vec * dE
                # print(f"ionisation ratecoeff: {integralgamma:.2e} [/s]")
                assert np.isclose(
                    self._nt_ionisation_ratecoeff[(Z, ion_stage)],
                    integralgamma,
                    rtol=0.01,
                )

        # n_e_nt = get_n_e_nt(engrid, yvec)
        # print(f'               n_e_nt: {n_e_nt:.2e} /s/cm3')

        if self.verbose:
            print()
            print(f"  frac_excitation_tot: {self._frac_excitation_tot:.4f}")
            print(f"  frac_ionisation_tot: {self._frac_ionisation_tot:.4f}")

        self.calculate_frac_heating()
        frac_heating = self.get_frac_heating()

        if self.verbose:
            print(f"         frac_heating: {frac_heating:.4f}")
            print(f"             frac_sum: {self._frac_excitation_tot + self._frac_ionisation_tot + frac_heating:.4f}")

    def get_n_e_nt(self) -> float:
        assert self._solved
        n_e_nt = 0.0
        for i, en in enumerate(self.engrid):
            # oneovervelocity = np.sqrt(9.10938e-31 / 2 / en / 1.60218e-19) / 100.
            velocity = np.sqrt(2 * en * 1.60218e-19 / 9.10938e-31) * 100.0  # cm/s
            n_e_nt += self.yvec[i] / velocity * self.deltaen

        return n_e_nt

    def get_frac_heating(self) -> float:
        assert self._solved
        if not hasattr(self, "_frac_heating"):
            self.calculate_frac_heating()

        return self._frac_heating

    def get_frac_excitation_tot(self) -> float:
        assert self._solved
        if not hasattr(self, "_frac_excitation_tot"):
            self.analyse_ntspectrum()

        return self._frac_excitation_tot

    def get_frac_ionisation_tot(self) -> float:
        assert self._solved
        if not hasattr(self, "_frac_ionisation_tot"):
            self.analyse_ntspectrum()

        return self._frac_ionisation_tot

    def get_frac_ionisation_ion(self, Z: int, ion_stage: int) -> float:
        assert self._solved
        if not hasattr(self, "_frac_excitation_ion"):
            self.analyse_ntspectrum()

        return self._frac_excitation_ion[(Z, ion_stage)]

    def get_ionisation_ratecoeff(self, Z: int, ion_stage: int) -> float:
        assert self._solved
        return self._nt_ionisation_ratecoeff[(Z, ion_stage)]

    def get_excitation_ratecoeff(self, Z: int, ion_stage: int, transitionkey: t.Any) -> float:
        # integral in Kozma & Fransson equation 9
        levelnumberdensity, xsvec, epsilon_trans_ev = self.excitationlists[(Z, ion_stage)][transitionkey]

        return np.dot(xsvec, self.yvec) * self.deltaen / self.depositionratedensity_ev

    def get_frac_sum(self) -> float:
        return self.get_frac_heating() + self.get_frac_excitation_tot() + self.get_frac_ionisation_tot()

    def get_d_etaheating_by_d_en_vec(self) -> list[float]:
        assert self._solved
        return [
            self.electronlossfunction(self.engrid[i]) * self.yvec[i] / self.depositionratedensity_ev
            for i in range(len(self.engrid))
        ]

    def get_d_etaexcitation_by_d_en_vec(self) -> npt.NDArray[np.float64]:
        assert self._solved
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
        assert self._solved
        part_integrand = np.zeros(len(self.engrid))

        for Z, ion_stage in self.ionpopdict:
            n_ion = self.ionpopdict[(Z, ion_stage)]
            dfcollion_thision = self.dfcollion.filter(pl.col("Z") == Z).filter(pl.col("ion_stage") == ion_stage)

            for shell in dfcollion_thision.iter_rows(named=True):
                xsvec = pynonthermal.collion.get_arxs_array_shell(self.engrid, shell)

                part_integrand += n_ion * shell["ionpot_ev"] * xsvec / self.depositionratedensity_ev

        return self.yvec * part_integrand

    def plot_yspectrum(
        self,
        en_y_on_d_en: bool = False,
        xscalelog: bool = False,
        outputfilename: Path | str | None = None,
        axis: mplax.Axes | None = None,
    ) -> None:
        assert self._solved
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
        assert self._solved
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

        npts = len(self.engrid)
        E_0 = self.engrid[0]

        # E_init_ev = np.dot(engrid, sourcevec) * deltaen
        # d_etasource_by_d_en_vec = engrid * sourcevec / E_init_ev
        # axes[0].plot(engrid[1:], d_etasource_by_d_en_vec[1:], marker="None", lw=1.5, color='blue', label='Source')

        d_etaion_by_d_en_vec = self.get_d_etaion_by_d_en_vec()

        d_etaexc_by_d_en_vec = self.get_d_etaexcitation_by_d_en_vec()

        d_etaheat_by_d_en_vec = self.get_d_etaheating_by_d_en_vec()

        deltaen = self.engrid[1:] - self.engrid[:-1]
        etaion_int = np.zeros(npts)
        etaexc_int = np.zeros(npts)
        etaheat_int = np.zeros(npts)
        for i in reversed(range(len(self.engrid) - 1)):
            etaion_int[i] = etaion_int[i + 1] + d_etaion_by_d_en_vec[i] * deltaen[i]
            etaexc_int[i] = etaexc_int[i + 1] + d_etaexc_by_d_en_vec[i] * deltaen[i]
            etaheat_int[i] = etaheat_int[i + 1] + d_etaheat_by_d_en_vec[i] * deltaen[i]

        etaheat_int[0] += E_0 * self.yvec[0] * self.electronlossfunction(E_0)

        # etatot_int = etaion_int + etaexc_int + etaheat_int

        # go below E_0
        deltaen2 = E_0 / 20.0
        engrid_low = np.arange(0.0, E_0, deltaen2, dtype=float)
        npts_low = len(engrid_low)
        d_etaheat_by_d_en_low = np.zeros(len(engrid_low))
        etaheat_int_low = np.zeros(len(engrid_low))
        etaion_int_low = np.zeros(len(engrid_low))
        etaexc_int_low = np.zeros(len(engrid_low))

        for i in reversed(range(len(engrid_low))):
            en_ev = engrid_low[i]
            N_e = self.calculate_N_e(en_ev)
            d_etaheat_by_d_en_low[i] += (
                N_e * en_ev / self.depositionratedensity_ev
            )  # + (yvec[0] * lossfunction(E_0, n_e, n_e_tot) / depositionratedensity_ev)
            etaheat_int_low[i] = (
                etaheat_int_low[i + 1] if i < len(engrid_low) - 1 else etaheat_int[0]
            ) + d_etaheat_by_d_en_low[i] * deltaen2

            etaion_int_low[i] = etaion_int[0]  # cross sections start above E_0
            etaexc_int_low[i] = etaexc_int[0]

        # etatot_int_low = etaion_int_low + etaexc_int_low + etaheat_int_low
        engridfull = np.append(engrid_low, self.engrid)

        # axes[0].plot(engridfull, np.append(etaion_int_low, etaion_int), marker="None", lw=1.5, color='C0', label='Ionisation')
        #
        # if not noexcitation:
        #     axes[0].plot(engridfull, np.append(etaexc_int_low, etaexc_int), marker="None", lw=1.5,
        #                  color='C1', label='Excitation')
        #
        # axes[0].plot(engridfull, np.append(etaheat_int_low, etaheat_int), marker="None", lw=1.5,
        #              color='C2', label='Heating')
        #
        # axes[0].plot(engridfull, np.append(etatot_int_low, etatot_int), marker="None", lw=1.5, color='black', label='Total')
        #
        # axes[0].set_ylim(bottom=0)
        # axes[0].legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 10})
        # axes[0].set_ylabel(r'$\eta$ E to Emax', fontsize=fs)

        # delta_E_y_on_dE = np.zeros(npts)
        # for i in range(len(engrid) - 1):
        #     # delta_E_y_on_dE[i] = ((yvec[i + 1] * engrid[i + 1]) - (yvec[i] * engrid[i])) / (engrid[i + 1] - engrid[i])
        #     delta_E_y_on_dE[i] = yvec[i] * engrid[i]
        # axes[0].plot(engrid, np.log10(delta_E_y_on_dE), marker="None", lw=1.5, color='black', label='')
        # axes[0].set_ylabel(r'log d(E y(E)) / dE', fontsize=fs)

        detaymax = max(
            *(d_etaion_by_d_en_vec * self.engrid),
            *(d_etaexc_by_d_en_vec * self.engrid),
            *(d_etaheat_by_d_en_vec * self.engrid),
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

        # axis.plot(engridfull, np.append(d_etaheat_by_d_en_low, d_etaheat_by_d_en_vec) * engridfull / detaymax,
        #           marker="None", lw=1.5, color='C2', label='Heating')
        ax.plot(
            self.engrid,
            d_etaheat_by_d_en_vec * self.engrid / detaymax,
            marker="None",
            lw=1.5,
            color="C2",
            label="Heating",
        )

        ax.set_ylim(bottom=0, top=1.0)
        ax.legend(loc="best", handlelength=2, frameon=False, numpoints=1, prop={"size": 10})
        ax.set_ylabel(r"E d$\eta$ / dE [eV$^{-1}$]", fontsize=fs)

        # etatot_int = etaion_int + etaexc_int + etaheat_int

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
