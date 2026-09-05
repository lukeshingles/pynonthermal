import math
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import pynonthermal

pytestmark = pytest.mark.benchmark
outputfolder = Path(__file__).absolute().parent / "output"


def test_lotz_heavy_element() -> None:
    # elements heavier than Ni (Z>28) use the Axelrod 1980/Lotz 1967 cross section approximation
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=400, verbose=True) as sf:
        sf.add_ionisation(56, 2, n_ion=1.0)
        sf.solve(depositionratedensity_ev=100, override_n_e=1.0)

        # per-ion getter triggers the analysis lazily
        assert sf.get_frac_ionisation_ion(56, 2) > 0.0
        assert sf.get_eff_ionpot(56, 2) > 0.0
        assert math.isclose(sf.get_frac_sum(), 1.0, abs_tol=0.01)
        assert sf.get_ionisation_ratecoeff(56, 2) > 0.0


class CountingAnalysisSolver(pynonthermal.SpencerFanoSolver):
    """A solver that counts the calls to analyse_ntspectrum."""

    analyse_count: int = 0

    def analyse_ntspectrum(self) -> None:
        self.analyse_count += 1
        super().analyse_ntspectrum()


def test_api_guards() -> None:
    with CountingAnalysisSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        sf.add_ionisation(2, 1, n_ion=1.0)

        # getters require solve() to have been called
        with pytest.raises(RuntimeError):
            sf.get_frac_heating()
        with pytest.raises(RuntimeError):
            sf.get_excitation_ratecoeff(2, 1, 0)

        # the same ion can't be added twice: its channel keys are already present
        with pytest.raises(ValueError, match="already added"):
            sf.add_ionisation(2, 1, n_ion=1.0)

        # an ion with no cross-section data (here H II, which has no bound electrons) is rejected
        with pytest.raises(ValueError, match="No ionisation cross-section data"):
            sf.add_ionisation(1, 2, n_ion=1.0)

        # xs_vec must be defined on the full energy grid
        with pytest.raises(ValueError, match="engrid"):
            sf.add_excitation(2, 1, levelnumberdensity=1.0, xs_vec=np.zeros(3), epsilon_trans_ev=10.0)

        # an omitted transitionkey gets an automatic index, and duplicate keys are rejected
        sf.add_excitation(2, 1, levelnumberdensity=1.0, xs_vec=np.zeros(200), epsilon_trans_ev=25.0)
        with pytest.raises(ValueError, match="already added"):
            sf.add_excitation(
                2, 1, levelnumberdensity=1.0, xs_vec=np.zeros(200), epsilon_trans_ev=25.0, transitionkey=0
            )

        # excitation data must exist for the requested ion
        with pytest.raises(ValueError, match="No excitation data"):
            sf.add_ion_ltepopexcitation(
                3, 1, n_ion=1.0, temperature=3000, adata_polars=pl.DataFrame({"Z": [2], "ion_stage": [1]})
            )

        # a zero-population ion is a no-op and a negative population is rejected
        sf.add_ionisation(8, 3, n_ion=0.0)
        assert (8, 3) not in sf.ionpopdict
        with pytest.raises(ValueError, match="non-negative"):
            sf.add_ionisation(8, 3, n_ion=-1.0)

        # second ion with no excitation channels (exercises the zero-excitation analysis path)
        sf.add_ionisation(8, 2, n_ion=0.1)

        sf.solve(depositionratedensity_ev=100, override_n_e=1e-4)

        # a legitimately-zero excitation fraction must not trigger repeated re-analysis
        assert sf.get_frac_excitation_tot() == 0.0
        count_after_first_call = sf.analyse_count
        assert count_after_first_call >= 1
        assert sf.get_frac_excitation_tot() == 0.0
        assert sf.analyse_count == count_after_first_call

        # each getter triggers the analysis lazily from a freshly-solved (un-analysed) state
        sf.solve(depositionratedensity_ev=100, override_n_e=1e-4)
        assert sf.get_frac_ionisation_tot() > 0.0
        sf.solve(depositionratedensity_ev=100, override_n_e=1e-4)
        assert sf.get_frac_ionisation_ion(2, 1) > 0.0
        sf.solve(depositionratedensity_ev=100, override_n_e=1e-4)
        assert sf.get_eff_ionpot(2, 1) > 0.0
        sf.solve(depositionratedensity_ev=100, override_n_e=1e-4)
        assert sf.get_ionisation_ratecoeff(2, 1) > 0.0

        # additions are locked after solving
        with pytest.raises(RuntimeError):
            sf.add_ionisation(2, 2, n_ion=1e-4)


def test_invalid_excitation_fraction_reported() -> None:
    # an unphysically-large excitation cross section produces frac_excitation > 1. It is reported
    # but kept in the total, as for frac_ionisation_shell, so that the problem stays visible
    # rather than being hidden behind a total that silently dropped the channel.
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=100) as sf:
        sf.add_ionisation(2, 1, n_ion=1.0)
        sf.add_excitation(2, 1, levelnumberdensity=1e30, xs_vec=np.full(100, 1e-10), epsilon_trans_ev=25.0)
        sf.solve(depositionratedensity_ev=100, override_n_e=1e-4)

        with pytest.warns(UserWarning, match="invalid frac_excitation_ion"):
            frac_excitation_tot = sf.get_frac_excitation_tot()

        assert frac_excitation_tot > 1.0


def test_helium() -> None:
    # KF1992 Figure 3. Pure-Helium Plasma
    x_e = 1e-4
    ions = [
        # Z ion_stage numberdensity
        (2, 1, 1.0 - x_e),
        (2, 2, x_e),
    ]

    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=2000, verbose=True) as sf:
        for Z, ion_stage, n_ion in ions:
            sf.add_ionisation(Z, ion_stage, n_ion=n_ion)
            sf.add_ion_ltepopexcitation(Z, ion_stage, n_ion=n_ion, temperature=3000, use_collstrengths=False)

        # re-adding an ion's excitations with a different population is not allowed
        with pytest.raises(ValueError, match="different populations"):
            sf.add_ion_ltepopexcitation(2, 1, n_ion=99.9, temperature=3000, use_collstrengths=False)

        # call solve twice to test that it can be called multiple times without error
        sf.solve(depositionratedensity_ev=1000)
        sf.solve(depositionratedensity_ev=100)

        sf.analyse_ntspectrum()
        frac_excitation_tot = sf.get_frac_excitation_tot()
        frac_ionisation_tot = sf.get_frac_ionisation_tot()
        frac_heating = sf.get_frac_heating()

        frac_sum = frac_excitation_tot + frac_ionisation_tot + frac_heating
        assert math.isclose(frac_sum, 1.0, abs_tol=0.005)
        assert math.isclose(frac_excitation_tot, 0.3315, abs_tol=0.05)
        assert math.isclose(frac_ionisation_tot, 0.4849, abs_tol=0.05)
        assert math.isclose(sf.get_frac_ionisation_ion(Z=ions[0][0], ion_stage=ions[0][1]), 0.4807, abs_tol=0.05)
        assert math.isclose(sf.get_frac_ionisation_ion(Z=ions[1][0], ion_stage=ions[1][1]), 0.0, abs_tol=0.05)

        sf.plot_spec_channels(outputfilename=outputfolder / "spec_channels.pdf", xscalelog=True)
        sf.plot_yspectrum(outputfilename=outputfolder / "yspectrum.pdf")
        sf.plot_channels(outputfilename=outputfolder / "channels.pdf", xscalelog=True)


def test_heating_only_approximation() -> None:
    # with heating_only_approximation=True, the matrix contains no excitation or ionisation loss
    # terms, but the analysis still calculates the channel rates from the heating-only solution
    x_e = 1e-4
    ions = [
        # Z ion_stage numberdensity
        (2, 1, 1.0 - x_e),
        (2, 2, x_e),
    ]

    def build_solver(heating_only_approximation: bool) -> pynonthermal.SpencerFanoSolver:
        # the assertions are qualitative, so a coarse grid is enough (test_helium covers npts=2000)
        sf = pynonthermal.SpencerFanoSolver(
            emin_ev=1, emax_ev=3000, npts=512, heating_only_approximation=heating_only_approximation
        )
        for Z, ion_stage, n_ion in ions:
            sf.add_ionisation(Z, ion_stage, n_ion=n_ion)
            sf.add_ion_ltepopexcitation(Z, ion_stage, n_ion=n_ion, temperature=3000, use_collstrengths=False)
        return sf

    with (
        build_solver(heating_only_approximation=True) as sf_heat,
        build_solver(heating_only_approximation=False) as sf_full,
    ):
        # the excitation and ionisation terms stay out of the matrix
        assert not sf_heat.sfmatrix.any()
        assert sf_full.sfmatrix.any()

        sf_heat.solve(depositionratedensity_ev=100)
        sf_full.solve(depositionratedensity_ev=100)

        # the analysis of the heating-only solution must not show a solver warning
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            sf_heat.analyse_ntspectrum()

        # heating takes almost all of the deposited energy
        assert math.isclose(sf_heat.get_frac_heating(), 1.0, rel_tol=0.05)

        # the channel rates are still available from the stored data
        assert sf_heat.get_frac_ionisation_tot() > 0.0
        assert sf_heat.get_frac_excitation_tot() > 0.0
        assert sf_heat.get_ionisation_ratecoeff(2, 1) > 0.0
        assert sf_heat.get_frac_ionisation_ion(2, 1) > 0.0

        # without the competing loss terms, y(E) is larger, so the rates are larger
        assert sf_heat.get_frac_ionisation_tot() > sf_full.get_frac_ionisation_tot()
        assert sf_heat.get_frac_excitation_tot() > sf_full.get_frac_excitation_tot()


def test_iron() -> None:
    ions = [
        # Z ion_stage numberdensity
        (26, 2, 0.3),
        (26, 3, 0.7),
    ]

    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=16000, npts=1024, verbose=True) as sf:
        for Z, ion_stage, n_ion in ions:
            sf.add_ionisation(Z, ion_stage, n_ion=n_ion)
            sf.add_ion_ltepopexcitation(Z, ion_stage, n_ion=n_ion, temperature=3000, use_collstrengths=False)

        sf.solve(depositionratedensity_ev=100)

        sf.analyse_ntspectrum()
        frac_excitation_tot = sf.get_frac_excitation_tot()
        frac_ionisation_tot = sf.get_frac_ionisation_tot()
        frac_heating = sf.get_frac_heating()

        frac_sum = frac_excitation_tot + frac_ionisation_tot + frac_heating
        assert math.isclose(frac_sum, 1.0, abs_tol=0.005)
        assert math.isclose(frac_excitation_tot, 0.0204, abs_tol=0.05)
        assert math.isclose(frac_ionisation_tot, 0.1391, abs_tol=0.05)

        assert math.isclose(sf.get_ionisation_ratecoeff(26, 2), 4.44e-01, rel_tol=0.05)
        assert math.isclose(sf.get_ionisation_ratecoeff(26, 3), 3.70e-01, rel_tol=0.05)

        # rate coefficients are in s^-1 and so scale with the deposition rate density of 100 eV/s/cm3
        assert math.isclose(sf.get_excitation_ratecoeff(26, 2, (0, 100)), 3.9930269946568673e-05, rel_tol=0.05)
        assert math.isclose(sf.get_excitation_ratecoeff(26, 2, (1, 100)), 6.654239325994856e-06, rel_tol=0.05)

        sf.plot_spec_channels(outputfilename=outputfolder / "spec_channels.pdf", xscalelog=True)
