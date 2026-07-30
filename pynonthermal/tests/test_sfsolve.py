import math
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


def test_grid_validation() -> None:
    with pytest.raises(ValueError, match="npts must be at least 2"):
        pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=1)
    with pytest.raises(ValueError, match="emin_ev must be greater than zero"):
        pynonthermal.SpencerFanoSolver(emin_ev=0, emax_ev=3000, npts=100)
    with pytest.raises(ValueError, match="must be greater than emin_ev"):
        pynonthermal.SpencerFanoSolver(emin_ev=3000, emax_ev=1, npts=100)


def test_ionpot_below_emin_rejected() -> None:
    # Fe I has 7.9 and 9.0 eV shells. Kozma & Fransson 1992 require all thresholds above the
    # low-energy cutoff, so a grid starting above them would not conserve energy.
    with (
        pynonthermal.SpencerFanoSolver(emin_ev=12.0, emax_ev=3000, npts=200) as sf,
        pytest.raises(ValueError, match="below emin_ev"),
    ):
        sf.add_ionisation(26, 1, n_ion=1.0)

    # the same ion is accepted once the cutoff is below its lowest shell, and then conserves energy
    with pynonthermal.SpencerFanoSolver(emin_ev=7.9, emax_ev=3000, npts=1000) as sf:
        sf.add_ionisation(26, 1, n_ion=0.99)
        sf.add_ionisation(26, 2, n_ion=0.01)
        sf.solve(depositionratedensity_ev=1e6)
        assert math.isclose(sf.get_frac_sum(), 1.0, abs_tol=0.05)


def test_zero_free_electron_density() -> None:
    # a plasma of only neutral ions has no thermal electron loss channel
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        sf.add_ionisation(2, 1, n_ion=1.0)
        assert sf.get_n_e() == 0.0
        with pytest.raises(ValueError, match="free electron density is zero"):
            sf.solve(depositionratedensity_ev=100)

        # override_n_e is the documented way out, but must be a usable density
        for badvalue in (0.0, -1.0):
            with pytest.raises(ValueError, match="override_n_e must be greater than zero"):
                sf.solve(depositionratedensity_ev=100, override_n_e=badvalue)

        sf.solve(depositionratedensity_ev=100, override_n_e=1e-4)
        assert sf.get_n_e() == 1e-4


def test_override_n_e_not_confused_with_cache() -> None:
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        sf.add_ionisation(8, 2, n_ion=1e8)
        sf.add_ionisation(8, 3, n_ion=1e8)
        assert sf.calculate_free_electron_density() == 3e8

        sf.solve(depositionratedensity_ev=1e8, override_n_e=1e6)
        assert sf.get_n_e() == 1e6

        # omitting the override falls back to the ion populations
        sf.solve(depositionratedensity_ev=1e8)
        assert sf.get_n_e() == 3e8


def test_excitation_xs_zero_above_grid() -> None:
    # a transition that no electron on the grid can drive must have a zero cross section
    # everywhere, rather than a spurious (and for the E1 branch, negative) value at the top point
    engrid = np.linspace(1.0, 100.0, 200)
    for row in (
        {"collstr": -1, "epsilon_trans_ev": 500.0, "forbidden": 0, "lower_g": 1, "upper_g": 3, "A": 1e8},
        {"collstr": 1.0, "epsilon_trans_ev": 500.0, "forbidden": 1, "lower_g": 1, "upper_g": 3, "A": 0.0},
    ):
        xs_vec = pynonthermal.excitation.get_xs_excitation_vector(engrid, row)
        assert not xs_vec.any()
        # the scalar implementation has always agreed on this point
        assert pynonthermal.excitation.get_xs_excitation(engrid[-1], row) == 0.0

    # a transition right at the top of the grid is still open
    row_attop = {"collstr": -1, "epsilon_trans_ev": 100.0, "forbidden": 0, "lower_g": 1, "upper_g": 3, "A": 1e8}
    assert pynonthermal.excitation.get_xs_excitation_vector(engrid, row_attop)[-1] > 0.0

    # no cross section anywhere may be negative for real atomic data on a grid far below the
    # highest transition energy
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=5.0, npts=300) as sf:
        sf.add_ionisation(26, 3, n_ion=0.7)
        sf.add_ion_ltepopexcitation(26, 3, n_ion=0.7)
        for transitions in sf.excitationlists.values():
            for _levelpop, xs_vec, epsilon_trans_ev in transitions.values():
                assert epsilon_trans_ev <= sf.engrid[-1]
                assert (xs_vec >= 0.0).all()


def test_excitation_only_ion_counted() -> None:
    # an ion given an excitation channel but no ionisation channel still spends deposited energy,
    # so its excitation fraction has to appear in the totals
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=500) as sf:
        sf.add_ionisation(8, 2, n_ion=1e8)
        xs_vec = np.where(sf.engrid >= 20.0, 1e-16, 0.0)
        sf.add_excitation(26, 2, levelnumberdensity=1e8, xs_vec=xs_vec, epsilon_trans_ev=20.0)
        sf.solve(depositionratedensity_ev=1e8)

        assert sf.get_frac_excitation_tot() > 0.3
        assert math.isclose(sf.get_frac_sum(), 1.0, abs_tol=0.01)


def test_ltepopexcitation_registers_population() -> None:
    # add_ion_ltepopexcitation() without add_ionisation() must still contribute the ion's free
    # electrons and nuclei, and must not invent an ionisation channel for it
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=600) as sf:
        sf.add_ionisation(8, 2, n_ion=1e8)
        sf.add_ion_ltepopexcitation(26, 3, n_ion=5e7, use_collstrengths=False)

        assert sf.ionpopdict[(26, 3)] == 5e7
        assert sf.get_n_e() == 1e8 * 1 + 5e7 * 2
        assert sf.get_n_ion_tot() == 1.5e8

        sf.solve(depositionratedensity_ev=1e8)
        assert sf.get_frac_excitation_tot() > 0.0
        assert sf.get_ionisation_ratecoeff(26, 3) == 0.0
        assert sf.get_eff_ionpot(26, 3) == float("inf")
        assert math.isclose(sf.get_frac_sum(), 1.0, abs_tol=0.01)


class CountingAnalysisSolver(pynonthermal.SpencerFanoSolver):
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

        # the same ion can't be added twice
        with pytest.raises(ValueError, match="twice"):
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
            sf.add_ion_ltepopexcitation(3, 1, n_ion=1.0, adata_polars=pl.DataFrame({"Z": [2], "ion_stage": [1]}))

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


def test_invalid_excitation_fraction_ignored() -> None:
    # an unphysically-large excitation cross section produces frac_excitation > 1,
    # which is reported and excluded from the total
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=100) as sf:
        sf.add_ionisation(2, 1, n_ion=1.0)
        sf.add_excitation(2, 1, levelnumberdensity=1e30, xs_vec=np.full(100, 1e-10), epsilon_trans_ev=25.0)
        sf.solve(depositionratedensity_ev=100, override_n_e=1e-4)

        assert sf.get_frac_excitation_tot() == 0.0


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
            sf.add_ion_ltepopexcitation(Z, ion_stage, n_ion=n_ion, use_collstrengths=False)

        # re-adding an ion's excitations with a different population is not allowed
        with pytest.raises(ValueError, match="different populations"):
            sf.add_ion_ltepopexcitation(2, 1, n_ion=99.9, use_collstrengths=False)

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


def test_iron() -> None:
    ions = [
        # Z ion_stage numberdensity
        (26, 2, 0.3),
        (26, 3, 0.7),
    ]

    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=16000, npts=1024, verbose=True) as sf:
        for Z, ion_stage, n_ion in ions:
            sf.add_ionisation(Z, ion_stage, n_ion=n_ion)
            sf.add_ion_ltepopexcitation(Z, ion_stage, n_ion=n_ion, use_collstrengths=False)

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

        assert math.isclose(sf.get_excitation_ratecoeff(26, 2, (0, 100)), 3.9930269946568673e-07, rel_tol=0.05)
        assert math.isclose(sf.get_excitation_ratecoeff(26, 2, (1, 100)), 6.654239325994856e-08, rel_tol=0.05)

        sf.plot_spec_channels(outputfilename=outputfolder / "spec_channels.pdf", xscalelog=True)
