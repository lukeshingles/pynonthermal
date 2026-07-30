"""Regression tests for solver correctness and input validation.

Deliberately not marked as benchmarks: these exercise error paths and cross section edge cases
rather than solver performance, so instrumenting them would only lengthen the CodSpeed run and
add noise to the tracked benchmark set. Performance-relevant cases live in test_sfsolve.py.
"""

import math

import numpy as np
import polars as pl
import pytest

import pynonthermal


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


def test_n_e_cache_invalidated_by_later_adds() -> None:
    # reading n_e between add_* calls must not freeze it at the earlier value
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        sf.add_ionisation(8, 2, n_ion=1e8)
        assert sf.get_n_e() == 1e8
        sf.add_ionisation(8, 3, n_ion=1e8)
        assert sf.get_n_e() == 3e8
        sf.add_ion_ltepopexcitation(26, 3, n_ion=5e7, use_collstrengths=False)
        assert sf.get_n_e() == 4e8


def test_lotz_xs_relativistic() -> None:
    # the Lotz/Axelrod cross section must fall off smoothly rather than dropping to zero at the
    # 255 keV energy where the classical beta^2 = 2E/mc^2 reaches one
    shell = (
        pynonthermal.collion.read_colliondata()
        .filter((pl.col("Z") == 56) & (pl.col("ion_stage") == 2) & (pl.col("n") < 0))
        .to_dicts()[0]
    )
    arr_en_ev = np.array([1e2, 1e3, 1e4, 1e5, 2.54e5, 2.6e5, 5.11e5, 1e6])
    xs_vec = pynonthermal.collion.get_arxs_array_shell(arr_en_ev, shell)

    assert (xs_vec > 0.0).all()
    assert np.isfinite(xs_vec).all()
    # and the scalar implementation agrees instead of raising a math domain error
    assert np.allclose(xs_vec, [pynonthermal.axelrod.get_lotz_xs_ionisation(shell, float(en)) for en in arr_en_ev])


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
