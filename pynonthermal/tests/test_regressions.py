"""Regression tests for solver correctness and input validation.

Deliberately not marked as benchmarks: these exercise error paths and cross section edge cases
rather than solver performance, so instrumenting them would only lengthen the CodSpeed run and
add noise to the tracked benchmark set. Performance-relevant cases live in test_sfsolve.py.
"""

import itertools
import math
import warnings

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

import pynonthermal
from pynonthermal import spencerfano


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

    # the same ion is accepted once the cutoff is below its lowest shell, and then conserves energy.
    # npts has to resolve the 7.9 and 9.0 eV shells: at npts=1000 the 3 eV grid spacing leaves the
    # fractions 5.6% short of unity, almost all of it in the ionisation term.
    with pynonthermal.SpencerFanoSolver(emin_ev=7.9, emax_ev=3000, npts=4000) as sf:
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

    # the loss function itself also rejects a non-positive density
    with pytest.raises(ValueError, match="positive free electron density"):
        pynonthermal.electronlossfunction(100.0, 0.0)


def test_deposition_rate_validated() -> None:
    # every fraction and rate coefficient is divided by the deposition rate density. Zero raised a bare
    # ZeroDivisionError from inside the analysis, and a negative value was accepted outright: it flipped
    # the sign of yvec and of every rate coefficient while still leaving the fractions summing to one,
    # so the conservation diagnostic never noticed.
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=300, npts=100) as sf:
        sf.add_ionisation(8, 2, n_ion=1e8)
        for badvalue in (0.0, -100.0):
            with pytest.raises(ValueError, match="depositionratedensity_ev must be greater than zero"):
                sf.solve(depositionratedensity_ev=badvalue)

        sf.solve(depositionratedensity_ev=1e8)
        assert (sf.yvec > 0.0).all()
        assert sf.get_ionisation_ratecoeff(8, 2) > 0.0


def test_excitation_inputs_validated() -> None:
    npts = 100
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=300, npts=npts) as sf:
        sf.add_ionisation(8, 2, n_ion=1e8)
        xs_vec = np.full(npts, 1e-16)

        # a non-positive transition energy puts entries below the diagonal, where the triangular
        # solve silently discards them, and the solution comes out looking plausible
        for badvalue in (-50.0, 0.0):
            with pytest.raises(ValueError, match="epsilon_trans_ev must be greater than zero"):
                sf.add_excitation(8, 2, levelnumberdensity=1e10, xs_vec=xs_vec, epsilon_trans_ev=badvalue)

        # above the top of the grid there is no electron that could drive the transition
        with pytest.raises(ValueError, match="above the top of the energy grid"):
            sf.add_excitation(8, 2, levelnumberdensity=1e10, xs_vec=xs_vec, epsilon_trans_ev=301.0)

        # negative populations and cross sections give a negative excitation fraction
        with pytest.raises(ValueError, match="levelnumberdensity must be non-negative"):
            sf.add_excitation(8, 2, levelnumberdensity=-1e10, xs_vec=xs_vec, epsilon_trans_ev=20.0)
        with pytest.raises(ValueError, match="xs_vec must be non-negative"):
            sf.add_excitation(8, 2, levelnumberdensity=1e10, xs_vec=-xs_vec, epsilon_trans_ev=20.0)

        # a transition exactly at the top of the grid is still legal
        sf.add_excitation(8, 2, levelnumberdensity=1e10, xs_vec=xs_vec, epsilon_trans_ev=300.0)
        assert np.array_equal(sf.sfmatrix, np.triu(sf.sfmatrix))


def test_N_e_excitation_above_grid() -> None:
    # eq. 6 evaluates y and the cross section at energy_ev + epsilon_trans_ev. Above the top of the
    # grid no such electron exists, but get_energyindex_lteq clamps to the last bin, so the term came
    # out equal to its value at emax_ev instead of zero. This is the same clamp that the second
    # ionisation integral is already guarded against.
    npts, emax = 300, 300.0
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=emax, npts=npts) as sf:
        epsilon_trans_ev = 299.5
        sf.add_excitation(
            26,
            2,
            levelnumberdensity=1e8,
            xs_vec=np.where(sf.engrid >= epsilon_trans_ev, 1e-16, 0.0),
            epsilon_trans_ev=epsilon_trans_ev,
        )
        sf.solve(depositionratedensity_ev=1e8, override_n_e=1e8)

        # at 0.5 eV the excited electron is exactly at emax_ev and the term is real
        assert sf.calculate_N_e(emax - epsilon_trans_ev) > 0.0
        # above that it would have to have come from an electron the solver never had
        for energy_ev in (0.75, 1.0):
            assert energy_ev + epsilon_trans_ev > emax
            assert sf.calculate_N_e(energy_ev) == 0.0


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


def test_N_e_empty_second_integral_domain() -> None:
    # Kozma & Fransson eq. 6 has an integral over [2E + I, E_max]. When 2E + I lies beyond the
    # top of the grid that domain is empty, but get_energyindex_lteq clamps to the last bin,
    # which used to add one spurious term. Na XI's single shell (I = 1465 eV) on a 1400-3000 eV
    # grid makes the domain empty for every E > 767.5 eV.
    with pynonthermal.SpencerFanoSolver(emin_ev=1400, emax_ev=3000, npts=400) as sf:
        sf.add_ionisation(11, 10, n_ion=1e8)
        sf.solve(depositionratedensity_ev=1e8)

        shell = (
            pynonthermal.collion.read_colliondata()
            .filter((pl.col("Z") == 11) & (pl.col("ion_stage") == 10))
            .to_dicts()[0]
        )
        ionpot_ev = float(shell["ionpot_ev"])
        J = pynonthermal.collion.get_J(11, 10, ionpot_ev)

        energy_ev = 1200.0
        assert 2 * energy_ev + ionpot_ev > sf.engrid[-1] + sf.deltaen  # second domain is empty

        # reference: eq. 6 with only the first integral, epsilon in [I, min(E_max - E, E + I)],
        # integrated on a grid far finer than the solver's own
        enlambda = min(float(sf.engrid[-1]) - energy_ev, energy_ev + ionpot_ev)
        arr_epsilon: npt.NDArray[np.float64] = np.linspace(ionpot_ev, enlambda, num=20001)
        arr_e_p: npt.NDArray[np.float64] = energy_ev + arr_epsilon
        arr_y: npt.NDArray[np.float64] = np.interp(arr_e_p, sf.engrid, sf.yvec, left=0.0, right=0.0)
        arr_xs = pynonthermal.collion.get_arxs_array_shell(arr_e_p, shell)
        # the closed form of Psecondary, written out so that the reference does not depend on the
        # implementation it is checking
        arr_p = 1.0 / J / np.arctan((arr_e_p - ionpot_ev) / 2 / J) / (1 + ((arr_epsilon - ionpot_ev) / J) ** 2)
        expected = 1e8 * float(np.trapezoid(arr_y * arr_xs * arr_p, arr_epsilon))

        # the tolerance covers the solver's own coarser sub-grids for these integrals, and is tight
        # enough to notice the empty-domain guard going away (asserted just below)
        got = sf.calculate_N_e(energy_ev)
        assert math.isclose(got, expected, rel_tol=1e-4)

        # reproduce the term the old clamping added, and check the result would no longer pass
        clamped_index = sf.get_energyindex_lteq(2 * energy_ev + ionpot_ev)
        spurious = 1e8 * float(
            sf.deltaen
            * sf.yvec[clamped_index]
            * pynonthermal.collion.get_arxs_array_shell(sf.engrid, shell)[clamped_index]
            * pynonthermal.collion.Psecondary_vec(
                e_p=float(sf.engrid[clamped_index]), e_s=energy_ev, ionpot_ev=ionpot_ev, J=J
            )
        )
        assert spurious > 0.0
        assert not math.isclose(got + spurious, expected, rel_tol=1e-4)


def test_N_e_epsilon_integral_not_keyed_to_the_grid() -> None:
    # The epsilon integral of eq. 6 spans at most emin_ev, so it is narrower than one cell of a
    # typical energy grid and needs a sub-grid of its own. Evaluating it on the solver's own grid
    # instead made the answer depend on where the grid points happened to fall relative to the
    # ionisation threshold rather than on the resolution. Al I has a 6.0 eV shell, and npts of 600,
    # 1200, 1800 and 2400 each put a grid point within 10 meV of it on a 1-3000 eV grid, where the
    # Psecondary normalisation 1/atan((e_p - I) / 2J) diverges. That gave frac_sum of 1.86, 1.37,
    # 1.15 and 1.07 at those four npts while the next npts up gave 0.99-1.00 each time.
    # assert the premise rather than trusting the hard-coded npts: each of the first of these pairs
    # must put a grid point within a few meV of an Al I threshold, and each second one must not
    ionpots = [
        float(row["ionpot_ev"])
        for row in pynonthermal.collion.read_colliondata()
        .filter((pl.col("Z") == 13) & (pl.col("ion_stage") == 1))
        .to_dicts()
    ]

    def distance_to_threshold(npts: int) -> float:
        engrid = np.linspace(1.0, 3000.0, num=npts)
        return min(float(np.abs(engrid - ionpot).min()) for ionpot in ionpots)

    pairs = ((600, 700), (1200, 1300), (1800, 1900), (2400, 2900))
    for npts_onthreshold, npts_neighbour in pairs:
        assert distance_to_threshold(npts_onthreshold) < 0.01, (
            f"npts={npts_onthreshold} is no longer on an Al I threshold; this test needs one that is"
        )
        assert distance_to_threshold(npts_neighbour) > 0.05, (
            f"npts={npts_neighbour} is itself on an Al I threshold, so it is not a clean control"
        )

    fracsums: dict[int, float] = {}
    for npts in [npts for pair in pairs for npts in pair]:
        with (
            pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=npts) as sf,
            # the coarsest grids here are genuinely under-resolved, so the conservation diagnostic
            # fires as designed. This test is about the spikes, not about the residual error.
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", UserWarning)
            sf.add_ionisation(13, 1, n_ion=1.0)
            sf.solve(depositionratedensity_ev=100, override_n_e=1e-4)
            fracsums[npts] = sf.get_frac_sum()

    for npts_onthreshold, npts_neighbour in pairs:
        assert math.isclose(fracsums[npts_onthreshold], fracsums[npts_neighbour], abs_tol=0.03), (
            f"npts={npts_onthreshold} gives frac_sum={fracsums[npts_onthreshold]} but the neighbouring"
            f" npts={npts_neighbour} gives {fracsums[npts_neighbour]}"
        )

    # and the remaining error is ordinary discretisation error, which shrinks with resolution
    assert abs(fracsums[2900] - 1.0) < abs(fracsums[600] - 1.0)


def test_N_e_epsilon_integral_value_on_a_narrow_domain() -> None:
    # The checks above pin the symptom (no spikes) rather than the value, so they would still pass if
    # the epsilon integral returned zero. Pin the value itself, in the regime the fix is for: emin_ev=1
    # makes the domain [I, E + I] at most 1 eV wide, narrower than the 0.75 eV grid spacing here.
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=4000) as sf:
        sf.add_ionisation(13, 1, n_ion=1.0)
        sf.solve(depositionratedensity_ev=100, override_n_e=1e-4)

        shells = (
            pynonthermal.collion.read_colliondata().filter((pl.col("Z") == 13) & (pl.col("ion_stage") == 1)).to_dicts()
        )
        energy_ev = 0.7
        emax = float(sf.engrid[-1])

        def psecondary(
            arr_e_p: npt.NDArray[np.float64], e_s: npt.NDArray[np.float64] | float, ionpot_ev: float, J: float
        ) -> npt.NDArray[np.float64]:
            # the closed form, so the reference does not depend on the implementation it is checking
            return 1.0 / J / np.arctan((arr_e_p - ionpot_ev) / 2 / J) / (1 + (np.asarray(e_s) / J) ** 2)

        # both eq. 6 integrals over every shell, each on a grid far finer than the solver's own
        expected = 0.0
        narrow_domain_seen = False
        for shell in shells:
            ionpot_ev = float(shell["ionpot_ev"])
            J = pynonthermal.collion.get_J(13, 1, ionpot_ev)

            enlambda = min(emax - energy_ev, energy_ev + ionpot_ev)
            if enlambda > ionpot_ev:
                narrow_domain_seen |= (enlambda - ionpot_ev) < sf.deltaen
                arr_epsilon = np.linspace(ionpot_ev, enlambda, num=20001)
                arr_e_p = arr_epsilon + energy_ev
                arr_y = np.interp(arr_e_p, sf.engrid, sf.yvec, left=0.0, right=0.0)
                arr_xs = pynonthermal.collion.get_arxs_array_shell(arr_e_p, shell)
                arr_p = psecondary(arr_e_p, arr_epsilon - ionpot_ev, ionpot_ev, J)
                expected += float(np.trapezoid(arr_y * arr_xs * arr_p, arr_epsilon))

            en_lower2 = 2 * energy_ev + ionpot_ev
            if en_lower2 < emax:
                arr_e_p2 = np.linspace(en_lower2, emax, num=40001)
                arr_y2 = np.interp(arr_e_p2, sf.engrid, sf.yvec, left=0.0, right=0.0)
                arr_xs2 = pynonthermal.collion.get_arxs_array_shell(arr_e_p2, shell)
                arr_p2 = psecondary(arr_e_p2, energy_ev, ionpot_ev, J)
                expected += float(np.trapezoid(arr_y2 * arr_xs2 * arr_p2, arr_e_p2))

        # the epsilon domain really is narrower than one grid cell here, the regime the sub-grid is for
        assert narrow_domain_seen
        assert expected > 0.0
        assert math.isclose(sf.calculate_N_e(energy_ev), expected, rel_tol=0.02)


def test_quadrature_resolution_constants_are_adequate() -> None:
    # NPTS_SUB_E0_INTEGRAL and NPTS_EPSILON_SUBGRID carry accuracy claims in their comments. Pin them,
    # so that lowering either to buy speed cannot silently degrade frac_heating. emin_ev=7.9 is the
    # demanding case, where the integral over [0, E_0] is about a third of the heating.
    def frac_heating(monkeypatch: pytest.MonkeyPatch, npts_sub_e0: int, emin_ev: float, Z: int) -> float:
        monkeypatch.setattr(spencerfano, "NPTS_SUB_E0_INTEGRAL", npts_sub_e0)
        with pynonthermal.SpencerFanoSolver(emin_ev=emin_ev, emax_ev=3000, npts=2000) as sf:
            sf.add_ionisation(Z, 1, n_ion=0.99)
            sf.add_ionisation(Z, 2, n_ion=0.01)
            sf.solve(depositionratedensity_ev=1e6)
            return sf.get_frac_heating()

    # emin_ev has to cover the large values add_ionisation's own error message steers users towards
    # ("set emin_ev <= {lowest ionpot} eV"), not just the default grid where this term is negligible
    for emin_ev, Z in ((7.9, 26), (15.7, 18), (24.5, 2)):
        with pytest.MonkeyPatch.context() as mp:
            converged = frac_heating(mp, 129, emin_ev, Z)
            asconfigured = frac_heating(mp, spencerfano.NPTS_SUB_E0_INTEGRAL, emin_ev, Z)
        assert math.isclose(asconfigured, converged, rel_tol=1e-3), (
            f"emin_ev={emin_ev}: the configured resolution gives frac_heating={asconfigured}"
            f" against a converged {converged}"
        )


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
    # the cross section must fall monotonically through the former 255 keV cliff
    assert (np.diff(xs_vec) < 0.0).all()


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

    # a negative population is rejected on the excitation path too
    with (
        pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf,
        pytest.raises(ValueError, match="non-negative"),
    ):
        sf.add_ion_ltepopexcitation(26, 3, n_ion=-1.0, use_collstrengths=False)


def test_conservation_warning_on_coarse_grid() -> None:
    # the energy fractions of a helium plasma are ~6.5% off unity at npts=100,
    # which must trigger the conservation diagnostic
    x_e = 1e-4
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=100) as sf:
        for Z, ion_stage, n_ion in ((2, 1, 1.0 - x_e), (2, 2, x_e)):
            sf.add_ionisation(Z, ion_stage, n_ion=n_ion)
            sf.add_ion_ltepopexcitation(Z, ion_stage, n_ion=n_ion, use_collstrengths=False)
        sf.solve(depositionratedensity_ev=100)

        with pytest.warns(UserWarning, match="energy fractions sum to"):
            frac_sum = sf.get_frac_sum()
        assert not math.isclose(frac_sum, 1.0, rel_tol=0.05)


def test_opal_J_applies_to_every_shell_of_its_ion() -> None:
    # The measured secondary-electron widths of Opal et al. 1971 are whole-atom values dominated by
    # valence-shell ionisation, but ARTIS applies each to every shell of its ion and pynonthermal
    # deliberately matches that, so Ne I's 48.5 eV 2s shell gets the same J as its 21.6 eV 2p shell.
    assert pynonthermal.collion.get_J(2, 1, 24.6) == 15.8  # He I, its only shell
    for ionpot_ev in (21.6, 48.5):  # Ne I 2p and 2s
        assert pynonthermal.collion.get_J(10, 1, ionpot_ev) == 24.2
    for ionpot_ev in (15.8, 29.2):  # Ar I 3p and 3s
        assert pynonthermal.collion.get_J(18, 1, ionpot_ev) == 10.0

    # every other ion, including the ionised stages of those three, uses the general rule
    for Z, ion_stage, ionpot_ev in ((26, 2, 16.18), (2, 2, 54.4), (10, 2, 41.0), (18, 2, 27.6)):
        assert pynonthermal.collion.get_J(Z, ion_stage, ionpot_ev) == 0.6 * ionpot_ev


def test_psecondary_requires_an_energy() -> None:
    # this guard was an assert, so it vanished under python -O and left the sentinel default of -1
    # to be used as a real secondary energy
    with pytest.raises(ValueError, match="e_s or the primary energy loss epsilon"):
        pynonthermal.collion.Psecondary(e_p=100.0, ionpot_ev=10.0, J=6.0)

    # the scalar and vectorised forms have to agree, since the solver uses both
    e_s = np.array([0.0, 5.0, 20.0])
    arr = pynonthermal.collion.Psecondary_vec(e_p=100.0, e_s=e_s, ionpot_ev=10.0, J=6.0)
    for i, value in enumerate(e_s):
        assert math.isclose(
            float(arr[i]), pynonthermal.collion.Psecondary(e_p=100.0, e_s=float(value), ionpot_ev=10.0, J=6.0)
        )

    # below threshold there are no secondaries
    assert not pynonthermal.collion.Psecondary_vec(e_p=np.array([5.0, 10.0]), e_s=0.0, ionpot_ev=10.0, J=6.0).any()


def test_n_e_nt_relativistic() -> None:
    # the electron speed used to be the classical sqrt(2E/m_e), which exceeds c above 511 keV and is
    # already 2.5 per cent fast at the 16 keV emax_ev that the iron benchmark uses
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=16000, npts=400) as sf:
        sf.add_ionisation(26, 2, n_ion=1.0)
        sf.solve(depositionratedensity_ev=1e8)
        n_e_nt = sf.get_n_e_nt()
        assert n_e_nt > 0.0

        # the classical speed exceeds the relativistic one, so it gives a lower density. Comparing
        # against it pins the physics without restating the implementation.
        arr_velocity_classical = np.sqrt(2 * sf.engrid * pynonthermal.constants.EV / pynonthermal.constants.ME)
        assert n_e_nt > float(np.sum(sf.yvec / arr_velocity_classical) * sf.deltaen)

        # and no electron may travel faster than light, which the classical form allows above 511 keV
        assert (pynonthermal.base.get_betasq(sf.engrid) < 1.0).all()
        assert n_e_nt > float(np.sum(sf.yvec) * sf.deltaen / pynonthermal.constants.CLIGHT)


def test_lossfunction_plasma_energy_guard() -> None:
    # the high-energy branch is ln(2E / hbar*omega_p), which is zero or negative once the plasma
    # energy reaches 2E. That was an assert, so under python -O it returned a non-positive loss rate.
    with pytest.raises(ValueError, match="not valid at"):
        pynonthermal.electronlossfunction(100.0, 1e30)

    # for densities the solver is actually used at, the loss rate is positive and falls with energy
    arr_loss = [pynonthermal.electronlossfunction(en_ev, 1e8) for en_ev in (1.0, 5.0, 14.0, 100.0, 3000.0)]
    assert all(loss > 0.0 for loss in arr_loss)
    assert all(b < a for a, b in itertools.pairwise(arr_loss))
