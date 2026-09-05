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


def test_solve_inputs_reject_nonfinite() -> None:
    # nan slipped past the old "<= 0.0" guards because nan comparisons are always False, and inf
    # passed them outright; either then contaminated the solution without any exception being raised
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=300, npts=100) as sf:
        sf.add_ionisation(8, 2, n_ion=1e8)
        for badvalue in (math.nan, math.inf):
            with pytest.raises(ValueError, match="depositionratedensity_ev must be greater than zero and finite"):
                sf.solve(depositionratedensity_ev=badvalue)
            with pytest.raises(ValueError, match="override_n_e must be greater than zero and finite"):
                sf.solve(depositionratedensity_ev=1e8, override_n_e=badvalue)


def test_solve_upper_triangular_guards() -> None:
    # scipy.linalg.solve_triangular raised on non-finite entries and zero diagonals; the numpy
    # back-substitution that replaced it must do the same rather than return nan/inf solutions
    a = np.triu(np.ones((3, 3)))
    b = np.ones(3)
    assert np.allclose(spencerfano.solve_upper_triangular(a, b), [0.0, 0.0, 1.0])

    a_nonfinite = a.copy()
    a_nonfinite[0, 2] = math.nan
    with pytest.raises(ValueError, match="finite"):
        spencerfano.solve_upper_triangular(a_nonfinite, b)

    with pytest.raises(ValueError, match="finite"):
        spencerfano.solve_upper_triangular(a, np.array([1.0, math.inf, 1.0]))

    a_singular = a.copy()
    a_singular[1, 1] = 0.0
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        spencerfano.solve_upper_triangular(a_singular, b)


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
        sf.set_temperature(3000)
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
        sf.set_temperature(3000)
        sf.add_ionisation(26, 3, n_ion=0.7)
        sf.add_ion_ltepopexcitation(26, 3, n_ion=0.7)
        for transitions in sf.excitationlists.values():
            for trans in transitions.values():
                assert trans.epsilon_trans_ev <= sf.engrid[-1]
                assert (trans.xs_vec >= 0.0).all()


def test_ionisation_ratecoeff_matches_eff_ionpot_form() -> None:
    # the rate coefficient is the direct integral of y(E) sigma(E) dE over the shells. For a
    # populated ion that equals Kozma & Fransson 1992 equation 13 with the effective potential.
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=400) as sf:
        sf.add_ionisation(8, 1, n_ion=9e8)
        sf.add_ionisation(8, 2, n_ion=1e8)
        sf.add_ionisation(26, 2, n_ion=2e7)
        sf.solve(depositionratedensity_ev=1e9)
        n_ion_tot = sf.get_n_ion_tot()
        for Z, ion_stage in sf.ionpopdict:
            ratecoeff = sf.get_ionisation_ratecoeff(Z, ion_stage)
            assert ratecoeff > 0.0
            assert math.isclose(ratecoeff, 1e9 / n_ion_tot / sf.get_eff_ionpot(Z, ion_stage), rel_tol=1e-12)


def test_excitation_only_ion_counted() -> None:
    # an ion given an excitation channel but no ionisation channel still spends deposited energy,
    # so its excitation fraction has to appear in the totals
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=500) as sf:
        sf.set_temperature(3000)
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
        sf.set_temperature(3000)
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
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        sf.set_temperature(3000)
        with pytest.raises(ValueError, match="non-negative"):
            sf.add_ion_ltepopexcitation(26, 3, n_ion=-1.0, use_collstrengths=False)


def test_conservation_warning_on_coarse_grid() -> None:
    # the energy fractions of a helium plasma are ~6.5% off unity at npts=100,
    # which must trigger the conservation diagnostic
    x_e = 1e-4
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=100) as sf:
        sf.set_temperature(3000)
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


def _reference_excitation_fill(
    sf: pynonthermal.SpencerFanoSolver,
    levelnumberdensity: float,
    xs_vec: npt.NDArray[np.float64],
    epsilon_trans_ev: float,
) -> npt.NDArray[np.float64]:
    # straightforward row-by-row construction of one transition's contribution: row i covers k
    # full-width bins from max(i, threshold index) plus a partial final bin of weight frac,
    # truncated at the top of the grid where every remaining bin is full width. Kept independent
    # of the production code so that a regression in the banded fill cannot hide here.
    engrid = sf.engrid
    npts = len(engrid)
    deltaen = sf.deltaen
    contribution = np.zeros((npts, npts))
    vec = levelnumberdensity * deltaen * xs_vec
    xsstartindex = sf.get_energyindex_lteq(en_ev=epsilon_trans_ev)
    k = int(epsilon_trans_ev / deltaen)
    frac = epsilon_trans_ev / deltaen - k
    for i in range(npts):
        startindex = max(i, xsstartindex)
        if i + k < npts:
            contribution[i, startindex : i + k] += vec[startindex : i + k]
            contribution[i, i + k] += vec[i + k] * frac
        else:
            contribution[i, startindex:] += vec[startindex:]
    return contribution


def test_excitation_band_fill_matches_rowwise() -> None:
    # add_excitation writes each transition as one windowed band plus a partial-bin diagonal
    # (every row repeats the same vec[j] values over the overlap of its window with the next
    # row's). The result must be bit-identical to the row-by-row fill, including the rows
    # truncated by the excitation threshold and the rows clipped at the top of the grid.
    rng = np.random.default_rng(42)
    npts = 157
    deltaen = (300.0 - 1.0) / (npts - 1)  # approximate spacing, only used to place the cases below
    cases = [
        (1.0, 300.0, 0.5 * deltaen),  # narrower than one grid cell: band width zero, partial bins only
        (1.0, 300.0, 5 * deltaen),  # within rounding of an exact multiple of the grid spacing
        (1.0, 300.0, 13.3),  # a typical transition energy, not aligned with the grid
        (1.0, 300.0, 299.0),  # band nearly as wide as the whole grid
        (1.0, 300.0, 300.0),  # at the top of the grid: every row's window is clipped
        (200.0, 300.0, 250.0),  # band wider than the whole grid: every row is clipped and truncated
    ]
    for emin, emax, epsilon_trans_ev in cases:
        with pynonthermal.SpencerFanoSolver(emin_ev=emin, emax_ev=emax, npts=npts) as sf:
            # nonzero cross sections below threshold too: the caller is not required to zero them,
            # and the partial-bin term is written for every row
            xs_vec = rng.random(npts) * 1e-16
            sf.add_excitation(8, 2, levelnumberdensity=1e5, xs_vec=xs_vec, epsilon_trans_ev=epsilon_trans_ev)
            expected = _reference_excitation_fill(sf, 1e5, xs_vec, epsilon_trans_ev)
            assert sf.sfmatrix.tobytes() == expected.tobytes(), f"mismatch for {epsilon_trans_ev=}"


def _reference_ionisation_fill(
    sf: pynonthermal.SpencerFanoSolver, n_ion: float, channel: pynonthermal.IonisationChannel
) -> npt.NDArray[np.float64]:
    # the masked full-tail construction of one channel's matrix contribution that predates the
    # forward-only cut pointers, kept verbatim as the reference the slice fill must reproduce
    engrid = sf.engrid
    npts = len(engrid)
    deltaen = sf.deltaen
    ionpot_ev = channel.ionpot_ev
    J = channel.J_ev
    ar_xs_array = channel.xs_grid
    xsstartindex = 0 if ionpot_ev <= engrid[0] else sf.get_energyindex_gteq(en_ev=ionpot_ev)
    atan_epsilon = np.arctan((engrid - ionpot_ev) / 2.0 / J)
    with np.errstate(divide="ignore", invalid="ignore"):
        prefactors = np.divide(
            n_ion * ar_xs_array * deltaen, atan_epsilon, out=np.zeros(npts), where=atan_epsilon != 0.0
        )
    epsilon_uppers = np.minimum((engrid + ionpot_ev) / 2, engrid)
    int_eps_uppers = np.arctan((epsilon_uppers - ionpot_ev) / J)
    epsilon_lowers1 = np.maximum(engrid - engrid[0], ionpot_ev)
    int_eps_lowers1 = np.arctan((epsilon_lowers1 - ionpot_ev) / J)
    contribution = np.zeros((npts, npts))
    for i, en in enumerate(engrid):
        jstart = max(i, xsstartindex)
        contribution[i, jstart:] += np.where(
            epsilon_lowers1[jstart - i : npts - i] <= epsilon_uppers[jstart:],
            prefactors[jstart:] * (int_eps_uppers[jstart:] - int_eps_lowers1[jstart - i : npts - i]),
            0.0,
        )
        if 2 * en + ionpot_ev < engrid[-1] + deltaen:
            secondintegralstartindex = sf.get_energyindex_lteq(float(2 * en + ionpot_ev))
            epsilon_lower2 = en + ionpot_ev
            int_eps_lower2 = math.atan((epsilon_lower2 - ionpot_ev) / J)
            contribution[i, secondintegralstartindex:] -= np.where(
                epsilon_lower2 <= epsilon_uppers[secondintegralstartindex:],
                prefactors[secondintegralstartindex:] * (int_eps_uppers[secondintegralstartindex:] - int_eps_lower2),
                0.0,
            )
    return contribution


def test_ltepop_excitation_grouped_fill() -> None:
    # add_ion_ltepopexcitation writes the matrix once per distinct band width, with the vectors
    # of same-width transitions pre-summed. Rebuilding from the stored per-transition data must
    # agree to within summation-order rounding.
    npts = 400
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=npts) as sf:
        sf.set_temperature(3000)
        sf.add_ion_ltepopexcitation(2, 1, n_ion=1.0, use_collstrengths=False)
        assert len(sf.excitationlists[(2, 1)]) > 100  # the grouping must actually see many transitions
        expected = np.zeros((npts, npts))
        for trans in sf.excitationlists[(2, 1)].values():
            expected += _reference_excitation_fill(sf, trans.levelnumberdensity, trans.xs_vec, trans.epsilon_trans_ev)
        assert np.allclose(sf.sfmatrix, expected, rtol=1e-12, atol=0.0)


def test_ionisation_fill_matches_masked_reference() -> None:
    # _add_ionisation_channel_to_matrix writes only each row's non-empty integral range, located by
    # forward-only cut pointers that evaluate the same comparisons the per-row np.where masks
    # used. The resulting matrix must be bit-identical to the masked full-tail construction.
    for emin, emax, npts, Z, ion_stage in [
        (1.0, 300.0, 193, 8, 2),
        (7.9, 16000.0, 217, 26, 2),  # emin well above 1 eV and a heavy ion with many shells
        (1.0, 3000.0, 149, 2, 1),  # helium: J takes the Opal et al. 1971 special-case value
    ]:
        with pynonthermal.SpencerFanoSolver(emin_ev=emin, emax_ev=emax, npts=npts) as sf:
            n_ion = 1e5
            sf.add_ionisation(Z, ion_stage, n_ion=n_ion)
            expected = np.zeros((npts, npts))
            for channel in sf._ionisation_channels[(Z, ion_stage)]:
                expected += _reference_ionisation_fill(sf, n_ion, channel)
            assert sf.sfmatrix.tobytes() == expected.tobytes(), f"mismatch for {Z=} {ion_stage=}"


def test_solve_upper_triangular_diag_add() -> None:
    # solve() passes the free-electron loss function as a separate diagonal vector instead of
    # copying the whole matrix to add it, so the diag_add path must reproduce the added-in-place
    # result exactly
    rng = np.random.default_rng(7)
    n = 40
    a = np.triu(rng.random((n, n))) + np.diag(np.full(n, 2.0))
    b = rng.random(n)
    d = rng.random(n)
    expected = spencerfano.solve_upper_triangular(a + np.diag(d), b)
    assert np.array_equal(spencerfano.solve_upper_triangular(a, b, diag_add=d), expected)

    # the singularity check applies to the summed diagonal: a zero on the matrix diagonal is
    # fine when diag_add makes the sum non-zero...
    a_zerodiag = a.copy()
    a_zerodiag[3, 3] = 0.0
    assert np.isfinite(spencerfano.solve_upper_triangular(a_zerodiag, b, diag_add=d)).all()

    # ...and a cancellation to exactly zero is singular
    d_cancel = d.copy()
    d_cancel[3] = -a[3, 3]
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        spencerfano.solve_upper_triangular(a, b, diag_add=d_cancel)

    with pytest.raises(ValueError, match="finite"):
        spencerfano.solve_upper_triangular(a, b, diag_add=np.array([math.nan] + [0.0] * (n - 1)))


def _bethe_xs_grid(
    engrid: npt.NDArray[np.float64], ionpot_ev: float, scale_cm2_ev2: float = 3e-13
) -> npt.NDArray[np.float64]:
    # a plain Bethe-like cross section [cm^2] on the solver energy grid, which stands for a custom
    # cross section. It is zero at and below the ionisation potential, as every channel must be.
    out = np.zeros_like(engrid)
    above = engrid > ionpot_ev
    out[above] = scale_cm2_ev2 * np.log(engrid[above] / ionpot_ev) / (engrid[above] * ionpot_ev)
    return out


def test_custom_ionisation_channel_matches_the_builtin_path() -> None:
    # a custom channel must take exactly the same path as a shell of the built-in table. Adding the
    # built-in cross sections back one call at a time must give the same matrix, bit for bit, and
    # the same energy fractions after the solve.
    Z, ion_stage, n_ion = 8, 2, 1e8
    shells = (
        pynonthermal.collion.read_colliondata()
        .filter((pl.col("Z") == Z) & (pl.col("ion_stage") == ion_stage))
        .to_dicts()
    )
    assert len(shells) > 1  # the ion must have more than one shell for this to mean anything

    with (
        pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf_custom,
        pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf_builtin,
    ):
        sf_builtin.add_ionisation(Z, ion_stage, n_ion=n_ion)
        for shell in shells:
            sf_custom.add_ionisation_channel(
                Z,
                ion_stage,
                n_ion=n_ion,
                ionpot_ev=float(shell["ionpot_ev"]),
                xs_vec=pynonthermal.collion.get_arxs_array_shell(sf_custom.engrid, shell),
            )

        assert sf_custom.sfmatrix.tobytes() == sf_builtin.sfmatrix.tobytes()

        for sf in (sf_custom, sf_builtin):
            sf.solve(depositionratedensity_ev=1e8)

        assert math.isclose(
            sf_custom.get_frac_ionisation_ion(Z, ion_stage), sf_builtin.get_frac_ionisation_ion(Z, ion_stage)
        )
        # frac_heating differs only through calculate_N_e, which needs the cross section between
        # the grid points and interpolates the array there instead of using the exact formula
        assert math.isclose(sf_custom.get_frac_heating(), sf_builtin.get_frac_heating(), rel_tol=1e-5)


def test_custom_ionisation_channel_conserves_energy() -> None:
    # the whole custom path, from the matrix fill to calculate_N_e, must still put every
    # deposited eV into heating, excitation, or ionisation
    ionpot_ev = 35.0
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=2000) as sf:
        sf.add_ionisation_channel(8, 2, n_ion=1e8, ionpot_ev=ionpot_ev, xs_vec=_bethe_xs_grid(sf.engrid, ionpot_ev))
        sf.solve(depositionratedensity_ev=1e8)

        assert sf.get_frac_ionisation_tot() > 0.1  # the channel must carry a real share
        assert math.isclose(sf.get_frac_sum(), 1.0, abs_tol=0.01)

        # calculate_N_e integrates over a domain narrower than one grid cell just above the
        # ionisation potential, so the channel must give values between the grid points
        channel = sf._ionisation_channels[(8, 2)][0]
        off_grid = np.array([ionpot_ev * 1.001, ionpot_ev * 1.002, ionpot_ev * 1.003])
        assert not np.isin(off_grid, sf.engrid).any()
        assert (channel.xs(off_grid) > 0.0).all()
        # and it stays at zero below its own ionisation potential
        assert not channel.xs(np.array([ionpot_ev * 0.999, ionpot_ev])).any()


def test_custom_ionisation_channel_adds_to_the_builtin_shells() -> None:
    # add_ionisation() and add_ionisation_channel() may both be called for one ion
    Z, ion_stage, n_ion = 8, 2, 1e8
    with (
        pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf_builtin,
        pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf_extra,
    ):
        for sf in (sf_builtin, sf_extra):
            sf.add_ionisation(Z, ion_stage, n_ion=n_ion)
        sf_extra.add_ionisation_channel(
            Z,
            ion_stage,
            n_ion=n_ion,
            ionpot_ev=120.0,
            xs_vec=_bethe_xs_grid(sf_extra.engrid, 120.0),
            channelkey="my inner shell",
        )

        assert (
            len(sf_extra._ionisation_channels[(Z, ion_stage)])
            == len(sf_builtin._ionisation_channels[(Z, ion_stage)]) + 1
        )
        for sf in (sf_builtin, sf_extra):
            sf.solve(depositionratedensity_ev=1e8)
        assert sf_extra.get_frac_ionisation_ion(Z, ion_stage) > sf_builtin.get_frac_ionisation_ion(Z, ion_stage)


def test_custom_ionisation_channel_validation() -> None:
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=200) as sf:
        good = _bethe_xs_grid(sf.engrid, 35.0)

        # the cross section must be non-negative, finite, and on the whole grid
        with pytest.raises(ValueError, match="non-negative"):
            sf.add_ionisation_channel(8, 2, n_ion=1e8, ionpot_ev=35.0, xs_vec=-good)
        with pytest.raises(ValueError, match="finite"):
            sf.add_ionisation_channel(8, 2, n_ion=1e8, ionpot_ev=35.0, xs_vec=np.full_like(good, np.inf))
        with pytest.raises(ValueError, match="one value for each"):
            sf.add_ionisation_channel(8, 2, n_ion=1e8, ionpot_ev=35.0, xs_vec=good[:-1])

        # a cross section below the ionisation potential would count towards the ionisation
        # fraction without a matching loss term in the matrix
        with pytest.raises(ValueError, match="zero at and below its ionisation potential"):
            sf.add_ionisation_channel(8, 2, n_ion=1e8, ionpot_ev=35.0, xs_vec=np.full_like(good, 1e-18))

        # the ionisation potential must lie inside the energy grid
        with pytest.raises(ValueError, match="below emin_ev"):
            sf.add_ionisation_channel(8, 2, n_ion=1e8, ionpot_ev=0.5, xs_vec=_bethe_xs_grid(sf.engrid, 0.5))
        with pytest.raises(ValueError, match="above the top of the energy grid"):
            sf.add_ionisation_channel(8, 2, n_ion=1e8, ionpot_ev=5000.0, xs_vec=np.zeros_like(good))
        with pytest.raises(ValueError, match="ionpot_ev must be greater than zero"):
            sf.add_ionisation_channel(8, 2, n_ion=1e8, ionpot_ev=-1.0, xs_vec=np.zeros_like(good))

        # a nan or infinite population used to reach ionpopdict and make every result nan
        for bad in (float("nan"), float("inf")):
            with pytest.raises(ValueError, match="non-negative and finite"):
                sf.add_ionisation_channel(8, 2, n_ion=bad, ionpot_ev=35.0, xs_vec=good)
            with pytest.raises(ValueError, match="non-negative and finite"):
                sf.add_ionisation(8, 2, n_ion=bad)

        # ion_stage below one gives a negative charge, which only a bare assert used to catch
        with pytest.raises(ValueError, match="ion_stage must be at least 1"):
            sf.add_ionisation_channel(8, 0, n_ion=1e8, ionpot_ev=35.0, xs_vec=good)
        with pytest.raises(ValueError, match="Z must be at least 1"):
            sf.add_ionisation_channel(0, 1, n_ion=1e8, ionpot_ev=35.0, xs_vec=good)

        assert not sf.ionpopdict
        assert not sf._ionisation_channels

        # a zero population adds no channel, but the cross section is still checked
        with pytest.raises(ValueError, match="non-negative"):
            sf.add_ionisation_channel(8, 2, n_ion=0.0, ionpot_ev=35.0, xs_vec=-good)
        sf.add_ionisation_channel(8, 2, n_ion=0.0, ionpot_ev=35.0, xs_vec=good)
        assert (8, 2) not in sf.ionpopdict
        assert not sf.sfmatrix.any()

        # an omitted channelkey gets an automatic index, and a duplicate key is rejected
        sf.add_ionisation_channel(8, 2, n_ion=1e8, ionpot_ev=35.0, xs_vec=good)
        assert sf._ionisation_channels[(8, 2)][0].key == 0
        with pytest.raises(ValueError, match="already added"):
            sf.add_ionisation_channel(8, 2, n_ion=1e8, ionpot_ev=35.0, xs_vec=good, channelkey=0)
        # the population must agree with the one an earlier call for this ion gave
        with pytest.raises(ValueError, match="different populations"):
            sf.add_ionisation_channel(8, 2, n_ion=2e8, ionpot_ev=42.0, xs_vec=_bethe_xs_grid(sf.engrid, 42.0))

        sf.solve(depositionratedensity_ev=1e8)
        with pytest.raises(RuntimeError, match="after solving"):
            sf.add_ionisation_channel(8, 2, n_ion=1e8, ionpot_ev=60.0, xs_vec=good)


def test_add_ionisation_is_atomic() -> None:
    # a key clash inside add_ionisation must leave the solver exactly as it was, so that the
    # caller can correct the key and try again
    Z, ion_stage = 8, 2
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=150) as sf:
        # "n 2 l 0" is the second built-in shell of O II, so the clash comes after the first
        sf.add_ionisation_channel(
            Z,
            ion_stage,
            n_ion=1e8,
            ionpot_ev=42.6,
            xs_vec=_bethe_xs_grid(sf.engrid, 42.6),
            channelkey="n 2 l 0",
        )
        matrix_before = sf.sfmatrix.copy()

        with pytest.raises(ValueError, match="already added"):
            sf.add_ionisation(Z, ion_stage, n_ion=1e8)

        assert np.array_equal(sf.sfmatrix, matrix_before)
        assert [channel.key for channel in sf._ionisation_channels[(Z, ion_stage)]] == ["n 2 l 0"]

        # the ion is not poisoned: a call with a free key still works
        sf.add_ionisation_channel(Z, ion_stage, n_ion=1e8, ionpot_ev=35.1, xs_vec=_bethe_xs_grid(sf.engrid, 35.1))
        assert len(sf._ionisation_channels[(Z, ion_stage)]) == 2


def test_cross_sections_are_copied_from_the_caller() -> None:
    # the solver keeps a read-only copy of every cross section, so a later write by the caller
    # cannot silently change the physics it already recorded
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf:
        xs_vec = np.where(sf.engrid >= 20.0, 1e-16, 0.0)
        sf.add_excitation(8, 2, levelnumberdensity=1e8, xs_vec=xs_vec, epsilon_trans_ev=20.0)
        xs_vec[:] = 0.0
        assert sf.excitationlists[(8, 2)][0].xs_vec.max() > 0.0
        assert not sf.excitationlists[(8, 2)][0].xs_vec.flags.writeable

        ion_xs = _bethe_xs_grid(sf.engrid, 35.0)
        sf.add_ionisation_channel(8, 2, n_ion=1e8, ionpot_ev=35.0, xs_vec=ion_xs)
        ion_xs[:] = 0.0
        channel = sf._ionisation_channels[(8, 2)][0]
        assert channel.xs_grid.max() > 0.0
        assert not channel.xs_grid.flags.writeable
        # the interpolation the channel uses off the grid kept the copy too
        assert channel.xs(np.array([100.5]))[0] > 0.0

    # the built-in shells of one ion must not share an array
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf_builtin:
        sf_builtin.add_ionisation(8, 2, n_ion=1e8)
        channels = sf_builtin._ionisation_channels[(8, 2)]
        assert len(channels) > 1
        assert not np.shares_memory(channels[0].xs_grid, channels[1].xs_grid)


def test_engrid_is_read_only() -> None:
    # deltaen is derived from engrid once, so a write in place would leave the two inconsistent
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=100) as sf:
        assert not sf.engrid.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            sf.engrid[0] = 2.0
        assert sf.engrid[0] == 1.0


def test_excitation_xs_must_be_finite() -> None:
    # an infinite cross section used to pass the non-negative test and put inf into the matrix
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=100) as sf:
        xs_vec = np.zeros(100)
        xs_vec[50:] = np.inf
        with pytest.raises(ValueError, match="finite"):
            sf.add_excitation(8, 2, levelnumberdensity=1e8, xs_vec=xs_vec, epsilon_trans_ev=20.0)
        assert not sf.sfmatrix.any()

        # a cross section of the wrong shape is named, rather than reaching the matrix band
        with pytest.raises(ValueError, match="one value for each"):
            sf.add_excitation(8, 2, levelnumberdensity=1e8, xs_vec=np.zeros((100, 1)), epsilon_trans_ev=20.0)
        assert not sf.excitationlists.get((8, 2))


def test_every_builtin_ion_builds_channels() -> None:
    # the checks in IonisationChannel.from_xs must accept every row of the built-in tables, on both
    # the Arnaud & Rothenflug fit branch and the Lotz branch
    engrid = np.linspace(1.0, 3000.0, num=64, dtype=np.float64)
    for collionfilename in ("collion.txt", "collion-AR1985.txt"):
        dfcollion = pynonthermal.collion.read_colliondata(collionfilename)
        n_lotz = 0
        n_fit = 0
        for Z, ion_stage in dfcollion.select(["Z", "ion_stage"]).unique().iter_rows():
            channels = pynonthermal.collion.get_ion_ionisation_channels(dfcollion, Z, ion_stage, engrid)
            assert channels
            for channel in channels:
                if str(channel.key).startswith("Lotz"):
                    n_lotz += 1
                else:
                    n_fit += 1
        assert n_lotz > 0
        assert n_fit > 0
