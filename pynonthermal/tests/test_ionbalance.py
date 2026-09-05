"""Tests of the ion populations from an ionisation/recombination balance or from the Saha equation.

Not marked as benchmarks: these check the balance algebra, the consistency of the balanced solver
with a solver that gets the same populations directly, and the error paths.
"""

import math
import warnings

import numpy as np
import pytest

import pynonthermal

# illustrative recombination rate coefficients [cm^3 s^-1] keyed by the recombining ion stage
OXYGEN_ALPHAS = {2: 3e-13, 3: 3e-12, 4: 1e-11}
HELIUM_ALPHAS = {2: 4e-13, 3: 2e-12}


def test_saha_constant() -> None:
    # (2 pi m_e k_B / h^2)^(3/2) = 2.4147e15 cm^-3 K^-3/2
    assert math.isclose(pynonthermal.ionbalance.SAHA_CONST, 2.4147e15, rel_tol=1e-4)


def test_saha_factor() -> None:
    # hydrogen at 10^4 K with U_I = 2 and U_II = 1: 2 * (1/2) * SAHA_CONST * T^1.5 * exp(-13.598 eV / kT)
    T = 1e4
    expected = pynonthermal.ionbalance.SAHA_CONST * T**1.5 * math.exp(-13.598 / (8.617333262145e-5 * T))
    assert math.isclose(pynonthermal.ionbalance.get_saha_factor(T, 13.598, 2.0, 1.0), expected, rel_tol=1e-12)

    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="temperature"):
            pynonthermal.ionbalance.get_saha_factor(bad, 13.598, 2.0, 1.0)
        with pytest.raises(ValueError, match="ionpot_ev"):
            pynonthermal.ionbalance.get_saha_factor(T, bad, 2.0, 1.0)
        with pytest.raises(ValueError, match="partition functions"):
            pynonthermal.ionbalance.get_saha_factor(T, 13.598, bad, 1.0)


def test_ion_fractions() -> None:
    # a well-conditioned case against the direct formula
    n_e = 1e8
    c = [2e8, 5e7]
    r1, r2 = c[0] / n_e, c[1] / n_e
    total = 1.0 + r1 + r1 * r2
    fractions = pynonthermal.ionbalance.get_ion_fractions(c, n_e)
    assert np.allclose(fractions, [1.0 / total, r1 / total, r1 * r2 / total], rtol=1e-14)
    assert math.isclose(sum(fractions), 1.0, rel_tol=1e-14)

    # a zero coefficient makes every higher stage exactly zero
    fractions = pynonthermal.ionbalance.get_ion_fractions([2e8, 0.0, 1e30], n_e)
    assert fractions[2] == 0.0
    assert fractions[3] == 0.0
    assert math.isclose(fractions[0] + fractions[1], 1.0, rel_tol=1e-14)
    assert math.isclose(fractions[1] / fractions[0], 2.0, rel_tol=1e-14)

    # very large and very small coefficients neither overflow nor warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fractions = pynonthermal.ionbalance.get_ion_fractions([1e300, 1e300, 1e300], n_e)
        assert fractions[3] == 1.0
        fractions = pynonthermal.ionbalance.get_ion_fractions([1e-300, 1e-300], n_e)
        assert fractions[0] == 1.0
        assert fractions[1] > 0.0

    with pytest.raises(ValueError, match="n_e"):
        pynonthermal.ionbalance.get_ion_fractions([1.0], 0.0)
    with pytest.raises(ValueError, match="ratio coefficients"):
        pynonthermal.ionbalance.get_ion_fractions([-1.0], n_e)


def test_charge_neutral_n_e_two_stages() -> None:
    # one element with two stages: n_2 n_e = c n_1 and n_e = n_e_fixed + n_2, so
    # n_e^2 - n_e_fixed n_e = c (n_elem - n_e + n_e_fixed) is a quadratic in n_e
    n_elem = 1e10
    # the 1e-130 case has its root at 1e-65, far below the first lower bracket of the bisection
    for c, n_e_fixed in ((1e6, 0.0), (1e12, 0.0), (1e6, 3e7), (1e-20, 0.0), (1e-130, 0.0)):
        b = -(n_e_fixed - c)
        a_c = -c * (n_elem + n_e_fixed)
        n_e_expected = 0.5 * (-b + math.sqrt(b * b - 4 * a_c))
        n_e = pynonthermal.ionbalance.solve_charge_neutral_n_e(n_e_fixed, [(n_elem, (1, 2), [c])])
        assert math.isclose(n_e, n_e_expected, rel_tol=1e-10)
        # the populations at the result are charge neutral
        fractions = pynonthermal.ionbalance.get_ion_fractions([c], n_e)
        assert math.isclose(n_e, n_e_fixed + n_elem * fractions[1], rel_tol=1e-10)

    # a chain that starts above the neutral stage has an exact lower bound: with zero ratios, every ion
    # sits in the lowest stage
    assert math.isclose(
        pynonthermal.ionbalance.solve_charge_neutral_n_e(0.0, [(n_elem, (2, 3), [0.0])]), n_elem, rel_tol=1e-12
    )

    # zero ratios and no fixed ionised ion give no free electrons
    with pytest.raises(ValueError, match="free electron density is zero"):
        pynonthermal.ionbalance.solve_charge_neutral_n_e(0.0, [(n_elem, (1, 2), [0.0])])
    # but a fixed ionised ion carries the result
    assert math.isclose(
        pynonthermal.ionbalance.solve_charge_neutral_n_e(5.0, [(n_elem, (1, 2), [0.0])]), 5.0, rel_tol=1e-12
    )

    with pytest.raises(ValueError, match="ratio coefficients"):
        pynonthermal.ionbalance.solve_charge_neutral_n_e(0.0, [(n_elem, (1, 2, 3), [1.0])])
    with pytest.raises(ValueError, match="n_elem"):
        pynonthermal.ionbalance.solve_charge_neutral_n_e(0.0, [(0.0, (1, 2), [1.0])])


def test_charge_neutral_n_e_two_elements() -> None:
    # the result is charge neutral for two elements together with fixed ions
    elements = [(1e10, (1, 2, 3), [1e9, 1e6]), (2e9, (2, 3), [3e8])]
    n_e_fixed = 4e8
    n_e = pynonthermal.ionbalance.solve_charge_neutral_n_e(n_e_fixed, elements)
    charge = n_e_fixed
    for n_elem, ion_stages, ratio_coeffs in elements:
        fractions = pynonthermal.ionbalance.get_ion_fractions(ratio_coeffs, n_e)
        charge += n_elem * sum((ion_stage - 1) * frac for ion_stage, frac in zip(ion_stages, fractions, strict=True))
    assert math.isclose(n_e, charge, rel_tol=1e-10)


def check_balance_identity(sf: pynonthermal.SpencerFanoSolver, Z: int, alphas: dict[int, float], tol: float) -> None:
    # n_i Gamma_i = n_{i+1} n_e alpha_{i+1} for every pair of adjacent stages
    n_e = sf.get_n_e()
    for upper, alpha in alphas.items():
        rate_ionisation = sf.ionpopdict[(Z, upper - 1)] * sf.get_ionisation_ratecoeff(Z, upper - 1)
        rate_recombination = sf.ionpopdict[(Z, upper)] * n_e * alpha
        assert math.isclose(rate_ionisation, rate_recombination, rel_tol=tol)


def build_direct_solver(
    sf_balanced: pynonthermal.SpencerFanoSolver, Z: int, excitation_stages: tuple[int, ...], temperature: float
) -> pynonthermal.SpencerFanoSolver:
    # a solver that gets the converged populations of the balanced solver as fixed populations
    sf = pynonthermal.SpencerFanoSolver(
        emin_ev=sf_balanced.engrid[0], emax_ev=sf_balanced.engrid[-1], npts=len(sf_balanced.engrid)
    )
    for (Z_ion, ion_stage), n_ion in sf_balanced.ionpopdict.items():
        if Z_ion == Z and ion_stage <= Z:
            sf.add_ionisation(Z, ion_stage, n_ion)
    for ion_stage in excitation_stages:
        sf.add_ion_ltepopexcitation(
            Z, ion_stage, n_ion=sf_balanced.ionpopdict[(Z, ion_stage)], temperature=temperature, use_collstrengths=False
        )
    return sf


def test_recombination_balance_oxygen() -> None:
    n_oxygen = 1e10
    deposition = 2950.49 * n_oxygen * 1e-5
    balance_tol = 1e-5
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=400) as sf:
        sf.add_element_ionbalance(8, n_oxygen, OXYGEN_ALPHAS)
        # the provisional populations are equal fractions, and the top stage O IV has channels too
        assert all(sf.ionpopdict[(8, ion_stage)] == n_oxygen / 4 for ion_stage in (1, 2, 3, 4))
        assert math.isclose(sf.get_n_ion_tot(), n_oxygen, rel_tol=1e-12)
        assert len(sf._ionisation_channels[(8, 4)]) > 0
        for ion_stage in (1, 2, 3):
            sf.add_ion_ltepopexcitation(8, ion_stage, n_ion=None, temperature=6000, use_collstrengths=False)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            sf.solve(depositionratedensity_ev=deposition, balance_tol=balance_tol)

        assert 1 < sf.balance_iterations < 100
        assert math.isclose(sf.get_n_ion_tot(), n_oxygen, rel_tol=1e-12)
        assert math.isclose(sf.get_n_e(), sf.calculate_free_electron_density(), rel_tol=1e-12)
        assert math.isclose(sf.get_frac_sum(), 1.0, abs_tol=0.02)
        check_balance_identity(sf, 8, OXYGEN_ALPHAS, tol=2 * balance_tol)
        fractions_low = {ion_stage: sf.ionpopdict[(8, ion_stage)] / n_oxygen for ion_stage in (1, 2, 3, 4)}
        assert math.isclose(sum(fractions_low.values()), 1.0, rel_tol=1e-12)

        # the balanced solver has the same matrix and solution as a solver with the converged
        # populations given directly. The delta updates leave only rounding differences.
        sf_direct = build_direct_solver(sf, 8, (1, 2, 3), 6000)
        sf_direct.solve(depositionratedensity_ev=deposition)
        assert np.allclose(sf.sfmatrix, sf_direct.sfmatrix, rtol=1e-9, atol=1e-12 * np.abs(sf_direct.sfmatrix).max())
        assert np.allclose(sf.yvec, sf_direct.yvec, rtol=1e-9)
        for ion_stage in (1, 2, 3):
            assert math.isclose(
                sf.get_ionisation_ratecoeff(8, ion_stage),
                sf_direct.get_ionisation_ratecoeff(8, ion_stage),
                rel_tol=1e-9,
            )
            assert sf.excitationlists[(8, ion_stage)].keys() == sf_direct.excitationlists[(8, ion_stage)].keys()
            transitionkey = next(iter(sf.excitationlists[(8, ion_stage)]))
            assert math.isclose(
                sf.get_excitation_ratecoeff(8, ion_stage, transitionkey),
                sf_direct.get_excitation_ratecoeff(8, ion_stage, transitionkey),
                rel_tol=1e-9,
            )
            trans_balanced = sf.excitationlists[(8, ion_stage)][transitionkey]
            trans_direct = sf_direct.excitationlists[(8, ion_stage)][transitionkey]
            assert math.isclose(trans_balanced.levelnumberdensity, trans_direct.levelnumberdensity, rel_tol=1e-12)

        # a second solve at a higher deposition rate starts from the converged rates, reconverges,
        # and ionises the element further
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            sf.solve(depositionratedensity_ev=deposition * 10, balance_tol=balance_tol)
        check_balance_identity(sf, 8, OXYGEN_ALPHAS, tol=2 * balance_tol)
        assert sf.ionpopdict[(8, 1)] < fractions_low[1] * n_oxygen
        assert sf.get_n_e() > sum(fractions_low[ion_stage] * (ion_stage - 1) for ion_stage in (2, 3, 4)) * n_oxygen

        # additions are locked after solving
        with pytest.raises(RuntimeError):
            sf.add_element_ionbalance(26, 1.0, {2: 1e-12})
        with pytest.raises(RuntimeError):
            sf.add_element_saha(26, 1.0, 5000.0, [1, 2])


def test_recombination_balance_helium_bare_nucleus() -> None:
    # the chain can end at the bare nucleus, which has no ionisation channel
    n_helium = 1e8
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf:
        sf.add_element_ionbalance(2, n_helium, HELIUM_ALPHAS)
        assert (2, 3) not in sf._ionisation_channels
        sf.solve(depositionratedensity_ev=1e8)
        assert sf.get_ionisation_ratecoeff(2, 3) == 0.0
        assert sf.get_ionisation_ratecoeff(2, 2) > 0.0
        assert sf.ionpopdict[(2, 3)] > 0.0
        check_balance_identity(sf, 2, HELIUM_ALPHAS, tol=2e-4)
        assert math.isclose(sf.get_frac_sum(), 1.0, abs_tol=0.02)


def test_top_stage_leak_warning() -> None:
    # a chain that stops at a stage with a large ionisation rate gets a warning
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf:
        sf.add_element_ionbalance(8, 1e10, {2: 3e-13})
        with pytest.warns(UserWarning, match="top stage 2 of Z=8"):
            sf.solve(depositionratedensity_ev=1e12)


def test_saha_populations() -> None:
    n_oxygen = 1e10
    temperature = 12000.0
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf:
        sf.add_element_saha(8, n_oxygen, temperature, [1, 2, 3])
        sf.add_ion_ltepopexcitation(8, 1, n_ion=None, temperature=temperature, use_collstrengths=False)
        sf.add_ion_ltepopexcitation(8, 2, n_ion=None, temperature=temperature, use_collstrengths=False)
        sf.solve(depositionratedensity_ev=1e8)
        # the Saha ratios do not depend on the solution, so one pass is enough
        assert sf.balance_iterations == 1

        # the populations satisfy the Saha equation with the partition functions of the level data
        adata = sf.adata_polars
        assert adata is not None
        ionpots = pynonthermal.collion.get_nist_ionisation_energies_ev()
        n_e = sf.get_n_e()
        assert math.isclose(n_e, sf.calculate_free_electron_density(), rel_tol=1e-12)
        assert math.isclose(n_e, sf.ionpopdict[(8, 2)] + 2 * sf.ionpopdict[(8, 3)], rel_tol=1e-12)
        for lower in (1, 2):
            partfuncs = [at_get_lte_partfunc(adata, 8, ion_stage, temperature) for ion_stage in (lower, lower + 1)]
            saha_factor = pynonthermal.ionbalance.get_saha_factor(
                temperature, ionpots[(8, lower)], partfuncs[0], partfuncs[1]
            )
            ratio = sf.ionpopdict[(8, lower + 1)] * n_e / sf.ionpopdict[(8, lower)]
            assert math.isclose(ratio, saha_factor, rel_tol=1e-9)

        # the matrix agrees with a solver that gets the populations directly
        sf_direct = build_direct_solver(sf, 8, (1, 2), temperature)
        sf_direct.solve(depositionratedensity_ev=1e8)
        assert np.allclose(sf.sfmatrix, sf_direct.sfmatrix, rtol=1e-9, atol=1e-12 * np.abs(sf_direct.sfmatrix).max())
        assert np.allclose(sf.yvec, sf_direct.yvec, rtol=1e-9)
        assert math.isclose(sf.get_frac_sum(), 1.0, abs_tol=0.02)

    # given partition functions replace the level data, and the bare nucleus gets one
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf:
        sf.add_element_saha(2, 1e8, 30000.0, [1, 2, 3], partfuncs={1: 1.0, 2: 2.0})
        sf.solve(depositionratedensity_ev=1e8)
        n_e = sf.get_n_e()
        saha_factor = pynonthermal.ionbalance.get_saha_factor(30000.0, ionpots[(2, 2)], 2.0, 1.0)
        assert math.isclose(sf.ionpopdict[(2, 3)] * n_e / sf.ionpopdict[(2, 2)], saha_factor, rel_tol=1e-9)


def at_get_lte_partfunc(adata: object, Z: int, ion_stage: int, temperature: float) -> float:
    import artistools as at  # noqa: PLC0415
    import polars as pl  # noqa: PLC0415

    assert isinstance(adata, pl.DataFrame)
    ion = adata.filter(pl.col("Z") == Z).filter(pl.col("ion_stage") == ion_stage)
    return at.transitions.get_lte_partfunc(ion["levels"].item(), temperature)


def test_mixed_fixed_saha_and_recombination() -> None:
    # fixed ions, a Saha element, and a recombination-balance element in one solver
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf:
        sf.add_ionisation(26, 2, n_ion=1e8)
        sf.add_ionisation(26, 3, n_ion=2e8)
        sf.add_element_saha(8, 1e9, 12000.0, [1, 2, 3])
        sf.add_element_ionbalance(2, 1e9, HELIUM_ALPHAS)
        sf.solve(depositionratedensity_ev=1e10)

        n_e = sf.get_n_e()
        charge = (
            1e8 + 2 * 2e8 + sum((ion_stage - 1) * n_ion for (Z, ion_stage), n_ion in sf.ionpopdict.items() if Z != 26)
        )
        assert math.isclose(n_e, charge, rel_tol=1e-12)
        assert math.isclose(sf.ionpopdict[(8, 1)] + sf.ionpopdict[(8, 2)] + sf.ionpopdict[(8, 3)], 1e9, rel_tol=1e-12)
        check_balance_identity(sf, 2, HELIUM_ALPHAS, tol=2e-4)
        assert sf.ionpopdict[(26, 2)] == 1e8
        assert math.isclose(sf.get_frac_sum(), 1.0, abs_tol=0.02)


def test_override_n_e_with_balance() -> None:
    # the balance uses the override density, and get_n_e() keeps it
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf:
        sf.add_element_ionbalance(2, 1e8, HELIUM_ALPHAS)
        sf.solve(depositionratedensity_ev=1e8, override_n_e=1e5)
        assert sf.get_n_e() == 1e5
        check_balance_identity(sf, 2, HELIUM_ALPHAS, tol=2e-4)
        # the populations are not charge neutral with the override
        assert not math.isclose(sf.calculate_free_electron_density(), 1e5, rel_tol=0.1)


def test_heating_only_approximation_with_balance() -> None:
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300, heating_only_approximation=True) as sf:
        sf.add_element_ionbalance(2, 1e8, HELIUM_ALPHAS)
        sf.add_ion_ltepopexcitation(2, 1, n_ion=None, use_collstrengths=False)
        assert not sf.sfmatrix.any()
        sf.solve(depositionratedensity_ev=1e8)
        assert not sf.sfmatrix.any()
        check_balance_identity(sf, 2, HELIUM_ALPHAS, tol=2e-4)


def test_zero_population_stage_has_a_rate_coefficient() -> None:
    # a stage that the balance leaves empty still has a positive ionisation rate coefficient
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf:
        sf.add_element_saha(8, 1e10, 500.0, [1, 2, 3, 4])
        sf.solve(depositionratedensity_ev=1e8, override_n_e=1e8)
        # at 500 K the Boltzmann factor of the 35 eV O II ionisation potential underflows to zero
        assert sf.ionpopdict[(8, 2)] > 0.0
        assert sf.ionpopdict[(8, 3)] == 0.0
        assert sf.ionpopdict[(8, 4)] == 0.0
        assert sf.get_ionisation_ratecoeff(8, 3) > 0.0
        assert sf.get_ionisation_ratecoeff(8, 4) > 0.0
        assert sf.get_eff_ionpot(8, 4) == math.inf


def test_balance_not_converged() -> None:
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf:
        sf.add_element_ionbalance(2, 1e8, HELIUM_ALPHAS)
        with pytest.raises(RuntimeError, match="did not converge in 1 iteration"):
            sf.solve(depositionratedensity_ev=1e8, balance_maxiter=1)
        # the solver is not marked as solved after a failed balance
        with pytest.raises(RuntimeError, match="must be solved first"):
            sf.get_frac_heating()
        with pytest.raises(ValueError, match="balance_tol"):
            sf.solve(depositionratedensity_ev=1e8, balance_tol=0.0)
        with pytest.raises(ValueError, match="balance_maxiter"):
            sf.solve(depositionratedensity_ev=1e8, balance_maxiter=0)


def test_balanced_element_input_validation() -> None:
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=300) as sf:
        with pytest.raises(ValueError, match="at least one recombination"):
            sf.add_element_ionbalance(8, 1e10, {})
        with pytest.raises(ValueError, match="contiguous"):
            sf.add_element_ionbalance(8, 1e10, {2: 1e-12, 4: 1e-12})
        with pytest.raises(ValueError, match="between 1 and 9"):
            sf.add_element_ionbalance(8, 1e10, {10: 1e-12})
        with pytest.raises(ValueError, match="between 1 and 9"):
            sf.add_element_ionbalance(8, 1e10, {1: 1e-12})
        for bad in (0.0, -1e-12, math.nan, math.inf):
            with pytest.raises(ValueError, match="greater than zero"):
                sf.add_element_ionbalance(8, 1e10, {2: bad})
        for bad in (0.0, -1.0, math.nan, math.inf):
            with pytest.raises(ValueError, match="n_elem"):
                sf.add_element_ionbalance(8, bad, {2: 1e-12})
        with pytest.raises(ValueError, match="Z must be at least 1"):
            sf.add_element_ionbalance(0, 1e10, {2: 1e-12})

        with pytest.raises(ValueError, match="at least two contiguous"):
            sf.add_element_saha(8, 1e10, 5000.0, [2])
        with pytest.raises(ValueError, match="at least two contiguous"):
            sf.add_element_saha(8, 1e10, 5000.0, [1, 3])
        with pytest.raises(ValueError, match="temperature"):
            sf.add_element_saha(8, 1e10, 0.0, [1, 2])
        # Ba has no level data in the internal database, so its partition functions must be given
        with pytest.raises(ValueError, match="No level data for Z=56 ion_stage 1"):
            sf.add_element_saha(56, 1e10, 5000.0, [1, 2])
        with pytest.raises(ValueError, match="partition function of Z=56 ion_stage 2"):
            sf.add_element_saha(56, 1e10, 5000.0, [1, 2], partfuncs={1: 1.0, 2: 0.0})

        # every rejected call leaves the solver unchanged
        assert not sf.ionpopdict
        assert not sf._balanced_elements
        assert not sf.sfmatrix.any()

        # a balanced element cannot overlap with fixed ions or with a second balanced element
        sf.add_ionisation(8, 2, n_ion=1e8)
        with pytest.raises(ValueError, match="already has ions"):
            sf.add_element_ionbalance(8, 1e10, {2: 1e-12})
        sf.add_element_ionbalance(2, 1e8, HELIUM_ALPHAS)
        with pytest.raises(ValueError, match="already added as a balanced element"):
            sf.add_element_saha(2, 1e8, 5000.0, [1, 2])
        with pytest.raises(ValueError, match="come from the ionisation balance"):
            sf.add_ionisation(2, 1, n_ion=1e8)
        with pytest.raises(ValueError, match="come from the ionisation balance"):
            sf.add_ionisation_channel(2, 1, 1e8, 24.6, np.where(sf.engrid > 24.6, 1e-17, 0.0))
        with pytest.raises(ValueError, match="come from the ionisation balance"):
            sf.add_excitation(2, 1, 1e8, np.where(sf.engrid > 21.0, 1e-17, 0.0), 21.0)
        with pytest.raises(ValueError, match="come from the ionisation balance"):
            sf.add_ion_ltepopexcitation(2, 1, n_ion=1e8, use_collstrengths=False)

        # n_ion=None needs a balanced ion
        with pytest.raises(ValueError, match="n_ion is required"):
            sf.add_ion_ltepopexcitation(8, 1, n_ion=None, use_collstrengths=False)
        with pytest.raises(ValueError, match="n_ion is required"):
            sf.add_ion_ltepopexcitation(26, 2, use_collstrengths=False)

        # the excitations of a balanced ion can be added once
        sf.add_ion_ltepopexcitation(2, 1, n_ion=None, use_collstrengths=False)
        with pytest.raises(ValueError, match="already added"):
            sf.add_ion_ltepopexcitation(2, 1, n_ion=None, use_collstrengths=False)
        assert len(sf.excitationlists[(2, 1)]) > 0
