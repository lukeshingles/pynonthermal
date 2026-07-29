import math
from pathlib import Path

import numpy as np
import pytest

import pynonthermal

pytestmark = pytest.mark.benchmark
outputfolder = Path(__file__).absolute().parent / "output"


def test_lotz_heavy_element() -> None:
    # elements heavier than Ni (Z>28) use the Axelrod 1980/Lotz 1967 cross section approximation
    with pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=400) as sf:
        sf.add_ionisation(56, 2, n_ion=1.0)
        sf.solve(depositionratedensity_ev=100, override_n_e=1.0)

        assert math.isclose(sf.get_frac_sum(), 1.0, abs_tol=0.01)
        assert sf.get_ionisation_ratecoeff(56, 2) > 0.0


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

        sf.solve(depositionratedensity_ev=100, override_n_e=1e-4)

        # rate coefficient is available without an explicit analyse_ntspectrum() call
        assert sf.get_ionisation_ratecoeff(2, 1) > 0.0

        # a legitimately-zero excitation fraction must not trigger repeated re-analysis
        assert sf.get_frac_excitation_tot() == 0.0
        count_after_first_call = sf.analyse_count
        assert count_after_first_call >= 1
        assert sf.get_frac_excitation_tot() == 0.0
        assert sf.analyse_count == count_after_first_call

        # additions are locked after solving
        with pytest.raises(RuntimeError):
            sf.add_ionisation(2, 2, n_ion=1e-4)


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
