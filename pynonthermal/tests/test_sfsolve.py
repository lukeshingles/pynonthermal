import numpy as np
from pathlib import Path
import pynonthermal

outputfolder = Path(__file__).absolute().parent / "output"


def test_helium():
    # KF1992 Figure 3. Pure-Helium Plasma
    x_e = 1e-4
    ions = [
        # Z ion_stage numberdensity
        (2, 1, 1.0 - x_e),
        (2, 2, x_e),
    ]

    with pynonthermal.SpencerFanoSolver(
        emin_ev=1, emax_ev=3000, npts=2000, verbose=True
    ) as sf:
        for Z, ion_stage, n_ion in ions:
            sf.add_ionisation(Z, ion_stage, n_ion=n_ion)
            sf.add_ion_ltepopexcitation(Z, ion_stage, n_ion=n_ion)

        sf.solve(depositionratedensity_ev=100)

        sf.analyse_ntspectrum()
        frac_excitation_tot = sf.get_frac_excitation_tot()
        frac_ionisation_tot = sf.get_frac_ionisation_tot()
        frac_heating = sf.get_frac_heating()

        frac_sum = frac_excitation_tot + frac_ionisation_tot + frac_heating
        assert np.isclose(frac_sum, 1.0, atol=0.005)
        assert np.isclose(frac_excitation_tot, 0.3315, atol=0.05)
        assert np.isclose(frac_ionisation_tot, 0.4849, atol=0.05)

        sf.plot_spec_channels(
            outputfilename=outputfolder / "spec_channels.pdf", xscalelog=True
        )
