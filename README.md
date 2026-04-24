# pynonthermal
[![DOI](https://zenodo.org/badge/359805556.svg)](https://zenodo.org/badge/latestdoi/359805556)
[![PyPI - Version](https://img.shields.io/pypi/v/pynonthermal)](https://pypi.org/project/pynonthermal)
[![License](https://img.shields.io/github/license/lukeshingles/pynonthermal)](https://github.com/lukeshingles/pynonthermal/blob/main/LICENSE)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/pynonthermal)](https://pypi.org/project/pynonthermal/)
[![Build and test](https://github.com/lukeshingles/pynonthermal/actions/workflows/pytest.yml/badge.svg)](https://github.com/lukeshingles/pynonthermal/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/lukeshingles/pynonthermal/branch/main/graph/badge.svg?token=574XDCYFIi)](https://codecov.io/gh/lukeshingles/pynonthermal)

pynonthermal is a Spencer-Fano equation solver for non-thermal electron energy deposition in plasmas. It computes how deposited energy is partitioned into heating, ionisation, and excitation, and provides non-thermal ionisation and excitation rate coefficients.

## Contents
- [Quick start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Units and conventions](#units-and-conventions)
- [Example output](#example-output)
- [Method background](#method-background)
- [Cross-section dataset choice](#cross-section-dataset-choice)
- [Citing pynonthermal](#citing-pynonthermal)
- [License](#license)

## Quick start

```python
import pynonthermal

sf = pynonthermal.SpencerFanoSolver(emin_ev=1.0, emax_ev=3000.0, npts=4000)

# Add ions that can be non-thermally ionised.
# Here: O II (ion_stage=2, i.e. charge +1) with number density in cm^-3.
sf.add_ionisation(Z=8, ion_stage=2, n_ion=1.0e8)

# Solve for a deposition rate density in eV s^-1 cm^-3.
sf.solve(depositionratedensity_ev=1.0e8)

print("heating fraction:", sf.get_frac_heating())
print("total ionisation fraction:", sf.get_frac_ionisation_tot())
print("total excitation fraction:", sf.get_frac_excitation_tot())
print("sum of fractions:", sf.get_frac_sum())
print("ionisation rate coeff [cm^3 s^-1]:", sf.get_ionisation_ratecoeff(Z=8, ion_stage=2))
```

The [quickstart notebook](https://github.com/lukeshingles/pynonthermal/blob/main/quickstart.ipynb) contains a fuller worked example, and can be launched on Binder:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lukeshingles/pynonthermal/HEAD?filepath=quickstart.ipynb)

## Installation

Released package (recommended for most users):

```sh
pip install pynonthermal
```

Development install with uv:

```sh
git clone https://github.com/lukeshingles/pynonthermal.git
cd pynonthermal
uv sync --frozen
source ./.venv/bin/activate
uv pip install --editable .
prek install
```

## Usage

Typical solver workflow:

1. Create `SpencerFanoSolver` with an energy grid (`emin_ev`, `emax_ev`, `npts`).
2. Add ionisation channels with `add_ionisation(Z, ion_stage, n_ion)`.
3. Optionally add excitation channels with `add_excitation(...)` or `add_ion_ltepopexcitation(...)`.
4. Call `solve(depositionratedensity_ev=..., override_n_e=...)`.
5. Query outputs such as `get_frac_heating()`, `get_frac_ionisation_tot()`, `get_frac_excitation_tot()`, `get_ionisation_ratecoeff(Z, ion_stage)`, and `get_excitation_ratecoeff(Z, ion_stage, transitionkey)`.

## Units and conventions

- Energies are in eV.
- Number densities are in cm^-3.
- Cross sections are in cm^2.
- `ion_stage = charge + 1` (for example, Fe I has `ion_stage=1`, Fe II has `ion_stage=2`).
- `depositionratedensity_ev` in `solve()` is in eV s^-1 cm^-3.

## Example output

The following plot shows the energy distribution of contributions to ionisation, excitation, and heating for a pure oxygen plasma (electron fraction 0.01), reproducing Figure 2 of Kozma and Fransson (1992). The area under each curve gives the fraction of deposited energy in that channel.

![Emission plot](https://raw.githubusercontent.com/lukeshingles/pynonthermal/main/docs/oxygen_channels.svg)

This figure is generated from the same solver setup demonstrated in the quickstart workflow.

## Method background

When high-energy leptons (electrons and positrons) are injected into a plasma, they lose energy through ionisation, excitation, and Coulomb interactions with thermal electrons. Tracking these rates is important, for example, in late-time Type Ia supernova modelling.

The numerical solver is similar to the Spencer-Fano implementation in the [ARTIS](https://github.com/artis-mcrt/artis) radiative transfer code ([Shingles et al. 2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.492.2029S/abstract)), itself an independent implementation of [Kozma and Fransson (1992)](https://ui.adsabs.harvard.edu/abs/1992ApJ...390..602K/abstract), based on [Spencer and Fano (1945)](https://ui.adsabs.harvard.edu/abs/1954PhRv...93.1172S/abstract). A similar approach is used in [CMFGEN](https://kookaburra.phyast.pitt.edu/hillier/web/CMFGEN.htm).

Impact ionisation cross sections use fits from [Arnaud and Rothenflug (1985)](https://ui.adsabs.harvard.edu/abs/1985A%26AS...60..425A/abstract) and [Arnaud and Raymond (1992)](https://ui.adsabs.harvard.edu/abs/1992ApJ...398..394A/abstract) for Z=1 to 28 (H to Ni). Heavier elements use the approximation of [Axelrod (1980), Eq. 3.38](https://ui.adsabs.harvard.edu/abs/1980PhDT.........1A/abstract) with [Lotz (1967)](https://doi.org/10.1007/BF01325928).

If internal level/transition data are used (for example, via `add_ion_ltepopexcitation()`), they are imported from the CMFGEN atomic data compilation. See source data files for references.

## Cross-section datasets
Ionization cross sections from H (Z=1) to Ni (Z=28) are sourced from the analytical fits to data from M. Arnaud & R. Rothenflug (1985, A&AS, 60, 425), with updates to Fe from M. Arnaud & J. Raymond (1992, ApJ, 398, 394). For heavier elements (Z>28), the approximation of Axelrod (1980, PhD thesis, Eq. 3.38) is used, with Lotz (1967, Z. Phys., 206, 205) for the required parameters.

## Advanced usage: custom excitation cross sections

You can supply your own excitation cross section table:

```python
sf.add_excitation(Z, ion_stage, n_level, xs_vec, epsilon_trans_ev, transitionkey=(lower, upper))
```

- `Z`: atomic number.
- `ion_stage`: one more than ion charge.
- `xs_vec`: NumPy array of cross sections (cm^2), defined at every energy in `sf.engrid` (eV).
- `transitionkey`: any unique key within the ion, used to retrieve the excitation rate coefficient.

Retrieve the non-thermal excitation rate coefficient with:

```python
nt_exc = sf.get_excitation_ratecoeff(Z, ion_stage, transitionkey)
```


## Citing pynonthermal

If you use pynonthermal, please cite it via the [Zenodo record](https://zenodo.org/badge/latestdoi/359805556).

## License

Distributed under the MIT license. See [LICENSE](LICENSE) for details.
