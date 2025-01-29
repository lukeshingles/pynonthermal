# pynonthermal
[![DOI](https://zenodo.org/badge/359805556.svg)](https://zenodo.org/badge/latestdoi/359805556)
[![PyPI - Version](https://img.shields.io/pypi/v/pynonthermal)](https://pypi.org/project/pynonthermal)
[![License](https://img.shields.io/github/license/lukeshingles/pynonthermal)](https://github.com/lukeshingles/pynonthermal/blob/main/LICENSE)

[![Supported Python versions](https://img.shields.io/pypi/pyversions/pynonthermal)](https://pypi.org/project/pynonthermal/)
[![Build and test](https://github.com/lukeshingles/pynonthermal/actions/workflows/pytest.yml/badge.svg)](https://github.com/lukeshingles/pynonthermal/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/lukeshingles/pynonthermal/branch/main/graph/badge.svg?token=574XDCYFIi)](https://codecov.io/gh/lukeshingles/pynonthermal)

pynonthermal is a non-thermal energy deposition (Spencer-Fano equation) solver.

When high-energy leptons (electron and positrons) are injected into a plasma, they slow down by ionising and exciting atoms and ions, and Coulomb scattering with free (thermal) electrons. Keeping track of the rates of the processes is important for example, when modelling Type Ia supernovae at late times (>100 days). At late times, ionisation by high-energy non-thermal leptons (seeded by radioactive decay products) generally overtakes photoionisation, and the non-thermal contribution to ionisation is needed to obtain reasonable agreement with observed emission lines of singly- and doubly-ionised species.

The numerical details of the solver are similar to the Spencer-Fano solver in the [ARTIS](https://github.com/artis-mcrt/artis) radiative transfer code ([Shingles et al. 2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.492.2029S/abstract)), which itself is an independent implementation of the [Kozma & Fransson (1992)](https://ui.adsabs.harvard.edu/abs/1992ApJ...390..602K/abstract) solution to the [Spencer & Fano (1945)](https://ui.adsabs.harvard.edu/abs/1954PhRv...93.1172S/abstract) equation. A similar solver is also applied in the [CMFGEN code](https://kookaburra.phyast.pitt.edu/hillier/web/CMFGEN.htm).

The impact ionisation cross sections are formula fits from [Arnaud & Rothenflug (1985)](https://ui.adsabs.harvard.edu/abs/1985A%26AS...60..425A/abstract) and [Arnaud & Raymond (1992)](https://ui.adsabs.harvard.edu/abs/1992ApJ...398..394A/abstract) for Z=1 to 28 (H to Ni). Heavier elements use the approximation of [Axelrod (1980)](https://ui.adsabs.harvard.edu/abs/1980PhDT.........1A/abstract) Eq 3.38 ([Lotz 1967](https://doi.org/10.1007/BF01325928)).

If the internal set of levels and transitions are applied (e.g., using ```add_ion_ltepopexcitation()```), these are imported from the [CMFGEN](https://kookaburra.phyast.pitt.edu/hillier/web/CMFGEN.htm) atomic data compilation. See the individual source files for atomic data references.

## Example output
The following plot shows the energy distribution of contributions to ionisation, excitation, and heating for a pure oxygen plasma (electron fraction 1e-2), reproducing figure 2 of KF92. The area under each curve gives the fraction of deposited energy going into that particular channel.

![Emission plot](https://raw.githubusercontent.com/lukeshingles/pynonthermal/main/docs/oxygen_channels.svg)

## Installation
The latest released version can be installed from PyPI:
```sh
pip install pynonthermal
```

For development, pynonthermal can be installed into a uv project environment:
```sh
git clone https://github.com/lukeshingles/pynonthermal.git
cd pynonthermal
uv sync --frozen
source ./.venv/bin/activate
uv pip install --editable .
pre-commit install
```

## Usage
See the [quickstart notebook](https://github.com/lukeshingles/pynonthermal/blob/main/quickstart.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lukeshingles/pynonthermal/HEAD?filepath=quickstart.ipynb) for an example of how to set up the composition and use the solver to determine ionisation and heating rates.

## Advanced Usage - Excitation cross sections
Advanced users will likely want to control the particular excitation cross sections that are included in the solver. Individual excitation transitions can be added with:

```python
SpencerFanoSolver.add_excitation(Z, ion_stage, n_level, xs_vec, epsilon_trans_ev, transitionkey=(lower, upper))
```
Z is the atomic number. ion_stage is the one more than the ion charge (e.g., Fe I or ion stage 1 has charge zero). The argument xs_vec is a numpy array of cross sections [cm<sup>2</sup>] defined at every energy in the sf.engrid array [eV]. The transition key can be almost anything that is unique within the ion and is used to refer back to the level pair when requesting the excitation rate coefficient.

```python
nt_exc = SpencerFanoSolver.get_excitation_ratecoeff(Z, ion_stage, transitionkey)
```

## Meta

Distributed under the MIT license. See ``LICENSE`` for more information.

https://github.com/lukeshingles/pynonthermal
