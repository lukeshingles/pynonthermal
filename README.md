# pynonthermal
[![Build and test](https://github.com/lukeshingles/pynonthermal/actions/workflows/pythonapp.yml/badge.svg)](https://github.com/lukeshingles/pynonthermal/actions/workflows/pythonapp.yml)
[![codecov](https://codecov.io/gh/lukeshingles/pynonthermal/branch/main/graph/badge.svg?token=574XDCYFIi)](https://codecov.io/gh/lukeshingles/pynonthermal)

pynonthermal is a non-thermal energy deposition (Spencer-Fano equation) solver.

When high-energy leptons (electron and positrons) are injected into a plasma, they slow down by ionising and exciting atoms and ions, and Colomb scattering with free (thermal) electrons. Keeping track of the rates of the processes is important for example, when modelling Type Ia supernovae at late times (>100 days). At late times, ionisation by high-energy non-thermal leptons (seeded by radioactive decay products) generally overtakes photoionisation, and the non-thermal contribution to ionisation is needed to obtain reasonable agreement with observed emission lines of singly- and doubly-ionised species.

The numerical details of the solver are similar to the Spencer-Fano solver in the [ARTIS](https://github.com/artis-mcrt/artis) radiative transfer code ([Shingles et al. 2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.492.2029S/abstract)), which itself is an independent implementation of the [Kozma & Fransson (1992)](https://ui.adsabs.harvard.edu/abs/1992ApJ...390..602K/abstract) solution to the [Spencer & Fano (1945)](https://ui.adsabs.harvard.edu/abs/1954PhRv...93.1172S/abstract) equation. A similar solver is also applied in the [CMFGEN code](https://kookaburra.phyast.pitt.edu/hillier/web/CMFGEN.htm).

The impact ionisation cross sections are formula fits from [Arnaud & Rothenflug (1985)](https://ui.adsabs.harvard.edu/abs/1985A%26AS...60..425A/abstract) and [Arnaud & Raymond (1992)](https://ui.adsabs.harvard.edu/abs/1992ApJ...398..394A/abstract).

If the internal set of levels and transitions are applied (e.g., using ```add_ion_ltepopexcitation()```), these are imported from the [CMFGEN](https://kookaburra.phyast.pitt.edu/hillier/web/CMFGEN.htm) atomic data compilation. See the individual source files for atomic data references.

## Installation
For the latest experimental version, pynonthermal can be installed with:
```sh
python3 -m pip install git+git://github.com/lukeshingles/pynonthermal.git
```

If this version crashes or causes problems, you can try dropping back to a released version.
```sh
python3 -m pip install pynonthermal
```

## Usage
See the [quickstart notebook](https://github.com/lukeshingles/pynonthermal/blob/main/quickstart.ipynb) for an example of how to set up the composition and use the solver to determine ionisation and heating rates.

## Example output
The following plot shows the energy distribution of contributions to ionisation, excitation, and heating for a pure oxygen plasma (electron fraction 1e-2), reproducing figure 2 of KF92. The area under each curve gives the fraction of deposited energy going into that particular channel.

![Emission plot](https://raw.githubusercontent.com/lukeshingles/pynonthermal/main/docs/oxygen_channels.svg)

## Advanced Usage
Advanced users will likely want to control the particular excitation transitions that are included in the solver. Individual excitation transitions can be added with:

```python
SpencerFanoSolver.add_excitation(
  Z, ionstage, n_level, xs_vec, epsilon_trans_ev, transitionkey=(lower, upper)
)
```
Z is the atomic number. ionstage is the one more than the ion charge (e.g., Fe I or ion stage 1 has charge zero). The argument xs_vec is a numpy array of cross sections [cm<sup>2</sup>] defined at every energy in the sf.engrid array [eV]. The transition key can be almost anything that is unique within the ion and is used to refer back to the level pair when requesting the excitation rate coefficient.

```python
nt_exc = SpencerFanoSolver.get_excitation_ratecoeff(Z, ionstage, transitionkey)
```

## Meta

Distributed under the MIT license. See ``LICENSE`` for more information.

https://github.com/lukeshingles/pynonthermal


