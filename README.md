# pynonthermal
A non-thermal energy deposition (Spencer-Fano equation) solver

When high-energy leptons (electron and positrons) are injected into a plasma, they will slow down by ionising and exciting atoms and ions, and Colomb scattering with free (thermal) electrons.

The numerical details are nearly identical to the Spencer-Fano solver in the [ARTIS](https://github.com/artis-mcrt/artis) radiative transfer code (described in [Shingles et al. 2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.492.2029S/abstract)).
We solve the [Kozma & Fransson (1992)](https://ui.adsabs.harvard.edu/abs/1992ApJ...390..602K/abstract) form of the [Spencer & Fano (1945)](https://ui.adsabs.harvard.edu/abs/1954PhRv...93.1172S/abstract) equation.

Ionisation cross sections are formula fits from Arnaud & Rothenflug (1985) and Arnaud & Raymond (1992).


## Installation
This package depends on artistools, which currently must be installed by:
```sh
git clone https://github.com/artis-mcrt/artistools.git
python3 -m pip install -e artistools
```

Then, pynonthermal can be installed with:
```sh
git clone https://github.com/lukeshingles/pynonthermal.git
python3 -m pip install -e pynonthermal
```