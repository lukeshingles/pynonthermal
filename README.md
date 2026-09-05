# pynonthermal
[![DOI](https://zenodo.org/badge/359805556.svg)](https://zenodo.org/badge/latestdoi/359805556)
[![PyPI - Version](https://img.shields.io/pypi/v/pynonthermal)](https://pypi.org/project/pynonthermal)
[![License](https://img.shields.io/github/license/lukeshingles/pynonthermal)](https://github.com/lukeshingles/pynonthermal/blob/main/LICENSE)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/pynonthermal)](https://pypi.org/project/pynonthermal/)
[![Build and test](https://github.com/lukeshingles/pynonthermal/actions/workflows/pytest.yml/badge.svg)](https://github.com/lukeshingles/pynonthermal/actions/workflows/pytest.yml)

pynonthermal is a Python solver for the Spencer-Fano equation, which describes the energy distribution of non-thermal (fast) electrons slowing down in a plasma. When high-energy leptons — such as the Compton, photoelectric, and pair-production electrons and positrons produced by radioactive decay in supernova ejecta — are injected into a partially ionised gas, they lose energy through three competing channels: Coulomb heating of the free thermal electrons, collisional ionisation, and collisional excitation of bound states.

Given a set of ions (with number densities) and an energy deposition rate, pynonthermal computes:

- the **degradation spectrum** y(E) of the non-thermal electron population,
- the **fraction of deposited energy** going to heating, ionisation, and excitation (per channel and per ion),
- **non-thermal ionisation rate coefficients** for each ion and **excitation rate coefficients** for individual bound-bound transitions, ready to be used in non-LTE plasma modelling.

These quantities are important, for example, in modelling the late-time spectra and light curves of Type Ia and core-collapse supernovae, where non-thermal ionisation can dominate over photoionisation. The solver follows the method of [Kozma & Fransson (1992)](https://ui.adsabs.harvard.edu/abs/1992ApJ...390..602K/abstract) (see [Method background](#method-background) for details and further references) and ships with the atomic data needed to run out of the box: ionisation cross sections for a wide range of ions, and level/transition data for bound-bound excitation.

## Contents
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage guide](#usage-guide)
- [Ion populations from an ionisation balance](#ion-populations-from-an-ionisation-balance)
- [Complete example: pure-oxygen plasma](#complete-example-pure-oxygen-plasma)
- [Units and conventions](#units-and-conventions)
- [Method background](#method-background)
- [Cross-section datasets](#cross-section-datasets)
- [Advanced usage: custom cross sections](#advanced-usage-custom-cross-sections)
- [Citing pynonthermal](#citing-pynonthermal)
- [License](#license)

## Installation

Released package (recommended for most users):

```sh
pip install pynonthermal
```

Development install with [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/lukeshingles/pynonthermal.git
cd pynonthermal
uv sync --frozen
source ./.venv/bin/activate
uv pip install --editable .
prek install
```

Run the test suite with:

```sh
uv run -- python3 -m pytest
```

## Quick start

```python
import pynonthermal

sf = pynonthermal.SpencerFanoSolver(emin_ev=0.1, emax_ev=3000.0, npts=4096)

# Add ions that can be non-thermally ionised.
# Here: O II (ion_stage=2, i.e. charge +1) with number density in cm^-3.
sf.add_ionisation(Z=8, ion_stage=2, n_ion=1.0e8)

# Solve for a deposition rate density in eV s^-1 cm^-3.
sf.solve(depositionratedensity_ev=1.0e8)

print("heating fraction:", sf.get_frac_heating())
print("total ionisation fraction:", sf.get_frac_ionisation_tot())
print("total excitation fraction:", sf.get_frac_excitation_tot())
print("sum of fractions:", sf.get_frac_sum())
print("ionisation rate coeff [s^-1]:", sf.get_ionisation_ratecoeff(Z=8, ion_stage=2))
```

The [quickstart notebook](https://github.com/lukeshingles/pynonthermal/blob/main/quickstart.ipynb) contains a fuller worked example, and can be launched on Binder:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lukeshingles/pynonthermal/HEAD?filepath=quickstart.ipynb)

## Usage guide

All ionisation and excitation channels must be added before calling `solve()`.

### 1. Create the solver

```python
sf = pynonthermal.SpencerFanoSolver(emin_ev=0.1, emax_ev=3000.0, npts=4096, verbose=False)
```

- `emin_ev`, `emax_ev`: bounds of the uniform energy grid in eV. Electrons that degrade below `emin_ev` are assumed to have thermalised, and their energy is counted as heating. The examples use `emin_ev=0.1`, the ARTIS default.
- `npts`: number of energy grid points. More points give better accuracy at the cost of memory and time; check `get_frac_sum()` after solving. The examples use `npts=4096`, the ARTIS default.
- `verbose`: print details of the setup, each added channel, and a per-ion, per-shell breakdown during analysis.
- `use_ar1985`: use the original Arnaud & Rothenflug (1985) ionisation cross sections (see [Cross-section datasets](#cross-section-datasets)).
- `heating_only_approximation`: remove the excitation and ionisation loss terms from the matrix and solve with the heating loss only. The solver still calculates the excitation and ionisation rates from this approximate solution, so the channel fractions do not sum to one.

The grid is available as `sf.engrid` (a NumPy array), which is needed if you supply [custom excitation cross sections](#custom-excitation-cross-sections).

### 2. Add ionisation channels

```python
sf.add_ionisation(Z=8, ion_stage=2, n_ion=1.0e8)
```

Adds every ionisation shell of the ion to the equation, using the built-in cross-section data. `Z` is the atomic number, `ion_stage` is one more than the ion charge (so `ion_stage=1` is neutral), and `n_ion` is the ion number density in cm^-3.

Each ion may be added once through this method; an ion with `n_ion=0.0` is silently skipped. If any of the ion's shells has an ionisation potential below `emin_ev`, a `ValueError` explains which lower `emin_ev` to use. To add a channel that the built-in table does not hold, or to replace the built-in shells of an ion, use [`add_ionisation_channel()`](#custom-ionisation-cross-sections).

The free electron density is computed automatically from the charges and densities of the added ions (`sf.get_n_e()`). At least one ionised species (or an explicit `override_n_e` in `solve()`) is required.

If you do not know the ion densities, the solver can find them for you from an ionisation balance. See [Ion populations from an ionisation balance](#ion-populations-from-an-ionisation-balance).

### 3. Add excitation channels (optional)

For bound-bound excitation using the built-in atomic database (levels and transitions from the CMFGEN compilation), with LTE level populations at a chosen temperature:

```python
sf.add_ion_ltepopexcitation(Z=8, ion_stage=1, n_ion=1.0e10, temperature=6000)
```

Optional parameters:

- `temperature`: temperature in K for the LTE Boltzmann level populations. A solver has one temperature: the first call that gives one (`add_ion_ltepopexcitation()`, `add_element_ltepopexcitation()`, or `add_element_saha()`) sets `sf.temperature`, later calls can leave it as `None` to use that value, and a different value raises a `ValueError`.
- `maxnlevelslower`, `maxnlevelsupper`: only include transitions from the lowest `maxnlevelslower` levels up to the lowest `maxnlevelsupper` levels (defaults 5 and 250, matching ARTIS). Pass `None` to include all.
- `use_collstrengths`: use tabulated collision strengths where available (default `True`); otherwise cross sections come from the oscillator strength via the van Regemorter approximation.

Transitions with energies outside the energy grid are dropped. If the internal database has no data for the ion, a `ValueError` is raised — you can then either supply your own level/transition table via `adata_polars` or add [custom cross sections](#custom-excitation-cross-sections) with `add_excitation()`.

An ion added only for excitation still contributes its charge to the free electron density.

### 4. Solve

```python
sf.solve(depositionratedensity_ev=1.0e8)
```

- `depositionratedensity_ev`: the rate of energy deposition per volume in eV s^-1 cm^-3 (must be positive and finite). The energy *fractions* are independent of this value; the *rate coefficients* scale linearly with it.
- `override_n_e`: optionally override the free electron density (cm^-3) instead of deriving it from the ion populations.
- `balance_tol`: the relative tolerance of the [ionisation balance](#ion-populations-from-an-ionisation-balance) (default `1e-4`). It has no effect without a balanced element.

The solution spectrum is stored as `sf.yvec` over `sf.engrid` (see [Method background](#method-background) for the numerical scheme).

### 5. Read the results

All getters require `solve()` to have been called first. Deposition fractions:

```python
sf.get_frac_heating()  # energy fraction to thermal electron heating
sf.get_frac_ionisation_tot()  # energy fraction to ionisation (all ions)
sf.get_frac_excitation_tot()  # energy fraction to excitation (all ions)
sf.get_frac_sum()  # sum of the above; ~1.0 if numerically accurate
sf.get_frac_ionisation_ion(Z, ion_stage)  # one ion's share of the ionisation fraction
```

Rate coefficients and derived quantities:

```python
sf.get_ionisation_ratecoeff(Z, ion_stage)  # non-thermal ionisation rate coefficient [s^-1]
sf.get_excitation_ratecoeff(Z, ion_stage, transitionkey)  # excitation rate coefficient [s^-1]
sf.get_eff_ionpot(Z, ion_stage)  # effective ionisation potential [eV] (KF92 eq. 12)
sf.get_n_e()  # free (thermal) electron density [cm^-3]
sf.get_n_e_nt()  # non-thermal electron density [cm^-3]
```

Multiply `get_ionisation_ratecoeff()` by the ion's number density to get ionisations per second per cm^3, and `get_excitation_ratecoeff()` by the lower level's population density to get excitations per second per cm^3. For excitations added by `add_ion_ltepopexcitation()`, the `transitionkey` is the tuple `(lower_level_index, upper_level_index)`, e.g. `(0, 8)` for ground level to the eighth excited level.

Call `sf.analyse_ntspectrum()` (with `verbose=True` on the solver) to print a detailed per-ion and per-shell breakdown.

### 6. Plot the solution

```python
sf.plot_yspectrum()  # degradation spectrum y(E)
sf.plot_channels(xscalelog=True)  # energy going to each channel vs electron energy
sf.plot_spec_channels("channels.pdf")  # both panels in one figure, saved to file
```

Each method shows the figure interactively, or saves it when `outputfilename` is given; `plot_yspectrum()` and `plot_channels()` also accept a Matplotlib `axis` to draw into an existing figure.

## Ion populations from an ionisation balance

Instead of a number density for each ion, you can give the number density of an element. The solver then finds the ion densities in `solve()`, in one of two ways.

### Non-thermal ionisation against recombination

```python
sf = pynonthermal.SpencerFanoSolver(emin_ev=0.1, emax_ev=3000, npts=4096)

# recombination rate coefficients in cm^3 s^-1, keyed by the ion stage that recombines
sf.add_element_ionbalance(Z=8, n_elem=1.0e10, recomb_ratecoeffs={2: 3.0e-13, 3: 3.0e-12, 4: 1.0e-11})

# LTE excitations of every stage of the element that has level data. The populations follow the balance.
sf.add_element_ltepopexcitation(Z=8, temperature=6000)

sf.solve(depositionratedensity_ev=2.95e8)

print(sf.ionpopdict)  # the converged ion densities [cm^-3]
print(sf.get_n_e())  # the charge-neutral free electron density [cm^-3]
```

For each pair of adjacent ion stages `i` and `i+1`, the balance is `n_i Gamma_i = n_{i+1} n_e alpha_{i+1}`. `Gamma_i` is the non-thermal ionisation rate coefficient of stage `i` from the Spencer-Fano solution (`get_ionisation_ratecoeff()`), and `alpha_{i+1}` is the recombination rate coefficient that you give for stage `i+1`. The chain of stages runs from one below the lowest key of `recomb_ratecoeffs` to the highest key. In the example, the chain is O I to O IV.

The Spencer-Fano solution depends on the ion densities, so `solve()` iterates: it solves the equation, updates the ion densities from the balance and the free electron density from charge neutrality, and repeats until the population ratios agree to `balance_tol`. A `RuntimeError` reports a balance that did not converge within 100 iterations. Typical cases converge in about 5 to 10 iterations.

Points to note:

- The balance includes only non-thermal ionisation and the recombination that you give. It does not include thermal collisional ionisation, photoionisation, or charge exchange. The ion fractions therefore depend on `depositionratedensity_ev`, unlike the fixed-population case.
- The top stage of the chain is a sink. Its ionisation is an energy loss in the matrix, but the ions it makes have no stage to go to. `solve()` warns if the ionisation rate out of the top stage exceeds 1 % of the total ionisation rate of the element, because about that fraction of the element then belongs in a higher stage. Then extend the chain with a rate coefficient for the next stage.
- Every stage of the chain gets the built-in ionisation channels of `add_ionisation()`. You cannot call `add_ionisation()`, `add_ionisation_channel()`, or `add_excitation()` for an ion of a balanced element.
- To choose the stages that get excitations, or to give each stage its own options, call `add_ion_ltepopexcitation(Z, ion_stage, n_ion=None, ...)` per stage instead of `add_element_ltepopexcitation()`. After `add_element_saha()`, the solver temperature is set, so the excitation calls can omit it.
- Until `solve()` runs, `sf.ionpopdict`, `sf.get_n_e()`, and `sf.get_n_ion_tot()` hold a provisional population of equal fractions for the stages of the chain.
- A second call to `solve()` starts from the converged rates of the first call.

### Saha equation

```python
sf.add_element_saha(Z=8, n_elem=1.0e10, temperature=12000, ion_stages=[1, 2, 3])
```

No recombination rate coefficients are needed. For each pair of adjacent stages, `n_{i+1} n_e / n_i = 2 (U_{i+1} / U_i) (2 pi m_e k_B T / h^2)^(3/2) exp(-chi_i / (k_B T))`, with the ionisation potentials `chi_i` from the NIST table. The partition functions `U_i` come from the LTE level populations of the built-in level data at `temperature`, which covers He, O, and Fe. For other elements, give them with `partfuncs={ion_stage: U, ...}`, or supply a level table via `adata_polars`. The bare nucleus (`ion_stage = Z + 1`) has a partition function of 1. `solve()` finds the free electron density from charge neutrality in one pass.

Both kinds of balanced element can be in one solver together with ions that have fixed number densities. The free electron density then sums the charges of all of them. With `override_n_e`, the balance uses that density instead.

The functions behind the balance are in `pynonthermal.ionbalance`: `get_saha_factor()`, `get_ion_fractions()`, and `solve_charge_neutral_n_e()`.

The [iron ionisation balance notebook](https://github.com/lukeshingles/pynonthermal/blob/main/fe_ionbalance_sn1a.ipynb) is a worked example: the ion fractions of iron in the core of a Type Ia supernova at 250 days, with the deposition rate from the 56Co decay, a comparison with the Saha equation, and the evolution from 150 to 400 days.

## Complete example: pure-oxygen plasma

This reproduces Figure 2 of Kozma & Fransson (1992): a pure-oxygen plasma with electron fraction x_e = 0.01, including both ionisation and excitation channels. With `verbose=True` the solver prints its setup and a per-ion, per-shell breakdown as it runs.

```python
import pynonthermal

n_e = 1e8  # free electron density [cm^-3]
x_e = 1e-2  # ionisation fraction n_OII / (n_OI + n_OII)
n_oxygen = n_e / x_e

ions = [
    # (Z, ion_stage, number_density)
    (8, 1, n_oxygen * (1 - x_e)),  # O I
    (8, 2, n_oxygen * x_e),  # O II
]

# emin_ev=1 matches the low-energy cutoff E_0 of Kozma & Fransson (1992)
sf = pynonthermal.SpencerFanoSolver(emin_ev=1, emax_ev=3000, npts=4096, verbose=True)
for Z, ion_stage, n_ion in ions:
    sf.add_ionisation(Z, ion_stage, n_ion)
    sf.add_ion_ltepopexcitation(Z, ion_stage, n_ion, temperature=6000)

# with fixed ion densities, any positive deposition rate works here: the energy fractions
# are independent of it (with a balanced element they would not be)
sf.solve(depositionratedensity_ev=2950.49 * n_oxygen)
sf.analyse_ntspectrum()  # print the full breakdown

sf.plot_channels(xscalelog=True)
```

The resulting plot shows the energy distribution of contributions to ionisation, excitation, and heating; the area under each curve gives the fraction of deposited energy in that channel:

![Energy deposition channels for a pure oxygen plasma](https://raw.githubusercontent.com/lukeshingles/pynonthermal/main/docs/oxygen_channels.svg)

## Units and conventions

- Energies are in eV.
- Number densities are in cm^-3.
- Cross sections are in cm^2.
- `ion_stage = charge + 1` (for example, Fe I has `ion_stage=1`, Fe II has `ion_stage=2`).
- `depositionratedensity_ev` in `solve()` is in eV s^-1 cm^-3.
- `get_ionisation_ratecoeff()` and `get_excitation_ratecoeff()` both return rates in s^-1.
- The recombination rate coefficients of `add_element_ionbalance()` are in cm^3 s^-1, keyed by the ion stage that recombines.

## Method background

The numerical solver is similar to the Spencer-Fano implementation in the [ARTIS](https://github.com/artis-mcrt/artis) radiative transfer code ([Shingles et al. 2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.492.2029S/abstract)), itself an independent implementation of [Kozma and Fransson (1992, ApJ, 390, 602)](https://ui.adsabs.harvard.edu/abs/1992ApJ...390..602K/abstract), based on the electron slowing-down equation of [Spencer and Fano (1954, Phys. Rev., 93, 1172)](https://ui.adsabs.harvard.edu/abs/1954PhRv...93.1172S/abstract). A similar approach is used in [CMFGEN](https://kookaburra.phyast.pitt.edu/hillier/web/CMFGEN.htm).

The integral form of the Kozma and Fransson degradation equation (their equation 7) is discretised on a uniform energy grid as an upper-triangular matrix equation and solved by back-substitution from the highest energy downward. The `SpencerFanoSolver` class docstring maps each term of the equation to the method that implements it, and the code comments cite the specific Kozma and Fransson equations at each site. The secondary-electron energy distribution follows [Opal, Peterson and Beaty (1971)](https://ui.adsabs.harvard.edu/abs/1971JChPh..55.4100O/abstract) as applied by Kozma and Fransson, and the energy loss rate to thermal electrons uses their Coulomb-logarithm prescription (after [Schunk and Hays 1971](https://ui.adsabs.harvard.edu/abs/1971P%26SS...19..113S/abstract)).

If internal level/transition data are used (for example, via `add_ion_ltepopexcitation()`), they are imported from the CMFGEN atomic data compilation (see the source data files for references), with excitation cross sections computed from the tabulated collision strengths ([Li, Dessart and Hillier 2012, equation 11](https://doi.org/10.1111/j.1365-2966.2012.21198.x)) or, for permitted transitions without one, from the oscillator strength via the van Regemorter (1962) approximation with the g-bar factor of [Mewe (1972)](https://ui.adsabs.harvard.edu/abs/1972A%26A....20..215M/abstract), as described in [Shingles et al. (2020, section 2.5)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.492.2029S/abstract).

## Cross-section datasets

Ionization cross sections from H (Z=1) to Ni (Z=28) use the shell-resolved analytical fits compiled by [Arnaud and Rothenflug (1985, A&AS, 60, 425)](https://ui.adsabs.harvard.edu/abs/1985A%26AS...60..425A/abstract), with updates to Fe from [Arnaud and Raymond (1992, ApJ, 398, 394)](https://ui.adsabs.harvard.edu/abs/1992ApJ...398..394A/abstract). For heavier elements (Z>28) and any other ions missing from the fit data, the approximation of [Axelrod (1980, PhD thesis, Eq. 3.38)](https://ui.adsabs.harvard.edu/abs/1980PhDT.........1A/abstract) is used — the high-energy limit of the [Lotz (1967, Z. Phys., 206, 205)](https://doi.org/10.1007/BF01325928) formula with relativistic corrections — with subshell binding energies from [Lotz (1970, J. Opt. Soc. Am., 60, 206)](https://doi.org/10.1364/JOSA.60.000206).

Passing `use_ar1985=True` to the solver selects the original Arnaud and Rothenflug (1985) compilation without the Fe updates, which can be useful for comparison with older published results.

## Advanced usage: custom cross sections

Give a custom cross section as a NumPy array of cross sections (cm^2) at every energy in `sf.engrid` (eV),
with `add_excitation()` or `add_ionisation_channel()`. Interpolate your own table onto `sf.engrid` first.

A custom cross section follows the same path through the solver as a built-in one. The matrix, the energy
fractions, and the rate coefficients therefore stay consistent.

The examples below use NumPy:

```python
import numpy as np
import pynonthermal
```

### Custom excitation cross sections

```python
sf.add_excitation(
    Z=8,
    ion_stage=2,
    levelnumberdensity=1.0e8,
    epsilon_trans_ev=20.0,
    transitionkey=(0, 3),
    xs_vec=np.interp(sf.engrid, my_en_ev, my_xs_cm2, left=0.0, right=0.0),
)
```

- `Z`: atomic number.
- `ion_stage`: one more than ion charge.
- `levelnumberdensity`: population density of the lower level (cm^-3), non-negative.
- `xs_vec`: a NumPy array of cross sections (cm^2), non-negative and finite, at every energy in
  `sf.engrid` (eV). The solver keeps a read-only copy, so a later write to your own array cannot change it.
- `epsilon_trans_ev`: transition energy (eV). Must be positive and no greater than `emax_ev`, since no
  electron the solver represents could otherwise drive the transition.
- `transitionkey`: any unique key within the ion, used to retrieve the excitation rate coefficient.

Transitions below `emin_ev` are allowed here, but `add_ion_ltepopexcitation()` drops them: Kozma and
Fransson (1992) take every electron below `emin_ev` to have thermalised, so that energy is accounted for
as heating instead.

Retrieve the rate coefficient afterwards with `get_excitation_ratecoeff()` as in [step 5](#5-read-the-results).

### Custom ionisation cross sections

```python
sf.add_ionisation_channel(
    Z=8,
    ion_stage=2,
    n_ion=1.0e8,
    ionpot_ev=35.0,
    xs_vec=np.interp(sf.engrid, my_en_ev, my_xs_cm2, left=0.0, right=0.0),
)
```

- `n_ion`: the ion number density (cm^-3). It must agree with the value that any other call for this ion
  gives.
- `ionpot_ev`: the ionisation potential of the channel (eV). It must be between `emin_ev` and `emax_ev`,
  and the cross section must be zero at and below it. Any value in that range is allowed, so a channel need
  not be a subshell of the built-in table. A total ionisation cross section for the ion works too.
- `xs_vec`: a NumPy array of cross sections (cm^2), non-negative and finite, at every energy in
  `sf.engrid` (eV).
- `channelkey`: any unique key within the ion. The default is the number of channels the ion already has.

Call `add_ionisation_channel()` once for each channel of the ion. To keep the built-in shells as well,
also call `add_ionisation()` for the ion. To replace them, do not call `add_ionisation()` for it.

A channel with `n_ion=0.0` is checked and then skipped, as `add_ionisation()` skips an ion.

`calculate_N_e()` integrates over a domain just above the ionisation potential that is narrower than one
grid cell, so the solver interpolates `xs_vec` between the grid points there. Resolve that region with
`npts` if it matters for your ion: the term it feeds is the energy that thermalises below `emin_ev`, which
is a small part of the heating fraction.

The solver keeps the Lorentzian secondary-electron distribution of Kozma and Fransson (1992, equation 4),
whose width comes from `pynonthermal.collion.get_J()`. The matrix fill integrates that distribution
analytically, so its shape is not adjustable.

Retrieve the rate coefficient afterwards with `get_ionisation_ratecoeff()` as in [step 5](#5-read-the-results).

## Citing pynonthermal

If you use pynonthermal, please cite it via the [Zenodo record](https://zenodo.org/badge/latestdoi/359805556). Please also consider citing the papers describing the method: [Kozma and Fransson (1992)](https://ui.adsabs.harvard.edu/abs/1992ApJ...390..602K/abstract) and [Shingles et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.492.2029S/abstract).

## License

Distributed under the MIT license. See [LICENSE](LICENSE) for details.
