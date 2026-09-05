"""Ionisation balance helpers that do not depend on the Spencer-Fano solver.

Every function here works with the ratio coefficient c_i = n_{i+1} n_e / n_i [cm^-3] of two
adjacent ion stages i and i+1. The Saha equation gives c_i from a temperature (get_saha_factor()).
A balance of non-thermal ionisation against recombination gives c_i = Gamma_i / alpha_{i+1},
with Gamma_i [s^-1] the ionisation rate coefficient of stage i and alpha_{i+1} [cm^3 s^-1] the
recombination rate coefficient of stage i+1. In both cases n_{i+1} / n_i = c_i / n_e, so the same
two functions give the ion fractions (get_ion_fractions()) and the charge-neutral free electron
density (solve_charge_neutral_n_e_ratios()). solve_charge_neutral_n_e() is the general root find
behind it, for any population model whose charge density does not increase with n_e.
"""

import math
from collections.abc import Callable
from collections.abc import Sequence

from pynonthermal.constants import EV
from pynonthermal.constants import H
from pynonthermal.constants import K_B
from pynonthermal.constants import ME

# (2 pi m_e k_B / h^2)^(3/2) in cm^-3 K^-3/2, the thermal factor of the Saha equation. K_B is in
# eV/K, so EV converts it to erg/K to match ME [g] and H [erg s].
SAHA_CONST: float = (2.0 * math.pi * ME * (K_B * EV) / H**2) ** 1.5

# the number of bisection steps in ln(n_e). Each step halves the bracket, so 100 steps take a
# bracket of 60 decades down to a relative width far below the double precision limit.
_N_BISECTION_STEPS: int = 100

# the first lower bracket of the bisection when no exact lower bound exists, as a fraction of the
# upper bracket, and the factor by which it falls until the residual is positive there. The
# populations are smooth in ln(n_e), so a wide bracket costs only steps.
_N_E_LOWER_BRACKET_FRACTION: float = 1e-60


def get_saha_factor(temperature: float, ionpot_ev: float, partfunc_lower: float, partfunc_upper: float) -> float:
    """Get the Saha ratio coefficient n_{i+1} n_e / n_i [cm^-3] of two adjacent ion stages.

    The result is 2 (U_{i+1} / U_i) (2 pi m_e k_B T / h^2)^(3/2) exp(-chi_i / (k_B T)), where
    the factor 2 is the statistical weight of the free electron.

    temperature:
        the temperature in K
    ionpot_ev:
        the ionisation potential chi_i of the lower stage in eV
    partfunc_lower, partfunc_upper:
        the partition functions U_i and U_{i+1} of the lower and the upper stage
    """
    if not 0.0 < temperature < math.inf:
        msg = f"temperature must be greater than zero and finite but is {temperature}"
        raise ValueError(msg)
    if not 0.0 < ionpot_ev < math.inf:
        msg = f"ionpot_ev must be greater than zero and finite but is {ionpot_ev}"
        raise ValueError(msg)
    if not 0.0 < partfunc_lower < math.inf or not 0.0 < partfunc_upper < math.inf:
        msg = f"partition functions must be greater than zero and finite but are {partfunc_lower}, {partfunc_upper}"
        raise ValueError(msg)

    return (
        2.0 * partfunc_upper / partfunc_lower * SAHA_CONST * temperature**1.5 * math.exp(-ionpot_ev / K_B / temperature)
    )


def get_ion_fractions(ratio_coeffs: Sequence[float], n_e: float) -> list[float]:
    """Get the fractions of a contiguous chain of ion stages from the ratio coefficients.

    ratio_coeffs holds c_i = n_{i+1} n_e / n_i [cm^-3] for each pair of adjacent stages, so the
    chain has len(ratio_coeffs) + 1 stages. The calculation runs in log space, so very large
    and very small coefficients do not overflow. A coefficient of exactly zero makes every
    higher stage exactly zero.

    n_e:
        the free electron density in cm^-3
    """
    if not 0.0 < n_e < math.inf:
        msg = f"n_e must be greater than zero and finite but is {n_e}"
        raise ValueError(msg)
    for c in ratio_coeffs:
        if not 0.0 <= c < math.inf:
            msg = f"ratio coefficients must be non-negative and finite but one is {c}"
            raise ValueError(msg)

    # ln(n_i / n_1) for each stage, with -inf after a zero coefficient
    ln_n_e = math.log(n_e)
    ln_relative = [0.0]
    for c in ratio_coeffs:
        ln_relative.append(ln_relative[-1] + (math.log(c) - ln_n_e) if c > 0.0 else -math.inf)

    ln_max = max(ln_relative)
    relative = [math.exp(ln_n - ln_max) if ln_n > -math.inf else 0.0 for ln_n in ln_relative]
    total = sum(relative)

    return [value / total for value in relative]


def solve_charge_neutral_n_e(
    n_e_fixed: float, charge_density: Callable[[float], float], charge_density_min: float, charge_density_max: float
) -> float:
    """Get the free electron density [cm^-3] that makes the plasma charge neutral.

    The result n_e satisfies n_e = n_e_fixed + charge_density(n_e). A bisection in ln(n_e) finds
    it. The function works for any population model whose charge density does not increase with
    the free electron density, for example the ratio coefficients of solve_charge_neutral_n_e_ratios()
    or a collisional-radiative model.

    n_e_fixed:
        the free electron density [cm^-3] from ions whose populations are fixed
    charge_density:
        a function of the free electron density n_e [cm^-3] that gives the electrons [cm^-3] from
        the ion charges of the modelled populations at that n_e. It must not increase with n_e, so
        that the solution is unique.
    charge_density_min:
        a lower bound of charge_density(n_e) at every n_e, for example the charge of the lowest ion
        stage of every element. Zero if no positive bound exists.
    charge_density_max:
        an upper bound of charge_density(n_e) at every n_e, for example the charge of the highest
        ion stage of every element
    """
    if not 0.0 <= n_e_fixed < math.inf:
        msg = f"n_e_fixed must be non-negative and finite but is {n_e_fixed}"
        raise ValueError(msg)
    if not 0.0 <= charge_density_min <= charge_density_max < math.inf:
        msg = (
            "the bounds of the charge density must satisfy 0 <= charge_density_min <= charge_density_max and be"
            f" finite but are {charge_density_min} and {charge_density_max}"
        )
        raise ValueError(msg)

    n_e_lower = n_e_fixed + charge_density_min
    n_e_upper = n_e_fixed + charge_density_max
    if n_e_upper <= 0.0:
        msg = (
            "no modelled population has an ionised stage and no fixed ion is ionised, so the free electron"
            " density is zero"
        )
        raise ValueError(msg)

    def residual(n_e: float) -> float:
        return n_e_fixed + charge_density(n_e) - n_e

    # the residual falls with n_e, and it is at most zero at the upper bracket
    if n_e_lower > 0.0:
        # the lower bracket is exact: the charge density is at least charge_density_min, so the residual
        # there is at least zero. Zero means that the populations sit at the bound, and then the lower
        # bracket is the solution.
        if residual(n_e_lower) <= 0.0:
            return n_e_lower
    else:
        # no exact positive lower bound. As n_e falls, the charge density tends to a positive constant
        # (every population that can be ionised is), so the residual turns positive at a small enough
        # n_e unless the charge density is zero everywhere.
        n_e_lower = n_e_upper * _N_E_LOWER_BRACKET_FRACTION
        while (charge_density_lower := charge_density(n_e_lower)) <= n_e_lower:
            if charge_density_lower <= 0.0:
                msg = "the charge density is zero and no fixed ion is ionised, so the free electron density is zero"
                raise ValueError(msg)
            n_e_lower *= _N_E_LOWER_BRACKET_FRACTION
            if n_e_lower == 0.0:
                msg = "the charge-neutral free electron density is below the range of a double precision number"
                raise ValueError(msg)

    ln_lower = math.log(n_e_lower)
    ln_upper = math.log(n_e_upper)
    for _ in range(_N_BISECTION_STEPS):
        ln_mid = 0.5 * (ln_lower + ln_upper)
        if residual(math.exp(ln_mid)) > 0.0:
            ln_lower = ln_mid
        else:
            ln_upper = ln_mid
        if ln_upper - ln_lower < 1e-15:
            break

    return math.exp(0.5 * (ln_lower + ln_upper))


def solve_charge_neutral_n_e_ratios(n_e_fixed: float, elements: Sequence[tuple[float, int, Sequence[float]]]) -> float:
    """Get the charge-neutral free electron density [cm^-3] for elements with ratio coefficients.

    This calls solve_charge_neutral_n_e() with the ion fractions of each element from
    get_ion_fractions() at n_e. The mean charge of every element decreases with n_e, so the
    solution is unique.

    n_e_fixed:
        the free electron density [cm^-3] from ions whose populations are fixed
    elements:
        one tuple (n_elem, lowest_stage, ratio_coeffs) per element: the element number density
        [cm^-3], the lowest ion stage of its chain, and the ratio coefficients
        n_{i+1} n_e / n_i [cm^-3] of each pair of adjacent stages. The chain has one stage more
        than ratio coefficients.
    """
    charge_density_min = 0.0
    charge_density_max = 0.0
    for n_elem, lowest_stage, ratio_coeffs in elements:
        if not 0.0 < n_elem < math.inf:
            msg = f"n_elem must be greater than zero and finite but is {n_elem}"
            raise ValueError(msg)
        if lowest_stage < 1:
            msg = f"the lowest ion stage must be at least 1 but is {lowest_stage}"
            raise ValueError(msg)
        # every stage is at least as charged as the lowest one and at most as the highest one
        charge_density_min += (lowest_stage - 1) * n_elem
        charge_density_max += (lowest_stage - 1 + len(ratio_coeffs)) * n_elem

    def charge_density(n_e: float) -> float:
        # the free electron density that the ion charges of the elements give at n_e
        total = 0.0
        for n_elem, lowest_stage, ratio_coeffs in elements:
            fractions = get_ion_fractions(ratio_coeffs, n_e)
            total += n_elem * sum((lowest_stage - 1 + index) * frac for index, frac in enumerate(fractions))
        return total

    return solve_charge_neutral_n_e(n_e_fixed, charge_density, charge_density_min, charge_density_max)
