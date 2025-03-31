import math
from functools import lru_cache
from math import atan
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

import pynonthermal
from pynonthermal.axelrod import get_binding_energies
from pynonthermal.axelrod import get_lotz_xs_ionisation
from pynonthermal.axelrod import get_shell_configs
from pynonthermal.constants import EV


@lru_cache
def get_nist_ionization_energies_ev() -> dict[tuple[int, int], float]:
    """Get a dictionary where dictioniz[(atomic_number, ion_sage)] = ionization_energy_ev."""
    dfnist = (
        pl.read_csv(
            Path(pynonthermal.DATADIR / "nist_ionization.txt.zst"),
            separator="\t",
            infer_schema_length=50,
            columns=["At. num", "Ion Charge", "Ionization Energy (a) (eV)"],
            ignore_errors=True,
        )
        .rename({"At. num": "Z", "Ionization Energy (a) (eV)": "ioniz_ev"})
        .with_columns(ion_stage=pl.col("Ion Charge").cast(pl.Int64) + 1)
        .drop("Ion Charge")
        .drop_nulls()
    )

    return {
        (atomic_number, ion_stage): ioniz_ev
        for atomic_number, ion_stage, ioniz_ev in dfnist.select(["Z", "ion_stage", "ioniz_ev"]).iter_rows(named=False)
    }


def read_colliondata(collionfilename: str | Path = "collion.txt") -> pl.DataFrame:
    dfcollion = pl.read_csv(
        Path(pynonthermal.DATADIR, collionfilename),
        separator=" ",
        has_header=False,
        skip_lines=1,
        truncate_ragged_lines=True,
        schema={
            "Z": pl.Int64,
            "nelec": pl.Int64,
            "n": pl.Int64,
            "l": pl.Int64,
            "ionpot_ev": pl.Float64,
            "A": pl.Float64,
            "B": pl.Float64,
            "C": pl.Float64,
            "D": pl.Float64,
        },
    )

    elements_electron_binding = get_binding_energies()
    all_shells_q = get_shell_configs()
    new_rows = []
    for Z in range(1, len(elements_electron_binding)):
        for ionstage in range(1, 6):
            any_data_matched = (
                dfcollion.filter(pl.col("Z") == Z).filter(pl.col("nelec") == (Z - ionstage + 1)).height > 0
            )

            if not any_data_matched:
                ioncharge = ionstage - 1
                nbound = Z - ioncharge  # number of bound electrons
                if nbound <= 0:
                    continue

                # ion_shells_q = get_shell_occupancies(Z, ionstage, elements_electron_binding, all_shells_q)
                ion_shells_q = all_shells_q[Z - 1]
                ionpot = get_nist_ionization_energies_ev()[(Z, ionstage)] * EV

                electron_count = 0
                for shellindex in range(len(ion_shells_q)):
                    electronsinshell = ion_shells_q[shellindex]

                    electron_count += electronsinshell

                    if electronsinshell <= 0:
                        continue
                    enbinding = elements_electron_binding[Z - 1][shellindex]
                    if enbinding <= 0:
                        # if we don't have the shell's binding energy, use the previous one
                        enbinding = elements_electron_binding[Z - 1][shellindex - 1]
                        assert enbinding > 0

                    p = max(ionpot, enbinding)
                    collionrow: dict[str, int | float] = {
                        "Z": Z,
                        "nelec": Z - ionstage + 1,
                        "n": -1,
                        "l": -shellindex,
                        "ionpot_ev": p / EV,
                        "A": -1.0,
                        "B": -1.0,
                        "C": -1.0,
                        "D": -1.0,
                    }

                    new_rows.append(collionrow)
                    if electron_count >= nbound:
                        break

    # Append Lotz approximate cross sections to the Arnaud/Rothenflug data
    return (
        pl.concat([dfcollion, pl.DataFrame(new_rows)])
        .with_columns(ion_stage=pl.col("Z") - pl.col("nelec") + 1)
        .sort(by=["Z", "ion_stage", "ionpot_ev", "n", "l"])
    )


def Psecondary(e_p: float, ionpot_ev: float, J: float, e_s: float = -1, epsilon: float = -1) -> float:
    # probability distribution function for secondaries energy e_s [eV] (or equivalently the energy loss of
    # the primary electron epsilon [eV]) given a primary energy e_p [eV] for an impact ionisation event

    assert e_s >= 0 or epsilon >= 0
    # if e_p < I:
    #     return 0.
    #
    if e_s < 0:
        e_s = epsilon - ionpot_ev
    if epsilon < 0:
        epsilon = e_s + ionpot_ev

    #
    # if epsilon < I:
    #     return 0.
    # if e_s < 0:
    #     return 0.
    # if e_s > e_p - I:
    #     return 0.
    # if e_s > e_p:
    #     return 0.

    # test case: constant, always below ionisation
    # Psecondary_e_s_max = 1. / J / 2.
    # return 1. / Psecondary_e_s_max if (e_s < Psecondary_e_s_max) else 0.

    return 1.0 / J / atan((e_p - ionpot_ev) / 2.0 / J) / (1 + ((e_s / J) ** 2))


def get_J(Z: int, ion_stage: int, ionpot_ev: float) -> float:
    # returns an energy in eV
    # values from Opal et al. 1971 as applied by Kozma & Fransson 1992
    if ion_stage == 1:
        if Z == 2:  # He I
            return 15.8
        if Z == 10:  # Ne I
            return 24.2
        if Z == 18:  # Ar I
            return 10.0

    return 0.6 * ionpot_ev


def ar_xs(energy_ev: float, ionpot_ev: float, A: float, B: float, C: float, D: float) -> float:
    u = energy_ev / ionpot_ev
    if u <= 1:
        return 0

    return (
        1e-14
        * (A * (1 - 1 / u) + B * pow((1 - 1 / u), 2) + C * math.log(u) + D * math.log(u) / u)
        / (u * pow(ionpot_ev, 2))
    )


def get_arxs_array_shell(arr_enev: npt.NDArray[np.float64], shell: dict[str, int | float]) -> npt.NDArray[np.float64]:
    if shell["n"] < 0:
        return np.array([get_lotz_xs_ionisation(shell, en_ev=en_ev) for en_ev in arr_enev])
    return np.array(
        [ar_xs(energy_ev, shell["ionpot_ev"], shell["A"], shell["B"], shell["C"], shell["D"]) for energy_ev in arr_enev]
    )


def get_arxs_array_ion(
    arr_enev: npt.NDArray[np.float64], dfcollion: pl.DataFrame, Z: int, ion_stage: int
) -> npt.NDArray[np.float64]:
    ar_xs_array = np.zeros(len(arr_enev))
    dfcollion_thision = dfcollion.filter(pl.col("Z") == Z).filter(pl.col("ion_stage") == ion_stage)
    for shell in dfcollion_thision.iter_rows(named=True):
        ar_xs_array += get_arxs_array_shell(arr_enev, shell)

    return ar_xs_array
