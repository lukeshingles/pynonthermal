from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

import pynonthermal
from pynonthermal.axelrod import get_binding_energies
from pynonthermal.axelrod import get_lotz_xs_ionisation_vec
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


@lru_cache
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

    nist_ionisation_energies_ev = get_nist_ionization_energies_ev()
    elements_electron_binding = get_binding_energies()
    all_shells_q = get_shell_configs()
    covered_z_nelec = set(dfcollion.select(["Z", "nelec"]).iter_rows())
    new_rows: list[dict[str, int | float]] = []
    for Z in range(1, len(elements_electron_binding) + 1):
        for ionstage in range(1, Z + 1):
            any_data_matched = (Z, Z - ionstage + 1) in covered_z_nelec

            if not any_data_matched:
                ioncharge = ionstage - 1
                nbound = Z - ioncharge  # number of bound electrons
                if nbound <= 0:
                    continue

                try:
                    ion_shells_q = all_shells_q[Z - 1]
                    ionpot = nist_ionisation_energies_ev[(Z, ionstage)] * EV
                except (KeyError, IndexError):
                    continue

                electron_count = 0
                for shellindex in range(len(ion_shells_q)):
                    electronsinshell = ion_shells_q[shellindex]

                    electron_count += electronsinshell

                    if electronsinshell <= 0:
                        continue
                    enbinding = elements_electron_binding[Z - 1][shellindex]
                    if enbinding <= 0:
                        # if we don't have the shell's binding energy, use the previous one.
                        # shellindex 0 has no previous shell, and a negative index would silently
                        # wrap around to the outermost shell.
                        if shellindex == 0:
                            msg = f"No binding energy for the innermost shell of Z={Z}"
                            raise ValueError(msg)
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


def Psecondary_vec(
    e_p: npt.NDArray[np.float64] | float, e_s: npt.NDArray[np.float64] | float, ionpot_ev: float, J: float
) -> npt.NDArray[np.float64]:
    """Evaluate the secondary-electron energy distribution over arrays of e_p and/or e_s.

    This is the Lorentzian distribution of Kozma & Fransson 1992 equation 4 (a fit by Opal,
    Peterson & Beaty 1971 to their measurements), with the width parameter J from get_J().

    e_p is the primary electron energy [eV] and e_s the secondary electron energy [eV]. The two
    arguments broadcast against each other, so either may be a scalar. Psecondary() is the scalar form.

    The distribution is normalised over its physical support, e_s in [0, (e_p - ionpot_ev) / 2], but is
    not truncated to it here, so the caller must supply limits of integration that stay inside it. Note
    also that the normalisation diverges as e_p approaches ionpot_ev from above, so a limit of
    integration that is snapped onto a coarse energy grid near the threshold can be badly wrong.
    """
    arr_e_p = np.asarray(e_p, dtype=np.float64)
    arr_e_s = np.asarray(e_s, dtype=np.float64)

    # below the ionisation threshold there are no secondaries (and the normalisation
    # atan((e_p - ionpot_ev) / 2 / J) would be zero or negative)
    norm = np.arctan((arr_e_p - ionpot_ev) / 2.0 / J)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(arr_e_p > ionpot_ev, 1.0 / J / norm / (1 + (arr_e_s / J) ** 2), 0.0)


def Psecondary(e_p: float, ionpot_ev: float, J: float, e_s: float = -1, epsilon: float = -1) -> float:
    # probability distribution function for secondaries energy e_s [eV] (or equivalently the energy loss of
    # the primary electron epsilon [eV]) given a primary energy e_p [eV] for an impact ionisation event

    # see Psecondary_vec() for the support that the caller's limits of integration must stay inside
    if e_s < 0 and epsilon < 0:
        msg = "either the secondary energy e_s or the primary energy loss epsilon must be given"
        raise ValueError(msg)

    if e_s < 0:
        e_s = epsilon - ionpot_ev

    return float(Psecondary_vec(e_p=e_p, e_s=e_s, ionpot_ev=ionpot_ev, J=J))


def get_J(Z: int, ion_stage: int, ionpot_ev: float) -> float:
    # returns an energy in eV
    # values from Opal et al. 1971 as applied by Kozma & Fransson 1992. They are whole-atom measurements
    # dominated by valence-shell ionisation, but are used for every shell of the ion to match ARTIS.
    if ion_stage == 1:
        if Z == 2:  # He I
            return 15.8
        if Z == 10:  # Ne I
            return 24.2
        if Z == 18:  # Ar I
            return 10.0

    return 0.6 * ionpot_ev


def get_arxs_array_shell(arr_enev: npt.NDArray[np.float64], shell: dict[str, int | float]) -> npt.NDArray[np.float64]:
    # the shell's total impact-ionisation cross section in cm^2 at each energy [eV]: the
    # sigma_ic of Kozma & Fransson 1992 (equations 5, 10, and 11 and the ionisation term of
    # equation 7). Shells with fit data use the formula of Younger 1981 with the A-D
    # coefficients of the Arnaud & Rothenflug 1985 compilation; shells marked n < 0 have no
    # fit data and use the Lotz approximation instead.
    if shell["n"] < 0:
        return get_lotz_xs_ionisation_vec(shell, arr_en_ev=arr_enev)

    ionpot_ev = float(shell["ionpot_ev"])
    xs = np.zeros_like(arr_enev)
    abovethreshold = arr_enev > ionpot_ev
    u = arr_enev[abovethreshold] / ionpot_ev
    # a few of the fits (Ne I n=2 l=0, Na II n=2 l=1) go slightly negative just above threshold,
    # so clamp to zero to keep the cross section physical
    xs[abovethreshold] = np.maximum(
        0.0,
        1e-14
        * (
            shell["A"] * (1 - 1 / u)
            + shell["B"] * (1 - 1 / u) ** 2
            + shell["C"] * np.log(u)
            + shell["D"] * np.log(u) / u
        )
        / (u * ionpot_ev**2),
    )
    return xs
