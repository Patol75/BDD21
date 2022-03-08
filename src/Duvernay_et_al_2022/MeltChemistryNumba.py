#!/usr/bin/env python3
from numba import njit
from numba.pycc import CC
import numpy as np
from scipy.constants import Avogadro, pi, R

from constants import const, hfse, radio


cc = CC("MeltChemistryCompiled")
cc.target_cpu = "host"
cc.verbose = True


@cc.export(
    "mineralogy",
    "Tuple((f8[:], f8[:], f8[:, :]))"
    "(f8, f8, f8[:], u8[:], f8[:, :], f8[:, :], u8[:],"
    " DictType(unicode_type, f8[:]), f8, DictType(unicode_type, f8[:]))",
)
def mineralogy(
    X, X_0, Fn_0, ele_ind, D, ri, val, mnrl_mode_coeff, eNd, part_arr
):
    """Determine modal mineral abundances, reaction coefficients and partition
    coefficients."""
    # Pressure and temperature interpolated from melt fraction
    P = np.interp(X, part_arr["melt_fraction"], part_arr["pressure"])
    T = np.interp(X, part_arr["melt_fraction"], part_arr["temperature"])

    # Modal mineral abundances in the residual solid
    Fn = solid_composition(X, P, mnrl_mode_coeff, eNd)

    # Reaction coefficients | Proportions of phases entering the melt
    if X > X_0:
        pn = (Fn_0 * (1 - X_0) - Fn * (1 - X)) / (X - X_0)
    else:
        pn = np.zeros(6)

    # Partition coefficients between all elements and mineral phases considered
    Dn = partition_coefficient(P, T, X, ele_ind, D, ri, val, Fn)

    return Fn, pn, Dn


@cc.export("al_phase_select", "f8(f8[:], f8, f8)")
@cc.export("al_phase_select_array", "f8[:](f8[:], f8[:], f8[:])")
@njit(["f8(f8[:], f8, f8)", "f8[:](f8[:], f8[:], f8[:])"])
def al_phase_select(Fn, spl, gnt):
    """Select appropriate values based on the stability of the aluminous phase;
    linear interpolation is performed within transitions.
    """
    if Fn[4] + Fn[5] > 0:
        return (spl * Fn[4] + gnt * Fn[5]) / (Fn[4] + Fn[5])
    else:
        return spl


@cc.export(
    "solid_composition", "f8[:](f8, f8, DictType(unicode_type, f8[:]), f8)"
)
@njit("f8[:](f8, f8, DictType(unicode_type, f8[:]), f8)")
def solid_composition(X, P, mnrl_mode_coeff, eNd):
    """Determine modal mineral abundances in the residual solid"""
    modal_ab = {}
    Fn = np.zeros(6)  # Order: [ol, opx, cpx, plg, spl, gnt]
    poly_var = np.asarray(
        [
            X**2 * P**2,
            X**2 * P,
            X**2,
            X * P**2,
            X * P,
            X,
            P**2,
            P,
            1,
        ]
    )

    # Modal abundances of parameterised minerals
    for mnrl, coeff in mnrl_mode_coeff.items():
        modal_ab[mnrl] = np.ascontiguousarray(coeff) @ poly_var
    Fn[0], Fn[2] = modal_ab["ol"], modal_ab["cpx"]
    Fn[3], Fn[4], Fn[5] = modal_ab["plg"], modal_ab["spl"], modal_ab["gnt"]
    Fn.clip(0, 1, Fn)

    # Deduce orthopyroxene mode and ensure modal abundances sum to 1
    sum_modes_ol_cpx_al = Fn.sum()
    if sum_modes_ol_cpx_al <= 1:  # Remaining proportion -> orthopyroxene
        Fn[1] = 1 - sum_modes_ol_cpx_al
    else:  # Enforce proportions sum to 1
        Fn /= sum_modes_ol_cpx_al

    # Account for systematic differences between primitive and depleted modal
    # abundances using the correction (Figure 3) of
    # Kimura and Kawabata - G-Cubed (2014)
    KK2014 = 0.04 * eNd / 10
    Fn[0] += KK2014
    Fn[1] -= max(KK2014 - Fn[2], 0)
    Fn[2] -= min(KK2014, Fn[2])

    return Fn


@cc.export("part_coeff_ideal_radius", "f8[:](f8, f8, f8, f8[:], f8)")
@cc.export("part_coeff_ideal_radius_single", "f8(f8, f8, f8, f8, f8)")
@njit(["f8[:](f8, f8, f8, f8[:], f8)", "f8(f8, f8, f8, f8, f8)"])
def part_coeff_ideal_radius(D0, E, r0, ri, T):
    """Calculate the partition coefficient between a cation and a mineral at a
    crystal site according to the lattice strain model (more information can be
    found at https://doi.org/10.1007/978-3-319-39312-4_347).
    Equations 11 and 12 of Brice - Journal of Crystal Growth (1975)"""
    return D0 * np.exp(
        (-4 * pi * E * Avogadro / R / T)
        * (r0 / 2 * (ri - r0) ** 2 + (ri - r0) ** 3 / 3)
    )


@cc.export("part_coeff_same_charge", "f8[:](f8, f8, f8, f8[:], f8, f8)")
@njit("f8[:](f8, f8, f8, f8[:], f8, f8)")
def part_coeff_same_charge(Da, E, r0, ri, ra, T):
    """Calculate the partition coefficient between a cation and a mineral at a
    crystal site from the value of a cation of similar charge but different
    radius.
    Equation 3 of Blundy and Wood - Nature (1994)"""
    return Da * np.exp(
        (-4 * pi * E * Avogadro / R / T)
        * (r0 / 2 * (ra**2 - ri**2) + (ri**3 - ra**3) / 3)
    )


@cc.export(
    "partition_coefficient",
    "f8[:, :](f8, f8, f8, u8[:], f8[:, :], f8[:, :], u8[:], f8[:])",
)
@njit("f8[:, :](f8, f8, f8, u8[:], f8[:, :], f8[:, :], u8[:], f8[:])")
def partition_coefficient(P, T, X, ele_ind, D, ri, val, Fn):
    """Determine the partition coefficient of each cation/mineral pair."""
    Dn = D.copy()  # Initialise Dn with constant partition coefficients

    # Isolate desired combinations of valence, coordination and element groups
    mask_11 = (val == 1) & np.isfinite(ri[:, 1])
    mask_21 = (val == 2) & np.isfinite(ri[:, 1])
    # mask_30_const = (
    #     (val == 3)
    #     & np.isfinite(ri[:, 0])
    #     & np.asarray([item not in const for item in ele_ind])
    # )
    mask_31_const = (
        (val == 3)
        & np.isfinite(ri[:, 1])
        & np.asarray([item not in const for item in ele_ind])
    )
    mask_40_hfse = (
        (val == 4)
        & np.isfinite(ri[:, 0])
        & np.asarray([item in hfse for item in ele_ind])
    )
    mask_41_radio = (
        (val == 4)
        & np.isfinite(ri[:, 1])
        & np.asarray([item in radio for item in ele_ind])
    )

    # # # # # #
    # Olivine #
    # # # # # #
    if mask_31_const.any():
        # Al content in olivine in mole per four-oxygen
        X_Al = al_phase_select(Fn, 1.56e-3, 5.64e-3)
        # Forsterite content in olivine | Magnesium number
        Mg_num = al_phase_select(Fn, 0.059 * X + 0.904, 0.07 * X + 0.897)

        # if mask_30_const.any():
        #     # Equations 13-15 of Sun and Liang - Chemical Geology (2013)
        #     D0 = np.exp(-0.67 - 0.17 * P + 117.3 * X_Al - 1.47 * Mg_num)
        #     r0, E = 0.725, 442
        #     # Equations 17-19 of Sun and Liang - Chemical Geology (2013)
        #     D0 = np.exp(-0.45 - 0.11 * P + 1.54 * X_Al_melt - 1.94 * Mg_num)
        #     r0, E = 0.720, 426
        #     Dn[mask_30_const, 0] = part_coeff_ideal_radius(
        #         D0, E * 1e9, r0 * 1e-10, ri[mask_30_const, 0], T
        #     )
        if mask_31_const.any():
            # Equations A1a-c of Sun and Liang - Chemical Geology (2014)
            D0 = np.exp(-0.44 - 0.18 * P + 123.75 * X_Al - 1.49 * Mg_num)
            r0, E = 0.809, 298
            Dn[mask_31_const, 0] = part_coeff_ideal_radius(
                D0, E * 1e9, r0 * 1e-10, ri[mask_31_const, 1], T
            )
    # # # # # # # # #
    # Orthopyroxene #
    # # # # # # # # #
    if mask_21.any() or mask_31_const.any() or mask_40_hfse.any():
        # Cation content of Ca on the M2 site in pyroxene per six-oxygen
        X_Ca_M2 = -0.756 * X**2 + 0.273 * X + 0.063
        # Cation content of Mg on the M1 site in pyroxene per six-oxygen
        X_Mg_M1 = -0.395 * X**2 + 0.299 * X + 0.801
        # Cation content of Mg on the M2 site in pyroxene per six-oxygen
        X_Mg_M2 = 0.692 * X**2 - 0.176 * X + 0.834
        # Cation content of Fe on the M1 site in pyroxene per six-oxygen
        X_Fe_M1 = -0.037 * X**2 - 0.030 * X + 0.083
        # Cation content of the tetrahedral Al in pyroxene per six-oxygen
        X_Al_T = -0.675 * X**2 + 0.041 * X + 0.146
        # Cation fraction of Ti in the melt per six-oxygen
        X_Ti_melt = al_phase_select(Fn, -0.014 * X + 0.012, -0.047 * X + 0.027)

        # # Equations 11-13 from Yao et al. - Contrib. to Min. and Pet. (2012)
        # D0 = np.exp(-5.37 + 3.87e4 / R / T + 3.56 * X_Ca_M2 + 3.54 * X_Al_T)
        # r0 = 0.69 + 0.43 * X_Ca_M2 + 0.23 * X_Mg_M2
        # E = (-1.37 + 1.85 * r0 - 0.53 * X_Ca_M2) * 1e3
        # Equations A2a-c of Sun and Liang - Chemical Geology (2014)
        D0 = np.exp(
            (-5.37 + 3.87e4 / R / T)
            + (3.54 * X_Al_T + 3.56 * X_Ca_M2 - 0.84 * X_Ti_melt)
        )
        r0 = 0.693 + 0.432 * X_Ca_M2 + 0.228 * X_Mg_M2
        E = (1.85 * r0 - 1.37 - 0.53 * X_Ca_M2) * 1e3

        if mask_21.any():
            # Section 3.11.10.6.3 of Wood and Blundy - Treatise on Geo. (2014)
            D_Mg, r_Mg = 1, 0.89e-10
            r0_v2, E_v2 = r0 + 0.08, 2 / 3 * E
            Dn[mask_21, 1] = part_coeff_same_charge(
                D_Mg, E_v2 * 1e9, r0_v2 * 1e-10, ri[mask_21, 1], r_Mg, T
            )
        if mask_31_const.any():
            Dn[mask_31_const, 1] = part_coeff_ideal_radius(
                D0, E * 1e9, r0 * 1e-10, ri[mask_31_const, 1], T
            )
        if mask_40_hfse.any():
            # Equations 9-11 of Sun and Liang - G. et C. Acta (2013)
            D0 = np.exp(
                (-4.825 + 3.178e4 / R / T + 4.172 * X_Al_T)
                + (8.551 * X_Ca_M2 * X_Mg_M2 - 2.616 * X_Fe_M1)
            )
            r0 = 0.618 + 0.032 * X_Ca_M2 + 0.03 * X_Mg_M1
            E = 2203
            Dn[mask_40_hfse, 1] = part_coeff_ideal_radius(
                D0, E * 1e9, r0 * 1e-10, ri[mask_40_hfse, 0], T
            )
    # # # # # # # # #
    # Clinopyroxene #
    # # # # # # # # #
    if (
        mask_11.any()
        or mask_21.any()
        or mask_31_const.any()
        or mask_41_radio.any()
    ):
        # Atomic fraction of Mg atoms on the M1 site per six-oxygen
        X_Mg_M1 = al_phase_select(Fn, 0.425 * X + 0.741, 0.191 * X + 0.793)
        # Cation content of Mg on the M2 site in pyroxene per six-oxygen
        X_Mg_M2 = al_phase_select(Fn, 0.583 * X + 0.223, 0.422 * X + 0.547)
        # Total Mg atoms per six oxygens in the liquid
        X_Mg_melt = al_phase_select(Fn, 0.14 * X + 0.722, 0.207 * X + 0.701)
        # Cation content of Al on the M1 site in pyroxene per six-oxygen
        X_Al_M1 = al_phase_select(Fn, -0.438 * X + 0.137, -0.114 * X + 0.099)
        # Cation content of the tetrahedral Al in pyroxene per six-oxygen
        X_Al_T = al_phase_select(Fn, -0.177 * X + 0.154, -0.013 * X + 0.061)
        # # Atomic fraction of cations with charge 1+ in the M2-site
        # X_Na_M2 = al_phase_select(
        #     Fn, 4.7e-3 / (X + 0.0615) + 9.6e-4, -0.0696 * X + 0.0509
        # )
        # Molar fraction of H2O in the melt
        X_H2O_melt = 0

        # Equations 8-10 from Sun and Liang - Contrib. to Min. and Pet. (2012)
        D0 = np.exp(
            (-7.14 + 7.19e4 / R / T)
            + (4.37 * X_Al_T + 1.98 * X_Mg_M2 - 0.91 * X_H2O_melt)
        )
        r0 = 1.066 - 0.104 * X_Al_M1 - 0.212 * X_Mg_M2
        E = (2.27 * r0 - 2) * 1e3

        if mask_11.any():
            # Section 3.11.10.1.3 of Wood and Blundy - Treatise on Geo. (2014)
            D_Na = np.exp(
                (10_367 + 2100 * P - 165 * P**2) / T
                + (-10.27 + 0.358 * P - 0.0184 * P**2)
            )
            r_Na = 1.18e-10
            r0_v1, E_v1 = r0 + 0.12, E / 3
            Dn[mask_11, 2] = part_coeff_same_charge(
                D_Na, E_v1 * 1e9, r0_v1 * 1e-10, ri[mask_11, 1], r_Na, T
            )
        if mask_21.any():
            # Section 3.11.10.1.2 of Wood and Blundy - Treatise on Geo. (2014)
            # Partition coefficient for Ca could be parameterised using data
            # from Hill, Blundy and Wood - Contrib. to Min. and Pet. (2011)
            D_Ca, r_Ca = 2, 1.12e-10
            r0_v2, E_v2 = r0 + 0.06, 2 / 3 * E
            Dn[mask_21, 2] = part_coeff_same_charge(
                D_Ca, E_v2 * 1e9, r0_v2 * 1e-10, ri[mask_21, 1], r_Ca, T
            )
        if mask_31_const.any():
            Dn[mask_31_const, 2] = part_coeff_ideal_radius(
                D0, E * 1e9, r0 * 1e-10, ri[mask_31_const, 1], T
            )
        # if mask_40_hfse.any():
        #     # Equations 7 and 3 from Erratum (2012) to
        #     # Hill, Blundy and Wood - Contrib. to Min. and Pet. (2011)
        #     r0_v4 = 0.659 - 0.008 * P + 0.028 * X_Al_M1
        #     E_v4 = 11_228 - 5.74 * T + 15_204 * X_Al_T
        #     # Titanium partitioning section from
        #     # Hill, Blundy and Wood - Contrib. to Min. and Pet. (2011)
        #     # Assumes X_Si_T = 1 - X_Al_T and Na accounts for all 1+ cations
        #     # in the M2 site
        #     X_Si_T = 1 - X_Al_T
        #     X_v2 = (1 - X_Na_M2) * X_Si_T**2
        #     X_v3 = (
        #         X_Na_M2 * X_Si_T**2
        #         + 2 * (1 - X_Na_M2) * X_Al_M1 * X_Si_T * X_Al_T
        #     )
        #     X_v4 = (1 - X_Na_M2) * X_Al_T**2 + 2 * X_Na_M2 * X_Al_T * X_Si_T
        #     # Equation 15 from Erratum (2012) to
        #     # Hill, Blundy and Wood - Contrib. to Min. and Pet. (2011)
        #     delta_G = 14_000
        #     D_Ti = (
        #         X_v4
        #         + X_v3 * np.exp(-delta_G / R / T)
        #         + X_v2 * np.exp(-4 * delta_G / R / T)
        #     ) * np.exp((35_730 - 2183 * P - 1457 * P**2) / R / T)
        #     r_Ti = 0.605e-10
        #     Dn[mask_40_hfse, 2] = part_coeff_same_charge(
        #         D_Ti, E_v4 * 1e9, r0_v4 * 1e-10, ri[mask_40_hfse, 0], r_Ti, T
        #     )
        if mask_41_radio.any():
            # Section 3.11.10.1.4 of Wood and Blundy - Treatise (2014)
            r_Th = 1.041e-10
            r0_v4, E_v4 = r0, 4 / 3 * E
            gam_Th_M2 = 1 / part_coeff_ideal_radius(
                1, E_v4 * 1e9, r0_v4 * 1e-10, r_Th, T
            )
            gam_Mg_M1 = np.exp(902 * (1 - X_Mg_M1) ** 2 / T)
            D_Th = (X_Mg_melt / X_Mg_M1 / gam_Mg_M1 / gam_Th_M2) * np.exp(
                (214_790 - 175.7 * T + 16_420 * P - 1500 * P**2) / R / T
            )
            Dn[mask_41_radio, 2] = part_coeff_same_charge(
                D_Th, E_v4 * 1e9, r0_v4 * 1e-10, ri[mask_41_radio, 1], r_Th, T
            )
    # # # # # # # #
    # Plagioclase #
    # # # # # # # #
    # if mask_11.any() or mask_21.any() or mask_31_const.any():
    #     # Ca content in plagioclase per eight-oxygen
    #     X_Ca = np.nan
    #     # Na content in plagioclase per eight-oxygen
    #     X_Na = np.nan
    #     if mask_11.any():
    #         # Equations 8a-c from Sun, Graff and Liang - G. et C. Acta (2017)
    #         D0 = np.exp(
    #             -9.99 + (11.37 + 0.49 * P) / R / T * 1e4 + 1.70 * X_Ca**2
    #         )
    #         r0, E = 1.213, 47
    #         Dn[mask_11, 3] = part_coeff_ideal_radius(
    #             D0, E * 1e9, r0 * 1e-10, ri[mask_11, 1], T
    #         )
    #     if mask_21.any():
    #         # Equations 7a-c from Sun, Graff and Liang - G. et C. Acta (2017)
    #         D0 = np.exp((6910 - 2542 * P**2) / R / T + 2.39 * X_Na**2)
    #         r0 = 1.189 + 0.075 * X_Na
    #         E = 719 - 487 * r0
    #         Dn[mask_21, 3] = part_coeff_ideal_radius(
    #             D0, E * 1e9, r0 * 1e-10, ri[mask_21, 1], T
    #         )
    #     if mask_31_const.any():
    #         # Equations 6a-c from Sun, Graff and Liang - G. et C. Acta (2017)
    #         D0 = np.exp(
    #             16.05
    #             - (19.45 + 1.17 * P**2) / R / T * 1e4
    #             - 5.17 * X_Ca**2
    #         )
    #         r0, E = 1.179, 196
    #         Dn[mask_31_const, 3] = part_coeff_ideal_radius(
    #             D0, E * 1e9, r0 * 1e-10, ri[mask_31_const, 1], T
    #         )
    # # # # # #
    # Spinel  #
    # # # # # #
    # # # # # #
    # Garnet  #
    # # # # # #
    if mask_21.any() or mask_31_const.any():
        # Cation content of Ca in garnet per 12-oxygen
        X_Ca = -0.247 * X + 0.355

        # Equations A4a-c of Sun and Liang - Chemical Geology (2014)
        D0 = np.exp(
            -2.01 + (9.03e4 - 93.02 * P * (37.78 - P)) / R / T - 1.04 * X_Ca
        )
        r0 = 0.785 + 0.153 * X_Ca
        E = (-1.67 + 2.35 * r0) * 1e3

        if mask_21.any():
            # Section 3.11.10.3.3 of Wood and Blundy - Treatise on Geo. (2014)
            D_Mg = np.exp(
                (258_210 - 141.5 * T + 5418 * P) / 3 / R / T
            ) / np.exp(1.9e4 * X_Ca**2 / R / T)
            r_Mg = 0.89e-10
            r0_v2, E_v2 = 0.053 + r0, 2 / 3 * E
            Dn[mask_21, 5] = part_coeff_same_charge(
                D_Mg, E_v2 * 1e9, r0_v2 * 1e-10, ri[mask_21, 1], r_Mg, T
            )
        if mask_31_const.any():
            Dn[mask_31_const, 5] = part_coeff_ideal_radius(
                D0, E * 1e9, r0 * 1e-10, ri[mask_31_const, 1], T
            )
        # if mask_40_hfse.any():
        #     # Figure 7 of Mallmann and O'Neill - G. et C. Acta (2007)
        #     D0_v4, r0_v4, E_v4 = 4.38, 0.6626, 2753
        #     # Figure 7 of Dwarzski et al. - American Mineralo. (2006)
        #     D0_v4, r0_v4, E_v4 = 0.4382, 0.6624, 1737
        #     # Table 1 of van Westrenen et al. - Geoch., Geoph., Geosy. (2001)
        #     D0_v4, r0_v4, E_v4 = 1.7, 0.67, 3273
        #     Dn[mask_40_hfse, 5] = part_coeff_ideal_radius(
        #         D0_v4, E_v4 * 1e9, r0_v4 * 1e-10, ri[mask_40_hfse, 0], T
        #     )

    return Dn


if __name__ == "__main__":
    cc.compile()
