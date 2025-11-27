#!/usr/bin/env python3
import numpy as np
from numba import njit
from numba.pycc import CC
from scipy.constants import Avogadro, R, pi

cc = CC("MeltChemistryCompiled")
cc.target_cpu = "host"
cc.verbose = True


@cc.export(
    "mineralogy",
    (
        "Tuple((f8[:], f8[:], f8[:, :]))(f8, f8, f8[:], u8[:], DictType(unicode_type,"
        " u8), u8[:], f8[:, :], f8[:, :], u8[:], DictType(unicode_type, f8[:]), f8,"
        " DictType(unicode_type, f8[:]))"
    ),
)
def mineralogy(
    X,
    X_0,
    Fn_0,
    global_ele_ind,
    ele_ind_map,
    const_ele_ind,
    D,
    ri,
    val,
    mnrl_mode_coeff,
    src_depletion,
    part_arr,
):
    """
    Determine modal mineral abundances, reaction coefficients
    and partition coefficients.

    """
    # Pressure and temperature interpolated from melt fraction
    P = np.interp(X, part_arr["melt_fraction"], part_arr["pressure"])
    T = np.interp(X, part_arr["melt_fraction"], part_arr["temperature"])

    # Modal mineral abundances in the residual solid
    Fn = solid_composition(X, P, mnrl_mode_coeff, src_depletion)

    # Reaction coefficients | Proportions of phases entering the melt
    if X > X_0:
        pn = (Fn_0 * (1 - X_0) - Fn * (1 - X)) / (X - X_0)
    else:
        pn = np.zeros(6)

    # Partition coefficients between all elements and mineral phases considered
    Dn = partition_coefficient(
        P, T, X, global_ele_ind, ele_ind_map, const_ele_ind, D, ri, val, Fn
    )
    return Fn, pn, Dn


@cc.export("al_phase_select", "f8(f8[:], f8, f8)")
@cc.export("al_phase_select_array", "f8[:](f8[:], f8[:], f8[:])")
@njit(["f8(f8[:], f8, f8)", "f8[:](f8[:], f8[:], f8[:])"])
def al_phase_select(Fn, spl, grt):
    """
    Select appropriate values based on the stability of the aluminous phase;
    linear interpolation is performed within transitions.

    """
    if Fn[4] + Fn[5] > 0:
        return (spl * Fn[4] + grt * Fn[5]) / (Fn[4] + Fn[5])
    else:
        return spl


@cc.export("solid_composition", "f8[:](f8, f8, DictType(unicode_type, f8[:]), f8)")
@njit("f8[:](f8, f8, DictType(unicode_type, f8[:]), f8)")
def solid_composition(X, P, mnrl_mode_coeff, src_depletion):
    """
    Determine modal mineral abundances in the residual solid

    """
    modal_ab = {}
    Fn = np.zeros(6)  # Order: [ol, opx, cpx, pl, spl, grt]
    poly_var = np.kron(np.array([X**2, X, 1]), np.array([P**2, P, 1]))

    # Modal abundances of parameterised minerals
    for mnrl, coeff in mnrl_mode_coeff.items():
        modal_ab[mnrl] = np.ascontiguousarray(coeff) @ poly_var
    Fn[0], Fn[2] = modal_ab["ol"], modal_ab["cpx"]
    Fn[3], Fn[4], Fn[5] = modal_ab["pl"], modal_ab["spl"], modal_ab["grt"]
    Fn.clip(0, 1, Fn)

    # Deduce orthopyroxene mode and ensure modal abundances sum to 1
    sum_modes_ol_cpx_al = Fn.sum()
    if sum_modes_ol_cpx_al <= 1:  # Remaining proportion -> orthopyroxene
        Fn[1] = 1 - sum_modes_ol_cpx_al
    else:  # Enforce proportions sum to 1
        Fn /= sum_modes_ol_cpx_al

    # Account for systematic differences between primitive and depleted modal abundances
    # using the correction (Figure 3) of Kimura and Kawabata - G-Cubed (2014)
    KK2014 = 0.04 * src_depletion
    Fn[0] += KK2014
    Fn[1] -= max(KK2014 - Fn[2], 0)
    Fn[2] -= min(KK2014, Fn[2])

    return Fn


@cc.export("part_coeff_ideal_radius", "f8[:](f8, f8, f8, f8[:], f8)")
@cc.export("part_coeff_ideal_radius_single", "f8(f8, f8, f8, f8, f8)")
@njit(["f8[:](f8, f8, f8, f8[:], f8)", "f8(f8, f8, f8, f8, f8)"])
def part_coeff_ideal_radius(D0, E, r0, ri, T):
    """
    Calculate the partition coefficient between a cation and a mineral at a crystal site
    according to the lattice strain model (more information can be found at
    https://doi.org/10.1007/978-3-319-39312-4_347).
    Equations 11 and 12 of Brice - Journal of Crystal Growth (1975)

    """
    return D0 * np.exp(
        -4 * pi * E * Avogadro / R / T * (r0 / 2 * (ri - r0) ** 2 + (ri - r0) ** 3 / 3)
    )


@cc.export("part_coeff_same_charge", "f8[:](f8, f8, f8, f8[:], f8, f8)")
@cc.export("part_coeff_same_charge_single", "f8(f8, f8, f8, f8, f8, f8)")
@njit("f8[:](f8, f8, f8, f8[:], f8, f8)")
def part_coeff_same_charge(Da, E, r0, ri, ra, T):
    """
    Calculate the partition coefficient between a cation and a mineral at a crystal site
    from the value of a cation of similar charge but different radius.
    Equation 3 of Blundy and Wood - Nature (1994)

    """
    return Da * np.exp(
        (-4 * pi * E * Avogadro / R / T)
        * (r0 / 2 * (ra**2 - ri**2) + (ri**3 - ra**3) / 3)
    )


@cc.export(
    "partition_coefficient",
    (
        "f8[:, :](f8, f8, f8, u8[:], DictType(unicode_type, u8), u8[:], f8[:, :], f8[:,"
        " :], u8[:], f8[:])"
    ),
)
@njit(
    "f8[:, :](f8, f8, f8, u8[:], DictType(unicode_type, u8), u8[:], f8[:, :], f8[:, :],"
    " u8[:], f8[:])"
)
def partition_coefficient(
    P, T, X, global_ele_ind, ele_ind_map, const_ele_ind, D, ri, val, Fn
):
    """
    Determine the partition coefficient of each cation/mineral pair.

    """
    # Isolate desired combinations of valence, coordination and element groups
    discard_const = np.asarray(
        [ele_ind not in const_ele_ind for ele_ind in global_ele_ind]
    )

    ele_mask_Na = (global_ele_ind == ele_ind_map["Na"]) & discard_const
    ele_mask_Co = (global_ele_ind == ele_ind_map["Co"]) & discard_const
    ele_mask_Ni = (global_ele_ind == ele_ind_map["Ni"]) & discard_const
    ele_mask_Cu = (global_ele_ind == ele_ind_map["Cu"]) & discard_const
    ele_mask_Zn = (global_ele_ind == ele_ind_map["Zn"]) & discard_const
    ele_mask_Sr = (global_ele_ind == ele_ind_map["Sr"]) & discard_const
    ele_mask_Pb = (global_ele_ind == ele_ind_map["Pb"]) & discard_const
    ele_mask_Sc = (global_ele_ind == ele_ind_map["Sc"]) & discard_const
    ele_mask_Cr = (global_ele_ind == ele_ind_map["Cr"]) & discard_const
    ele_mask_Ti = (global_ele_ind == ele_ind_map["Ti"]) & discard_const
    ele_mask_Zr = (global_ele_ind == ele_ind_map["Zr"]) & discard_const
    ele_mask_Hf = (global_ele_ind == ele_ind_map["Hf"]) & discard_const
    ele_mask_Th = (global_ele_ind == ele_ind_map["Th"]) & discard_const
    ele_mask_U = (global_ele_ind == ele_ind_map["U"]) & discard_const
    ele_mask_V = (global_ele_ind == ele_ind_map["V"]) & discard_const
    ele_mask_Nb = (global_ele_ind == ele_ind_map["Nb"]) & discard_const
    ele_mask_Ta = (global_ele_ind == ele_ind_map["Ta"]) & discard_const

    ele_mask_11 = (val == 1) & np.isfinite(ri[:, 1]) & discard_const
    # ele_mask_20 = (val == 2) & np.isfinite(ri[:, 0]) & discard_const
    # ele_mask_21 = (val == 2) & np.isfinite(ri[:, 1]) & discard_const
    ele_mask_31 = (
        (val == 3)
        & np.isfinite(ri[:, 1])
        & discard_const
        & ~(ele_mask_Sc | ele_mask_Cr)
    )
    ele_mask_40_hfse = (
        (val == 4)
        & np.isfinite(ri[:, 0])
        & discard_const
        & (ele_mask_Ti | ele_mask_Zr | ele_mask_Hf)
    )
    # ele_mask_41_radio = (
    #     (val == 4)
    #     & np.isfinite(ri[:, 1])
    #     & discard_const
    #     & (ele_mask_Th | ele_mask_U)
    # )

    # # # # # #
    # Olivine #
    # # # # # #

    # Al content in olivine in mole per four-oxygen
    X_Al = al_phase_select(
        Fn,
        -1.166e-4 * X * T + 1.628e-1 * X + 6.632e-5 * T - 1.001e-1,
        4.717e-6 * X * T - 7.203e-3 * X - 2.738e-6 * T + 1.069e-2,
    )
    # Forsterite content in olivine | Magnesium number
    Mg_num = al_phase_select(
        Fn,
        4.746e-5 * X * T - 2.941e-2 * X + 4.057e-6 * T + 8.990e-1,
        -6.102e-7 * X * T + 6.992e-2 * X + 1.893e-5 * T + 8.604e-1,
    )

    # Table 3 from Le Roux et al. - American Mineralogist (2015)
    D[ele_mask_Co, 0] = al_phase_select(Fn, 2.1, 2.37)
    D[ele_mask_Ni, 0] = al_phase_select(Fn, 6.2, 6.2)
    D[ele_mask_Cu, 0] = al_phase_select(Fn, 0.13, 0.13)
    D[ele_mask_Zn, 0] = al_phase_select(Fn, 0.99, 0.96)
    D[ele_mask_Sc, 0] = al_phase_select(Fn, 0.20, 0.15)
    D[ele_mask_Cr, 0] = al_phase_select(Fn, 0.8, 0.79)
    D[ele_mask_V, 0] = al_phase_select(Fn, 0.10, 0.14)

    # Equations A1a-c of Sun and Liang - Chemical Geology (2014)
    D0 = np.exp(-0.44 - 0.18 * P + 123.75 * X_Al - 1.49 * Mg_num)
    r0, E = 0.809, 298
    D[ele_mask_31, 0] = part_coeff_ideal_radius(
        D0, E * 1e9, r0 * 1e-10, ri[ele_mask_31, 1], T
    )

    # # # # # # # # #
    # Orthopyroxene #
    # # # # # # # # #

    # # Cation content of Al per six-oxygen
    # X_Al = al_phase_select(
    #     Fn,
    #     -9.015e-4 * X * T + 7.717e-1 * X + 5.431e-4 * T - 5.914e-1,
    #     1.832e-3 * X * T - 3.710 * X - 9.000e-4 * T + 1.953,
    # )
    # Cation content of Al on the tetrahedral site per six-oxygen
    X_Al_T = al_phase_select(
        Fn,
        -8.137e-4 * X * T + 1.060 * X + 1.786e-4 * T - 1.333e-1,
        1.070e-3 * X * T - 2.114 * X - 5.671e-4 * T + 1.196,
    )
    # # Cation content of Ca per six-oxygen
    # X_Ca = al_phase_select(
    #     Fn,
    #     -1.097e-3 * X * T + 1.747 * X + 2.037e-4 * T - 2.385e-1,
    #     -1.862e-4 * X * T + 2.338e-1 * X + 1.426e-4 * T - 1.559e-1,
    # )
    # Cation content of Ca on the M2 site per six-oxygen
    X_Ca_M2 = al_phase_select(
        Fn,
        -1.097e-3 * X * T + 1.747 * X + 2.037e-4 * T - 2.385e-1,
        -1.862e-4 * X * T + 2.338e-1 * X + 1.426e-4 * T - 1.559e-1,
    )
    # Cation content of Fe on the M1 site per six-oxygen
    X_Fe_M1 = al_phase_select(
        Fn,
        -1.392e-4 * X * T + 2.231e-1 * X - 2.464e-5 * T + 1.158e-1,
        -1.032e-5 * X * T - 1.845e-2 * X - 3.754e-6 * T + 8.845e-2,
    )
    # Cation content of Mg on the M1 site per six-oxygen
    X_Mg_M1 = al_phase_select(
        Fn,
        -6.692e-5 * X * T + 6.084e-1 * X - 4.662e-4 * T + 1.507,
        -6.528e-4 * X * T + 1.442 * X + 3.373e-4 * T + 1.330e-1,
    )
    # Cation content of Mg on the M2 site per six-oxygen
    X_Mg_M2 = al_phase_select(
        Fn,
        1.274e-3 * X * T - 1.969 * X - 2.631e-4 * T + 1.242,
        1.386e-4 * X * T - 6.299e-2 * X - 1.497e-4 * T + 1.063,
    )
    # Cation fraction of Ti in the melt per six-oxygen
    X_Ti_melt = al_phase_select(
        Fn,
        2.087e-4 * X * T - 4.405e-1 * X + 1.498e-5 * T + 1.156e-2,
        1.225e-4 * X * T - 3.037e-1 * X - 2.206e-5 * T + 8.261e-2,
    )

    # Equations 56a and 56b from Bedard - Chemical Geology (2007)
    if P < 3.14:
        D[ele_mask_Na, 1] = np.exp(-4.291548 + 0.719040 * P)
    else:
        D[ele_mask_Na, 1] = np.exp(-3.041418 + 0.320851 * P)

    # Table 3 from Le Roux et al. - American Mineralogist (2015)
    D[ele_mask_Co, 1] = al_phase_select(Fn, 1.04, 1.29)
    D[ele_mask_Ni, 1] = al_phase_select(Fn, 3.7, 3.7)
    D[ele_mask_Cu, 1] = al_phase_select(Fn, 0.12, 0.12)
    D[ele_mask_Zn, 1] = al_phase_select(Fn, 0.68, 0.451)
    D[ele_mask_Sc, 1] = al_phase_select(Fn, 0.35, 0.495)
    D[ele_mask_Cr, 1] = al_phase_select(Fn, 2.5, 8.8)
    D[ele_mask_V, 1] = al_phase_select(Fn, 0.30, 1.06)

    # # Section 3.11.10.6.3 of Wood and Blundy - Treatise on Geochemistry (2014)
    # D_Mg, r_Mg = 1, 0.72e-10
    # r0 = 0.753 + 0.118 * X_Al + 0.114 * X_Ca + 0.08
    # E = 240
    # D[ele_mask_20, 1] = part_coeff_same_charge(
    #     D_Mg, E * 1e9, r0 * 1e-10, ri[ele_mask_20, 0], r_Mg, T
    # )

    # Equations 4-6 of Sun and Liang - Geochimica et Cosmochimica Acta (2013)
    D0 = np.exp(
        -5.37 + 3.87e4 / R / T + 3.54 * X_Al_T + 3.56 * X_Ca_M2 - 0.84 * X_Ti_melt
    )
    r0 = 0.693 + 0.432 * X_Ca_M2 + 0.228 * X_Mg_M2
    E = (1.85 * r0 - 1.37 - 0.53 * X_Ca_M2) * 1e3
    D[ele_mask_31, 1] = part_coeff_ideal_radius(
        D0, E * 1e9, r0 * 1e-10, ri[ele_mask_31, 1], T
    )

    # Equations 9-11 of Sun and Liang - Geochimica et Cosmochimica Acta (2013)
    D0 = np.exp(
        -4.825
        + 3.178e4 / R / T
        + 4.172 * X_Al_T
        + 8.551 * X_Ca_M2 * X_Mg_M2
        - 2.616 * X_Fe_M1
    )
    r0 = 0.618 + 0.032 * X_Ca_M2 + 0.03 * X_Mg_M1
    E = 2203
    D[ele_mask_40_hfse, 1] = part_coeff_ideal_radius(
        D0, E * 1e9, r0 * 1e-10, ri[ele_mask_40_hfse, 0], T
    )

    # # # # # # # # #
    # Clinopyroxene #
    # # # # # # # # #

    # Cation content of Al on the M1 site per six-oxygen
    X_Al_M1 = al_phase_select(
        Fn,
        -3.546e-3 * X * T + 5.281 * X + 5.916e-4 * T - 7.892e-1,
        5.612e-5 * X * T - 4.683e-2 * X - 3.754e-4 * T + 8.188e-1,
    )
    # Cation content of Al on the tetrahedral site per six-oxygen
    X_Al_T = al_phase_select(
        Fn,
        -1.493e-3 * X * T + 2.142 * X + 2.126e-4 * T - 1.765e-1,
        -1.008e-4 * X * T + 3.649e-1 * X - 4.349e-4 * T + 8.978e-1,
    )
    # Molar fraction of Al2O3 in the melt
    X_Al2O3_melt = al_phase_select(
        Fn,
        3.055e-4 * X * T - 6.387e-1 * X - 1.313e-4 * T + 3.259e-1,
        1.264e-4 * X * T - 2.287e-1 * X - 1.855e-4 * T + 4.052e-1,
    )
    # Cation content of Ca on the M2 site per six-oxygen
    X_Ca_M2 = al_phase_select(
        Fn,
        -2.459e-3 * X * T + 4.017 * X - 1.587e-3 * T + 3.098,
        -1.699e-4 * X * T + 1.979e-1 * X - 2.198e-4 * T + 7.507e-1,
    )
    # # Cation content of Mg on the M1 site per six-oxygen
    # X_Mg_M1 = al_phase_select(
    #     Fn,
    #     3.120e-3 * X * T - 4.628 * X - 6.414e-4 * T + 1.744,
    #     -1.519e-4 * X * T + 2.845e-1 * X + 4.539e-4 * T - 8.000e-2,
    # )
    # Cation content of Mg on the M2 site per six-oxygen
    X_Mg_M2 = al_phase_select(
        Fn,
        3.000e-3 * X * T - 4.675 * X + 1.270e-3 * T - 1.688,
        -5.157e-6 * X * T + 2.422e-1 * X + 2.201e-4 * T + 1.312e-1,
    )
    # # Cation content of Mg in the melt per six-oxygen
    # X_Mg_melt = al_phase_select(
    #     Fn,
    #     -1.040e-4 * X * T + 1.011 * X + 1.333e-3 * T - 1.608,
    #     -3.530e-4 * X * T + 1.011 * X + 1.668e-3 * T - 2.098,
    # )
    # Weight per cent of MgO in the melt
    X_MgO_melt = al_phase_select(
        Fn,
        1.039e-3 * X * T + 1.433e1 * X + 2.330e-2 * T - 2.742e1,
        -3.266e-3 * X * T + 1.423e1 * X + 2.762e-2 * T - 3.341e1,
    )
    # # Cation content of Na on the M2 site per six-oxygen
    # X_Na_M2 = al_phase_select(
    #     Fn,
    #     -6.168e-4 * X * T + 7.971e-1 * X + 1.321e-4 * T - 1.560e-1,
    #     8.687e-5 * X * T - 2.509e-1 * X + 2.250e-5 * T + 7.368e-3,
    # )
    # # Cation content of the tetrahedral Si per six-oxygen
    # X_Si_T = al_phase_select(
    #     Fn,
    #     1.493e-3 * X * T - 2.142 * X - 2.126e-4 * T + 2.177,
    #     1.008e-4 * X * T - 3.649e-1 * X + 4.349e-4 * T + 1.102,
    # )
    # Molar fraction of SiO2 in the melt
    X_SiO2_melt = al_phase_select(
        Fn,
        5.050e-4 * X * T - 8.057e-1 * X - 3.469e-4 * T + 1.065,
        8.899e-8 * X * T + 2.692e-2 * X - 1.265e-4 * T + 6.800e-1,
    )
    # # Molar ratio Mg / (Mg + Fe) in the melt
    # Mg_num_melt = al_phase_select(
    #     Fn,
    #     -5.064e-5 * X * T + 2.595e-1 * X + 5.130e-5 * T + 6.581e-1,
    #     8.293e-5 * X * T - 1.510e-2 * X + 3.967e-5 * T + 6.735e-1,
    # )
    # Molar ratio Mg / (Mg + Fe) in clinopyroxene
    Mg_num = al_phase_select(
        Fn,
        7.865e-4 * X * T - 1.185 * X - 1.340e-4 * T + 1.112,
        -1.785e-4 * X * T + 4.067e-1 * X + 7.077e-5 * T + 7.590e-1,
    )

    # # Molar fraction of H2O in the melt (not parameterised)
    # X_H2O_melt = 0

    # Shannon - Acta Crystallographica (1976)
    r_Na = 1.18e-10
    # r_Ca = 1.12e-10
    r_Sm = 1.079e-10
    # Blundy and Wood - Reviews in Mineralogy and Geochemistry (2003)
    # r_Th = 1.041e-10

    # Partition coefficient for Ca - Run 1948
    # Adam and Green - Contributions to Mineralogy and Petrology (2006)
    D_Ca = 1.99
    # Equation 84 from Schoneveld - ANU Thesis (2018)
    D0 = np.exp(-2.07 + 0.52e4 / T)
    r0 = 1.036 - 0.08 * X_Mg_M2
    E = 238
    γ_REE = 2.54 * np.log(X_SiO2_melt)
    D_Sm = (
        part_coeff_ideal_radius(D0, E * 1e9, r0 * 1e-10, r_Sm, T)
        * np.exp(γ_REE)
        * 2
        * X_Al2O3_melt
        / X_SiO2_melt
        * D_Ca
    )

    # Equation S388c from Bedard - Geochemistry, Geophysics, Geosystems (2014)
    D_Ti = np.exp((np.log(D_Sm) + 0.162190) / 0.999163)
    D[ele_mask_Ti, 2] = D_Ti
    # Equation S133 from Bedard - Geochemistry, Geophysics, Geosystems (2014)
    D_Hf = np.exp(-0.436056 + 1.027121 * np.log(D_Ti))
    D[ele_mask_Hf, 2] = D_Hf
    # Equation S214 from Bedard - Geochemistry, Geophysics, Geosystems (2014)
    D_Th = np.exp(-2.78091 + 1.76865 * np.log(D_Ti))
    D[ele_mask_Th, 2] = D_Th
    # # Equation 50 of Wood and Blundy - Treatise on Geochemistry (2014)
    # D_Ta = 10 ** (-2.127 + 3.769 * X_Al_T)
    # Equation S92 from Bedard - Geochemistry, Geophysics, Geosystems (2014)
    D_Ta = np.exp(-4.92448 + 8.40847 * X_Al_T)
    D[ele_mask_Ta, 2] = D_Ta

    # Equation S19 from Bedard - Geochemistry, Geophysics, Geosystems (2014)
    D[ele_mask_Sr, 2] = np.exp(-1.87504 - 0.23387 * np.log(X_MgO_melt))
    # Equation S240 from Bedard - Geochemistry, Geophysics, Geosystems (2014)
    D[ele_mask_Pb, 2] = np.exp(-0.20031 - 4.51647 * Mg_num)
    # Equation S119 from Bedard - Geochemistry, Geophysics, Geosystems (2014)
    D[ele_mask_Zr, 2] = np.exp(-0.48986 + 1.071278 * np.log(D_Hf))
    # Equation S224 from Bedard - Geochemistry, Geophysics, Geosystems (2014)
    D[ele_mask_U, 2] = np.exp(-0.78777 + 0.885892 * np.log(D_Th))
    # # Equation 51 of Wood and Blundy - Treatise on Geochemistry (2014)
    # D[ele_mask_Nb, 2] = 0.003 + 0.292 * D_Ta
    # Equation S90 from Bedard - Geochemistry, Geophysics, Geosystems (2014)
    D[ele_mask_Nb, 2] = np.exp(-1.31570 + 0.88397 * np.log(D_Ta))

    # Table 3 from Le Roux et al. - American Mineralogist (2015)
    D[ele_mask_Co, 2] = al_phase_select(Fn, 1.06, 0.86)
    D[ele_mask_Ni, 2] = al_phase_select(Fn, 3.2, 22.0)
    D[ele_mask_Cu, 2] = al_phase_select(Fn, 0.09, 0.09)
    D[ele_mask_Zn, 2] = al_phase_select(Fn, 0.48, 0.333)
    D[ele_mask_Sc, 2] = al_phase_select(Fn, 1.51, 0.84)
    D[ele_mask_Cr, 2] = al_phase_select(Fn, 8.0, 7.5)
    D[ele_mask_V, 2] = al_phase_select(Fn, 0.8, 1.48)

    # Section 3.11.10.1.3 of Wood and Blundy - Treatise on Geochemistry (2014)
    # r0 revision deduced from Figure 8 of Mollo et al. - Earth-Science Reviews (2020)
    D_Na = np.exp(
        (10_367 + 2100 * P - 165 * P**2) / T - 10.27 + 0.358 * P - 0.0184 * P**2
    )
    r0 = 0.974 + 0.067 * X_Ca_M2 - 0.051 * X_Al_M1 + 0.23
    E = (318.6 + 6.9 * P - 0.036 * T) / 3
    # # Equation 78 from Schoneveld - ANU Thesis (2018)
    # D_Na = np.exp(-2.62 + 0.63 * P + 493 / T)
    # r0 = 1.05 + 0.03 * X_Al_M1
    # E = 29.46
    D[ele_mask_11, 2] = part_coeff_same_charge(
        D_Na, E * 1e9, r0 * 1e-10, ri[ele_mask_11, 1], r_Na, T
    )

    # Section 3.11.10.1.2 of Wood and Blundy - Treatise on Geochemistry (2014)
    # r0 = 0.974 + 0.067 * X_Ca_M2 - 0.051 * X_Al_M1 + 0.06
    # E = 2 * (318.6 + 6.9 * P - 0.036 * T) / 3
    # D[ele_mask_21, 2] = part_coeff_same_charge(
    #     D_Ca, E * 1e9, r0 * 1e-10, ri[ele_mask_21, 1], r_Ca, T
    # )

    # # Section 3.11.10.1.1 of Wood and Blundy - Treatise on Geochemistry (2014)
    # D0 = (
    #     Mg_num_melt
    #     / X_Mg_M1
    #     * np.exp((88_750 - 65.644 * T + 7050 * P - 770 * P**2) / R / T)
    # )
    # r0 = 0.974 + 0.067 * X_Ca_M2 - 0.051 * X_Al_M1
    # E = 318.6 + 6.9 * P - 0.036 * T
    # # Equations 8-10 from
    # # Sun and Liang - Contributions to Mineralogy and Petrology (2012)
    # D0 = np.exp(
    #     -7.14 + 7.19e4 / R / T + 4.37 * X_Al_T + 1.98 * X_Mg_M2 - 0.91 * X_H2O_melt
    # )
    # r0 = 1.066 - 0.104 * X_Al_M1 - 0.212 * X_Mg_M2
    # E = (2.27 * r0 - 2) * 1e3
    # D[ele_mask_31, 2] = part_coeff_ideal_radius(
    #     D0, E * 1e9, r0 * 1e-10, ri[ele_mask_31, 1], T
    # )

    # Equation 84 from Schoneveld - ANU Thesis (2018)
    D0 = np.exp(-2.07 + 0.52e4 / T)
    r0 = 1.036 - 0.08 * X_Mg_M2
    E = 238
    D[ele_mask_31, 2] = (
        part_coeff_ideal_radius(D0, E * 1e9, r0 * 1e-10, ri[ele_mask_31, 1], T)
        * np.exp(γ_REE)
        * 2
        * X_Al2O3_melt
        / X_SiO2_melt
        * D_Ca
    )

    # # Table 2 from Bedard - Geochemistry, Geophysics, Geosystems (2014)
    # ln_D_Sm = 1.8e4 / T - 13

    # D0_M2 = np.exp(
    #     0.17177 + 0.86352 * ln_D_Sm + 0.02549 * ln_D_Sm**2 - 0.0020082 * ln_D_Sm**3
    # )
    # r0_M2 = (
    #     1.0401
    #     + 0.013443 * ln_D_Sm
    #     - 0.0017107 * ln_D_Sm**2
    #     + 0.00020599 * ln_D_Sm**3
    # )
    # if ln_D_Sm < -3.0176:
    #     E_M2 = 308.96 + 1.2721 * ln_D_Sm - 1.4171 * ln_D_Sm**2
    # elif ln_D_Sm < -1.95982:
    #     E_M2 = 429.54 + 71.2 * ln_D_Sm + 8.5143 * ln_D_Sm**2
    # else:
    #     E_M2 = 350.71 + 13.104 * ln_D_Sm - 0.60535 * ln_D_Sm**2

    # D0_M1 = np.exp(1.3636 + 1.2849 * ln_D_Sm)
    # if ln_D_Sm < -0.17112:
    #     r0_M1 = 0.74173 - 0.019187 * ln_D_Sm - 0.0040115 * ln_D_Sm**2
    # else:
    #     r0_M1 = 0.744896
    # if ln_D_Sm < -0.15489:
    #     E_M1 = 694.21 - 135.08 * ln_D_Sm + 47.214 * ln_D_Sm**2
    # else:
    #     E_M1 = 721.21 + 31.794 * ln_D_Sm - 0.86319 * ln_D_Sm**2

    # D[ele_mask_31, 2] = (
    #     part_coeff_ideal_radius(
    #         D0_M2, E_M2 * 1e9, r0_M2 * 1e-10, ri[ele_mask_31, 1], T
    #     )
    #     + part_coeff_ideal_radius(
    #         D0_M1, E_M1 * 1e9, r0_M1 * 1e-10, ri[ele_mask_31, 0], T
    #     )
    # )

    # # Equations 7 and 3 from Erratum (2012) to
    # # Hill, Blundy and Wood - Contributions to Mineralogy and Petrology (2011)
    # r0 = 0.659 - 0.008 * P + 0.028 * X_Al_M1
    # E = 11_228 - 5.74 * T + 15_204 * X_Al_T
    # # Titanium partitioning section from
    # # Hill, Blundy and Wood - Contributions to Mineralogy and Petrology (2011)
    # # Assume Na accounts for all 1+ cations in the M2 site
    # X_v2 = (1 - X_Na_M2) * X_Si_T**2
    # X_v3 = X_Na_M2 * X_Si_T**2 + 2 * (1 - X_Na_M2) * X_Al_M1 * X_Si_T * X_Al_T
    # X_v4 = (1 - X_Na_M2) * X_Al_T**2 + 2 * X_Na_M2 * X_Al_T * X_Si_T
    # # Equation 15 from Erratum (2012) to
    # # Hill, Blundy and Wood - Contributions to Mineralogy and Petrology (2011)
    # delta_G = 1.4e4
    # D_Ti = (
    #     X_v4
    #     + X_v3 * np.exp(-delta_G / R / T)
    #     + X_v2 * np.exp(-4 * delta_G / R / T)
    # ) * np.exp((35_730 - 2183 * P - 1457 * P**2) / R / T)
    # r_Ti = 0.605e-10
    # D[ele_mask_40_hfse, 2] = part_coeff_same_charge(
    #     D_Ti, E * 1e9, r0 * 1e-10, ri[ele_mask_40_hfse, 0], r_Ti, T
    # )

    # # Equations from Corrigendum (2015) to
    # # Dygert et al. - Geochimica et Cosmochimica Acta (2014)
    # D0 = np.exp(-15.16 + 16.90e4 / R / T + 2.64 * X_Al_T)
    # r0 = 0.655
    # E = 2.22e3
    # D[ele_mask_40_hfse, 2] = part_coeff_ideal_radius(
    #     D0, E * 1e9, r0 * 1e-10, ri[ele_mask_40_hfse, 0], T
    # )

    # # Section 7.1 of Blundy and Wood - Reviews in Mineralogy and Geochem. (2003)
    # # Section 3.11.10.1.4 of Wood and Blundy - Treatise on Geochemistry (2014)
    # r0 = 0.974 + 0.067 * X_Ca_M2 - 0.051 * X_Al_M1
    # E = 4 * (318.6 + 6.9 * P - 0.036 * T) / 3
    # γ_Th_M2 = 1 / part_coeff_ideal_radius(1, E * 1e9, r0 * 1e-10, r_Th, T)
    # γ_Mg_M1 = np.exp(7.5e3 / R / T)
    # # γ_Mg_M1 = np.exp(902 * (1 - X_Mg_M1) ** 2 / T)
    # D_Th = (X_Mg_melt / X_Mg_M1 / γ_Mg_M1 / γ_Th_M2) * np.exp(
    #     (214_790 - 175.7 * T + 16_420 * P - 1500 * P**2) / R / T
    # )
    # D[ele_mask_41_radio, 2] = part_coeff_same_charge(
    #     D_Th, E * 1e9, r0 * 1e-10, ri[ele_mask_41_radio, 1], r_Th, T
    # )

    # # # # # # # #
    # Plagioclase #
    # # # # # # # #

    # # Ca content in plagioclase per eight-oxygen
    # X_Ca =
    # # Na content in plagioclase per eight-oxygen
    # X_Na =

    # # Equations 8a-c from Sun, Graff and Liang - G. et C. Acta (2017)
    # D0 = np.exp(-9.99 + (11.37 + 0.49 * P) / R / T * 1e4 + 1.70 * X_Ca**2)
    # r0, E = 1.213, 47
    # D[ele_mask_11, 3] = part_coeff_ideal_radius(
    #     D0, E * 1e9, r0 * 1e-10, ri[ele_mask_11, 1], T
    # )

    # # Equations 7a-c from Sun, Graff and Liang - G. et C. Acta (2017)
    # D0 = np.exp((6910 - 2542 * P**2) / R / T + 2.39 * X_Na**2)
    # r0 = 1.189 + 0.075 * X_Na
    # E = 719 - 487 * r0
    # D[ele_mask_21, 3] = part_coeff_ideal_radius(
    #     D0, E * 1e9, r0 * 1e-10, ri[ele_mask_21, 1], T
    # )

    # # Equations 6a-c from Sun, Graff and Liang - G. et C. Acta (2017)
    # D0 = np.exp(16.05 - (19.45 + 1.17 * P**2) / R / T * 1e4 - 5.17 * X_Ca**2)
    # r0, E = 1.179, 196
    # D[ele_mask_31, 3] = part_coeff_ideal_radius(
    #     D0, E * 1e9, r0 * 1e-10, ri[ele_mask_31, 1], T
    # )

    # # # # # #
    # Spinel  #
    # # # # # #

    # # # # # #
    # Garnet  #
    # # # # # #

    # # Cation content of Al in garnet per 12-oxygen
    # X_Al = -5.415e-4 * X * T + 1.158 * X - 3.741e-4 * T + 2.533
    # Cation content of Ca in garnet per 12-oxygen
    X_Ca = 4.516e-5 * X * T - 2.685e-1 * X - 2.325e-4 * T + 7.989e-1
    # # Cation content of Cr in garnet per 12-oxygen
    # X_Cr = -2.164e-4 * X * T + 4.607e-1 * X - 1.692e-5 * T + 1.051e-1
    # # Cation content of Fe in garnet per 12-oxygen
    # X_Fe = 2.836e-4 * X * T - 8.115e-1 * X - 6.649e-5 * T + 4.910e-01
    # # Cation content of Mg in garnet per 12-oxygen
    # X_Mg = 1.707e-4 * X * T + 2.927e-2 * X + 4.541e-4 * T + 1.515
    # # Cation content of Mn in garnet per 12-oxygen
    # X_Mn = -3.803e-6 * X * T + 2.833e-3 * X - 3.300e-6 * T + 1.731e-02

    # # Equations 3, 6 and 18 from
    # # van Westrenen and Draper - Contributions to Mineralogy and Petrology (2007)
    # X_site_grt = X_Mg + X_Ca + X_Fe + X_Mn
    # Y_site_grt = X_Al + X_Cr

    # frac_Ca_X = X_Ca / X_site_grt
    # frac_Al_Y = X_Al / Y_site_grt

    # X_Py = X_Mg / X_site_grt
    # X_Gr = frac_Ca_X * frac_Al_Y
    # X_Alm = X_Fe / X_site_grt
    # X_Spes = X_Mn / X_site_grt
    # X_And = 0
    # X_Uv = frac_Ca_X * (1 - frac_Al_Y)

    # # γ_Fe_grt = np.exp(1.9e4 * frac_Ca_X**2 / R / T)
    # # D_Fe = 2
    # # D0 = np.exp((400_290 + 4586 * P - 218 * T) / R / T) / (γ_Fe_grt * D_Fe) ** 2

    # r0 = (
    #     0.9302 * X_Py
    #     + 0.993 * X_Gr
    #     + 0.916 * X_Alm
    #     + 0.946 * X_Spes
    #     + 1.05 * (X_And + X_Uv)
    #     - 0.0044 * (P - 3)
    #     + 0.000058 * (T - 1818)
    # )
    # E = 2826 * (1.38 + r0) ** -3 + 12.4 * P - 0.072 * T + 237 * (X_Al + X_Cr)
    # # Equations 55, 56 and 57 from
    # # Wood and Blundy - Treatise on Geochemistry (2014)
    # D_Mg = np.exp((258_210 - 141.5 * T + 5418 * P) / 3 / R / T) / np.exp(
    #     1.9e4 * frac_Ca_X**2 / R / T
    # )
    # r0 += 0.053
    # E *= 2 / 3

    # D[ele_mask_21, 5] = part_coeff_same_charge(
    #     D_Mg, E * 1e9, r0 * 1e-10, ri[ele_mask_21, 1], r_Mg, T
    # )

    # Equations A4a-c of Sun and Liang - Chemical Geology (2014)
    D0 = np.exp(-2.01 + (9.03e4 - 93.02 * P * (37.78 - P)) / R / T - 1.04 * X_Ca)
    r0 = 0.785 + 0.153 * X_Ca
    E = (-1.67 + 2.35 * r0) * 1e3
    D[ele_mask_31, 5] = part_coeff_ideal_radius(
        D0, E * 1e9, r0 * 1e-10, ri[ele_mask_31, 1], T
    )

    return D


if __name__ == "__main__":
    cc.compile()
