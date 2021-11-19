#!/usr/bin/env python3
from numba import njit
from numba.pycc import CC
from numpy import exp, interp, isfinite, zeros
from scipy.constants import Avogadro, pi, R


cc = CC("MeltChemistryCompiled")
cc.target_cpu = "host"
cc.verbose = True


# Main melt chemistry function that returns the bulk partition coeffients
@cc.export('mineralogy', 'Tuple((f8[:], f8[:], f8[:, :]))'
           '(f8, f8, f8, f8[:, :], f8[:, :], u8[:], '
           'DictType(unicode_type, f8[:, :]), f8, '
           'DictType(unicode_type, f8[:]), f8, f8)')
def mineralogy(X, X_0, P_0, D, ri, val, quad_poly_coeff, eNd, part_arr,
               X_spl_in, X_gnt_out):
    # Pressure and temperature interpolated from melt fraction
    P = interp(X, part_arr["melt_fraction"], part_arr["pressure"])
    T = interp(X, part_arr["melt_fraction"], part_arr["temperature"])

    # Proportions of phases remaining in the residual solid per aluminous phase
    Fn_spl, Fn_gnt = solid_composition(X, P, quad_poly_coeff, eNd)
    Fn_0_spl, Fn_0_gnt = solid_composition(X_0, P_0, quad_poly_coeff, eNd)

    # Proportions of phases entering the melt per aluminous phase
    if X > 0:
        pn_spl = (Fn_0_spl * (1 - X_0) - Fn_spl * (1 - X)) / (X - X_0)
        pn_gnt = (Fn_0_gnt * (1 - X_0) - Fn_gnt * (1 - X)) / (X - X_0)
    else:
        pn_spl, pn_gnt = zeros(6), zeros(6)

    # Final proportions based on the stability of the aluminous phase
    Fn = al_phase_select(X, X_spl_in, X_gnt_out, Fn_spl, Fn_gnt)
    pn = al_phase_select(X, X_spl_in, X_gnt_out, pn_spl, pn_gnt)

    # Partition coefficients between all elements and mineral phases considered
    Dn = partition_coefficient(P, T, X, D, ri, val, X_spl_in, X_gnt_out)

    return Fn, pn, Dn


# Select appropriate variable values based on the stability of the aluminous
# phase; linear interpolation is performed within transitions
@cc.export('al_phase_select', 'f8(f8, f8, f8, f8, f8)')
@cc.export('al_phase_select_array', 'f8[:](f8, f8, f8, f8[:], f8[:])')
@njit(['f8(f8, f8, f8, f8, f8)', 'f8[:](f8, f8, f8, f8[:], f8[:])'])
def al_phase_select(X, X_spl_in, X_gnt_out, spl, gnt):
    if X < X_spl_in:  # Garnet stability field
        return gnt
    elif X >= X_gnt_out:  # Spinel stability field
        return spl
    else:  # Spinel-Garnet transition
        return (spl - gnt) / (X_gnt_out - X_spl_in) * (X - X_spl_in) + gnt


# Proportions of mineral phases in the residual solid for each aluminous phase
@cc.export('solid_composition', 'Tuple((f8[:], f8[:]))'
           '(f8, f8, DictType(unicode_type, f8[:, :]), f8)')
@njit('Tuple((f8[:], f8[:]))(f8, f8, DictType(unicode_type, f8[:, :]), f8)')
def solid_composition(X, P, quad_poly_coeff, eNd):
    mnrl_prop = {}
    # Proportions of parameterised mineral phases in the residual solid
    for mnrl, coeff in quad_poly_coeff.items():
        a = coeff[0, 0] * P ** 2 + coeff[0, 1] * P + coeff[0, 2]
        b = coeff[1, 0] * P ** 2 + coeff[1, 1] * P + coeff[1, 2]
        mnrl_prop[mnrl] = min(1 - X, max(a * X + b, 0.))

    # Proportions of mineral phases per aluminous phase (sum to 1 - X)
    Fn_spl, Fn_gnt = zeros(6), zeros(6)  # Order: [ol, opx, cpx, plg, spl, gnt]
    Fn_spl[0], Fn_gnt[0] = mnrl_prop["ol_spl"], mnrl_prop["ol_gnt"]
    Fn_spl[2], Fn_gnt[2] = mnrl_prop["cpx"], mnrl_prop["cpx"]
    Fn_spl[4], Fn_gnt[5] = mnrl_prop["spl"], mnrl_prop["gnt"]

    # Incorporate orthopyroxene and adjust mineral proportions (sum to 1)
    for Fn_al in (Fn_spl, Fn_gnt):
        sum_ol_cpx_al = sum(Fn_al)
        if sum_ol_cpx_al <= 1 - X:  # Remaining proportion -> orthopyroxene
            Fn_al[1] = 1 - X - sum_ol_cpx_al
            Fn_al /= (1 - X)
        else:  # Proportions in excess (re-calibrate quad_poly_coeff?)
            Fn_al /= sum_ol_cpx_al

        # Correction from Figure 3 of Kimura and Kawabata - G-Cubed (2014)
        KK2014 = 0.04 * eNd / 10
        Fn_al[0] += KK2014
        Fn_al[1] -= max(KK2014 - Fn_al[2], 0)
        Fn_al[2] -= min(KK2014, Fn_al[2])

    return Fn_spl, Fn_gnt


# Partition coefficient between an element and a mineral at a crystal site
# Model relies on a fictive element of ideal radius -> strain-free interaction
# Equations 11 and 12 of Brice - Journal of Crystal Growth (1975)
@cc.export('part_coeff_ideal_radius', 'f8[:](f8, f8, f8, f8[:], f8)')
@cc.export('part_coeff_ideal_radius_single', 'f8(f8, f8, f8, f8, f8)')
@njit(['f8[:](f8, f8, f8, f8[:], f8)', 'f8(f8, f8, f8, f8, f8)'])
def part_coeff_ideal_radius(Do, E, ro, ri, T):
    return Do * exp(-4 * pi * E * Avogadro / R / T
                    * (ro / 2 * (ri - ro) ** 2 + (ri - ro) ** 3 / 3))


# Partition coefficient between an element and a mineral at a crystal site
# Model relies on a ion of same charge but different radius
# Equation 3 of Blundy and Wood - Nature (1994)
@cc.export('part_coeff_same_charge', 'f8[:](f8, f8, f8, f8[:], f8, f8)')
@njit('f8[:](f8, f8, f8, f8[:], f8, f8)')
def part_coeff_same_charge(Da, E, ro, ri, ra, T):
    return Da * exp(-4 * pi * E * Avogadro / R / T
                    * (ro / 2 * (ra ** 2 - ri ** 2) + (ri ** 3 - ra ** 3) / 3))


# Partition coefficients between all elements and mineral phases considered
@cc.export('partition_coefficient',
           'f8[:, :](f8, f8, f8, f8[:, :], f8[:, :], u8[:], f8, f8)')
@njit('f8[:, :](f8, f8, f8, f8[:, :], f8[:, :], u8[:], f8, f8)')
def partition_coefficient(P, T, X, D, ri, val, X_spl_in, X_gnt_out):
    Dn = D.copy()  # Initialise Dn with constant partition coefficients

    mask_11 = (val == 1) & isfinite(ri[:, 1])
    mask_21 = (val == 2) & isfinite(ri[:, 1])
    mask_30 = (val == 3) & isfinite(ri[:, 0])
    mask_31 = (val == 3) & isfinite(ri[:, 1])
    mask_40 = (val == 4) & isfinite(ri[:, 0])
    mask_41 = (val == 4) & isfinite(ri[:, 1])

    # # # # # #
    # Olivine #
    # # # # # #
    if mask_30.any():
        F_Al = al_phase_select(X, X_spl_in, X_gnt_out, 1.56e-3, 5.64e-3)
        Mg_num = al_phase_select(X, X_spl_in, X_gnt_out,
                                 0.059 * X + 0.904, 0.07 * X + 0.897)
        # Equations 17-19 of Sun and Liang - Chemical Geology (2013)
        Do = exp(-0.45 - 0.11 * P + 1.54 * F_Al - 1.94 * Mg_num)
        Dn[mask_30, 0] = part_coeff_ideal_radius(
            Do, 426e9, 0.72e-10, ri[mask_30, 0], T)
    # # # # # # # # #
    # Orthopyroxene #
    # # # # # # # # #
    if mask_21.any() or mask_31.any():
        X_Ca_M2 = -0.756 * X ** 2 + 0.273 * X + 0.063
        X_Mg_M2 = 0.692 * X ** 2 - 0.176 * X + 0.834
        X_Al_T = -0.675 * X ** 2 + 0.041 * X + 0.146
        # Equations 11-13 from Yao et al. - Contrib. to Min. and Pet. (2012)
        Do = exp(-5.37 + 3.87e4 / R / T + 3.56 * X_Ca_M2 + 3.54 * X_Al_T)
        ro = 0.69 + 0.43 * X_Ca_M2 + 0.23 * X_Mg_M2
        E = (-1.37 + 1.85 * ro - 0.53 * X_Ca_M2) * 1e12
        if mask_21.any():
            # Section 3.11.10.6.3 of Wood and Blundy - Treatise on Geo. (2014)
            D_Mg, r_Mg = 1, 0.89e-10
            ro_v2, E_v2 = (ro + 0.08) * 1e-10, 2 / 3 * E
            Dn[mask_21, 1] = part_coeff_same_charge(
                D_Mg, E_v2, ro_v2, ri[mask_21, 1], r_Mg, T)
        if mask_31.any():
            Dn[mask_31, 1] = part_coeff_ideal_radius(
                Do, E, ro * 1e-10, ri[mask_31, 1], T)
    # # # # # # # # #
    # Clinopyroxene #
    # # # # # # # # #
    if mask_11.any() or mask_21.any() or mask_31.any() or mask_41.any():
        X_Mg_M1 = al_phase_select(X, X_spl_in, X_gnt_out,
                                  0.425 * X + 0.741, 0.191 * X + 0.793)
        X_Mg_M2 = al_phase_select(X, X_spl_in, X_gnt_out,
                                  0.583 * X + 0.223, 0.422 * X + 0.547)
        X_Mg_Mel = al_phase_select(X, X_spl_in, X_gnt_out,
                                   0.14 * X + 0.722, 0.207 * X + 0.701)
        X_Al_M1 = al_phase_select(X, X_spl_in, X_gnt_out,
                                  -0.438 * X + 0.137, -0.114 * X + 0.099)
        X_Al_T = al_phase_select(X, X_spl_in, X_gnt_out,
                                 -0.177 * X + 0.154, -0.013 * X + 0.061)
        X_H2O_Mel = 0  # Neglect effect of water in melt
        # Equations 8-10 from Sun and Liang - Contrib. to Min. and Pet. (2012)
        Do = exp(-7.14 + 7.19e4 / R / T + 4.37 * X_Al_T + 1.98 * X_Mg_M2
                 - 0.91 * X_H2O_Mel)
        ro = 1.066 - 0.104 * X_Al_M1 - 0.212 * X_Mg_M2
        E = (2.27 * ro - 2) * 1e12
        if mask_11.any():
            # Section 3.11.10.1.3 of Wood and Blundy - Treatise on Geo. (2014)
            D_Na = exp((10_367 + 2100 * P - 165 * P ** 2) / T - 10.27
                       + 0.358 * P - 0.0184 * P ** 2)
            r_Na = 1.18e-10
            ro_v1, E_v1 = (ro + 0.12) * 1e-10, E / 3
            Dn[mask_11, 2] = part_coeff_same_charge(
                D_Na, E_v1, ro_v1, ri[mask_11, 1], r_Na, T)
        if mask_21.any():
            # Section 3.11.10.1.2 of Wood and Blundy - Treatise on Geo. (2014)
            # Partition coefficient for Ca could be parameterised using data
            # from Hill, Blundy and Wood - Contrib. to Min. and Pet. (2011)
            D_Ca, r_Ca = 2, 1.12e-10
            ro_v2, E_v2 = (ro + 0.06) * 1e-10, 2 / 3 * E
            Dn[mask_21, 2] = part_coeff_same_charge(
                D_Ca, E_v2, ro_v2, ri[mask_21, 1], r_Ca, T)
        if mask_31.any():
            Dn[mask_31, 2] = part_coeff_ideal_radius(
                Do, E, ro * 1e-10, ri[mask_31, 1], T)
        if mask_41.any():
            # Section 3.11.10.1.4 of Wood and Blundy - Treatise (2014)
            r_Th = 1.041e-10
            ro_v4, E_v4 = ro * 1e-10, 4 / 3 * E
            Y_Th_M2 = 1 / part_coeff_ideal_radius(1, E_v4, ro_v4, r_Th, T)
            Y_Mg_M1 = exp(902 * (1 - X_Mg_M1) ** 2 / T)
            D_Th = exp((214_790 - 175.7 * T + 16_420 * P - 1500 * P ** 2)
                       / R / T) * X_Mg_Mel / X_Mg_M1 / Y_Mg_M1 / Y_Th_M2
            Dn[mask_41, 2] = part_coeff_same_charge(
                D_Th, E_v4, ro_v4, ri[mask_41, 1], r_Th, T)
    # # # # # # # #
    # Plagioclase #
    # # # # # # # #
    # # # # # #
    # Spinel  #
    # # # # # #
    # # # # # #
    # Garnet  #
    # # # # # #
    if mask_21.any() or mask_31.any() or mask_40.any():
        F_Ca = -0.247 * X + 0.355
        # Equations 10-12 from Sun and Liang - Chemical Geology (2013)
        Do = exp(-2.05 + (9.17e4 - 91.35 * P * (38 - P)) / R / T - 1.02 * F_Ca)
        ro = 0.78 + 0.155 * F_Ca
        E = (-1.62 + 2.29 * ro) * 1e12
        if mask_21.any():
            # Section 3.11.10.3.3 of Wood and Blundy - Treatise on Geo. (2014)
            D_Mg = (exp((258_210 - 141.5 * T + 5418 * P) / 3 / R / T)
                    / exp(1.9e4 * F_Ca ** 2 / R / T))
            r_Mg = 0.89e-10
            ro_v2, E_v2 = (0.053 + ro) * 1e-10, 2 / 3 * E
            Dn[mask_21, 5] = part_coeff_same_charge(
                D_Mg, E_v2, ro_v2, ri[mask_21, 1], r_Mg, T)
        if mask_31.any():
            Dn[mask_31, 5] = part_coeff_ideal_radius(
                Do, E, ro * 1e-10, ri[mask_31, 1], T)
        if mask_40.any():
            # Figure 7 (caption) of Mallmann and O'Neill - G. et C. Acta (2007)
            Do_v4, ro_v4, E_v4 = 4.38, 0.6626e-10, 2753e9
            Dn[mask_40, 5] = part_coeff_ideal_radius(
                Do_v4, E_v4, ro_v4, ri[mask_40, 0], T)

    return Dn


if __name__ == "__main__":
    cc.compile()
